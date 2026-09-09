import Tyr.Mctx.Math

/-!
# Seeded MCTS exploration

Shared, tensor-free samplers for the tree, batched, and DAG policies. SplitMix64
mixes both seeds and streams; uniforms use the midpoints of 2^52 bins, strictly
inside (0, 1). Gamma sampling uses Marsaglia and Tsang's rejection method
(2000), with the shape-boost identity below one. Dirichlet normalization stays
in log space so small concentrations do not underflow to an all-zero draw.

These streams are reproducible, but are not bit-compatible with JAX's PRNG.
-/

namespace torch.mctx.Sampling

private def mix64 (x : UInt64) : UInt64 :=
  let x := (x ^^^ (x >>> 30)) * 0xbf58476d1ce4e5b9
  let x := (x ^^^ (x >>> 27)) * 0x94d049bb133111eb
  x ^^^ (x >>> 31)

/-- Derive distinct deterministic streams for root noise, search, and acting. -/
def splitKey (key stream : UInt64) : UInt64 :=
  mix64 (key + (stream + 1) * 0x9e3779b97f4a7c15)

private def uniform (state : UInt64) : Float × UInt64 :=
  let state := state + 0x9e3779b97f4a7c15
  let bits := mix64 state >>> 12
  ((bits.toFloat + 0.5) / 4503599627370496.0, state)

private def normal (state : UInt64) : Float × UInt64 :=
  let (u, state) := uniform state
  let (v, state) := uniform state
  (Float.sqrt (-2.0 * Float.log u) * Float.cos (6.283185307179586 * v), state)

-- The rejection loop consumes fresh random values until a proposal is accepted.
-- Its acceptance probability is at least 0.95 for the shapes used here (>= 1).
private partial def logGammaAboveOne (state : UInt64) (shape : Float) : Float × UInt64 :=
  let d := shape - 1.0 / 3.0
  let c := 1.0 / (3.0 * Float.sqrt d)
  let (x, state) := normal state
  let v := 1.0 + c * x
  if v <= 0.0 then logGammaAboveOne state shape
  else
    let v3 := v * v * v
    let (u, state) := uniform state
    let x2 := x * x
    if u < 1.0 - 0.0331 * x2 * x2 ||
        Float.log u < 0.5 * x2 + d * (1.0 - v3 + Float.log v3) then
      (Float.log d + 3.0 * Float.log v, state)
    else logGammaAboveOne state shape

/-- A symmetric Dirichlet draw. Invalid concentrations return uniform weights.
    Empty and singleton draws are respectively empty and `[1]`. -/
def dirichlet (key : UInt64) (n : Nat) (alpha : Float) : Array Float := Id.run do
  if n == 0 then return #[]
  if n == 1 then return #[1.0]
  if alpha <= 0.0 || !alpha.isFinite then
    return Array.replicate n (1.0 / Float.ofNat n)
  let mut state := key
  let mut logs := Array.mkEmpty n
  for _ in [:n] do
    let (g, next) := logGammaAboveOne state (if alpha < 1.0 then alpha + 1.0 else alpha)
    state := next
    if alpha < 1.0 then
      let (u, next) := uniform state
      state := next
      -- Store alpha * log(Gamma(alpha)): subtract the maximum before dividing
      -- by alpha, avoiding overflow even for subnormal positive alpha.
      logs := logs.push (alpha * g + Float.log u)
    else
      logs := logs.push g
  let largest := maxD logs (logs.getD 0 0.0)
  let scale := if alpha < 1.0 then alpha else 1.0
  let weights := logs.map (fun x => Float.exp ((x - largest) / scale))
  let total := sum weights
  return weights.map (· / total)

/-- Independent Gumbel(0, scale) samples; scale zero disables the noise. -/
def gumbel (key : UInt64) (n : Nat) (scale : Float) : Array Float := Id.run do
  if scale == 0.0 then return Array.replicate n 0.0
  let mut state := key
  let mut result := Array.mkEmpty n
  for _ in [:n] do
    let (u, next) := uniform state
    state := next
    result := result.push (scale * (-Float.log (-Float.log u)))
  return result

private def legal (invalid : Option (Array Bool)) (i : Nat) : Bool :=
  !(invalid.getD #[]).getD i false

/-- Mask and normalize nonnegative finite weights. With no positive legal
    weights, use a uniform distribution over legal actions. An all-invalid mask
    retains the legacy uniform fallback; callers must supply a legal action to
    obtain a valid selection. -/
def normalizeWeights (weights : Array Float) (invalid : Option (Array Bool)) : Array Float := Id.run do
  let clean := weights.mapIdx fun i w =>
    if legal invalid i && w > 0.0 && w.isFinite then w else 0.0
  let largest := maxD clean
  if largest > 0.0 then
    let scaled := clean.map (· / largest)
    let total := sum scaled
    return scaled.map (· / total)
  let count := (List.range weights.size).countP (legal invalid)
  if count == 0 then
    return Array.replicate weights.size (1.0 / Float.ofNat (max 1 weights.size))
  return weights.mapIdx fun i _ => if legal invalid i then 1.0 / Float.ofNat count else 0.0

/-- Sample from legal `weights^(1/temperature)`. Nonpositive or NaN temperature
    chooses the first maximum; positive infinity is uniform on positive support.
    Zero-mass actions are excluded exactly. Empty input returns the sentinel 0. -/
def sampleAction (key : UInt64) (weights : Array Float) (temperature : Float)
    (invalid : Option (Array Bool)) : Nat := Id.run do
  -- Temper raw positive weights in log space. Normalizing first could erase
  -- tiny positive weights that become significant again at high temperature.
  let positive := weights.mapIdx fun i w =>
    if legal invalid i && w > 0.0 && w.isFinite then w else 0.0
  let probs := if maxD positive > 0.0 then positive else normalizeWeights weights invalid
  if probs.isEmpty then return 0
  if temperature <= 0.0 || temperature.isNaN then return argmax probs
  let maxLog := Float.log (maxD probs)
  let tempered := probs.map fun p =>
    if p > 0.0 then Float.exp ((Float.log p - maxLog) / temperature) else 0.0
  let (u, _) := uniform key
  let threshold := u * sum tempered
  let mut cumulative := 0.0
  let mut lastPositive := 0
  for i in [:tempered.size] do
    let p := tempered[i]!
    if p > 0.0 then
      lastPositive := i
      cumulative := cumulative + p
      if threshold < cumulative then return i
  -- Rounding at the top of the CDF must not select a trailing zero-mass action.
  return lastPositive

/-- Mix root priors with symmetric Dirichlet noise over legal actions only.
    Fractions are clamped to [0, 1]; NaN fractions or invalid alpha disable noise.
    Invalid logits are removed before softmax so they cannot suppress legal priors. -/
def rootLogits (key : UInt64) (logits : Array Float) (invalid : Option (Array Bool))
    (fraction alpha : Float) : Array Float := Id.run do
  let indices := (List.range logits.size).filter (legal invalid)
  if indices.isEmpty then return Array.replicate logits.size (-1e30)
  let fraction :=
    if fraction.isNaN || alpha <= 0.0 || !alpha.isFinite then 0.0
    else max 0.0 (min 1.0 fraction)
  -- Avoid a softmax/log round trip when noise is disabled; preserve zero priors.
  if fraction == 0.0 then
    let largest := maxD (indices.toArray.map fun i => logits.getD i 0.0)
      (logits.getD (indices.headD 0) 0.0)
    return logits.mapIdx fun i x => if legal invalid i then x - largest else (-1.0 / 0.0)
  let priors := softmax (indices.toArray.map fun i => logits.getD i 0.0)
  let noise := dirichlet key indices.length alpha
  let mut result := Array.replicate logits.size (-1.0 / 0.0)
  for (i, j) in indices.zipIdx do
    let p := (1.0 - fraction) * priors[j]! + fraction * noise[j]!
    result := result.set! i (Float.log p)
  return result

end torch.mctx.Sampling

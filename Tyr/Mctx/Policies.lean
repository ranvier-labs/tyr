import Tyr.Mctx.Search

/-!
# Tyr.Mctx.Policies

Unbatched search policies (Lean port of mctx policies): `muzeroPolicy`,
`alphazeroPolicy` with optional subtree persistence, and
`gumbelMuZeroPolicy`, plus invalid-action masking, temperature, and
Dirichlet-noise helpers.
-/

namespace torch.mctx

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

private def clamp01 (x : Float) : Float :=
  if x < 1e-6 then 1e-6 else if x > 1.0 - 1e-6 then 1.0 - 1e-6 else x

private def pseudoUniform01 (key : UInt64) (i : Nat) : Float :=
  let x := (key + UInt64.ofNat (i + 1) * 0x9e3779b97f4a7c15)
  let mant := (x >>> 11).toNat
  let denom : Float := Float.ofNat (Nat.pow 2 53)
  clamp01 (Float.ofNat mant / denom)

private def pseudoProbs (key : UInt64) (n : Nat) : Array Float :=
  let vals := (List.range n).toArray.map (fun i => pseudoUniform01 key i)
  let z := sum vals
  if z <= 0.0 then
    Array.replicate n (1.0 / Float.ofNat (Nat.max 1 n))
  else
    vals.map (fun v => v / z)

/-- Returns logits with near-zero mass on invalid actions (`true` = invalid). -/
def maskInvalidActions (logits : Array Float) (invalidActions : Option (Array Bool)) : Array Float :=
  match invalidActions with
  | none => logits
  | some invalid =>
    let shifted :=
      if logits.isEmpty then logits
      else
        let m := maxD logits (logits.getD 0 0.0)
        logits.map (fun x => x - m)
    (List.range shifted.size).toArray.map fun i =>
      if invalid.getD i false then -1e30 else shifted.getD i 0.0

private def getLogitsFromProbs (probs : Array Float) : Array Float :=
  probs.map logSafe

private def addDirichletNoise
    (key : UInt64)
    (probs : Array Float)
    (dirichletFraction : Float)
    : Array Float :=
  let noise := pseudoProbs key probs.size
  (List.range probs.size).toArray.map fun i =>
    (1.0 - dirichletFraction) * probs.getD i 0.0 + dirichletFraction * noise.getD i 0.0

/-- Returns temperature-scaled logits (`logits / temperature`) with safe `temperature=0` handling. -/
def applyTemperature (logits : Array Float) (temperature : Float) : Array Float :=
  if logits.isEmpty then
    #[]
  else
    let m := maxD logits (logits.getD 0 0.0)
    let shifted := logits.map (fun x => x - m)
    let t := if temperature <= 0.0 then 1e-30 else temperature
    shifted.map (fun x => x / t)

/-- Lean4 MuZero policy port (unbatched). -/
def muzeroPolicy
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (numSimulations : Nat)
    (invalidActions : Option (Array Bool) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (_dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (Tree S Unit) :=
  let noisyLogits :=
    let probs := softmax root.priorLogits
    let noisyProbs := addDirichletNoise rngKey probs dirichletFraction
    getLogitsFromProbs noisyProbs
  let root := { root with priorLogits := maskInvalidActions noisyLogits invalidActions }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let searchTree := search params rngKey root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions ()

  let summary := searchTree.summary
  let actionWeights := summary.visitProbs
  let actionLogits := applyTemperature (getLogitsFromProbs actionWeights) temperature
  let action := argmax actionLogits
  { action := action, actionWeights := actionWeights, searchTree := searchTree }

/-- AlphaZero-style policy with optional subtree persistence (`mctx-az` style).
    - If `searchTree = none`, initializes a fresh tree with capacity `maxNodes`
      (default `numSimulations + 1`).
    - If `searchTree = some t`, updates root priors/raw value and continues
      search from `t`.
-/
def alphazeroPolicy
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (numSimulations : Nat)
    (searchTree : Option (Tree S Unit) := none)
    (maxNodes : Option Nat := none)
    (invalidActions : Option (Array Bool) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (_dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (Tree S Unit) :=
  let noisyLogits :=
    let probs := softmax root.priorLogits
    let noisyProbs := addDirichletNoise rngKey probs dirichletFraction
    getLogitsFromProbs noisyProbs
  let root := { root with priorLogits := maskInvalidActions noisyLogits invalidActions }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let rootInvalid := invalidActions.getD (Array.replicate root.priorLogits.size false)
  let initialTree :=
    match searchTree with
    | none =>
      instantiateTreeFromRootWithCapacity root (maxNodes.getD (numSimulations + 1)) rootInvalid ()
    | some t =>
      updateTreeWithRoot t root rootInvalid ()

  let searchTree := searchWithTree
    params rngKey initialTree recurrentFn rootFn interiorFn numSimulations maxDepth

  let summary := searchTree.summary
  let actionWeights := summary.visitProbs
  let actionLogits := applyTemperature (getLogitsFromProbs actionWeights) temperature
  let action := argmax actionLogits
  { action := action, actionWeights := actionWeights, searchTree := searchTree }

private def sampleGumbel (key : UInt64) (n : Nat) (scale : Float) : Array Float :=
  (List.range n).toArray.map fun i =>
    let u := pseudoUniform01 key i
    scale * (-(Float.log (-Float.log u)))

/-- Lean4 Gumbel MuZero policy port (unbatched). -/
def gumbelMuZeroPolicy
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (numSimulations : Nat)
    (invalidActions : Option (Array Bool) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S GumbelMuZeroExtraData := qtransformCompletedByMixValue)
    (maxNumConsideredActions : Nat := 16)
    (gumbelScale : Float := 1.0)
    : PolicyOutput (Tree S GumbelMuZeroExtraData) :=
  let root := { root with priorLogits := maskInvalidActions root.priorLogits invalidActions }
  let gumbel := sampleGumbel rngKey root.priorLogits.size gumbelScale
  let extraData : GumbelMuZeroExtraData := { rootGumbel := gumbel }

  let rootFn : RootActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex =>
    gumbelMuZeroRootActionSelection tree nodeIndex numSimulations maxNumConsideredActions qtransform
  let interiorFn : InteriorActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex _depth =>
    gumbelMuZeroInteriorActionSelection tree nodeIndex qtransform

  let searchTree := search params rngKey root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions extraData

  let summary := searchTree.summary
  let consideredVisit := summary.visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc
  let completedQvalues := qtransform searchTree ROOT_INDEX
  let toArgmax := scoreConsidered consideredVisit gumbel root.priorLogits completedQvalues summary.visitCounts
  let action := maskedArgmax toArgmax invalidActions

  let completedSearchLogits := maskInvalidActions (addArrays root.priorLogits completedQvalues) invalidActions
  let actionWeights := softmax completedSearchLogits

  { action := action, actionWeights := actionWeights, searchTree := searchTree }

end torch.mctx

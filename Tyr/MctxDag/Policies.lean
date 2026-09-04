import Tyr.MctxDag.Search

/-!
# Tyr.MctxDag.Policies

Unbatched DAG-backed policies, mirroring `Tyr.Mctx.Policies`:
`muzeroPolicyDag`, `alphazeroPolicyDag` with optional graph persistence,
and `gumbelMuZeroPolicyDag`.
-/

namespace torch.mctxdag

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
        let m := torch.mctx.maxD logits (logits.getD 0 0.0)
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
    let m := torch.mctx.maxD logits (logits.getD 0 0.0)
    let shifted := logits.map (fun x => x - m)
    let t := if temperature <= 0.0 then 1e-30 else temperature
    shifted.map (fun x => x / t)

private def sampleGumbel (key : UInt64) (n : Nat) (scale : Float) : Array Float :=
  (List.range n).toArray.map fun i =>
    let u := pseudoUniform01 key i
    scale * (-(Float.log (-Float.log u)))

/-- DAG-backed MuZero policy (unbatched). -/
def muzeroPolicyDag
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (keyFn : S → K)
    (numSimulations : Nat)
    (invalidActions : Option (Array Bool) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S K Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (_dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (DagTree S K Unit) :=
  let noisyLogits :=
    let probs := softmax root.priorLogits
    let noisyProbs := addDirichletNoise rngKey probs dirichletFraction
    getLogitsFromProbs noisyProbs
  let root := { root with priorLogits := maskInvalidActions noisyLogits invalidActions }

  let interiorFn : InteriorActionSelectionFn S K Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S K Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let searchTree := searchDag params rngKey root recurrentFn keyFn rootFn interiorFn
    numSimulations maxDepth invalidActions ()

  let summary := searchTree.summary
  let actionWeights := summary.visitProbs
  let actionLogits := applyTemperature (getLogitsFromProbs actionWeights) temperature
  let action := argmax actionLogits
  { action := action, actionWeights := actionWeights, searchTree := searchTree }

/-- DAG-backed AlphaZero-style policy with optional graph persistence. -/
def alphazeroPolicyDag
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (keyFn : S → K)
    (numSimulations : Nat)
    (searchTree : Option (DagTree S K Unit) := none)
    (maxNodes : Option Nat := none)
    (invalidActions : Option (Array Bool) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S K Unit := qtransformByParentAndSiblings)
    (dirichletFraction : Float := 0.25)
    (_dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (DagTree S K Unit) :=
  let noisyLogits :=
    let probs := softmax root.priorLogits
    let noisyProbs := addDirichletNoise rngKey probs dirichletFraction
    getLogitsFromProbs noisyProbs
  let root := { root with priorLogits := maskInvalidActions noisyLogits invalidActions }

  let interiorFn : InteriorActionSelectionFn S K Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S K Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let rootInvalid := invalidActions.getD (Array.replicate root.priorLogits.size false)
  let rootKey := keyFn root.embedding
  let initialTree :=
    match searchTree with
    | none =>
      instantiateDagTreeFromRootWithCapacity root rootKey (maxNodes.getD (numSimulations + 1)) rootInvalid ()
    | some t =>
      updateDagTreeWithRoot t root rootKey rootInvalid ()

  let searchTree := searchWithDag
    params rngKey initialTree recurrentFn keyFn rootFn interiorFn numSimulations maxDepth

  let summary := searchTree.summary
  let actionWeights := summary.visitProbs
  let actionLogits := applyTemperature (getLogitsFromProbs actionWeights) temperature
  let action := argmax actionLogits
  { action := action, actionWeights := actionWeights, searchTree := searchTree }

/-- DAG-backed Gumbel MuZero policy (unbatched). -/
def gumbelMuZeroPolicyDag
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (keyFn : S → K)
    (numSimulations : Nat)
    (invalidActions : Option (Array Bool) := none)
    (maxDepth : Option Nat := none)
    (qtransform : QTransform S K GumbelMuZeroExtraData := qtransformCompletedByMixValue)
    (maxNumConsideredActions : Nat := 16)
    (gumbelScale : Float := 1.0)
    : PolicyOutput (DagTree S K GumbelMuZeroExtraData) :=
  let root := { root with priorLogits := maskInvalidActions root.priorLogits invalidActions }
  let gumbel := sampleGumbel rngKey root.priorLogits.size gumbelScale
  let extraData : GumbelMuZeroExtraData := { rootGumbel := gumbel }

  let rootFn : RootActionSelectionFn S K GumbelMuZeroExtraData := fun _ tree nodeIndex =>
    gumbelMuZeroRootActionSelection tree nodeIndex numSimulations maxNumConsideredActions qtransform
  let interiorFn : InteriorActionSelectionFn S K GumbelMuZeroExtraData := fun _ tree nodeIndex _depth =>
    gumbelMuZeroInteriorActionSelection tree nodeIndex qtransform

  let searchTree := searchDag params rngKey root recurrentFn keyFn rootFn interiorFn
    numSimulations maxDepth invalidActions extraData

  let summary := searchTree.summary
  let consideredVisit := summary.visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc
  let completedQvalues := qtransform searchTree ROOT_INDEX
  let toArgmax :=
    torch.mctx.scoreConsidered consideredVisit gumbel root.priorLogits completedQvalues summary.visitCounts
  let action := maskedArgmax toArgmax invalidActions

  let completedSearchLogits := maskInvalidActions (addArrays root.priorLogits completedQvalues) invalidActions
  let actionWeights := softmax completedSearchLogits

  { action := action, actionWeights := actionWeights, searchTree := searchTree }

end torch.mctxdag

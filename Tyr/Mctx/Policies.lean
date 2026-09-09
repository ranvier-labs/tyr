import Tyr.Mctx.Search
import Tyr.Mctx.Sampling

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

/-- Returns logits with zero mass on invalid actions (`true` = invalid).
    An all-invalid mask retains the legacy uniform fallback. -/
def maskInvalidActions (logits : Array Float) (invalidActions : Option (Array Bool)) : Array Float :=
  match invalidActions with
  | none => logits
  | some _ => Sampling.rootLogits 0 logits invalidActions 0.0 0.3

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
    (dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (Tree S Unit) :=
  let noisyLogits := Sampling.rootLogits (Sampling.splitKey rngKey 0)
    root.priorLogits invalidActions dirichletFraction dirichletAlpha
  let root := { root with priorLogits := noisyLogits }

  let interiorFn : InteriorActionSelectionFn S Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let searchTree := search params (Sampling.splitKey rngKey 1) root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions ()

  let summary := searchTree.summary
  let actionWeights := Sampling.normalizeWeights summary.visitProbs invalidActions
  let action := Sampling.sampleAction (Sampling.splitKey rngKey 2)
    actionWeights temperature invalidActions
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
    (dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (Tree S Unit) :=
  let noisyLogits := Sampling.rootLogits (Sampling.splitKey rngKey 0)
    root.priorLogits invalidActions dirichletFraction dirichletAlpha
  let root := { root with priorLogits := noisyLogits }

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
    params (Sampling.splitKey rngKey 1) initialTree recurrentFn rootFn interiorFn numSimulations maxDepth

  let summary := searchTree.summary
  let actionWeights := Sampling.normalizeWeights summary.visitProbs invalidActions
  let action := Sampling.sampleAction (Sampling.splitKey rngKey 2)
    actionWeights temperature invalidActions
  { action := action, actionWeights := actionWeights, searchTree := searchTree }

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
  let gumbel := Sampling.gumbel (Sampling.splitKey rngKey 0) root.priorLogits.size gumbelScale
  let numConsidered := countConsideredActions
    maxNumConsideredActions root.priorLogits.size (invalidActions.getD #[])
  let extraData : GumbelMuZeroExtraData := {
    rootGumbel := gumbel
    consideredVisitSchedule := some (ConsideredVisitSchedule.create numConsidered numSimulations)
  }

  let rootFn : RootActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex =>
    gumbelMuZeroRootActionSelection tree nodeIndex numSimulations maxNumConsideredActions qtransform
  let interiorFn : InteriorActionSelectionFn S GumbelMuZeroExtraData := fun _ tree nodeIndex _depth =>
    gumbelMuZeroInteriorActionSelection tree nodeIndex qtransform

  let searchTree := search params (Sampling.splitKey rngKey 1) root recurrentFn rootFn interiorFn
    numSimulations maxDepth invalidActions extraData

  let summary := searchTree.summary
  let consideredVisit := summary.visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc
  let completedQvalues := qtransform searchTree ROOT_INDEX
  let toArgmax := scoreConsidered consideredVisit gumbel root.priorLogits completedQvalues summary.visitCounts
  let action := maskedArgmax toArgmax invalidActions

  let completedSearchLogits := maskInvalidActions (addArrays root.priorLogits completedQvalues) invalidActions
  let actionWeights := Sampling.normalizeWeights (softmax completedSearchLogits) invalidActions

  { action := action, actionWeights := actionWeights, searchTree := searchTree }

end torch.mctx

import Tyr.MctxDag.Search
import Tyr.Mctx.Policies
import Tyr.Mctx.Sampling

/-!
# Tyr.MctxDag.Policies

Unbatched DAG-backed policies, mirroring `Tyr.Mctx.Policies`:
`muzeroPolicyDag`, `alphazeroPolicyDag` with optional graph persistence,
and `gumbelMuZeroPolicyDag`.
-/

namespace torch.mctxdag

private def addArrays (a b : Array Float) : Array Float :=
  (List.range a.size).toArray.map fun i => a.getD i 0.0 + b.getD i 0.0

/-- Returns logits with near-zero mass on invalid actions (`true` = invalid). -/
def maskInvalidActions (logits : Array Float) (invalidActions : Option (Array Bool)) : Array Float :=
  torch.mctx.maskInvalidActions logits invalidActions

/-- Returns temperature-scaled logits (`logits / temperature`) with safe `temperature=0` handling. -/
def applyTemperature (logits : Array Float) (temperature : Float) : Array Float :=
  torch.mctx.applyTemperature logits temperature

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
    (dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (DagTree S K Unit) :=
  let noisyLogits := torch.mctx.Sampling.rootLogits (torch.mctx.Sampling.splitKey rngKey 0)
    root.priorLogits invalidActions dirichletFraction dirichletAlpha
  let root := { root with priorLogits := noisyLogits }

  let interiorFn : InteriorActionSelectionFn S K Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransform pbCInit pbCBase
  let rootFn : RootActionSelectionFn S K Unit := fun _ tree nodeIndex =>
    interiorFn 0 tree nodeIndex 0

  let searchTree := searchDag params (torch.mctx.Sampling.splitKey rngKey 1) root recurrentFn keyFn rootFn interiorFn
    numSimulations maxDepth invalidActions ()

  let summary := searchTree.summary
  let actionWeights := torch.mctx.Sampling.normalizeWeights summary.visitProbs invalidActions
  let action := torch.mctx.Sampling.sampleAction (torch.mctx.Sampling.splitKey rngKey 2)
    actionWeights temperature invalidActions
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
    (dirichletAlpha : Float := 0.3)
    (pbCInit : Float := 1.25)
    (pbCBase : Float := 19652.0)
    (temperature : Float := 1.0)
    : PolicyOutput (DagTree S K Unit) :=
  let noisyLogits := torch.mctx.Sampling.rootLogits (torch.mctx.Sampling.splitKey rngKey 0)
    root.priorLogits invalidActions dirichletFraction dirichletAlpha
  let root := { root with priorLogits := noisyLogits }

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
    params (torch.mctx.Sampling.splitKey rngKey 1) initialTree recurrentFn keyFn rootFn interiorFn numSimulations maxDepth

  let summary := searchTree.summary
  let actionWeights := torch.mctx.Sampling.normalizeWeights summary.visitProbs invalidActions
  let action := torch.mctx.Sampling.sampleAction (torch.mctx.Sampling.splitKey rngKey 2)
    actionWeights temperature invalidActions
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
  let gumbel := torch.mctx.Sampling.gumbel (torch.mctx.Sampling.splitKey rngKey 0)
    root.priorLogits.size gumbelScale
  let numConsidered := torch.mctx.countConsideredActions
    maxNumConsideredActions root.priorLogits.size (invalidActions.getD #[])
  let extraData : GumbelMuZeroExtraData := {
    rootGumbel := gumbel
    consideredVisitSchedule := some (torch.mctx.ConsideredVisitSchedule.create numConsidered numSimulations)
  }

  let rootFn : RootActionSelectionFn S K GumbelMuZeroExtraData := fun _ tree nodeIndex =>
    gumbelMuZeroRootActionSelection tree nodeIndex numSimulations maxNumConsideredActions qtransform
  let interiorFn : InteriorActionSelectionFn S K GumbelMuZeroExtraData := fun _ tree nodeIndex _depth =>
    gumbelMuZeroInteriorActionSelection tree nodeIndex qtransform

  let searchTree := searchDag params (torch.mctx.Sampling.splitKey rngKey 1) root recurrentFn keyFn rootFn interiorFn
    numSimulations maxDepth invalidActions extraData

  let summary := searchTree.summary
  let consideredVisit := summary.visitCounts.foldl (init := 0) fun acc c => if c > acc then c else acc
  let completedQvalues := qtransform searchTree ROOT_INDEX
  let toArgmax :=
    torch.mctx.scoreConsidered consideredVisit gumbel root.priorLogits completedQvalues summary.visitCounts
  let action := maskedArgmax toArgmax invalidActions

  let completedSearchLogits := maskInvalidActions (addArrays root.priorLogits completedQvalues) invalidActions
  let actionWeights := torch.mctx.Sampling.normalizeWeights (softmax completedSearchLogits) invalidActions

  { action := action, actionWeights := actionWeights, searchTree := searchTree }

end torch.mctxdag

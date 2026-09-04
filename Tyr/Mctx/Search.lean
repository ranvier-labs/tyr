import Tyr.Mctx.ActionSelection

/-!
# Tyr.Mctx.Search

Core MCTS loop over `Tree`: root instantiation, simulation, edge expansion
through the recurrent model, value backup, and the unbatched `search` /
`searchWithTree` entry points.
-/

namespace torch.mctx

private def updateAt (xs : Array α) (i : Nat) (v : α) : Array α :=
  if i < xs.size then xs.set! i v else xs

private def updateAt2D (xs : Array (Array α)) (i j : Nat) (v : α) : Array (Array α) :=
  if i < xs.size then
    let row := xs.getD i #[]
    if j < row.size then
      let row' := row.set! j v
      xs.set! i row'
    else
      xs
  else
    xs

private def intToNatNonneg (x : Int) : Nat :=
  if x < 0 then 0 else Int.toNat x

/-- Updates one node's value/prior/embedding and increments visit count. -/
def updateTreeNode
    (tree : Tree S E)
    (nodeIndex : Nat)
    (priorLogits : Array Float)
    (value : Float)
    (embedding : S)
    : Tree S E :=
  let newVisit := tree.nodeVisits.getD nodeIndex 0 + 1
  { tree with
    childrenPriorLogits := updateAt tree.childrenPriorLogits nodeIndex priorLogits
    rawValues := updateAt tree.rawValues nodeIndex value
    nodeValues := updateAt tree.nodeValues nodeIndex value
    nodeVisits := updateAt tree.nodeVisits nodeIndex newVisit
    embeddings := updateAt tree.embeddings nodeIndex embedding
  }

/-- Initializes an empty tree and fills the root node. -/
def instantiateTreeFromRootWithCapacity
    [Inhabited S]
    (root : RootFnOutput S)
    (numNodes : Nat)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : Tree S E :=
  let numActions := root.priorLogits.size
  let intRow : Array Int := Array.replicate numActions UNVISITED
  let visitRow : Array UInt64 := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  let tree : Tree S E := {
    nodeVisits := Array.replicate numNodes 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    parents := Array.replicate numNodes NO_PARENT
    actionFromParent := Array.replicate numNodes NO_PARENT
    childrenIndex := Array.replicate numNodes intRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes visitRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := updateAt (Array.replicate numNodes default) ROOT_INDEX root.embedding
    rootInvalidActions := rootInvalidActions
    extraData := extraData
  }
  updateTreeNode tree ROOT_INDEX root.priorLogits root.value root.embedding

/-- Initializes an empty tree and fills the root node.
    Capacity is `numSimulations + 1` (upstream mctx default). -/
def instantiateTreeFromRoot
    [Inhabited S]
    (root : RootFnOutput S)
    (numSimulations : Nat)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : Tree S E :=
  instantiateTreeFromRootWithCapacity root (numSimulations + 1) rootInvalidActions extraData

/-- Updates root priors/raw value/embedding in an existing tree. If the root was
    uninitialized, initializes root visit/value as well. -/
def updateTreeWithRoot
    [Inhabited S]
    (tree : Tree S E)
    (root : RootFnOutput S)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : Tree S E :=
  let rootUninitialized := tree.nodeVisits.getD ROOT_INDEX 0 = 0
  let tree := { tree with
    childrenPriorLogits := updateAt tree.childrenPriorLogits ROOT_INDEX root.priorLogits
    rawValues := updateAt tree.rawValues ROOT_INDEX root.value
    embeddings := updateAt tree.embeddings ROOT_INDEX root.embedding
    rootInvalidActions := rootInvalidActions
    extraData := extraData
  }
  if rootUninitialized then
    { tree with
      nodeValues := updateAt tree.nodeValues ROOT_INDEX root.value
      nodeVisits := updateAt tree.nodeVisits ROOT_INDEX 1
    }
  else
    tree

/-- Simulates from root until reaching an unvisited edge or depth cutoff. -/
def simulate
    (rngKey : UInt64)
    (tree : Tree S E)
    (actionSelectionFn : InteriorActionSelectionFn S E)
    (maxDepth : Nat)
    : NodeIndex × Action := Id.run do
  let mut nodeIndex := ROOT_INDEX
  let mut parentIndex := ROOT_INDEX
  let mut action : Action := 0
  let mut depth := 0
  let mut continuing := true

  while continuing do
    parentIndex := nodeIndex
    action := actionSelectionFn rngKey tree nodeIndex depth
    let nextNode := (tree.childrenIndex.getD nodeIndex #[]).getD action UNVISITED
    let nextDepth := depth + 1
    let isVisited := nextNode != UNVISITED
    let beforeCutoff := nextDepth < maxDepth
    if isVisited && beforeCutoff then
      nodeIndex := intToNatNonneg nextNode
      depth := nextDepth
    else
      continuing := false

  return (parentIndex, action)

/-- Expands one `(parent, action)` edge and evaluates recurrent dynamics. -/
def expand
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (tree : Tree S E)
    (recurrentFn : RecurrentFn P S)
    (parentIndex : NodeIndex)
    (action : Action)
    (nextNodeIndex : NodeIndex)
    : Tree S E :=
  let embedding := tree.embeddings.getD parentIndex default
  let (step, nextEmbedding) := recurrentFn params rngKey action embedding
  let inBounds := nextNodeIndex < tree.nodeVisits.size
  let tree :=
    if inBounds then
      updateTreeNode tree nextNodeIndex step.priorLogits step.value nextEmbedding
    else
      tree
  let childIdx : Int := if inBounds then Int.ofNat nextNodeIndex else UNVISITED
  { tree with
    childrenIndex := updateAt2D tree.childrenIndex parentIndex action childIdx
    childrenRewards := updateAt2D tree.childrenRewards parentIndex action step.reward
    childrenDiscounts := updateAt2D tree.childrenDiscounts parentIndex action step.discount
    parents := updateAt tree.parents nextNodeIndex (Int.ofNat parentIndex)
    actionFromParent := updateAt tree.actionFromParent nextNodeIndex (Int.ofNat action)
  }

/-- Backs up values from a leaf to the root. -/
def backward (tree : Tree S E) (leafIndex : NodeIndex) : Tree S E := Id.run do
  let mut t := tree
  let mut index := leafIndex
  let mut leafValue := t.nodeValues.getD leafIndex 0.0

  while index != ROOT_INDEX do
    let parentInt := t.parents.getD index NO_PARENT
    if parentInt < 0 then
      index := ROOT_INDEX
    else
      let parent := intToNatNonneg parentInt
      let count := t.nodeVisits.getD parent 0
      let action := intToNatNonneg (t.actionFromParent.getD index NO_PARENT)
      let reward := (t.childrenRewards.getD parent #[]).getD action 0.0
      let discount := (t.childrenDiscounts.getD parent #[]).getD action 0.0
      leafValue := reward + discount * leafValue
      let parentValue :=
        ((t.nodeValues.getD parent 0.0) * Float.ofNat count + leafValue) /
          Float.ofNat (count + 1)
      let childValue := t.nodeValues.getD index 0.0
      let childCount := (t.childrenVisits.getD parent #[]).getD action 0 + 1

      t := { t with
        nodeValues := updateAt t.nodeValues parent parentValue
        nodeVisits := updateAt t.nodeVisits parent (count + 1)
        childrenValues := updateAt2D t.childrenValues parent action childValue
        childrenVisits := updateAt2D t.childrenVisits parent action childCount
      }
      index := parent

  return t

/-- Runs full MCTS search (unbatched Lean port). -/
def searchWithTree
    [Inhabited S]
    (params : P)
    (rngKey : UInt64)
    (tree : Tree S E)
    (recurrentFn : RecurrentFn P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    : Tree S E := Id.run do
  let actionSelectionFn :=
    switchingActionSelectionWrapper rootActionSelectionFn interiorActionSelectionFn
  let depthCutoff := maxDepth.getD tree.numSimulations
  let mut tree := tree

  let mut sim := 0
  while sim < numSimulations do
    let key := rngKey + UInt64.ofNat (sim + 1)
    let (parentIndex, action) := simulate key tree actionSelectionFn depthCutoff
    let existing := (tree.childrenIndex.getD parentIndex #[]).getD action UNVISITED
    let nextNodeIndex := if existing = UNVISITED then tree.nextNodeIndex else intToNatNonneg existing
    let inBounds := nextNodeIndex < tree.nodeVisits.size
    tree := expand params key tree recurrentFn parentIndex action nextNodeIndex
    tree := backward tree (if inBounds then nextNodeIndex else parentIndex)
    sim := sim + 1

  return tree

/-- Runs full MCTS search (unbatched Lean port). -/
def search
    [Inhabited S]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (rootActionSelectionFn : RootActionSelectionFn S E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    (invalidActions : Option (Array Bool) := none)
    (extraData : E := default)
    : Tree S E := Id.run do
  let rootInvalid := invalidActions.getD (Array.replicate root.priorLogits.size false)
  let tree := instantiateTreeFromRoot root numSimulations rootInvalid extraData
  searchWithTree params rngKey tree recurrentFn rootActionSelectionFn interiorActionSelectionFn
    numSimulations maxDepth

end torch.mctx

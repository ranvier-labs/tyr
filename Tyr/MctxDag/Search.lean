import Tyr.MctxDag.ActionSelection

/-!
# Tyr.MctxDag.Search

Core DAG-MCTS loop over `DagTree`: root instantiation, path simulation,
edge expansion with transposition reuse, value backup along the simulated
path, and the `searchDag` / `searchWithDag` entry points.
-/

namespace torch.mctxdag

private def updateAt (xs : Array α) (i : Nat) (v : α) : Array α :=
  if i < xs.size then xs.set! i v else xs

private def updateAt2D (xs : Array (Array α)) (i j : Nat) (v : α) : Array (Array α) :=
  if i < xs.size then
    let row := xs.getD i #[]
    if j < row.size then
      xs.set! i (row.set! j v)
    else
      xs
  else
    xs

private def intToNatNonneg (x : Int) : Nat :=
  if x < 0 then 0 else Int.toNat x

/-- Updates a node's value/prior/embedding and increments node visit count. -/
def updateDagNode
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (nodeIndex : Nat)
    (priorLogits : Array Float)
    (value : Float)
    (embedding : S)
    : DagTree S K E :=
  let newVisit := tree.nodeVisits.getD nodeIndex 0 + 1
  { tree with
    childrenPriorLogits := updateAt tree.childrenPriorLogits nodeIndex priorLogits
    rawValues := updateAt tree.rawValues nodeIndex value
    nodeValues := updateAt tree.nodeValues nodeIndex value
    nodeVisits := updateAt tree.nodeVisits nodeIndex newVisit
    embeddings := updateAt tree.embeddings nodeIndex embedding
  }

/-- Initializes an empty DAG tree and fills the root node. -/
def instantiateDagTreeFromRootWithCapacity
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (root : RootFnOutput S)
    (rootKey : K)
    (numNodes : Nat)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : DagTree S K E :=
  let numActions := root.priorLogits.size
  let intRow : Array Int := Array.replicate numActions UNVISITED
  let visitRow : Array UInt64 := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  let tree : DagTree S K E := {
    nodeVisits := Array.replicate numNodes 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    childrenIndex := Array.replicate numNodes intRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes visitRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := updateAt (Array.replicate numNodes default) ROOT_INDEX root.embedding
    keys := updateAt (Array.replicate numNodes default) ROOT_INDEX rootKey
    rootInvalidActions := rootInvalidActions
    extraData := extraData
    keyToNode := ({} : Std.HashMap K NodeIndex).insert rootKey ROOT_INDEX
    numAllocated := 1
  }
  updateDagNode tree ROOT_INDEX root.priorLogits root.value root.embedding

/-- Initializes an empty DAG tree with capacity `numSimulations + 1`. -/
def instantiateDagTreeFromRoot
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (root : RootFnOutput S)
    (rootKey : K)
    (numSimulations : Nat)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : DagTree S K E :=
  instantiateDagTreeFromRootWithCapacity root rootKey (numSimulations + 1) rootInvalidActions extraData

/-- Updates root priors/raw value/embedding in an existing DAG tree. -/
def updateDagTreeWithRoot
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (root : RootFnOutput S)
    (rootKey : K)
    (rootInvalidActions : Array Bool)
    (extraData : E)
    : DagTree S K E :=
  let rootUninitialized := tree.nodeVisits.getD ROOT_INDEX 0 = 0
  let prevRootKey := tree.keys.getD ROOT_INDEX default
  let keyMap := ((tree.keyToNode.erase prevRootKey).erase rootKey).insert rootKey ROOT_INDEX
  let tree := { tree with
    childrenPriorLogits := updateAt tree.childrenPriorLogits ROOT_INDEX root.priorLogits
    rawValues := updateAt tree.rawValues ROOT_INDEX root.value
    embeddings := updateAt tree.embeddings ROOT_INDEX root.embedding
    keys := updateAt tree.keys ROOT_INDEX rootKey
    rootInvalidActions := rootInvalidActions
    extraData := extraData
    keyToNode := keyMap
    numAllocated := if tree.numAllocated = 0 then 1 else tree.numAllocated
  }
  if rootUninitialized then
    { tree with
      nodeValues := updateAt tree.nodeValues ROOT_INDEX root.value
      nodeVisits := updateAt tree.nodeVisits ROOT_INDEX 1
    }
  else
    tree

/-- Simulates from root until reaching an unvisited edge or depth cutoff.
    Returns the traversed parent/action path and the final `(parent, action)` to expand. -/
def simulatePath
    [BEq K]
    [Hashable K]
    (rngKey : UInt64)
    (tree : DagTree S K E)
    (actionSelectionFn : InteriorActionSelectionFn S K E)
    (maxDepth : Nat)
    : Array NodeIndex × Array Action × NodeIndex × Action := Id.run do
  let mut nodeIndex := ROOT_INDEX
  let mut depth := 0
  let mut parents : Array NodeIndex := #[]
  let mut actions : Array Action := #[]

  while true do
    let action := actionSelectionFn rngKey tree nodeIndex depth
    let nextNode := (tree.childrenIndex.getD nodeIndex #[]).getD action UNVISITED
    let nextDepth := depth + 1
    let isVisited := nextNode != UNVISITED
    let beforeCutoff := nextDepth < maxDepth
    if isVisited && beforeCutoff then
      parents := parents.push nodeIndex
      actions := actions.push action
      nodeIndex := intToNatNonneg nextNode
      depth := nextDepth
    else
      return (parents, actions, nodeIndex, action)
  unreachable!

/-- Expands one edge. If the edge is already expanded (e.g. depth cutoff), returns the
    existing child without re-running `recurrentFn` and without touching the stored
    reward/discount. If the resulting key already exists, reuses that node.
    If capacity is exhausted, no node is added and returns `parentIndex` as backup leaf. -/
def expandEdge
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (params : P)
    (rngKey : UInt64)
    (tree : DagTree S K E)
    (recurrentFn : RecurrentFn P S)
    (keyFn : S → K)
    (parentIndex : NodeIndex)
    (action : Action)
    : DagTree S K E × NodeIndex :=
  let existingEdge := (tree.childrenIndex.getD parentIndex #[]).getD action UNVISITED
  if existingEdge != UNVISITED then
    (tree, intToNatNonneg existingEdge)
  else
    let embedding := tree.embeddings.getD parentIndex default
    let (step, nextEmbedding) := recurrentFn params rngKey action embedding

    let setEdge := fun (t : DagTree S K E) (child : Int) =>
      { t with
        childrenIndex := updateAt2D t.childrenIndex parentIndex action child
        childrenRewards := updateAt2D t.childrenRewards parentIndex action step.reward
        childrenDiscounts := updateAt2D t.childrenDiscounts parentIndex action step.discount
      }

    let nextKey := keyFn nextEmbedding
    match tree.keyToNode[nextKey]? with
    | some existingNode =>
      (setEdge tree (Int.ofNat existingNode), existingNode)
    | none =>
      if tree.numAllocated < tree.capacity then
        let nodeIndex := tree.numAllocated
        let tree := updateDagNode tree nodeIndex step.priorLogits step.value nextEmbedding
        let tree := { tree with
          keys := updateAt tree.keys nodeIndex nextKey
          keyToNode := tree.keyToNode.insert nextKey nodeIndex
          numAllocated := nodeIndex + 1
        }
        (setEdge tree (Int.ofNat nodeIndex), nodeIndex)
      else
        -- Out-of-bounds: keep the edge unvisited but still back up through parent.
        let tree := { tree with
          childrenIndex := updateAt2D tree.childrenIndex parentIndex action UNVISITED
          childrenRewards := updateAt2D tree.childrenRewards parentIndex action step.reward
          childrenDiscounts := updateAt2D tree.childrenDiscounts parentIndex action step.discount
        }
        (tree, parentIndex)

/-- Backs up values along the concrete simulated path. -/
def backwardPath
    [BEq K]
    [Hashable K]
    (tree : DagTree S K E)
    (pathParents : Array NodeIndex)
    (pathActions : Array Action)
    (leafIndex : NodeIndex)
    : DagTree S K E := Id.run do
  let mut t := tree
  let mut child := leafIndex
  let mut leafValue := t.nodeValues.getD leafIndex 0.0
  let mut i := pathParents.size

  while i > 0 do
    let j := i - 1
    let parent := pathParents.getD j ROOT_INDEX
    let action := pathActions.getD j 0
    let count := t.nodeVisits.getD parent 0
    let reward := (t.childrenRewards.getD parent #[]).getD action 0.0
    let discount := (t.childrenDiscounts.getD parent #[]).getD action 0.0
    leafValue := reward + discount * leafValue
    let parentValue :=
      ((t.nodeValues.getD parent 0.0) * Float.ofNat count + leafValue) /
        Float.ofNat (count + 1)
    let childValue := t.nodeValues.getD child 0.0
    let childCount := (t.childrenVisits.getD parent #[]).getD action 0 + 1

    t := { t with
      nodeValues := updateAt t.nodeValues parent parentValue
      nodeVisits := updateAt t.nodeVisits parent (count + 1)
      childrenValues := updateAt2D t.childrenValues parent action childValue
      childrenVisits := updateAt2D t.childrenVisits parent action childCount
    }
    child := parent
    i := j

  return t

/-- Runs DAG-MCTS from a pre-initialized DAG tree. -/
def searchWithDag
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    (params : P)
    (rngKey : UInt64)
    (tree : DagTree S K E)
    (recurrentFn : RecurrentFn P S)
    (keyFn : S → K)
    (rootActionSelectionFn : RootActionSelectionFn S K E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S K E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    : DagTree S K E := Id.run do
  let actionSelectionFn :=
    switchingActionSelectionWrapper rootActionSelectionFn interiorActionSelectionFn
  let depthCutoff := maxDepth.getD tree.numSimulations
  let mut tree := tree

  let mut sim := 0
  while sim < numSimulations do
    let key := rngKey + UInt64.ofNat (sim + 1)
    let (pathParents, pathActions, parentIndex, action) := simulatePath key tree actionSelectionFn depthCutoff
    let (tree', leafIndex) := expandEdge params key tree recurrentFn keyFn parentIndex action
    tree := backwardPath tree' (pathParents.push parentIndex) (pathActions.push action) leafIndex
    sim := sim + 1

  return tree

/-- Runs DAG-MCTS from root model output. -/
def searchDag
    [Inhabited S]
    [Inhabited K]
    [BEq K]
    [Hashable K]
    [Inhabited E]
    (params : P)
    (rngKey : UInt64)
    (root : RootFnOutput S)
    (recurrentFn : RecurrentFn P S)
    (keyFn : S → K)
    (rootActionSelectionFn : RootActionSelectionFn S K E)
    (interiorActionSelectionFn : InteriorActionSelectionFn S K E)
    (numSimulations : Nat)
    (maxDepth : Option Nat := none)
    (invalidActions : Option (Array Bool) := none)
    (extraData : E := default)
    : DagTree S K E :=
  let rootInvalid := invalidActions.getD (Array.replicate root.priorLogits.size false)
  let rootKey := keyFn root.embedding
  let tree := instantiateDagTreeFromRoot root rootKey numSimulations rootInvalid extraData
  searchWithDag params rngKey tree recurrentFn keyFn rootActionSelectionFn interiorActionSelectionFn
    numSimulations maxDepth

end torch.mctxdag

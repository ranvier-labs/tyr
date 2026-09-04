import Tyr.Mctx.Base
import Tyr.Mctx.Math

/-!
# Tyr.Mctx.Tree

`Tree S E`, the unbatched search-tree state (Lean port of `mctx.Tree`):
flat arrays of node and edge statistics, with Q-value, root-summary,
reset, and subtree-extraction helpers.
-/

namespace torch.mctx

/-- Search tree state (unbatched Lean port of mctx.Tree). -/
structure Tree (S E : Type) where
  nodeVisits : Array Nat
  rawValues : Array Float
  nodeValues : Array Float
  parents : Array Int
  actionFromParent : Array Int
  childrenIndex : Array (Array Int)
  childrenPriorLogits : Array (Array Float)
  childrenVisits : Array (Array UInt64)
  childrenRewards : Array (Array Float)
  childrenDiscounts : Array (Array Float)
  childrenValues : Array (Array Float)
  embeddings : Array S
  rootInvalidActions : Array Bool
  extraData : E
  deriving Repr

instance [Inhabited S] [Inhabited E] : Inhabited (Tree S E) where
  default := {
    nodeVisits := #[]
    rawValues := #[]
    nodeValues := #[]
    parents := #[]
    actionFromParent := #[]
    childrenIndex := #[]
    childrenPriorLogits := #[]
    childrenVisits := #[]
    childrenRewards := #[]
    childrenDiscounts := #[]
    childrenValues := #[]
    embeddings := #[]
    rootInvalidActions := #[]
    extraData := default
  }

/-- Root node index. -/
def ROOT_INDEX : Nat := 0

/-- Sentinel for missing parent. -/
def NO_PARENT : Int := -1

/-- Sentinel for unexpanded edge. -/
def UNVISITED : Int := -1

/-- Number of action branches per node. -/
def Tree.numActions (tree : Tree S E) : Nat :=
  (tree.childrenIndex.getD ROOT_INDEX #[]).size

/-- Number of simulations represented in this tree capacity. -/
def Tree.numSimulations (tree : Tree S E) : Nat :=
  tree.nodeVisits.size - 1

/-- Q-values `r + gamma * v` for all actions from `nodeIndex`. -/
def Tree.qvalues (tree : Tree S E) (nodeIndex : Nat) : Array Float :=
  let rewards := tree.childrenRewards.getD nodeIndex #[]
  let discounts := tree.childrenDiscounts.getD nodeIndex #[]
  let values := tree.childrenValues.getD nodeIndex #[]
  (List.range rewards.size).toArray.map fun a =>
    rewards.getD a 0.0 + discounts.getD a 0.0 * values.getD a 0.0

private def visitProbsFromCounts (counts : Array UInt64) (numActions : UInt64) : Array Float :=
  let total : UInt64 := counts.foldl (init := 0) (· + ·)
  if total = 0 then
    if numActions = 0 then #[] else Array.replicate numActions.toNat (1.0 / numActions.toFloat)
  else
    counts.map (fun c => c.toFloat / total.toFloat)

/-- Root summary statistics used by policies. -/
def Tree.summary (tree : Tree S E) : SearchSummary :=
  let value := tree.nodeValues.getD ROOT_INDEX 0.0
  let visitCounts := tree.childrenVisits.getD ROOT_INDEX #[]
  let qvalues := tree.qvalues ROOT_INDEX
  let visitProbs := visitProbsFromCounts visitCounts tree.numActions.toUInt64
  { visitCounts := visitCounts
  , visitProbs := visitProbs
  , value := value
  , qvalues := qvalues
  }

/-- Returns the next free node slot index (or capacity if full). -/
def Tree.nextNodeIndex (tree : Tree S E) : Nat := Id.run do
  for i in [:tree.nodeVisits.size] do
    if tree.nodeVisits.getD i 0 = 0 then
      return i
  return tree.nodeVisits.size

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

/-- Resets a search tree in-place shape to empty/unvisited state. -/
def resetSearchTree [Inhabited S] (tree : Tree S E) : Tree S E :=
  let numNodes := tree.nodeVisits.size
  let numActions := tree.numActions
  let intRow : Array Int := Array.replicate numActions UNVISITED
  let natRow : Array UInt64 := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  { tree with
    nodeVisits := Array.replicate numNodes 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    parents := Array.replicate numNodes NO_PARENT
    actionFromParent := Array.replicate numNodes NO_PARENT
    childrenIndex := Array.replicate numNodes intRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes natRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := Array.replicate numNodes default
    rootInvalidActions := Array.replicate numActions false
  }

/-- Extracts the subtree rooted at a root child action and remaps node indices. -/
def getSubtree [Inhabited S] (tree : Tree S E) (childAction : Nat) : Tree S E := Id.run do
  let rootChildInt := (tree.childrenIndex.getD ROOT_INDEX #[]).getD childAction UNVISITED
  if rootChildInt = UNVISITED then
    return resetSearchTree tree

  let rootChild := intToNatNonneg rootChildInt
  let numNodes := tree.nodeVisits.size
  let numActions := tree.numActions

  let mut keep : Array Bool := Array.replicate numNodes false
  let mut stack : Array Nat := #[rootChild]

  while !stack.isEmpty do
    let some node := stack.back? | break
    stack := stack.pop
    if node < numNodes && !(keep.getD node false) then
      keep := keep.set! node true
      for a in [:numActions] do
        let child := (tree.childrenIndex.getD node #[]).getD a UNVISITED
        if child != UNVISITED then
          stack := stack.push (intToNatNonneg child)

  let mut retained : Array Nat := #[]
  for i in [:numNodes] do
    if keep.getD i false then
      retained := retained.push i

  let mut oldToNew : Array Int := Array.replicate numNodes UNVISITED
  for newIdx in [:retained.size] do
    let oldIdx := retained.getD newIdx 0
    oldToNew := oldToNew.set! oldIdx (Int.ofNat newIdx)

  let intRow : Array Int := Array.replicate numActions UNVISITED
  let natRow : Array UInt64 := Array.replicate numActions 0
  let floatRow : Array Float := Array.replicate numActions 0.0
  let mut out : Tree S E := {
    nodeVisits := Array.replicate numNodes 0
    rawValues := Array.replicate numNodes 0.0
    nodeValues := Array.replicate numNodes 0.0
    parents := Array.replicate numNodes NO_PARENT
    actionFromParent := Array.replicate numNodes NO_PARENT
    childrenIndex := Array.replicate numNodes intRow
    childrenPriorLogits := Array.replicate numNodes floatRow
    childrenVisits := Array.replicate numNodes natRow
    childrenRewards := Array.replicate numNodes floatRow
    childrenDiscounts := Array.replicate numNodes floatRow
    childrenValues := Array.replicate numNodes floatRow
    embeddings := Array.replicate numNodes default
    rootInvalidActions := Array.replicate numActions false
    extraData := tree.extraData
  }

  for newIdx in [:retained.size] do
    let oldIdx := retained.getD newIdx 0
    out := { out with
      nodeVisits := updateAt out.nodeVisits newIdx (tree.nodeVisits.getD oldIdx 0)
      rawValues := updateAt out.rawValues newIdx (tree.rawValues.getD oldIdx 0.0)
      nodeValues := updateAt out.nodeValues newIdx (tree.nodeValues.getD oldIdx 0.0)
      embeddings := updateAt out.embeddings newIdx (tree.embeddings.getD oldIdx default)
    }

    let parentNew : Int :=
      if newIdx = ROOT_INDEX then NO_PARENT
      else
        let pOld := tree.parents.getD oldIdx NO_PARENT
        if pOld = NO_PARENT then NO_PARENT else oldToNew.getD (intToNatNonneg pOld) UNVISITED
    out := { out with
      parents := updateAt out.parents newIdx parentNew
      actionFromParent := updateAt out.actionFromParent newIdx (
        if newIdx = ROOT_INDEX then NO_PARENT else tree.actionFromParent.getD oldIdx NO_PARENT
      )
    }

    for a in [:numActions] do
      let childOld := (tree.childrenIndex.getD oldIdx #[]).getD a UNVISITED
      let childNew :=
        if childOld = UNVISITED then UNVISITED
        else oldToNew.getD (intToNatNonneg childOld) UNVISITED
      out := { out with
        childrenIndex := updateAt2D out.childrenIndex newIdx a childNew
        childrenPriorLogits := updateAt2D out.childrenPriorLogits newIdx a
          ((tree.childrenPriorLogits.getD oldIdx #[]).getD a 0.0)
        childrenVisits := updateAt2D out.childrenVisits newIdx a
          ((tree.childrenVisits.getD oldIdx #[]).getD a 0)
        childrenRewards := updateAt2D out.childrenRewards newIdx a
          ((tree.childrenRewards.getD oldIdx #[]).getD a 0.0)
        childrenDiscounts := updateAt2D out.childrenDiscounts newIdx a
          ((tree.childrenDiscounts.getD oldIdx #[]).getD a 0.0)
        childrenValues := updateAt2D out.childrenValues newIdx a
          ((tree.childrenValues.getD oldIdx #[]).getD a 0.0)
      }

  return out

end torch.mctx

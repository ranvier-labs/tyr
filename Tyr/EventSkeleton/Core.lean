/-!
# Tyr.EventSkeleton.Core

An event-skeleton representation for hybrid and stochastic differentiation.

This module is intentionally independent from `Tyr.AD.Elim`.  The existing
AlphaGrad eliminator contracts local Jacobian vertices.  Event-skeleton
differentiation needs a richer move vocabulary first: interval adjoints,
saltation timing, reset transposes, branch aggregation, mark marginalization,
score estimators, BEL/noise eliminations, and learned complements.
-/

namespace Tyr.EventSkeleton

abbrev VertexId := Nat
abbrev SegmentId := Nat
abbrev EventId := Nat

inductive StateRole where
  | boundary
  | interior
  | checkpoint
  deriving Repr, BEq, Inhabited

/-- Coarse vertex classes in an event-skeleton computation graph. -/
inductive SkeletonVertexKind where
  | state (role : StateRole)
  | interval
  | eventTime
  | reset
  | mark
  | branch
  | noise
  | checkpoint
  | learnedComplement
  | frozen
  | opaque
  deriving Repr, BEq, Inhabited

/--
The legal elimination/action moves for the separate event-skeleton path.

`localSchurBlock` is the eventual bridge back to the existing AlphaGrad
vertex-elimination implementation; the other moves are hybrid/stochastic
specific and should remain first-class.
-/
inductive SkeletonMoveKind where
  | intervalAdjoint
  | saltationTime
  | resetTranspose
  | checkpointBoundary
  | rematerializeSegment
  | freezeControl
  | clockedUpdate
  | localSchurBlock
  | markMarginalize
  | markScoreSample
  | branchAggregate
  | learnedComplement
  deriving Repr, BEq, Inhabited

inductive MoveExactness where
  | exact
  | unbiasedEstimator
  | learnedApproximation
  | controlledApproximation
  deriving Repr, BEq, Inhabited

namespace SkeletonMoveKind

/-- Default exactness class for a move before any local approximation metadata. -/
def defaultExactness : SkeletonMoveKind → MoveExactness
  | .markScoreSample => .unbiasedEstimator
  | .learnedComplement => .learnedApproximation
  | _ => .exact

end SkeletonMoveKind

/-- Simple planning cost metadata.  These are deliberately unitless weights. -/
structure MoveCost where
  work : Float := 0.0
  memory : Float := 0.0
  variance : Float := 0.0
  bias : Float := 0.0
  deriving Repr, Inhabited

namespace MoveCost

def zero : MoveCost := {}

def add (a b : MoveCost) : MoveCost :=
  {
    work := a.work + b.work
    memory := a.memory + b.memory
    variance := a.variance + b.variance
    bias := a.bias + b.bias
  }

end MoveCost

structure SkeletonVertex where
  id : VertexId
  kind : SkeletonVertexKind
  label : String := ""
  deriving Repr, Inhabited

structure SkeletonMove where
  kind : SkeletonMoveKind
  targets : Array VertexId := #[]
  reads : Array VertexId := #[]
  writes : Array VertexId := #[]
  cost : MoveCost := {}
  exactness : MoveExactness := kind.defaultExactness
  label : String := ""
  deriving Repr, Inhabited

namespace SkeletonMove

def exact (kind : SkeletonMoveKind) (label : String := "") : SkeletonMove :=
  { kind := kind, exactness := .exact, label := label }

end SkeletonMove

/-- A sparse skeleton graph plus a proposed elimination schedule. -/
structure SkeletonGraph where
  vertices : Array SkeletonVertex := #[]
  moves : Array SkeletonMove := #[]
  deriving Repr, Inhabited

namespace SkeletonGraph

def empty : SkeletonGraph := {}

def addVertex (g : SkeletonGraph) (v : SkeletonVertex) : SkeletonGraph :=
  { g with vertices := g.vertices.push v }

def addMove (g : SkeletonGraph) (m : SkeletonMove) : SkeletonGraph :=
  { g with moves := g.moves.push m }

def vertexIds (g : SkeletonGraph) : Array VertexId :=
  g.vertices.map (fun v => v.id)

def moveKinds (g : SkeletonGraph) : Array SkeletonMoveKind :=
  g.moves.map (fun m => m.kind)

def containsMoveKind (g : SkeletonGraph) (kind : SkeletonMoveKind) : Bool :=
  g.moves.any (fun m => m.kind == kind)

def totalCost (g : SkeletonGraph) : MoveCost :=
  g.moves.foldl (fun acc m => acc.add m.cost) {}

end SkeletonGraph

end Tyr.EventSkeleton

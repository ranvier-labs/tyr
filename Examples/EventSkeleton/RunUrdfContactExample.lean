import Tyr.EventSkeleton.Examples.UrdfContact

/-!
# URDF Contact Event-Skeleton Runner

Runs the small URDF-backed hybrid contact example on the local machine and
prints the forward contact state plus the reverse event-skeleton messages.
-/

namespace Examples.EventSkeleton.RunUrdfContactExample

open Tyr.EventSkeleton
open Tyr.EventSkeleton.Examples.UrdfContact

private def printSection (title : String) : IO Unit := do
  IO.println ""
  IO.println s!"== {title} =="

private def printMove (i : Nat) (move : SkeletonMove) : IO Unit := do
  IO.println s!"{i}. {reprStr move.kind} exactness={reprStr move.exactness} label={move.label}"

private def printMoves (moves : Array SkeletonMove) : IO Unit := do
  for h : i in [:moves.size] do
    printMove (i + 1) moves[i]

def run (csvPath? : Option String := none) : IO UInt32 := do
  match buildEndToEnd? with
  | .error msg =>
      IO.eprintln s!"URDF contact simulation failed: {msg}"
      return 1
  | .ok result =>
      IO.println "URDF-backed contact / hybrid ODE event-skeleton example"

      printSection "URDF model"
      IO.println s!"robot={robot.name}"
      IO.println s!"links={reprStr ContactProbeUrdf.linkNames}"
      IO.println s!"joints={reprStr ContactProbeUrdf.jointNames}"
      IO.println s!"joint={slideJoint.name} type={reprStr slideJoint.jointType} axis={reprStr slideJoint.axis}"
      IO.println s!"probe mass={probeMass}"
      IO.println s!"collision sphere radius={contactRadius}"

      printSection "Forward hybrid ODE"
      IO.println s!"x0={reprStr initialState}"
      IO.println s!"attempted segment: t=0.0 -> {acceptedContactSegment.tAttempt}"
      IO.println s!"localized impact time={impactTime}"
      IO.println s!"x(tau-)={reprStr preImpactState}"
      IO.println s!"guard(x(tau-))={contactGuard preImpactState}"
      IO.println s!"x(tau+)={reprStr postImpactState}"

      printSection "Reverse contact event"
      IO.println s!"terminal p+={reprStr terminalPostImpactAdjoint}"
      IO.println s!"saltation alpha={result.saltationAlpha}"
      IO.println s!"p-={reprStr result.preImpactAdjoint}"
      IO.println s!"dL/d restitution={reprStr result.restitutionGrad}"

      printSection "Dynamic contact-mode mark"
      IO.println s!"support policy={reprStr contactModeSupport.policy}"
      IO.println s!"selected mode ids={reprStr contactModeSupport.selectedIds}"
      IO.println s!"retained probabilities={reprStr contactModeMarkData.probs}"
      IO.println s!"eliminated value={result.markMessage.value}"
      IO.println s!"mark state adjoint={reprStr result.markMessage.stateAdjoint}"

      printSection "Projected elimination moves"
      printMoves result.moves
      if let some path := csvPath? then
        writeContactTrajectoryCsv path
        IO.println s!"wrote contact trajectory {path}"
      return 0

end Examples.EventSkeleton.RunUrdfContactExample

def main (args : List String) : IO UInt32 :=
  let csvPath? :=
    match args with
    | "--csv" :: path :: _ => some path
    | _ => none
  Examples.EventSkeleton.RunUrdfContactExample.run csvPath?

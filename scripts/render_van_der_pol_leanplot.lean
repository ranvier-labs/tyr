import LeanPlot
import Tyr.EventSkeleton.Examples.VanDerPol

/-!
# Van der Pol LeanPlot Renderer

External renderer for the EventSkeleton Van der Pol rollout.  This intentionally
lives outside Tyr's Lake roots because `lean-plot` currently brings in a
transitive `Tests` module root that conflicts with Tyr's test runner.
-/

namespace Scripts.RenderVanDerPolLeanPlot

open LeanPlot
open Tyr.EventSkeleton.Examples.VanDerPol

private def defaultOutput : System.FilePath :=
  "output" / "event-skeleton" / limitCycleLeanPlotSpec.outputPath

private def outputPathFromArgs : List String → System.FilePath
  | path :: _ => System.FilePath.mk path
  | [] => defaultOutput

private def ensureParentDir (path : System.FilePath) : IO Unit := do
  match path.parent with
  | some parent =>
      unless parent.toString.isEmpty do
        IO.FS.createDirAll parent
  | none => pure ()

def phasePoints (rollout : VanDerPolResult) : Array DataPoint :=
  rollout.samples.map fun state => { x := state.q, y := state.qdot }

def limitCycleSvgPlot
    (rollout : VanDerPolResult)
    (spec : LimitCycleLeanPlotSpec := limitCycleLeanPlotSpec) : PlotSpec :=
  {
    width := Length.inches 5.0
    height := Length.inches 4.0
    title? := some "Van der Pol limit cycle"
    caption? := some "Rendered from the EventSkeleton CalcLimitCycle rollout."
    xAxis := AxisSpec.withDomain spec.xMin spec.xMax { label := spec.xLabel, tickCount := 5 }
    yAxis := AxisSpec.withDomain spec.yMin spec.yMax { label := spec.yLabel, tickCount := 5 }
    marks := #[
      Mark.lineSeries
        (phasePoints rollout)
        (some "CalcLimitCycle")
        {
          color := Color.black
          width := spec.lineWidth
          opacity := 1.0
        }
    ]
  }

def run (args : List String) : IO UInt32 := do
  match build? with
  | .error msg =>
      IO.eprintln s!"Van der Pol EventSkeleton build failed: {msg}"
      return 1
  | .ok result =>
      let outputPath := outputPathFromArgs args
      ensureParentDir outputPath
      LeanPlot.Export.writeSvg outputPath (limitCycleSvgPlot result.rollout result.leanPlotSpec)
      IO.println s!"wrote {outputPath}"
      IO.println s!"samples={result.rollout.samples.size}, closure q={result.closure.qError}, qdot={result.closure.qdotError}"
      return 0

end Scripts.RenderVanDerPolLeanPlot

def main (args : List String) : IO UInt32 :=
  Scripts.RenderVanDerPolLeanPlot.run args

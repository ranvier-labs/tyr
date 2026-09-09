import LeanTest
import Lean.Util.Path
import Tests.TestDiffusion

unsafe def main : IO UInt32 := do
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let env ← Lean.importModules #[{ module := `LeanTest }, { module := `Tests.TestDiffusion }] {}
  LeanTest.runTestsAndExit env {} {}

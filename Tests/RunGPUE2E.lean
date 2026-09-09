import LeanTest
import Lean.Util.Path
import Tests.TestGPUE2E

/-- Parse command line arguments into a LeanTest config. -/
def parseArgs (args : List String) : IO LeanTest.RunConfig := do
  let mut config : LeanTest.RunConfig := {}
  let mut remaining := args
  while _h : !remaining.isEmpty do
    match remaining with
    | "--filter" :: pattern :: rest =>
      config := { config with filter := some pattern }
      remaining := rest
    | "--ignored" :: rest =>
      config := { config with includeIgnored := true }
      remaining := rest
    | "--fail-fast" :: rest =>
      config := { config with failFast := true }
      remaining := rest
    | "--help" :: _ =>
      IO.println "Usage: TestGPUE2E [OPTIONS]"
      IO.println ""
      IO.println "Options:"
      IO.println "  --filter PATTERN  Only run tests matching PATTERN"
      IO.println "  --ignored         Include tests marked as ignored"
      IO.println "  --fail-fast       Stop on first failure"
      IO.println "  --help            Show this help"
      IO.Process.exit 0
    | _ :: rest =>
      remaining := rest
    | [] =>
      remaining := []
  pure config

unsafe def main (args : List String) : IO UInt32 := do
  let config ← parseArgs args
  Tests.GPUCoverage.checkPreflight (← Tests.GPUCoverage.strictMode)
    (← torch.cuda_is_available) true
  Lean.initSearchPath (← Lean.findSysroot)
  Lean.enableInitializersExecution
  let env ← Lean.importModules
    #[{ module := `LeanTest }, { module := `Tests.TestGPUE2E }]
    {}
  Tests.GPUCoverage.runSuite env config

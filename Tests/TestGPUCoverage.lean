import Tests.GPUCoverage

namespace Tests.TestGPUCoverage

open LeanTest GPUCoverage

private def expectError (action : IO Unit) (message : String) : IO Unit := do
  let failed ← try
    action
    pure false
  catch error =>
    assertTrue (error.toString.contains message) s!"Unexpected error: {error}"
    pure true
  assertTrue failed "Expected the GPU coverage gate to fail"

@[test]
def testStrictGpuPreflight : IO Unit := do
  expectError (checkPreflight true false true) "LibTorch cannot use CUDA"
  expectError (checkPreflight true true false) "GPU family"
  checkPreflight true true true
  checkPreflight false false false

@[test]
def testGpuCoverageRequiresExecution : IO Unit := do
  assertTrue (!coverageOkay true 0 0 0) "Empty strict selections must fail"
  assertTrue (!coverageOkay true 0 10 0) "All-skipped strict suites must fail"
  assertTrue (!coverageOkay true 9 1 0) "Partially skipped strict suites must fail"
  assertTrue (!coverageOkay true 10 0 1) "Numerical failures must fail"
  assertTrue (coverageOkay true 10 0 0) "Executed successful strict suite must pass"
  assertTrue (coverageOkay false 0 10 0) "Local optional skips remain supported"

@[test]
def testGpuSkipIsErrorInStrictMode : IO Unit := do
  expectError (skipWithMode true "fixture" "CUDA unavailable") "GPU coverage required"

end Tests.TestGPUCoverage

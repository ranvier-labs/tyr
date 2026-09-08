import LeanTest

/-! Shared accounting for optional local GPU tests and strict CI runs. -/
namespace Tests.GPUCoverage

initialize skippedCases : IO.Ref Nat ← IO.mkRef 0

def strictMode : IO Bool := do
  pure ((← IO.getEnv "TYR_GPU_TEST_STRICT") == some "1")

def skipWithMode (strict : Bool) (label reason : String) : IO Unit := do
  if strict then
    throw <| IO.userError s!"GPU coverage required: {label}: {reason}"
  skippedCases.modify (· + 1)
  IO.println s!"[skip] {label}: {reason}"

def skip (label reason : String) : IO Unit := do
  skipWithMode (← strictMode) label reason

/-- Kept separate from hardware discovery so hosted CI can test gate behavior. -/
def checkPreflight (strict cudaAvailable familyMatches : Bool) : IO Unit := do
  if strict && !cudaAvailable then
    throw <| IO.userError "GPU coverage required: linked LibTorch cannot use CUDA"
  if strict && !familyMatches then
    throw <| IO.userError "GPU coverage required: configured GPU family does not match the suite"

def coverageOkay (strict : Bool) (executed skipped failed : Nat) : Bool :=
  failed == 0 && (!strict || (executed > 0 && skipped == 0))

unsafe def runSuite (env : Lean.Environment) (config : LeanTest.RunConfig) : IO UInt32 := do
  let strict ← strictMode
  skippedCases.set 0
  let summary ← LeanTest.runTests env {} config
  let runtimeSkipped ← skippedCases.get
  let executed := summary.passed + summary.failed - runtimeSkipped
  let skipped := summary.skipped + runtimeSkipped
  IO.println s!"[gpu-coverage] executed={executed} skipped={skipped} failed={summary.failed} selected={summary.total} strict={strict}"
  if coverageOkay strict executed skipped summary.failed then
    return 0
  IO.eprintln "GPU coverage gate failed: strict runs require executed tests and no skips."
  return 1

end Tests.GPUCoverage

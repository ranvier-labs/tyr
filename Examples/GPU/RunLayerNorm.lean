/- End-to-end ThunderKittens-style layernorm validation. -/
import Tyr.Torch
import Tyr.GPU.Kernels.FusedLayerNorm
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunLayerNorm

open torch
open Tyr.GPU.Kernels

private def launchBlockX : UInt64 := 32

def suiteName : String := "layernorm"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/layernorm64x1024"⟩
  names := #["x", "residual", "weight", "bias", "expected_out", "expected_resid"]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

private abbrev LayerNormInput := T #[1, 64, 1024]
private abbrev LayerNormWeight := T #[1024]

private structure LayerNormInputs where
  x : LayerNormInput
  residual : LayerNormInput
  weight : LayerNormWeight
  bias : LayerNormWeight

private def layerNormReference
    (inputs : LayerNormInputs)
    : LayerNormInput × LayerNormInput :=
  let eps : Float := 1.0e-5
  let expectedResid := inputs.x + inputs.residual
  let expectedOut := torch.nn.layer_norm expectedResid inputs.weight inputs.bias eps
  (expectedOut, expectedResid)

private def layerNormReferenceBFloat16
    (inputs : LayerNormInputs)
    : LayerNormInput × LayerNormInput :=
  let eps : Float := 1.0e-5
  let expectedResid := inputs.x + inputs.residual
  let expectedOut32 : LayerNormInput :=
    torch.nn.layer_norm
      (torch.toFloat' expectedResid)
      (torch.toFloat' inputs.weight)
      (torch.toFloat' inputs.bias)
      eps
  let expectedOut := torch.toBFloat16' expectedOut32
  (expectedOut, expectedResid)

private def asBFloat16Inputs (inputs : LayerNormInputs) : LayerNormInputs := {
  x := torch.toBFloat16' inputs.x
  residual := torch.toBFloat16' inputs.residual
  weight := torch.toBFloat16' inputs.weight
  bias := torch.toBFloat16' inputs.bias
}

private def inputDType? (inputs : LayerNormInputs) : Except String DType := do
  let dtype := inputs.x.dtype
  if inputs.residual.dtype != dtype then
    throw s!"residual dtype {inputs.residual.dtype} does not match x dtype {dtype}"
  if inputs.weight.dtype != dtype then
    throw s!"weight dtype {inputs.weight.dtype} does not match x dtype {dtype}"
  if inputs.bias.dtype != dtype then
    throw s!"bias dtype {inputs.bias.dtype} does not match x dtype {dtype}"
  pure dtype

private def logInputDTypes (label : String) (inputs : LayerNormInputs) : IO Unit := do
  IO.println <|
    s!"[{suiteName}:{label}] input_dtypes x={inputs.x.dtype} residual={inputs.residual.dtype} " ++
    s!"weight={inputs.weight.dtype} bias={inputs.bias.dtype}"

private def compareResults
    (mode : String)
    (expectedOut : LayerNormInput)
    (expectedResid : LayerNormInput)
    (out : LayerNormInput)
    (outResid : LayerNormInput)
    (rtol atol : Float)
    : IO Bool := do
  let outCheck := compareTensors s!"layernorm.{mode}.output" (torch.toFloat' expectedOut) (torch.toFloat' out) rtol atol
  let residCheck := compareTensors s!"layernorm.{mode}.residual" (torch.toFloat' expectedResid) (torch.toFloat' outResid) rtol atol
  logTensorCheck outCheck
  logTensorCheck residCheck
  pure (outCheck.ok && residCheck.ok)

private def runInputs (label : String) (inputs : LayerNormInputs) : IO Bool := do
  logInputDTypes label inputs
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  match inputDType? inputs with
  | .error msg =>
      IO.eprintln s!"[{suiteName}:{label}] unsupported_dtype_mix error={msg}"
      pure false
  | .ok .Float32 => do
      let (expectedOut, expectedResid) := layerNormReference inputs
      let out := torch.zeros_like inputs.x
      let outResid := torch.zeros_like inputs.residual
      fusedLayerNormResidual64x1024F32Direct.launch
        inputs.x inputs.residual inputs.weight inputs.bias out outResid
        64 1 1 256 1 1 0 stream
      let _ ← torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.f32" else "f32") expectedOut expectedResid out outResid 5e-3 5e-3
  | .ok .BFloat16 => do
      let (expectedOut, expectedResid) := layerNormReferenceBFloat16 inputs
      let out := torch.zeros_like inputs.x
      let outResid := torch.zeros_like inputs.residual
      fusedLayerNormResidual64x1024Bf16Direct.launch
        inputs.x inputs.residual inputs.weight inputs.bias out outResid
        64 1 1 256 1 1 0 stream
      let _ ← torch.cuda_synchronize
      compareResults (if blackwell then "blackwell.bf16" else "bf16") expectedOut expectedResid out outResid 2e-2 2e-2
  | .ok dtype =>
      IO.eprintln s!"[{suiteName}:{label}] unsupported_dtype dtype={dtype}"
      pure false

private def randomInputs (dtype : DType) : IO LayerNormInputs := do
  let device := Device.CUDA 0
  let base : LayerNormInputs := {
    x := ← torch.rand #[1, 64, 1024] false device
    residual := ← torch.rand #[1, 64, 1024] false device
    weight := ← torch.rand #[1024] false device
    bias := ← torch.rand #[1024] false device
  }
  match dtype with
  | .Float32 => pure base
  | .BFloat16 => pure (asBFloat16Inputs base)
  | _ => throw <| IO.userError s!"unsupported random layernorm dtype {dtype}"

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate layernorm fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let inputs ← randomInputs .Float32
  let vendoredInputs := asBFloat16Inputs inputs
  let (expectedOut, expectedResid) := layerNormReferenceBFloat16 vendoredInputs

  torch.data.saveTensor inputs.x (fixtureFile "x").toString
  torch.data.saveTensor inputs.residual (fixtureFile "residual").toString
  torch.data.saveTensor inputs.weight (fixtureFile "weight").toString
  torch.data.saveTensor inputs.bias (fixtureFile "bias").toString
  torch.data.saveTensor expectedOut (fixtureFile "expected_out").toString
  torch.data.saveTensor expectedResid (fixtureFile "expected_resid").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let residMean := torch.nn.item (torch.nn.meanAll expectedResid)
  IO.println <|
    s!"Generated layernorm fixtures in {fixtureSpec.dir} " ++
    s!"input_dtype={inputs.x.dtype} vendored_ref_dtype={vendoredInputs.x.dtype} " ++
    s!"outMean={outMean} residMean={residMean}"

def runFloat32Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.f32" 0
  runInputs "random_f32" (← randomInputs .Float32)

def runBFloat16Once : IO Bool := do
  if !(← requireCuda suiteName) then
    return false
  seedFixtures s!"{suiteName}.bf16" 0
  runInputs "random_bf16" (← randomInputs .BFloat16)

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let inputs : LayerNormInputs := {
    x := ← torch.data.loadTensor #[1, 64, 1024] (fixtureFile "x").toString
    residual := ← torch.data.loadTensor #[1, 64, 1024] (fixtureFile "residual").toString
    weight := ← torch.data.loadTensor #[1024] (fixtureFile "weight").toString
    bias := ← torch.data.loadTensor #[1024] (fixtureFile "bias").toString
  }
  runInputs "fixtures" inputs


private def benchmarkFloat32 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .Float32
  let (expectedOut, expectedResid) := layerNormReference inputs
  let out := torch.zeros_like inputs.x
  let outResid := torch.zeros_like inputs.residual
  let launch := do
    fusedLayerNormResidual64x1024F32Direct.launch
      inputs.x inputs.residual inputs.weight inputs.bias out outResid
      64 1 1 256 1 1 0 stream
  launch
  torch.cuda_synchronize
  let correct ← compareResults "bench.f32" expectedOut expectedResid out outResid 5e-3 5e-3
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "layernorm_residual_f32_64x1024" "tyr" route samples correct, correct)

private def benchmarkBFloat16 (cfg : Benchmark.Config) (stream : UInt64)
    (_blackwell : Bool) : IO (String × Bool) := do
  let inputs ← randomInputs .BFloat16
  let (expectedOut, expectedResid) := layerNormReferenceBFloat16 inputs
  let out := torch.zeros_like inputs.x
  let outResid := torch.zeros_like inputs.residual
  let launch := do
    fusedLayerNormResidual64x1024Bf16Direct.launch
      inputs.x inputs.residual inputs.weight inputs.bias out outResid
      64 1 1 256 1 1 0 stream
  launch
  torch.cuda_synchronize
  let correct ← compareResults "bench.bf16" expectedOut expectedResid out outResid 2e-2 2e-2
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let route := "generated_direct_row_cta_256x4"
  pure (Benchmark.summaryJson cfg "layernorm_residual_bf16_64x1024" "tyr" route samples correct, correct)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let cfg ← Benchmark.parseConfig args "layernorm_bench"
  seedFixtures s!"{suiteName}.benchmark" 0
  let stream ← torch.cuda_current_stream
  let blackwell ← isBlackwellFamily
  let (f32Line, f32Ok) ← benchmarkFloat32 cfg stream blackwell
  let (bf16Line, bf16Ok) ← benchmarkBFloat16 cfg stream blackwell
  IO.println f32Line
  IO.println bf16Line
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path #[f32Line, bf16Line]
  | none => pure ()
  pure (if f32Ok && bf16Ok then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunLayerNorm

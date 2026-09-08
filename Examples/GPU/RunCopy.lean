/-
  Examples/GPU/RunCopy.lean

  End-to-end GPU demo and allocation-free copy benchmark.
-/
import Examples.GPU.Benchmark
import Tyr.Torch
import Tyr.GPU.Kernels.Copy
import Examples.GPU.Parity

namespace Examples.GPU.RunCopy

open torch
open Tyr.GPU.Kernels
open Examples.GPU.Benchmark

def suiteName : String := "copy"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  seedFixtures suiteName 1001
  let device := Device.CUDA 0
  let input ← torch.rand #[1, 1, 64, 64] false device
  let output := torch.zeros_like input
  let stream ← torch.cuda_current_stream

  copy64x64Float4.launch input output 1 1 1 512 1 1 0 stream
  torch.cuda_synchronize

  reportTensorComparison "copy.output" input output 1e-7 1e-7

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  let cfg ← parseConfig args "copy_bench"

  seedFixtures suiteName 1001
  let input ← torch.rand #[1, 1, 64, 64] false (Device.CUDA 0)
  let tyrOut := torch.zeros_like input
  let torchOut := torch.zeros_like input
  let stream ← torch.cuda_current_stream

  copy64x64Float4.launch input tyrOut 1 1 1 512 1 1 0 stream
  torch.copy_ torchOut input
  torch.cuda_synchronize
  let tyrCheck ← reportTensorComparison "copy.bench.tyr" input tyrOut 1e-7 1e-7
  let torchCheck ← reportTensorComparison "copy.bench.torch" input torchOut 1e-7 1e-7

  let tyrSamples ← timeCudaEvents cfg stream do
    copy64x64Float4.launch input tyrOut 1 1 1 512 1 1 0 stream
  let torchSamples ← timeCudaEvents cfg stream do
    torch.copy_ torchOut input

  let tyrLine := summaryJson cfg "copy_f32_64x64" "tyr" "generated_float4_512x2" tyrSamples tyrCheck
  let torchLine := summaryJson cfg "copy_f32_64x64" "torch_eager" "libtorch_copy_" torchSamples torchCheck
  IO.println tyrLine
  IO.println torchLine
  match cfg.jsonlOut? with
  | some path => writeJsonl path #[tyrLine, torchLine]
  | none => pure ()
  pure (if tyrCheck && torchCheck then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    let ok ← runOnce
    pure (if ok then 0 else 1)

end Examples.GPU.RunCopy

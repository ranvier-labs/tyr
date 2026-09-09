/- End-to-end GB10/Blackwell MHA validation using the reduced 2-block path. -/
import Tyr.Torch
import Tyr.GPU.Kernels.MhaGB10
import Examples.GPU.Parity
import Examples.GPU.FixtureRunner
import Examples.GPU.Benchmark

namespace Examples.GPU.RunMhaGB10

open torch
open Tyr.GPU.Kernels

def suiteName : String := "mha_gb10"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/mha_gb10_128x64"⟩
  names := #[
    "q", "k", "v", "dO",
    "expected_o", "expected_l",
    "expected_dq", "expected_dk", "expected_dv"
  ]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

def generateFixtures : IO Unit := do
  if !(← requireCuda suiteName) then
    throw <| IO.userError "CUDA is not available; cannot generate mha_gb10 fixtures."

  IO.FS.createDirAll fixtureSpec.dir
  let device := Device.CUDA 0

  let qf ← torch.randn #[1, 1, 128, 64] false device
  let kf ← torch.randn #[1, 1, 128, 64] false device
  let vf ← torch.randn #[1, 1, 128, 64] false device
  let dOf ← torch.randn #[1, 1, 128, 64] false device

  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  let dO := torch.toBFloat16' dOf

  let expectedOut := torch.nn.scaled_dot_product_attention q k v 0.0 false

  let q32 := torch.toFloat' q
  let k32 := torch.toFloat' k
  let kT : T #[1, 1, 64, 128] := torch.nn.transpose k32 2 3
  let scores : T #[1, 1, 128, 128] := torch.nn.bmm4d q32 kT
  let scaled : T #[1, 1, 128, 128] := scores / 8.0
  let expScores : T #[1, 1, 128, 128] := torch.nn.exp scaled
  let sumExp : T #[1, 1, 128] := torch.nn.sumDim expScores 3 false
  let lse3 : T #[1, 1, 128] := torch.nn.log sumExp
  let expectedL3 : T #[1, 1, 128] := torch.mul_scalar lse3 (-8.0)
  let expectedL : T #[2, 64] := torch.reshape expectedL3 #[2, 64]

  let qRef := torch.autograd.set_requires_grad q true
  let kRef := torch.autograd.set_requires_grad k true
  let vRef := torch.autograd.set_requires_grad v true
  let outRef := torch.nn.scaled_dot_product_attention qRef kRef vRef 0.0 false
  torch.autograd.backward outRef dO
  let expectedDQ := torch.toFloat' (torch.autograd.grad_of qRef)
  let expectedDK := torch.toFloat' (torch.autograd.grad_of kRef)
  let expectedDV := torch.toFloat' (torch.autograd.grad_of vRef)

  torch.data.saveTensor q (fixtureFile "q").toString
  torch.data.saveTensor k (fixtureFile "k").toString
  torch.data.saveTensor v (fixtureFile "v").toString
  torch.data.saveTensor dO (fixtureFile "dO").toString
  torch.data.saveTensor expectedOut (fixtureFile "expected_o").toString
  torch.data.saveTensor expectedL (fixtureFile "expected_l").toString
  torch.data.saveTensor expectedDQ (fixtureFile "expected_dq").toString
  torch.data.saveTensor expectedDK (fixtureFile "expected_dk").toString
  torch.data.saveTensor expectedDV (fixtureFile "expected_dv").toString

  let outMean := torch.nn.item (torch.nn.meanAll expectedOut)
  let lMean := torch.nn.item (torch.nn.meanAll expectedL)
  let dqMean := torch.nn.item (torch.nn.meanAll expectedDQ)
  IO.println s!"Generated mha_gb10 fixtures in {fixtureSpec.dir} outMean={outMean} lMean={lMean} dqMean={dqMean}"

def runOnce : IO Bool := do
  if !(← requireCuda suiteName) then
    return false

  if !(← isBlackwellFamily) then
    IO.println s!"[skip] {suiteName}: requires TYR_GPU_FAMILY=BLACKWELL"
    return true

  if !(← fixturesPresent fixtureSpec) then
    generateFixtures

  let stream ← torch.cuda_current_stream

  let q ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "q").toString
  let k ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "k").toString
  let v ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "v").toString
  let dO ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "dO").toString
  let expectedOut ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_o").toString
  let expectedL ← torch.data.loadTensor #[2, 64] (fixtureFile "expected_l").toString
  let expectedDQ ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dq").toString
  let expectedDK ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dk").toString
  let expectedDV ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dv").toString

  let outLse := torch.zeros_like q
  let lseOut := torch.zeros #[2, 64] false (Device.CUDA 0)
  tkFlashAttnGb10Fwd2BlockLse.launch q k v outLse lseOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let lFromLse : T #[2, 64] := torch.mul_scalar lseOut (-8.0)

  let out := torch.zeros_like q
  let lOut : T #[2, 64] := torch.zeros_like lseOut
  tkMhaGb10Fwd2Block.launch q k v out lOut 128 64 1 8 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dVec : T #[2, 64] := torch.mul_scalar lOut 0.0
  tkMhaGb10BwdPrep2Block.launch dO out dVec 128 64 1 8 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dQ := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  let dKSeed := torch.ones #[1, 1, 128, 64] false (Device.CUDA 0)
  let dK : T #[1, 1, 128, 64] := torch.mul_scalar dKSeed 0.0
  let dV : T #[1, 1, 128, 64] := torch.add_scalar dK 0.0
  tkMhaGb10BwdDQDirect.launch q k v dO lOut dVec dQ 128 64 1 8 1 32 1 1 0 stream
  tkMhaGb10BwdDKDVDirect.launch q k v dO lOut dVec dK dV 128 64 1 8 1 32 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let outCheck := compareTensors "mha_gb10.forward" expectedOut out 3e-2 3e-2
  let lCheck := compareTensors "mha_gb10.l" expectedL lOut 3e-2 3e-2
  let lKernelCheck := compareTensors "mha_gb10.l_vs_lse_kernel" lFromLse lOut 3e-2 3e-2
  let lFixtureCheck := compareTensors "mha_gb10.lse_kernel_vs_fixture" expectedL lFromLse 3e-2 3e-2
  let dqCheck := compareTensors "mha_gb10.dq" expectedDQ dQ 3e-2 3e-2
  let dkCheck := compareTensors "mha_gb10.dk" expectedDK dK 3e-2 3e-2
  let dvCheck := compareTensors "mha_gb10.dv" expectedDV dV 3e-2 3e-2
  for check in #[outCheck, lCheck, lKernelCheck, lFixtureCheck, dqCheck, dkCheck, dvCheck] do
    logTensorCheck check

  pure (outCheck.ok && lCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok)

private structure ForwardShape where
  batch : UInt64
  qHeads : UInt64
  seqLen : UInt64

private def forwardMatrix : LeanBenchmark.Matrix ForwardShape := {
  defaultProfile := "gb10-realistic"
  cases := #[
    { id := "s64", payload := ⟨1, 1, 64⟩, profiles := #["sequence-sweep"], tags := #["edge", "launch-bound"] },
    { id := "s128", payload := ⟨1, 1, 128⟩, profiles := #["quick", "sequence-sweep"], tags := #["small"] },
    { id := "s256", payload := ⟨1, 1, 256⟩, profiles := #["sequence-sweep"], tags := #["medium"] },
    { id := "s512", payload := ⟨1, 1, 512⟩, profiles := #["sequence-sweep"], tags := #["medium"] },
    { id := "s768", payload := ⟨1, 1, 768⟩, profiles := #["sequence-sweep"], tags := #["training", "micro"] },
    { id := "b1_h16_s768", payload := ⟨1, 16, 768⟩, profiles := #["gb10-realistic", "model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "latency"] },
    { id := "b2_h16_s768", payload := ⟨2, 16, 768⟩, profiles := #["model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "throughput"] },
    { id := "b4_h16_s768", payload := ⟨4, 16, 768⟩, profiles := #["model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "throughput"] },
    { id := "b8_h16_s768", payload := ⟨8, 16, 768⟩, profiles := #["model-shapes", "batch-sweep"], tags := #["training", "specialized", "multi-head", "throughput"] },
    { id := "s1024", payload := ⟨1, 1, 1024⟩, profiles := #["sequence-sweep"], tags := #["large"] },
    { id := "s2048", payload := ⟨1, 1, 2048⟩, profiles := #["sequence-sweep"], tags := #["large"] }
  ]
}

private def benchmarkForwardSeq (cfg : Benchmark.Config) (stream : UInt64)
    (shape : ForwardShape) : IO (String × Bool) := do
  let { batch, qHeads, seqLen } := shape
  let device := Device.CUDA 0
  let qf ← torch.randn #[batch, qHeads, seqLen, 64] false device
  let kf ← torch.randn #[batch, qHeads, seqLen, 64] false device
  let vf ← torch.randn #[batch, qHeads, seqLen, 64] false device
  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  let expected := torch.nn.scaled_dot_product_attention q k v 0.0 false
  let q32 := torch.toFloat' q
  let k32 := torch.toFloat' k
  let kT : T #[batch, qHeads, 64, seqLen] := torch.nn.transpose k32 2 3
  let scores : T #[batch, qHeads, seqLen, seqLen] := torch.nn.bmm4d q32 kT
  let scaled : T #[batch, qHeads, seqLen, seqLen] := scores / 8.0
  let expScores : T #[batch, qHeads, seqLen, seqLen] := torch.nn.exp scaled
  let sumExp : T #[batch, qHeads, seqLen] := torch.nn.sumDim expScores 3 false
  let expectedL3 : T #[batch, qHeads, seqLen] :=
    torch.mul_scalar (torch.nn.log sumExp) (-8.0)
  let out := torch.zeros_like q
  let useS768Specialization := seqLen == 768
  let useLongSchedule := seqLen >= 2048
  let rowTileSize : UInt64 := if useS768Specialization then 32 else if useLongSchedule then 32 else 16
  let queryBlocks := (seqLen + rowTileSize - 1) / rowTileSize
  let launchQueryBlocks := if useS768Specialization then seqLen / (4 * rowTileSize) else queryBlocks
  let lBlocks := if useS768Specialization then batch * qHeads * queryBlocks else queryBlocks
  let lOut : T #[lBlocks, rowTileSize] := torch.zeros #[lBlocks, rowTileSize] false device
  let expectedL : T #[lBlocks, rowTileSize] := torch.reshape expectedL3 #[lBlocks, rowTileSize]
  let launch : IO Unit := if useS768Specialization then
    tkMhaGb10FwdS768D64Warp4.launch q k v out lOut seqLen 64 qHeads
      launchQueryBlocks qHeads batch 128 1 1 0 stream
  else if useLongSchedule then
    tkMhaGb10FwdLong.launch q k v out lOut seqLen 64 1 queryBlocks 1 128 1 1 0 stream
  else
    tkMhaGb10Fwd2Block.launch q k v out lOut seqLen 64 1 queryBlocks 1 128 1 1 0 stream
  launch
  torch.cuda_synchronize
  let check := compareTensors s!"mha_gb10.bench.forward.s{seqLen}" expected out 3e-2 3e-2
  let lCheck := compareTensors s!"mha_gb10.bench.l.s{seqLen}" expectedL lOut 3e-2 3e-2
  logTensorCheck check
  logTensorCheck lCheck
  let samples ← Benchmark.timeCudaEvents cfg stream launch
  let line := Benchmark.summaryJson cfg s!"mha_forward_b{batch}_h{qHeads}_s{seqLen}_d64" "tyr"
    (if useS768Specialization then "generated_gb10_specialized_s768_d64_warp4_rows32_splitkv_tma1_exp2"
     else if useLongSchedule then "generated_gb10_specialized_rows32_d64"
     else "generated_gb10_specialized_rows16_d64") samples (check.ok && lCheck.ok)
    "kernel_only" true (some (batch * seqLen).toFloat) (some "tokens")
  pure (line, check.ok && lCheck.ok)

private def benchmarkBackwardS768 (cfg : Benchmark.Config) (stream : UInt64) (batch : UInt64)
    : IO (Array String × Bool) := do
  let device := Device.CUDA 0
  let shape := #[batch, 16, 768, 64]
  let qf ← torch.randn shape false device
  let kf ← torch.randn shape false device
  let vf ← torch.randn shape false device
  let dOf ← torch.randn shape false device
  let q := torch.toBFloat16' qf
  let k := torch.toBFloat16' kf
  let v := torch.toBFloat16' vf
  let dO := torch.toBFloat16' dOf
  let qRef := torch.autograd.set_requires_grad q true
  let kRef := torch.autograd.set_requires_grad k true
  let vRef := torch.autograd.set_requires_grad v true
  let outRef := torch.nn.scaled_dot_product_attention qRef kRef vRef 0.0 false
  torch.autograd.backward outRef dO
  let expectedDQ := torch.toFloat' (torch.autograd.grad_of qRef)
  let expectedDK := torch.toFloat' (torch.autograd.grad_of kRef)
  let expectedDV := torch.toFloat' (torch.autograd.grad_of vRef)
  let out := torch.zeros_like q
  let lOut32 : T #[batch * 384, 32] := torch.zeros #[batch * 384, 32] false device
  let launchForward := tkMhaGb10FwdS768D64Warp4.launch q k v out lOut32 768 64 16
    6 16 batch 128 1 1 0 stream
  launchForward
  let lOut : T #[batch * 768, 16] := torch.reshape lOut32 #[batch * 768, 16]
  torch.cuda_synchronize
  let dVec : T #[batch * 768, 16] := torch.mul_scalar lOut 0.0
  let dO32 := torch.toFloat' dO
  let out32 := torch.toFloat' out
  let expectedD3 : T #[batch, 16, 768] := torch.nn.sumDim (dO32 * out32) 3 false
  let expectedD : T #[batch * 768, 16] := torch.reshape expectedD3 #[batch * 768, 16]
  let dQ : T #[batch, 16, 768, 64] := torch.mul_scalar q 0.0
  let dK : T #[batch, 16, 768, 64] := torch.mul_scalar k 0.0
  let dV : T #[batch, 16, 768, 64] := torch.mul_scalar v 0.0
  let launchDQ := tkMhaGb10BwdDQS768D64.launch q k v dO out lOut dVec dQ 768 64 16
    12 16 batch 128 1 1 0 stream
  let launchDKDV := tkMhaGb10BwdDKDVS768D64.launch q k v dO lOut dVec dK dV 768 64 16
    12 16 batch 128 1 1 0 stream
  let launch := do
    launchDQ
    launchDKDV
  let launchTraining := do
    launchForward
    launch
  launch
  torch.cuda_synchronize
  let dCheck := compareTensors "mha_gb10.bench.s768.d" expectedD dVec 3e-2 3e-2
  let dqCheck := compareTensors "mha_gb10.bench.s768.dq" expectedDQ (torch.toFloat' dQ) 3e-2 3e-2
  let dkCheck := compareTensors "mha_gb10.bench.s768.dk" expectedDK (torch.toFloat' dK) 3e-2 3e-2
  let dvCheck := compareTensors "mha_gb10.bench.s768.dv" expectedDV (torch.toFloat' dV) 3e-2 3e-2
  for check in #[dCheck, dqCheck, dkCheck, dvCheck] do logTensorCheck check
  let ok := dCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok
  let fullSamples ← Benchmark.timeCudaEvents cfg stream launch
  let trainingSamples ← Benchmark.timeCudaEvents cfg stream launchTraining
  let dqSamples ← Benchmark.timeCudaEvents cfg stream launchDQ
  let dkdvSamples ← Benchmark.timeCudaEvents cfg stream launchDKDV
  let workItems := some (batch * 768).toFloat
  let workUnit := some "tokens"
  let dqLine := Benchmark.summaryJson cfg s!"mha_backward_dq_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_dq_fusedprep_rvshuffle_frag16_kv32x2_bf16grad" dqSamples ok "kernel_only" true workItems workUnit
  let dkdvLine := Benchmark.summaryJson cfg s!"mha_backward_dkdv_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_dkdv_warp4_tma_vrow_q48stream_qdo3_bf16grad" dkdvSamples ok
    "kernel_only" true workItems workUnit
  let fullLine := Benchmark.summaryJson cfg s!"mha_backward_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_dq_fusedprep_rvshuffle_frag16_kv32x2_dkdv_vrow_q48stream_qdo3_bf16grad" fullSamples ok
    "full_generated_kernel_sequence" true workItems workUnit
  let trainingLine := Benchmark.summaryJson cfg s!"mha_training_step_b{batch}_h16_s768_d64" "tyr"
    "generated_gb10_s768_fwd_saved_lse_dq_fusedprep_rvshuffle_frag16_kv32x2_dkdv_q48stream_bf16grad" trainingSamples ok
    "forward_saved_state_plus_backward_kernel_sequence" true workItems workUnit
  pure (#[dqLine, dkdvLine, fullLine, trainingLine], ok)

private def runBenchmark (args : List String) : IO UInt32 := do
  if !(← requireCuda suiteName) then return 1
  if !(← isBlackwellFamily) then
    IO.eprintln "[mha_gb10] benchmark requires TYR_GPU_FAMILY=BLACKWELL"
    return 1
  if !(← fixturesPresent fixtureSpec) then generateFixtures
  -- Keep correctness inputs stable across candidate/control benchmark runs.
  torch.manualSeed 0
  let cfg ← Benchmark.parseConfig args "mha_gb10_bench"
  let stream ← torch.cuda_current_stream
  let selection := LeanBenchmark.MatrixSelection.parse args forwardMatrix.defaultProfile
  let selected ← match forwardMatrix.select selection with
    | .ok cases => pure cases
    | .error msg => throw <| IO.userError msg
  let mut forwardLines : Array String := #[]
  let mut forwardOk := true
  for matrixCase in selected do
    let (line, ok) ← benchmarkForwardSeq cfg stream matrixCase.payload
    forwardLines := forwardLines.push line
    forwardOk := forwardOk && ok
  let mut s768BackwardLines : Array String := #[]
  let mut s768BackwardOk := true
  for matrixCase in selected do
    if matrixCase.payload.seqLen == 768 && matrixCase.payload.qHeads == 16 then
      let (lines, ok) ← benchmarkBackwardS768 cfg stream matrixCase.payload.batch
      s768BackwardLines := s768BackwardLines ++ lines
      s768BackwardOk := s768BackwardOk && ok
  let q ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "q").toString
  let k ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "k").toString
  let v ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "v").toString
  let expectedOut ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_o").toString
  let dO ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "dO").toString
  let expectedDQ ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dq").toString
  let expectedDK ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dk").toString
  let expectedDV ← torch.data.loadTensor #[1, 1, 128, 64] (fixtureFile "expected_dv").toString
  let out := torch.zeros_like q
  let lOut : T #[2, 64] := torch.zeros #[2, 64] false (Device.CUDA 0)
  let launch := tkMhaGb10Fwd2Block.launch q k v out lOut
    128 64 1 8 1 128 1 1 0 stream
  launch
  torch.cuda_synchronize
  let check := compareTensors "mha_gb10.bench.forward" expectedOut out 3e-2 3e-2
  logTensorCheck check
  let dVec : T #[2, 64] := torch.mul_scalar lOut 0.0
  let dO32 := torch.toFloat' dO
  let out32 := torch.toFloat' out
  let expectedD3 : T #[1, 1, 128] := torch.nn.sumDim (dO32 * out32) 3 false
  let expectedD : T #[2, 64] := torch.reshape expectedD3 #[2, 64]
  let dQ := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  -- Keep the mutable kernel outputs syntactically distinct: identical pure
  -- allocation expressions can be commoned by Lean and alias each other.
  let dKSeed := torch.ones #[1, 1, 128, 64] false (Device.CUDA 0)
  let dK : T #[1, 1, 128, 64] := torch.mul_scalar dKSeed 0.0
  let dV : T #[1, 1, 128, 64] := torch.add_scalar dK 0.0
  let launchBackward := do
    tkMhaGb10BwdPrep2Block.launch dO out dVec 128 64 1 8 1 32 1 1 0 stream
    tkMhaGb10BwdDQDirect.launch q k v dO lOut dVec dQ
      128 64 1 8 1 32 1 1 0 stream
    tkMhaGb10BwdDKDVDirect.launch q k v dO lOut dVec dK dV
      128 64 1 8 1 32 1 1 0 stream
  launchBackward
  torch.cuda_synchronize
  let dCheck := compareTensors "mha_gb10.bench.d" expectedD dVec 3e-2 3e-2
  let dqCheck := compareTensors "mha_gb10.bench.dq" expectedDQ dQ 3e-2 3e-2
  let dkCheck := compareTensors "mha_gb10.bench.dk" expectedDK dK 3e-2 3e-2
  let dvCheck := compareTensors "mha_gb10.bench.dv" expectedDV dV 3e-2 3e-2
  for gradCheck in #[dCheck, dqCheck, dkCheck, dvCheck] do logTensorCheck gradCheck
  let backwardOk := dCheck.ok && dqCheck.ok && dkCheck.ok && dvCheck.ok
  let backwardSamples ← Benchmark.timeCudaEvents cfg stream launchBackward
  let backwardLine := Benchmark.summaryJson cfg "mha_backward_b1_h1_s128_d64" "tyr"
    "generated_gb10_prep_direct_dq_dkdv" backwardSamples backwardOk
    "full_generated_kernel_sequence" true
  for line in forwardLines do IO.println line
  IO.println backwardLine
  for line in s768BackwardLines do IO.println line
  let allLines := (forwardLines.push backwardLine) ++ s768BackwardLines
  match cfg.jsonlOut? with
  | some path => Benchmark.writeJsonl path allLines
  | none => pure ()
  pure (if forwardOk && check.ok && backwardOk && s768BackwardOk then 0 else 1)

def main (args : List String) : IO UInt32 := do
  if args.contains "--benchmark" then runBenchmark args else
    runWithFixtures args suiteName fixtureSpec generateFixtures runOnce

end Examples.GPU.RunMhaGB10

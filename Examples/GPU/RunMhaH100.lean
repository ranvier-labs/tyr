/- End-to-end `mha_h100`-style forward/backward validation for the reduced
   2-block (`seq=128`, `d=64`) kernel variant in this repo. -/
import Tyr.Torch
import Tyr.GPU.Kernels.MhaH100
import Examples.GPU.FixtureRunner

namespace Examples.GPU.RunMhaH100

open torch
open Tyr.GPU.Kernels

private abbrev MhaTensor := T #[1, 1, 128, 64]
private abbrev LTensor := T #[2, 64]
private abbrev GradTiles := T #[2, 64, 64]

private def contractLabel : String := "dq_direct_kv_sweep_store_add"
private def seqLen : Nat := 128
private def headDim : Nat := 64
private def kvTiles : Nat := 2
def suiteName : String := "mha_h100"

def fixtureSpec : FixtureSpec := {
  dir := ⟨"data/gpu_fixtures/mha_h100_128x64"⟩
  names := #[
    "q", "k", "v", "dO",
    "expected_o", "expected_l",
    "expected_dq", "expected_dk", "expected_dv"
  ]
}

def fixtureFile (name : String) : System.FilePath :=
  Examples.GPU.fixturePath fixtureSpec name

private def partialDumpFile (name : String) : System.FilePath :=
  fixtureSpec.dir / name

private def dumpAccumulatedGrads (dK dV : MhaTensor) : IO Unit := do
  let dKTiles : GradTiles := torch.reshape dK #[2, 64, 64]
  let dVTiles : GradTiles := torch.reshape dV #[2, 64, 64]
  let dKPath := partialDumpFile "diag_dK_accum_tiles.pt"
  let dVPath := partialDumpFile "diag_dV_accum_tiles.pt"
  torch.data.saveTensor dKTiles dKPath.toString
  torch.data.saveTensor dVTiles dVPath.toString
  IO.println s!"mha_h100 grad_dump=true dK_tiles={dKPath} dV_tiles={dVPath}"

def generateFixtures : IO Unit := do
  if !(← torch.cuda_is_available) then
    throw <| IO.userError "CUDA is not available; cannot generate mha_h100 fixtures."

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

  -- Forward reference output.
  let expectedOut := torch.nn.scaled_dot_product_attention q k v 0.0 false

  -- ThunderKittens-style L vector: L = -8 * lse for d=64.
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

  -- Backward references from PyTorch autograd.
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
  IO.println s!"Generated mha_h100 fixtures in {fixtureSpec.dir} outMean={outMean} lMean={lMean} dqMean={dqMean}"

def runOnce (dumpPartials : Bool := false) : IO Bool := do
  if !(← torch.cuda_is_available) then
    IO.eprintln "CUDA is not available on this host."
    return false

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

  -- Forward.
  -- Cross-check L against the existing LSE kernel path.
  let outLse := torch.zeros_like q
  let lseOut := torch.zeros #[2, 64] false (Device.CUDA 0)
  tkFlashAttnFwd2BlockLse.launch q k v outLse lseOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  let lFromLse : T #[2, 64] := torch.mul_scalar lseOut (-8.0)

  let out := torch.zeros_like q
  let lOut : T #[2, 64] := torch.zeros_like lseOut
  tkMhaH100Fwd2Block.launch q k v out lOut 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  -- Backward prep + backward.
  -- Keep distinct allocation expressions for mutable outputs; these FFI tensors
  -- are mutated by kernels even though constructors are pure at the Lean level.
  let dVec : T #[2, 64] := torch.mul_scalar lOut 0.0
  tkMhaH100BwdPrep2Block.launch dO out dVec 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize

  let dQ : MhaTensor := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  let dK : MhaTensor := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  let dV : MhaTensor := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
  tkMhaH100Bwd2BlockDq.launch q k v dO lOut dVec dQ 128 64 1 2 1 128 1 1 0 stream
  tkMhaH100Bwd2BlockKvSweep.launch q k v dO lOut dVec dQ dK dV 128 64 1 2 1 128 1 1 0 stream
  let _ ← torch.cuda_synchronize
  if dumpPartials then
    dumpAccumulatedGrads dK dV

  -- Validate.
  let outRefOk := torch.allclose expectedOut out 3e-2 3e-2
  let lRefOk := torch.allclose expectedL lOut 3e-2 3e-2
  let lVsLseOk := torch.allclose lFromLse lOut 3e-2 3e-2
  let lseRefOk := torch.allclose expectedL lFromLse 3e-2 3e-2
  let dqRefOk := torch.allclose expectedDQ dQ 3e-2 3e-2
  let dkRefOk := torch.allclose expectedDK dK 3e-2 3e-2
  let dvRefOk := torch.allclose expectedDV dV 3e-2 3e-2
  let kernelRefOk := outRefOk && lRefOk && dqRefOk && dkRefOk && dvRefOk
  let lRouteOk := lVsLseOk && lseRefOk
  let overallOk := kernelRefOk && lRouteOk

  let outMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (out - expectedOut)))
  let outMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (out - expectedOut)))
  let lMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lOut - expectedL)))
  let lMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (lOut - expectedL)))
  let lKernelMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lOut - lFromLse)))
  let lFixtureMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (lFromLse - expectedL)))
  let dqMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dQ - expectedDQ)))
  let dqMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dQ - expectedDQ)))
  let dkMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dK - expectedDK)))
  let dkMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dK - expectedDK)))
  let dvMae := torch.nn.item (torch.nn.meanAll (torch.nn.abs (dV - expectedDV)))
  let dvMaxErr := torch.nn.item (torch.nn.maxAll (torch.nn.abs (dV - expectedDV)))
  IO.println s!"mha_h100 contract={contractLabel} seq={seqLen} head_dim={headDim} kv_tiles={kvTiles} overall_ok={overallOk} kernel_ref_ok={kernelRefOk} l_route_ok={lRouteOk} out_ref_ok={outRefOk} l_ref_ok={lRefOk} l_vs_lse_ok={lVsLseOk} lse_ref_ok={lseRefOk} dq_ref_ok={dqRefOk} dk_ref_ok={dkRefOk} dv_ref_ok={dvRefOk} out_mae={outMae} out_max={outMaxErr} l_mae={lMae} l_max={lMaxErr} l_kernel_mae={lKernelMae} l_fixture_mae={lFixtureMae} dq_mae={dqMae} dq_max={dqMaxErr} dk_mae={dkMae} dk_max={dkMaxErr} dv_mae={dvMae} dv_max={dvMaxErr}"

  pure overallOk

def main (args : List String) : IO UInt32 := do
  let dumpPartials := args.contains "--dump-partials"
  runWithFixtures args suiteName fixtureSpec generateFixtures (runOnce dumpPartials)

end Examples.GPU.RunMhaH100

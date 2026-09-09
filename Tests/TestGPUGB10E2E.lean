import LeanTest
import Tests.GPUCoverage
import Tyr.Torch
import Examples.GPU.Parity
import Examples.GPU.RunLayerNorm
import Examples.GPU.RunRMSNorm
import Examples.GPU.RunMhaGB10
import Examples.GPU.RunRKCombine
import Examples.GPU.RunRKFusedSolve
import Examples.GPU.RunBrownianSample
import Examples.GPU.RunBrownianDescent
import Examples.GPU.RunEulerMaruyamaFused

namespace Tests.GPUGB10E2E

open LeanTest
open torch

private def runGpuBoolTestIf
    (label : String)
    (enabled : IO Bool)
    (reason : String)
    (action : IO Bool)
    : IO Unit := do
  if !(← torch.cuda_is_available) then
    Tests.GPUCoverage.skip label "CUDA unavailable"
  else if !(← enabled) then
    Tests.GPUCoverage.skip label reason
  else
    let ok ← action
    LeanTest.assertTrue ok label

@[test]
def testMhaGB10TorchParity : IO Unit :=
  runGpuBoolTestIf
    "mha_gb10_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunMhaGB10.runOnce

@[test]
def testLayerNormGB10Float32TorchParity : IO Unit :=
  runGpuBoolTestIf
    "layernorm_gb10_f32_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunLayerNorm.runFloat32Once

@[test]
def testLayerNormGB10BFloat16TorchParity : IO Unit :=
  runGpuBoolTestIf
    "layernorm_gb10_bf16_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunLayerNorm.runBFloat16Once

@[test]
def testRMSNormGB10Float32TorchParity : IO Unit :=
  runGpuBoolTestIf
    "rmsnorm_gb10_f32_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRMSNorm.runFloat32Once

@[test]
def testRMSNormGB10BFloat16TorchParity : IO Unit :=
  runGpuBoolTestIf
    "rmsnorm_gb10_bf16_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRMSNorm.runBFloat16Once

@[test]
def testRKCombineGB10TorchParity : IO Unit :=
  runGpuBoolTestIf
    "rk_combine_gb10_tyr_vs_torch"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRKCombine.runOnce

@[test]
def testRKFusedSolveGB10Parity : IO Unit :=
  runGpuBoolTestIf
    "rk_fused_solve_gb10_vs_generic"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunRKFusedSolve.runOnce

@[test]
def testKeyedNormalGB10CpuParity : IO Unit :=
  runGpuBoolTestIf
    "keyed_normal_gb10_vs_cpu"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunBrownianSample.runOnce

@[test]
def testBrownianDescentGB10CpuParity : IO Unit :=
  runGpuBoolTestIf
    "brownian_descent_gb10_vs_cpu"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunBrownianDescent.runOnce

@[test]
def testEulerMaruyamaFusedGB10CpuParity : IO Unit :=
  runGpuBoolTestIf
    "euler_maruyama_fused_gb10_vs_cpu"
    Examples.GPU.isBlackwellFamily
    "requires TYR_GPU_FAMILY=BLACKWELL"
    Examples.GPU.RunEulerMaruyamaFused.runOnce

end Tests.GPUGB10E2E

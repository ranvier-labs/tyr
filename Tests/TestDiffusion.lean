/-
  Tests for Diffusion Model Implementation
-/
import Tyr
import Examples.Diffusion.Diffusion
import Examples.Diffusion.DiffusionSchedule
import Examples.Diffusion.DiffusionTrain
import LeanTest

namespace Tests.Diffusion

open torch
open torch.diffusion



@[test]
def testMaskSchedule : IO Unit := do
  let schedule := MaskedDiffusionSchedule.init 64 0 16
  -- Verify mask_probs goes from 1/64 to ~1.0
  let firstProb := nn.item (data.slice1d (n := 64) (m := 1) schedule.mask_probs 0 1)
  let lastProb := nn.item (data.slice1d (n := 64) (m := 1) schedule.mask_probs 63 64)
  LeanTest.assertTrue (firstProb > 0.01) "First prob > 0.01"
  LeanTest.assertTrue (firstProb < 0.05) "First prob < 0.05"
  LeanTest.assertTrue (lastProb > 0.95) "Last prob > 0.95"

@[test]
def testMaskPreservesContext : IO Unit := do
  let schedule := MaskedDiffusionSchedule.init 64 0 16  -- context_len=16
  -- Create random tokens (avoiding 0 which is mask)
  let x_0 ← randint 1 127 #[(2 : UInt64), 128]
  -- High timestep = lots of masking
  let t := full_int #[(2 : UInt64)] 63
  let x_t ← schedule.addMasks x_0 t
  -- Count total masks and check some were added
  let numMasked := nn.item (nn.sumAll (toFloat' (eq_scalar x_t 0)))
  -- With context_len=16 and seq=128, we have 112 non-context positions per sample
  -- At timestep 63 with linear schedule, mask_prob ~= 1.0, so most should be masked
  LeanTest.assertTrue (numMasked > 50) "Many positions are masked"
  let originalContext := data.slice x_0 1 0 16
  let maskedContext := data.slice x_t 1 0 16
  LeanTest.assertTrue (allclose originalContext maskedContext 0.0 0.0)
    "Every context token is preserved"

@[test]
def testMaskedLoss : IO Unit := do
  -- Create dummy logits and targets
  let logits ← randn #[(2 : UInt64), 32, 128] false
  let targets ← randint 0 127 #[(2 : UInt64), 32]
  let x_t := full_int #[(2 : UInt64), 32] 0  -- All masked
  let loss := maskedCrossEntropyLoss logits targets x_t 0
  let lossVal := nn.item loss
  LeanTest.assertTrue (not lossVal.isNaN) "Loss not NaN"
  LeanTest.assertTrue (lossVal > 0) "Loss positive"
  -- Test with no masked positions
  let x_t_none ← randint 1 127 #[(2 : UInt64), 32]  -- No masks
  let loss_none := maskedCrossEntropyLoss logits targets x_t_none 0
  let lossValNone := nn.item loss_none
  -- When no positions are masked, loss should be ~0 (masked loss / small epsilon)
  LeanTest.assertTrue (lossValNone < 1.0) "Loss with no masks is small"

@[test]
def testDiffusionForward : IO Unit := do
  let cfg := Config.tiny
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  -- Create input tensors
  let x_t ← randint 0 cfg.vocab_size.toInt64 #[(2 : UInt64), cfg.seq_len]
  let t ← randint 0 cfg.diffusion_steps.toInt64 #[(2 : UInt64)]
  let logits := forward params x_t t rotaryCache
  -- Verify output is finite
  let sum := nn.item (nn.sumAll logits)
  LeanTest.assertTrue (not sum.isNaN) "Output not NaN"
  LeanTest.assertTrue (not sum.isInf) "Output not Inf"

@[test]
def testSampleConfidence : IO Unit := do
  let cfg := Config.tiny
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  let result ← sampleConfidence (batch := 1) params rotaryCache (confidenceThreshold := 0.3) (maxSteps := 50)
  -- Check if any positions still masked
  let _anyMasked := any (eq_scalar result cfg.mask_token_id.toInt64)
  -- Note: with random weights, sampling may not fully decode
  -- Just verify it doesn't crash and produces valid output
  let sum := nn.item (nn.sumAll (toFloat' result))
  LeanTest.assertTrue (not sum.isNaN) "Output not NaN"

@[test]
def testSampleTopK : IO Unit := do
  let cfg := Config.tiny
  let params ← DiffusionParams.init cfg
  let rotaryCache ← RotaryCache.init cfg.seq_len cfg.headDim
  let result ← sampleTopK (batch := 1) params rotaryCache (k := 8) (maxSteps := 50)
  -- Verify output is valid
  let sum := nn.item (nn.sumAll (toFloat' result))
  LeanTest.assertTrue (not sum.isNaN) "Output not NaN"

@[test]
def testTrainStep : IO Unit := do
  manualSeed 20260908
  let modelCfg := Config.tiny
  let trainCfg : diffusion.train.TrainConfig := { maxIters := 1, batchSize := 2 }
  let params ← DiffusionParams.init modelCfg
  let rotaryCache ← RotaryCache.init modelCfg.seq_len modelCfg.headDim
  let schedule := MaskedDiffusionSchedule.init modelCfg.diffusion_steps modelCfg.mask_token_id modelCfg.context_len
  -- Initialize optimizer
  let opt := Optim.adamw (lr := 1e-4)
  let optState := opt.init params
  -- Create dummy batch
  let x_0 ← randint 1 127 #[trainCfg.batchSize, modelCfg.seq_len]
  let (params', _optState', loss) ← diffusion.train.trainStep trainCfg params optState schedule x_0 rotaryCache 1e-4
  let params' : DiffusionParams modelCfg := params'
  LeanTest.assertTrue (not (Float.isNaN loss)) "Loss not NaN"
  LeanTest.assertTrue (loss > 0) "Loss positive"
  -- Verify parameters changed
  let diffTensor := params.token_emb - params'.token_emb
  let diffSum := nn.item (nn.sumAll (nn.abs diffTensor))
  
  -- Parameters should change after training step
  LeanTest.assertTrue (diffSum > 1e-6) s!"Parameters updated (diff={diffSum})"
  -- The zero-initialized output head receives the first step's learning signal;
  -- embedding changes alone can be explained by decoupled weight decay.
  let headDiff := nn.item (nn.sumAll (nn.abs (params.output_head - params'.output_head)))
  LeanTest.assertTrue (headDiff > 1e-6) s!"Output head learned (diff={headDiff})"

end Tests.Diffusion

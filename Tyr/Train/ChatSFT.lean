/-
  Tyr/Train/ChatSFT.lean

  Supervised Fine-Tuning for chat models.

  Based on nanochat's chat_sft.py:
  - Task mixture training with masked loss
  - Linear LR decay schedule
  - Evaluation of validation loss and chat metrics

  Divergence from nanochat: `trainLoop` currently consumes only
  `embeddingLr` (scaled by `initLrFrac` and the linear decay multiplier)
  via a single AdamW optimizer applied to all parameters. The `matrixLr`,
  `unembeddingLr`, `weightDecay`, and `gradClip` config fields are parsed
  but unused: there is no Muon/Adam dual-optimizer split, no configurable
  weight decay, and no gradient clipping yet.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module
import Tyr.Optim
import Tyr.Distributed
import Tyr.Data.Task
import Tyr.Train.RunLedger

namespace torch.Train.ChatSFT

open torch
open torch.Data.Task
open torch.dist

/-! ## Configuration -/

/-- Chat SFT training configuration.
    Based on nanochat's chat_sft.py CLI arguments. -/
structure SFTConfig where
  /-- Number of training epochs -/
  numEpochs : Nat := 1
  /-- Override number of iterations (-1 = use numEpochs) -/
  numIterations : Int := -1
  /-- Per-device batch size -/
  deviceBatchSize : Nat := 4
  /-- Target examples per optimization step (for gradient accumulation) -/
  targetExamplesPerStep : Nat := 32
  /-- Learning rate for embedding parameters (Adam) -/
  embeddingLr : Float := 0.2
  /-- Learning rate for unembedding/LM head parameters (Adam) -/
  unembeddingLr : Float := 0.004
  /-- Learning rate for matrix parameters (Muon) -/
  matrixLr : Float := 0.02
  /-- Weight decay for Adam parameters -/
  weightDecay : Float := 0.0
  /-- Initial LR as fraction of base LR (for warmup) -/
  initLrFrac : Float := 0.02
  /-- Maximum sequence length -/
  maxSeqLen : Nat := 2048
  /-- Evaluate validation loss every N steps -/
  evalEvery : Nat := 100
  /-- Number of batches for validation loss evaluation -/
  evalSteps : Nat := 100
  /-- Evaluate chat metrics every N steps -/
  evalMetricsEvery : Nat := 200
  /-- Device to train on -/
  device : Device := Device.CPU
  /-- Gradient clipping (0 = disabled) -/
  gradClip : Float := 1.0
  /-- Logging interval (steps) -/
  logInterval : Nat := 10
  deriving Repr, Inhabited

/-- Compute gradient accumulation steps from config -/
def SFTConfig.gradAccumSteps (cfg : SFTConfig) (worldSize : Nat := 1) : Nat :=
  let examplesPerStep := cfg.deviceBatchSize * worldSize
  cfg.targetExamplesPerStep / examplesPerStep

/-! ## Learning Rate Schedule -/

/-- Linear LR decay schedule (nanochat style).
    Decays from 1.0 to 0.0 over num_iterations. -/
def linearLRMultiplier (step numIterations : Nat) : Float :=
  if numIterations == 0 then 1.0
  else 1.0 - step.toFloat / numIterations.toFloat

/-! ## Training State -/

/-- SFT training state tracking -/
structure SFTState where
  /-- Current step number -/
  step : Nat := 0
  /-- Current epoch number -/
  epoch : Nat := 0
  /-- Best validation loss seen -/
  bestValLoss : Float := 1e10
  /-- Running training loss for logging -/
  runningLoss : Float := 0.0
  /-- Number of tokens trained on -/
  totalTokens : Nat := 0
  deriving Repr, Inhabited

/-! ## SFT Batch -/

/-- SFT batch with inputs, targets, and mask -/
structure SFTBatch where
  /-- Input token IDs: [batch, seq] as Int64 tensor -/
  inputs : T #[]
  /-- Target token IDs: [batch, seq] as Int64 tensor -/
  targets : T #[]
  /-- Training mask: [batch, seq] as Float tensor (1.0 = train, 0.0 = ignore) -/
  mask : T #[]
  /-- Batch size -/
  batchSize : UInt64
  /-- Sequence length -/
  seqLen : UInt64
  /-- Number of valid (non-masked) target tokens -/
  numValidTokens : Nat
  deriving Repr

/-- Create an empty SFT batch -/
def SFTBatch.empty : SFTBatch :=
  { inputs := zeros #[0, 0]
  , targets := zeros #[0, 0]
  , mask := zeros #[0, 0]
  , batchSize := 0
  , seqLen := 0
  , numValidTokens := 0 }

/-- Move all tensor fields of an SFT batch to a target device. -/
private def moveBatchToDevice (batch : SFTBatch) (device : Device) : SFTBatch :=
  { batch with
    inputs := batch.inputs.to device
    targets := batch.targets.to device
    mask := batch.mask.to device
  }

/-- Convert a collated conversation batch into the SFT batch format. -/
def SFTBatch.ofConversationBatch (batch : ConversationBatch) : SFTBatch :=
  let shape := batch.tokens.runtimeShape
  let batchSize := shape.getD 0 0
  let seqLen := shape.getD 1 0
  let numValidInt := nn.itemInt (nn.sumAll batch.mask)
  let numValidTokens := if numValidInt <= 0 then 0 else numValidInt.toUInt64.toNat
  {
    inputs := batch.tokens
    targets := batch.tokens
    mask := batch.mask
    batchSize := batchSize
    seqLen := seqLen
    numValidTokens := numValidTokens
  }

/-! ## Masked Loss Computation -/

/-- Compute cross-entropy loss with mask.
    Only positions where mask > 0 contribute to the loss.

    logits: [batch, seq, vocab] - Model output logits
    targets: [batch, seq] - Target token IDs
    mask: [batch, seq] - Training mask (1.0 = train, 0.0 = ignore)

    Returns scalar loss averaged over valid positions.
-/
def maskedCrossEntropy {batch seq vocab : UInt64}
    (logits : T #[batch, seq, vocab])
    (targets : T #[batch, seq])
    (mask : T #[batch, seq])
    : T #[] :=
  -- Flatten for cross-entropy
  let batchSeq := batch * seq
  let logitsFlat := reshape logits #[batchSeq, vocab]
  let targetsFlat := reshape targets #[batchSeq]
  -- Compute per-element loss
  let lossesFlat := nn.cross_entropy_none logitsFlat targetsFlat
  -- Apply mask and compute mean over valid positions
  let maskFlat := reshape mask #[batchSeq]
  let maskedLosses := mul lossesFlat maskFlat
  -- Sum losses and divide by number of valid tokens
  let totalLoss := nn.sumAll maskedLosses
  let numValid := nn.sumAll maskFlat
  -- Divide by number of valid tokens to get mean loss
  -- Use mul_scalar with reciprocal to avoid needing tensor division
  let numValidFloat := nn.item numValid
  let epsFloat : Float := 1e-8
  mul_scalar totalLoss (1.0 / (numValidFloat + epsFloat))

/-! ## Data Preparation -/

/-- Prepare SFT batch from tokenized conversations.
    Creates input/target pairs where targets are shifted and masked.

    The mask ensures we only train on assistant responses.
    - Input: tokens[:-1]
    - Target: tokens[1:]
    - Mask: mask[1:] (shifted to align with targets)
-/
def prepareSFTBatch (convs : Array TokenizedConversation) (maxLen : Nat) (padTokenId : UInt64)
    : IO SFTBatch := do
  let batchSize := convs.size
  if batchSize == 0 then
    return SFTBatch.empty

  -- Find max length (minus 1 for shift, capped at maxLen)
  let seqLen := convs.foldl (fun acc c => max acc (min (c.tokens.size - 1) maxLen)) 0

  if seqLen == 0 then
    return SFTBatch.empty

  -- Build input, target, and mask arrays
  let mut allInputs : Array Int64 := #[]
  let mut allTargets : Array Int64 := #[]
  let mut allMasks : Array Float := #[]
  let mut numValidTokens : Nat := 0

  for conv in convs do
    let n := conv.tokens.size
    -- Input: tokens[:-1], Target: tokens[1:], Mask: mask[1:]
    for i in [:seqLen] do
      if i < n - 1 then
        -- Valid position
        allInputs := allInputs.push conv.tokens[i]!.toInt64
        allTargets := allTargets.push conv.tokens[i + 1]!.toInt64
        -- Mask for target position (i+1)
        let maskVal : Float := if i + 1 < conv.mask.size && conv.mask[i + 1]! == 1 then 1.0 else 0.0
        allMasks := allMasks.push maskVal
        if maskVal > 0 then
          numValidTokens := numValidTokens + 1
      else
        -- Padding position
        allInputs := allInputs.push padTokenId.toInt64
        allTargets := allTargets.push padTokenId.toInt64  -- Target doesn't matter, mask is 0
        allMasks := allMasks.push 0.0  -- Don't train on padding

  -- Convert to tensors
  let inputsTensor := data.fromInt64Array allInputs
  let inputsTensor := reshape inputsTensor #[batchSize.toUInt64, seqLen.toUInt64]

  let targetsTensor := data.fromInt64Array allTargets
  let targetsTensor := reshape targetsTensor #[batchSize.toUInt64, seqLen.toUInt64]

  -- Convert float mask to tensor via Int64 (workaround: no fromFloatArray)
  let maskInt := allMasks.map (fun m => if m > 0.5 then (1 : Int64) else 0)
  let maskTensor := data.fromInt64Array maskInt
  let maskTensor := toFloat' (reshape maskTensor #[batchSize.toUInt64, seqLen.toUInt64])

  return {
    inputs := inputsTensor
    targets := targetsTensor
    mask := maskTensor
    batchSize := batchSize.toUInt64
    seqLen := seqLen.toUInt64
    numValidTokens
  }

/-! ## Training Functions -/

/-- Single forward pass for SFT with masked loss.
    Returns loss tensor for backprop.

    forwardFn should take inputs [batch, seq] and return logits [batch, seq, vocab]
-/
def sftForward {batch seq vocab : UInt64}
    (forwardFn : T #[batch, seq] → IO (T #[batch, seq, vocab]))
    (inputs : T #[batch, seq])
    (targets : T #[batch, seq])
    (mask : T #[batch, seq])
    : IO (T #[]) := do
  -- Get logits from model
  let logits ← forwardFn inputs
  -- Compute masked loss
  return maskedCrossEntropy logits targets mask

/-- Evaluate validation loss over multiple batches -/
def evalValidationLoss
    (dataFn : IO SFTBatch)
    (forwardFn : T #[] → IO (T #[]))  -- Generic forward: inputs → logits
    (numSteps : Nat)
    (device : Device := Device.CPU)
    : IO Float := autograd.no_grad do
  let mut totalLoss : Float := 0.0
  let mut totalValid : Nat := 0

  for _ in [:numSteps] do
    let batch ← dataFn
    let batch := moveBatchToDevice batch device
    if batch.numValidTokens > 0 then
      let logits ← forwardFn batch.inputs
      -- Reshape for loss computation (dimensions come from batch)
      let batchSeq := batch.batchSize * batch.seqLen
      let logitsFlat := reshape logits #[batchSeq]
      let targetsFlat := reshape batch.targets #[batchSeq]
      let maskFlat := reshape batch.mask #[batchSeq]
      -- Compute loss using standard cross-entropy then mask
      let lossPerToken := nn.cross_entropy_none
        (reshape logitsFlat #[batchSeq, 1])  -- Placeholder vocab dim
        targetsFlat
      let maskedLoss := mul lossPerToken maskFlat
      let batchLoss := nn.sumAll maskedLoss
      totalLoss := totalLoss + nn.item batchLoss
      totalValid := totalValid + batch.numValidTokens

  if totalValid == 0 then
    return 0.0
  return totalLoss / totalValid.toFloat

/-- Logging callback type -/
def LogCallback := Nat → Float → Float → Nat → IO Unit

/-- Silent training-step callback. -/
def silentLogger : LogCallback := fun _ _ _ _ => pure ()

/-- Validation logging callback type. -/
def ValidationCallback := Nat → Float → IO Unit

/-- Silent validation callback. -/
def silentValidationLogger : ValidationCallback := fun _ _ => pure ()

/-- Render a human-readable training-step update. -/
def formatTrainUpdate (step : Nat) (trainLoss lrm : Float) (numTokens : Nat) : String :=
  s!"Step {step}: loss={trainLoss}, lrm={lrm}, tokens={numTokens}"

/-- Render a human-readable validation update. -/
def formatValidationUpdate (step : Nat) (valLoss : Float) : String :=
  s!"Step {step}: val_loss={valLoss}"

/-- Hook set for SFT training progress and evaluation. -/
structure Callbacks where
  onTrainStep : LogCallback := silentLogger
  onValidation : ValidationCallback := silentValidationLogger

/-- Compose two callback sets. -/
def Callbacks.combine (lhs rhs : Callbacks) : Callbacks := {
  onTrainStep := fun step trainLoss lrm numTokens => do
    lhs.onTrainStep step trainLoss lrm numTokens
    rhs.onTrainStep step trainLoss lrm numTokens
  onValidation := fun step valLoss => do
    lhs.onValidation step valLoss
    rhs.onValidation step valLoss
}

/-- Emit SFT metrics to the run ledger. -/
def artifactCallbacks
    (artifacts : RunLedger.RunArtifacts)
    (scopePrefix : String := "sft") : Callbacks := {
  onTrainStep := fun step trainLoss lrm numTokens => do
    RunLedger.appendMetricEvent artifacts {
      scope := s!"{scopePrefix}/train"
      step := some step
      metrics := [
        RunLedger.metricFloat "loss" trainLoss,
        RunLedger.metricFloat "lr_multiplier" lrm,
        RunLedger.metricNat "num_tokens" numTokens
      ]
    }
  onValidation := fun step valLoss => do
    RunLedger.appendMetricEvent artifacts {
      scope := s!"{scopePrefix}/eval"
      step := some step
      metrics := [RunLedger.metricFloat "val_loss" valLoss]
    }
}

/-! ## Main Training Loop -/

/-- Run SFT training loop.

    This implements the nanochat chat_sft.py training algorithm:
    1. Linear LR decay schedule
    2. Gradient accumulation
    3. Masked loss on assistant responses only
    4. Periodic validation evaluation

    Parameters:
    - cfg: Training configuration
    - params: Model parameters to train (must implement TensorStruct)
    - optState: Initial optimizer state
    - forwardFn: Forward pass function that takes params and inputs, returns logits
    - lossFn: Loss function that takes logits, targets, mask, returns scalar loss
    - trainDataFn: Function to get next training batch
    - valDataFn: Function to get next validation batch (optional)
    - numTrainExamples: Total training examples (for computing iterations)
    - logger: Logging callback
-/
def trainLoop [TensorStruct P]
    (cfg : SFTConfig)
    (params : P)
    (optState : Optim.AdamWState P)
    (forwardFn : P → T #[] → IO (T #[]))  -- params → inputs → logits
    (lossFn : T #[] → T #[] → T #[] → T #[])  -- logits → targets → mask → loss
    (trainDataFn : IO SFTBatch)
    (valDataFn : Option (IO SFTBatch) := none)
    (numTrainExamples : Nat := 10000)
    (callbacks : Callbacks := {})
    : IO (P × Optim.AdamWState P × SFTState) := do
  let isDistributed ← dist.isInitialized
  let worldSizeU64 ← if isDistributed then dist.getWorldSize else pure 1
  let worldSize := worldSizeU64.toNat

  -- Compute number of iterations
  let numIterations : Nat :=
    if cfg.numIterations >= 0 then
      cfg.numIterations.toNat
    else
      (numTrainExamples / cfg.targetExamplesPerStep) * cfg.numEpochs

  let gradAccumSteps := max 1 (cfg.gradAccumSteps worldSize)

  let mut currentParams := TensorStruct.makeLeafParams params
  let mut currentOptState := optState
  let mut state : SFTState := {}

  for step in [:numIterations] do
    let isLastStep := step == numIterations - 1
    state := { state with step }

    -- Validation evaluation
    if let some valFn := valDataFn then
      if isLastStep || (step > 0 && step % cfg.evalEvery == 0) then
        let valLoss ← evalValidationLoss valFn (forwardFn currentParams) cfg.evalSteps cfg.device
        callbacks.onValidation step valLoss
        if valLoss < state.bestValLoss then
          state := { state with bestValLoss := valLoss }

    if isLastStep then
      break

    -- Zero gradients at start of accumulation
    let workingParams := TensorStruct.zeroGrads (TensorStruct.makeLeafParams currentParams)

    -- Gradient accumulation loop
    let mut accumLoss : Float := 0.0
    let mut numTokens : Nat := 0

    for _ in [:gradAccumSteps] do
      let batch ← trainDataFn
      let batch := moveBatchToDevice batch cfg.device

      if batch.numValidTokens > 0 then
        -- Forward pass
        let logits ← forwardFn workingParams batch.inputs
        -- Compute masked loss
        let loss := lossFn logits batch.targets batch.mask
        -- Scale loss for accumulation
        let scaledLoss := mul_scalar loss (1.0 / gradAccumSteps.toFloat)
        -- Backward pass (gradients accumulate)
        autograd.backwardLoss scaledLoss

        accumLoss := accumLoss + nn.item loss
        numTokens := numTokens + batch.numValidTokens

    -- LR schedule: linear decay with initial warmup fraction
    let lrm := linearLRMultiplier step numIterations
    let lr := cfg.embeddingLr * cfg.initLrFrac * lrm

    -- Synchronize gradients across ranks in distributed training.
    if isDistributed then
      let gradsToSync := TensorStruct.grads workingParams
      let _ ← dist.allReduceGrads gradsToSync .avg
      pure ()

    -- Extract gradients and update parameters.
    let grads := TensorStruct.grads workingParams
    let opt := Optim.adamw (lr := lr) (b1 := 0.9) (b2 := 0.999)
    let (newParams, newOptState) := Optim.step opt workingParams grads currentOptState

    currentParams := newParams
    currentOptState := newOptState

    state := { state with
      totalTokens := state.totalTokens + numTokens
      runningLoss := state.runningLoss + accumLoss / gradAccumSteps.toFloat
    }

    -- Logging
    if step % cfg.logInterval == 0 then
      let avgLoss := if gradAccumSteps > 0
        then accumLoss / gradAccumSteps.toFloat
        else 0.0
      callbacks.onTrainStep step avgLoss lrm numTokens

  return (currentParams, currentOptState, state)

/-! ## Convenience Functions -/

/-- Create a training data generator from a TaskIterator.
    Yields SFTBatch on each call, handling epoch boundaries automatically. -/
def makeTaskDataGenerator
    (iter : TaskIterator)
    (_maxLen : Nat)
    (_padTokenId : UInt64)
    : IO (IO SFTBatch) := do
  let iterRef ← IO.mkRef iter
  return do
    let mut currentIter ← iterRef.get

    -- Get batch of tokenized conversations
    let (maybeBatch, newIter) ← currentIter.nextBatch
    currentIter := newIter

    -- Handle epoch boundary
    let batch ← match maybeBatch with
      | none =>
        -- Epoch ended, get first batch of new epoch
        let (maybeBatch', newerIter) ← currentIter.nextBatch
        currentIter := newerIter
        match maybeBatch' with
        | none => pure SFTBatch.empty
        | some convBatch => pure (SFTBatch.ofConversationBatch convBatch)
      | some convBatch =>
        pure (SFTBatch.ofConversationBatch convBatch)

    iterRef.set currentIter
    return batch

/-- Simple SFT training helper for models with standard interface.
    Wraps trainLoop with common defaults.

    Note: lossFn must handle the actual shape computation since
    we use shape-erased tensors (T #[]). -/
def trainSFT [TensorStruct P]
    (cfg : SFTConfig)
    (params : P)
    (forwardFn : P → T #[] → IO (T #[]))
    (lossFn : T #[] → T #[] → T #[] → T #[])
    (trainDataFn : IO SFTBatch)
    (numTrainExamples : Nat)
    : IO (P × SFTState) := do
  -- Initialize optimizer state via the adamw optimizer
  let opt := Optim.adamw (lr := cfg.embeddingLr) (b1 := 0.9) (b2 := 0.999)
  let optState := opt.init params

  let (finalParams, _, finalState) ← trainLoop cfg params optState forwardFn lossFn trainDataFn none numTrainExamples

  return (finalParams, finalState)

end torch.Train.ChatSFT

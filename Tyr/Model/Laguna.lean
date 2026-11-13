/-
  Tyr/Model/Laguna.lean

  Umbrella module for the poolside Laguna text causal-LM implementation
  (Laguna-S-2.1: 118B-A8B MoE, NVFP4 packed experts, interleaved
  full/sliding-window attention, per-head softplus output gating).
-/
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.ConfigIO
import Tyr.Model.Laguna.Rope
import Tyr.Model.Laguna.NvFp4
import Tyr.Model.Laguna.MoE
import Tyr.Model.Laguna.Model
import Tyr.Model.Laguna.Weights
import Tyr.Model.Laguna.Pretrained

/-! # Laguna

Umbrella import for the poolside Laguna hybrid-attention MoE model in
`torch.laguna`: configuration, dual RoPE tables, NVFP4 dequantization, the MoE
block, model, weights and pretrained loading.
-/

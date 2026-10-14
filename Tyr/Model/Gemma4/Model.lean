/-
  Tyr/Model/Gemma4/Model.lean

  Standalone Gemma 4 text causal-LM for Tyr.
  Supports:
  - Sliding + full attention hybrid schedule
  - Small-model KV sharing (E2B/E4B)
  - Per-layer input embeddings/projections (E2B/E4B)
  - Dense and MoE feed-forward variants
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Attention
import Tyr.Model.Gemma4.Config
import Tyr.Model.Generation

/-! # Gemma 4 text model

Gemma 4 causal language model: `torch.gemma4.Gemma4RMSNorm`, dense and
routed-expert MLPs, attention with the per-layer schedule from
`torch.gemma4.Gemma4Config`, and the decoder stack used by generation.
-/

namespace torch.gemma4

open torch
open torch.Model

private def reqGradFalse {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad t false

@[extern "lean_torch_gemma4_text_experts_forward"]
private opaque routedTextExpertsForward
    {tokens numExperts topK inter hidden : UInt64}
    (hiddenStates : @& T #[tokens, hidden])
    (gateUpProj : @& T #[numExperts, 2 * inter, hidden])
    (downProj : @& T #[numExperts, hidden, inter])
    (topVals : @& T #[tokens, topK])
    (topIdx : @& T #[tokens, topK])
    : T #[tokens, hidden]

/-- Gemma 4 RMSNorm with standard learned scale. -/
structure Gemma4RMSNorm (dim : UInt64) where
  weight : T #[dim]
  eps : Float := 1e-6
  deriving TensorStruct

namespace Gemma4RMSNorm

def fromCheckpointWeight {dim : UInt64}
    (weight : T #[dim])
    (eps : Float := 1e-6)
    : Gemma4RMSNorm dim :=
  { weight := weight, eps }

def forward2d {n dim : UInt64}
    (m : Gemma4RMSNorm dim)
    (x : T #[n, dim])
    : T #[n, dim] :=
  let xf : T #[n, dim] := toFloat' x
  let var : T #[n, 1] := nn.meanDim (xf * xf) 1 true
  let inv : T #[n, 1] := nn.rsqrt (var + m.eps)
  let inv : T #[n, dim] := nn.expand inv #[n, dim]
  let scale : T #[n, dim] := nn.expand (reshape (toFloat' m.weight) #[1, dim]) #[n, dim]
  restoreInputDType x (xf * inv * scale)

def forward3d {batch seq dim : UInt64}
    (m : Gemma4RMSNorm dim)
    (x : T #[batch, seq, dim])
    : T #[batch, seq, dim] :=
  let flat : T #[batch * seq, dim] := reshape x #[batch * seq, dim]
  reshape (forward2d m flat) #[batch, seq, dim]

def forward4d {a b c dim : UInt64}
    (m : Gemma4RMSNorm dim)
    (x : T #[a, b, c, dim])
    : T #[a, b, c, dim] :=
  let flat : T #[a * b * c, dim] := reshape x #[a * b * c, dim]
  reshape (forward2d m flat) #[a, b, c, dim]

end Gemma4RMSNorm

private def rmsNormNoScale2d {n dim : UInt64}
    (x : T #[n, dim])
    (eps : Float := 1e-6)
    : T #[n, dim] :=
  let xf : T #[n, dim] := toFloat' x
  let var : T #[n, 1] := nn.meanDim (xf * xf) 1 true
  let inv : T #[n, 1] := nn.rsqrt (var + eps)
  let inv : T #[n, dim] := nn.expand inv #[n, dim]
  restoreInputDType x (xf * inv)

private def rmsNormNoScale4d {a b c dim : UInt64}
    (x : T #[a, b, c, dim])
    (eps : Float := 1e-6)
    : T #[a, b, c, dim] :=
  let flat : T #[a * b * c, dim] := reshape x #[a * b * c, dim]
  reshape (rmsNormNoScale2d flat eps) #[a, b, c, dim]

private def rmsNormWeighted4dSlice {a b c dim maxDim : UInt64}
    (weight : T #[maxDim])
    (x : T #[a, b, c, dim])
    (eps : Float := 1e-6)
    : T #[a, b, c, dim] :=
  let w : T #[dim] := data.slice weight 0 0 dim
  let xf : T #[a * b * c, dim] := reshape x #[a * b * c, dim]
  let var : T #[a * b * c, 1] := nn.meanDim (xf * xf) 1 true
  let inv : T #[a * b * c, 1] := nn.rsqrt (var + eps)
  let inv : T #[a * b * c, dim] := nn.expand inv #[a * b * c, dim]
  let scale : T #[a * b * c, dim] := nn.expand (reshape (toFloat' w) #[1, dim]) #[a * b * c, dim]
  restoreInputDType x (reshape (xf * inv * scale) #[a, b, c, dim])

private def rmsNormNoScale4dSlice {a b c dim : UInt64}
    (x : T #[a, b, c, dim])
    (eps : Float := 1e-6)
    : T #[a, b, c, dim] :=
  rmsNormNoScale4d x eps

/-- Dense Gemma 4 MLP.
    Layers whose checkpoint width is smaller than `Config.maxIntermediateSize`
    are zero-padded during load. -/
structure Gemma4MLP (cfg : Config) where
  gate_proj : T #[Config.maxIntermediateSize cfg, cfg.hidden_size]
  up_proj : T #[Config.maxIntermediateSize cfg, cfg.hidden_size]
  down_proj : T #[cfg.hidden_size, Config.maxIntermediateSize cfg]
  deriving TensorStruct

namespace Gemma4MLP

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4MLP cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.hidden_size] :=
  linear3d (nn.gelu (linear3d x m.gate_proj) * linear3d x m.up_proj) m.down_proj

end Gemma4MLP

structure Gemma4TextRouter (cfg : Config) where
  proj : T #[cfg.num_experts, cfg.hidden_size]
  scale : T #[cfg.hidden_size]
  per_expert_scale : T #[cfg.num_experts]
  deriving TensorStruct

namespace Gemma4TextRouter

def forward {tokens : UInt64}
    (cfg : Config)
    (m : Gemma4TextRouter cfg)
    (hidden : T #[tokens, cfg.hidden_size])
    : T #[tokens, cfg.num_experts]
      × T #[tokens, cfg.top_k_experts]
      × T #[tokens, cfg.top_k_experts] :=
  let normed : T #[tokens, cfg.hidden_size] := rmsNormNoScale2d hidden cfg.rms_norm_eps
  let scaleRoot := 1.0 / Float.sqrt cfg.hidden_size.toFloat
  let scale : T #[tokens, cfg.hidden_size] :=
    nn.expand (reshape (toFloat' m.scale) #[1, cfg.hidden_size]) #[tokens, cfg.hidden_size]
  let normed : T #[tokens, cfg.hidden_size] := mul_scalar (normed * scale) scaleRoot
  let logitsDyn : T #[] := torch.einsum2 "eh,th->te" m.proj normed
  let logits : T #[tokens, cfg.num_experts] := reshape logitsDyn #[tokens, cfg.num_experts]
  let probs : T #[tokens, cfg.num_experts] := nn.softmax logits (-1)
  let (topValsRaw, topIdx) := torch.topk_2d probs cfg.top_k_experts 1
  let denom : T #[tokens, 1] := nn.sumDim topValsRaw 1 true
  let denom : T #[tokens, cfg.top_k_experts] := nn.expand denom #[tokens, cfg.top_k_experts]
  let topVals : T #[tokens, cfg.top_k_experts] := nn.div topValsRaw denom
  let flatIdx : T #[tokens * cfg.top_k_experts] := reshape topIdx #[tokens * cfg.top_k_experts]
  let scalesFlat : T #[tokens * cfg.top_k_experts] :=
    data.indexSelect (reshape m.per_expert_scale #[cfg.num_experts]) 0 flatIdx
  let scales : T #[tokens, cfg.top_k_experts] := reshape scalesFlat #[tokens, cfg.top_k_experts]
  (probs, topVals * scales, topIdx)

end Gemma4TextRouter

structure Gemma4TextExperts (cfg : Config) where
  gate_up_proj : T #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]
  down_proj : T #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]
  deriving TensorStruct

namespace Gemma4TextExperts

def forward2d {tokens : UInt64}
    (cfg : Config)
    (m : Gemma4TextExperts cfg)
    (hidden : T #[tokens, cfg.hidden_size])
    (topVals : T #[tokens, cfg.top_k_experts])
    (topIdx : T #[tokens, cfg.top_k_experts])
    : T #[tokens, cfg.hidden_size] :=
  routedTextExpertsForward hidden m.gate_up_proj m.down_proj topVals topIdx

end Gemma4TextExperts

structure Gemma4MoeBranch (cfg : Config) where
  router : Gemma4TextRouter cfg
  experts : Gemma4TextExperts cfg
  post_feedforward_layernorm_1 : Gemma4RMSNorm cfg.hidden_size
  post_feedforward_layernorm_2 : Gemma4RMSNorm cfg.hidden_size
  pre_feedforward_layernorm_2 : Gemma4RMSNorm cfg.hidden_size
  deriving TensorStruct

structure Gemma4PerLayerInputBlock (cfg : Config) where
  per_layer_input_gate : T #[cfg.hidden_size_per_layer_input, cfg.hidden_size]
  per_layer_projection : T #[cfg.hidden_size, cfg.hidden_size_per_layer_input]
  post_per_layer_input_norm : Gemma4RMSNorm cfg.hidden_size
  deriving TensorStruct

structure Gemma4Attention (cfg : Config) where
  q_proj : T #[Config.maxAttentionDim cfg, cfg.hidden_size]
  k_proj : T #[Config.maxKVProjDim cfg, cfg.hidden_size]
  v_proj : T #[Config.maxKVProjDim cfg, cfg.hidden_size]
  o_proj : T #[cfg.hidden_size, Config.maxAttentionDim cfg]
  q_norm : Gemma4RMSNorm (Config.maxHeadDim cfg)
  k_norm : Gemma4RMSNorm (Config.maxHeadDim cfg)
  deriving TensorStruct

namespace Gemma4Attention

abbrev KVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch (Config.maxKVHeads cfg) (Config.maxHeadDim cfg)

private def qProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] :=
  let qW : T #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size] :=
    data.slice m.q_proj 0 0 (cfg.num_attention_heads * cfg.head_dim)
  reshape (linear3d x qW) #[batch, seq, cfg.num_attention_heads, cfg.head_dim]

private def qProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] :=
  let dim := cfg.num_attention_heads * Config.fullHeadDim cfg
  let qW : T #[dim, cfg.hidden_size] := data.slice m.q_proj 0 0 dim
  reshape (linear3d x qW) #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg]

private def kProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let kW : T #[rows, cfg.hidden_size] := data.slice m.k_proj 0 0 rows
  reshape (linear3d x kW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def kProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let kW : T #[rows, cfg.hidden_size] := data.slice m.k_proj 0 0 rows
  reshape (linear3d x kW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def vProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let vW : T #[rows, cfg.hidden_size] := data.slice m.v_proj 0 0 rows
  reshape (linear3d x vW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def vProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let vW : T #[rows, cfg.hidden_size] := data.slice m.v_proj 0 0 rows
  reshape (linear3d x vW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def oProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim])
    : T #[batch, seq, cfg.hidden_size] :=
  let cols := cfg.num_attention_heads * cfg.head_dim
  let oW : T #[cfg.hidden_size, cols] := data.slice m.o_proj 1 0 cols
  linear3d x oW

private def oProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg])
    : T #[batch, seq, cfg.hidden_size] :=
  let cols := cfg.num_attention_heads * Config.fullHeadDim cfg
  let oW : T #[cfg.hidden_size, cols] := data.slice m.o_proj 1 0 cols
  linear3d x oW

private def padSlidingKVForCache {batch seq : UInt64}
    (cfg : Config)
    (x : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim])
    : T #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] :=
  let padDim : UInt64 := Config.maxHeadDim cfg - cfg.head_dim
  let xDim : T #[batch, cfg.num_key_value_heads, seq, Config.maxHeadDim cfg] :=
    nn.cat x (zerosOn x.device : T #[batch, cfg.num_key_value_heads, seq, padDim]) 3
  let padHeads : UInt64 := Config.maxKVHeads cfg - cfg.num_key_value_heads
  nn.cat xDim (zerosOn x.device : T #[batch, padHeads, seq, Config.maxHeadDim cfg]) 1

private def padFullKVForCache {batch seq : UInt64}
    (cfg : Config)
    (x : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg])
    : T #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] :=
  let padDim : UInt64 := Config.maxHeadDim cfg - Config.fullHeadDim cfg
  let xDim : T #[batch, Config.fullNumKVHeads cfg, seq, Config.maxHeadDim cfg] :=
    nn.cat x (zerosOn x.device : T #[batch, Config.fullNumKVHeads cfg, seq, padDim]) 3
  let padHeads : UInt64 := Config.maxKVHeads cfg - Config.fullNumKVHeads cfg
  nn.cat xDim (zerosOn x.device : T #[batch, padHeads, seq, Config.maxHeadDim cfg]) 1

private def cacheSliceSliding {batch kvLen : UInt64}
    (cfg : Config)
    (cache : KVCache cfg batch)
    : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim]
      × T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] :=
  let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let k0 : T #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg] :=
    data.slice (data.slice kStore 1 0 cfg.num_key_value_heads) 2 0 kvLen
  let v0 : T #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg] :=
    data.slice (data.slice vStore 1 0 cfg.num_key_value_heads) 2 0 kvLen
  let k : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] := data.slice k0 3 0 cfg.head_dim
  let v : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] := data.slice v0 3 0 cfg.head_dim
  (k, v)

private def cacheSliceFull {batch kvLen : UInt64}
    (cfg : Config)
    (cache : KVCache cfg batch)
    : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg]
      × T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] :=
  let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let k0 : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg] :=
    data.slice (data.slice kStore 1 0 (Config.fullNumKVHeads cfg)) 2 0 kvLen
  let v0 : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg] :=
    data.slice (data.slice vStore 1 0 (Config.fullNumKVHeads cfg)) 2 0 kvLen
  let k : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] :=
    data.slice k0 3 0 (Config.fullHeadDim cfg)
  let v : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] :=
    data.slice v0 3 0 (Config.fullHeadDim cfg)
  (k, v)

private def attentionScale (d : UInt64) : Float :=
  Float.sqrt d.toFloat

private def attentionSlidingPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    (attnMask : T #[batch, seq, seq])
    (cache : KVCache cfg batch)
    : T #[batch, seq, cfg.hidden_size] × KVCache cfg batch :=
  let q0 := qProjSliding cfg m x
  let kRaw := kProjSliding cfg m x
  let vRaw := vProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
  let q2 : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] := rotary.applyRotaryEmb q1 cos sin
  let k2 : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] := rotary.applyRotaryEmb k1 cos sin
  let v2 : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale cfg.head_dim)
  let kh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention k2
  let vh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention v2
  let attn : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    nn.scaledDotProductAttentionGQAMaskQKV qh kh vh attnMask 0.0 true
  let out : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim] :=
    reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  let paddedK : T #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] := padSlidingKVForCache cfg kh
  let paddedV : T #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] := padSlidingKVForCache cfg vh
  let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let kStore' := data.sliceScatter kStore 2 0 paddedK
  let vStore' := data.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache cfg batch := {
    kStoreDyn := nn.eraseShape kStore'
    vStoreDyn := nn.eraseShape vStore'
    seq := seq
    maxLen := cache.maxLen
  }
  (oProjSliding cfg m out, cache')

private def attentionFullPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, Config.fullHeadDim cfg / 2])
    (sin : T #[seq, Config.fullHeadDim cfg / 2])
    (attnMask : T #[batch, seq, seq])
    (cache : KVCache cfg batch)
    : T #[batch, seq, cfg.hidden_size] × KVCache cfg batch :=
  let q0 := qProjFull cfg m x
  let kRaw := kProjFull cfg m x
  let vRaw :=
    if cfg.attention_k_eq_v then
      kRaw
    else
      vProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
  let q2 : T #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] := rotary.applyRotaryEmb q1 cos sin
  let k2 : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rotary.applyRotaryEmb k1 cos sin
  let v2 : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale (Config.fullHeadDim cfg))
  let kh : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] := nn.transpose_for_attention k2
  let vh : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] := nn.transpose_for_attention v2
  let attn : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
    nn.scaledDotProductAttentionGQAMaskQKV qh kh vh attnMask 0.0 true
  let out : T #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] :=
    reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  let paddedK : T #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] := padFullKVForCache cfg kh
  let paddedV : T #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] := padFullKVForCache cfg vh
  let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
    reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let kStore' := data.sliceScatter kStore 2 0 paddedK
  let vStore' := data.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache cfg batch := {
    kStoreDyn := nn.eraseShape kStore'
    vStoreDyn := nn.eraseShape vStore'
    seq := seq
    maxLen := cache.maxLen
  }
  (oProjFull cfg m out, cache')

private def attentionSlidingNoCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    : T #[batch, seq, cfg.hidden_size] :=
  let q0 := qProjSliding cfg m x
  let k0 := kProjSliding cfg m x
  let v0 := vProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight k0 m.k_norm.eps
  let q2 : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] := rotary.applyRotaryEmb q1 cos sin
  let k2 : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] := rotary.applyRotaryEmb k1 cos sin
  let v2 : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] := rmsNormNoScale4dSlice v0 cfg.rms_norm_eps
  let qh : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale cfg.head_dim)
  let kh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention k2
  let vh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention v2
  let attn : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    nn.scaledDotProductAttentionGQAWindow qh kh vh 0.0 true true cfg.sliding_window
  let out : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim] :=
    reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  oProjSliding cfg m out

private def attentionFullNoCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, Config.fullHeadDim cfg / 2])
    (sin : T #[seq, Config.fullHeadDim cfg / 2])
    : T #[batch, seq, cfg.hidden_size] :=
  let q0 := qProjFull cfg m x
  let kRaw := kProjFull cfg m x
  let vRaw :=
    if cfg.attention_k_eq_v then
      kRaw
    else
      vProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
  let q2 : T #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] := rotary.applyRotaryEmb q1 cos sin
  let k2 : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rotary.applyRotaryEmb k1 cos sin
  let v2 : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale (Config.fullHeadDim cfg))
  let kh : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] := nn.transpose_for_attention k2
  let vh : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] := nn.transpose_for_attention v2
  let attn : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
    nn.scaledDotProductAttentionGQA qh kh vh 0.0 true true
  let out : T #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] :=
    reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  oProjFull cfg m out

private def attentionSlidingWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, cfg.head_dim / 2])
    (sin : T #[seq, cfg.head_dim / 2])
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : T #[batch, seq, cfg.hidden_size] × KVCache cfg batch :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : T #[batch, seq, cfg.num_attention_heads, cfg.head_dim] := rotary.applyRotaryEmb q1 cos sin
  let qh : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := seq) cfg src
    let attn : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
      nn.scaledDotProductAttentionGQAWindow qh kAll0 vAll0 0.0 true true cfg.sliding_window
    let out : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim] :=
      reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache)
  | none =>
    let kRaw := kProjSliding cfg m x
    let vRaw := vProjSliding cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] := rotary.applyRotaryEmb k1 cos sin
    let v2 : T #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention k2
    let vh : T #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] := nn.transpose_for_attention v2
    let paddedK := padSlidingKVForCache cfg kh
    let paddedV := padSlidingKVForCache cfg vh
    let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let kStore' := data.sliceScatter kStore 2 0 paddedK
    let vStore' := data.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache cfg batch := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := seq
      maxLen := cache.maxLen
    }
    let attn : T #[batch, cfg.num_attention_heads, seq, cfg.head_dim] :=
      nn.scaledDotProductAttentionGQAWindow qh kh vh 0.0 true true cfg.sliding_window
    let out : T #[batch, seq, cfg.num_attention_heads * cfg.head_dim] :=
      reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache')

private def attentionFullWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (cos : T #[seq, Config.fullHeadDim cfg / 2])
    (sin : T #[seq, Config.fullHeadDim cfg / 2])
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : T #[batch, seq, cfg.hidden_size] × KVCache cfg batch :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : T #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] := rotary.applyRotaryEmb q1 cos sin
  let qh : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let (kAll, vAll) := cacheSliceFull (kvLen := seq) cfg src
    let attn : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
      nn.scaledDotProductAttentionGQA qh kAll vAll 0.0 true true
    let out : T #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] :=
      reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache)
  | none =>
    let kRaw := kProjFull cfg m x
    let vRaw :=
      if cfg.attention_k_eq_v then
        kRaw
      else
        vProjFull cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rotary.applyRotaryEmb k1 cos sin
    let v2 : T #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kh : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] := nn.transpose_for_attention k2
    let vh : T #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] := nn.transpose_for_attention v2
    let paddedK := padFullKVForCache cfg kh
    let paddedV := padFullKVForCache cfg vh
    let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let kStore' := data.sliceScatter kStore 2 0 paddedK
    let vStore' := data.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache cfg batch := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := seq
      maxLen := cache.maxLen
    }
    let attn : T #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] :=
      nn.scaledDotProductAttentionGQA qh kh vh 0.0 true true
    let out : T #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] :=
      reshape (nn.transpose_from_attention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache')

private def attentionSlidingStep {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, 1, cfg.hidden_size])
    (cos : T #[1, cfg.head_dim / 2])
    (sin : T #[1, cfg.head_dim / 2])
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : T #[batch, 1, cfg.hidden_size] × KVCache cfg batch :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : T #[batch, 1, cfg.num_attention_heads, cfg.head_dim] := rotary.applyRotaryEmb q1 cos sin
  let qh : T #[batch, cfg.num_attention_heads, 1, cfg.head_dim] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let kvLen := src.seq
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := kvLen) cfg src
    let kAll : T #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] := data.slice kAll0 2 start useLen
    let vAll : T #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] := data.slice vAll0 2 start useLen
    let attn : T #[batch, cfg.num_attention_heads, 1, cfg.head_dim] :=
      nn.scaledDotProductAttentionGQAQKV qh kAll vAll 0.0 false true
    let out : T #[batch, 1, cfg.num_attention_heads * cfg.head_dim] :=
      reshape (nn.transpose_from_attention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache)
  | none =>
    let kRaw := kProjSliding cfg m x
    let vRaw := vProjSliding cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : T #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] := rotary.applyRotaryEmb k1 cos sin
    let v2 : T #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kNew : T #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] := nn.transpose_for_attention k2
    let vNew : T #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] := nn.transpose_for_attention v2
    let paddedK : T #[batch, Config.maxKVHeads cfg, 1, Config.maxHeadDim cfg] := padSlidingKVForCache cfg kNew
    let paddedV : T #[batch, Config.maxKVHeads cfg, 1, Config.maxHeadDim cfg] := padSlidingKVForCache cfg vNew
    let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let writePos := if cache.seq < cache.maxLen then cache.seq else cache.maxLen - 1
    let kStore' := data.sliceScatter kStore 2 writePos paddedK
    let vStore' := data.sliceScatter vStore 2 writePos paddedV
    let kvLen := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    let kAll0 : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStoreDyn := nn.eraseShape kStore', vStoreDyn := nn.eraseShape vStore', seq := kvLen }) |>.1
    let vAll0 : T #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStoreDyn := nn.eraseShape kStore', vStoreDyn := nn.eraseShape vStore', seq := kvLen }) |>.2
    let kAll : T #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] := data.slice kAll0 2 start useLen
    let vAll : T #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] := data.slice vAll0 2 start useLen
    let attn : T #[batch, cfg.num_attention_heads, 1, cfg.head_dim] :=
      nn.scaledDotProductAttentionGQAQKV qh kAll vAll 0.0 false true
    let out : T #[batch, 1, cfg.num_attention_heads * cfg.head_dim] :=
      reshape (nn.transpose_from_attention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    let cache' : KVCache cfg batch := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (oProjSliding cfg m out, cache')

private def attentionFullStep {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : T #[batch, 1, cfg.hidden_size])
    (cos : T #[1, Config.fullHeadDim cfg / 2])
    (sin : T #[1, Config.fullHeadDim cfg / 2])
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : T #[batch, 1, cfg.hidden_size] × KVCache cfg batch :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : T #[batch, 1, cfg.num_attention_heads, Config.fullHeadDim cfg] := rotary.applyRotaryEmb q1 cos sin
  let qh : T #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg] :=
    mul_scalar (nn.transpose_for_attention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let kvLen := src.seq
    let (kAll, vAll) := cacheSliceFull (kvLen := kvLen) cfg src
    let attn : T #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg] :=
      nn.scaledDotProductAttentionGQAQKV qh kAll vAll 0.0 false true
    let out : T #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg] :=
      reshape (nn.transpose_from_attention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache)
  | none =>
    let kRaw := kProjFull cfg m x
    let vRaw :=
      if cfg.attention_k_eq_v then
        kRaw
      else
        vProjFull cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : T #[batch, 1, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rotary.applyRotaryEmb k1 cos sin
    let v2 : T #[batch, 1, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kNew : T #[batch, Config.fullNumKVHeads cfg, 1, Config.fullHeadDim cfg] := nn.transpose_for_attention k2
    let vNew : T #[batch, Config.fullNumKVHeads cfg, 1, Config.fullHeadDim cfg] := nn.transpose_for_attention v2
    let paddedK := padFullKVForCache cfg kNew
    let paddedV := padFullKVForCache cfg vNew
    let kStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.kStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : T #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] :=
      reshape cache.vStoreDyn #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let writePos := if cache.seq < cache.maxLen then cache.seq else cache.maxLen - 1
    let kStore' := data.sliceScatter kStore 2 writePos paddedK
    let vStore' := data.sliceScatter vStore 2 writePos paddedV
    let kvLen := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
    let kAll : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStoreDyn := nn.eraseShape kStore', vStoreDyn := nn.eraseShape vStore', seq := kvLen }) |>.1
    let vAll : T #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStoreDyn := nn.eraseShape kStore', vStoreDyn := nn.eraseShape vStore', seq := kvLen }) |>.2
    let attn : T #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg] :=
      nn.scaledDotProductAttentionGQAQKV qh kAll vAll 0.0 false true
    let out : T #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg] :=
      reshape (nn.transpose_from_attention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    let cache' : KVCache cfg batch := {
      kStoreDyn := nn.eraseShape kStore'
      vStoreDyn := nn.eraseShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (oProjFull cfg m out, cache')

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : T #[batch, seq, cfg.hidden_size])
    (slidingCos : T #[seq, cfg.head_dim / 2])
    (slidingSin : T #[seq, cfg.head_dim / 2])
    (fullCos : T #[seq, Config.fullHeadDim cfg / 2])
    (fullSin : T #[seq, Config.fullHeadDim cfg / 2])
    : T #[batch, seq, cfg.hidden_size] :=
  match layerType with
  | .slidingAttention => attentionSlidingNoCache cfg m x slidingCos slidingSin
  | .fullAttention => attentionFullNoCache cfg m x fullCos fullSin

def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : T #[batch, seq, cfg.hidden_size])
    (slidingCos : T #[seq, cfg.head_dim / 2])
    (slidingSin : T #[seq, cfg.head_dim / 2])
    (fullCos : T #[seq, Config.fullHeadDim cfg / 2])
    (fullSin : T #[seq, Config.fullHeadDim cfg / 2])
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : T #[batch, seq, cfg.hidden_size] × KVCache cfg batch :=
  match layerType with
  | .slidingAttention => attentionSlidingWithCache cfg m x slidingCos slidingSin cache sharedSource
  | .fullAttention => attentionFullWithCache cfg m x fullCos fullSin cache sharedSource

def forwardStep {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : T #[batch, 1, cfg.hidden_size])
    (slidingCos : T #[1, cfg.head_dim / 2])
    (slidingSin : T #[1, cfg.head_dim / 2])
    (fullCos : T #[1, Config.fullHeadDim cfg / 2])
    (fullSin : T #[1, Config.fullHeadDim cfg / 2])
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : T #[batch, 1, cfg.hidden_size] × KVCache cfg batch :=
  match layerType with
  | .slidingAttention => attentionSlidingStep cfg m x slidingCos slidingSin cache sharedSource
  | .fullAttention => attentionFullStep cfg m x fullCos fullSin cache sharedSource

def forwardPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : T #[batch, seq, cfg.hidden_size])
    (slidingCos : T #[seq, cfg.head_dim / 2])
    (slidingSin : T #[seq, cfg.head_dim / 2])
    (fullCos : T #[seq, Config.fullHeadDim cfg / 2])
    (fullSin : T #[seq, Config.fullHeadDim cfg / 2])
    (slidingAttnMask : T #[batch, seq, seq])
    (fullAttnMask : T #[batch, seq, seq])
    (cache : KVCache cfg batch)
    : T #[batch, seq, cfg.hidden_size] × KVCache cfg batch :=
  match layerType with
  | .slidingAttention =>
    attentionSlidingPrefillWithMask cfg m x slidingCos slidingSin slidingAttnMask cache
  | .fullAttention =>
    attentionFullPrefillWithMask cfg m x fullCos fullSin fullAttnMask cache

end Gemma4Attention

structure Gemma4Layer (cfg : Config) where
  layerIdx : UInt64
  layerType : LayerType
  self_attn : Gemma4Attention cfg
  mlp : Gemma4MLP cfg
  input_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_attention_layernorm : Gemma4RMSNorm cfg.hidden_size
  pre_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  per_layer_input : Option (Gemma4PerLayerInputBlock cfg) := none
  moe : Option (Gemma4MoeBranch cfg) := none
  layer_scalar : T #[1]
  deriving TensorStruct

structure Gemma4Cache (cfg : Config) (batch : UInt64) where
  attnCaches : Array (Gemma4Attention.KVCache cfg batch)

namespace Gemma4Layer

private def applyLayerScalar {batch seq : UInt64}
    (cfg : Config)
    (scale : T #[1])
    (x : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.hidden_size] :=
  let s : T #[batch, seq, cfg.hidden_size] := nn.expand (reshape scale #[1, 1, 1]) #[batch, seq, cfg.hidden_size]
  x * s

private def applyPerLayerInput {batch seq : UInt64}
    (cfg : Config)
    (block : Gemma4PerLayerInputBlock cfg)
    (hidden : T #[batch, seq, cfg.hidden_size])
    (perLayerInput : T #[batch, seq, cfg.hidden_size_per_layer_input])
    : T #[batch, seq, cfg.hidden_size] :=
  let residual := hidden
  let gated : T #[batch, seq, cfg.hidden_size_per_layer_input] :=
    nn.gelu (linear3d hidden block.per_layer_input_gate)
  let mixed := gated * perLayerInput
  let projected : T #[batch, seq, cfg.hidden_size] := linear3d mixed block.per_layer_projection
  let projected := block.post_per_layer_input_norm.forward3d projected
  residual + projected

def forward {batch seq : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (slidingCos : T #[seq, cfg.head_dim / 2])
    (slidingSin : T #[seq, cfg.head_dim / 2])
    (fullCos : T #[seq, Config.fullHeadDim cfg / 2])
    (fullSin : T #[seq, Config.fullHeadDim cfg / 2])
    (perLayerInput : Option (T #[batch, seq, cfg.hidden_size_per_layer_input]) := none)
    : T #[batch, seq, cfg.hidden_size] :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let attn := layer.self_attn.forward cfg layer.layerType h1 slidingCos slidingSin fullCos fullSin
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : T #[tokens, cfg.hidden_size] := reshape residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : T #[tokens, cfg.hidden_size] :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : T #[batch, seq, cfg.hidden_size] := reshape expert2d #[batch, seq, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  applyLayerScalar cfg layer.layer_scalar h5

def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (slidingCos : T #[seq, cfg.head_dim / 2])
    (slidingSin : T #[seq, cfg.head_dim / 2])
    (fullCos : T #[seq, Config.fullHeadDim cfg / 2])
    (fullSin : T #[seq, Config.fullHeadDim cfg / 2])
    (layerCache : Gemma4Attention.KVCache cfg batch)
    (sharedSource : Option (Gemma4Attention.KVCache cfg batch) := none)
    (perLayerInput : Option (T #[batch, seq, cfg.hidden_size_per_layer_input]) := none)
    : T #[batch, seq, cfg.hidden_size] × Gemma4Attention.KVCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let (attn, layerCache') :=
    layer.self_attn.forwardWithCache
      cfg
      layer.layerType
      h1
      slidingCos
      slidingSin
      fullCos
      fullSin
      layerCache
      sharedSource
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : T #[tokens, cfg.hidden_size] := reshape residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : T #[tokens, cfg.hidden_size] :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : T #[batch, seq, cfg.hidden_size] := reshape expert2d #[batch, seq, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardStep {batch : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : T #[batch, 1, cfg.hidden_size])
    (slidingCos : T #[1, cfg.head_dim / 2])
    (slidingSin : T #[1, cfg.head_dim / 2])
    (fullCos : T #[1, Config.fullHeadDim cfg / 2])
    (fullSin : T #[1, Config.fullHeadDim cfg / 2])
    (layerCache : Gemma4Attention.KVCache cfg batch)
    (sharedSource : Option (Gemma4Attention.KVCache cfg batch) := none)
    (perLayerInput : Option (T #[batch, 1, cfg.hidden_size_per_layer_input]) := none)
    : T #[batch, 1, cfg.hidden_size] × Gemma4Attention.KVCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let (attn, layerCache') :=
    layer.self_attn.forwardStep
      cfg
      layer.layerType
      h1
      slidingCos
      slidingSin
      fullCos
      fullSin
      layerCache
      sharedSource
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let routerInput : T #[batch, cfg.hidden_size] := reshape residual2 #[batch, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : T #[batch, cfg.hidden_size] :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : T #[batch, 1, cfg.hidden_size] := reshape expert2d #[batch, 1, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : T #[batch, seq, cfg.hidden_size])
    (slidingCos : T #[seq, cfg.head_dim / 2])
    (slidingSin : T #[seq, cfg.head_dim / 2])
    (fullCos : T #[seq, Config.fullHeadDim cfg / 2])
    (fullSin : T #[seq, Config.fullHeadDim cfg / 2])
    (slidingAttnMask : T #[batch, seq, seq])
    (fullAttnMask : T #[batch, seq, seq])
    (layerCache : Gemma4Attention.KVCache cfg batch)
    (perLayerInput : Option (T #[batch, seq, cfg.hidden_size_per_layer_input]) := none)
    : T #[batch, seq, cfg.hidden_size] × Gemma4Attention.KVCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let (attn, layerCache') :=
    layer.self_attn.forwardPrefillWithMask
      cfg
      layer.layerType
      h1
      slidingCos
      slidingSin
      fullCos
      fullSin
      slidingAttnMask
      fullAttnMask
      layerCache
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : T #[tokens, cfg.hidden_size] := reshape residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : T #[tokens, cfg.hidden_size] :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : T #[batch, seq, cfg.hidden_size] := reshape expert2d #[batch, seq, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

end Gemma4Layer

structure Gemma4Model (cfg : Config) where
  embed_tokens : T #[cfg.vocab_size, cfg.hidden_size]
  embed_tokens_per_layer :
    Option (T #[cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input]) := none
  per_layer_model_projection :
    Option (T #[cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.hidden_size]) := none
  per_layer_projection_norm : Option (Gemma4RMSNorm cfg.hidden_size_per_layer_input) := none
  layers : Array (Gemma4Layer cfg)
  norm : Gemma4RMSNorm cfg.hidden_size
  deriving TensorStruct

namespace Gemma4Model

private def extractPerLayerInput? {batch seq : UInt64}
    (cfg : Config)
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (layerIdx : UInt64)
    : Option (T #[batch, seq, cfg.hidden_size_per_layer_input]) :=
  match perLayerInputs with
  | some t =>
    let one : T #[batch, seq, 1, cfg.hidden_size_per_layer_input] := data.slice t 2 layerIdx 1
    some (reshape one #[batch, seq, cfg.hidden_size_per_layer_input])
  | none => none

def embedTokens {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  mul_scalar (nn.embedding inputIds m.embed_tokens) (Float.sqrt cfg.hidden_size.toFloat)

def computePerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]) :=
  match m.embed_tokens_per_layer, m.per_layer_model_projection, m.per_layer_projection_norm with
  | some tokEmbTable, some projW, some projNorm =>
    let tokEmbFlat : T #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input] :=
      mul_scalar
        (nn.embedding inputIds tokEmbTable)
        (Float.sqrt cfg.hidden_size_per_layer_input.toFloat)
    let tokEmb : T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] :=
      reshape tokEmbFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let projScale := 1.0 / Float.sqrt cfg.hidden_size.toFloat
    let projFlat : T #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input] :=
      mul_scalar (linear3d inputsEmbeds projW) projScale
    let proj4d : T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] :=
      reshape projFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let proj4d := projNorm.forward4d proj4d
    some (mul_scalar (proj4d + tokEmb) (Float.sqrt 0.5))
  | _, _, _ =>
    none

private def precomputeSlidingRotary {seq : UInt64}
    (cfg : Config)
    (device : Device)
    : T #[seq, cfg.head_dim / 2] × T #[seq, cfg.head_dim / 2] :=
  rotary.computeFreqsOnDevicePure seq cfg.head_dim cfg.sliding_rope_theta device

private def precomputeFullRotary {seq : UInt64}
    (cfg : Config)
    (device : Device)
    : T #[seq, Config.fullHeadDim cfg / 2] × T #[seq, Config.fullHeadDim cfg / 2] :=
  let fullHalf := Config.fullHeadDim cfg / 2
  let rotHalf := Config.fullRotaryHalfDim cfg
  let (cos0, sin0) := rotary.computeFreqsOnDevicePure seq (Config.fullHeadDim cfg) cfg.full_rope_theta device
  let cosRot : T #[seq, rotHalf] := data.slice cos0 1 0 rotHalf
  let sinRot : T #[seq, rotHalf] := data.slice sin0 1 0 rotHalf
  let passHalf : UInt64 := fullHalf - rotHalf
  let cosPass : T #[seq, passHalf] := onesOn device
  let sinPass : T #[seq, passHalf] := zerosOn device
  (nn.cat cosRot cosPass 1, nn.cat sinRot sinPass 1)

def initCache {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (maxLen : UInt64)
    (device : Device)
    : Gemma4Cache cfg batch :=
  Id.run do
    let mut caches : Array (Gemma4Attention.KVCache cfg batch) := #[]
    for _ in [:m.layers.size] do
      let kv := qwen.QwenAttention.initKVCache
        maxLen
        (batch := batch)
        (num_kv_heads := Config.maxKVHeads cfg)
        (head_dim := Config.maxHeadDim cfg)
        device
      caches := caches.push kv
    return { attnCaches := caches }

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let (slidingCos, slidingSin) := precomputeSlidingRotary cfg inputsEmbeds.device
    let (fullCos, fullSin) := precomputeFullRotary cfg inputsEmbeds.device
    let mut hidden := inputsEmbeds
    for i in [:m.layers.size] do
      match m.layers[i]? with
      | some layer =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        hidden := Gemma4Layer.forward cfg layer hidden slidingCos slidingSin fullCos fullSin perLayer
      | none => pure ()
    return m.norm.forward3d hidden

def forwardEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.hidden_size] :=
  Id.run do
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let (slidingCos, slidingSin) := precomputeSlidingRotary cfg inputsEmbeds.device
    let (fullCos, fullSin) := precomputeFullRotary cfg inputsEmbeds.device
    let mut hidden := inputsEmbeds
    for i in [:m.layers.size] do
      match m.layers[i]? with
      | some layer =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        hidden := Gemma4Layer.forward cfg layer hidden slidingCos slidingSin fullCos fullSin perLayer
      | none => pure ()
    return m.norm.forward3d hidden

def forwardEmbedsWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    : T #[batch, seq, cfg.hidden_size] :=
  Id.run do
    let _ := inputIds
    let (slidingCos, slidingSin) := precomputeSlidingRotary cfg inputsEmbeds.device
    let (fullCos, fullSin) := precomputeFullRotary cfg inputsEmbeds.device
    let mut hidden := inputsEmbeds
    for i in [:m.layers.size] do
      match m.layers[i]? with
      | some layer =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        hidden := Gemma4Layer.forward cfg layer hidden slidingCos slidingSin fullCos fullSin perLayer
      | none => pure ()
    return m.norm.forward3d hidden

def forwardWithCache {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : T #[batch, seq, cfg.hidden_size] × Gemma4Cache cfg batch :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : T #[seq, cfg.head_dim / 2] := data.slice slidingCosAll 0 0 seq
    let slidingSin : T #[seq, cfg.head_dim / 2] := data.slice slidingSinAll 0 0 seq
    let fullCos : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullCosAll 0 0 seq
    let fullSin : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardWithCache
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbeds {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : T #[batch, seq, cfg.hidden_size] × Gemma4Cache cfg batch :=
  Id.run do
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : T #[seq, cfg.head_dim / 2] := data.slice slidingCosAll 0 0 seq
    let slidingSin : T #[seq, cfg.head_dim / 2] := data.slice slidingSinAll 0 0 seq
    let fullCos : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullCosAll 0 0 seq
    let fullSin : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardWithCache
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbedsWithPerLayerInputs {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : T #[batch, seq, cfg.hidden_size] × Gemma4Cache cfg batch :=
  Id.run do
    let _ := inputIds
    let slidingCos : T #[seq, cfg.head_dim / 2] := data.slice slidingCosAll 0 0 seq
    let slidingSin : T #[seq, cfg.head_dim / 2] := data.slice slidingSinAll 0 0 seq
    let fullCos : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullCosAll 0 0 seq
    let fullSin : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardWithCache
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (slidingAttnMask : T #[batch, seq, seq])
    (fullAttnMask : T #[batch, seq, seq])
    : T #[batch, seq, cfg.hidden_size] × Gemma4Cache cfg batch :=
  Id.run do
    let _ := inputIds
    let slidingCos : T #[seq, cfg.head_dim / 2] := data.slice slidingCosAll 0 0 seq
    let slidingSin : T #[seq, cfg.head_dim / 2] := data.slice slidingSinAll 0 0 seq
    let fullCos : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullCosAll 0 0 seq
    let fullSin : T #[seq, Config.fullHeadDim cfg / 2] := data.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardPrefillWithMask
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            slidingAttnMask
            fullAttnMask
            layerCache
            perLayer
        hidden := hiddenNext
        cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardStep {batch : UInt64} {maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : T #[batch, 1])
    (position : UInt64)
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : T #[batch, 1, cfg.hidden_size] × Gemma4Cache cfg batch :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : T #[1, cfg.head_dim / 2] := data.slice slidingCosAll 0 position 1
    let slidingSin : T #[1, cfg.head_dim / 2] := data.slice slidingSinAll 0 position 1
    let fullCos : T #[1, Config.fullHeadDim cfg / 2] := data.slice fullCosAll 0 position 1
    let fullSin : T #[1, Config.fullHeadDim cfg / 2] := data.slice fullSinAll 0 position 1
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer :=
          match perLayerInputs with
          | some p =>
            let one : T #[batch, 1, 1, cfg.hidden_size_per_layer_input] := data.slice p 2 i.toUInt64 1
            some (reshape one #[batch, 1, cfg.hidden_size_per_layer_input])
          | none => none
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardStep
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

end Gemma4Model

structure Gemma4ForCausalLM (cfg : Config) where
  model : Gemma4Model cfg
  lmHead : T #[cfg.vocab_size, cfg.hidden_size]
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Gemma4ForCausalLM

def embedTokens {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.hidden_size] :=
  m.model.embedTokens cfg inputIds

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    : T #[batch, seq, cfg.vocab_size] :=
  let hidden := m.model.forward cfg inputIds
  nn.softcap (linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    : T #[batch, seq, cfg.vocab_size] :=
  let hidden := m.model.forwardEmbeds cfg inputIds inputsEmbeds
  nn.softcap (linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbedsWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    : T #[batch, seq, cfg.vocab_size] :=
  let hidden := m.model.forwardEmbedsWithPerLayerInputs cfg inputIds inputsEmbeds perLayerInputs
  nn.softcap (linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbedsWithPerLayerInputsBidirectionalVision {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (slidingAttnMask : T #[batch, seq, seq])
    (fullAttnMask : T #[batch, seq, seq])
    : T #[batch, seq, cfg.vocab_size] :=
  let cache := Gemma4Model.initCache cfg m.model seq inputsEmbeds.device
  let (slidingCosAll, slidingSinAll) := Gemma4Model.precomputeSlidingRotary cfg inputsEmbeds.device
  let (fullCosAll, fullSinAll) := Gemma4Model.precomputeFullRotary cfg inputsEmbeds.device
  let (hidden, _cache') :=
    m.model.forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision
      (maxLen := seq)
      cfg
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      slidingAttnMask
      fullAttnMask
  nn.softcap (linear3d hidden m.lmHead) cfg.final_logit_softcapping

private def precomputeDecodeRotary {maxLen : UInt64}
    (cfg : Config)
    (device : Device)
    : T #[maxLen, cfg.head_dim / 2]
      × T #[maxLen, cfg.head_dim / 2]
      × T #[maxLen, Config.fullHeadDim cfg / 2]
      × T #[maxLen, Config.fullHeadDim cfg / 2] :=
  let (slCos, slSin) := Gemma4Model.precomputeSlidingRotary cfg device
  let (fullCos, fullSin) := Gemma4Model.precomputeFullRotary cfg device
  (slCos, slSin, fullCos, fullSin)

private def prefillCaches {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : IO (T #[batch, cfg.vocab_size] × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCache
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : T #[batch, 1, cfg.hidden_size] := data.slice hidden 1 (seq - 1) 1
  let logits3 : T #[batch, 1, cfg.vocab_size] := nn.softcap (linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (reshape logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : IO (T #[batch, cfg.vocab_size] × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbeds
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : T #[batch, 1, cfg.hidden_size] := data.slice hidden 1 (seq - 1) 1
  let logits3 : T #[batch, 1, cfg.vocab_size] := nn.softcap (linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (reshape logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputs {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : IO (T #[batch, cfg.vocab_size] × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbedsWithPerLayerInputs
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : T #[batch, 1, cfg.hidden_size] := data.slice hidden 1 (seq - 1) 1
  let logits3 : T #[batch, 1, cfg.vocab_size] := nn.softcap (linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (reshape logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (slidingAttnMask : T #[batch, seq, seq])
    (fullAttnMask : T #[batch, seq, seq])
    : IO (T #[batch, cfg.vocab_size] × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      slidingAttnMask
      fullAttnMask
  let lastHidden : T #[batch, 1, cfg.hidden_size] := data.slice hidden 1 (seq - 1) 1
  let logits3 : T #[batch, 1, cfg.vocab_size] := nn.softcap (linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (reshape logits3 #[batch, cfg.vocab_size], cache')

private def decodeStep {batch maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, 1])
    (position : UInt64)
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    : IO (T #[batch, cfg.vocab_size] × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardStep
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      position
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let logits3 : T #[batch, 1, cfg.vocab_size] := nn.softcap (linear3d hidden m.lmHead) cfg.final_logit_softcapping
  pure (reshape logits3 #[batch, cfg.vocab_size], cache')

private partial def decodeLoopCached {batch maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : T #[batch])
    (remaining : Nat)
    (cache : Gemma4Cache cfg batch)
    (lastLogits : T #[batch, cfg.vocab_size])
    (slidingCosAll : T #[maxLen, cfg.head_dim / 2])
    (slidingSinAll : T #[maxLen, cfg.head_dim / 2])
    (fullCosAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (fullSinAll : T #[maxLen, Config.fullHeadDim cfg / 2])
    (onStep : Option (StreamCallback batch))
    (generatedSoFar : UInt64)
    {curSeq : UInt64}
    (curIds : T #[batch, curSeq])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw ← sampleFromLogits lastLogits strategy
  let finished' : T #[batch] :=
    if eosTokenIds.isEmpty then
      finished
    else
      logicalOr finished (tokenInSet nextTokRaw eosTokenIds)
  let nextTok : T #[batch] :=
    if eosTokenIds.isEmpty then
      nextTokRaw
    else
      applyFinishedEos nextTokRaw finished (eosTokenIds.getD 0 0)

  match onStep with
  | some cb => cb generatedSoFar nextTok
  | none => pure ()

  let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
  let appended : T #[batch, curSeq + 1] := nn.cat curIds nextCol 1

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(any (logical_not finished'))
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    let (nextLogits, cache') ←
      decodeStep
        cfg
        m
        nextCol
        curSeq
        cache
        slidingCosAll
        slidingSinAll
        fullCosAll
        fullSinAll
    decodeLoopCached
      cfg
      m
      strategy
      eosTokenIds
      finished'
      (remaining - 1)
      cache'
      nextLogits
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      onStep
      (generatedSoFar + 1)
      appended

private def generateCore {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputIds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputIds.device
  let (logits, cache') ←
    prefillCaches
      (maxLen := maxLen)
      cfg m inputIds cache
      slidingCosAll slidingSinAll fullCosAll fullSinAll
  let finished0 : T #[batch] := falseMask (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens.toNat
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
    onStep
    0
    inputIds

private def generateFromEmbedsCore {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputsEmbeds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputsEmbeds.device
  let (logits, cache') ←
    prefillCachesFromEmbeds
      (maxLen := maxLen)
      cfg
      m
      inputIds
      inputsEmbeds
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let finished0 : T #[batch] := falseMask (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens.toNat
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
      onStep
      0
      inputIds

private def generateFromEmbedsCoreWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    (prefillMasks : Option (T #[batch, seq, seq] × T #[batch, seq, seq]) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputsEmbeds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputsEmbeds.device
  let (logits, cache') ←
    match prefillMasks with
    | some (slidingAttnMask, fullAttnMask) =>
      prefillCachesFromEmbedsWithPerLayerInputsBidirectionalVision
        (maxLen := maxLen)
        cfg
        m
        inputIds
        inputsEmbeds
        perLayerInputs
        cache
        slidingCosAll
        slidingSinAll
        fullCosAll
        fullSinAll
        slidingAttnMask
        fullAttnMask
    | none =>
      prefillCachesFromEmbedsWithPerLayerInputs
        (maxLen := maxLen)
        cfg
        m
        inputIds
        inputsEmbeds
        perLayerInputs
        cache
        slidingCosAll
        slidingSinAll
        fullCosAll
        fullSinAll
  let finished0 : T #[batch] := falseMask (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens.toNat
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
    onStep
    0
    inputIds

def generate {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds none

def generateFromEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

def generateFromEmbedsWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (prefillMasks : Option (T #[batch, seq, seq] × T #[batch, seq, seq]) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds none prefillMasks

def generateStream {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds (some onStep)

def generateFromEmbedsStream {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

def generateFromEmbedsStreamWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, cfg.hidden_size])
    (perLayerInputs : Option (T #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]))
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (prefillMasks : Option (T #[batch, seq, seq] × T #[batch, seq, seq]) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds (some onStep) prefillMasks

end Gemma4ForCausalLM

end torch.gemma4

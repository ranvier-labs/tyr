/-
  Tyr/Model/Qwen35/Config.lean

  Configuration for the standalone Qwen3.5 text causal-LM implementation.
  Includes dense and MoE variants plus hybrid layer scheduling.
-/
import Tyr.Basic
import Tyr.TensorStruct

/-! # Qwen3.5 configuration

`torch.qwen35.Qwen35Config` with per-layer token-mixer kinds
`torch.qwen35.LayerType` and presets including
`torch.qwen35.Qwen35Config.qwen35_9B` and
`torch.qwen35.Qwen35Config.qwen35_35B_A3B`.
-/

namespace torch.qwen35

/-- Per-layer token-mixer kind used by Qwen3.5 hybrid blocks. -/
inductive LayerType where
  | linearAttention
  | fullAttention
  deriving Repr, Inhabited, BEq

namespace LayerType

/-- Parse a HuggingFace `layer_types` string entry. -/
def ofString? (s : String) : Option LayerType :=
  if s == "linear_attention" then
    some .linearAttention
  else if s == "full_attention" then
    some .fullAttention
  else
    none

/-- Serialize to HuggingFace-style string. -/
def toString : LayerType → String
  | .linearAttention => "linear_attention"
  | .fullAttention => "full_attention"

end LayerType

instance : TensorStruct LayerType where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

/-- Qwen3.5 text model configuration.
    Dense checkpoints use `num_experts = 0`.
    MoE checkpoints set `num_experts > 0`. -/
structure Qwen35Config where
  vocab_size : UInt64 := 248320
  hidden_size : UInt64 := 4096
  intermediate_size : UInt64 := 12288
  num_hidden_layers : UInt64 := 32
  num_attention_heads : UInt64 := 16
  num_key_value_heads : UInt64 := 4
  head_dim : UInt64 := 256

  rope_theta : Float := 1000000.0
  partial_rotary_factor : Float := 0.25
  rms_norm_eps : Float := 1e-6
  max_position_embeddings : UInt64 := 32768

  attention_bias : Bool := false
  attention_dropout : Float := 0.0
  attn_output_gate : Bool := true
  hidden_act : String := "silu"

  linear_conv_kernel_dim : UInt64 := 4
  linear_key_head_dim : UInt64 := 128
  linear_value_head_dim : UInt64 := 128
  linear_num_key_heads : UInt64 := 16
  linear_num_value_heads : UInt64 := 32
  mamba_ssm_dtype : String := "float32"

  layer_types : Array LayerType := #[]
  full_attention_interval : UInt64 := 4
  mlp_only_layers : Array UInt64 := #[]
  mrope_interleaved : Bool := false
  mrope_section : Array UInt64 := #[]
  mtp_num_hidden_layers : UInt64 := 0
  mtp_use_dedicated_embeddings : Bool := false

  moe_intermediate_size : UInt64 := 512
  shared_expert_intermediate_size : UInt64 := 512
  num_experts_per_tok : UInt64 := 8
  num_experts : UInt64 := 0

  use_cache : Bool := true
  tie_word_embeddings : Bool := false

  pad_token_id : Option UInt64 := none
  bos_token_id : Option UInt64 := none
  eos_token_id : Option UInt64 := none
  deriving Repr, Inhabited

namespace Qwen35Config

/-- Default dense Qwen3.5-9B style config. -/
def qwen35_9B : Qwen35Config :=
  { num_experts := 0 }

/-- Default MoE Qwen3.5-35B-A3B style config. -/
def qwen35_35B_A3B : Qwen35Config := {
  vocab_size := 248320
  hidden_size := 2048
  intermediate_size := 0
  num_hidden_layers := 40
  num_attention_heads := 16
  num_key_value_heads := 2
  head_dim := 256
  rope_theta := 1000000.0
  partial_rotary_factor := 0.25
  rms_norm_eps := 1e-6
  max_position_embeddings := 32768
  attention_bias := false
  attention_dropout := 0.0
  attn_output_gate := true
  hidden_act := "silu"
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 128
  linear_value_head_dim := 128
  linear_num_key_heads := 16
  linear_num_value_heads := 32
  mamba_ssm_dtype := "float32"
  layer_types := #[]
  full_attention_interval := 4
  mlp_only_layers := #[]
  mrope_interleaved := false
  mrope_section := #[]
  mtp_num_hidden_layers := 0
  mtp_use_dedicated_embeddings := false
  moe_intermediate_size := 512
  shared_expert_intermediate_size := 512
  num_experts_per_tok := 8
  num_experts := 256
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := none
  bos_token_id := none
  eos_token_id := none
}

/-- Build default hybrid schedule: every `interval`th layer is full attention, rest linear attention. -/
def defaultLayerTypes (numLayers interval : UInt64) : Array LayerType :=
  Id.run do
    let mut out : Array LayerType := Array.mkEmpty numLayers.toNat
    if interval == 0 then
      for _ in [:numLayers.toNat] do
        out := out.push .fullAttention
      return out
    for i in [:numLayers.toNat] do
      let isFull := ((i + 1).toUInt64 % interval) == 0
      out := out.push (if isFull then .fullAttention else .linearAttention)
    return out

/-- Default multimodal Qwen3.6-35B-A3B text config from Hugging Face `config.json`. -/
def qwen36_35B_A3B : Qwen35Config := {
  vocab_size := 248320
  hidden_size := 2048
  intermediate_size := 0
  num_hidden_layers := 40
  num_attention_heads := 16
  num_key_value_heads := 2
  head_dim := 256
  rope_theta := 10000000.0
  partial_rotary_factor := 0.25
  rms_norm_eps := 1e-6
  max_position_embeddings := 262144
  attention_bias := false
  attention_dropout := 0.0
  attn_output_gate := true
  hidden_act := "silu"
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 128
  linear_value_head_dim := 128
  linear_num_key_heads := 16
  linear_num_value_heads := 32
  mamba_ssm_dtype := "float32"
  layer_types := defaultLayerTypes 40 4
  full_attention_interval := 4
  mlp_only_layers := #[]
  mrope_interleaved := true
  mrope_section := #[11, 11, 10]
  mtp_num_hidden_layers := 1
  mtp_use_dedicated_embeddings := false
  moe_intermediate_size := 512
  shared_expert_intermediate_size := 512
  num_experts_per_tok := 8
  num_experts := 256
  use_cache := true
  tie_word_embeddings := false
  pad_token_id := none
  bos_token_id := some 248044
  eos_token_id := some 248044
}

/-- Return `layer_types` if valid, otherwise synthesize the default interval schedule. -/
def normalizedLayerTypes (cfg : Qwen35Config) : Array LayerType :=
  if cfg.layer_types.size == cfg.num_hidden_layers.toNat then
    cfg.layer_types
  else
    defaultLayerTypes cfg.num_hidden_layers cfg.full_attention_interval

/-- Default dense Qwen3.5-0.8B style config. -/
def qwen35_0_8B : Qwen35Config := {
  vocab_size := 248320
  hidden_size := 1024
  intermediate_size := 3584
  num_hidden_layers := 24
  num_attention_heads := 8
  num_key_value_heads := 2
  head_dim := 256
  rope_theta := 10000000.0
  partial_rotary_factor := 0.25
  rms_norm_eps := 1e-6
  max_position_embeddings := 262144
  attention_bias := false
  attention_dropout := 0.0
  attn_output_gate := true
  hidden_act := "silu"
  linear_conv_kernel_dim := 4
  linear_key_head_dim := 128
  linear_value_head_dim := 128
  linear_num_key_heads := 16
  linear_num_value_heads := 16
  mamba_ssm_dtype := "float32"
  layer_types := defaultLayerTypes 24 4
  full_attention_interval := 4
  mlp_only_layers := #[]
  mrope_interleaved := true
  mrope_section := #[11, 11, 10]
  mtp_num_hidden_layers := 1
  mtp_use_dedicated_embeddings := false
  moe_intermediate_size := 512
  shared_expert_intermediate_size := 512
  num_experts_per_tok := 8
  num_experts := 0
  use_cache := true
  tie_word_embeddings := true
  pad_token_id := none
  bos_token_id := none
  eos_token_id := some 248044
}

/-- Config with `layer_types` normalized for runtime use. -/
def normalize (cfg : Qwen35Config) : Qwen35Config :=
  { cfg with layer_types := normalizedLayerTypes cfg }

/-- Query heads per KV head (full-attention GQA). -/
def numHeadsPerKVGroup (cfg : Qwen35Config) : UInt64 :=
  if cfg.num_key_value_heads == 0 then 1 else cfg.num_attention_heads / cfg.num_key_value_heads

/-- Linear-attention key projection dimension. -/
def linearKeyDim (cfg : Qwen35Config) : UInt64 :=
  cfg.linear_num_key_heads * cfg.linear_key_head_dim

/-- Linear-attention value projection dimension. -/
def linearValueDim (cfg : Qwen35Config) : UInt64 :=
  cfg.linear_num_value_heads * cfg.linear_value_head_dim

/-- Depthwise-conv channel count for linear-attention fused QKV path. -/
def linearConvDim (cfg : Qwen35Config) : UInt64 :=
  (linearKeyDim cfg) * 2 + (linearValueDim cfg)

/-- Repeat factor to broadcast linear K/Q heads to V heads. -/
def linearKVRepeat (cfg : Qwen35Config) : UInt64 :=
  if cfg.linear_num_key_heads == 0 then 1 else cfg.linear_num_value_heads / cfg.linear_num_key_heads

/-- Whether this config uses MoE FFN blocks. -/
def isMoE (cfg : Qwen35Config) : Bool :=
  cfg.num_experts > 0 && cfg.num_experts_per_tok > 0

/-- Rotary dimension for partial RoPE. Clamped to an even value in `[2, head_dim]`. -/
def rotaryDim (cfg : Qwen35Config) : UInt64 :=
  let raw := (cfg.head_dim.toFloat * cfg.partial_rotary_factor).toUInt64
  let base :=
    if raw == 0 then cfg.head_dim
    else if raw > cfg.head_dim then cfg.head_dim
    else raw
  if base <= 2 then
    2
  else if base % 2 == 0 then
    base
  else
    base - 1

/-- Number of rotary frequencies (`rotary_dim / 2`). -/
def rotaryHalfDim (cfg : Qwen35Config) : UInt64 :=
  (rotaryDim cfg) / 2

end Qwen35Config

abbrev Config := Qwen35Config

namespace Config

def qwen35_0_8B : Config := Qwen35Config.qwen35_0_8B

def qwen35_9B : Config := Qwen35Config.qwen35_9B

def qwen35_35B_A3B : Config := Qwen35Config.qwen35_35B_A3B

def qwen36_35B_A3B : Config := Qwen35Config.qwen36_35B_A3B

def normalize (cfg : Config) : Config := Qwen35Config.normalize cfg

def normalizedLayerTypes (cfg : Config) : Array LayerType :=
  Qwen35Config.normalizedLayerTypes cfg

def isMoE (cfg : Config) : Bool := Qwen35Config.isMoE cfg

def rotaryDim (cfg : Config) : UInt64 := Qwen35Config.rotaryDim cfg

def rotaryHalfDim (cfg : Config) : UInt64 := Qwen35Config.rotaryHalfDim cfg

def linearKeyDim (cfg : Config) : UInt64 := Qwen35Config.linearKeyDim cfg

def linearValueDim (cfg : Config) : UInt64 := Qwen35Config.linearValueDim cfg

def linearConvDim (cfg : Config) : UInt64 := Qwen35Config.linearConvDim cfg

def linearKVRepeat (cfg : Config) : UInt64 := Qwen35Config.linearKVRepeat cfg

def numHeadsPerKVGroup (cfg : Config) : UInt64 := Qwen35Config.numHeadsPerKVGroup cfg

end Config

end torch.qwen35

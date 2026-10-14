/-
  Tyr/Model/Gemma4/Config.lean

  Configuration for the standalone Gemma 4 text causal-LM implementation.
  Covers the public Gemma 4 family checkpoints routed through the text
  submodel of the official multimodal repos.
-/
import Tyr.Basic
import Tyr.TensorStruct

/-! # Gemma 4 configuration

`torch.gemma4.Gemma4Config` describes the Gemma 4 text model, including the
per-layer attention schedule `torch.gemma4.LayerType`. Presets:
`torch.gemma4.Gemma4Config.gemma4_E4B`, `torch.gemma4.Gemma4Config.gemma4_E2B`,
`torch.gemma4.Gemma4Config.gemma4_26B_A4B` and
`torch.gemma4.Gemma4Config.gemma4_31B`.
-/

namespace torch.gemma4

/-- Per-layer attention schedule entry used by Gemma 4. -/
inductive LayerType where
  | slidingAttention
  | fullAttention
  deriving Repr, Inhabited, BEq

namespace LayerType

def ofString? (s : String) : Option LayerType :=
  if s == "sliding_attention" then
    some .slidingAttention
  else if s == "full_attention" then
    some .fullAttention
  else
    none

def toString : LayerType → String
  | .slidingAttention => "sliding_attention"
  | .fullAttention => "full_attention"

end LayerType

instance : TensorStruct LayerType where
  map _ x := x
  mapM _ x := pure x
  zipWith _ x _ := x
  fold _ init _ := init

/-- Gemma 4 text model configuration. -/
structure Gemma4Config where
  vocab_size : UInt64 := 262144
  hidden_size : UInt64 := 2560
  intermediate_size : UInt64 := 10240
  num_hidden_layers : UInt64 := 42
  num_attention_heads : UInt64 := 8
  num_key_value_heads : UInt64 := 2
  num_global_key_value_heads : UInt64 := 0
  head_dim : UInt64 := 256
  global_head_dim : UInt64 := 512

  sliding_window : UInt64 := 512
  sliding_rope_theta : Float := 10000.0
  full_rope_theta : Float := 1000000.0
  full_partial_rotary_factor : Float := 0.25
  rms_norm_eps : Float := 1e-6
  max_position_embeddings : UInt64 := 32768

  attention_bias : Bool := false
  attention_dropout : Float := 0.0
  hidden_activation : String := "gelu_pytorch_tanh"
  attention_k_eq_v : Bool := false
  use_bidirectional_attention : String := ""

  layer_types : Array LayerType := #[]
  full_attention_interval : UInt64 := 6
  num_kv_shared_layers : UInt64 := 0
  use_double_wide_mlp : Bool := false

  hidden_size_per_layer_input : UInt64 := 0
  vocab_size_per_layer_input : UInt64 := 262144

  enable_moe_block : Bool := false
  num_experts : UInt64 := 0
  top_k_experts : UInt64 := 0
  moe_intermediate_size : UInt64 := 0

  use_cache : Bool := true
  tie_word_embeddings : Bool := true
  final_logit_softcapping : Float := 30.0

  pad_token_id : Option UInt64 := some 0
  bos_token_id : Option UInt64 := some 2
  eos_token_id : Option UInt64 := some 1
  deriving Repr, Inhabited

namespace Gemma4Config

/-- Default Gemma 4 E4B-style config. -/
def gemma4_E4B : Gemma4Config :=
  {
    vocab_size := 262144
    hidden_size := 2560
    intermediate_size := 10240
    num_hidden_layers := 42
    num_attention_heads := 8
    num_key_value_heads := 2
    num_global_key_value_heads := 0
    head_dim := 256
    global_head_dim := 512
    sliding_window := 512
    sliding_rope_theta := 10000.0
    full_rope_theta := 1000000.0
    full_partial_rotary_factor := 0.25
    rms_norm_eps := 1e-6
    max_position_embeddings := 32768
    attention_bias := false
    attention_dropout := 0.0
    hidden_activation := "gelu_pytorch_tanh"
    attention_k_eq_v := false
    use_bidirectional_attention := ""
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    full_attention_interval := 6
    num_kv_shared_layers := 18
    use_double_wide_mlp := false
    hidden_size_per_layer_input := 256
    vocab_size_per_layer_input := 262144
    enable_moe_block := false
    num_experts := 0
    top_k_experts := 0
    moe_intermediate_size := 0
    use_cache := true
    tie_word_embeddings := true
    final_logit_softcapping := 30.0
    pad_token_id := some 0
    bos_token_id := some 2
    eos_token_id := some 1
  }

/-- Default Gemma 4 E2B-style config. -/
def gemma4_E2B : Gemma4Config :=
  {
    gemma4_E4B with
    hidden_size := 1536
    intermediate_size := 6144
    num_hidden_layers := 35
    num_key_value_heads := 1
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    full_attention_interval := 5
    num_kv_shared_layers := 20
    use_double_wide_mlp := true
  }

/-- Default Gemma 4 26B-A4B-style config. -/
def gemma4_26B_A4B : Gemma4Config :=
  {
    vocab_size := 262144
    hidden_size := 2816
    intermediate_size := 2112
    num_hidden_layers := 30
    num_attention_heads := 16
    num_key_value_heads := 8
    num_global_key_value_heads := 2
    head_dim := 256
    global_head_dim := 512
    sliding_window := 1024
    sliding_rope_theta := 10000.0
    full_rope_theta := 1000000.0
    full_partial_rotary_factor := 0.25
    rms_norm_eps := 1e-6
    max_position_embeddings := 32768
    attention_bias := false
    attention_dropout := 0.0
    hidden_activation := "gelu_pytorch_tanh"
    attention_k_eq_v := true
    use_bidirectional_attention := "vision"
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
    full_attention_interval := 6
    num_kv_shared_layers := 0
    use_double_wide_mlp := false
    hidden_size_per_layer_input := 0
    vocab_size_per_layer_input := 262144
    enable_moe_block := true
    num_experts := 128
    top_k_experts := 8
    moe_intermediate_size := 704
    use_cache := true
    tie_word_embeddings := true
    final_logit_softcapping := 30.0
    pad_token_id := some 0
    bos_token_id := some 2
    eos_token_id := some 1
  }

/-- Default Gemma 4 31B-style config. -/
def gemma4_31B : Gemma4Config :=
  {
    gemma4_26B_A4B with
    hidden_size := 5376
    intermediate_size := 21504
    num_hidden_layers := 60
    num_attention_heads := 32
    num_key_value_heads := 16
    num_global_key_value_heads := 4
    enable_moe_block := false
    num_experts := 0
    top_k_experts := 0
    moe_intermediate_size := 0
    layer_types := #[
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention,
      .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .slidingAttention, .fullAttention
    ]
  }

/-- Default Gemma 4 hybrid schedule: every `interval`th layer is full attention,
    and the final layer is always full attention. -/
def defaultLayerTypes (numLayers interval : UInt64) : Array LayerType :=
  Id.run do
    let n := numLayers.toNat
    let mut out : Array LayerType := Array.mkEmpty n
    if n == 0 then
      return out
    if interval == 0 then
      for _ in [:n] do
        out := out.push .fullAttention
      return out
    for i in [:n] do
      let idx1 := (i + 1).toUInt64
      let isFull := (idx1 % interval) == 0 || i + 1 == n
      out := out.push (if isFull then .fullAttention else .slidingAttention)
    return out

def normalizedLayerTypes (cfg : Gemma4Config) : Array LayerType :=
  let base :=
    if cfg.layer_types.size == cfg.num_hidden_layers.toNat then
      cfg.layer_types
    else
      defaultLayerTypes cfg.num_hidden_layers cfg.full_attention_interval
  if base.isEmpty then
    base
  else
    base.set! (base.size - 1) .fullAttention

def normalize (cfg : Gemma4Config) : Gemma4Config :=
  { cfg with
    layer_types := normalizedLayerTypes cfg
    vocab_size_per_layer_input :=
      if cfg.hidden_size_per_layer_input == 0 then cfg.vocab_size_per_layer_input
      else if cfg.vocab_size_per_layer_input == 0 then cfg.vocab_size else cfg.vocab_size_per_layer_input
  }

def fullHeadDim (cfg : Gemma4Config) : UInt64 :=
  if cfg.global_head_dim == 0 then cfg.head_dim else cfg.global_head_dim

def maxHeadDim (cfg : Gemma4Config) : UInt64 :=
  if cfg.head_dim >= fullHeadDim cfg then cfg.head_dim else fullHeadDim cfg

def fullNumKVHeads (cfg : Gemma4Config) : UInt64 :=
  if cfg.num_global_key_value_heads == 0 then cfg.num_key_value_heads else cfg.num_global_key_value_heads

def maxKVHeads (cfg : Gemma4Config) : UInt64 :=
  if cfg.num_key_value_heads >= fullNumKVHeads cfg then cfg.num_key_value_heads else fullNumKVHeads cfg

def maxAttentionDim (cfg : Gemma4Config) : UInt64 :=
  cfg.num_attention_heads * maxHeadDim cfg

def maxKVProjDim (cfg : Gemma4Config) : UInt64 :=
  maxKVHeads cfg * maxHeadDim cfg

def fullRotaryDim (cfg : Gemma4Config) : UInt64 :=
  let raw := ((fullHeadDim cfg).toFloat * cfg.full_partial_rotary_factor).toUInt64
  let base :=
    if raw == 0 then fullHeadDim cfg
    else if raw > fullHeadDim cfg then fullHeadDim cfg
    else raw
  if base <= 2 then
    2
  else if base % 2 == 0 then
    base
  else
    base - 1

def fullRotaryHalfDim (cfg : Gemma4Config) : UInt64 :=
  (fullRotaryDim cfg) / 2

def maxIntermediateSize (cfg : Gemma4Config) : UInt64 :=
  if cfg.use_double_wide_mlp then
    2 * cfg.intermediate_size
  else
    cfg.intermediate_size

def layerTypeAt (cfg : Gemma4Config) (layerIdx : UInt64) : LayerType :=
  (normalizedLayerTypes cfg).getD layerIdx.toNat .fullAttention

def firstKVSharedLayer (cfg : Gemma4Config) : UInt64 :=
  if cfg.num_kv_shared_layers == 0 || cfg.num_kv_shared_layers >= cfg.num_hidden_layers then
    0
  else
    cfg.num_hidden_layers - cfg.num_kv_shared_layers

def isKVSharedLayer (cfg : Gemma4Config) (layerIdx : UInt64) : Bool :=
  let first := firstKVSharedLayer cfg
  cfg.num_kv_shared_layers > 0 && first > 0 && layerIdx >= first

def layerIntermediateSize (cfg : Gemma4Config) (layerIdx : UInt64) : UInt64 :=
  if cfg.use_double_wide_mlp && isKVSharedLayer cfg layerIdx then
    2 * cfg.intermediate_size
  else
    cfg.intermediate_size

def sharedSourceLayer? (cfg : Gemma4Config) (layerIdx : UInt64) : Option UInt64 :=
  if !(isKVSharedLayer cfg layerIdx) then
    none
  else
    Id.run do
      let target := layerTypeAt cfg layerIdx
      let first := firstKVSharedLayer cfg
      if first == 0 then
        return none
      let mut src : Option UInt64 := none
      for i in [:first.toNat] do
        let idx := i.toUInt64
        if layerTypeAt cfg idx == target then
          src := some idx
      return src

def isMoE (cfg : Gemma4Config) : Bool :=
  cfg.enable_moe_block && cfg.num_experts > 0 && cfg.top_k_experts > 0 && cfg.moe_intermediate_size > 0

def hasPerLayerInput (cfg : Gemma4Config) : Bool :=
  cfg.hidden_size_per_layer_input > 0

end Gemma4Config

abbrev Config := Gemma4Config

namespace Config

def gemma4_E2B : Config := Gemma4Config.gemma4_E2B
def gemma4_E4B : Config := Gemma4Config.gemma4_E4B
def gemma4_26B_A4B : Config := Gemma4Config.gemma4_26B_A4B
def gemma4_31B : Config := Gemma4Config.gemma4_31B

def normalize (cfg : Config) : Config := Gemma4Config.normalize cfg
def normalizedLayerTypes (cfg : Config) : Array LayerType := Gemma4Config.normalizedLayerTypes cfg
def defaultLayerTypes (numLayers interval : UInt64) : Array LayerType := Gemma4Config.defaultLayerTypes numLayers interval
def fullHeadDim (cfg : Config) : UInt64 := Gemma4Config.fullHeadDim cfg
def maxHeadDim (cfg : Config) : UInt64 := Gemma4Config.maxHeadDim cfg
def fullNumKVHeads (cfg : Config) : UInt64 := Gemma4Config.fullNumKVHeads cfg
def maxKVHeads (cfg : Config) : UInt64 := Gemma4Config.maxKVHeads cfg
def maxAttentionDim (cfg : Config) : UInt64 := Gemma4Config.maxAttentionDim cfg
def maxKVProjDim (cfg : Config) : UInt64 := Gemma4Config.maxKVProjDim cfg
def fullRotaryDim (cfg : Config) : UInt64 := Gemma4Config.fullRotaryDim cfg
def fullRotaryHalfDim (cfg : Config) : UInt64 := Gemma4Config.fullRotaryHalfDim cfg
def maxIntermediateSize (cfg : Config) : UInt64 := Gemma4Config.maxIntermediateSize cfg
def layerTypeAt (cfg : Config) (layerIdx : UInt64) : LayerType := Gemma4Config.layerTypeAt cfg layerIdx
def firstKVSharedLayer (cfg : Config) : UInt64 := Gemma4Config.firstKVSharedLayer cfg
def isKVSharedLayer (cfg : Config) (layerIdx : UInt64) : Bool := Gemma4Config.isKVSharedLayer cfg layerIdx
def layerIntermediateSize (cfg : Config) (layerIdx : UInt64) : UInt64 := Gemma4Config.layerIntermediateSize cfg layerIdx
def sharedSourceLayer? (cfg : Config) (layerIdx : UInt64) : Option UInt64 := Gemma4Config.sharedSourceLayer? cfg layerIdx
def isMoE (cfg : Config) : Bool := Gemma4Config.isMoE cfg
def hasPerLayerInput (cfg : Config) : Bool := Gemma4Config.hasPerLayerInput cfg

end Config

end torch.gemma4

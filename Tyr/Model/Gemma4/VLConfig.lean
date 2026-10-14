/-
  Tyr/Model/Gemma4/VLConfig.lean

  Multimodal (vision + text) configuration for Gemma 4.
-/
import Tyr.Model.Gemma4.Config

/-! # Gemma 4 multimodal configuration

`torch.gemma4.Gemma4VisionConfig`, `torch.gemma4.Gemma4ImageProcessorConfig`
(from `processor_config.json`) and the composite `torch.gemma4.Gemma4VLConfig`,
with derived sizes such as `torch.gemma4.Gemma4VisionConfig.patchDim`.
-/

namespace torch.gemma4

/-- Vision backbone config published in Gemma 4 `config.json`. -/
structure Gemma4VisionConfig where
  hidden_size : UInt64 := 768
  intermediate_size : UInt64 := 3072
  num_hidden_layers : UInt64 := 16
  num_attention_heads : UInt64 := 12
  num_key_value_heads : UInt64 := 12
  head_dim : UInt64 := 64
  patch_size : UInt64 := 16
  pooling_kernel_size : UInt64 := 3
  position_embedding_size : UInt64 := 10240
  rms_norm_eps : Float := 1e-6
  rope_theta : Float := 100.0
  standardize : Bool := false
  use_clipped_linears : Bool := false
  deriving Repr, Inhabited

namespace Gemma4VisionConfig

def in_channels (_cfg : Gemma4VisionConfig) : UInt64 := 3

def patchDim (cfg : Gemma4VisionConfig) : UInt64 :=
  in_channels cfg * cfg.patch_size * cfg.patch_size

def pooledTokenCount (cfg : Gemma4VisionConfig) (patchRows patchCols : UInt64) : UInt64 :=
  let pool := if cfg.pooling_kernel_size == 0 then 1 else cfg.pooling_kernel_size
  (patchRows / pool) * (patchCols / pool)

end Gemma4VisionConfig

/-- Image processor config published in Gemma 4 `processor_config.json`. -/
structure Gemma4ImageProcessorConfig where
  do_resize : Bool := true
  do_rescale : Bool := true
  do_normalize : Bool := false
  image_mean : Array Float := #[0.0, 0.0, 0.0]
  image_std : Array Float := #[1.0, 1.0, 1.0]
  image_seq_length : UInt64 := 280
  max_soft_tokens : UInt64 := 280
  patch_size : UInt64 := 16
  pooling_kernel_size : UInt64 := 3
  rescale_factor : Float := 1.0 / 255.0
  deriving Repr, Inhabited

namespace Gemma4ImageProcessorConfig

def softTokenCount (cfg : Gemma4ImageProcessorConfig) : UInt64 :=
  if cfg.max_soft_tokens > 0 then cfg.max_soft_tokens else cfg.image_seq_length

end Gemma4ImageProcessorConfig

/-- Composite Gemma 4 multimodal config. -/
structure Gemma4VLConfig where
  text_config : Config := Config.gemma4_E4B
  vision_config : Gemma4VisionConfig := {}
  image_processor : Gemma4ImageProcessorConfig := {}
  boi_token_id : UInt64 := 255999
  image_token_id : UInt64 := 258880
  eoi_token_id : UInt64 := 258882
  video_token_id : UInt64 := 258884
  audio_token_id : UInt64 := 258881
  vision_soft_tokens_per_image : UInt64 := 280
  tie_word_embeddings : Bool := true
  deriving Repr, Inhabited

namespace Gemma4VLConfig

def normalize (cfg : Gemma4VLConfig) : Gemma4VLConfig :=
  let textCfg := Config.normalize cfg.text_config
  let imageProc :=
    { cfg.image_processor with
      patch_size :=
        if cfg.image_processor.patch_size == 0 then cfg.vision_config.patch_size else cfg.image_processor.patch_size
      pooling_kernel_size :=
        if cfg.image_processor.pooling_kernel_size == 0 then
          cfg.vision_config.pooling_kernel_size
        else
          cfg.image_processor.pooling_kernel_size
      image_seq_length :=
        if cfg.image_processor.image_seq_length == 0 then cfg.vision_soft_tokens_per_image else cfg.image_processor.image_seq_length
      max_soft_tokens :=
        if cfg.image_processor.max_soft_tokens == 0 then cfg.vision_soft_tokens_per_image else cfg.image_processor.max_soft_tokens
    }
  {
    cfg with
    text_config := textCfg
    image_processor := imageProc
    vision_soft_tokens_per_image :=
      if cfg.vision_soft_tokens_per_image == 0 then imageProc.softTokenCount else cfg.vision_soft_tokens_per_image
  }

def imageSoftTokenCount (cfg : Gemma4VLConfig) : UInt64 :=
  let procBudget := cfg.image_processor.softTokenCount
  if procBudget > 0 then procBudget else cfg.vision_soft_tokens_per_image

end Gemma4VLConfig

abbrev VisionConfig := Gemma4VisionConfig
abbrev ImageProcessorConfig := Gemma4ImageProcessorConfig
abbrev VLConfig := Gemma4VLConfig

namespace VisionConfig

def in_channels (cfg : VisionConfig) : UInt64 := Gemma4VisionConfig.in_channels cfg
def patchDim (cfg : VisionConfig) : UInt64 := Gemma4VisionConfig.patchDim cfg
def pooledTokenCount (cfg : VisionConfig) (patchRows patchCols : UInt64) : UInt64 :=
  Gemma4VisionConfig.pooledTokenCount cfg patchRows patchCols

end VisionConfig

namespace ImageProcessorConfig

def softTokenCount (cfg : ImageProcessorConfig) : UInt64 := Gemma4ImageProcessorConfig.softTokenCount cfg

end ImageProcessorConfig

namespace VLConfig

def normalize (cfg : VLConfig) : VLConfig := Gemma4VLConfig.normalize cfg
def imageSoftTokenCount (cfg : VLConfig) : UInt64 := Gemma4VLConfig.imageSoftTokenCount cfg

end VLConfig

end torch.gemma4

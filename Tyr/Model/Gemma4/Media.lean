/-
  Tyr/Model/Gemma4/Media.lean

  Apple-only image preprocessing helpers for Gemma 4 multimodal inputs.
-/
import Tyr.Torch
import Tyr.Model.Gemma4.VLConfig

/-! # Gemma 4 media loading

`torch.gemma4.media.loadImagePatchGrid` loads one image and returns its patch
grid `[patchRows, patchCols, patchDim]` for the vision tower.
-/

namespace torch.gemma4.media

open torch
open torch.gemma4

/-- Load one image and return a patch grid `[patchRows, patchCols, patchDim]`
    sized for the configured Gemma 4 image token budget. -/
def loadImagePatchGrid (cfg : VLConfig) (path : String)
    : IO (Sigma (fun patchRows =>
        Sigma (fun patchCols =>
          T #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config]))) := do
  let dyn ← data.loadGemma4ImagePatchGrid
    path
    cfg.vision_config.patch_size
    cfg.vision_config.pooling_kernel_size
    cfg.imageSoftTokenCount
    cfg.image_processor.rescale_factor
  let shp := dyn.runtimeShape
  if shp.size != 3 then
    throw <| IO.userError s!"Expected Gemma4 image patch grid rank 3, got shape={shp}"
  let patchRows := shp.getD 0 0
  let patchCols := shp.getD 1 0
  let patchDim := shp.getD 2 0
  if patchDim != VisionConfig.patchDim cfg.vision_config then
    throw <| IO.userError
      s!"Gemma4 image patch dim mismatch: expected {VisionConfig.patchDim cfg.vision_config}, got {patchDim}"
  pure ⟨patchRows, ⟨patchCols, reshape dyn #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config]⟩⟩

end torch.gemma4.media

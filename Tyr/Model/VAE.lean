/-
  Tyr/Model/VAE.lean

  VAE (Variational Autoencoder) for Flux latent space.
  Re-exports all VAE components and provides full AutoEncoder.
-/

import Tyr.Model.VAE.ResnetBlock
import Tyr.Model.VAE.AttnBlock
import Tyr.Model.VAE.Upsample
import Tyr.Model.VAE.Decoder
import Tyr.Model.VAE.Weights
import Tyr.Log

/-! # Flux VAE

`torch.vae.AutoEncoder` decodes Flux latents: it inverts the BatchNorm latent
normalization (`torch.vae.AutoEncoder.invNormalize`), unpacks $16\times16$
latents and runs the decoder.
-/

namespace torch.vae

open torch.Log

/-- Full AutoEncoder with BatchNorm statistics for proper normalization.
    The encode/decode pipeline uses BN normalization on packed latents. -/
structure AutoEncoder where
  /-- The VAE decoder -/
  decoder : Decoder
  /-- BatchNorm running mean for latent normalization [128] -/
  bn_running_mean : T #[128]
  /-- BatchNorm running variance for latent normalization [128] -/
  bn_running_var : T #[128]
  deriving TensorStruct

namespace AutoEncoder

/-- BatchNorm epsilon for numerical stability -/
def bn_eps : Float := 1e-4

/-- Inverse normalize packed latents using BN statistics.
    z: [batch, 128, h, w] → [batch, 128, h, w]
    Formula: z * sqrt(running_var + eps) + running_mean -/
def invNormalize (ae : AutoEncoder) (z : T #[1, 128, 16, 16]) : T #[1, 128, 16, 16] :=
  -- Compute scale: sqrt(running_var + eps)
  let var_plus_eps := ae.bn_running_var + (torch.full #[128] bn_eps)
  let scale := nn.sqrt var_plus_eps
  -- Reshape and expand for element-wise ops: [128] → [1, 128, 1, 1] → [1, 128, 16, 16]
  let scale := reshape scale #[1, 128, 1, 1]
  let mean := reshape ae.bn_running_mean #[1, 128, 1, 1]
  -- Expand to full shape using concrete literal shapes (avoids type variable FFI issue)
  let scale := nn.expand scale #[1, 128, 16, 16]
  let mean := nn.expand mean #[1, 128, 16, 16]
  z * scale + mean

/-- Unpack latents from packed format to VAE decoder format (16x16 packed → 32x32 unpacked).
    Packed: [1, 128, 16, 16]
    Unpacked: [1, 32, 32, 32]
    Uses concrete shapes to avoid FFI issues with type variables. -/
def unpackLatents16x16 (z : T #[1, 128, 16, 16]) : T #[1, 32, 32, 32] :=
  -- Reshape: [1, 128, 16, 16] → [1, 32, 2, 2, 16, 16]
  let z := reshape z #[1, 32, 2, 2, 16, 16]
  -- Permute: [1, 32, 2, 2, 16, 16] → [1, 32, 16, 2, 16, 2]
  let z := permute z #[0, 1, 4, 2, 5, 3]
  -- Reshape to final: [1, 32, 32, 32]
  reshape z #[1, 32, 32, 32]

/-- Full decode pipeline: inv_normalize → unpack → decoder.
    Input: packed latents [1, 128, 16, 16]
    Output: RGB image [1, 3, 256, 256]
    Uses concrete shapes to avoid FFI issues with type variables. -/
def decode (ae : AutoEncoder) (z : T #[1, 128, 16, 16]) : T #[1, 3, 256, 256] :=
  -- Step 1: Inverse normalize using BN stats (uses broadcasting)
  let z := ae.invNormalize z
  -- Step 2: Unpack latents [1, 128, 16, 16] → [1, 32, 32, 32]
  let z := unpackLatents16x16 z
  -- Step 3: Decode through VAE decoder (batch=1 specialized version)
  ae.decoder.forward1 z

end AutoEncoder

/-- Load full AutoEncoder from SafeTensors including BN stats. -/
def loadAutoEncoder (path : String) (log : Handlers := {}) : IO AutoEncoder := do
  log.onInfo "Loading AutoEncoder..."

  -- Load decoder
  let decoder ← loadDecoder path log

  -- Load BN running stats
  let bn_running_mean ← safetensors.loadTensor path "bn.running_mean" #[128]
  let bn_running_var ← safetensors.loadTensor path "bn.running_var" #[128]
  log.onInfo "  Loaded BN running stats"

  log.onInfo "AutoEncoder loaded successfully!"
  pure {
    decoder
    bn_running_mean := autograd.set_requires_grad bn_running_mean false
    bn_running_var := autograd.set_requires_grad bn_running_var false
  }

end torch.vae

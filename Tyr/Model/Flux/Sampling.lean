/-
  Tyr/Model/Flux/Sampling.lean

  Sampling / denoising loop for Flux diffusion.
  Implements flow matching / rectified flow sampling.
-/
import Tyr.Torch
import Tyr.Model.Flux.Config
import Tyr.Model.Flux.Model

/-! # Flux sampling

Flow-matching sampler for Flux2. `torch.flux.computeTimesteps` builds the
shifted schedule using `torch.flux.computeEmpiricalMu` and
`torch.flux.generalizedTimeSnrShift`; `torch.flux.eulerStep`,
`torch.flux.denoise` and `torch.flux.sample` integrate the velocity field from
noise to latents, configured by `torch.flux.SamplingConfig`.
-/

namespace torch.flux

/-- Sampling configuration -/
structure SamplingConfig where
  /-- Number of sampling steps -/
  num_steps : Nat := 4
  /-- CFG scale (if using classifier-free guidance) -/
  guidance_scale : Float := 1.0
  deriving Repr, Inhabited

/-- Compute empirical mu for Flux2 schedule. -/
private def computeEmpiricalMu (image_seq_len : UInt64) (num_steps : Nat) : Float :=
  let a1 := 8.73809524e-05
  let b1 := 1.89833333
  let a2 := 0.00016927
  let b2 := 0.45666666
  let img := image_seq_len.toFloat
  if image_seq_len > 4300 then
    a2 * img + b2
  else
    let m_200 := a2 * img + b2
    let m_10 := a1 * img + b1
    let a := (m_200 - m_10) / 190.0
    let b := m_200 - 200.0 * a
    a * num_steps.toFloat + b

/-- Generalized time SNR shift. -/
private def generalizedTimeSnrShift (t : Float) (mu : Float) (sigma : Float) : Float :=
  if t == 0.0 then
    0.0
  else
    let expMu := Float.exp mu
    let denom := expMu + Float.pow (1.0 / t - 1.0) sigma
    expMu / denom

/-- Compute timestep schedule for rectified flow (Flux2 schedule).
    Returns array of (t_curr, t_next) pairs. -/
def computeTimesteps (image_seq_len : UInt64) (num_steps : Nat) : Array (Float × Float) :=
  let mu := computeEmpiricalMu image_seq_len num_steps
  let denom := num_steps.toFloat
  Array.ofFn fun (i : Fin num_steps) =>
    let t_curr := 1.0 - (i.val.toFloat / denom)
    let t_next := 1.0 - ((i.val + 1).toFloat / denom)
    (generalizedTimeSnrShift t_curr mu 1.0, generalizedTimeSnrShift t_next mu 1.0)

/-- Single denoising step (Euler method).
    x: current state
    pred: model prediction (velocity)
    t_curr, t_next: timestep boundaries
    Returns: next state -/
def eulerStep {s : Shape}
    (x : T s)
    (pred : T s)
    (t_curr t_next : Float)
    : T s :=
  let dt := t_next - t_curr
  x + (pred * dt)

/-- Denoise latents using rectified flow.
    noise: [batch, seq, dim] - starting noise
    txt: [batch, txt_seq, txt_dim] - text conditioning
    model: Flux model
    Returns: [batch, seq, dim] - denoised latents -/
def denoise {batch img_seq txt_seq : UInt64} (cfg : FluxConfig)
    (model : FluxModel cfg)
    (noise : T #[batch, img_seq, cfg.in_channels])
    (txt : T #[batch, txt_seq, cfg.context_in_dim])
    (img_ids : T #[batch, img_seq, 4])
    (txt_ids : T #[batch, txt_seq, 4])
    (num_steps : Nat := 4)
    : T #[batch, img_seq, cfg.in_channels] :=
  let timesteps := computeTimesteps img_seq num_steps
  -- Use fold instead of Id.run do with for loop to avoid runtime bug
  timesteps.foldl (init := noise) fun x (t_curr, t_next) =>
    -- Create timestep tensor
    let t := torch.full #[batch] t_curr
    -- Model prediction
    let pred := model.forward cfg x txt t img_ids txt_ids
    -- Euler step
    eulerStep x pred t_curr t_next

/-- Sample from pure noise.
    Returns denoised latents ready for VAE decoding. -/
def sample {batch img_seq txt_seq : UInt64} (cfg : FluxConfig)
    (model : FluxModel cfg)
    (txt : T #[batch, txt_seq, cfg.context_in_dim])
    (img_ids : T #[batch, img_seq, 4])
    (txt_ids : T #[batch, txt_seq, 4])
    (num_steps : Nat := 4)
    : IO (T #[batch, img_seq, cfg.in_channels]) := do
  -- Start from pure noise
  let noise ← torch.randn #[batch, img_seq, cfg.in_channels]
  pure (denoise cfg model noise txt img_ids txt_ids num_steps)

/-- Compute image position IDs for RoPE.
    For a 32x32 latent, returns position IDs with 4 axes: [t, h, w, l]
    where t=0 (timestep), h/w are spatial coordinates, l=0 (layer).
    Returns: [1, seq, 4] tensor (shape-erased to T #[] at compile time) -/
def computeImagePositionIds (height width : UInt64) : IO (T #[1, height * width, 4]) := do
  let seq := height * width
  let h := torch.arange 0 height
  let w := torch.arange 0 width
  let h_grid := reshape h #[height, 1]
  let h_grid := nn.expand h_grid #[height, width]
  let w_grid := reshape w #[1, width]
  let w_grid := nn.expand w_grid #[height, width]
  let h_flat := reshape h_grid #[seq]
  let w_flat := reshape w_grid #[seq]
  let t_flat := torch.full_int #[seq] 0
  let l_flat := torch.full_int #[seq] 0
  let t_col := reshape t_flat #[seq, 1]
  let h_col := reshape h_flat #[seq, 1]
  let w_col := reshape w_flat #[seq, 1]
  let l_col := reshape l_flat #[seq, 1]
  let ids := nn.cat (nn.cat (nn.cat t_col h_col 1) w_col 1) l_col 1
  pure (reshape ids #[1, seq, 4])

/-- Compute text position IDs for RoPE.
    Text positions are simple sequential IDs.
    Returns: [1, seq_len, 4] tensor -/
def computeTextPositionIds (seq_len : UInt64) : IO (T #[1, seq_len, 4]) := do
  let t := torch.full_int #[seq_len] 0
  let h := torch.full_int #[seq_len] 0
  let w := torch.full_int #[seq_len] 0
  let l := torch.arange 0 seq_len
  let t_col := reshape t #[seq_len, 1]
  let h_col := reshape h #[seq_len, 1]
  let w_col := reshape w #[seq_len, 1]
  let l_col := reshape l #[seq_len, 1]
  let ids := nn.cat (nn.cat (nn.cat t_col h_col 1) w_col 1) l_col 1
  pure (reshape ids #[1, seq_len, 4])

/-- Unpack Flux latents from sequence format to VAE spatial format.
    Flux uses 2×2 spatial patchification, packing 32 channels into 128-dim patches.

    Input: [batch, seq, 128] where seq = packed_H × packed_W
    Output: [batch, 32, packed_H × 2, packed_W × 2]

    The einops equivalent is:
      rearrange(z, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", p1=2, p2=2, h=H, w=W)

    For 256×256 output images:
    - packed_H = packed_W = 16 (so seq = 256)
    - VAE input becomes [batch, 32, 32, 32]
    - VAE outputs [batch, 3, 256, 256] (8× upscaling) -/
def unpackLatents {batch packed_H packed_W : UInt64}
    (z : T #[batch, packed_H * packed_W, 128])
    : T #[batch, 32, packed_H * 2, packed_W * 2] :=
  -- Step 1: Reshape sequence to spatial grid with packed channels
  -- [batch, seq, 128] → [batch, packed_H, packed_W, 32, 2, 2]
  let z := reshape z #[batch, packed_H, packed_W, 32, 2, 2]
  -- Step 2: Permute to interleave spatial patches
  -- [batch, packed_H, packed_W, 32, 2, 2] → [batch, 32, packed_H, 2, packed_W, 2]
  let z := permute z #[0, 3, 1, 4, 2, 5]
  -- Step 3: Reshape to final VAE format
  -- [batch, 32, packed_H, 2, packed_W, 2] → [batch, 32, packed_H*2, packed_W*2]
  reshape z #[batch, 32, packed_H * 2, packed_W * 2]

/-- Pack VAE spatial format back to Flux sequence format (inverse of unpackLatents).
    Used when encoding images to latents for img2img or inpainting.

    Input: [batch, 32, H, W] where H and W are divisible by 2
    Output: [batch, (H/2) × (W/2), 128] -/
def packLatents {batch H W : UInt64}
    (z : T #[batch, 32, H, W])
    : T #[batch, (H / 2) * (W / 2), 128] :=
  let packed_H := H / 2
  let packed_W := W / 2
  -- Step 1: Reshape to expose 2×2 patches
  -- [batch, 32, H, W] → [batch, 32, packed_H, 2, packed_W, 2]
  let z := reshape z #[batch, 32, packed_H, 2, packed_W, 2]
  -- Step 2: Permute to group patches with channels
  -- [batch, 32, packed_H, 2, packed_W, 2] → [batch, packed_H, packed_W, 32, 2, 2]
  let z := permute z #[0, 2, 4, 1, 3, 5]
  -- Step 3: Flatten to sequence format
  -- [batch, packed_H, packed_W, 32, 2, 2] → [batch, packed_H*packed_W, 128]
  reshape z #[batch, packed_H * packed_W, 128]

end torch.flux

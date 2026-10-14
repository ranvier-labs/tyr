/-
  Tyr/Model/Flux/RoPE.lean

  4-axis rotary position embeddings for Flux.
  Splits head_dim across axes and applies RoPE per axis.
-/
import Tyr.Torch

/-! # Flux rotary embeddings

4-axis rotary position embeddings for Flux. `torch.flux.ropeEmbed` builds the
per-axis cos/sin tables from position ids, and `torch.flux.applyRope` /
`torch.flux.applyRopeAttn` rotate the Q/K tensors.
-/

namespace torch.flux

private def axisIds {batch seq : UInt64} (ids : T #[batch, seq, 4]) (axis : UInt64) : T #[batch, seq] :=
  reshape (data.slice ids 2 axis 1) #[batch, seq]

private def ropeAxis {batch seq : UInt64}
    (ids : T #[batch, seq]) (dim : UInt64) (theta : UInt64)
    : T #[batch, seq, dim / 2, 2, 2] :=
  let scale := torch.arange 0 dim 2
  let scale := _root_.torch.toFloat' scale
  let scale := div_scalar scale dim.toFloat
  let logTheta := Float.log theta.toFloat
  let omega := nn.exp (mul_scalar scale (-logTheta))  -- 1 / (theta ** scale)

  let pos := _root_.torch.toFloat' ids
  let pos := nn.unsqueeze pos 2  -- [batch, seq, 1]
  let pos := nn.expand pos #[batch, seq, dim / 2]
  let omega := reshape omega #[1, 1, dim / 2]
  let omega := nn.expand omega #[batch, seq, dim / 2]
  let out := pos * omega

  let cos := nn.cos out
  let sin := nn.sin out
  let neg_sin := mul_scalar sin (-1.0)

  let cos_u := nn.unsqueeze cos 3
  let neg_sin_u := nn.unsqueeze neg_sin 3
  let sin_u := nn.unsqueeze sin 3
  let cos_u2 := nn.unsqueeze cos 3

  let stacked := nn.cat (nn.cat (nn.cat cos_u neg_sin_u 3) sin_u 3) cos_u2 3
  reshape stacked #[batch, seq, dim / 2, 2, 2]

/-- Build 4-axis RoPE embeddings for Flux.
    ids: [batch, seq, 4] position IDs (t, h, w, l)
    axes_dims: per-axis head_dim split (sum = head_dim)
    theta: RoPE base (Flux uses 2000) -/
def ropeEmbed {batch seq head_dim : UInt64}
    (ids : T #[batch, seq, 4])
    (axes_dims : Array UInt64)
    (theta : UInt64 := 2000)
    : T #[batch, 1, seq, head_dim / 2, 2, 2] :=
  let d0 := axes_dims.getD 0 0
  let d1 := axes_dims.getD 1 0
  let d2 := axes_dims.getD 2 0
  let d3 := axes_dims.getD 3 0

  let ids0 := axisIds ids 0
  let ids1 := axisIds ids 1
  let ids2 := axisIds ids 2
  let ids3 := axisIds ids 3

  let rope0 := ropeAxis ids0 d0 theta
  let rope1 := ropeAxis ids1 d1 theta
  let rope2 := ropeAxis ids2 d2 theta
  let rope3 := ropeAxis ids3 d3 theta

  let emb := nn.cat (nn.cat (nn.cat rope0 rope1 2) rope2 2) rope3 2
  let emb := nn.unsqueeze emb 1
  reshape emb #[batch, 1, seq, head_dim / 2, 2, 2]

/-- Apply Flux RoPE embeddings (from `ropeEmbed`) to Q/K tensors. -/
private def applyRopeAttn {batch n_head seq head_dim : UInt64}
    (x : T #[batch, n_head, seq, head_dim])
    (pe : T #[batch, 1, seq, head_dim / 2, 2, 2])
    : T #[batch, n_head, seq, head_dim] :=
  let x := _root_.torch.toFloat' x
  let x := reshape x #[batch, n_head, seq, head_dim / 2, 1, 2]
  let x0 := reshape (data.slice x 5 0 1) #[batch, n_head, seq, head_dim / 2, 1]
  let x1 := reshape (data.slice x 5 1 1) #[batch, n_head, seq, head_dim / 2, 1]
  let x0 := nn.expand x0 #[batch, n_head, seq, head_dim / 2, 2]
  let x1 := nn.expand x1 #[batch, n_head, seq, head_dim / 2, 2]

  let pe0 := reshape (data.slice pe 5 0 1) #[batch, 1, seq, head_dim / 2, 2]
  let pe1 := reshape (data.slice pe 5 1 1) #[batch, 1, seq, head_dim / 2, 2]
  let pe0 := nn.expand pe0 #[batch, n_head, seq, head_dim / 2, 2]
  let pe1 := nn.expand pe1 #[batch, n_head, seq, head_dim / 2, 2]

  let out := pe0 * x0 + pe1 * x1
  reshape out #[batch, n_head, seq, head_dim]

def applyRope {batch seq n_head head_dim : UInt64}
    (x : T #[batch, seq, n_head, head_dim])
    (pe : T #[batch, 1, seq, head_dim / 2, 2, 2])
    : T #[batch, seq, n_head, head_dim] :=
  let x_attn := nn.transpose_for_attention x
  let out := applyRopeAttn x_attn pe
  nn.transpose_from_attention out

end torch.flux

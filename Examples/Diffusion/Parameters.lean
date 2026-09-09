import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Derive

/-! Parameter and rotary types owned by the discrete diffusion model.

These retain the shapes and zero-initialized residual projections previously
shared with the removed NanoProof example.
-/
namespace torch.diffusion

structure RotaryCache (seq_len head_dim : UInt64) where
  cos : T #[seq_len, head_dim / 2]
  sin : T #[seq_len, head_dim / 2]

def RotaryCache.init (seq_len head_dim : UInt64) (base : Float := 10000.0)
    : IO (RotaryCache seq_len head_dim) := do
  let (cos, sin) ← rotary.computeFreqs seq_len head_dim base
  return { cos, sin }

structure AttentionParams (n_embd n_head n_kv_head : UInt64) where
  c_q : T #[n_head * (n_embd / n_head), n_embd]
  c_k : T #[n_kv_head * (n_embd / n_head), n_embd]
  c_v : T #[n_kv_head * (n_embd / n_head), n_embd]
  c_proj : T #[n_embd, n_embd]
  deriving TensorStruct

structure MLPParams (n_embd : UInt64) where
  c_fc : T #[4 * n_embd, n_embd]
  c_proj : T #[n_embd, 4 * n_embd]
  deriving TensorStruct

structure BlockParams (n_embd n_head n_kv_head : UInt64) where
  attn : AttentionParams n_embd n_head n_kv_head
  mlp : MLPParams n_embd
  deriving TensorStruct

def makeLeafParam {s : Shape} (t : T s) : T s :=
  autograd.set_requires_grad (autograd.detach t) true

def AttentionParams.init (n_embd n_head n_kv_head : UInt64)
    : IO (AttentionParams n_embd n_head n_kv_head) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_q ← randn #[n_head * (n_embd / n_head), n_embd] false
  let c_k ← randn #[n_kv_head * (n_embd / n_head), n_embd] false
  let c_v ← randn #[n_kv_head * (n_embd / n_head), n_embd] false
  return {
    c_q := makeLeafParam (mul_scalar c_q scale)
    c_k := makeLeafParam (mul_scalar c_k scale)
    c_v := makeLeafParam (mul_scalar c_v scale)
    c_proj := makeLeafParam (zeros #[n_embd, n_embd])
  }

def MLPParams.init (n_embd : UInt64) : IO (MLPParams n_embd) := do
  let scale := 1.0 / Float.sqrt n_embd.toFloat
  let c_fc ← randn #[4 * n_embd, n_embd] false
  return {
    c_fc := makeLeafParam (mul_scalar c_fc scale)
    c_proj := makeLeafParam (zeros #[n_embd, 4 * n_embd])
  }

def BlockParams.init (n_embd n_head n_kv_head : UInt64)
    : IO (BlockParams n_embd n_head n_kv_head) := do
  return { attn := ← AttentionParams.init n_embd n_head n_kv_head,
           mlp := ← MLPParams.init n_embd }

end torch.diffusion

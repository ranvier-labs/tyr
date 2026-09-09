import Tyr.Torch

namespace torch.data

/-- Effectful inference-only slice write. Callers must own the destination and
serialize access; aliases observe this mutation. Unlike the legacy pure
`sliceScatterInplace`, the IO effect orders reads and writes explicitly.
The native operation disables autograd for the copy. -/
@[extern "lean_torch_copy_slice_io"]
opaque copySliceIO {s src : Shape} (dst : @& T s) (dim start : UInt64)
    (src : @& T src) : IO Unit

/-- Allocate an independent inference tensor copy. The IO effect prevents Lean
from merging identical allocations when distinct mutable buffers are required. -/
@[extern "lean_torch_clone_inference_io"]
opaque cloneInferenceIO {s : Shape} (src : @& T s) : IO (T s)

end torch.data

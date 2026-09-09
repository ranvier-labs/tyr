import Tyr.Torch

namespace torch.data

/-- Effectful inference-only slice write. Callers must own the destination and
serialize access; aliases observe this mutation. Unlike the legacy pure
`sliceScatterInplace`, the IO effect orders reads and writes explicitly.
The native operation disables autograd for the copy. -/
@[extern "lean_torch_copy_slice_io"]
opaque copySliceIO {s src : Shape} (dst : @& T s) (dim start : UInt64)
    (src : @& T src) : IO Unit

end torch.data

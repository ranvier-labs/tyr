#pragma once
#include <lean/lean.h>

// Compare with Tyr/Inference/OwnedKV.lean generated C. Lean 4.29 erases the
// IO world argument; shape arguments are owned and tensors are borrowed.
#ifdef __cplusplus
extern "C" {
#endif
lean_object* lean_torch_copy_slice_io(lean_object*, lean_object*, lean_object*, uint64_t, uint64_t, lean_object*);
#ifdef __cplusplus
}
#endif

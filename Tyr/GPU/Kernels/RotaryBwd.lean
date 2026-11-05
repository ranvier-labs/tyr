/-
  Compatibility import for the working rotary forward/backward kernels.

  The former `rotaryBwd` definition in this module allocated conceptual
  register/shared tiles but never loaded global inputs or wrote the output.
  Keeping that GPU entrypoint would silently expose a no-op backward pass.
  The measured BF16 D64 implementation is
  `Rotary.rotaryBwdQKD64Bf16Direct` in `Rotary.lean`.
-/

import Tyr.GPU.Kernels.Rotary

/-! # Rotary backward (compatibility)

Compatibility import of `Tyr.GPU.Kernels.Rotary`. The former no-op `rotaryBwd`
was removed; the measured BF16 D64 backward is
`Tyr.GPU.Kernels.Rotary.rotaryBwdQKD64Bf16Direct`.
-/

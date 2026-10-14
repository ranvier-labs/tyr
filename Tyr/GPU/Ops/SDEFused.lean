/-
  Tyr/GPU/Ops/SDEFused.lean

  Fused fixed-step Euler–Maruyama SDE solve on device, with diagonal
  noise drawn inline by the virtual-Brownian-tree descent kernel
  (`Tyr.GPU.Kernels.emStepVbt`). Per step this issues one drift and one
  diffusion evaluation, one `mul_scalar` (folding the step size into the
  drift, as in `Tyr.GPU.Ops.RKFused`), and a single fused kernel launch —
  the Brownian increment never materializes in a tensor.

  The descent is time-driven, so the value-query node schedules for the
  whole uniform step grid are precomputed and uploaded once before the
  loop (see `Tyr.GPU.Ops.BrownianFused.buildValueSchedule` for the record
  encoding and the f32-exactness constraint on time tags).

  State layout: flat `T #[n]` Float32; the state is double-buffered, so
  the kernel writing `y1` never aliases the `y0` it reads.
-/
import Tyr.Torch
import Tyr.GPU.Kernels.BrownianSample
import Tyr.GPU.Ops.BrownianFused

/-! # Fused Euler–Maruyama

Fused Euler–Maruyama SDE steps (`Tyr.GPU.Ops.SDEFused.emStepFused`,
`Tyr.GPU.Ops.SDEFused.emSolveFused`) driven by uploaded Brownian schedules.
-/

namespace Tyr.GPU.Ops.SDEFused

open torch
open Tyr.GPU.Kernels
open Tyr.GPU.Ops.BrownianFused

/-- One uploaded value-query schedule: device records plus record count. -/
structure DeviceSchedule where
  records : T #[]
  len : UInt64
  deriving Inhabited

/-- Build and upload the `w(t)` schedule for one query time. -/
def DeviceSchedule.make (T0 T1 tol : Float) (maxDepth : Nat) (t : Float)
    (device : Device) : DeviceSchedule :=
  let sched := buildValueSchedule T0 T1 tol maxDepth t
  { records := uploadSchedule sched device, len := (sched.size / 6).toUInt64 }

/-- One fused Euler–Maruyama step: reads the state from `y0`, writes the
    new state into `y1Out`. `drift`/`diffusion` are the vector fields
    `f(t, y)` and `g(t, y)`; noise is diagonal, `g ∘ ΔW`. -/
def emStepFused {n : UInt64}
    (drift diffusion : Float → T #[n] → T #[n])
    (treeSeed : UInt64) (t h : Float)
    (schedA schedB : DeviceSchedule)
    (y0 : T #[n]) (y1Out : T #[n])
    (blackwell : Bool) (stream : UInt64) : IO Unit := do
  let fh := mul_scalar (drift t y0) h
  let g := diffusion t y0
  let root := tensorRootState treeSeed
  if blackwell then
    emStepVbtBlackwell.launch y1Out y0 fh g schedA.records schedB.records
      root n schedA.len schedB.len 1 1 1 256 1 1 0 stream
  else
    emStepVbt.launch y1Out y0 fh g schedA.records schedB.records
      root n schedA.len schedB.len 1 1 1 256 1 1 0 stream

/-- Fused fixed-step Euler–Maruyama solve from `t0` to `t1` in `steps`
    uniform steps, driven by the VBT over `[T0, T1]` with seed
    `treeSeed` (matching the CPU `VirtualBrownianTree` tensor instance).
    Returns a fresh tensor holding the final state. -/
def emSolveFused {n : UInt64}
    (drift diffusion : Float → T #[n] → T #[n])
    (treeSeed : UInt64) (T0 T1 tol : Float) (maxDepth : Nat)
    (t0 t1 : Float) (steps : Nat)
    (y0 : T #[n])
    (blackwell : Bool) (stream : UInt64)
    (device : Device := Device.CUDA 0) : IO (T #[n]) := do
  let m := Nat.max steps 1
  let h := (t1 - t0) / Float.ofNat m
  -- schedules for the whole step grid, uploaded once
  let mut scheds : Array DeviceSchedule := #[]
  for k in [:m + 1] do
    let t := t0 + Float.ofNat k * h
    scheds := scheds.push (DeviceSchedule.make T0 T1 tol maxDepth t device)
  -- fresh copy of the initial state; ping-pong with one scratch buffer
  let mut cur : T #[n] := torch.add y0 (mul_scalar y0 0.0)
  let mut nxt ← torch.rand #[n] false device
  for k in [:m] do
    let t := t0 + Float.ofNat k * h
    emStepFused drift diffusion treeSeed t h
      (scheds.getD k default) (scheds.getD (k + 1) default)
      cur nxt blackwell stream
    let tmp := cur
    cur := nxt
    nxt := tmp
  let _ ← torch.cuda_synchronize
  pure cur

end Tyr.GPU.Ops.SDEFused

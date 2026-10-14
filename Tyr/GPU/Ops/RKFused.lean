/-
  Tyr/GPU/Ops/RKFused.lean

  IO-layer wrappers and a fused fixed-step Dopri5 integrator over the
  generated RK kernels (`Tyr.GPU.Kernels.RKCombine`), following the
  kernel-op layering used by `Tyr.GPU.Ops.MhaH100`.

  State layout contract: `T #[1, 1, 64, cols]` with `cols % 64 == 0`
  (one 64×64 tile per block, grid x = cols / 64).

  Per step this issues 7 vector-field evaluations, 7 `mul_scalar` products
  (folding the step size into the stage derivatives), 6 fused stage-sum
  launches, and 1 fused combine launch — versus ~60 separate tensor ops
  for the generic solver path (measured ~1.7µs of dispatch glue each).

  Two hazards this module is careful about (both bit us on hardware):
  - Scratch buffers are allocated through `IO` (`Workspace.make`); pure
    factories with identical arguments get common-subexpression-eliminated
    into one shared tensor, which mutating kernel launches then alias.
  - The state is double-buffered: the combine kernel writes `y1` while the
    stage kernels of the same step read `y0`, so the two must never be the
    same tensor.
-/
import Tyr.Torch
import Tyr.GPU.Kernels.RKCombine

/-! # Fused Dopri5 integrator

Preallocated workspace (`Tyr.GPU.Ops.RKFused.Workspace`) and fused Dopri5
step/solve (`Tyr.GPU.Ops.RKFused.dopri5SolveFused`) on GPU tiles.
-/

namespace Tyr.GPU.Ops.RKFused

open torch
open Tyr.GPU.Kernels

/-- Tile-layout state used by the fused integrator. -/
abbrev Tile (cols : UInt64) : Type := T #[1, 1, 64, cols]

/-- Preallocated scratch buffers for one solver instance. -/
structure Workspace (cols : UInt64) where
  z : Array (Tile cols)      -- six stage-state buffers (z₂ … z₇)
  yA : Tile cols             -- ping
  yB : Tile cols             -- pong
  err : Tile cols

/-- Allocate the workspace through IO so every buffer is distinct. -/
def Workspace.make (cols : UInt64) (device : Device) : IO (Workspace cols) := do
  let mut z : Array (Tile cols) := #[]
  for _ in [:6] do
    z := z.push (← torch.rand #[1, 1, 64, cols] false device)
  let yA ← torch.rand #[1, 1, 64, cols] false device
  let yB ← torch.rand #[1, 1, 64, cols] false device
  let err ← torch.rand #[1, 1, 64, cols] false device
  pure { z, yA, yB, err }

private def dopri5C : Array Float :=
  #[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]

/-- One fused Dopri5 step: reads the state from `y0`, writes the new state
    into `y1Out` (and the embedded error estimate into `ws.err`). -/
def dopri5StepFused {cols : UInt64}
    (vf : Float → Tile cols → Tile cols)
    (t h : Float)
    (y0 : Tile cols) (y1Out : Tile cols)
    (ws : Workspace cols)
    (blackwell : Bool) (stream : UInt64) : IO Unit := do
  let gx := cols / 64
  let kOf := fun (i : Nat) (z : Tile cols) =>
    mul_scalar (vf (t + dopri5C.getD i 0.0 * h) z) h
  let k1 := kOf 0 y0
  let z2 := ws.z.getD 0 y0
  if blackwell then
    dopri5Stage2Blackwell.launch y0 k1 z2 gx 1 1 32 1 1 0 stream
  else
    dopri5Stage2.launch y0 k1 z2 gx 1 1 32 1 1 0 stream
  let k2 := kOf 1 z2
  let z3 := ws.z.getD 1 y0
  if blackwell then
    dopri5Stage3Blackwell.launch y0 k1 k2 z3 gx 1 1 32 1 1 0 stream
  else
    dopri5Stage3.launch y0 k1 k2 z3 gx 1 1 32 1 1 0 stream
  let k3 := kOf 2 z3
  let z4 := ws.z.getD 2 y0
  if blackwell then
    dopri5Stage4Blackwell.launch y0 k1 k2 k3 z4 gx 1 1 32 1 1 0 stream
  else
    dopri5Stage4.launch y0 k1 k2 k3 z4 gx 1 1 32 1 1 0 stream
  let k4 := kOf 3 z4
  let z5 := ws.z.getD 3 y0
  if blackwell then
    dopri5Stage5Blackwell.launch y0 k1 k2 k3 k4 z5 gx 1 1 32 1 1 0 stream
  else
    dopri5Stage5.launch y0 k1 k2 k3 k4 z5 gx 1 1 32 1 1 0 stream
  let k5 := kOf 4 z5
  let z6 := ws.z.getD 4 y0
  if blackwell then
    dopri5Stage6Blackwell.launch y0 k1 k2 k3 k4 k5 z6 gx 1 1 32 1 1 0 stream
  else
    dopri5Stage6.launch y0 k1 k2 k3 k4 k5 z6 gx 1 1 32 1 1 0 stream
  let k6 := kOf 5 z6
  let z7 := ws.z.getD 5 y0
  if blackwell then
    dopri5Stage7Blackwell.launch y0 k1 k2 k3 k4 k5 k6 z7 gx 1 1 32 1 1 0 stream
  else
    dopri5Stage7.launch y0 k1 k2 k3 k4 k5 k6 z7 gx 1 1 32 1 1 0 stream
  let k7 := kOf 6 z7
  if blackwell then
    dopri5Combine64Blackwell.launch y0 k1 k2 k3 k4 k5 k6 k7 y1Out ws.err
      gx 1 1 32 1 1 0 stream
  else
    dopri5Combine64.launch y0 k1 k2 k3 k4 k5 k6 k7 y1Out ws.err
      gx 1 1 32 1 1 0 stream

/-- Fused fixed-step Dopri5 solve from `t0` to `t1` in `steps` uniform
    steps. Returns the tensor holding the final state (one of the two
    ping-pong buffers; clone it if you need it to outlive the workspace). -/
def dopri5SolveFused {cols : UInt64}
    (vf : Float → Tile cols → Tile cols)
    (t0 t1 : Float) (steps : Nat)
    (y0 : Tile cols)
    (ws : Workspace cols)
    (blackwell : Bool) (stream : UInt64) : IO (Tile cols) := do
  let n := Nat.max steps 1
  let h := (t1 - t0) / Float.ofNat n
  -- fresh copy of the initial state so the caller's tensor is not mutated;
  -- ping-pong between it and the workspace buffers
  let mut cur : Tile cols := torch.add y0 (mul_scalar y0 0.0)
  let mut nxt := ws.yA
  for k in [:n] do
    let t := t0 + Float.ofNat k * h
    dopri5StepFused vf t h cur nxt ws blackwell stream
    let tmp := cur
    cur := nxt
    nxt := tmp
  let _ ← torch.cuda_synchronize
  pure cur

end Tyr.GPU.Ops.RKFused

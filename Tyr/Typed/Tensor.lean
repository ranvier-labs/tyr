import Tyr.Torch
import Tyr.Typed.Device

/-!
# Typed tensor facade

`Tensor (σ : StaticSpec)` is a lightweight typed view over Tyr's existing
`T shape`, graded by ONE struct index carrying shape, dtype, and device
policy. The index is phantom: a `Tensor σ` contains exactly a `T σ.shape`
handle, so wrappers cost nothing at runtime and the FFI surface is
untouched.

Grading: `dtype` and `device` have "dynamic" defaults, so
`Tensor {shape := s}` claims only the shape, while
`Tensor {shape := s, dtype := .Float32, device := .exact (.CUDA 0)}` pins
everything. `DTensor shape dtype` abbreviates the shape+dtype instantiation.

Checked constructors validate runtime metadata when crossing from raw
tensors into the typed facade; `assumeSpec`/`assumeDType` are available for
code that already established the invariant externally (each such use
encodes an assumption about libtorch behavior — keep them covered by the
parity tests in `Tests/TestTyped.lean`).
-/

namespace torch

/-- Static specification carried by a typed tensor: shape always; dtype and
    device policy graded, with defaults meaning "dynamic" / "anywhere". -/
structure StaticSpec where
  shape : Shape
  dtype : DType := .Unknown "dynamic"
  device : DevicePolicy := .any
deriving Repr, BEq

namespace StaticSpec

/-- Forget the device policy (runtime-metadata view). -/
@[reducible] def toTensorSpec (σ : StaticSpec) : TensorSpec :=
  { shape := σ.shape, dtype := σ.dtype }

@[reducible] def withShape (σ : StaticSpec) (shape : Shape) : StaticSpec :=
  { σ with shape }

@[reducible] def withDType (σ : StaticSpec) (dtype : DType) : StaticSpec :=
  { σ with dtype }

/-- Result spec of a broadcasting pointwise op. The device policy of the
    left operand propagates. -/
@[reducible] def pointwise (l r : StaticSpec) : StaticSpec :=
  { shape := TensorSpec.broadcastShape l.shape r.shape
    dtype := DType.promote l.dtype r.dtype
    device := l.device }

/-- Result spec of true division: pointwise, then floated. -/
@[reducible] def division (l r : StaticSpec) : StaticSpec :=
  { l.pointwise r with dtype := (DType.promote l.dtype r.dtype).atLeastFloat }

/-- Result spec of a full reduction with sum accumulation rules. -/
@[reducible] def sumSpec (σ : StaticSpec) : StaticSpec :=
  { σ with shape := #[], dtype := σ.dtype.sumResult }

end StaticSpec

/-- Typed tensor: phantom `StaticSpec` index over the runtime `T` handle. -/
structure Tensor (σ : StaticSpec) where
  private mk ::
  raw : T σ.shape

/-- Shape+dtype instantiation of the graded tensor (device unconstrained). -/
abbrev DTensor (shape : Shape) (dtype : DType) : Type :=
  Tensor { shape, dtype }

abbrev TT (shape : Shape) (dtype : DType) : Type :=
  DTensor shape dtype

abbrev SomeDTensor :=
  Sigma fun spec : TensorSpec => DTensor spec.shape spec.dtype

namespace Tensor

/-- Wrap a raw tensor, assuming the full static spec holds. -/
def assumeSpec {σ : StaticSpec} (raw : T σ.shape) : Tensor σ :=
  .mk raw

/-- Compat alias for `assumeSpec` (the shape+dtype reading). -/
abbrev assumeDType {σ : StaticSpec} (raw : T σ.shape) : Tensor σ :=
  assumeSpec raw

def toTensor {σ : StaticSpec} (t : Tensor σ) : T σ.shape :=
  t.raw

instance {σ : StaticSpec} : CoeOut (Tensor σ) (T σ.shape) where
  coe := toTensor

/-- The static spec, as runtime metadata (shape + dtype). -/
def spec {σ : StaticSpec} (_t : Tensor σ) : TensorSpec :=
  σ.toTensorSpec

/-- The full static spec, including the device policy. -/
def staticSpec {σ : StaticSpec} (_t : Tensor σ) : StaticSpec :=
  σ

def dtype {σ : StaticSpec} (_t : Tensor σ) : DType :=
  σ.dtype

def shape {σ : StaticSpec} (_t : Tensor σ) : Shape :=
  σ.shape

/-- Runtime metadata actually carried by the handle. -/
def actualSpec {σ : StaticSpec} (t : Tensor σ) : TensorSpec :=
  { shape := t.raw.runtimeShape, dtype := t.raw.dtype }

/-- Validate every phantom claim — shape, dtype, and device policy —
    against the runtime handle. Graded components make no claim at their
    dynamic level: an `Unknown` dtype matches any runtime dtype, and the
    `.any` device policy accepts any placement. Used by tests and debug
    assertions at trust boundaries. -/
def validate {σ : StaticSpec} (t : Tensor σ) : Except String Unit := do
  TensorSpec.checkShape σ.shape t.raw.runtimeShape
  match σ.dtype with
  | .Unknown _ => pure ()
  | d => TensorSpec.checkDType d t.raw.dtype
  σ.device.check t.raw.device

def ofTensor? {shape : Shape} (dtype : DType) (raw : T shape) : Option (DTensor shape dtype) :=
  if raw.runtimeShape == shape && raw.dtype == dtype then
    some (.mk raw)
  else
    none

def ofTensor {shape : Shape} (dtype : DType) (raw : T shape) : Except String (DTensor shape dtype) := do
  TensorSpec.checkShape shape raw.runtimeShape
  TensorSpec.checkDType dtype raw.dtype
  pure (.mk raw)

def ofTensorWithContract {shape : Shape} (contract : TensorContract) (raw : T shape) :
    Except String (DTensor shape contract.spec.dtype) := do
  let actual : TensorSpec := { shape := raw.runtimeShape, dtype := raw.dtype }
  contract.check actual raw.device
  if h : contract.spec.shape = shape then
    pure (h ▸ .mk raw)
  else
    .error s!"Expected static shape {reprStr contract.spec.shape}, got {reprStr shape}"

def toSome {shape : Shape} {dtype : DType} (tensor : DTensor shape dtype) : SomeDTensor :=
  ⟨{ shape, dtype }, tensor⟩

end Tensor

end torch

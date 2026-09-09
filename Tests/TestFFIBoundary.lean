import LeanTest
import Tyr.Torch

open torch

namespace Tests.TestFFIBoundary

private def values {s : Shape} (t : T s) : Array Float :=
  t.getValues |>.toList.toArray

/-- Exercise boxed Lean Int/Nat arguments through the real native bridge. -/
@[test]
def testBoxedIndexAndUnbind : IO Unit := do
  let x := torch.reshape (torch.arange 0 6) #[2, 3]
  LeanTest.assertEqual (values (x.getOp 0)) #[0.0, 1.0, 2.0]
  LeanTest.assertEqual (values (x.getOp 1)) #[3.0, 4.0, 5.0]
  LeanTest.assertEqual (values (x.getOp (-1))) #[3.0, 4.0, 5.0]
  let rows := torch.unbind x 0
  LeanTest.assertEqual rows.size 2
  LeanTest.assertEqual rows[0]!.runtimeShape #[3]
  LeanTest.assertEqual (values rows[1]!) #[3.0, 4.0, 5.0]
  let columns := torch.unbind x 1
  LeanTest.assertEqual columns.size 3
  LeanTest.assertEqual columns[0]!.runtimeShape #[2]
  LeanTest.assertEqual (values columns[2]!) #[2.0, 5.0]

@[test]
def testArangeLengthAndScalarWidths : IO Unit := do
  let cases : Array (UInt64 × UInt64 × UInt64) :=
    #[(0, 5, 2), (0, 6, 2), (2, 2, 1), (0, 1, 4), (4294967296, 4294967299, 1)]
  for (start, stop, step) in cases do
    let x := torch.arange start stop step
    LeanTest.assertEqual x.runtimeShape #[arangeLength start stop step]
  LeanTest.assertEqual (values (torch.arange 0 5 2)) #[0.0, 2.0, 4.0]
  -- getValues is a float32 display helper; item preserves these int64 values.
  let wide := torch.arange 4294967296 4294967299
  LeanTest.assertEqual (nn.item (wide.getOp 0)) 4294967296.0
  LeanTest.assertEqual (nn.item (wide.getOp 1)) 4294967297.0
  LeanTest.assertEqual (nn.item (wide.getOp 2)) 4294967298.0
  LeanTest.assertEqual (torch.eye 2).runtimeShape #[2, 2]
  LeanTest.assertEqual (values (torch.linspace 0.0 1.0 3)) #[0.0, 0.5, 1.0]
  LeanTest.assertEqual (values (torch.logspace 0.0 2.0 3)) #[1.0, 10.0, 100.0]

@[test]
def testExactScalarReshapeAndLegacyErasure : IO Unit := do
  let x := torch.ones #[1]
  LeanTest.assertEqual (torch.reshapeExact x #[]).runtimeShape #[]
  -- Existing raw callers deliberately use an empty target for erasure.
  LeanTest.assertEqual (torch.reshape x #[]).runtimeShape #[1]
  LeanTest.assertEqual (nn.eraseShape x).runtimeShape #[1]

@[test]
def testExactLoadRejectsSameNumelWrongShape : IO Unit :=
  IO.FS.withTempDir fun dir => do
    let path := (dir / "tensor.pt").toString
    data.saveTensor (torch.ones #[2, 3]) path
    let exact ← data.loadTensorExact #[2, 3] path
    LeanTest.assertEqual exact.runtimeShape #[2, 3]
    try
      let _ ← data.loadTensorExact #[3, 2] path
      LeanTest.fail "exact loading must reject a shape mismatch"
    catch error =>
      LeanTest.assertTrue (error.toString.containsSubstr "Tensor shape mismatch")
    let legacy ← data.loadTensor #[3, 2] path
    LeanTest.assertEqual legacy.runtimeShape #[3, 2]

end Tests.TestFFIBoundary

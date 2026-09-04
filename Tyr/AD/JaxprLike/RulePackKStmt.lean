import Std.Data.HashSet
import Tyr.AD.JaxprLike.RuleRegistry
import Tyr.AD.JaxprLike.KStmtNames

/-!
# Tyr.AD.JaxprLike.RulePackKStmt

Built-in local-Jacobian rule-pack registration for `FromKStmt` op names.
-/

namespace Tyr.AD.JaxprLike

open Tyr.GPU.Codegen

private def atomName (value : String) : Lean.Name :=
  Lean.Name.str Lean.Name.anonymous value

private def reduceOpTagName (op : ReduceOp) : Lean.Name :=
  atomName (kstmtReduceOpTag op)

private def reduceAxisTagName (axis : ReduceAxis) : Lean.Name :=
  atomName (kstmtReduceAxisTag axis)

private def broadcastAxisTagName (axis : BroadcastAxis) : Lean.Name :=
  atomName (kstmtBroadcastAxisTag axis)

private def unaryJacTag (op : UnaryOp) (mode : Tyr.AD.Sparse.JacMode := .none) :
    Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.unary (kstmtUnaryOpName op) .x mode)

private def binaryJacTag (op : BinaryOp) (arg : Tyr.AD.Sparse.JacArgRole)
    (mode : Tyr.AD.Sparse.JacMode := .none) :
    Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.binary (kstmtBinaryOpName op) arg mode)

/-- 1x1 sparse map used for scalar local Jacobians in early KStmt rule packs. -/
private def scalarJacMap (tag : Tyr.AD.Sparse.SparseMapTag) (weight : Float) : SparseLinearMap :=
  {
    repr := tag
    inDim? := some 1
    outDim? := some 1
    entries := #[({ src := 0, dst := 0, weight := weight } : Tyr.AD.Sparse.SparseEntry)]
  }

private def shapeFlatDim? (shape? : Option (Array Nat)) : Option Nat :=
  match shape? with
  | some dims =>
    if dims.isEmpty then
      some 1
    else
      some (dims.foldl (init := 1) (· * ·))
  | none => none

private def varFlatDim? (v : JVar) : Option Nat :=
  shapeFlatDim? v.metaInfo.shape

private def diagEntries (n : Nat) (weight : Float) : Array Tyr.AD.Sparse.SparseEntry :=
  (Array.range n).map fun i =>
    ({ src := i, dst := i, weight := weight } : Tyr.AD.Sparse.SparseEntry)

private def shapeAwareJacMap
    (tag : Tyr.AD.Sparse.SparseMapTag)
    (inDim? outDim? : Option Nat)
    (weight : Float) :
    SparseLinearMap :=
  match inDim?, outDim? with
  | some inDim, some outDim =>
    if inDim = 0 || outDim = 0 then
      scalarJacMap tag weight
    else
      {
        repr := tag
        inDim? := some inDim
        outDim? := some outDim
        entries := diagEntries (min inDim outDim) weight
      }
  | _, _ =>
    scalarJacMap tag weight

private def unaryJacMapForVars
    (inv outv : JVar)
    (tag : Tyr.AD.Sparse.SparseMapTag)
    (weight : Float) :
    SparseLinearMap :=
  shapeAwareJacMap tag (varFlatDim? inv) (varFlatDim? outv) weight

private def binaryJacMapForVars
    (inv outv : JVar)
    (tag : Tyr.AD.Sparse.SparseMapTag)
    (weight : Float) :
    SparseLinearMap :=
  shapeAwareJacMap tag (varFlatDim? inv) (varFlatDim? outv) weight

private def shapeDimAt? (v : JVar) (idx : Nat) : Option Nat := do
  let shape ← v.metaInfo.shape
  shape[idx]?

private def shapeRowsCols? (v : JVar) : Option (Nat × Nat) := do
  let r ← shapeDimAt? v 0
  let c ← shapeDimAt? v 1
  some (r, c)

private def buildSparseMap
    (tag : Tyr.AD.Sparse.SparseMapTag)
    (inDim outDim : Nat)
    (entries : Array Tyr.AD.Sparse.SparseEntry) :
    SparseLinearMap :=
  {
    repr := tag
    inDim? := some inDim
    outDim? := some outDim
    entries := entries
  }

private def transposeEntries (rows cols : Nat) : Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:cols] do
      entries := entries.push {
        src := i * cols + j
        dst := j * rows + i
        weight := 1.0
      }
  return entries

private def sliceRowsEntries
    (rows cols startRow outRows outCols : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:outRows] do
    for j in [:outCols] do
      let srcRow := startRow + i
      if srcRow < rows && j < cols then
        entries := entries.push {
          src := srcRow * cols + j
          dst := i * outCols + j
          weight := 1.0
        }
  return entries

private def sliceColsEntries
    (rows cols startCol outRows outCols : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:outRows] do
    for j in [:outCols] do
      let srcCol := startCol + j
      if i < rows && srcCol < cols then
        entries := entries.push {
          src := i * cols + srcCol
          dst := i * outCols + j
          weight := 1.0
        }
  return entries

private def concatColsEntriesLhs
    (rows lhsCols outCols : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:lhsCols] do
      entries := entries.push {
        src := i * lhsCols + j
        dst := i * outCols + j
        weight := 1.0
      }
  return entries

private def concatColsEntriesRhs
    (rows rhsCols outCols offset : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:rhsCols] do
      let dstCol := offset + j
      if dstCol < outCols then
        entries := entries.push {
          src := i * rhsCols + j
          dst := i * outCols + dstCol
          weight := 1.0
        }
  return entries

private def broadcastEntries
    (axis : BroadcastAxis)
    (inDim outRows outCols : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  if inDim = 0 then
    return entries
  for i in [:outRows] do
    for j in [:outCols] do
      let srcIdx :=
        match axis with
        | .Row =>
          if inDim = outCols then j
          else if inDim = outRows then i
          else j % inDim
        | .Col =>
          if inDim = outRows then i
          else if inDim = outCols then j
          else i % inDim
      entries := entries.push {
        src := srcIdx
        dst := i * outCols + j
        weight := 1.0
      }
  return entries

private def reduceSumEntries
    (axis : ReduceAxis)
    (rows cols outDim : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:cols] do
      let dst :=
        match axis with
        | .Row =>
          if outDim = cols then j else j % outDim
        | .Col =>
          if outDim = rows then i else i % outDim
        | .Full => 0
      entries := entries.push {
        src := i * cols + j
        dst := dst
        weight := 1.0
      }
  return entries

private def cumsumRowEntries (rows cols : Nat) : Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:cols] do
      for k in [0:i+1] do
        entries := entries.push {
          src := k * cols + j
          dst := i * cols + j
          weight := 1.0
        }
  return entries

private def cumsumColEntries (rows cols : Nat) : Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:cols] do
      for k in [0:j+1] do
        entries := entries.push {
          src := i * cols + k
          dst := i * cols + j
          weight := 1.0
        }
  return entries

private def cumsumFullEntries (inDim outDim : Nat) : Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  let n := min inDim outDim
  for dst in [:n] do
    for src in [0:dst+1] do
      entries := entries.push { src := src, dst := dst, weight := 1.0 }
  return entries

private def fillEntries (outDim : Nat) : Array Tyr.AD.Sparse.SparseEntry :=
  (Array.range outDim).map fun dst =>
    ({ src := 0, dst := dst, weight := 1.0 } : Tyr.AD.Sparse.SparseEntry)

private def sliceVectorEntries
    (inDim start size : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:size] do
    let src := start + i
    if src < inDim then
      entries := entries.push {
        src := src
        dst := i
        weight := 1.0
      }
  return entries

private def concatRowsEntries
    (rows cols outCols rowOffset : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:cols] do
      entries := entries.push {
        src := i * cols + j
        dst := (rowOffset + i) * outCols + j
        weight := 1.0
      }
  return entries

private def concatVectorEntries
    (len offset : Nat) :
    Array Tyr.AD.Sparse.SparseEntry :=
  (Array.range len).map fun i =>
    ({ src := i, dst := offset + i, weight := 1.0 } : Tyr.AD.Sparse.SparseEntry)

private def concatColsEntries
    (rows cols outCols colOffset : Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for i in [:rows] do
    for j in [:cols] do
      entries := entries.push {
        src := i * cols + j
        dst := i * outCols + (colOffset + j)
        weight := 1.0
      }
  return entries

private def exactUnaryMapOrFallback
    (inv outv : JVar)
    (tag : Tyr.AD.Sparse.SparseMapTag)
    (entries? : Option (Array Tyr.AD.Sparse.SparseEntry)) :
    SparseLinearMap :=
  match entries?, varFlatDim? inv, varFlatDim? outv with
  | some entries, some inDim, some outDim =>
    buildSparseMap tag inDim outDim entries
  | _, _, _ =>
    unaryJacMapForVars inv outv tag 1.0

private def exactBinaryMapOrFallback
    (inv outv : JVar)
    (tag : Tyr.AD.Sparse.SparseMapTag)
    (entries? : Option (Array Tyr.AD.Sparse.SparseEntry)) :
    SparseLinearMap :=
  match entries?, varFlatDim? inv, varFlatDim? outv with
  | some entries, some inDim, some outDim =>
    buildSparseMap tag inDim outDim entries
  | _, _, _ =>
    binaryJacMapForVars inv outv tag 1.0

private def transposeAliasEntries? (inv outv : JVar) : Option (Array Tyr.AD.Sparse.SparseEntry) := do
  let (rows, cols) ← shapeRowsCols? inv
  let (outRows, outCols) ← shapeRowsCols? outv
  let inDim ← varFlatDim? inv
  let outDim ← varFlatDim? outv
  if
      inDim = 0 || outDim = 0 ||
      inDim != outDim ||
      inDim != rows * cols ||
      outRows != cols ||
      outCols != rows then
    none
  else
    some (transposeEntries rows cols)

private def identityAliasEntries? (inv outv : JVar) : Option (Array Tyr.AD.Sparse.SparseEntry) := do
  let inDim ← varFlatDim? inv
  let outDim ← varFlatDim? outv
  if inDim = 0 || outDim = 0 || inDim != outDim then
    none
  else
    some (diagEntries inDim 1.0)

private def rowMajorStrides (shape : Array Nat) : Array Nat := Id.run do
  let mut strides := Array.replicate shape.size 1
  let mut stride := 1
  for i in (List.range shape.size).reverse do
    strides := strides.set! i stride
    stride := stride * shape.getD i 1
  return strides

private def flattenCoord (strides coord : Array Nat) : Nat := Id.run do
  let mut flat := 0
  for i in [:coord.size] do
    flat := flat + coord[i]! * strides.getD i 1
  return flat

private def unflattenIndex (shape strides : Array Nat) (flat : Nat) : Array Nat := Id.run do
  let mut rem := flat
  let mut coord := Array.replicate shape.size 0
  for i in [:shape.size] do
    let stride := strides.getD i 1
    let extent := shape.getD i 1
    let value :=
      if extent = 0 || stride = 0 then
        0
      else
        rem / stride
    coord := coord.set! i value
    if extent != 0 && stride != 0 then
      rem := rem % stride
  return coord

private def padConfig? (eqn : JEqn) : Option (Array Nat × Array Nat × Array Nat) := do
  let low ← eqn.params.findNats? .padLow
  let high ← eqn.params.findNats? .padHigh
  let interior ← eqn.params.findNats? .padInterior
  if low.size = high.size && high.size = interior.size then
    some (low, high, interior)
  else
    none

private def expectedPaddedExtent (input low high interior : Nat) : Nat :=
  if input = 0 then
    low + high
  else
    low + input + high + interior * (input - 1)

private def padBaseEntries
    (inShape outShape low interior : Array Nat) :
    Array Tyr.AD.Sparse.SparseEntry := Id.run do
  let inDim := shapeFlatDim? (some inShape) |>.getD 0
  let inStrides := rowMajorStrides inShape
  let outStrides := rowMajorStrides outShape
  let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
  for src in [:inDim] do
    let inCoord := unflattenIndex inShape inStrides src
    let mut outCoord := Array.replicate inCoord.size 0
    for i in [:inCoord.size] do
      outCoord := outCoord.set! i (low.getD i 0 + inCoord[i]! * (interior.getD i 0 + 1))
    entries := entries.push {
      src := src
      dst := flattenCoord outStrides outCoord
      weight := 1.0
    }
  return entries

private def padBaseEntries? (eqn : JEqn) (base outv : JVar) :
    Option (Array Tyr.AD.Sparse.SparseEntry) := do
  let inShape ← base.metaInfo.shape
  let outShape ← outv.metaInfo.shape
  let (low, high, interior) ← padConfig? eqn
  let inDim ← varFlatDim? base
  let outDim ← varFlatDim? outv
  if
      inDim = 0 || outDim = 0 ||
      inShape.size != outShape.size ||
      low.size != inShape.size ||
      high.size != inShape.size ||
      interior.size != inShape.size then
    none
  else
    let shapeMatches :=
      (List.range inShape.size).all fun i =>
        outShape.getD i 0 = expectedPaddedExtent (inShape.getD i 0) (low.getD i 0) (high.getD i 0) (interior.getD i 0)
    if shapeMatches then
      some (padBaseEntries inShape outShape low interior)
    else
      none

private def padValueEntries?
    (padv outv : JVar)
    (baseEntries : Array Tyr.AD.Sparse.SparseEntry) :
    Option (Array Tyr.AD.Sparse.SparseEntry) := do
  let padDim ← varFlatDim? padv
  let outDim ← varFlatDim? outv
  if padDim != 1 || outDim = 0 then
    none
  else
    let occupied : Std.HashSet Nat :=
      baseEntries.foldl (init := {}) fun acc entry => acc.insert entry.dst
    let mut entries : Array Tyr.AD.Sparse.SparseEntry := #[]
    for dst in [:outDim] do
      if !occupied.contains dst then
        entries := entries.push { src := 0, dst := dst, weight := 1.0 }
    some entries

private def broadcastAliasEntries? (inv outv : JVar) : Option (Array Tyr.AD.Sparse.SparseEntry) := do
  let inShape ← inv.metaInfo.shape
  let outShape ← outv.metaInfo.shape
  let inDim ← varFlatDim? inv
  let outDim ← varFlatDim? outv
  if inDim = 0 || outDim = 0 then
    none
  else if outShape.size = 1 then
    let outLen := outShape.getD 0 1
    if inDim = outLen then
      some (diagEntries inDim 1.0)
    else if inDim = 1 then
      some (fillEntries outDim)
    else
      none
  else if outShape.size >= 2 then
    let outRows := outShape.getD 0 1
    let outCols := outShape.getD 1 1
    if inShape.size >= 2 && inShape.getD 0 1 = outRows && inShape.getD 1 1 = outCols then
      some (diagEntries inDim 1.0)
    else if inDim = 1 then
      some (fillEntries outDim)
    else if inDim = outCols then
      some (broadcastEntries .Row inDim outRows outCols)
    else if inDim = outRows then
      some (broadcastEntries .Col inDim outRows outCols)
    else if
        inShape.size >= 2 &&
        inShape.getD 0 1 = 1 &&
        inShape.getD 1 1 = outCols then
      some (broadcastEntries .Row inDim outRows outCols)
    else if
        inShape.size >= 2 &&
        inShape.getD 1 1 = 1 &&
        inShape.getD 0 1 = outRows then
      some (broadcastEntries .Col inDim outRows outCols)
    else
      none
  else
    none

private def sliceAliasEntries? (eqn : JEqn) (inv outv : JVar) :
    Option (Array Tyr.AD.Sparse.SparseEntry) := do
  let inDim ← varFlatDim? inv
  let outDim ← varFlatDim? outv
  let inShape ← inv.metaInfo.shape
  let outShape ← outv.metaInfo.shape
  match eqn.params.findNat? .startRow, eqn.params.findNat? .numRows,
      eqn.params.findNat? .startCol, eqn.params.findNat? .numCols with
  | some startRow, some numRows, _, _ =>
    if inShape.size >= 2 && outShape.size >= 2 then
      let rows := inShape.getD 0 1
      let cols := inShape.getD 1 1
      let outRows := outShape.getD 0 1
      let outCols := outShape.getD 1 1
      if outRows = numRows && outCols = cols then
        some (sliceRowsEntries rows cols startRow outRows outCols)
      else
        none
    else if outDim = numRows then
      some (sliceVectorEntries inDim startRow outDim)
    else
      none
  | _, _, some startCol, some numCols =>
    if inShape.size >= 2 && outShape.size >= 2 then
      let rows := inShape.getD 0 1
      let cols := inShape.getD 1 1
      let outRows := outShape.getD 0 1
      let outCols := outShape.getD 1 1
      if outCols = numCols && outRows = rows then
        some (sliceColsEntries rows cols startCol outRows outCols)
      else
        none
    else if outDim = numCols then
      some (sliceVectorEntries inDim startCol outDim)
    else
      none
  | _, _, _, _ => none

private def concatAliasEntries?
    (invars : Array JVar)
    (outv : JVar) :
    Option (Array (Array Tyr.AD.Sparse.SparseEntry)) := do
  let outShape ← outv.metaInfo.shape
  let outDim ← varFlatDim? outv
  let inShapes? := invars.map (·.metaInfo.shape)
  let inDims? := invars.map varFlatDim?
  if inShapes?.any Option.isNone || inDims?.any Option.isNone then
    none
  else
    let inShapes := inShapes?.map Option.get!
    let inDims := inDims?.map Option.get!
    if outShape.size = 1 && inShapes.all (fun s => s.size = 1) then
      let total := inDims.foldl (init := 0) (· + ·)
      if total != outDim then
        none
      else
        let init : Nat × Array (Array Tyr.AD.Sparse.SparseEntry) := (0, #[])
        let (_, maps) :=
          inDims.foldl
            (init := init)
            fun (state : Nat × Array (Array Tyr.AD.Sparse.SparseEntry)) len =>
              let offset := state.1
              let entries := concatVectorEntries len offset
              (offset + len, state.2.push entries)
        some maps
    else if outShape.size >= 2 && inShapes.all (fun s => s.size >= 2) then
      let outRows := outShape.getD 0 1
      let outCols := outShape.getD 1 1
      let sameRows := inShapes.all (fun s => s.getD 0 1 = outRows)
      let sameCols := inShapes.all (fun s => s.getD 1 1 = outCols)
      let totalCols := inShapes.foldl (init := 0) fun acc s => acc + s.getD 1 1
      let totalRows := inShapes.foldl (init := 0) fun acc s => acc + s.getD 0 1
      if sameRows && totalCols = outCols then
        let init : Nat × Array (Array Tyr.AD.Sparse.SparseEntry) := (0, #[])
        let (_, maps) :=
          inShapes.foldl
            (init := init)
            fun (state : Nat × Array (Array Tyr.AD.Sparse.SparseEntry)) shape =>
              let rows := shape.getD 0 1
              let cols := shape.getD 1 1
              let offset := state.1
              (offset + cols, state.2.push (concatColsEntries rows cols outCols offset))
        some maps
      else if sameCols && totalRows = outRows then
        let init : Nat × Array (Array Tyr.AD.Sparse.SparseEntry) := (0, #[])
        let (_, maps) :=
          inShapes.foldl
            (init := init)
            fun (state : Nat × Array (Array Tyr.AD.Sparse.SparseEntry)) shape =>
              let rows := shape.getD 0 1
              let cols := shape.getD 1 1
              let offset := state.1
              (offset + rows, state.2.push (concatRowsEntries rows cols outCols offset))
        some maps
      else
        none
    else
      none

/--
If an op name is `jax.lax.<prim>`, return the corresponding
`jax.lax.<prim>_p` alias used in some Graphax/AlphaGrad source paths.
-/
private def jaxLaxPrimAlias? (op : OpName) : Option OpName :=
  match op with
  | .str (.str (.str .anonymous "jax") "lax") base =>
    some (Lean.Name.str (Lean.Name.str (Lean.Name.str .anonymous "jax") "lax") s!"{base}_p")
  | _ => none

private def registerAliasWithLaxPrimVariant (op : OpName) (rule : LocalJacRule) :
    Lean.CoreM Unit := do
  registerLocalJacRule op rule
  match jaxLaxPrimAlias? op with
  | some prim => registerLocalJacRule prim rule
  | none => pure ()

private def requireUnaryShape (op : OpName) (eqn : JEqn) : Except RuleError (JVar × JVar) := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Unary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Unary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else
    let inv ←
      match eqn.invars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Unary rule `{op}` requires one input variable.")
    if eqn.invars.size != 1 then
      .error (.malformedEqn s!"Unary rule `{op}` expects exactly one input variable, got {eqn.invars.size}.")
    else
      .ok (inv, outv)

private def requireBinaryShape (op : OpName) (eqn : JEqn) : Except RuleError (JVar × JVar × JVar) := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Binary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Binary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else
    let a ←
      match eqn.invars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Binary rule `{op}` requires two input variables.")
    let b ←
      match eqn.invars[1]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Binary rule `{op}` requires two input variables.")
    if eqn.invars.size != 2 then
      .error (.malformedEqn s!"Binary rule `{op}` expects exactly two input variables, got {eqn.invars.size}.")
    else
      .ok (a, b, outv)

private def requireTernaryShape (op : OpName) (eqn : JEqn) : Except RuleError (JVar × JVar × JVar × JVar) := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Ternary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Ternary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else
    let a ←
      match eqn.invars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Ternary rule `{op}` requires three input variables.")
    let b ←
      match eqn.invars[1]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Ternary rule `{op}` requires three input variables.")
    let c ←
      match eqn.invars[2]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Ternary rule `{op}` requires three input variables.")
    if eqn.invars.size != 3 then
      .error (.malformedEqn s!"Ternary rule `{op}` expects exactly three input variables, got {eqn.invars.size}.")
    else
      .ok (a, b, c, outv)

private def requireNullaryShape (op : OpName) (eqn : JEqn) : Except RuleError JVar := do
  let outv ←
    match eqn.outvars[0]? with
    | some v => .ok v
    | none => .error (.malformedEqn s!"Nullary rule `{op}` requires one output variable.")
  if eqn.outvars.size != 1 then
    .error (.malformedEqn s!"Nullary rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
  else if eqn.invars.size != 0 then
    .error (.malformedEqn s!"Nullary rule `{op}` expects zero input variables, got {eqn.invars.size}.")
  else
    .ok outv

private def unaryConstJac? (op : UnaryOp) : Option (Tyr.AD.Sparse.SparseMapTag × Float) :=
  match op with
  | .Copy => some (unaryJacTag .Copy, 1.0)
  | .Neg => some (unaryJacTag .Neg, -1.0)
  | .Zero => some (unaryJacTag .Zero, 0.0)
  | .One => some (unaryJacTag .One, 0.0)
  | .PosInfty => some (unaryJacTag .PosInfty, 0.0)
  | .NegInfty => some (unaryJacTag .NegInfty, 0.0)
  | _ => none

private def unarySymbolicJac? (op : UnaryOp) : Option (Tyr.AD.Sparse.SparseMapTag × Float) :=
  match op with
  | .Exp => some (unaryJacTag .Exp, 1.0)
  | .Exp2 => some (unaryJacTag .Exp2, 1.0)
  | .Log => some (unaryJacTag .Log, 1.0)
  | .Log2 => some (unaryJacTag .Log2, 1.0)
  | .Abs => some (unaryJacTag .Abs, 1.0)
  | .Relu => some (unaryJacTag .Relu, 1.0)
  | .Sqrt => some (unaryJacTag .Sqrt, 1.0)
  | .Rsqrt => some (unaryJacTag .Rsqrt, 1.0)
  | .Tanh => some (unaryJacTag .Tanh, 1.0)
  | .FastTanh => some (unaryJacTag .FastTanh, 1.0)
  | .Sigmoid => some (unaryJacTag .Sigmoid, 1.0)
  | .Gelu => some (unaryJacTag .Gelu, 1.0)
  | .Silu => some (unaryJacTag .Silu, 1.0)
  | .Swish => some (unaryJacTag .Swish, 1.0)
  | .Sin => some (unaryJacTag .Sin, 1.0)
  | .Cos => some (unaryJacTag .Cos, 1.0)
  | .Recip => some (unaryJacTag .Recip, 1.0)
  | .Square => some (unaryJacTag .Square, 1.0)
  | _ => none

private def binaryConstJac? (op : BinaryOp) :
    Option ((Tyr.AD.Sparse.SparseMapTag × Float) × (Tyr.AD.Sparse.SparseMapTag × Float)) :=
  match op with
  | .Add => some ((binaryJacTag .Add .a, 1.0), (binaryJacTag .Add .b, 1.0))
  | .Sub => some ((binaryJacTag .Sub .a, 1.0), (binaryJacTag .Sub .b, -1.0))
  | _ => none

private def binarySymbolicJac? (op : BinaryOp) :
    Option ((Tyr.AD.Sparse.SparseMapTag × Float) × (Tyr.AD.Sparse.SparseMapTag × Float)) :=
  match op with
  | .Mul =>
    some
      ((binaryJacTag .Mul .a .rhsValue, 1.0), (binaryJacTag .Mul .b .lhsValue, 1.0))
  | .Div =>
    some
      ( (binaryJacTag .Div .a .reciprocalRhs, 1.0),
        (binaryJacTag .Div .b .negLhsOverRhsSq, -1.0) )
  | .Max =>
    some
      ((binaryJacTag .Max .a .mask, 1.0), (binaryJacTag .Max .b .complementMask, 1.0))
  | .Min =>
    some
      ((binaryJacTag .Min .a .mask, 1.0), (binaryJacTag .Min .b .complementMask, 1.0))
  | _ => none

private def unaryConstRule
    (op : UnaryOp)
    (jacTag : Tyr.AD.Sparse.SparseMapTag)
    (jacWeight : Float) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape (kstmtUnaryOpName op) eqn
    .ok #[{
      src := inv.id
      dst := outv.id
      map := unaryJacMapForVars inv outv jacTag jacWeight
    }]

private def binaryConstRule
    (op : BinaryOp)
    (left : Tyr.AD.Sparse.SparseMapTag × Float)
    (right : Tyr.AD.Sparse.SparseMapTag × Float) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (a, b, outv) ← requireBinaryShape (kstmtBinaryOpName op) eqn
    .ok #[
      { src := a.id, dst := outv.id, map := binaryJacMapForVars a outv left.1 left.2 },
      { src := b.id, dst := outv.id, map := binaryJacMapForVars b outv right.1 right.2 }
    ]

private def unarySemanticsRule? (op : UnaryOp) : Option LocalJacRule :=
  match unaryConstJac? op with
  | some (repr, weight) => some (unaryConstRule op repr weight)
  | none =>
    match unarySymbolicJac? op with
    | some (repr, weight) => some (unaryConstRule op repr weight)
    | none => none

private def binarySemanticsRule? (op : BinaryOp) : Option LocalJacRule :=
  match binaryConstJac? op with
  | some (left, right) => some (binaryConstRule op left right)
  | none =>
    match binarySymbolicJac? op with
    | some (left, right) => some (binaryConstRule op left right)
    | none => none

private def unarySymbolicRule
    (opName : OpName)
    (jacTag : Tyr.AD.Sparse.SparseMapTag)
    (jacWeight : Float := 1.0) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape opName eqn
    .ok #[{
      src := inv.id
      dst := outv.id
      map := unaryJacMapForVars inv outv jacTag jacWeight
    }]

private def binarySymbolicRule
    (opName : OpName)
    (left : Tyr.AD.Sparse.SparseMapTag × Float)
    (right : Tyr.AD.Sparse.SparseMapTag × Float) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (a, b, outv) ← requireBinaryShape opName eqn
    .ok #[
      { src := a.id, dst := outv.id, map := binaryJacMapForVars a outv left.1 left.2 },
      { src := b.id, dst := outv.id, map := binaryJacMapForVars b outv right.1 right.2 }
    ]

private def reduceJacTag (op : ReduceOp) (axis : ReduceAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.reduce (reduceOpTagName op) (reduceAxisTagName axis) .x .contract)

private def reduceAccumSrcJacTag (op : ReduceOp) (axis : ReduceAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.reduceAccum (reduceOpTagName op) (reduceAxisTagName axis) .src .contract)

private def reduceAccumAccumJacTag (op : ReduceOp) (axis : ReduceAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.reduceAccum (reduceOpTagName op) (reduceAxisTagName axis) .accum .carry)

private def broadcastJacTag (axis : BroadcastAxis) : Tyr.AD.Sparse.SparseMapTag :=
  .semantic (.broadcast (broadcastAxisTagName axis) .x .expand)

private def binaryBroadcastJacPair
    (op : BinaryOp)
    (axis : BroadcastAxis) :
    ((Tyr.AD.Sparse.SparseMapTag × Float) × (Tyr.AD.Sparse.SparseMapTag × Float)) :=
  let axisTagName := broadcastAxisTagName axis
  let defaultPair :=
    ( (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .tile .tileBroadcast), 1.0),
      (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .vec .vecBroadcast), 1.0) )
  match binaryConstJac? op with
  | some (left, right) =>
    ( (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .tile .tileBroadcast), left.2),
      (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .vec .vecBroadcast), right.2) )
  | none =>
    match binarySymbolicJac? op with
    | some (left, right) =>
      ( (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .tile .tileBroadcast), left.2),
        (.semantic (.binaryBroadcast (kstmtBinaryOpName op) axisTagName .vec .vecBroadcast), right.2) )
    | none =>
      defaultPair

private def reduceStructuralRule (op : ReduceOp) (axis : ReduceAxis) : LocalJacRule :=
  fun eqn _ctx => do
    let opName := kstmtReduceOpName op axis
    let (inv, outv) ← requireUnaryShape opName eqn
    let tag := reduceJacTag op axis
    let map :=
      match op with
      | .Sum =>
        match shapeRowsCols? inv, varFlatDim? inv, varFlatDim? outv with
        | some (rows, cols), some inDim, some outDim =>
          if inDim = 0 || outDim = 0 then
            unaryJacMapForVars inv outv tag 1.0
          else
            buildSparseMap tag inDim outDim (reduceSumEntries axis rows cols outDim)
        | _, _, _ =>
          unaryJacMapForVars inv outv tag 1.0
      | _ =>
        unaryJacMapForVars inv outv tag 1.0
    .ok #[{ src := inv.id, dst := outv.id, map := map }]

private def reduceAccumStructuralRule (op : ReduceOp) (axis : ReduceAxis) : LocalJacRule :=
  fun eqn _ctx => do
    let opName := kstmtReduceAccumOpName op axis
    let (srcv, accv, outv) ← requireBinaryShape opName eqn
    let srcTag := reduceAccumSrcJacTag op axis
    let accTag := reduceAccumAccumJacTag op axis
    let srcMap :=
      match op with
      | .Sum =>
        match shapeRowsCols? srcv, varFlatDim? srcv, varFlatDim? outv with
        | some (rows, cols), some inDim, some outDim =>
          if inDim = 0 || outDim = 0 then
            binaryJacMapForVars srcv outv srcTag 1.0
          else
            buildSparseMap srcTag inDim outDim (reduceSumEntries axis rows cols outDim)
        | _, _, _ =>
          binaryJacMapForVars srcv outv srcTag 1.0
      | _ =>
        binaryJacMapForVars srcv outv srcTag 1.0
    let accMap := binaryJacMapForVars accv outv accTag 1.0
    .ok #[
      { src := srcv.id, dst := outv.id, map := srcMap },
      { src := accv.id, dst := outv.id, map := accMap }
    ]

private def broadcastStructuralRule (axis : BroadcastAxis) : LocalJacRule :=
  fun eqn _ctx => do
    let opName := kstmtBroadcastOpName axis
    let (inv, outv) ← requireUnaryShape opName eqn
    let tag := broadcastJacTag axis
    let map :=
      match shapeRowsCols? outv, varFlatDim? inv, varFlatDim? outv with
      | some (outRows, outCols), some inDim, some outDim =>
        if inDim = 0 || outDim = 0 then
          unaryJacMapForVars inv outv tag 1.0
        else
          buildSparseMap tag inDim outDim (broadcastEntries axis inDim outRows outCols)
      | _, _, _ =>
        unaryJacMapForVars inv outv tag 1.0
    .ok #[{ src := inv.id, dst := outv.id, map := map }]

private def binaryBroadcastStructuralRule (op : BinaryOp) (axis : BroadcastAxis) : LocalJacRule :=
  fun eqn _ctx => do
    let opName := kstmtBinaryBroadcastOpName op axis
    let (tile, vec, outv) ← requireBinaryShape opName eqn
    let pair := binaryBroadcastJacPair op axis
    let tileTag := pair.1.1
    let tileWeight := pair.1.2
    let vecTag := pair.2.1
    let vecWeight := pair.2.2

    let tileMap :=
      match varFlatDim? tile, varFlatDim? outv with
      | some inDim, some outDim =>
        if inDim = 0 || outDim = 0 || inDim != outDim then
          binaryJacMapForVars tile outv tileTag tileWeight
        else
          buildSparseMap tileTag inDim outDim (diagEntries inDim tileWeight)
      | _, _ =>
        binaryJacMapForVars tile outv tileTag tileWeight

    let vecMap :=
      match shapeRowsCols? outv, varFlatDim? vec, varFlatDim? outv with
      | some (outRows, outCols), some inDim, some outDim =>
        if inDim = 0 || outDim = 0 || outDim != outRows * outCols then
          binaryJacMapForVars vec outv vecTag vecWeight
        else
          let entries :=
            (broadcastEntries axis inDim outRows outCols).map fun e =>
              ({ src := e.src, dst := e.dst, weight := e.weight * vecWeight } : Tyr.AD.Sparse.SparseEntry)
          buildSparseMap vecTag inDim outDim entries
      | _, _, _ =>
        binaryJacMapForVars vec outv vecTag vecWeight

    .ok #[
      { src := tile.id, dst := outv.id, map := tileMap },
      { src := vec.id, dst := outv.id, map := vecMap }
    ]

private def transposeStructuralRule : LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape kstmtTransposeOpName eqn
    let tag : Tyr.AD.Sparse.SparseMapTag := .semantic (.transpose .x .permute)
    let map :=
      match shapeRowsCols? inv, varFlatDim? inv, varFlatDim? outv with
      | some (rows, cols), some inDim, some outDim =>
        if inDim = 0 || outDim = 0 || inDim != outDim || inDim != rows * cols then
          unaryJacMapForVars inv outv tag 1.0
        else
          buildSparseMap tag inDim outDim (transposeEntries rows cols)
      | _, _, _ =>
        unaryJacMapForVars inv outv tag 1.0
    .ok #[{ src := inv.id, dst := outv.id, map := map }]

private def swapLayoutStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtSwapLayoutOpName (.semantic (.swapLayout .x .layoutPermute))

private def convertStructuralRule : LocalJacRule :=
  unarySymbolicRule kstmtConvertOpName (.semantic (.convert .x .cast))

private def sliceRowsStructuralRule : LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape kstmtSliceRowsOpName eqn
    let tag : Tyr.AD.Sparse.SparseMapTag := .semantic (.sliceRows .x .projection)
    let map :=
      match shapeRowsCols? inv, shapeRowsCols? outv, varFlatDim? inv, varFlatDim? outv with
      | some (rows, cols), some (outRows, outCols), some inDim, some outDim =>
        let startRow := (eqn.params.findNat? .startRow).getD 0
        buildSparseMap tag inDim outDim (sliceRowsEntries rows cols startRow outRows outCols)
      | _, _, _, _ =>
        unaryJacMapForVars inv outv tag 1.0
    .ok #[{ src := inv.id, dst := outv.id, map := map }]

private def sliceColsStructuralRule : LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape kstmtSliceColsOpName eqn
    let tag : Tyr.AD.Sparse.SparseMapTag := .semantic (.sliceCols .x .projection)
    let map :=
      match shapeRowsCols? inv, shapeRowsCols? outv, varFlatDim? inv, varFlatDim? outv with
      | some (rows, cols), some (outRows, outCols), some inDim, some outDim =>
        let startCol := (eqn.params.findNat? .startCol).getD 0
        buildSparseMap tag inDim outDim (sliceColsEntries rows cols startCol outRows outCols)
      | _, _, _, _ =>
        unaryJacMapForVars inv outv tag 1.0
    .ok #[{ src := inv.id, dst := outv.id, map := map }]

private def concatColsStructuralRule : LocalJacRule :=
  fun eqn _ctx => do
    let (lhs, rhs, outv) ← requireBinaryShape kstmtConcatColsOpName eqn
    let lhsTag : Tyr.AD.Sparse.SparseMapTag := .semantic (.concatCols .lhs .inject)
    let rhsTag : Tyr.AD.Sparse.SparseMapTag := .semantic (.concatCols .rhs .inject)
    let lhsMap :=
      match shapeRowsCols? lhs, shapeRowsCols? outv, varFlatDim? lhs, varFlatDim? outv with
      | some (rows, lhsCols), some (outRows, outCols), some inDim, some outDim =>
        if rows != outRows then
          binaryJacMapForVars lhs outv lhsTag 1.0
        else
          buildSparseMap lhsTag inDim outDim (concatColsEntriesLhs rows lhsCols outCols)
      | _, _, _, _ =>
        binaryJacMapForVars lhs outv lhsTag 1.0
    let rhsMap :=
      match shapeRowsCols? lhs, shapeRowsCols? rhs, shapeRowsCols? outv,
          varFlatDim? rhs, varFlatDim? outv with
      | some (_rows, lhsCols), some (rhsRows, rhsCols), some (outRows, outCols), some inDim, some outDim =>
        if rhsRows != outRows then
          binaryJacMapForVars rhs outv rhsTag 1.0
        else
          buildSparseMap rhsTag inDim outDim
            (concatColsEntriesRhs rhsRows rhsCols outCols lhsCols)
      | _, _, _, _, _ =>
        binaryJacMapForVars rhs outv rhsTag 1.0
    .ok #[
      { src := lhs.id, dst := outv.id, map := lhsMap },
      { src := rhs.id, dst := outv.id, map := rhsMap }
    ]

private def outerStructuralRule : LocalJacRule :=
  binarySymbolicRule
    kstmtOuterOpName
    (.semantic (.outer .a .kronOther), 1.0)
    (.semantic (.outer .b .kronOther), 1.0)

private def dotGeneralRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (a, b, outv) ← requireBinaryShape opName eqn
    let lhsContract := (eqn.params.findNats? .lhsContract).getD #[]
    let rhsContract := (eqn.params.findNats? .rhsContract).getD #[]
    let lhsBatch := (eqn.params.findNats? .lhsBatch).getD #[]
    let rhsBatch := (eqn.params.findNats? .rhsBatch).getD #[]
    let variant :=
      match eqn.params.findName? .variant with
      | some variant => variant
      | none =>
        match eqn.params.findName? .sourceOp with
        | some sourceOp => sourceOp
        | none => atomName "generic"
    .ok #[
      { src := a.id, dst := outv.id, map := binaryJacMapForVars a outv (.semantic (.dotGeneral {
          variant := variant
          arg := .lhs
          lhsContract := lhsContract
          rhsContract := rhsContract
          lhsBatch := lhsBatch
          rhsBatch := rhsBatch
        })) 1.0 },
      { src := b.id, dst := outv.id, map := binaryJacMapForVars b outv (.semantic (.dotGeneral {
          variant := variant
          arg := .rhs
          lhsContract := lhsContract
          rhsContract := rhsContract
          lhsBatch := lhsBatch
          rhsBatch := rhsBatch
        })) 1.0 }
    ]

private def mmaRule (trans : MMATranspose) : LocalJacRule :=
  fun eqn _ctx => do
    let opName := kstmtMmaOpName trans
    let (a, b, c, outv) ← requireTernaryShape opName eqn
    .ok #[
      {
        src := a.id
        dst := outv.id
        map := binaryJacMapForVars a outv (.semantic (.binary opName .a .rhsValue)) 1.0
      },
      {
        src := b.id
        dst := outv.id
        map := binaryJacMapForVars b outv (.semantic (.binary opName .b .lhsValue)) 1.0
      },
      {
        src := c.id
        dst := outv.id
        map := binaryJacMapForVars c outv (.semantic (.binary opName .accum .inject)) 1.0
      }
    ]

private def stopGradientNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (_inv, _outv) ← requireUnaryShape opName eqn
    .ok #[]

private def iotaNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let _outv ← requireNullaryShape opName eqn
    .ok #[]

private def requireAtLeastOneOutput (op : OpName) (eqn : JEqn) : Except RuleError Unit := do
  if eqn.outvars.isEmpty then
    .error (.malformedEqn s!"No-grad rule `{op}` requires at least one output variable.")
  else
    .ok ()

private def requireAtLeastOneInputOneOutput
    (op : OpName)
    (eqn : JEqn) :
    Except RuleError (Array JVar × JVar) := do
  if eqn.invars.isEmpty then
    .error (.malformedEqn s!"Variadic rule `{op}` expects at least one input variable, got 0.")
  else
    let outv ←
      match eqn.outvars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Variadic rule `{op}` requires one output variable.")
    if eqn.outvars.size != 1 then
      .error (.malformedEqn s!"Variadic rule `{op}` expects exactly one output variable, got {eqn.outvars.size}.")
    else
      .ok (eqn.invars, outv)

private def devicePutNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let _ ← requireAtLeastOneOutput opName eqn
    .ok #[]

private def pjitNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let _ ← requireAtLeastOneOutput opName eqn
    .ok #[]

private def binaryNoGradRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (_a, _b, _outv) ← requireBinaryShape opName eqn
    .ok #[]

private def unarySemanticAliasRule
    (opName semanticOp : OpName)
    (mode : Tyr.AD.Sparse.JacMode := .none) :
    LocalJacRule :=
  unarySymbolicRule opName (.semantic (.unary semanticOp .x mode))

private def binarySemanticAliasRule
    (opName semanticOp : OpName)
    (leftMode : Tyr.AD.Sparse.JacMode := .none)
    (rightMode : Tyr.AD.Sparse.JacMode := .none) :
    LocalJacRule :=
  binarySymbolicRule
    opName
    ((.semantic (.binary semanticOp .a leftMode)), 1.0)
    ((.semantic (.binary semanticOp .b rightMode)), 1.0)

private def communicationUnaryRule (opName : OpName) : LocalJacRule :=
  unarySemanticAliasRule opName opName

private def structuralUnaryAliasRule
    (opName : OpName)
    (mode : Tyr.AD.Sparse.JacMode) :
    LocalJacRule :=
  fun eqn _ctx => do
    let (inv, outv) ← requireUnaryShape opName eqn
    let semanticOp := eqn.sourceOpName
    let tag : Tyr.AD.Sparse.SparseMapTag := .semantic (.unary semanticOp .x mode)
    let entries? :=
      if semanticOp == transposeAliasOpName || semanticOp == `jax.lax.transpose_p || semanticOp == `Graphax.transpose then
        transposeAliasEntries? inv outv
      else if
          semanticOp == reshapeAliasOpName ||
          semanticOp == `jax.lax.reshape_p ||
          semanticOp == `Graphax.reshape ||
          semanticOp == squeezeAliasOpName ||
          semanticOp == `jax.lax.squeeze_p ||
          semanticOp == `Graphax.squeeze then
        identityAliasEntries? inv outv
      else if
          semanticOp == broadcastInDimAliasOpName ||
          semanticOp == `jax.lax.broadcast_in_dim_p ||
          semanticOp == `Graphax.broadcast_in_dim then
        broadcastAliasEntries? inv outv
      else if
          semanticOp == sliceAliasOpName ||
          semanticOp == `jax.lax.slice_p ||
          semanticOp == sliceInDimAliasOpName ||
          semanticOp == `jax.lax.slice_in_dim_p ||
          semanticOp == `Graphax.slice ||
          semanticOp == `Graphax.slice_in_dim then
        sliceAliasEntries? eqn inv outv
      else
        none
    .ok #[{
      src := inv.id
      dst := outv.id
      map := exactUnaryMapOrFallback inv outv tag entries?
    }]

private def concatLikeStructuralRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (invars, outv) ← requireAtLeastOneInputOneOutput opName eqn
    let exactEntries? := concatAliasEntries? invars outv
    let semanticOp := eqn.sourceOpName
    let mut edges : Array LocalJacEdge := #[]
    for h : idx in [:invars.size] do
      let inv := invars[idx]
      let tag : Tyr.AD.Sparse.SparseMapTag := .semantic (.unary semanticOp .x .inject)
      let entries? := exactEntries?.bind (·[idx]?)
      edges := edges.push {
        src := inv.id
        dst := outv.id
        map := exactBinaryMapOrFallback inv outv tag entries?
      }
    .ok edges

private def reductionUnaryAliasRule (opName : OpName) : LocalJacRule :=
  unarySemanticAliasRule opName opName .contract

private def selectNAliasRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    if eqn.invars.size < 2 then
      .error (.malformedEqn s!"Control rule `{opName}` expects at least two input variables, got {eqn.invars.size}.")
    else
      let semanticOp := eqn.sourceOpName
      let outv ←
        match eqn.outvars[0]? with
        | some v => .ok v
        | none => .error (.malformedEqn s!"Control rule `{opName}` requires one output variable.")
      if eqn.outvars.size != 1 then
        .error (.malformedEqn s!"Control rule `{opName}` expects exactly one output variable, got {eqn.outvars.size}.")
      else
        let mut edges : Array LocalJacEdge := #[]
        for i in [1:eqn.invars.size] do
          let inv := eqn.invars[i]!
          edges := edges.push {
            src := inv.id
            dst := outv.id
            map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .mask)) 1.0
          }
        .ok edges

private def sliceWindow {α} (xs : Array α) (start count : Nat) : Array α :=
  let lo := min start xs.size
  let hi := min (lo + count) xs.size
  xs.extract lo hi

private def condAliasRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    if eqn.outvars.isEmpty then
      .error (.malformedEqn s!"Control-flow rule `{opName}` expects at least one output variable, got 0.")
    else
      let semanticOp := eqn.sourceOpName
      let predDefault := if eqn.invars.isEmpty then 0 else 1
      let predCount :=
        min ((eqn.params.findNat? .condPredicateCount).getD predDefault) eqn.invars.size
      let maxData := eqn.invars.size - predCount
      let dataCount :=
        min ((eqn.params.findNat? .condDataInputCount).getD maxData) maxData
      let dataInvars := sliceWindow eqn.invars predCount dataCount
      let mut edges : Array LocalJacEdge := #[]
      for outv in eqn.outvars do
        for inv in dataInvars do
          edges := edges.push {
            src := inv.id
            dst := outv.id
            map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .projection)) 1.0
          }
      .ok edges

private def scanAliasRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    if eqn.outvars.isEmpty then
      .error (.malformedEqn s!"Control-flow rule `{opName}` expects at least one output variable, got 0.")
    else
      let semanticOp := eqn.sourceOpName
      let carryDefault := if eqn.invars.isEmpty then 0 else 1
      let carryCount :=
        min ((eqn.params.findNat? .scanCarryInputCount).getD carryDefault) eqn.invars.size
      let carryInvars := sliceWindow eqn.invars 0 carryCount
      let maxData := eqn.invars.size - carryCount
      let dataCount :=
        min ((eqn.params.findNat? .scanDataInputCount).getD maxData) maxData
      let dataInvars := sliceWindow eqn.invars carryCount dataCount

      let carryOutDefault := min carryCount eqn.outvars.size
      let carryOutCount :=
        min ((eqn.params.findNat? .scanCarryOutputCount).getD carryOutDefault) eqn.outvars.size
      let carryOutvars := sliceWindow eqn.outvars 0 carryOutCount
      let dataOutvars := sliceWindow eqn.outvars carryOutCount (eqn.outvars.size - carryOutCount)

      let mut edges : Array LocalJacEdge := #[]
      for outv in carryOutvars do
        for inv in carryInvars do
          edges := edges.push {
            src := inv.id
            dst := outv.id
            map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .carry)) 1.0
          }
        for inv in dataInvars do
          edges := edges.push {
            src := inv.id
            dst := outv.id
            map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .projection)) 1.0
          }

      for outv in dataOutvars do
        for inv in carryInvars do
          edges := edges.push {
            src := inv.id
            dst := outv.id
            map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .carry)) 1.0
          }
        for inv in dataInvars do
          edges := edges.push {
            src := inv.id
            dst := outv.id
            map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .projection)) 1.0
          }
      .ok edges

private def dynamicProjectionAliasRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let semanticOp := eqn.sourceOpName
    let outv ←
      match eqn.outvars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Dynamic-projection rule `{opName}` requires one output variable.")
    if eqn.outvars.size != 1 then
      .error (.malformedEqn s!"Dynamic-projection rule `{opName}` expects exactly one output variable, got {eqn.outvars.size}.")
    else
      let inv ←
        match eqn.invars[0]? with
        | some v => .ok v
        | none => .error (.malformedEqn s!"Dynamic-projection rule `{opName}` expects at least one input variable.")
      .ok #[{
        src := inv.id
        dst := outv.id
        map := binaryJacMapForVars inv outv (.semantic (.unary semanticOp .x .projection)) 1.0
      }]

private def dynamicUpdateAliasRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let semanticOp := eqn.sourceOpName
    let outv ←
      match eqn.outvars[0]? with
      | some v => .ok v
      | none => .error (.malformedEqn s!"Dynamic-update rule `{opName}` requires one output variable.")
    if eqn.outvars.size != 1 then
      .error (.malformedEqn s!"Dynamic-update rule `{opName}` expects exactly one output variable, got {eqn.outvars.size}.")
    else
      let base ←
        match eqn.invars[0]? with
        | some v => .ok v
        | none => .error (.malformedEqn s!"Dynamic-update rule `{opName}` expects at least two input variables.")
      let update ←
        match eqn.invars[1]? with
        | some v => .ok v
        | none => .error (.malformedEqn s!"Dynamic-update rule `{opName}` expects at least two input variables.")
      .ok #[
        {
          src := base.id
          dst := outv.id
          map := binaryJacMapForVars base outv (.semantic (.binary semanticOp .lhs .projection)) 1.0
        },
        {
          src := update.id
          dst := outv.id
          map := binaryJacMapForVars update outv (.semantic (.binary semanticOp .rhs .inject)) 1.0
        }
      ]

private def graphaxUnaryAliasSpecs : Array (UnaryOp × Array OpName) := #[
  (.Neg, #[`jax.lax.neg, `Graphax.neg]),
  (.Abs, #[`jax.lax.abs, `Graphax.abs]),
  (.Exp, #[`jax.lax.exp, `Graphax.exp]),
  (.Log, #[`jax.lax.log, `Graphax.log]),
  (.Sqrt, #[`jax.lax.sqrt, `Graphax.sqrt]),
  (.Rsqrt, #[`jax.lax.rsqrt, `Graphax.rsqrt]),
  (.Square, #[`jax.lax.square, `Graphax.square]),
  (.Sin, #[`jax.lax.sin, `Graphax.sin]),
  (.Cos, #[`jax.lax.cos, `Graphax.cos]),
  (.Tanh, #[`jax.lax.tanh, `Graphax.tanh]),
  (.Sigmoid, #[`jax.lax.logistic, `Graphax.logistic, `jax.nn.sigmoid, `Graphax.sigmoid])
]

private def graphaxBinaryAliasSpecs : Array (BinaryOp × Array OpName) := #[
  (.Add, #[`jax.lax.add, `jax.lax.add_any, `Graphax.add, `Graphax.add_any]),
  (.Sub, #[`jax.lax.sub, `Graphax.sub]),
  (.Mul, #[`jax.lax.mul, `Graphax.mul]),
  (.Div, #[`jax.lax.div, `Graphax.div]),
  (.Max, #[`jax.lax.max, `Graphax.max]),
  (.Min, #[`jax.lax.min, `Graphax.min])
]

private def graphaxStructuralUnaryAliasSpecs :
    Array (OpName × Tyr.AD.Sparse.JacMode) := #[
  (transposeAliasOpName, .permute),
  (reshapeAliasOpName, .projection),
  (squeezeAliasOpName, .projection),
  (broadcastInDimAliasOpName, .expand),
  (sliceAliasOpName, .projection),
  (sliceInDimAliasOpName, .projection),
  (convertElementTypeAliasOpName, .cast),
  (`Graphax.transpose, .permute),
  (`Graphax.reshape, .projection),
  (`Graphax.squeeze, .projection),
  (`Graphax.broadcast_in_dim, .expand),
  (`Graphax.slice, .projection),
  (`Graphax.slice_in_dim, .projection),
  (`Graphax.convert_element_type, .cast)
]

private def cumsumStructuralRule (axis : ReduceAxis) : LocalJacRule :=
  fun eqn _ctx => do
    let opName := kstmtCumsumOpName axis
    let (inv, outv) ← requireUnaryShape opName eqn
    let tag : Tyr.AD.Sparse.SparseMapTag := .semantic (.cumsum (reduceAxisTagName axis) .x .prefix)
    let map :=
      match axis with
      | .Full =>
        match varFlatDim? inv, varFlatDim? outv with
        | some inDim, some outDim =>
          if inDim = 0 || outDim = 0 then
            unaryJacMapForVars inv outv tag 1.0
          else
            buildSparseMap tag inDim outDim (cumsumFullEntries inDim outDim)
        | _, _ =>
          unaryJacMapForVars inv outv tag 1.0
      | .Row =>
        match shapeRowsCols? inv, varFlatDim? inv, varFlatDim? outv with
        | some (rows, cols), some inDim, some outDim =>
          if inDim = 0 || outDim = 0 || inDim != outDim then
            unaryJacMapForVars inv outv tag 1.0
          else
            buildSparseMap tag inDim outDim (cumsumRowEntries rows cols)
        | _, _, _ =>
          unaryJacMapForVars inv outv tag 1.0
      | .Col =>
        match shapeRowsCols? inv, varFlatDim? inv, varFlatDim? outv with
        | some (rows, cols), some inDim, some outDim =>
          if inDim = 0 || outDim = 0 || inDim != outDim then
            unaryJacMapForVars inv outv tag 1.0
          else
            buildSparseMap tag inDim outDim (cumsumColEntries rows cols)
        | _, _, _ =>
          unaryJacMapForVars inv outv tag 1.0
    .ok #[{ src := inv.id, dst := outv.id, map := map }]

private def cumprodStructuralRule (axis : ReduceAxis) : LocalJacRule :=
  unarySymbolicRule
    (kstmtCumprodOpName axis)
    (.semantic (.cumprod (reduceAxisTagName axis) .x .prefixProduct))

private def padAliasRule (opName : OpName) : LocalJacRule :=
  fun eqn _ctx => do
    let (base, padv, outv) ← requireBinaryShape opName eqn
    let semanticOp := eqn.sourceOpName
    let baseTag : Tyr.AD.Sparse.SparseMapTag := .semantic (.binary semanticOp .lhs .projection)
    let padTag : Tyr.AD.Sparse.SparseMapTag := .semantic (.binary semanticOp .rhs .inject)
    let baseEntries? := padBaseEntries? eqn base outv
    let padEntries? := baseEntries?.bind (padValueEntries? padv outv)
    .ok #[
      {
        src := base.id
        dst := outv.id
        map := exactBinaryMapOrFallback base outv baseTag baseEntries?
      },
      {
        src := padv.id
        dst := outv.id
        map := exactBinaryMapOrFallback padv outv padTag padEntries?
      }
    ]

/-! ## Placeholder and hybrid rule packs (experimental scaffolding)

WARNING: The `*PlaceholderRules` and `*HybridRules` registration functions
in this section are experimental scaffolding, not sound differentiation.
Any op without an explicit semantic rule falls back to
`defaultPlaceholderRule`, which fabricates an identity-like Jacobian edge
per input regardless of the op's actual math — so local Jacobians (and any
cross-derivative/VJP quantities computed from them) produced under these
packs are numerically meaningless for such ops. These packs currently have
no call sites in the repository. For results built only from explicit
semantic rules, use `registerKStmtAllSupportedSemanticsRules` (ops without
a real rule then fail instead of silently producing placeholder values).
-/

/-- Register placeholder local-Jac rules for every KStmt unary and binary op. -/
def registerKStmtUnaryBinaryPlaceholderRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    registerLocalJacRule (kstmtUnaryOpName op) defaultPlaceholderRule
  for op in allKStmtBinaryOps do
    registerLocalJacRule (kstmtBinaryOpName op) defaultPlaceholderRule

/-- Register placeholder local-Jac rules for non-unary/binary KStmt op names. -/
def registerKStmtExtendedPlaceholderRules : Lean.CoreM Unit := do
  for op in allKStmtExtendedOpNames do
    registerLocalJacRule op defaultPlaceholderRule

/-- Register placeholder local-Jac rules for all currently lowered KStmt op names. -/
def registerKStmtAllSupportedPlaceholderRules : Lean.CoreM Unit := do
  registerKStmtUnaryBinaryPlaceholderRules
  registerKStmtExtendedPlaceholderRules

/--
Register constant-Jacobian local-Jac rules for linear/constant unary and binary ops:
- unary: `Copy`, `Neg`, `Zero`, `One`, `PosInfty`, `NegInfty`
- binary: `Add`, `Sub`
-/
def registerKStmtLinearSemanticsRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    match unaryConstJac? op with
    | some (repr, weight) =>
      registerLocalJacRule (kstmtUnaryOpName op) (unaryConstRule op repr weight)
    | none => pure ()
  for op in allKStmtBinaryOps do
    match binaryConstJac? op with
    | some (left, right) =>
      registerLocalJacRule (kstmtBinaryOpName op) (binaryConstRule op left right)
    | none => pure ()

/--
Register hybrid local-Jac rules for KStmt unary/binary ops:
- use Graphax/AlphaGrad-aligned semantics where available
- use `defaultPlaceholderRule` for remaining ops
-/
def registerKStmtUnaryBinaryHybridRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    registerLocalJacRule
      (kstmtUnaryOpName op)
      ((unarySemanticsRule? op).getD defaultPlaceholderRule)
  for op in allKStmtBinaryOps do
    registerLocalJacRule
      (kstmtBinaryOpName op)
      ((binarySemanticsRule? op).getD defaultPlaceholderRule)

/--
Register semantic local-Jac rules for all KStmt unary/binary ops that currently
have Graphax/AlphaGrad-overlap formulas in this registry.
-/
def registerKStmtGraphaxAlphaGradSemanticsRules : Lean.CoreM Unit := do
  for op in allKStmtUnaryOps do
    match unarySemanticsRule? op with
    | some rule => registerLocalJacRule (kstmtUnaryOpName op) rule
    | none => pure ()
  for op in allKStmtBinaryOps do
    match binarySemanticsRule? op with
    | some rule => registerLocalJacRule (kstmtBinaryOpName op) rule
    | none => pure ()

/--
Register structural local-Jacobian rules for non-unary/binary KStmt ops lowered
by `FromKStmt`.
-/
def registerKStmtStructuralSemanticsRules : Lean.CoreM Unit := do
  registerLocalJacRule kstmtTransposeOpName transposeStructuralRule
  registerLocalJacRule kstmtSwapLayoutOpName swapLayoutStructuralRule
  registerLocalJacRule kstmtConvertOpName convertStructuralRule
  registerLocalJacRule kstmtSliceRowsOpName sliceRowsStructuralRule
  registerLocalJacRule kstmtSliceColsOpName sliceColsStructuralRule
  registerLocalJacRule kstmtConcatColsOpName concatColsStructuralRule
  registerLocalJacRule kstmtOuterOpName outerStructuralRule
  registerLocalJacRule kstmtDotGeneralOpName (dotGeneralRule kstmtDotGeneralOpName)
  for trans in allKStmtMMATransposes do
    registerLocalJacRule (kstmtMmaOpName trans) (mmaRule trans)

  for axis in allKStmtBroadcastAxes do
    registerLocalJacRule (kstmtBroadcastOpName axis) (broadcastStructuralRule axis)

  for op in allKStmtBinaryOps do
    for axis in allKStmtBroadcastAxes do
      registerLocalJacRule
        (kstmtBinaryBroadcastOpName op axis)
        (binaryBroadcastStructuralRule op axis)

  for op in allKStmtReduceOps do
    for axis in allKStmtReduceAxes do
      registerLocalJacRule
        (kstmtReduceOpName op axis)
        (reduceStructuralRule op axis)
      registerLocalJacRule
        (kstmtReduceAccumOpName op axis)
        (reduceAccumStructuralRule op axis)

  for axis in allKStmtReduceAxes do
    registerLocalJacRule (kstmtCumsumOpName axis) (cumsumStructuralRule axis)
    registerLocalJacRule (kstmtCumprodOpName axis) (cumprodStructuralRule axis)

/--
Register local-Jac rules for all currently lowered KStmt ops:
- unary/binary use semantic-or-placeholder hybrid behavior
- structural/transform ops use explicit structural semantics
-/
def registerKStmtAllSupportedHybridRules : Lean.CoreM Unit := do
  registerKStmtUnaryBinaryHybridRules
  registerKStmtStructuralSemanticsRules

/--
Register explicit semantic local-Jac rules for all currently lowered KStmt ops,
without placeholder fallback.
-/
def registerKStmtAllSupportedSemanticsRules : Lean.CoreM Unit := do
  registerKStmtGraphaxAlphaGradSemanticsRules
  registerKStmtStructuralSemanticsRules

/--
Register no-grad/control primitive semantics used in Graphax/AlphaGrad-style
LeanJaxpr paths:
- `stop_gradient` emits no local-Jacobian edges.
- `iota` emits no local-Jacobian edges.
- `device_put` emits no local-Jacobian edges.
- `pjit` emits no local-Jacobian edges.
-/
def registerGraphaxAlphaGradNoGradControlRules : Lean.CoreM Unit := do
  for op in allStopGradientOpNames do
    registerLocalJacRule op (stopGradientNoGradRule op)
  for op in allIotaOpNames do
    registerLocalJacRule op (iotaNoGradRule op)
  for op in allDevicePutOpNames do
    registerLocalJacRule op (devicePutNoGradRule op)
  for op in allPjitOpNames do
    registerLocalJacRule op (pjitNoGradRule op)

/--
Register Graphax/JAX primitive aliases that are not emitted by current KStmt
lowering but should map to the same local-Jacobian semantics in LeanJaxpr paths.
-/
def registerGraphaxAlphaGradPrimitiveAliasRules : Lean.CoreM Unit := do
  for entry in graphaxUnaryAliasSpecs do
    let op := entry.1
    let aliases := entry.2
    match unarySemanticsRule? op with
    | some rule =>
      for alias in aliases do
        registerAliasWithLaxPrimVariant alias rule
    | none => pure ()

  for entry in graphaxBinaryAliasSpecs do
    let op := entry.1
    let aliases := entry.2
    match binarySemanticsRule? op with
    | some rule =>
      for alias in aliases do
        registerAliasWithLaxPrimVariant alias rule
    | none => pure ()

  for op in allGraphaxExtraUnaryAliasOpNames do
    registerLocalJacRule op (unarySemanticAliasRule op op)

  for op in allGraphaxExtraBinaryAliasOpNames do
    registerLocalJacRule op (binarySemanticAliasRule op op)

  for op in allGraphaxZeroBinaryAliasOpNames do
    registerLocalJacRule op (binaryNoGradRule op)

/--
Register communication primitive aliases with unary structured semantic tags.
These preserve dependency edges while keeping local-Jac metadata explicit.
-/
def registerGraphaxAlphaGradCommunicationRules : Lean.CoreM Unit := do
  for op in allCommunicationUnaryOpNames do
    registerLocalJacRule op (communicationUnaryRule op)

/--
Register structural alias rules used by Graphax/JAX source-path primitives that
do not flow through the current KStmt constructor set.
-/
def registerGraphaxAlphaGradStructuralAliasRules : Lean.CoreM Unit := do
  for entry in graphaxStructuralUnaryAliasSpecs do
    registerAliasWithLaxPrimVariant entry.1 (structuralUnaryAliasRule entry.1 entry.2)
  for op in allPadAliasOpNames do
    registerLocalJacRule op (padAliasRule op)
  for op in allConcatLikeAliasOpNames do
    registerLocalJacRule op (concatLikeStructuralRule op)

/--
Register source-path reduction/control aliases that are part of Graphax/JAX
primitive coverage but outside current KStmt naming.
- `select`/`select_n`: selector input excluded, data inputs propagated.
- `cond`: predicate input excluded, data inputs propagated.
- `scan`: all inputs (data/carry) propagated.
-/
def registerGraphaxAlphaGradReductionControlAliasRules : Lean.CoreM Unit := do
  for op in allReductionUnaryAliasOpNames do
    registerLocalJacRule op (reductionUnaryAliasRule op)
  for op in allSelectNAliasOpNames do
    registerLocalJacRule op (selectNAliasRule op)
  for op in allScanAliasOpNames do
    registerLocalJacRule op (scanAliasRule op)
  for op in allCondAliasOpNames do
    registerLocalJacRule op (condAliasRule op)

/--
Register dynamic/update structural aliases from Graphax/JAX source paths.
These rules keep index operands non-differentiated in local-Jac extraction.
-/
def registerGraphaxAlphaGradDynamicAliasRules : Lean.CoreM Unit := do
  for op in allDynamicProjectionAliasOpNames do
    registerLocalJacRule op (dynamicProjectionAliasRule op)
  for op in allDynamicUpdateAliasOpNames do
    registerLocalJacRule op (dynamicUpdateAliasRule op)

/-- Register dedicated dot-general local-Jacobian semantics (not outer-only approximation). -/
def registerGraphaxAlphaGradDotGeneralRules : Lean.CoreM Unit := do
  for op in allDotGeneralOpNames do
    registerLocalJacRule op (dotGeneralRule op)

/--
Register a parity-oriented semantic pack aligned with Graphax/AlphaGrad overlap:
- unary/binary overlap semantics
- structural semantics
- no-grad/control semantics
- dot-general semantics
-/
def registerGraphaxAlphaGradParityRules : Lean.CoreM Unit := do
  registerKStmtGraphaxAlphaGradSemanticsRules
  registerGraphaxAlphaGradPrimitiveAliasRules
  registerKStmtStructuralSemanticsRules
  registerGraphaxAlphaGradStructuralAliasRules
  registerGraphaxAlphaGradReductionControlAliasRules
  registerGraphaxAlphaGradDynamicAliasRules
  registerGraphaxAlphaGradNoGradControlRules
  registerGraphaxAlphaGradCommunicationRules
  registerGraphaxAlphaGradDotGeneralRules

/-- Register placeholder local-Jac rules for an explicit op-name list. -/
def registerPlaceholderRulesForOps (ops : Array OpName) : Lean.CoreM Unit := do
  for op in ops do
    registerLocalJacRule op defaultPlaceholderRule

end Tyr.AD.JaxprLike

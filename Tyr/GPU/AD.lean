import Tyr.AutoGrad
import Tyr.Torch

/-!
# Tyr.GPU.AD

Handwritten Lean-IR JVP/VJP rules for a small set of `torch.*` ops
(`add`/`sub`/`mul`/`matmul`), registered with the `Tyr.AD` rule registry
by `init`.

Note: `init` currently has no call sites in the repository, so these rules
are inert unless downstream code invokes it explicitly.
-/

namespace Tyr.GPU.AD

open Lean.IR
open Tyr.AD

/-- Helper for linear binary ops (like add, sub) for JVP -/
def registerBinaryLinearJVP (op : Lean.Name) : Lean.CoreM Unit := do
  registerJVPRule op fun args tans resTy => do
    let prim ← getFreshVar
    let tan ← getFreshVar
    
    let builder (rest : FnBody) := 
      FnBody.vdecl prim resTy (Expr.fap op args) (
        FnBody.vdecl tan resTy (Expr.fap op tans) rest
      )
      
    return (builder, prim, tan)

/-- Helper for linear binary ops for VJP -/
def registerBinaryLinearVJP (op : Lean.Name) : Lean.CoreM Unit := do
  registerVJPRule op fun _args dz _resTy => do
    let dx_in := dz
    let dy_in ← getFreshVar
    
    let builder (rest : FnBody) :=
      if op == `torch.sub then
        FnBody.vdecl dy_in IRType.object (Expr.fap `torch.neg #[Arg.var dz]) rest
      else
        FnBody.vdecl dy_in IRType.object (Expr.fap `torch.copy #[Arg.var dz]) rest

    return (builder, #[dx_in, dy_in])

/-- Initialize AD rules for GPU operations -/
def init : Lean.CoreM Unit := do
  registerBinaryLinearJVP `torch.add
  registerBinaryLinearJVP `torch.sub
  registerBinaryLinearVJP `torch.add
  registerBinaryLinearVJP `torch.sub
  
  -- Multiplication JVP
  registerJVPRule `torch.mul fun args tans resTy => do
    let x := args[0]!
    let y := args[1]!
    let dx := tans[0]!
    let dy := tans[1]!
    let prim ← getFreshVar
    let t1 ← getFreshVar
    let t2 ← getFreshVar
    let tan ← getFreshVar
    let builder (rest : FnBody) :=
      FnBody.vdecl prim resTy (Expr.fap `torch.mul args) (
        FnBody.vdecl t1 resTy (Expr.fap `torch.mul #[x, dy]) (
          FnBody.vdecl t2 resTy (Expr.fap `torch.mul #[y, dx]) (
            FnBody.vdecl tan resTy (Expr.fap `torch.add #[Arg.var t1, Arg.var t2]) rest
          )
        )
      )
    return (builder, prim, tan)

  -- Multiplication VJP
  registerVJPRule `torch.mul fun args dz resTy => do
    let x := args[0]!
    let y := args[1]!
    let dx ← getFreshVar
    let dy ← getFreshVar
    let builder (rest : FnBody) :=
      FnBody.vdecl dx resTy (Expr.fap `torch.mul #[Arg.var dz, y]) (
        FnBody.vdecl dy resTy (Expr.fap `torch.mul #[Arg.var dz, x]) rest
      )
    return (builder, #[dx, dy])
    
  -- Matmul JVP
  registerJVPRule `torch.matmul fun args tans resTy => do
    let A := args[0]!
    let B := args[1]!
    let dA := tans[0]!
    let dB := tans[1]!
    let prim ← getFreshVar
    let t1 ← getFreshVar
    let t2 ← getFreshVar
    let tan ← getFreshVar
    let builder (rest : FnBody) :=
      FnBody.vdecl prim resTy (Expr.fap `torch.matmul args) (
        FnBody.vdecl t1 resTy (Expr.fap `torch.matmul #[dA, B]) (
          FnBody.vdecl t2 resTy (Expr.fap `torch.matmul #[A, dB]) (
            FnBody.vdecl tan resTy (Expr.fap `torch.add #[Arg.var t1, Arg.var t2]) rest
          )
        )
      )
    return (builder, prim, tan)

  -- Matmul VJP
  registerVJPRule `torch.matmul fun args dC resTy => do
    let A := args[0]!
    let B := args[1]!
    let dA ← getFreshVar
    let dB ← getFreshVar
    let BT ← getFreshVar
    let AT ← getFreshVar
    let v0 ← getFreshVar
    let v1 ← getFreshVar
    
    let builder (rest : FnBody) :=
      FnBody.vdecl v0 IRType.uint64 (Expr.lit (LitVal.num 0)) (
        FnBody.vdecl v1 IRType.uint64 (Expr.lit (LitVal.num 1)) (
          FnBody.vdecl BT resTy (Expr.fap `torch.transpose #[B, Arg.var v0, Arg.var v1]) (
            FnBody.vdecl AT resTy (Expr.fap `torch.transpose #[A, Arg.var v0, Arg.var v1]) (
              FnBody.vdecl dA resTy (Expr.fap `torch.matmul #[Arg.var dC, Arg.var BT]) (
                FnBody.vdecl dB resTy (Expr.fap `torch.matmul #[Arg.var AT, Arg.var dC]) rest
              )
            )
          )
        )
      )
    
    return (builder, #[dA, dB])

end Tyr.GPU.AD
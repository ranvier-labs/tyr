import Lean
import Tyr.GPU.Codegen.TileIR.Expr

/-! # TileIR specialization

`Tyr.GPU.Codegen.TileIR.recoverConstParamKinds?` recovers the ordered `ct.Const`
binder kinds (`Tyr.GPU.Codegen.TileIR.ConstParamKind`) from a TileIR kernel
type.
-/

open Lean

namespace Tyr.GPU.Codegen.TileIR

inductive ConstParamKind where
  | nat
  | int
  | bool
  deriving Repr, Inhabited, BEq

private partial def constParamKindsAux? (ty : Expr) (acc : Array ConstParamKind) : MetaM (Option (Array ConstParamKind)) := do
  let ty : Expr ← Lean.Meta.whnf ty
  if Expr.isConstOf ty ``Tyr.GPU.Codegen.TileIR.Module then
    pure (some acc)
  else if Expr.isForall ty then
    let domain : Expr ← Lean.Meta.whnf (Expr.bindingDomain! ty)
    let kind? :=
      if Expr.isConstOf domain ``Nat then
        some ConstParamKind.nat
      else if Expr.isConstOf domain ``Int then
        some ConstParamKind.int
      else if Expr.isConstOf domain ``Bool then
        some ConstParamKind.bool
      else
        none
    match kind? with
    | some kind =>
        constParamKindsAux? (Expr.bindingBody! ty).headBeta (acc.push kind)
    | none =>
        pure none
  else
    pure none

/-- Recover the ordered `ct.Const` binder kinds from a TileIR kernel type. -/
def recoverConstParamKinds? (ty : Expr) : MetaM (Option (Array ConstParamKind)) :=
  constParamKindsAux? ty #[]

end Tyr.GPU.Codegen.TileIR

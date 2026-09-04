import Tyr.DiffEq.Solver.Base

namespace torch
namespace DiffEq

/-! ## Reversible Heun Solver -/

attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hAddInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hSubInst
attribute [local instance] _root_.torch.DiffEq.DiffEqArithmetic.hMulInst

structure ReversibleHeun where
  deriving Inhabited

def ReversibleHeun.solver {Term Y VF Control Args : Type}
    [TermLike Term Y VF Control Args]
    [DiffEqSpace Y] :
    AbstractSolver Term Y VF Control Args := {
  SolverState := Y × VF
  DenseInfo := LocalLinearDenseInfo Y
  termStructure := TermStructure.single
  order := fun _ => 2
  strongOrder := fun _ => 0.5
  init := fun term t0 _t1 y0 args =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    let vf0 := inst.vf term t0 y0 args
    (y0, vf0)
  step := fun term t0 t1 y0 args state madeJump =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    let (yhat0, vf0Prev) := state
    let vf0 := if madeJump then inst.vf term t0 y0 args else vf0Prev
    let control := inst.contr term t0 t1
    let prod0 := inst.prod term vf0 control
    let yhat1 :=
      (y0 + (y0 - yhat0)) + prod0
    let vf1 := inst.vf term t1 yhat1 args
    let prod1 := inst.prod term vf1 control
    let y1 :=
      y0 + (0.5 * (prod0 + prod1))
    let yErr :=
      0.5 * (prod1 - prod0)
    let dense := { t0 := t0, t1 := t1, y0 := y0, y1 := y1 }
    {
      y1 := y1
      yError := some yErr
      denseInfo := dense
      solverState := (yhat1, vf1)
      result := Result.successful
    }
  func := fun term t y args =>
    let inst := (inferInstance : TermLike Term Y VF Control Args)
    inst.vf term t y args
  interpolation := fun info => info.toInterpolation
}

instance : ExplicitSolver ReversibleHeun := ⟨True.intro⟩
instance : StratonovichSolver ReversibleHeun := ⟨True.intro⟩
instance : AdaptiveSolver ReversibleHeun := ⟨True.intro⟩
instance : ReversibleSolver ReversibleHeun := ⟨True.intro⟩

end DiffEq
end torch

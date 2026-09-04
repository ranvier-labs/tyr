import Std

/-!
# Tyr.Mctx.Math

Small numeric helpers shared by the `Tyr.Mctx` search modules: array
reductions, argmax (with an invalid-action mask), numerically stable
softmax, and guarded log/division.
-/

namespace torch.mctx

/-- Sum of a float array. -/
def sum (xs : Array Float) : Float :=
  xs.foldl (init := 0.0) (· + ·)

/-- Maximum value of a float array, defaults to 0 for empty arrays. -/
def maxD (xs : Array Float) (default : Float := 0.0) : Float :=
  xs.foldl (init := default) fun acc x =>
    if x > acc then x else acc

/-- Argmax over an array, returns 0 for empty arrays. -/
def argmax (xs : Array Float) : Nat :=
  if xs.isEmpty then
    0
  else
    let init := (0, xs.getD 0 0.0)
    let (bestIdx, _) := (List.range xs.size).foldl (init := init) fun (acc : Nat × Float) i =>
      let xi := xs.getD i (-1.0 / 0.0)
      if xi > acc.2 then (i, xi) else acc
    bestIdx

/-- Argmax with an invalid-action mask (`true` means invalid). -/
def maskedArgmax (scores : Array Float) (invalid : Option (Array Bool)) : Nat :=
  let masked :=
    match invalid with
    | none => scores
    | some inv =>
      (List.range scores.size).toArray.map fun i =>
        let s := scores.getD i 0.0
        let isInvalid := inv.getD i false
        if isInvalid then -1e30 else s
  argmax masked

/-- Numerically stable softmax. -/
def softmax (xs : Array Float) : Array Float :=
  if xs.isEmpty then
    #[]
  else
    let m := maxD xs (xs.getD 0 0.0)
    let exps := xs.map (fun x => Float.exp (x - m))
    let z := sum exps
    if z <= 0.0 then
      let p := 1.0 / (Float.ofNat xs.size)
      Array.replicate xs.size p
    else
      exps.map (fun e => e / z)

/-- `log(max(x, tiny))` helper. -/
def logSafe (x : Float) (tiny : Float := 1e-30) : Float :=
  Float.log (if x < tiny then tiny else x)

/-- Safe division helper. -/
def divSafe (x y : Float) (eps : Float := 1e-8) : Float :=
  x / (if Float.abs y < eps then eps else y)

end torch.mctx

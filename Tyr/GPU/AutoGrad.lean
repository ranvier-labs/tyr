import Tyr.AutoGrad
import Tyr.GPU.Codegen.IR

/-!
# Tyr.GPU.AutoGrad

Symbolic AD over the GPU kernel IR (`Tyr.GPU.Codegen`): linearizes `KStmt`
programs into a linear trace (`LinearInst`) and transposes that trace into
VJP kernel statements. Custom op rules can be supplied through the GPU VJP
registry (`registerGpuVJPRule`/`getGpuVJPRule` in `Tyr.AutoGrad`).

Note: the GPU VJP registry currently has no registered rules anywhere in
the repository, so `getGpuVJPRule` lookups always fall back to the built-in
rules below.
-/

namespace Tyr.GPU.AD

open Tyr.AD
open Tyr.GPU.Codegen

/-- 
  Linear Intermediate Representation (LIR).
  Represents the linearized execution trace (JVP logic).
-/
inductive LinearInst where
  | id (dout din : VarId)
  | add (dout din1 din2 : VarId)
  | sub (dout din1 din2 : VarId)
  | mul (dout din1 din2 : VarId) (primal1 primal2 : VarId)
  | div (dout din1 din2 : VarId) (primal1 primal2 : VarId)
  | max (dout din1 din2 : VarId) (primalY primal1 primal2 : VarId)
  | min (dout din1 din2 : VarId) (primalY primal1 primal2 : VarId)
  | exp (dout din : VarId) (primalY : VarId)
  | exp2 (dout din : VarId) (primalY : VarId)
  | log (dout din : VarId) (primalX : VarId)
  | log2 (dout din : VarId) (primalX : VarId)
  | neg (dout din : VarId)
  | sqrt (dout din : VarId) (primalY : VarId)
  | rsqrt (dout din : VarId) (primalY : VarId)
  | tanh (dout din : VarId) (primalY : VarId)
  | sigmoid (dout din : VarId) (primalY : VarId)
  | recip (dout din : VarId) (primalX : VarId)
  | square (dout din : VarId) (primalX : VarId)
  | sin (dout din : VarId) (primalX : VarId)
  | cos (dout din : VarId) (primalX : VarId)
  | broadcast (axis : BroadcastAxis) (dout din : VarId)
  | sum (axis : ReduceAxis) (dout din : VarId)
  | reduceMax (axis : ReduceAxis) (dout din : VarId) (primalY primalX : VarId)
  | reduceMin (axis : ReduceAxis) (dout din : VarId) (primalY primalX : VarId)
  | reduceProd (axis : ReduceAxis) (dout din : VarId) (primalY primalX : VarId)
  | mma (trans : MMATranspose) (dout da db dc : VarId) (primalA primalB : VarId)
  | loop (v : VarId) (lo hi : Nat) (body : Array LinearInst)
  | loopVal (v : VarId) (lo : Nat) (hi : VarId) (body : Array LinearInst)
  | custom (name : String) (dout : VarId) (dins : Array VarId) (ctx : Array VarId)
  deriving Repr, Inhabited

/-- Convert BroadcastAxis to ReduceAxis for VJP -/
def broadcastToReduceAxis : BroadcastAxis → ReduceAxis
  | .Row => .Col
  | .Col => .Row

/-- Convert ReduceAxis to BroadcastAxis for VJP of reductions. -/
def reduceToBroadcastAxis? : ReduceAxis → Option BroadcastAxis
  | .Row => some .Col
  | .Col => some .Row
  | .Full => none

/-- Trace State -/
structure TraceState where
  primalStmts : Array KStmt := #[]
  linearTrace : Array LinearInst := #[]

abbrev TraceM := StateT TraceState ADM

def emitPrimal (s : KStmt) : TraceM Unit := 
  modify fun st => { st with primalStmts := st.primalStmts.push s }

def emitLinear (l : LinearInst) : TraceM Unit :=
  modify fun st => { st with linearTrace := st.linearTrace.push l }

/-- Lift ADM actions to TraceM -/
def liftADM (m : ADM α) : TraceM α := 
  liftM m

private def unsupportedLinearize {α} (kind detail : String) : TraceM α := do
  throwError s!"GPU autodiff linearization does not support {kind} `{detail}`"

private def ln2 : Float := 0.6931471805599453
private def invLn2 : Float := 1.4426950408889634

/-- Linearize a statement -/
partial def linearizeStmt (s : KStmt) : TraceM Unit := do
  match s with
  | .declRT v _ _ _ _ =>
    let dv ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent v dv
    emitPrimal s
    
  | .declST v _ _ _ _ =>
    let dv ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent v dv
    emitPrimal s

  | .declRV v _ _ =>
    let dv ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent v dv
    emitPrimal s

  | .declSV v _ _ =>
    let dv ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent v dv
    emitPrimal s

  | .load dst src =>
    let ddst ← liftADM getFreshGpuVar
    let dsrc ← liftADM $ getGpuTangent src
    liftADM $ setGpuTangent dst ddst
    emitPrimal s
    emitLinear $ .id ddst dsrc

  | .store dst src =>
    let ddst ← liftADM $ getGpuTangent dst
    let dsrc ← liftADM $ getGpuTangent src
    emitPrimal s
    emitLinear $ .id ddst dsrc

  | .binary op dst a b =>
    let da ← liftADM $ getGpuTangent a
    let db ← liftADM $ getGpuTangent b
    let ddst ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent dst ddst
    emitPrimal s
    
    -- Check for custom rule first
    let opName := "BinaryOp." ++ toString op
    let rule? ← liftADM $ liftM $ getGpuVJPRule opName
    match rule? with
    | some _ => 
      -- If custom rule exists, emit custom LIR instruction.
      -- We assume custom rules take (dout, [da, db], [a, b]) for binary ops.
      -- This requires convention. For now, just emit custom.
      emitLinear $ .custom opName ddst #[da, db] #[a, b]
    | none =>
      match op with
      | .Add => emitLinear $ .add ddst da db
      | .Sub => emitLinear $ .sub ddst da db
      | .Mul => emitLinear $ .mul ddst da db a b
      | .Div => emitLinear $ .div ddst da db a b
      | .Max => emitLinear $ .max ddst da db dst a b
      | .Min => emitLinear $ .min ddst da db dst a b

  | .unary op dst src =>
    let dsrc ← liftADM $ getGpuTangent src
    let ddst ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent dst ddst
    emitPrimal s
    
    let opName := "UnaryOp." ++ toString op
    let rule? ← liftADM $ liftM $ getGpuVJPRule opName
    match rule? with
    | some _ =>
      emitLinear $ .custom opName ddst #[dsrc] #[src, dst] -- Pass input and output as context
    | none =>
      match op with
      | .Copy => emitLinear $ .id ddst dsrc
      | .Exp => emitLinear $ .exp ddst dsrc dst
      | .Exp2 => emitLinear $ .exp2 ddst dsrc dst
      | .Log => emitLinear $ .log ddst dsrc src
      | .Log2 => emitLinear $ .log2 ddst dsrc src
      | .Neg => emitLinear $ .neg ddst dsrc
      | .Sqrt => emitLinear $ .sqrt ddst dsrc dst
      | .Rsqrt => emitLinear $ .rsqrt ddst dsrc dst
      | .Tanh => emitLinear $ .tanh ddst dsrc dst
      | .FastTanh => emitLinear $ .tanh ddst dsrc dst
      | .Sigmoid => emitLinear $ .sigmoid ddst dsrc dst
      | .Recip => emitLinear $ .recip ddst dsrc src
      | .Square => emitLinear $ .square ddst dsrc src
      | .Sin => emitLinear $ .sin ddst dsrc src
      | .Cos => emitLinear $ .cos ddst dsrc src
      | _ => unsupportedLinearize "unary op" (reprStr op)

  | .broadcast axis dst vec =>
    let dvec ← liftADM $ getGpuTangent vec
    let ddst ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent dst ddst
    emitPrimal s
    emitLinear $ .broadcast axis ddst dvec

  | .reduce op axis dst src =>
    let dsrc ← liftADM $ getGpuTangent src
    let ddst ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent dst ddst
    emitPrimal s
    match op with
    | .Sum => emitLinear $ .sum axis ddst dsrc
    | .Max =>
      match axis with
      | .Row | .Col => emitLinear $ .reduceMax axis ddst dsrc dst src
      | .Full => unsupportedLinearize "full reduction op" (reprStr op)
    | .Min =>
      match axis with
      | .Row | .Col => emitLinear $ .reduceMin axis ddst dsrc dst src
      | .Full => unsupportedLinearize "full reduction op" (reprStr op)
    | .Prod =>
      match axis with
      | .Row | .Col => emitLinear $ .reduceProd axis ddst dsrc dst src
      | .Full => unsupportedLinearize "full reduction op" (reprStr op)

  | .mma trans dst a b c =>
    let da ← liftADM $ getGpuTangent a
    let db ← liftADM $ getGpuTangent b
    let dc ← liftADM $ getGpuTangent c
    let ddst ← liftADM getFreshGpuVar
    liftADM $ setGpuTangent dst ddst
    emitPrimal s
    emitLinear $ .mma trans ddst da db dc a b

  | .forLoop v lo hi body =>
    let savedState ← get
    set { savedState with primalStmts := #[], linearTrace := #[] }
    body.forM linearizeStmt
    let innerState ← get
    set { savedState with 
          primalStmts := savedState.primalStmts.push (.forLoop v lo hi innerState.primalStmts)
          linearTrace := savedState.linearTrace.push (.loop v lo hi innerState.linearTrace) }

  | .forLoopVal v lo hi body =>
    let savedState ← get
    set { savedState with primalStmts := #[], linearTrace := #[] }
    body.forM linearizeStmt
    let innerState ← get
    set { savedState with
          primalStmts := savedState.primalStmts.push (.forLoopVal v lo hi innerState.primalStmts)
          linearTrace := savedState.linearTrace.push (.loopVal v lo hi innerState.linearTrace) }

  | other => emitPrimal other

/-- Transpose a Linear Trace to produce VJP statements -/
partial def transposeTrace (trace : Array LinearInst) : ADM (Array KStmt) := do
  let mut stmts := #[]
  
  for inst in trace.reverse do
    match inst with
    | .id dout din =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      stmts := stmts.push (.binary .Add ddin ddin ddout)

    | .add dout din1 din2 =>
      let ddout ← getGpuCotangent dout
      let ddin1 ← getGpuCotangent din1
      let ddin2 ← getGpuCotangent din2
      stmts := stmts.push (.binary .Add ddin1 ddin1 ddout)
      stmts := stmts.push (.binary .Add ddin2 ddin2 ddout)

    | .sub dout din1 din2 =>
      let ddout ← getGpuCotangent dout
      let ddin1 ← getGpuCotangent din1
      let ddin2 ← getGpuCotangent din2
      stmts := stmts.push (.binary .Add ddin1 ddin1 ddout)
      stmts := stmts.push (.binary .Sub ddin2 ddin2 ddout)

    | .mul dout da db a b =>
      let ddout ← getGpuCotangent dout
      let dda ← getGpuCotangent da
      let ddb ← getGpuCotangent db
      let t1 ← getFreshGpuVar
      let t2 ← getFreshGpuVar
      stmts := stmts.push (.binary .Mul t1 ddout b)
      stmts := stmts.push (.binary .Add dda dda t1)
      stmts := stmts.push (.binary .Mul t2 ddout a)
      stmts := stmts.push (.binary .Add ddb ddb t2)

    | .div dout da db a b =>
      let ddout ← getGpuCotangent dout
      let dda ← getGpuCotangent da
      let ddb ← getGpuCotangent db
      let inv_b ← getFreshGpuVar
      let inv_b_sq ← getFreshGpuVar
      let neg_a ← getFreshGpuVar
      let db_term ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      let t2 ← getFreshGpuVar
      -- da += dout * (1/b)
      stmts := stmts.push (.unary .Recip inv_b b)
      stmts := stmts.push (.binary .Mul t1 ddout inv_b)
      stmts := stmts.push (.binary .Add dda dda t1)
      -- db += dout * (-a / b^2)
      stmts := stmts.push (.binary .Mul inv_b_sq inv_b inv_b)
      stmts := stmts.push (.scalarMul neg_a a (-1.0))
      stmts := stmts.push (.binary .Mul db_term neg_a inv_b_sq)
      stmts := stmts.push (.binary .Mul t2 ddout db_term)
      stmts := stmts.push (.binary .Add ddb ddb t2)

    | .max dout da db y a b =>
      let ddout ← getGpuCotangent dout
      let dda ← getGpuCotangent da
      let ddb ← getGpuCotangent db
      let maskA ← getFreshGpuVar
      let maskB ← getFreshGpuVar
      let count ← getFreshGpuVar
      let weightA ← getFreshGpuVar
      let weightB ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      let t2 ← getFreshGpuVar
      stmts := stmts.push (.eqMask maskA a y)
      stmts := stmts.push (.eqMask maskB b y)
      stmts := stmts.push (.binary .Add count maskA maskB)
      stmts := stmts.push (.binary .Div weightA maskA count)
      stmts := stmts.push (.binary .Div weightB maskB count)
      stmts := stmts.push (.binary .Mul t1 ddout weightA)
      stmts := stmts.push (.binary .Add dda dda t1)
      stmts := stmts.push (.binary .Mul t2 ddout weightB)
      stmts := stmts.push (.binary .Add ddb ddb t2)

    | .min dout da db y a b =>
      let ddout ← getGpuCotangent dout
      let dda ← getGpuCotangent da
      let ddb ← getGpuCotangent db
      let maskA ← getFreshGpuVar
      let maskB ← getFreshGpuVar
      let count ← getFreshGpuVar
      let weightA ← getFreshGpuVar
      let weightB ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      let t2 ← getFreshGpuVar
      stmts := stmts.push (.eqMask maskA a y)
      stmts := stmts.push (.eqMask maskB b y)
      stmts := stmts.push (.binary .Add count maskA maskB)
      stmts := stmts.push (.binary .Div weightA maskA count)
      stmts := stmts.push (.binary .Div weightB maskB count)
      stmts := stmts.push (.binary .Mul t1 ddout weightA)
      stmts := stmts.push (.binary .Add dda dda t1)
      stmts := stmts.push (.binary .Mul t2 ddout weightB)
      stmts := stmts.push (.binary .Add ddb ddb t2)

    | .exp dout din y =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.binary .Mul t1 ddout y)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .exp2 dout din y =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.binary .Mul t1 ddout y)
      stmts := stmts.push (.scalarMul t1 t1 ln2)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .log dout din x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.binary .Div t1 ddout x)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .log2 dout din x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.binary .Div t1 ddout x)
      stmts := stmts.push (.scalarMul t1 t1 invLn2)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .neg dout din =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      stmts := stmts.push (.binary .Sub ddin ddin ddout)

    | .sqrt dout din y =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let twoY ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.scalarMul twoY y 2.0)
      stmts := stmts.push (.binary .Div t1 ddout twoY)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .rsqrt dout din y =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let ySq ← getFreshGpuVar
      let yCube ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.binary .Mul ySq y y)
      stmts := stmts.push (.binary .Mul yCube ySq y)
      stmts := stmts.push (.binary .Mul t1 ddout yCube)
      stmts := stmts.push (.scalarMul t1 t1 (-0.5))
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .tanh dout din y =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let t1 ← getFreshGpuVar
      let t2 ← getFreshGpuVar
      -- ddin += ddout * (1 - y^2)
      stmts := stmts.push (.binary .Mul t1 y y)
      stmts := stmts.push (.scalarMul t1 t1 (-1.0))
      stmts := stmts.push (.scalarAdd t1 t1 1.0)
      stmts := stmts.push (.binary .Mul t2 ddout t1)
      stmts := stmts.push (.binary .Add ddin ddin t2)

    | .sigmoid dout din y =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let oneMinusY ← getFreshGpuVar
      let deriv ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.scalarMul oneMinusY y (-1.0))
      stmts := stmts.push (.scalarAdd oneMinusY oneMinusY 1.0)
      stmts := stmts.push (.binary .Mul deriv y oneMinusY)
      stmts := stmts.push (.binary .Mul t1 ddout deriv)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .recip dout din x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let invX ← getFreshGpuVar
      let invXSq ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.unary .Recip invX x)
      stmts := stmts.push (.binary .Mul invXSq invX invX)
      stmts := stmts.push (.binary .Mul t1 ddout invXSq)
      stmts := stmts.push (.scalarMul t1 t1 (-1.0))
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .square dout din x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.binary .Mul t1 ddout x)
      stmts := stmts.push (.scalarMul t1 t1 2.0)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .sin dout din x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let cosX ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.unary .Cos cosX x)
      stmts := stmts.push (.binary .Mul t1 ddout cosX)
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .cos dout din x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let sinX ← getFreshGpuVar
      let t1 ← getFreshGpuVar
      stmts := stmts.push (.unary .Sin sinX x)
      stmts := stmts.push (.binary .Mul t1 ddout sinX)
      stmts := stmts.push (.scalarMul t1 t1 (-1.0))
      stmts := stmts.push (.binary .Add ddin ddin t1)

    | .broadcast axis dout din =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      let reduceAxis := broadcastToReduceAxis axis
      stmts := stmts.push (.reduceAccum .Sum reduceAxis ddin ddout ddin)

    | .sum axis dout din =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      match reduceToBroadcastAxis? axis with
      | some bAxis =>
        let btmp ← getFreshGpuVar
        stmts := stmts.push (.broadcast bAxis btmp ddout)
        stmts := stmts.push (.binary .Add ddin ddin btmp)
      | none =>
        -- Best-effort fallback for scalar/full reductions.
        stmts := stmts.push (.binary .Add ddin ddin ddout)

    | .reduceMax axis dout din y x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      match reduceToBroadcastAxis? axis with
      | some bAxis =>
        let yBroad ← getFreshGpuVar
        let mask ← getFreshGpuVar
        let count ← getFreshGpuVar
        let countBroad ← getFreshGpuVar
        let weight ← getFreshGpuVar
        let ddoutBroad ← getFreshGpuVar
        let contrib ← getFreshGpuVar
        stmts := stmts.push (.broadcast bAxis yBroad y)
        stmts := stmts.push (.eqMask mask x yBroad)
        stmts := stmts.push (.reduce .Sum axis count mask)
        stmts := stmts.push (.broadcast bAxis countBroad count)
        stmts := stmts.push (.binary .Div weight mask countBroad)
        stmts := stmts.push (.broadcast bAxis ddoutBroad ddout)
        stmts := stmts.push (.binary .Mul contrib ddoutBroad weight)
        stmts := stmts.push (.binary .Add ddin ddin contrib)
      | none =>
        throwError s!"GPU autodiff transpose does not support full Max reductions"

    | .reduceMin axis dout din y x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      match reduceToBroadcastAxis? axis with
      | some bAxis =>
        let yBroad ← getFreshGpuVar
        let mask ← getFreshGpuVar
        let count ← getFreshGpuVar
        let countBroad ← getFreshGpuVar
        let weight ← getFreshGpuVar
        let ddoutBroad ← getFreshGpuVar
        let contrib ← getFreshGpuVar
        stmts := stmts.push (.broadcast bAxis yBroad y)
        stmts := stmts.push (.eqMask mask x yBroad)
        stmts := stmts.push (.reduce .Sum axis count mask)
        stmts := stmts.push (.broadcast bAxis countBroad count)
        stmts := stmts.push (.binary .Div weight mask countBroad)
        stmts := stmts.push (.broadcast bAxis ddoutBroad ddout)
        stmts := stmts.push (.binary .Mul contrib ddoutBroad weight)
        stmts := stmts.push (.binary .Add ddin ddin contrib)
      | none =>
        throwError s!"GPU autodiff transpose does not support full Min reductions"

    | .reduceProd axis dout din y x =>
      let ddout ← getGpuCotangent dout
      let ddin ← getGpuCotangent din
      match reduceToBroadcastAxis? axis with
      | some bAxis =>
        let zeroX ← getFreshGpuVar
        let zeroVec ← getFreshGpuVar
        let oneVec ← getFreshGpuVar
        let zeroMask ← getFreshGpuVar
        let zeroCount ← getFreshGpuVar
        let caseNoZero ← getFreshGpuVar
        let caseOneZero ← getFreshGpuVar
        let safeX ← getFreshGpuVar
        let safeProd ← getFreshGpuVar
        let safeDenom ← getFreshGpuVar
        let yBroad ← getFreshGpuVar
        let safeProdBroad ← getFreshGpuVar
        let caseNoZeroBroad ← getFreshGpuVar
        let caseOneZeroBroad ← getFreshGpuVar
        let divTerm ← getFreshGpuVar
        let noZeroWeight ← getFreshGpuVar
        let oneZeroWeight ← getFreshGpuVar
        let weight ← getFreshGpuVar
        let ddoutBroad ← getFreshGpuVar
        let contrib ← getFreshGpuVar
        stmts := stmts.push (.unary .Zero zeroX x)
        stmts := stmts.push (.unary .Zero zeroVec ddout)
        stmts := stmts.push (.unary .One oneVec ddout)
        stmts := stmts.push (.eqMask zeroMask x zeroX)
        stmts := stmts.push (.reduce .Sum axis zeroCount zeroMask)
        stmts := stmts.push (.eqMask caseNoZero zeroCount zeroVec)
        stmts := stmts.push (.eqMask caseOneZero zeroCount oneVec)
        stmts := stmts.push (.binary .Add safeX x zeroMask)
        stmts := stmts.push (.reduce .Prod axis safeProd safeX)
        stmts := stmts.push (.broadcast bAxis yBroad y)
        stmts := stmts.push (.broadcast bAxis safeProdBroad safeProd)
        stmts := stmts.push (.broadcast bAxis caseNoZeroBroad caseNoZero)
        stmts := stmts.push (.broadcast bAxis caseOneZeroBroad caseOneZero)
        stmts := stmts.push (.binary .Add safeDenom x zeroMask)
        stmts := stmts.push (.binary .Div divTerm yBroad safeDenom)
        stmts := stmts.push (.binary .Mul noZeroWeight caseNoZeroBroad divTerm)
        stmts := stmts.push (.binary .Mul oneZeroWeight zeroMask safeProdBroad)
        stmts := stmts.push (.binary .Mul oneZeroWeight oneZeroWeight caseOneZeroBroad)
        stmts := stmts.push (.binary .Add weight noZeroWeight oneZeroWeight)
        stmts := stmts.push (.broadcast bAxis ddoutBroad ddout)
        stmts := stmts.push (.binary .Mul contrib ddoutBroad weight)
        stmts := stmts.push (.binary .Add ddin ddin contrib)
      | none =>
        throwError s!"GPU autodiff transpose does not support full Prod reductions"

    | .mma trans dout da db dc a b =>
      let ddout ← getGpuCotangent dout
      let dda ← getGpuCotangent da
      let ddb ← getGpuCotangent db
      let ddc ← getGpuCotangent dc
      -- dC += dOut
      stmts := stmts.push (.binary .Add ddc ddc ddout)
      -- dA/dB contributions for each transpose mode.
      match trans with
      | .AB =>
        -- dA += dOut * B^T, dB += A^T * dOut
        stmts := stmts.push (.mma .ABt dda ddout b dda)
        stmts := stmts.push (.mma .AtB ddb a ddout ddb)
      | .ABt =>
        -- dA += dOut * B, dB += dOut^T * A
        stmts := stmts.push (.mma .AB dda ddout b dda)
        stmts := stmts.push (.mma .AtB ddb ddout a ddb)
      | .AtB =>
        -- dA += B * dOut^T, dB += A * dOut
        stmts := stmts.push (.mma .ABt dda b ddout dda)
        stmts := stmts.push (.mma .AB ddb a ddout ddb)
      | .AtBt =>
        -- dA += B^T * dOut^T, dB += dOut^T * A^T
        stmts := stmts.push (.mma .AtBt dda b ddout dda)
        stmts := stmts.push (.mma .AtBt ddb ddout a ddb)

    | .loop v lo hi body =>
      let transBody ← transposeTrace body
      stmts := stmts.push (.forLoopRev v lo hi transBody)

    | .loopVal v lo hi body =>
      let transBody ← transposeTrace body
      stmts := stmts.push (.forLoopValRev v lo hi transBody)

    | .custom name dout dins ctx =>
      let rule? ← liftM $ getGpuVJPRule name
      match rule? with
      | some rule =>
        let ddout ← getGpuCotangent dout
        let mut ddins := #[]
        for din in dins do
          ddins := ddins.push (← getGpuCotangent din)
        
        let customStmts ← rule ddout ddins ctx
        stmts := stmts ++ customStmts
      | none => pure ()

  return stmts

end Tyr.GPU.AD

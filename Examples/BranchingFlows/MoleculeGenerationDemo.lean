import Tyr.Model.BranchingFlows.Molecule
import Tyr.Model.BranchingFlows.QM9

/-!
  Examples/BranchingFlows/MoleculeGenerationDemo.lean

  Molecule-shaped BranchingFlows target-conditioned generation demo.

  It uses the preprocessed QM9 JSONL boundary, constructs native molecule
  `BranchingState`s, samples a training bridge, runs a target-conditioned
  forward generation loop from the length-one masked source state, and writes
  XYZ point clouds for downstream OpenBabel/RDKit checks. The target supplies
  the coordinate and label predictions; this demonstrates the branching
  mechanism rather than a learned molecular distribution.
-/

namespace Examples.BranchingFlows

open torch.branching

private def noEventTimeDist : TimeDist :=
  { cdf := fun _ => 0.0,
    pdf := fun _ => 0.0,
    quantile := fun _ => 0.0 }

private def positiveIdentity (x : Float) : Float :=
  if x <= 0.0 then 0.0 else x

private def demoJsonl : String :=
  "{\"name\":\"water\",\"smiles\":\"O\",\"atoms\":[{\"label\":8,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.95,0.0,0.0]},{\"label\":1,\"coord\":[-0.24,0.93,0.0]}]}\n" ++
  "{\"name\":\"ammonia\",\"smiles\":\"N\",\"atoms\":[{\"label\":7,\"coord\":[0.0,0.0,0.0]},{\"label\":1,\"coord\":[0.94,0.0,0.0]},{\"label\":1,\"coord\":[-0.31,0.89,0.0]},{\"label\":1,\"coord\":[-0.31,-0.89,0.0]}]}\n"

private def exceptToIO {α : Type} (context : String) (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error e => throw (IO.userError s!"{context}: {e}")

private def hotLogits (vocabSize target : Nat) : Array Float :=
  (Array.replicate vocabSize (-8.0)).set! target 8.0

private def demoLabelSymbol : Nat → String :=
  labelSymbolFromArray #["X", "H", "X", "X", "X", "X", "C", "N", "O", "X"]

private def targetAtom (target : BranchingState MoleculeAtom) (i : Nat) : MoleculeAtom :=
  if target.state.isEmpty then
    default
  else
    target.state.getD (i % target.state.size) target.state[0]!

private def targetConditionedModel
    (vocabSize : Nat)
    (target : BranchingState MoleculeAtom)
    (splitLogit : Float)
    (t : Float)
    (state : BranchingState MoleculeAtom) :
    MoleculeModelPrediction :=
  let n := state.state.size
  let needsSplit := n < target.state.size && t < 0.75
  let gap := target.state.size - n
  let activeSplitLogit :=
    if gap <= 1 then
      min splitLogit 3.0
    else
      splitLogit
  let indices := Array.range n
  { coordTargets := indices.map (fun i => (targetAtom target i).coord)
    labelLogits := indices.map (fun i => hotLogits vocabSize (targetAtom target i).label)
    splitLogits := indices.map (fun i => if needsSplit && i == 0 then activeSplitLogit else -100.0)
    delLogits := Array.replicate n (-100.0) }

def runDemo (outputPrefix : String := "examples_branching_molecule") : IO Unit := do
  let vocabSize := 10
  let maskToken := 9
  let cfg := MoleculeBridgeConfig.qm9 vocabSize maskToken
  let records ← exceptToIO "parse demo QM9 JSONL" (parseQM9MoleculeJsonl demoJsonl)
  let states ← exceptToIO "convert demo QM9 states"
    (qm9RecordsToBranchingStates records { vocabSize? := some vocabSize, maskToken? := some maskToken })
  if states.isEmpty then
    throw (IO.userError "demo JSONL did not contain any molecules")
  let target := states[0]!
  writeMoleculeXYZ ⟨outputPrefix ++ "_target.xyz"⟩ target "preprocessed target molecule" demoLabelSymbol

  let (bridgeResult, rng) :=
    branchingBridge
      (fun cfg x0 x1 t0 t1 => MoleculeBridgeConfig.bridge cfg x0 x1 t0 t1)
      cfg
      (fun _root => MoleculeBridgeConfig.maskedAtom cfg)
      #[target]
      #[0.5]
      TimeDist.betaOneThreeHalves
      TimeDist.uniform
      (sequentialUniformPolicy MoleculeAtom)
      (MoleculeBridgeConfig.anchorMerge cfg)
      (lengthMins := .uniform 1)
      (deletionPad := 1.2)
      (x1Modifier := maskDeletedMoleculeLabels cfg)
      (rng := { state := 2026 })
  if bridgeResult.Xt.isEmpty then
    throw (IO.userError "bridge did not produce a state")
  let bridgeState := bridgeResult.Xt[0]!
  writeMoleculeXYZ ⟨outputPrefix ++ "_bridge.xyz"⟩ bridgeState "sampled bridge state at t=0.5" demoLabelSymbol

  let (x0Atom, rng) := MoleculeBridgeConfig.sampleInitialAtom cfg rng
  let x0 : BranchingState MoleculeAtom := BranchingState.mkDefault #[x0Atom] #[0]
  let flow : CoalescentFlow MoleculeBridgeConfig MoleculeAtom :=
    CoalescentFlow.mkDefault cfg TimeDist.betaOneThreeHalves noEventTimeDist positiveIdentity
  let model := targetConditionedModel vocabSize target 4.0
  let schedule := (Array.range 33).map (fun i => Float.ofNat i / 32.0)
  let (generated, _rng) :=
    moleculeBranchingGenerate flow x0 model schedule (rng := rng)
  for i in [:generated.trajectory.size] do
    let state := generated.trajectory[i]!
    let time := generated.times.getD i 0.0
    writeMoleculeXYZ ⟨s!"{outputPrefix}_step_{i}.xyz"⟩ state
      s!"branching trajectory step={i} t={time}" demoLabelSymbol
    IO.println s!"trajectory step={i} t={time} atoms={state.state.size}"
    if i > 0 then
      for event in generated.events.getD (i - 1) #[] do
        IO.println s!"  branch source_id={event.sourceId} splits={event.splitCount} deleted={event.deleted} interval=[{event.t0}, {event.t1}]"
  writeMoleculeXYZ ⟨outputPrefix ++ "_generated.xyz"⟩ generated.finalState
    "target-conditioned moleculeBranchingGenerate reference" demoLabelSymbol
  writeMoleculeTrajectoryJsonl ⟨outputPrefix ++ "_trajectory.jsonl"⟩ generated

  IO.println s!"target atoms: {target.state.size}"
  IO.println s!"bridge atoms: {bridgeState.state.size}"
  IO.println s!"generated atoms: {generated.finalState.state.size}"
  IO.println s!"wrote {outputPrefix}_target.xyz"
  IO.println s!"wrote {outputPrefix}_bridge.xyz"
  IO.println s!"wrote {outputPrefix}_generated.xyz"
  IO.println s!"wrote {outputPrefix}_trajectory.jsonl with stable runtime lineage"
  IO.println s!"wrote {generated.trajectory.size} branching trajectory frames"

def _root_.main (args : List String) : IO UInt32 := do
  let outputPrefix :=
    match args with
    | outPrefix :: _ => outPrefix
    | [] => "examples_branching_molecule"
  runDemo outputPrefix
  pure 0

end Examples.BranchingFlows

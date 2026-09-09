import LeanTest
import Tyr.Mctx

open torch.mctx

private def approx (a b : Float) (tol : Float := 1e-6) : Bool :=
  Float.abs (a - b) < tol

@[test]
def testMuZeroPolicyBatchedBandit : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[-1.0, 0.0, 2.0, 3.0],
      #[3.0, 1.0, 0.0, -1.0]
    ]
    value := #[0.0, 0.0]
    embedding := #[(), ()]
  }

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    ({
      reward := Array.replicate b 0.0
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 4 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let invalidActions : Array (Array Bool) := #[
    #[false, false, false, true],
    #[true, false, false, false]
  ]

  let out := muzeroPolicyBatched
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 1)
    (invalidActions := some invalidActions)
    (dirichletFraction := 0.0)

  LeanTest.assertEqual out.action.size 2 "Expected 2 batched actions"
  LeanTest.assertEqual (out.action.getD 0 0) 2 "Batch 0 should pick action 2"
  LeanTest.assertEqual (out.action.getD 1 0) 1 "Batch 1 should pick action 1"

@[test]
def testSearchBatchedSummaryShapes : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[0.0, 1.0, 2.0],
      #[2.0, 1.0, 0.0],
      #[0.5, 0.2, 0.1]
    ]
    value := #[0.0, 0.0, 0.0]
    embedding := #[(), (), ()]
  }

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    ({
      reward := Array.replicate b 0.0
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 3 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let rootFn : RootActionSelectionFn Unit Unit := fun _ tree nodeIndex =>
    muzeroActionSelection tree nodeIndex 0 qtransformByParentAndSiblings
  let interiorFn : InteriorActionSelectionFn Unit Unit := fun _ tree nodeIndex depth =>
    muzeroActionSelection tree nodeIndex depth qtransformByParentAndSiblings

  let tree := searchBatched
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrentFn)
    (rootActionSelectionFn := rootFn)
    (interiorActionSelectionFn := interiorFn)
    (numSimulations := 2)

  let summary := tree.summary
  LeanTest.assertEqual summary.value.size 3 "Summary should contain one value per batch element"
  LeanTest.assertEqual summary.visitCounts.size 3 "Visit counts should be batched"
  LeanTest.assertEqual (summary.visitCounts.getD 0 #[]).size 3 "Each row should have num_actions counts"

@[test]
def testGumbelPolicyBatchedMasking : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[0.0, -1.0, 2.0, 3.0],
      #[1.0, 2.0, 0.5, -3.0]
    ]
    value := #[-1.0, -1.0]
    embedding := #[(), ()]
  }

  let rewards : Array (Array Float) := #[
    #[20.0, 3.0, -1.0, 10.0],
    #[1.0, 4.0, 2.0, -5.0]
  ]

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    let reward := (List.range b).toArray.map fun i =>
      let a := actions.getD i 0
      (rewards.getD i #[]).getD a 0.0
    ({
      reward := reward
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 4 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let invalidActions : Array (Array Bool) := #[
    #[true, false, false, true],
    #[true, false, false, true]
  ]

  let out := gumbelMuZeroPolicyBatched
    (params := ())
    (rngKey := 7)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 12)
    (invalidActions := some invalidActions)

  let a0 := out.action.getD 0 0
  let a1 := out.action.getD 1 0
  LeanTest.assertTrue (a0 = 1 || a0 = 2) s!"Batch0 action should be valid, got {a0}"
  LeanTest.assertTrue (a1 = 1 || a1 = 2) s!"Batch1 action should be valid, got {a1}"

  let weights0 := out.actionWeights.getD 0 #[]
  let sum0 := weights0.foldl (init := 0.0) (· + ·)
  LeanTest.assertTrue (approx sum0 1.0 1e-5) s!"Action weights should sum to 1, got {sum0}"

@[test]
def testAlphaZeroPolicyBatchedPersistentSubtree : IO Unit := do
  let root1 : BatchedRootFnOutput Unit := {
    priorLogits := #[
      #[0.2, -0.1, 0.4, 0.0],
      #[-0.4, 0.6, 0.1, 0.2]
    ]
    value := #[0.1, -0.2]
    embedding := #[(), ()]
  }

  let rewards : Array (Array Float) := #[
    #[0.1, 0.0, -0.2, 0.3],
    #[-0.1, 0.4, 0.2, 0.0]
  ]

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    let reward := (List.range b).toArray.map fun i =>
      let a := actions.getD i 0
      (rewards.getD i #[]).getD a 0.0
    ({
      reward := reward
      discount := Array.replicate b 0.95
      priorLogits := Array.replicate b (Array.replicate 4 0.0)
      value := Array.replicate b 0.05
    }, Array.replicate b ())

  let out1 := alphazeroPolicyBatched
    (params := ())
    (rngKey := 3)
    (root := root1)
    (recurrentFn := recurrentFn)
    (numSimulations := 5)
    (searchTree := none)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let carried := getSubtreeBatched out1.searchTree out1.action
  let before0 := (carried.trees.getD 0 default).nodeVisits.getD ROOT_INDEX 0
  let before1 := (carried.trees.getD 1 default).nodeVisits.getD ROOT_INDEX 0

  let root2 : BatchedRootFnOutput Unit := {
    priorLogits := root1.priorLogits
    value := #[0.7, -0.8]
    embedding := #[(), ()]
  }

  let out2 := alphazeroPolicyBatched
    (params := ())
    (rngKey := 13)
    (root := root2)
    (recurrentFn := recurrentFn)
    (numSimulations := 2)
    (searchTree := some carried)
    (maxNodes := some 64)
    (dirichletFraction := 0.0)

  let tree0 := out2.searchTree.trees.getD 0 default
  let tree1 := out2.searchTree.trees.getD 1 default
  LeanTest.assertEqual (tree0.nodeVisits.getD ROOT_INDEX 0) (before0 + 2)
    "Batch 0 should continue search from carried subtree"
  LeanTest.assertEqual (tree1.nodeVisits.getD ROOT_INDEX 0) (before1 + 2)
    "Batch 1 should continue search from carried subtree"
  LeanTest.assertTrue (approx (tree0.rawValues.getD ROOT_INDEX 0.0) 0.7 1e-6)
    "Batch 0 root value should refresh from new root output"
  LeanTest.assertTrue (approx (tree1.rawValues.getD ROOT_INDEX 0.0) (-0.8) 1e-6)
    "Batch 1 root value should refresh from new root output"

@[test]
def testResetSearchTreeBatchedSelectMask : IO Unit := do
  let root : BatchedRootFnOutput Unit := {
    priorLogits := #[#[0.0, 1.0], #[1.0, 0.0]]
    value := #[0.0, 0.0]
    embedding := #[(), ()]
  }

  let recurrentFn : BatchedRecurrentFn Unit Unit := fun _params _rng actions _embeddings =>
    let b := actions.size
    ({
      reward := Array.replicate b 0.0
      discount := Array.replicate b 0.0
      priorLogits := Array.replicate b (Array.replicate 2 0.0)
      value := Array.replicate b 0.0
    }, Array.replicate b ())

  let out := muzeroPolicyBatched
    (params := ())
    (rngKey := 0)
    (root := root)
    (recurrentFn := recurrentFn)
    (numSimulations := 2)
    (dirichletFraction := 0.0)

  let reset := resetSearchTreeBatched out.searchTree (some #[true, false])
  let t0 := reset.trees.getD 0 default
  let t1 := reset.trees.getD 1 default

  LeanTest.assertTrue ((List.range t0.nodeVisits.size).all fun i => t0.nodeVisits.getD i 1 = 0)
    "Selected batch row should be reset"
  LeanTest.assertTrue (t1.nodeVisits.getD ROOT_INDEX 0 > 0)
    "Unselected batch row should be preserved"

/-! Batched exploration must use independent streams for identical rows. -/

private def explorationRoot (rows : Nat) : BatchedRootFnOutput Unit := {
  priorLogits := Array.replicate rows #[0.0, 0.0, 0.0, 0.0]
  value := Array.replicate rows 0.0
  embedding := Array.replicate rows ()
}

private def explorationRecurrent : BatchedRecurrentFn Unit Unit := fun _ _ actions _ =>
  ({ reward := Array.replicate actions.size 0.0, discount := Array.replicate actions.size 0.0,
     priorLogits := Array.replicate actions.size #[0.0, 0.0, 0.0, 0.0],
     value := Array.replicate actions.size 0.0 }, Array.replicate actions.size ())

private def explorationPolicy (alphaZero : Bool) (seed : UInt64) (simulations : Nat)
    (temperature fraction alpha : Float) (root : BatchedRootFnOutput Unit)
    (invalid : Option (Array (Array Bool)) := none) : BatchedPolicyOutput (BatchedTree Unit Unit) :=
  if alphaZero then
    alphazeroPolicyBatched () seed root explorationRecurrent simulations
      (invalidActions := invalid) (temperature := temperature)
      (dirichletFraction := fraction) (dirichletAlpha := alpha)
  else
    muzeroPolicyBatched () seed root explorationRecurrent simulations
      (invalidActions := invalid) (temperature := temperature)
      (dirichletFraction := fraction) (dirichletAlpha := alpha)

@[test] def testMctxBatchedExplorationStreamsAndReplay : IO Unit := do
  let root := explorationRoot 16
  for alphaZero in #[false, true] do
    let out := explorationPolicy alphaZero 0 16 1.0 0.0 0.3 root
    let replay := explorationPolicy alphaZero 0 16 1.0 0.0 0.3 root
    LeanTest.assertEqual out.action replay.action "same seed replays every batch row"
    LeanTest.assertEqual out.actionWeights replay.actionWeights "same seed replays batched weights"
    LeanTest.assertTrue (out.action.any (· != out.action[0]!))
      "identical batch rows must receive independent categorical draws"
    LeanTest.assertTrue ((explorationPolicy alphaZero 1 16 1.0 0.0 0.3 root).action != out.action)
      "adjacent small seeds change batched action draws"
    for temperature in #[0.0, -1.0] do
      let greedy := explorationPolicy alphaZero 0 16 temperature 0.0 0.3 root
      for row in [:16] do
        LeanTest.assertEqual greedy.action[row]! (argmax greedy.actionWeights[row]!)
          "nonpositive temperature is greedy in every row"

@[test] def testMctxBatchedExplorationZeroVisitsRespectMask : IO Unit := do
  let root := { explorationRoot 16 with priorLogits := Array.replicate 16 #[1e100, 0.0, 1e100, 1.0] }
  let invalid := some (Array.replicate 16 #[true, false, true, false])
  for alphaZero in #[false, true] do
    for temperature in #[0.0, 1.0, 2.0] do
      let out := explorationPolicy alphaZero 3 0 temperature 0.0 0.3 root invalid
      for row in [:16] do
        let prior := softmax out.searchTree.trees[row]!.childrenPriorLogits[ROOT_INDEX]!
        let expected := softmax #[0.0, 1.0]
        LeanTest.assertTrue (approx prior[1]! expected[0]! && approx prior[3]! expected[1]!)
          "masking a huge invalid prior preserves each row's legal prior ratio"
        LeanTest.assertTrue (out.action[row]! == 1 || out.action[row]! == 3)
          "zero-visit batched selection remains legal despite huge invalid priors"
        let weights := out.actionWeights[row]!
        LeanTest.assertEqual weights[0]! 0.0 "invalid action zero has exactly zero weight"
        LeanTest.assertEqual weights[2]! 0.0 "invalid action two has exactly zero weight"
        LeanTest.assertTrue (weights.all (fun w => w.isFinite && w >= 0.0)) "batched weights are finite probabilities"
        LeanTest.assertTrue (approx (sum weights) 1.0) "zero-visit legal row sums to one"

@[test] def testMctxBatchedExplorationDirichletAlpha : IO Unit := do
  let root := explorationRoot 32
  for alphaZero in #[false, true] do
    let priors := fun seed alpha =>
      (explorationPolicy alphaZero seed 0 0.0 1.0 alpha root).searchTree.trees.map fun tree =>
        softmax tree.childrenPriorLogits[ROOT_INDEX]!
    let draws := priors 0 0.3
    LeanTest.assertEqual draws (priors 0 0.3) "same seed replays batched Dirichlet noise"
    LeanTest.assertTrue (draws != priors 1 0.3) "nearby seeds change batched Dirichlet noise"
    LeanTest.assertTrue (draws[0]! != draws[1]!) "identical rows receive independent root noise"
    let concentration := fun (rows : Array (Array Float)) =>
      sum (rows.map fun row => sum (row.map fun p => p * p))
    LeanTest.assertTrue (concentration (priors 0 0.05) > concentration (priors 0 10.0) + 10.0)
      "Dirichlet alpha controls root prior spread independently across batch rows"

@[test] def testMctxBatchedAlphaZeroVisitTemperatureDistribution : IO Unit := do
  let root := explorationRoot 16
  let singleRoot : RootFnOutput Unit := { priorLogits := #[0.0, 0.0, 0.0, 0.0], value := 0.0, embedding := () }
  let base := instantiateTreeFromRootWithCapacity singleRoot 1 #[false, false, false, false] ()
  let carried : BatchedTree Unit Unit := {
    trees := Array.replicate 16 { base with childrenVisits := #[#[9, 1, 0, 0]], nodeVisits := #[10] }
  }
  let draw := fun seed temperature =>
    (alphazeroPolicyBatched () seed root explorationRecurrent 0
      (searchTree := some carried) (dirichletFraction := 0.0) (temperature := temperature)).action
  let mut cold := 0
  let mut hot := 0
  for seed in [:16] do
    let a := draw seed.toUInt64 1.0
    let b := draw seed.toUInt64 2.0
    for row in [:16] do
      LeanTest.assertTrue (a[row]! < 2 && b[row]! < 2) "unvisited actions have no sampling mass"
      if a[row]! == 0 then cold := cold + 1
      if b[row]! == 0 then hot := hot + 1
  LeanTest.assertTrue (cold > hot + 16 && cold >= 200 && hot >= 150 && hot <= 220)
    s!"batched temperature changes visit sampling, got T1={cold}/256 T2={hot}/256"

@[test] def testMctxBatchedGumbelIndependentRows : IO Unit := do
  let root := explorationRoot 16
  let run := fun seed => gumbelMuZeroPolicyBatched () seed root explorationRecurrent 0
  let out := run 0
  let draws := fun seed => (run seed).searchTree.trees.map (·.extraData.rootGumbel)
  LeanTest.assertEqual (draws 0) (draws 0) "same seed replays batched Gumbel noise"
  LeanTest.assertTrue (draws 0 != draws 1) "adjacent seeds change batched Gumbel noise"
  LeanTest.assertTrue ((draws 0)[0]! != (draws 0)[1]!) "batch rows use distinct Gumbel streams"
  LeanTest.assertTrue (out.action.any (· != out.action[0]!)) "Gumbel action draws differ across identical rows"

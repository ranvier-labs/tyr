import LeanTest
import Tyr.Mctx
import Tyr.MctxDag

namespace Tests.MctxSelection

open torch.mctx

@[test] def testMctxMaskedArgmaxExtremeScores : IO Unit := do
  LeanTest.assertEqual (maskedArgmax #[0.0, -1e100] (some #[true, false])) 1
    "A finite masking penalty must not admit an invalid action"
  LeanTest.assertEqual (maskedArgmax #[0.0, -1.0 / 0.0, -1.0 / 0.0] (some #[true, false, false])) 1
    "Negative infinity ties must select the first legal action"
  LeanTest.assertEqual (maskedArgmax #[0.0 / 0.0, -2.0, -1.0] (some #[true])) 2
    "Masked NaN scores are excluded and missing mask entries remain legal"
  LeanTest.assertEqual (maskedArgmax #[] none) 0 "Empty sentinel remains zero"
  LeanTest.assertEqual (maskedArgmax #[1.0, 2.0] (some #[true, true])) 0
    "All-invalid sentinel remains zero"

@[test] def testMctxSearchNeverExpandsMaskedExtremeScore : IO Unit := do
  let root : RootFnOutput Unit := { priorLogits := #[0.0, 0.0], value := 0.0, embedding := () }
  let step : RecurrentFn Unit Unit := fun _ _ _ _ =>
    ({ reward := 7.0, discount := 0.0, priorLogits := #[0.0, 0.0], value := 0.0 }, ())
  let out := muzeroPolicy () 0 root step 4 (maxDepth := some 1)
    (invalidActions := some #[true, false]) (dirichletFraction := 0.0)
    (qtransform := fun _ _ => #[-1e100, -1e100])
  LeanTest.assertEqual out.searchTree.summary.visitCounts #[0, 4]
    "The rollout itself, not only final action sampling, must respect legality"
  let dagRoot : torch.mctxdag.RootFnOutput Nat :=
    { priorLogits := root.priorLogits, value := 0.0, embedding := 0 }
  let dagStep : torch.mctxdag.RecurrentFn Unit Nat := fun _ _ action state =>
    ({ reward := 7.0, discount := 0.0, priorLogits := #[0.0, 0.0], value := 0.0 }, state + action + 1)
  let dag := torch.mctxdag.muzeroPolicyDag () 0 dagRoot dagStep id 4 (maxDepth := some 1)
    (invalidActions := some #[true, false]) (dirichletFraction := 0.0)
    (qtransform := fun _ _ => #[-1e100, -1e100])
  LeanTest.assertEqual dag.searchTree.summary.visitCounts #[0, 4]
    "DAG rollouts must also exclude invalid scores"
  let batchedRoot : BatchedRootFnOutput Unit :=
    { priorLogits := #[root.priorLogits], value := #[0.0], embedding := #[()] }
  let batchedStep : BatchedRecurrentFn Unit Unit := fun _ _ actions _ =>
    ({ reward := Array.replicate actions.size 7.0, discount := Array.replicate actions.size 0.0,
       priorLogits := Array.replicate actions.size #[0.0, 0.0], value := Array.replicate actions.size 0.0 },
     Array.replicate actions.size ())
  let batched := muzeroPolicyBatched () 0 batchedRoot batchedStep 4 (maxDepth := some 1)
    (invalidActions := some #[#[true, false]]) (dirichletFraction := 0.0)
    (qtransform := fun _ _ => #[-1e100, -1e100])
  LeanTest.assertEqual batched.searchTree.summary.visitCounts #[#[0, 4]]
    "Batched rollouts must also exclude invalid scores"

@[test] def testMctxCachedSchedulesMatchOriginalTables : IO Unit := do
  for budget in #[0, 1, 7, 32, 65] do
    let table := getTableOfConsideredVisits 16 budget
    for considered in [:17] do
      let cached := some (ConsideredVisitSchedule.create considered budget)
      for simulation in [:budget + 1] do
        let expected := (table.getD considered #[]).getD simulation 0
        LeanTest.assertEqual (consideredVisitAt cached considered budget simulation) expected
          "Cached schedule must preserve the original table's selection rounds"
        LeanTest.assertEqual (consideredVisitAt none considered budget simulation) expected
          "Uncached direct-selector calls must preserve selection rounds"

@[test] def testMctxCachedScheduleInvalidation : IO Unit := do
  let stale := some (ConsideredVisitSchedule.create 16 32)
  for (considered, budget) in #[(2, 32), (16, 8), (1, 7)] do
    for simulation in [:budget] do
      LeanTest.assertEqual (consideredVisitAt stale considered budget simulation)
        (consideredVisitAt none considered budget simulation)
        "Changed legal-action counts or budgets invalidate cached schedules"
  LeanTest.assertEqual (countConsideredActions 16 4 #[true, false]) 3
    "Short masks must count missing entries as legal"

end Tests.MctxSelection

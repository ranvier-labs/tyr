import LeanTest
import Tyr.Mctx.Math
import Tyr.Mctx.Sampling

namespace Tests.MctxSampling

open torch.mctx

private def assertApprox (actual expected tolerance : Float) (label : String) : IO Unit :=
  LeanTest.assertTrue (actual.isFinite && Float.abs (actual - expected) ≤ tolerance)
    s!"{label}: expected {expected} ± {tolerance}, got {actual}"

private def assertSimplex (values : Array Float) (size : Nat) (label : String) : IO Unit := do
  LeanTest.assertEqual values.size size s!"{label}: dimension"
  for value in values do
    LeanTest.assertTrue (value.isFinite && value ≥ 0.0 && value ≤ 1.0)
      s!"{label}: invalid probability {value}"
  assertApprox (values.foldl (· + ·) 0.0) 1.0 1e-12 s!"{label}: total mass"

private def assertFrequencies (counts : Array Nat) (draws : Nat)
    (expected : Array Float) (label : String) : IO Unit := do
  LeanTest.assertEqual (counts.foldl (· + ·) 0) draws s!"{label}: all draws counted"
  LeanTest.assertEqual counts.size expected.size s!"{label}: dimension"
  for i in [:expected.size] do
    let observed := Float.ofNat counts[i]! / Float.ofNat draws
    -- Fixed seeds make this deterministic. The bound is deliberately several
    -- standard errors wide, while distinguishing all tested distributions.
    assertApprox observed expected[i]! 0.03 s!"{label}: frequency of action {i}"

@[test] def testMctxSamplingKeyReproducibility : IO Unit := do
  for seed in #[0, 1, 17, 123456789] do
    let key := UInt64.ofNat seed
    LeanTest.assertEqual (Sampling.splitKey key 3) (Sampling.splitKey key 3)
      "Splitting a seed must be reproducible"
    LeanTest.assertTrue (Sampling.splitKey key 3 != Sampling.splitKey key 4)
      "Different streams must not share the same key"
    LeanTest.assertEqual (Sampling.dirichlet key 5 0.3) (Sampling.dirichlet key 5 0.3)
      "Dirichlet draws must be reproducible"
    LeanTest.assertEqual (Sampling.gumbel key 5 1.0) (Sampling.gumbel key 5 1.0)
      "Gumbel draws must be reproducible"
    LeanTest.assertEqual (Sampling.sampleAction key #[1.0, 3.0] 1.0 none)
      (Sampling.sampleAction key #[1.0, 3.0] 1.0 none)
      "Categorical draws must be reproducible"

@[test] def testMctxDirichletSimplexAndExtremeConcentrations : IO Unit := do
  LeanTest.assertEqual (Sampling.dirichlet 0 0 0.3) #[] "Empty Dirichlet draw"
  for alpha in #[(1e-8 : Float), 0.2, 1.0, 10.0, 1e8] do
    for seed in [:32] do
      let key := UInt64.ofNat seed
      LeanTest.assertEqual (Sampling.dirichlet key 1 alpha) #[1.0]
        "One-dimensional Dirichlet is a point mass"
      assertSimplex (Sampling.dirichlet key 5 alpha) 5 s!"Dirichlet alpha={alpha} seed={seed}"

@[test] def testMctxDirichletMomentsAndSequentialSeeds : IO Unit := do
  let draws := 2048
  for alpha in #[(0.2 : Float), 3.0] do
    let mut sums := Array.replicate 3 (0.0 : Float)
    let mut squares := Array.replicate 3 (0.0 : Float)
    let mut lastFirst := 0.0
    let mut successiveChanges := 0
    for seed in [:draws] do
      let values := Sampling.dirichlet (UInt64.ofNat seed) 3 alpha
      assertSimplex values 3 s!"Dirichlet moment sample alpha={alpha}"
      for i in [:3] do
        sums := sums.set! i (sums[i]! + values[i]!)
        squares := squares.set! i (squares[i]! + values[i]! * values[i]!)
      if seed > 0 && Float.abs (values[0]! - lastFirst) > 0.01 then
        successiveChanges := successiveChanges + 1
      lastFirst := values[0]!
    -- For symmetric Dirichlet(alpha, alpha, alpha), E[X_i] = 1/3
    -- and Var[X_i] = 2 / (9 * (3*alpha + 1)). In particular these
    -- concentrations must produce very different dispersion.
    let expectedVariance := 2.0 / (9.0 * (3.0 * alpha + 1.0))
    for i in [:3] do
      let mean := sums[i]! / Float.ofNat draws
      let variance := squares[i]! / Float.ofNat draws - mean * mean
      assertApprox mean (1.0 / 3.0) 0.04 s!"Dirichlet alpha={alpha} mean {i}"
      assertApprox variance expectedVariance 0.02 s!"Dirichlet alpha={alpha} variance {i}"
    LeanTest.assertTrue (successiveChanges > draws / 2)
      "Consecutive small integer seeds must change draws substantially, not by floating-point roundoff"

@[test] def testMctxCategoricalTemperatureFrequencies : IO Unit := do
  let draws := 4096
  -- The zero-weight action must stay impossible, even at high temperature.
  for (temperature, firstProb) in #[(0.5, 0.1), (1.0, 0.25), (2.0, 1.0 / (1.0 + Float.sqrt 3.0))] do
    let mut counts := Array.replicate 3 0
    for seed in [:draws] do
      let action := Sampling.sampleAction (UInt64.ofNat seed) #[1.0, 3.0, 0.0] temperature none
      LeanTest.assertTrue (action < 2) "A zero-weight action must never be sampled"
      counts := counts.set! action (counts[action]! + 1)
    assertFrequencies counts draws #[firstProb, 1.0 - firstProb, 0.0]
      s!"Categorical temperature={temperature}"

@[test] def testMctxCategoricalMasksAndZeroVisitFallback : IO Unit := do
  let invalid := some #[false, true, false, true]
  let weights := Sampling.normalizeWeights #[1.0, 100.0, 3.0, 100.0] invalid
  LeanTest.assertEqual weights #[0.25, 0.0, 0.75, 0.0]
    "Masked weights must be removed before normalization"
  let fallback := Sampling.normalizeWeights #[0.0, 0.0, 0.0, 0.0] invalid
  LeanTest.assertEqual fallback #[0.5, 0.0, 0.5, 0.0]
    "A zero-visit root must distribute fallback mass only across valid actions"
  LeanTest.assertEqual (Sampling.normalizeWeights #[1e308, 1e308] none) #[0.5, 0.5]
    "Normalization must avoid overflowing the sum of finite weights"
  LeanTest.assertEqual
    (Sampling.normalizeWeights #[0.0 / 0.0, -1.0, 2.0, 1.0 / 0.0] none) #[0.0, 0.0, 1.0, 0.0]
    "Nonfinite and negative weights must not acquire probability mass"
  let draws := 2048
  let mut fallbackCounts := Array.replicate 4 0
  let mut weightedCounts := Array.replicate 4 0
  for seed in [:draws] do
    let key := UInt64.ofNat seed
    let fallbackAction := Sampling.sampleAction key #[0.0, 0.0, 0.0, 0.0] 1.0 invalid
    let weightedAction := Sampling.sampleAction key #[1.0, 100.0, 3.0, 100.0] 1.0 invalid
    LeanTest.assertTrue (fallbackAction == 0 || fallbackAction == 2)
      "Zero-visit fallback must never select an invalid action"
    LeanTest.assertTrue (weightedAction == 0 || weightedAction == 2)
      "Sampling must never select an invalid action with positive weight"
    fallbackCounts := fallbackCounts.set! fallbackAction (fallbackCounts[fallbackAction]! + 1)
    weightedCounts := weightedCounts.set! weightedAction (weightedCounts[weightedAction]! + 1)
  assertFrequencies fallbackCounts draws #[0.5, 0.0, 0.5, 0.0] "Zero-visit fallback"
  assertFrequencies weightedCounts draws #[0.25, 0.0, 0.75, 0.0] "Masked categorical"

@[test] def testMctxCategoricalDeterministicAndExtremeTemperatures : IO Unit := do
  for seed in [:128] do
    let key := UInt64.ofNat seed
    for temperature in #[(0.0 : Float), -1.0, 1e-300] do
      LeanTest.assertEqual
        (Sampling.sampleAction key #[1.0, 100.0, 3.0] temperature (some #[false, true, false])) 2
        "Nonpositive or vanishing temperature must select the valid maximum"
    LeanTest.assertEqual (Sampling.sampleAction key #[0.0, 3.0, 0.0] 1e300 none) 1
      "High temperature must preserve exact zero mass"
    LeanTest.assertEqual (Sampling.sampleAction key #[0.0, 0.0, 0.0] 0.0 (some #[true, false, false])) 1
      "Zero-temperature fallback must choose a valid action deterministically"

@[test] def testMctxSamplingDegenerateInputFallbacks : IO Unit := do
  LeanTest.assertEqual (Sampling.normalizeWeights #[] none) #[] "Empty normalization"
  LeanTest.assertEqual (Sampling.sampleAction 0 #[] 1.0 none) 0 "Empty action sentinel"
  LeanTest.assertEqual (Sampling.rootLogits 0 #[] none 0.25 0.3) #[] "Empty root logits"
  let allInvalid := some #[true, true, true]
  let uniform := Array.replicate 3 (1.0 / 3.0 : Float)
  LeanTest.assertEqual (Sampling.normalizeWeights #[1.0, 100.0, 3.0] allInvalid) uniform
    "All-invalid roots retain the documented uniform fallback"
  LeanTest.assertEqual (Sampling.sampleAction 123 #[1.0, 100.0, 3.0] 0.0 allInvalid) 0
    "All-invalid zero-temperature fallback uses deterministic sentinel zero"
  let invalidRoot := Sampling.rootLogits 123 #[0.0, 1.0, 2.0] allInvalid 0.25 0.3
  LeanTest.assertTrue (invalidRoot.all Float.isFinite)
    "All-invalid root fallback must avoid an all-infinite softmax"
  LeanTest.assertEqual (softmax invalidRoot) uniform "All-invalid root probabilities"
  let logits := #[0.0, Float.log 3.0]
  let clean := Sampling.rootLogits 123 logits none 0.0 0.3
  for alpha in #[(0.0 : Float), -1.0, 0.0 / 0.0, 1.0 / 0.0] do
    LeanTest.assertEqual (Sampling.dirichlet 123 3 alpha) uniform
      "Invalid Dirichlet concentration falls back to a uniform simplex"
    LeanTest.assertEqual (Sampling.rootLogits 123 logits none 1.0 alpha) clean
      "Invalid root concentration must disable noise"
  LeanTest.assertEqual (Sampling.rootLogits 123 logits none (0.0 / 0.0) 0.3) clean
    "NaN noise fraction must disable noise"
  LeanTest.assertEqual (Sampling.rootLogits 123 logits none (-1.0) 0.3) clean
    "Negative noise fraction clamps to zero"
  LeanTest.assertEqual (Sampling.rootLogits 123 logits none 2.0 0.3)
    (Sampling.rootLogits 123 logits none 1.0 0.3)
    "Noise fraction above one clamps to one"
  LeanTest.assertEqual (Sampling.sampleAction 123 #[1.0, 3.0, 0.0] (0.0 / 0.0) none) 1
    "NaN temperature uses the greedy fallback"
  let draws := 2048
  let mut counts := Array.replicate 3 0
  for seed in [:draws] do
    let action := Sampling.sampleAction (UInt64.ofNat seed) #[1.0, 3.0, 0.0] (1.0 / 0.0) none
    LeanTest.assertTrue (action < 2) "Infinite temperature must preserve zero mass"
    counts := counts.set! action (counts[action]! + 1)
  assertFrequencies counts draws #[0.5, 0.5, 0.0] "Infinite-temperature positive support"

@[test] def testMctxTemperaturePreservesExtremePositiveSupport : IO Unit := do
  let draws := 2048
  -- Normalizing before log tempering would underflow the first positive
  -- weight to zero and incorrectly discard it even at infinite temperature.
  for temperature in #[(1e300 : Float), 1.0 / 0.0] do
    let mut counts := Array.replicate 3 0
    for seed in [:draws] do
      let action := Sampling.sampleAction (UInt64.ofNat seed) #[1e-300, 1e300, 0.0] temperature none
      LeanTest.assertTrue (action < 2) "Tempering must preserve the original positive support"
      counts := counts.set! action (counts[action]! + 1)
    assertFrequencies counts draws #[0.5, 0.5, 0.0] s!"Extreme weights at temperature={temperature}"

@[test] def testMctxGumbelMomentsAndCategoricalFrequencies : IO Unit := do
  let draws := 4096
  let logits := #[0.0, Float.log 2.0, Float.log 5.0]
  let mut counts := Array.replicate 3 0
  let mut sum := 0.0
  let mut sumSquares := 0.0
  for seed in [:draws] do
    let noise := Sampling.gumbel (UInt64.ofNat seed) 3 1.0
    LeanTest.assertEqual noise.size 3 "Gumbel dimension"
    LeanTest.assertTrue (noise.all Float.isFinite) "Gumbel draws must be finite"
    let first := noise[0]!
    sum := sum + first
    sumSquares := sumSquares + first * first
    let perturbed := (List.range 3).toArray.map fun i => logits[i]! + noise[i]!
    let action := argmax perturbed
    counts := counts.set! action (counts[action]! + 1)
  let mean := sum / Float.ofNat draws
  let variance := sumSquares / Float.ofNat draws - mean * mean
  assertApprox mean 0.5772156649 0.08 "Standard Gumbel mean"
  assertApprox variance 1.6449340668 0.25 "Standard Gumbel variance"
  assertFrequencies counts draws #[0.125, 0.25, 0.625] "Gumbel-max categorical"
  LeanTest.assertEqual (Sampling.gumbel 0 0 1.0) #[] "Empty Gumbel draw"
  LeanTest.assertEqual (Sampling.gumbel 123 4 0.0) #[0.0, 0.0, 0.0, 0.0]
    "Zero Gumbel scale must disable noise"
  let unit := Sampling.gumbel 123 4 1.0
  let scaled := Sampling.gumbel 123 4 2.0
  for i in [:4] do
    assertApprox scaled[i]! (2.0 * unit[i]!) 1e-12 "Gumbel noise scale"

@[test] def testMctxRootNoiseMixingAndMasks : IO Unit := do
  let logits := #[0.0, Float.log 3.0, Float.log 7.0]
  let invalid := some #[false, true, false]
  let clean := softmax (Sampling.rootLogits 17 logits invalid 0.0 0.3)
  assertApprox clean[0]! 0.125 1e-12 "Noise-free masked root first probability"
  assertApprox clean[2]! 0.875 1e-12 "Noise-free masked root last probability"
  LeanTest.assertEqual clean[1]! 0.0 "Invalid root action must have zero probability"
  let noise := softmax (Sampling.rootLogits 17 logits invalid 1.0 0.3)
  let changedPriors := softmax (Sampling.rootLogits 17 #[10.0, -20.0, -10.0] invalid 1.0 0.3)
  let mixed := softmax (Sampling.rootLogits 17 logits invalid 0.25 0.3)
  assertSimplex noise 3 "Root Dirichlet noise"
  assertSimplex mixed 3 "Mixed root probabilities"
  for i in [:3] do
    assertApprox changedPriors[i]! noise[i]! 1e-12 "Pure noise must ignore model priors"
    assertApprox mixed[i]! (0.75 * clean[i]! + 0.25 * noise[i]!) 1e-12 "Root noise mixture"
  LeanTest.assertEqual noise[1]! 0.0 "Root noise must not revive invalid actions"
  LeanTest.assertEqual mixed[1]! 0.0 "Mixing must not revive invalid actions"
  let concentrated := softmax (Sampling.rootLogits 17 logits invalid 1.0 0.01)
  let diffuse := softmax (Sampling.rootLogits 17 logits invalid 1.0 100.0)
  LeanTest.assertTrue (Float.abs (concentrated[0]! - diffuse[0]!) > 0.05)
    "The root noise alpha argument must affect exploration"

def run : IO Unit := do
  testMctxSamplingKeyReproducibility
  testMctxDirichletSimplexAndExtremeConcentrations
  testMctxDirichletMomentsAndSequentialSeeds
  testMctxCategoricalTemperatureFrequencies
  testMctxCategoricalMasksAndZeroVisitFallback
  testMctxCategoricalDeterministicAndExtremeTemperatures
  testMctxSamplingDegenerateInputFallbacks
  testMctxTemperaturePreservesExtremePositiveSupport
  testMctxGumbelMomentsAndCategoricalFrequencies
  testMctxRootNoiseMixingAndMasks

end Tests.MctxSampling

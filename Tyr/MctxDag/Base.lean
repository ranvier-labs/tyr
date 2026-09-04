import Tyr.Mctx.Base
import Tyr.Mctx.Math

/-!
# Tyr.MctxDag.Base

Shared aliases re-exporting the `torch.mctx` core types (actions, model
I/O, summaries, recurrent functions) for the DAG search namespace.
-/

namespace torch.mctxdag

abbrev Action := torch.mctx.Action
abbrev NodeIndex := torch.mctx.NodeIndex
abbrev Depth := torch.mctx.Depth

abbrev RecurrentFnOutput := torch.mctx.RecurrentFnOutput
abbrev RootFnOutput (S : Type) := torch.mctx.RootFnOutput S
abbrev PolicyOutput (TreeType : Type) := torch.mctx.PolicyOutput TreeType
abbrev SearchSummary := torch.mctx.SearchSummary

abbrev RecurrentFn (P S : Type) := torch.mctx.RecurrentFn P S

abbrev sum := torch.mctx.sum
abbrev maxD := torch.mctx.maxD
abbrev argmax := torch.mctx.argmax
abbrev maskedArgmax := torch.mctx.maskedArgmax
abbrev softmax := torch.mctx.softmax
abbrev logSafe := torch.mctx.logSafe

end torch.mctxdag

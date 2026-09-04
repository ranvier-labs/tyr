import Tyr.SafeTensors

open torch

/-
The command reads the checkpoint header while Lean elaborates this file and
generates typed metadata and loaders. No handwritten model schema is involved.
-/
safetensors_type_provider "Tests/fixtures/safetensors/indexed_dir" as DemoWeights

#check DemoWeights.load_embed_weightTyped
#check DemoWeights.load_proj_biasTyped
#check DemoWeights.loadAll
#check DemoWeights.embed_weightTensorSpec
#check DemoWeights.proj_biasTensorSpec

/-
  Tyr/Model/Qwen25Omni/Pretrained.lean

  Resolve/load Qwen2.5-Omni thinker text checkpoints from either:
  - local model directory, or
  - HuggingFace repo id (download to local cache on demand).
-/
import Tyr.Hub
import Tyr.Model.Qwen25Omni.ConfigIO
import Tyr.Model.Qwen25Omni.Weights

/-! # Qwen2.5-Omni pretrained loading

Covered repo ids (`torch.qwen25omni.hub.qwen25OmniCollectionRepoIds`), tokenizer
files, and `torch.qwen25omni.Qwen25OmniForCausalLM.loadFromPretrained` from a
local directory or repo id.
-/

namespace torch.qwen25omni.hub

open torch.Hub

/-- Qwen2.5-Omni model ids (HF collection `Qwen/qwen25-omni`). -/
def qwen25OmniCollectionRepoIds : Array String := #[
  "Qwen/Qwen2.5-Omni-3B",
  "Qwen/Qwen2.5-Omni-7B",
  "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
  "Qwen/Qwen2.5-Omni-7B-AWQ"
]

def isQwen25OmniCollectionRepoId (repoId : String) : Bool :=
  qwen25OmniCollectionRepoIds.contains repoId

/-- Qwen2.5-Omni tokenizer files to download. -/
def tokenizerFiles : Array String := #[
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json",
  "merges.txt"
]

export torch.Hub (resolvePretrainedDir detectWeightLayout findCachedSnapshot? shardFilesFromIndexFile)

end torch.qwen25omni.hub

namespace torch.qwen25omni

namespace Qwen25OmniForCausalLM

/-- Load Qwen2.5-Omni thinker text checkpoint from local dir or HF repo id. -/
def loadFromPretrained
    (source : String)
    (defaults : Config := Config.qwen25omni_3B)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir)
    : IO (Sigma (fun cfg => Qwen25OmniForCausalLM cfg)) := do
  Hub.loadModelFromPretrained
    source
    revision
    cacheDir
    hub.tokenizerFiles
    (fun modelDir cfg => Config.loadFromPretrainedDir modelDir cfg)
    defaults
    (fun modelDir cfg => Qwen25OmniForCausalLM.loadSharded modelDir cfg)
    (fun path cfg => Qwen25OmniForCausalLM.load path cfg)

end Qwen25OmniForCausalLM

end torch.qwen25omni

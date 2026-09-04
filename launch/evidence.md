# Selected evidence

The launch examples cross several assurance boundaries. Static checks are
distinguished from runtime integration, numerical comparison, and
hardware-specific measurements.

| Example | Reproduction or artifact | What it establishes |
|---|---|---|
| Static tensor contract | `./scripts/launch/run_shape_safety.sh` | A valid projection elaborates and an incompatible shape is rejected before numerical execution. |
| Hybrid contact adjoint | `lake exe RunUrdfContactExample` and `generated/contact/trajectory.csv` | The exported trajectory retains both event sides; the reverse event computes the saltation update and restitution derivative. |
| SafeTensors type provider | `./scripts/launch/run_safetensors_schema.sh` | Typed declarations are generated from a fixture checkpoint header during elaboration. |
| GPU language | `lake exe TestGPUDSL` | Forty-nine compiler and code-generation tests cover capability constraints and kernel structure. This local path uses CPU stubs and is not native CUDA execution. |
| Native CUDA parity | `generated/gpu/cuda-parity-gb10.txt` and companion manifest | Five generated modules passed ten numerical reference comparisons on the named device and software configuration in the manifest. |
| Model runtime integration | `generated/model-inference/qwen36-27b-gb10.txt` | Tyr loaded a Qwen3.6 27B checkpoint and generated 32 tokens through `Device.CUDA 0`; this is not a quality or performance result. |

These measurements describe the paths that were exercised. In particular, the
named CUDA device is metadata for one execution and does not define the scope
of Tyr's general tensor device interface.

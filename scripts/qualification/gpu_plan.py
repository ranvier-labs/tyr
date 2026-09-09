#!/usr/bin/env python3
"""Architecture-specific build inputs and the production decode route exercised."""
import argparse
import json


def configuration(gpu):
    if gpu == "H100":
        names = ["Copy", "Rotary", "FusedLayerNorm", "FusedRMSNorm", "MhaH100", "MhaH100Decode"]
        return {"gpu": gpu, "family": "HOPPER", "runner": "TestGPUE2E",
                "modules": ["Tyr.GPU.Kernels." + name for name in names],
                "decode_route": "hopper_custom_kernel_when_eligible"}
    if gpu in ("GB10", "B200", "B300"):
        names = ["MhaGB10", "FusedRMSNorm", "FusedLayerNorm", "RKCombine", "BrownianSample"]
        # GB10 cannot compile WGMMA/tcgen05. The production flash-attention
        # dispatcher also deliberately uses SDPA on B200/B300 (major != 9).
        # Keep RunMhaH100Decode as a generic production decode/cache parity
        # test, but do not claim that it exercises a Hopper custom kernel.
        return {"gpu": gpu, "family": "BLACKWELL", "runner": "TestGPUGB10E2E",
                "modules": ["Tyr.GPU.Kernels." + name for name in names],
                "decode_route": "sdpa_fallback"}
    raise ValueError(f"No strict qualification plan for GPU={gpu}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu")
    parser.add_argument("--field", choices=("family", "runner", "modules", "decode_route"))
    args = parser.parse_args()
    plan = configuration(args.gpu)
    if args.field:
        value = plan[args.field]
        print("\n".join(value) if isinstance(value, list) else value)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()

import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.Linear
import Tyr.Module.LayerNorm
import Tyr.Module.Affine
import Tyr.Module.Conv2d
import Tyr.Module.RMSNorm

/-!
# Tyr.Module

`Tyr.Module` is the umbrella import for Tyr's core neural-module layer.
It re-exports base module interfaces, deriving support, and common layer implementations.

## Major Components

- `Tyr.Module.Core`: base module abstractions and core contracts.
- `Tyr.Module.Derive`: deriving support for module/tensor-structure ergonomics.
- Common layers: `Linear`, `LayerNorm`, `Affine`, `Conv2d`, and `RMSNorm`.

## Scope

Use this module as the standard entrypoint for building model layers in Tyr.
Specialized layer families may live in additional submodules.
-/

import Tyr.Basic
import Tyr.Torch

namespace torch

structure Conv2d where
  in_channels : UInt64
  out_channels : UInt64
  kernel_size : UInt64 × UInt64
  stride : UInt64 × UInt64 := (1,1)
  padding : UInt64 × UInt64 := (0,0)
  dilation : UInt64 × UInt64 := (1,1)
  groups : UInt64 := 1
  bias : Bool := true

  weight : T #[out_channels, in_channels/groups, kernel_size.fst, kernel_size.snd] := torch.zeros #[out_channels, in_channels/groups, kernel_size.fst, kernel_size.snd]
  outputBias : T #[out_channels] := torch.zeros #[out_channels]


def Conv2d.init (c : Conv2d) : IO Conv2d := do
  let k := (Float.ofNat (c.groups.toNat) / Float.ofNat (c.in_channels * c.kernel_size.fst * c.kernel_size.snd).toNat).sqrt;
  let weight ← uniform c.weight.shape (min := -k) (max := k);
  return { c with weight := weight }

def Conv2d.outShape (c : Conv2d) (s : Shape) : Shape :=
  let (n, _, hin, win) := (s[0]!, s[1]!, s[2]!, s[3]!);
  let (px,py) := c.padding;
  let (sx,sy) := c.stride;
  let (dx,dy) := c.dilation;
  let (kx,ky) := c.kernel_size;
  let hout := (hin + 2 * px - dx * (kx-1) - 1)/sx + 1;
  let wout := (win + 2 * py - dy * (ky-1) - 1)/sy + 1;
  #[n, c.out_channels, hout, wout]



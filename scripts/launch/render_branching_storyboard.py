#!/usr/bin/env python3
"""Render exported BranchingFlows trajectory frames as a single SVG storyboard."""

from __future__ import annotations

import glob
import math
import pathlib
import re
import sys

from render_xyz import COLORS, read_xyz


def frame_index(path: str) -> int:
    match = re.search(r"_step_(\d+)\.xyz$", path)
    return int(match.group(1)) if match else 0


def miniature(path: pathlib.Path, x0: float, y0: float, width: float, height: float) -> list[str]:
    atoms = read_xyz(path)
    coords = []
    for symbol, x, y, z in atoms:
        angle = math.radians(28)
        px = x * math.cos(angle) - y * math.sin(angle)
        py = x * math.sin(angle) + y * math.cos(angle) - z * 0.55
        coords.append((symbol, px, py, x, y, z))
    xs = [a[1] for a in coords]
    ys = [a[2] for a in coords]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    scale = min(width, height) * 0.52 / span
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    placed = [(s, x0 + width / 2 + (px - cx) * scale, y0 + height / 2 + (py - cy) * scale, ox, oy, oz) for s, px, py, ox, oy, oz in coords]
    out = []
    for i, atom in enumerate(placed):
        for j in range(i + 1, len(placed)):
            other = placed[j]
            distance = math.sqrt((atom[3] - other[3]) ** 2 + (atom[4] - other[4]) ** 2 + (atom[5] - other[5]) ** 2)
            if distance < 1.65 and not (atom[0] == "H" and other[0] == "H"):
                out.append(f'<line x1="{atom[1]:.1f}" y1="{atom[2]:.1f}" x2="{other[1]:.1f}" y2="{other[2]:.1f}" stroke="#64748b" stroke-width="10" stroke-linecap="round"/>')
    for symbol, x, y, *_ in placed:
        radius = 20 if symbol == "H" else 27
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{COLORS.get(symbol, "#f59e0b")}" stroke="#e2e8f0" stroke-width="3"/>')
        out.append(f'<text x="{x:.1f}" y="{y + 6:.1f}" fill="#020617" font-family="Inter,Helvetica,sans-serif" font-size="15" font-weight="800" text-anchor="middle">{symbol}</text>')
    return out


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_branching_storyboard.py step-prefix output.svg")
    paths = [pathlib.Path(p) for p in sorted(glob.glob(sys.argv[1] + "_step_*.xyz"), key=frame_index)]
    if not paths:
        raise SystemExit("no trajectory frames found")
    # Long cosine schedules often contain many no-op states. Keep the first
    # state, atom-count changes, and the final state; then downsample change
    # points for a readable launch graphic.
    interesting = [paths[0]]
    previous_count = len(read_xyz(paths[0]))
    for path in paths[1:-1]:
        count = len(read_xyz(path))
        if count != previous_count:
            interesting.append(path)
            previous_count = count
    if paths[-1] != interesting[-1]:
        interesting.append(paths[-1])
    if len(interesting) > 8:
        last = len(interesting) - 1
        indices = sorted({round(i * last / 7) for i in range(8)})
        paths = [interesting[i] for i in indices]
    else:
        paths = interesting
    destination = pathlib.Path(sys.argv[2])
    card_width, gap = 310, 92
    total_width = 92 + len(paths) * card_width + (len(paths) - 1) * gap + 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="660" viewBox="0 0 {total_width} 660" role="img">',
        '<title>Tyr molecule branching trajectory</title>',
        f'<rect width="{total_width}" height="660" fill="#08111f"/>',
        '<text x="72" y="76" fill="#f8fafc" font-family="Inter,Helvetica,sans-serif" font-size="38" font-weight="750">Molecule generation as a branching process</text>',
        '<text x="72" y="112" fill="#94a3b8" font-family="Inter,Helvetica,sans-serif" font-size="20">Intermediate states exported directly by Tyr</text>',
    ]
    previous_atoms = None
    for i, path in enumerate(paths):
        x = 92 + i * (card_width + gap)
        atoms = read_xyz(path)
        comment = path.read_text().splitlines()[1]
        time_match = re.search(r"t=([0-9.]+)", comment)
        time = time_match.group(1) if time_match else ("1.0" if i == len(paths) - 1 else "?")
        parts.append(f'<rect x="{x}" y="164" width="{card_width}" height="400" rx="22" fill="#0f172a" stroke="#334155" stroke-width="2"/>')
        actual_step = frame_index(str(path))
        parts.append(f'<text x="{x + 28}" y="210" fill="#67e8f9" font-family="SFMono-Regular,Menlo,monospace" font-size="17">step {actual_step} · t={time}</text>')
        parts.extend(miniature(path, x + 18, 218, card_width - 36, 255))
        parts.append(f'<text x="{x + 28}" y="526" fill="#cbd5e1" font-family="Inter,Helvetica,sans-serif" font-size="19">{len(atoms)} atoms</text>')
        if i > 0:
            arrow_x1 = x - gap + 18
            arrow_x2 = x - 18
            parts.append(f'<line x1="{arrow_x1}" y1="360" x2="{arrow_x2}" y2="360" stroke="#a78bfa" stroke-width="4"/>')
            parts.append(f'<path d="M {arrow_x2 - 12} 350 L {arrow_x2} 360 L {arrow_x2 - 12} 370" fill="none" stroke="#a78bfa" stroke-width="4"/>')
            delta = len(atoms) - (previous_atoms or 0)
            label = f"+{delta} branch" if delta > 0 else "flow step"
            parts.append(f'<text x="{(arrow_x1 + arrow_x2) / 2}" y="337" fill="#c4b5fd" font-family="Inter,Helvetica,sans-serif" font-size="14" font-weight="700" text-anchor="middle">{label}</text>')
        previous_atoms = len(atoms)
    parts.append('<text x="72" y="622" fill="#64748b" font-family="Inter,Helvetica,sans-serif" font-size="17">Masked source → split events → generated geometry</text>')
    parts.append('</svg>')
    destination.write_text("\n".join(parts))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

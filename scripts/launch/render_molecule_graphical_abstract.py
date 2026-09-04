#!/usr/bin/env python3
"""Create a scientific figure directly from Tyr's lineage-aware JSONL trajectory."""

from __future__ import annotations

import csv
import html
import json
import pathlib

ROOT = pathlib.Path("launch/generated/molecule")
PARTICLE_COLORS = ["#2563eb", "#ea580c", "#059669", "#9333ea", "#ca8a04", "#0891b2"]
LABEL_SYMBOLS = {0: "X", 1: "H", 6: "C", 7: "N", 8: "O", 9: "X"}


def esc(value: object) -> str:
    return html.escape(str(value))


def load_trajectory() -> tuple[list[dict], list[dict]]:
    source = ROOT / "water_trajectory.jsonl"
    frames_by_step: dict[int, dict] = {}
    events: list[dict] = []
    metadata_rows = 0
    for line_number, raw in enumerate(source.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        row_type = row.get("type")
        if row_type == "metadata":
            metadata_rows += 1
            if row.get("schema") != "tyr.branching-trajectory.v1":
                raise ValueError(f"{source}:{line_number}: unsupported trajectory schema")
            if row.get("coordinates") != "raw":
                raise ValueError(f"{source}:{line_number}: expected raw coordinates")
        elif row_type == "state":
            step = int(row["step"])
            frame = frames_by_step.setdefault(step, {"step": step, "time": float(row["t"]), "atoms": []})
            if abs(frame["time"] - float(row["t"])) > 1e-12:
                raise ValueError(f"{source}:{line_number}: inconsistent time within step {step}")
            atom = dict(row)
            atom["symbol"] = LABEL_SYMBOLS.get(int(row["label"]), str(row["label"]))
            frame["atoms"].append(atom)
        elif row_type == "event":
            events.append(row)
        else:
            raise ValueError(f"{source}:{line_number}: unknown row type {row_type!r}")
    frames = [frames_by_step[step] for step in sorted(frames_by_step)]
    if metadata_rows != 1:
        raise ValueError(f"{source}: expected exactly one metadata row, found {metadata_rows}")
    if not frames:
        raise ValueError(f"no state rows found in {source}")
    for frame in frames:
        frame["atoms"].sort(key=lambda atom: int(atom["state_index"]))
    events.sort(key=lambda event: int(event["event_id"]))
    return frames, events


def particle_color(particle_id: int) -> str:
    return PARTICLE_COLORS[(particle_id - 1) % len(PARTICLE_COLORS)]


def line(parts: list[str], x1: float, y1: float, x2: float, y2: float, **attrs: object) -> None:
    style = " ".join(f'{k.replace("_", "-")}="{esc(v)}"' for k, v in attrs.items())
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {style}/>')


def text(parts: list[str], x: float, y: float, value: str, **attrs: object) -> None:
    defaults = {"font_family": "Inter,Helvetica,sans-serif", "font_size": 16, "fill": "#172033"}
    defaults.update(attrs)
    style = " ".join(f'{k.replace("_", "-")}="{esc(v)}"' for k, v in defaults.items())
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" {style}>{esc(value)}</text>')


def polyline(parts: list[str], points: list[tuple[float, float]], color: str, width: float = 3) -> None:
    value = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    parts.append(f'<polyline points="{value}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')


def axes(parts: list[str], x: float, y: float, w: float, h: float, xlabel: str, ylabel: str) -> None:
    line(parts, x, y + h, x + w, y + h, stroke="#475569", stroke_width=1.5)
    line(parts, x, y, x, y + h, stroke="#475569", stroke_width=1.5)
    for i in range(5):
        tx = x + i * w / 4
        line(parts, tx, y, tx, y + h, stroke="#e2e8f0", stroke_width=1)
        text(parts, tx, y + h + 23, f"{i/4:.2g}", text_anchor="middle", font_size=13, fill="#64748b")
    text(parts, x + w / 2, y + h + 48, xlabel, text_anchor="middle", font_size=15)
    text(parts, x - 48, y + h / 2, ylabel, text_anchor="middle", font_size=15, transform=f"rotate(-90 {x-48:.1f} {y+h/2:.1f})")


def main() -> None:
    frames, events = load_trajectory()
    max_atoms = max(len(f["atoms"]) for f in frames)

    tracks: dict[int, list[tuple[float, dict]]] = {}
    for frame in frames:
        for atom in frame["atoms"]:
            particle_id = int(atom["particle_id"])
            tracks.setdefault(particle_id, []).append((frame["time"], atom))

    csv_path = ROOT / "trajectory-data.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "step", "time", "state_index", "particle_id", "parent_id", "birth_event_id",
            "symbol", "label", "x", "y", "z",
        ])
        for frame in frames:
            for atom in frame["atoms"]:
                writer.writerow([
                    frame["step"], f'{frame["time"]:.6f}', atom["state_index"],
                    atom["particle_id"], atom["parent_id"], atom["birth_event_id"],
                    atom["symbol"], atom["label"], atom["x"], atom["y"], atom["z"],
                ])

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img">',
        '<title>Data-driven Tyr molecule branching trajectory</title>',
        '<rect width="1600" height="1000" fill="#ffffff"/>',
    ]
    text(parts, 65, 62, "Reference molecule path through continuous flow and discrete branching", font_size=34, font_weight=750, fill="#111827")
    text(parts, 65, 94, f"Target-conditioned diagnostic (not a learned sample) · {len(frames)} timepoints · {len(events)} split events · raw coordinates", font_size=16, fill="#64748b")

    # A: number of elements over time.
    text(parts, 65, 150, "A", font_size=24, font_weight=800)
    text(parts, 105, 150, "State dimension", font_size=20, font_weight=700)
    ax, ay, aw, ah = 110, 190, 590, 260
    axes(parts, ax, ay, aw, ah, "flow time, t", "number of atoms, N(t)")
    points = [(ax + f["time"] * aw, ay + ah - (len(f["atoms"]) - 0.8) / (max_atoms - 0.5) * ah) for f in frames]
    step_points: list[tuple[float, float]] = [points[0]]
    for previous, current in zip(points, points[1:]):
        step_points.extend([(current[0], previous[1]), current])
    polyline(parts, step_points, "#111827", 3.5)
    for event in events:
        if event["kind"] != "split":
            continue
        x = ax + float(event["t1"]) * aw
        line(parts, x, ay, x, ay + ah, stroke="#dc2626", stroke_width=1.5, stroke_dasharray="5 5")
        text(parts, x + 5, ay + 17, f"split {event['event_id']}", font_size=13, fill="#dc2626", font_weight=700)
    for n in range(1, max_atoms + 1):
        yy = ay + ah - (n - 0.8) / (max_atoms - 0.5) * ah
        text(parts, ax - 12, yy + 5, str(n), text_anchor="end", font_size=13, fill="#64748b")

    # B: data-derived projected coordinate tracks.
    text(parts, 790, 150, "B", font_size=24, font_weight=800)
    text(parts, 830, 150, "Coordinate trajectories", font_size=20, font_weight=700)
    bx, by, bw, bh = 835, 190, 650, 260
    axes(parts, bx, by, bw, bh, "flow time, t", "q = x + 0.35y − 0.20z")
    all_q = []
    q_tracks: dict[int, list[tuple[float, float]]] = {}
    for particle_id, track in tracks.items():
        q_track = []
        for time, atom in track:
            q = float(atom["x"]) + 0.35 * float(atom["y"]) - 0.20 * float(atom["z"])
            q_track.append((time, q))
            all_q.append(q)
        q_tracks[particle_id] = q_track
    qmin, qmax = min(all_q), max(all_q)
    qpad = max((qmax - qmin) * 0.08, 0.05)
    q_to_plot = lambda t, q: (bx + t * bw, by + bh - (q - qmin + qpad) / (qmax - qmin + 2*qpad) * bh)
    for particle_id in sorted(q_tracks):
        track = q_tracks[particle_id]
        color = particle_color(particle_id)
        pts = [(bx + t * bw, by + bh - (q - qmin + qpad) / (qmax - qmin + 2*qpad) * bh) for t, q in track]
        polyline(parts, pts, color, 3)
        for x, y in pts[::4]:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.7" fill="{color}"/>')
        legend_y = by + 24 + (particle_id - 1) * 22
        line(parts, bx + bw - 110, legend_y - 5, bx + bw - 84, legend_y - 5, stroke=color, stroke_width=3)
        text(parts, bx + bw - 76, legend_y, f"particle {particle_id}", font_size=13, fill=color, font_weight=700)
    for event in events:
        if event["kind"] != "split":
            continue
        parent_track = q_tracks[int(event["parent_id"])]
        parent_point = q_to_plot(*parent_track[-1])
        for child_id in event["child_ids"]:
            child_point = q_to_plot(*q_tracks[int(child_id)][0])
            line(parts, *parent_point, *child_point, stroke="#64748b", stroke_width=1.5, stroke_dasharray="4 4")
        x = bx + float(event["t1"]) * bw
        line(parts, x, by, x, by + bh, stroke="#dc2626", stroke_width=1.2, stroke_dasharray="5 5")

    # C: XY paths, using only observed coordinates.
    text(parts, 65, 535, "C", font_size=24, font_weight=800)
    text(parts, 105, 535, "Spatial paths", font_size=20, font_weight=700)
    cx, cy, cw, ch = 110, 575, 590, 280
    xs = [float(atom["x"]) for frame in frames for atom in frame["atoms"]]
    ys = [float(atom["y"]) for frame in frames for atom in frame["atoms"]]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 0.1)
    line(parts, cx, cy + ch, cx + cw, cy + ch, stroke="#475569", stroke_width=1.5)
    line(parts, cx, cy, cx, cy + ch, stroke="#475569", stroke_width=1.5)
    text(parts, cx + cw/2, cy + ch + 42, "x coordinate", text_anchor="middle", font_size=15)
    text(parts, cx - 48, cy + ch/2, "y coordinate", text_anchor="middle", font_size=15, transform=f"rotate(-90 {cx-48} {cy+ch/2})")
    text(parts, cx, cy + ch + 20, f"{xmin:.2f}", text_anchor="middle", font_size=12, fill="#64748b")
    text(parts, cx + (xmax-xmin)/span*(cw-60) + 30, cy + ch + 20, f"{xmax:.2f}", text_anchor="middle", font_size=12, fill="#64748b")
    text(parts, cx - 10, cy + ch - 25, f"{ymin:.2f}", text_anchor="end", font_size=12, fill="#64748b")
    text(parts, cx - 10, cy + ch - 25 - (ymax-ymin)/span*(ch-60), f"{ymax:.2f}", text_anchor="end", font_size=12, fill="#64748b")
    spatial_tracks: dict[int, list[tuple[float, float]]] = {}
    for particle_id, track in tracks.items():
        spatial = []
        for _time, atom in track:
            x, y = float(atom["x"]), float(atom["y"])
            px = cx + 30 + (x - xmin) / span * (cw - 60)
            py = cy + ch - 30 - (y - ymin) / span * (ch - 60)
            spatial.append((px, py))
        spatial_tracks[particle_id] = spatial
        color = particle_color(particle_id)
        polyline(parts, spatial, color, 3)
        parts.append(f'<circle cx="{spatial[0][0]:.1f}" cy="{spatial[0][1]:.1f}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{spatial[-1][0]:.1f}" cy="{spatial[-1][1]:.1f}" r="6" fill="{color}"/>')
    for event in events:
        if event["kind"] != "split":
            continue
        parent_point = spatial_tracks[int(event["parent_id"])][-1]
        for child_id in event["child_ids"]:
            child_point = spatial_tracks[int(child_id)][0]
            line(parts, *parent_point, *child_point, stroke="#64748b", stroke_width=1.5, stroke_dasharray="4 4")

    # D: exact final exported geometry.
    text(parts, 790, 535, "D", font_size=24, font_weight=800)
    text(parts, 830, 535, "Endpoint geometry", font_size=20, font_weight=700)
    dx, dy = 1160, 780
    final_atoms = frames[-1]["atoms"]
    placed = []
    for atom in final_atoms:
        symbol = atom["symbol"]
        x, y, z = float(atom["x"]), float(atom["y"]), float(atom["z"])
        placed.append((symbol, dx + x * 190, dy - y * 190, x, y, z))
    for i, atom in enumerate(placed):
        for other in placed[i+1:]:
            distance = ((atom[3]-other[3])**2 + (atom[4]-other[4])**2 + (atom[5]-other[5])**2) ** 0.5
            if distance < 1.25 and not (atom[0] == other[0] == "H"):
                line(parts, atom[1], atom[2], other[1], other[2], stroke="#94a3b8", stroke_width=13, stroke_linecap="round")
    atom_colors = {"O": "#ef4444", "H": "#f8fafc", "C": "#334155", "N": "#3b82f6", "X": "#a78bfa"}
    for symbol, x, y, *_ in placed:
        r = 30 if symbol == "H" else 38
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{atom_colors.get(symbol, "#f59e0b")}" stroke="#334155" stroke-width="3"/>')
        text(parts, x, y + 8, symbol, text_anchor="middle", font_size=22, font_weight=800)
    text(parts, 830, 872, "Final coordinates are plotted without geometric adjustment.", font_size=14, fill="#64748b")
    text(parts, 65, 930, "Methods.", font_size=14, font_weight=800)
    text(parts, 132, 930, "Solid lines connect raw states of one runtime particle; dashed connectors show parent-to-child lineage at logged split events.", font_size=14, fill="#475569")
    text(parts, 65, 958, "No display offsets or inferred correspondences are used. Sources: water_trajectory.jsonl and trajectory-data.csv.", font_size=14, fill="#475569")
    parts.append("</svg>")

    svg_path = ROOT / "graphical-abstract.svg"
    svg_path.write_text("\n".join(parts))
    print(f"wrote {svg_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

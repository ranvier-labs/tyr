#!/usr/bin/env python3
"""Render a lineage-aware BranchingFlows JSONL trajectory as a scientific SVG.

The renderer uses only values present in the exported trajectory.  In
particular, it does not fan out coincident children or otherwise alter spatial
coordinates for display.
"""

from __future__ import annotations

import html
import json
import pathlib
import sys
from collections import defaultdict


INK = "#22252a"
MUTED = "#666b73"
FAINT = "#8a8f96"
RULE = "#deded8"
GRID = "#e9e9e4"
BLUE = "#3974a4"
EVENT = "#b85a3c"
PAPER = "#fdfdfb"


def read_trajectory(path: pathlib.Path):
    metadata = {}
    states = []
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_number}: {error}") from error
        row_type = row.get("type")
        if row_type == "metadata":
            metadata = row
        elif row_type == "state":
            states.append(row)
        elif row_type == "event":
            events.append(row)
    if not states:
        raise SystemExit(f"{path}: no state rows")
    return metadata, states, events


def line(x1, y1, x2, y2, **attrs):
    values = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs}
    return "<line " + " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in values.items()) + "/>"


def text(x, y, value, **attrs):
    values = {"x": x, "y": y, **attrs}
    attributes = " ".join(f'{key.replace("_", "-")}="{val}"' for key, val in values.items())
    return f"<text {attributes}>{html.escape(str(value))}</text>"


def path(points, **attrs):
    values = {"d": "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points), **attrs}
    return "<path " + " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in values.items()) + "/>"


def panel_label(parts, letter, title, x, y):
    parts.append(text(x, y, letter, fill=INK, font_size=22, font_weight=700))
    parts.append(text(x + 34, y, title, fill=INK, font_size=18, font_weight=650))


def ticks(parts, x0, x1, y0, y1, x_values, y_values, sx, sy, x_label, y_label):
    for value in y_values:
        y = sy(value)
        parts.append(line(x0, y, x1, y, stroke=GRID, stroke_width=1))
        parts.append(text(x0 - 10, y + 4, f"{value:g}", fill=MUTED, font_size=12, text_anchor="end"))
    for value in x_values:
        x = sx(value)
        parts.append(line(x, y0, x, y1, stroke=GRID, stroke_width=1))
        parts.append(text(x, y1 + 21, f"{value:g}", fill=MUTED, font_size=12, text_anchor="middle"))
    parts.append(line(x0, y1, x1, y1, stroke=INK, stroke_width=1))
    parts.append(line(x0, y0, x0, y1, stroke=INK, stroke_width=1))
    parts.append(text((x0 + x1) / 2, y1 + 43, x_label, fill=MUTED, font_size=13, text_anchor="middle"))
    parts.append(text(x0 - 52, (y0 + y1) / 2, y_label, fill=MUTED, font_size=13,
                      text_anchor="middle", transform=f"rotate(-90 {x0 - 52} {(y0 + y1) / 2})"))


def lineage_positions(states, events):
    children = defaultdict(list)
    parents = {}
    for event in events:
        if event.get("kind") != "split":
            continue
        parent = event["parent_id"]
        for child in event.get("child_ids", []):
            children[parent].append(child)
            parents[child] = parent
    ids = sorted({row["particle_id"] for row in states})
    roots = [particle_id for particle_id in ids if particle_id not in parents]
    positions = {}
    next_leaf = 0.0

    def assign(particle_id):
        nonlocal next_leaf
        descendants = children.get(particle_id, [])
        if not descendants:
            positions[particle_id] = next_leaf
            next_leaf += 1.0
        else:
            for child in descendants:
                assign(child)
            positions[particle_id] = sum(positions[child] for child in descendants) / len(descendants)

    for root in roots:
        assign(root)
    return positions, children


def label_name(label):
    return {1: "H", 6: "C", 7: "N", 8: "O", 9: "mask"}.get(label, str(label))


def render(source: pathlib.Path, destination: pathlib.Path):
    metadata, states, events = read_trajectory(source)
    by_particle = defaultdict(list)
    by_step = defaultdict(list)
    for row in states:
        by_particle[row["particle_id"]].append(row)
        by_step[row["step"]].append(row)
    for rows in by_particle.values():
        rows.sort(key=lambda row: row["t"])
    steps = sorted(by_step)
    t_min = min(row["t"] for row in states)
    t_max = max(row["t"] for row in states)
    positions, children = lineage_positions(states, events)
    split_by_parent = {event["parent_id"]: event for event in events if event.get("kind") == "split"}

    width, height = 1600, 1260
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Lineage-aware Tyr BranchingFlows trajectory</title>',
        '<desc id="description">Stable particle identities, split events, raw coordinate traces, population size, and spatial states from a target-conditioned water trajectory.</desc>',
        f'<rect width="{width}" height="{height}" fill="{PAPER}"/>',
        '<style>text { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }</style>',
        text(80, 55, "BranchingFlows trajectory", fill=INK, font_size=30, font_weight=650),
        text(80, 82, "Target-conditioned sampler trace · raw coordinates · stable runtime lineage", fill=MUTED, font_size=15),
    ]

    # Panel A: stable runtime lineage.
    ax0, ax1, ay0, ay1 = 130, 1510, 135, 365
    panel_label(parts, "A", "Runtime lineage", 80, 122)
    sx_a = lambda t: ax0 + (t - t_min) / max(t_max - t_min, 1e-9) * (ax1 - ax0)
    max_pos = max(positions.values(), default=0.0)
    sy_a = lambda pos: ay0 + (pos + 0.5) / (max_pos + 1.0) * (ay1 - ay0)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = sx_a(t)
        parts.append(line(x, ay0, x, ay1, stroke=GRID, stroke_width=1))
        parts.append(text(x, ay1 + 20, f"{t:.2g}", fill=MUTED, font_size=12, text_anchor="middle"))
    for particle_id, rows in sorted(by_particle.items()):
        birth = rows[0]["t"]
        death = split_by_parent.get(particle_id, {}).get("t1", rows[-1]["t"])
        y = sy_a(positions[particle_id])
        parts.append(line(sx_a(birth), y, sx_a(death), y, stroke=BLUE, stroke_width=3))
        parts.append(text(sx_a(birth) + 6, y + 16, f"p{particle_id}", fill=BLUE, font_size=12))
        if particle_id not in children:
            parts.append(text(ax1 + 10, y + 4, label_name(rows[-1]["label"]), fill=INK, font_size=13))
    for event in events:
        if event.get("kind") != "split":
            continue
        parent = event["parent_id"]
        child_ids = event.get("child_ids", [])
        x = sx_a(event["t1"])
        parent_y = sy_a(positions[parent])
        child_ys = [sy_a(positions[child]) for child in child_ids]
        parts.append(line(x, min(child_ys + [parent_y]), x, max(child_ys + [parent_y]), stroke=EVENT, stroke_width=1.5))
        parts.append(f'<circle cx="{x}" cy="{parent_y}" r="5" fill="{PAPER}" stroke="{EVENT}" stroke-width="2"/>')
        parts.append(text(x + 7, min(child_ys + [parent_y]) - 16, f"split {event['event_id']} · t={event['t1']:.4f}", fill=EVENT, font_size=11))
    parts.append(text((ax0 + ax1) / 2, ay1 + 43, "time", fill=MUTED, font_size=13, text_anchor="middle"))

    # Panel B: exact x-coordinate traces.
    bx0, bx1, by0, by1 = 130, 970, 485, 770
    panel_label(parts, "B", "Particle x-coordinate", 80, 455)
    coord_min = min(row["x"] for row in states)
    coord_max = max(row["x"] for row in states)
    coord_pad = max((coord_max - coord_min) * 0.06, 0.1)
    coord_min -= coord_pad
    coord_max += coord_pad
    sx_b = lambda t: bx0 + (t - t_min) / max(t_max - t_min, 1e-9) * (bx1 - bx0)
    sy_b = lambda value: by1 - (value - coord_min) / max(coord_max - coord_min, 1e-9) * (by1 - by0)
    y_ticks = [round(coord_min + i * (coord_max - coord_min) / 4, 1) for i in range(5)]
    ticks(parts, bx0, bx1, by0, by1, [0, .25, .5, .75, 1], y_ticks, sx_b, sy_b, "time", "x coordinate")
    for event in events:
        x = sx_b(event["t1"])
        parts.append(line(x, by0, x, by1, stroke=EVENT, stroke_width=1, stroke_dasharray="5 5"))
    palette = ["#3974a4", "#517fa4", "#6a8aa2", "#817f9c", "#9b718b", "#ad646f"]
    for index, (particle_id, rows) in enumerate(sorted(by_particle.items())):
        points = [(sx_b(row["t"]), sy_b(row["x"])) for row in rows]
        if len(points) == 1:
            parts.append(f'<circle cx="{points[0][0]:.2f}" cy="{points[0][1]:.2f}" r="3" fill="{palette[index % len(palette)]}"/>')
        else:
            parts.append(path(points, fill="none", stroke=palette[index % len(palette)], stroke_width=2.2))
        last_x, last_y = points[-1]
        parts.append(text(last_x + 5, last_y - 5, f"p{particle_id}", fill=palette[index % len(palette)], font_size=11))

    # Panel C: population size and split events.
    cx0, cx1, cy0, cy1 = 1100, 1510, 485, 770
    panel_label(parts, "C", "Population size", 1050, 455)
    sx_c = lambda t: cx0 + (t - t_min) / max(t_max - t_min, 1e-9) * (cx1 - cx0)
    max_n = max(len(by_step[step]) for step in steps)
    sy_c = lambda n: cy1 - (n - 0.5) / max(max_n, 1) * (cy1 - cy0)
    ticks(parts, cx0, cx1, cy0, cy1, [0, .25, .5, .75, 1], list(range(1, max_n + 1)), sx_c, sy_c, "time", "N(t)")
    population_points = []
    previous = None
    for step in steps:
        t = by_step[step][0]["t"]
        n = len(by_step[step])
        if previous is not None:
            population_points.append((sx_c(t), sy_c(previous)))
        population_points.append((sx_c(t), sy_c(n)))
        previous = n
    parts.append(path(population_points, fill="none", stroke=BLUE, stroke_width=2.8))
    for event in events:
        x, y = sx_c(event["t1"]), sy_c(len(by_step[min(steps, key=lambda step: abs(by_step[step][0]["t"] - event["t1"]))]))
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{EVENT}" stroke="{PAPER}" stroke-width="2"/>')

    # Panel D: exact x-y states at the initial state, event frames, and endpoint.
    panel_label(parts, "D", "Spatial state (x-y plane, shared scale)", 80, 860)
    event_steps = []
    for event in events:
        event_steps.append(min(steps, key=lambda step: abs(by_step[step][0]["t"] - event["t1"])))
    snapshot_steps = []
    for step in [steps[0], *event_steps, steps[-1]]:
        if step not in snapshot_steps:
            snapshot_steps.append(step)
    spatial_x_min = min(row["x"] for row in states)
    spatial_x_max = max(row["x"] for row in states)
    spatial_y_min = min(row["y"] for row in states)
    spatial_y_max = max(row["y"] for row in states)
    spatial_span = max(spatial_x_max - spatial_x_min, spatial_y_max - spatial_y_min, 1e-9)
    spatial_x_mid = (spatial_x_min + spatial_x_max) / 2
    spatial_y_mid = (spatial_y_min + spatial_y_max) / 2
    spatial_x_min, spatial_x_max = spatial_x_mid - spatial_span * .56, spatial_x_mid + spatial_span * .56
    spatial_y_min, spatial_y_max = spatial_y_mid - spatial_span * .56, spatial_y_mid + spatial_span * .56
    gap = 24
    snapshot_width = (1430 - gap * (len(snapshot_steps) - 1)) / len(snapshot_steps)
    for index, step in enumerate(snapshot_steps):
        x0 = 80 + index * (snapshot_width + gap)
        x1 = x0 + snapshot_width
        y0, y1 = 900, 1105
        rows = sorted(by_step[step], key=lambda row: row["state_index"])
        sx_d = lambda value: x0 + 14 + (value - spatial_x_min) / (spatial_x_max - spatial_x_min) * (snapshot_width - 28)
        sy_d = lambda value: y1 - 14 - (value - spatial_y_min) / (spatial_y_max - spatial_y_min) * (y1 - y0 - 28)
        parts.append(f'<rect x="{x0}" y="{y0}" width="{snapshot_width}" height="{y1-y0}" fill="none" stroke="{RULE}"/>')
        parts.append(text(x0 + 8, y0 - 10, f"step {step} · t={rows[0]['t']:.4f} · N={len(rows)}", fill=MUTED, font_size=12))
        coincident = defaultdict(list)
        for row in rows:
            coincident[(row["x"], row["y"])].append(row)
        for (x_value, y_value), point_rows in coincident.items():
            px, py = sx_d(x_value), sy_d(y_value)
            fill = EVENT if point_rows[0]["label"] == 8 else BLUE if point_rows[0]["label"] == 1 else "#a0a4a8"
            stroke = EVENT if len(point_rows) > 1 else PAPER
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="{fill}" fill-opacity="0.82" stroke="{stroke}" stroke-width="{2.5 if len(point_rows) > 1 else 1.5}"/>')
            identities = ",".join(f"p{row['particle_id']}" for row in point_rows)
            parts.append(text(px + 9, py - 7, identities, fill=INK, font_size=10))
    parts.append(text(80, 1140, "Children that coincide at birth are drawn at the same coordinate; no display jitter is applied.", fill=MUTED, font_size=12))

    schema = metadata.get("schema", "unknown schema")
    parts.append(line(80, 1175, 1510, 1175, stroke=INK, stroke_width=1))
    parts.append(text(80, 1203, "Figure 1.", fill=INK, font_size=13, font_weight=700))
    parts.append(text(145, 1203, "A target-conditioned mechanism trace exported by Tyr. It is not a learned molecular sample.", fill=INK, font_size=13))
    parts.append(text(80, 1227, f"Source: {source.as_posix()} · schema: {schema} · {len(states)} state rows · {len(events)} events.", fill=MUTED, font_size=12))
    parts.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n")
    print(f"wrote {destination}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: render_branching_trajectory.py trajectory.jsonl output.svg")
    render(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))


if __name__ == "__main__":
    main()

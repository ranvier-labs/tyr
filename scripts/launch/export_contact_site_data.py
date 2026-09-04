#!/usr/bin/env python3
"""Convert the Lean-exported contact trajectory CSV to a browser data module."""

from __future__ import annotations

import csv
import json
import pathlib


SOURCE = pathlib.Path("launch/generated/contact/trajectory.csv")
DESTINATION = pathlib.Path("launch/site/contact-data.js")


def main() -> None:
    with SOURCE.open(newline="") as handle:
        rows = [
            {
                "time": float(row["time"]),
                "phase": row["phase"],
                "position": float(row["position"]),
                "velocity": float(row["velocity"]),
            }
            for row in csv.DictReader(handle)
        ]
    payload = {
        "source": "launch/generated/contact/trajectory.csv",
        "samples": rows,
        "impact": {
            "time": 0.2,
            "preVelocity": -2.962,
            "postVelocity": 1.1848,
            "saltationAlpha": 4.636732,
            "restitutionGradient": 2.962,
        },
    }
    DESTINATION.write_text(
        "window.TYR_CONTACT_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    )
    print(f"wrote {DESTINATION} from {SOURCE}")


if __name__ == "__main__":
    main()

"""
Regenerate calibration.json from a reference set of simulation trials.

The interactivity score normalizes each sample's `human_torque` against fixed
per-state bounds. Those bounds are derived ONCE here from a reference dataset and
then stored, so that scoring a single trial is deterministic and does not depend
on whatever other files happen to be present at scoring time.

Bounds = the [lo_pct, hi_pct] percentiles of human_torque within each State.
Using percentiles (default 5 / 95) makes the bounds robust to outliers.

Usage
-----
    python build_calibration.py "AI_evaluation/new_patient_data/virtual/task1/augmented_*.csv"
    python build_calibration.py "<glob>" --lo 5 --hi 95 --out calibration.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from inter_score import _resolve, _TORQUE_COLS, infer_state

HERE = os.path.dirname(os.path.abspath(__file__))


def build(pattern: str, lo_pct: float, hi_pct: float) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        tcol = _resolve(df, _TORQUE_COLS)
        if tcol is None:
            continue
        sub = pd.DataFrame({"human_torque": df[tcol].to_numpy(dtype=float),
                            "state": infer_state(df)})
        frames.append(sub)

    data = pd.concat(frames, ignore_index=True)
    states = {}
    for state, grp in data.groupby("state"):
        lo = float(grp["human_torque"].quantile(lo_pct / 100.0))
        hi = float(grp["human_torque"].quantile(hi_pct / 100.0))
        direction = (
            "higher_torque_higher_score" if state == "Flexion"
            else "higher_torque_lower_score"
        )
        states[state] = {"lo": round(lo, 4), "hi": round(hi, 4),
                         "direction": direction, "n_samples": int(len(grp))}

    # If the reference set has no Extension trials, fall back to Flexion magnitude
    # bounds (provisional) so Extension scoring is still defined.
    if "Extension" not in states and "Flexion" in states:
        fx = states["Flexion"]
        states["Extension"] = {"lo": fx["lo"], "hi": fx["hi"],
                               "direction": "higher_torque_lower_score",
                               "n_samples": 0, "provisional": True}

    return {
        "method": "per-state min-max normalization of human_torque (Module A port)",
        "source": "https://github.com/donghee/wde-interactivity",
        "reference_pattern": pattern,
        "reference_files": len(files),
        "percentiles": [lo_pct, hi_pct],
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build calibration.json from reference trials.")
    parser.add_argument("pattern", help="Glob for reference CSVs.")
    parser.add_argument("--lo", type=float, default=5.0, help="Lower percentile (default 5).")
    parser.add_argument("--hi", type=float, default=95.0, help="Upper percentile (default 95).")
    parser.add_argument("--out", default=os.path.join(HERE, "calibration.json"))
    args = parser.parse_args()

    calib = build(args.pattern, args.lo, args.hi)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")
    for state, cfg in calib["states"].items():
        print(f"  {state:10s} lo={cfg['lo']:+.4f} hi={cfg['hi']:+.4f} "
              f"dir={cfg['direction']} n={cfg.get('n_samples')}")


if __name__ == "__main__":
    main()

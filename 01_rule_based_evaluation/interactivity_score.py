"""
Exoskeleton-Human Interactivity Score (0-100)
=============================================

A deterministic, formula-based interactivity score for elbow exoskeleton trials.

Lineage
-------
This is a port of the ORIGINAL deterministic evaluation from
    https://github.com/donghee/wde-interactivity  (Module A / GET_interaction_score*)
which scores a trial directly from the user's `human_torque`.

It deliberately does NOT use the SVR model that later appeared in
`wearable_ui_frontend/wde/interactivity` -- that model was trained on 24 samples
with over-fit hyper-parameters (gamma=100, C=1000) and produced unstable scores
that swung 5-8 points under tiny sensor noise. This formula-based score is
deterministic: identical input + calibration always yields the identical score.

Method (Module A, per-angle / per-state direction)
--------------------------------------------------
1. Use `human_torque` directly from the simulation CSV (already provided by the
   Gazebo / augmented simulator -- no torque re-computation needed).
2. Determine motion State per sample:
       Flexion   if angular_velocity < 0   (elbow angle decreasing, 170 deg -> 64 deg)
       Extension otherwise                 (elbow angle increasing, 64 deg -> 170 deg)
   Physical CSVs that carry an explicit 'U_Status' column use it directly.
3. Apply the per-state direction (verified against Evaluation_Table2.csv, where the
   sign of (Range_1 - Range_0) splits perfectly by State: Flexion=+, Extension=-):
       Flexion   : higher human_torque -> higher score
       Extension : higher human_torque -> lower  score
4. Normalize each sample to [0, 1] against fixed per-state bounds (calibration.json),
   clip to [0, 1], average over the trial, and multiply by 100.

Usage
-----
    python interactivity_score.py data/augmented_01.csv
    python interactivity_score.py data/augmented_01.csv --calibration calibration.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.join(HERE, "calibration.json")

# Column-name aliases so both the augmented (simulation) and physical schemas work.
_TORQUE_COLS = ("human_torque", "Human_Torque", "HumanTorque")
_VELOCITY_COLS = ("angular_velocity", "Angular_Velocity")
_STATE_COLS = ("U_Status", "State", "state")


def _resolve(df: pd.DataFrame, candidates) -> str | None:
    """Return the first column in `candidates` that exists in `df`, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_calibration(path: str = DEFAULT_CALIBRATION) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_state(df: pd.DataFrame) -> np.ndarray:
    """
    Per-sample motion state as an array of 'Flexion' / 'Extension'.

    Priority:
      1. explicit State column (physical data: 'U_Status'),
      2. sign of angular_velocity (simulation data),
      3. fallback: sign of the step-to-step change in elbow angle.
    """
    state_col = _resolve(df, _STATE_COLS)
    if state_col is not None:
        return (
            df[state_col]
            .astype(str)
            .str.strip()
            .str.capitalize()
            .where(lambda s: s.isin(["Flexion", "Extension"]), "Flexion")
            .to_numpy()
        )

    vel_col = _resolve(df, _VELOCITY_COLS)
    if vel_col is not None:
        return np.where(df[vel_col].to_numpy() < 0, "Flexion", "Extension")

    # Fallback: derive direction from the elbow angle trajectory.
    angle_col = _resolve(df, ("elbow_angle_rad", "Elbow Angle", "Angle"))
    if angle_col is None:
        raise KeyError(
            "Cannot determine motion state: need one of "
            f"{_STATE_COLS} or {_VELOCITY_COLS} or an angle column."
        )
    d_angle = np.gradient(df[angle_col].to_numpy(dtype=float))
    return np.where(d_angle < 0, "Flexion", "Extension")


def score_dataframe(df: pd.DataFrame, calib: dict) -> float:
    """Compute the 0-100 interactivity score for a single loaded trial."""
    torque_col = _resolve(df, _TORQUE_COLS)
    if torque_col is None:
        raise KeyError(f"No human-torque column found; expected one of {_TORQUE_COLS}.")

    torque = df[torque_col].to_numpy(dtype=float)
    state = infer_state(df)
    states_cfg = calib["states"]

    per_sample = np.empty(len(df), dtype=float)
    for i, (t, s) in enumerate(zip(torque, state)):
        cfg = states_cfg.get(s, states_cfg["Flexion"])
        lo, hi = cfg["lo"], cfg["hi"]
        span = hi - lo
        if span == 0:
            per_sample[i] = 0.5
            continue
        frac = (t - lo) / span
        if cfg["direction"] == "higher_torque_lower_score":
            frac = 1.0 - frac
        per_sample[i] = frac

    per_sample = np.clip(per_sample, 0.0, 1.0)
    # NaN-safe mean (rows with missing torque are ignored).
    return float(np.nanmean(per_sample) * 100.0)


def interactivity_score(csv_path: str, calibration: str = DEFAULT_CALIBRATION) -> float:
    """Read a trial CSV and return its interactivity score in [0, 100]."""
    df = pd.read_csv(csv_path)
    calib = load_calibration(calibration)
    return score_dataframe(df, calib)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exoskeleton interactivity score (0-100).")
    parser.add_argument("csv", help="Path to a trial CSV (e.g. data/augmented_01.csv).")
    parser.add_argument(
        "--calibration",
        default=DEFAULT_CALIBRATION,
        help="Path to calibration.json (default: alongside this script).",
    )
    args = parser.parse_args()

    score = interactivity_score(args.csv, args.calibration)
    print(f"Interactivity Score: {score:.1f} / 100   ({os.path.basename(args.csv)})")


if __name__ == "__main__":
    main()

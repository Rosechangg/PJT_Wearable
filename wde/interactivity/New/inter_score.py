# inter_score.py
#
# Drop-in replacement for the previous SVR-based scorer.
# ONLY the scoring algorithm changed: the unstable SVR model
# (svr_y_score_model.joblib, trained on 24 samples, gamma=100/C=1000) is replaced
# by the ORIGINAL deterministic, torque-based evaluation ported from
#   https://github.com/donghee/wde-interactivity  (Module A).
#
# Everything else (the inter_graph() function and service_inter.py) is unchanged.
#
# Why: the SVR score swung 5-8 points under tiny sensor noise (non-reproducible).
# This score is deterministic -- identical input + calibration -> identical score.

import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless rendering for the web service
import matplotlib.pyplot as plt
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIBRATION = os.path.join(HERE, "calibration.json")

# Column-name aliases so both the simulation (augmented) and physical schemas work.
_TORQUE_COLS = ("human_torque", "Human_Torque", "HumanTorque")
_VELOCITY_COLS = ("angular_velocity", "Angular_Velocity")
_STATE_COLS = ("U_Status", "State", "state")


def _resolve(df, candidates):
    """Return the first column in `candidates` present in `df`, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def load_calibration(path=DEFAULT_CALIBRATION):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_state(df):
    """
    Per-sample motion state ('Flexion' / 'Extension').

    Priority: explicit State column (physical 'U_Status') -> sign of
    angular_velocity (simulation) -> sign of the elbow-angle step change.
    """
    state_col = _resolve(df, _STATE_COLS)
    if state_col is not None:
        s = df[state_col].astype(str).str.strip().str.capitalize()
        return np.where(s.isin(["Flexion", "Extension"]), s, "Flexion")

    vel_col = _resolve(df, _VELOCITY_COLS)
    if vel_col is not None:
        return np.where(df[vel_col].to_numpy() < 0, "Flexion", "Extension")

    angle_col = _resolve(df, ("elbow_angle_rad", "sensor_position", "Elbow Angle", "Angle"))
    if angle_col is None:
        raise KeyError(
            "Cannot determine motion state: need one of "
            f"{_STATE_COLS} or {_VELOCITY_COLS} or an angle column."
        )
    d_angle = np.gradient(df[angle_col].to_numpy(dtype=float))
    return np.where(d_angle < 0, "Flexion", "Extension")


def score_dataframe(df, calib):
    """
    Compute the 0-100 interactivity score for a single loaded trial.

    For each sample, normalize human_torque to [0,1] against per-state bounds
    (lo, hi) with the per-state direction, clip, average, x100:

        Flexion   : (human_torque - lo) / (hi - lo)        # higher torque -> higher
        Extension : (hi - human_torque) / (hi - lo)        # higher torque -> lower
    """
    torque_col = _resolve(df, _TORQUE_COLS)
    if torque_col is None:
        raise KeyError(
            "No human-torque column found in result.csv; expected one of "
            f"{_TORQUE_COLS}. The simulator must output human_torque."
        )

    torque = df[torque_col].to_numpy(dtype=float)
    state = infer_state(df)
    states_cfg = calib["states"]

    per_sample = np.empty(len(df), dtype=float)
    for i, (t, s) in enumerate(zip(torque, state)):
        cfg = states_cfg.get(str(s), states_cfg["Flexion"])
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
    return float(np.nanmean(per_sample) * 100.0)


def infer_y_score(result_csv_path, model_path=None, n_steps=None,
                  calibration_path=DEFAULT_CALIBRATION):
    """
    Deterministic torque-based interactivity score in [0, 100].

    Signature is kept compatible with the previous SVR version so that
    service_inter.py does not change: `model_path` and `n_steps` are accepted
    and ignored (no joblib model is loaded any more).
    """
    df = pd.read_csv(result_csv_path)
    calib = load_calibration(calibration_path)
    return score_dataframe(df, calib)


def inter_graph(result_csv_path):
    # UNCHANGED from the original extend/inter_score.py
    df = pd.read_csv(result_csv_path)
    df.rename(columns={'step': 'time_step', 'elbow_angle_rad': 'sensor_position', 'inter_force': 'sensor_force'}, inplace=True)

    # elbow angle rad to degree
    df['sensor_position'] = df['sensor_position'] * (180.0 / np.pi) * -1.0

    plt.figure(figsize=(15, 10))
    plt.plot(df['time_step'], df['sensor_position'], label='Sensor Position', color='blue')

    #plt.show()

    graph_img = BytesIO()
    plt.savefig(graph_img, format='png', dpi=72)
    plt.clf()
    graph_img.seek(0)

    return graph_img


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "augmented_01.csv")
    y = infer_y_score(path)
    print(f"Interactivity Score: {y:.1f} / 100   ({os.path.basename(path)})")


if __name__ == "__main__":
    main()

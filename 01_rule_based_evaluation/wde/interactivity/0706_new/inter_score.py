# inter_score.py
#
# Deterministic, engagement-based interactivity scorer.
#
# Previous versions (SVR model, then per-state torque min-max normalization)
# collapsed to near-identical scores on the M2/M6 trials: the min-max bounds
# were calibrated on a different torque scale (~+-0.8 N.m), so the new trials
# (human_torque ~1-10 N.m) all saturated at 1.0 and the scores only differed
# by ~5 points. That version also ignored the signals that actually separate
# conditions (motor cooperation, interaction power).
#
# This version scores "interactivity" as the ASSISTED POWER RATIO: of the
# mechanical power the human delivers (|human_torque * angular_velocity|), what
# fraction occurs while the motor is actively assisting (motor torque in the
# same direction as the human torque). One parameter-free number in [0, 100]
# that rewards, jointly:
#   - participation : larger human_torque -> more weight
#   - power transfer: faster motion (angular_velocity) -> more weight
#   - cooperation   : motor torque agrees in sign -> counts as assisted
#
# Same input + code -> identical score (pure arithmetic, no model, no RNG).

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless rendering for the web service
import matplotlib.pyplot as plt
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))

# Column-name aliases so both the simulation (augmented) and physical schemas work.
_TORQUE_COLS = ("human_torque", "Human_Torque", "HumanTorque")
_MOTOR_COLS = ("motor_torque", "Motor_Torque", "MotorTorque")
_VELOCITY_COLS = ("angular_velocity", "Angular_Velocity")


def _resolve(df, candidates):
    """Return the first column in `candidates` present in `df`, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def score_dataframe(df):
    """
    Assisted power ratio in [0, 100] for a single loaded trial.

        power_i    = |human_torque_i * angular_velocity_i|
        assisted_i = 1 if sign(human_torque_i) == sign(motor_torque_i) else 0
        score      = 100 * sum(assisted_i * power_i) / sum(power_i)
    """
    tcol = _resolve(df, _TORQUE_COLS)
    mcol = _resolve(df, _MOTOR_COLS)
    vcol = _resolve(df, _VELOCITY_COLS)
    missing = [name for name, col in
               (("human_torque", tcol), ("motor_torque", mcol),
                ("angular_velocity", vcol)) if col is None]
    if missing:
        raise KeyError(
            f"result.csv is missing required column(s) {missing}; "
            "the simulator must output human_torque, motor_torque and angular_velocity."
        )

    human = df[tcol].to_numpy(dtype=float)
    motor = df[mcol].to_numpy(dtype=float)
    vel = df[vcol].to_numpy(dtype=float)

    power = np.abs(human * vel)
    assisted = (np.sign(human) == np.sign(motor)).astype(float)

    total = power.sum()
    if total == 0:
        return 0.0
    return float(100.0 * np.sum(assisted * power) / total)


def infer_y_score(result_csv_path, model_path=None, n_steps=None,
                  calibration_path=None):
    """
    Deterministic interactivity score in [0, 100].

    Signature is kept compatible with the previous versions so that
    service_inter.py does not change: `model_path`, `n_steps` and
    `calibration_path` are accepted and ignored (the score needs no model
    and no calibration file any more).
    """
    df = pd.read_csv(result_csv_path)
    return score_dataframe(df)


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

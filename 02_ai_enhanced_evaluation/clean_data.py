"""
센서 이상치 제거 -> data/*_clean.csv 생성.

두 코호트의 오염원이 다르다.
  환자   : 모터 전류 포화. |Current| 가 174,433 부근 고정값으로 튄다. 정상 관측 최대는 1,853 이고
           그 사이 5,000~100,000 구간에는 샘플이 하나도 없어 경계가 명확하다.
  정상인 : 로드셀(Weight) 스파이크. Inter_Torque -> Human_Torque 로 전파된다.

전류 오염은 tau_motor = -0.00186*I + 0.26813 (Manuscript Eq.2) 를 통해 Motor_Torque 로,
다시 tau_human = tau_inter - tau_motor - tau_grav (Eq.4) 를 통해 Human_Torque 로 번진다.
그래서 전류만 걸러도 파생 토크가 함께 정리된다.

보간하지 않고 샘플 단위로만 제거한다. 시퀀스는 학습 단계에서 세그먼트 내 구간으로 다시
만들어지므로, 여기서 값을 채우면 없던 신호를 만드는 셈이 된다.

    python clean_data.py            # 리포트만
    python clean_data.py --write    # data/*_clean.csv 저장
"""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# 물리적으로 불가능하거나 센서 포화가 명백한 구간만 자른다.
# 경계는 각 신호의 p0.1~p99.9 와 실제 관측 극단값 사이의 빈 구간에서 골랐다.
LIMITS = {
    "Angle": (0.0, 180.0),          # 주관절 가동범위. 관측 p99.9 = 175
    "Current": (-10_000, 10_000),   # 정상 최대 |1,853| vs 포화 174,433
    "Weight": (-10_000, 10_000),    # 로드셀 스파이크
    "Human_Torque": (-30.0, 30.0),  # 관측 p0.1~p99.9 = -16.0 ~ 6.5
    "Inter_Torque": (-50.0, 50.0),
    "Motor_Torque": (-50.0, 50.0),
}

SEQUENCE_LENGTH = 20  # 학습 시 슬라이딩 윈도우 길이. 이보다 짧은 세그먼트는 쓸 수 없다.


def clean(df, name):
    keep = pd.Series(True, index=df.index)
    print(f"\n[{name}] {len(df):,} samples")
    for col, (lo, hi) in LIMITS.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        bad = (v < lo) | (v > hi) | v.isna()
        if bad.any():
            print(f"  {col:15} {int(bad.sum()):6,} 제거 ({100.0 * bad.mean():.4f}%)"
                  f"  관측 {v.min():.1f} ~ {v.max():.1f}")
        keep &= ~bad

    dropped = int((~keep).sum())
    print(f"  -> {dropped:,} 제거 ({100.0 * (~keep).mean():.4f}%), {int(keep.sum()):,} 유지")

    key = ["Number", "Task", "Repeat", "State"]
    sizes = df[keep].groupby(key).size()
    print(f"  세그먼트 {df.groupby(key).ngroups} -> {len(sizes)} "
          f"(길이>={SEQUENCE_LENGTH} 인 것 {int((sizes >= SEQUENCE_LENGTH).sum())})")
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    for fname, label in [("result_normal.csv", "정상인"), ("result_patient.csv", "환자")]:
        df = pd.read_csv(os.path.join(DATA, fname))
        df = df[(df["Repeat"] >= 1) & (df["Repeat"] <= 20)]
        out = clean(df, label)
        if args.write:
            path = os.path.join(DATA, fname.replace(".csv", "_clean.csv"))
            out.to_csv(path, index=False)
            print(f"  저장: {path}")


if __name__ == "__main__":
    main()

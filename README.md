# 엑소스켈레톤-사람 상호작용성 평가 (Interactivity Score)

엑소스켈레톤(팔꿈치) 시험 데이터로부터 **사람-기기 상호작용성 점수(0~100)** 를 산출하는
**결정론적(deterministic)** 평가 코드입니다.

---

## 왜 이 코드인가 (배경)

기존 `wearable_ui_frontend/wde/interactivity` 에는 **SVR 모델**(`svr_y_score_model.joblib`)이
들어가 있었는데, 다음 문제가 있었습니다.

- 학습 샘플 **24개**, 과적합 하이퍼파라미터(`gamma=100, C=1000`)로 반응면이 매우 뾰족함
- 그 결과 **입력이 조금만 달라져도 점수가 5~8점씩 출렁임** (재현 불가)
- 토크 기반 평가가 아니라, 각도 곡선의 기울기·힘 통계 6개 feature를 쓰는 정체불명 모델

이 코드는 그 SVR을 버리고, **원래의 결정론적·토크 기반 평가**
([donghee/wde-interactivity](https://github.com/donghee/wde-interactivity), Module A)를
시뮬레이션 데이터에 맞게 이식한 것입니다. **같은 입력 → 항상 같은 점수**가 보장됩니다.

---

## 평가 방법

시뮬레이터가 출력한 `human_torque` 를 **그대로** 사용해, 동작 구간(State)·각도에 따라 평가합니다.

1. **human_torque 직접 사용** — 시뮬레이션 CSV에 이미 들어 있음 (토크 재계산 불필요)

2. **동작 State 판정** (샘플별)
   - `angular_velocity < 0` → **Flexion** (굴곡, 각도 감소 170°→64°)
   - `angular_velocity ≥ 0` → **Extension** (신전, 각도 증가 64°→170°)
   - 실측(physical) CSV는 `U_Status` 컬럼을 직접 사용

3. **방향 (State에 따라 반대)** — 원본 `Evaluation_Table2.csv` 검증 결과,
   `(Range_1 − Range_0)` 부호가 State로 **정확히** 갈립니다 (예외 0):

   | State | 방향 | 의미 |
   |---|---|---|
   | **Flexion** | human_torque **클수록 高점** (73/73) | — |
   | **Extension** | human_torque **작을수록 高점** (55/55) | 보조가 편안할수록 사람 토크↓ |

4. **정규화 → 점수**
   - 각 샘플을 State별 고정 범위 `[lo, hi]`(→ `calibration.json`)로 0~1 정규화
   - 방향 적용 후 `clip(0,1)` → 전체 평균 → `× 100`

```
Filtered_Score_i = clip( (human_torque_i − lo) / (hi − lo), 0, 1 )      # Flexion
Filtered_Score_i = clip( (hi − human_torque_i) / (hi − lo), 0, 1 )      # Extension
Interactivity Score = mean(Filtered_Score) × 100
```

> **결정론 보장**: `calibration.json` 의 범위는 기준 데이터셋에서 **한 번** 산출해 고정합니다.
> 따라서 단일 파일 채점은 다른 파일 유무와 무관하게 항상 동일한 값을 냅니다.

---

## 사용법

```bash
# 단일 시험 채점
python interactivity_score.py data/augmented_01.csv
# -> Interactivity Score: 56.0 / 100   (augmented_01.csv)

# 파이썬에서 호출
python -c "from interactivity_score import interactivity_score as s; print(s('data/augmented_01.csv'))"
```

예시 점수 (virtual/task1 augmented):

| 파일 | 점수 |
|---|---|
| augmented_01.csv | 56.0 |
| augmented_02.csv | 47.8 |
| augmented_03.csv | 44.9 |

---

## 입력 데이터 형식

시뮬레이션(augmented) CSV 컬럼:

```
timestamp_sec, step, elbow_angle_rad, inter_force,
target_angle_rad, angular_velocity, human_torque, motor_torque
```

채점에 필요한 컬럼은 **`human_torque`** 와 (State 판정용) **`angular_velocity`** 입니다.
실측 데이터처럼 `U_Status`(Flexion/Extension) 컬럼이 있으면 그것을 우선 사용합니다.

---

## 캘리브레이션 재생성

기준 데이터가 바뀌면 범위를 다시 산출합니다.

```bash
python build_calibration.py "../AI_evaluation/new_patient_data/virtual/task1/augmented_*.csv" \
    --lo 5 --hi 95 --out calibration.json
```

현재 `calibration.json` 은 **virtual/task1 augmented 36개 파일(1258 샘플)** 기준입니다.

> **참고 (Extension 보정)**: 현재 기준 데이터(virtual/task1)는 전부 Flexion이라
> Extension 범위는 Flexion 크기를 잠정 사용합니다(`provisional: true`).
> Extension 시뮬 데이터가 확보되면 `build_calibration.py` 로 재보정하세요.

---

## 파일 구성

| 파일 | 설명 |
|---|---|
| `interactivity_score.py` | 메인 채점기 (CLI + 함수 API) |
| `build_calibration.py` | 기준 데이터에서 `calibration.json` 생성 |
| `calibration.json` | State별 고정 정규화 범위 + 방향 |
| `Evaluation_Table2.csv` | 원본 평가표 (State별 방향 근거) |
| `data/augmented_*.csv` | 예시 입력 |

---

## 의존성

```
pandas, numpy
```

(원본 SVR 방식의 `scikit-learn`, `joblib` 은 더 이상 필요 없습니다.)

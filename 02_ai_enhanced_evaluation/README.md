# 02_ai_enhanced_evaluation — AI 기반 상호작용 평가 (ICRA 투고용)

규칙 기반 점수([`../01_rule_based_evaluation/`](../01_rule_based_evaluation/))를 확장하여,
정상인과 환자의 토크 시계열로부터 **사람-엑소 상호작용의 성격**을 학습으로 추정합니다.

두 가지 구성요소로 이루어집니다.

| 구성요소 | 스크립트 | 출력 |
|---|---|---|
| A. Ordinal regression | `ordinal_regression_success.py` | 상호작용을 Resistance–Assistance 연속선 위의 값으로 예측 |
| B. 환자별 적합성 평가 | `suitability_calculator_4metrics.py` | 환자마다 어떤 제어 모드가 적합한지 4개 지표로 산출 |
| C. Sim2Real 검증 | [`sim2real_validation/`](sim2real_validation/) | 시뮬레이터가 실제 환자 움직임을 재현하는지 검증 (R² ≥ 0.97) |

---

## A. Ordinal Regression — Resistance–Assistance Continuum

기존 방식은 보조(assistance)를 **0~1 이진 분류**로 다뤘습니다. 그러나 실제 상호작용은
"로봇이 저항하는가 / 순응하는가 / 강하게 끌고 가는가"라는 **순서(order)를 가진 연속선**에
가깝습니다. 이를 단일 ordinal 값 `[-1, +1]`로 모델링합니다.

```
Resistance(-1)  →  Compliant(0)  →  Very Compliant(+0.5)  →  Stiff(+1)
```

정상인과 환자 실험은 **별도로 수행**되었고 Task 번호 방향이 서로 반대입니다.
정상인은 T1(약)→T3(강), 환자는 T1(강)→T3(저항)입니다.

| Task | 정상인 | 환자 |
|---|---|---|
| Task 1 | Highly compliant (+0.5) | **Stiff (+1)** |
| Task 2 | Compliant (0) | Compliant (0) |
| Task 3 | **Stiff (+1)** | **Resistant (−1)** |

이 매핑의 근거(원 논문 본문·제어 코드·궤적 반복성 검증)와, 파생 토크의 부호로
모드를 추론하면 안 되는 이유는 [LABEL_DETERMINATION.md](LABEL_DETERMINATION.md)에 정리했습니다.

> **저항 영역(−1)은 환자 Task 3 단독 출처**입니다(393 세그먼트). 정상인 라벨은 전부 0 이상이므로,
> 정상인 데이터만으로는 이 축의 보조측 절반만 채워집니다.

### 모델

- **Transformer backbone** (d_model 128, 4 layers, 8 heads, dropout 0.2) + ordinal head
- 입력: 길이 20 시퀀스 × 13 feature
  - 기본 10종: `Weight, Angle, Current, Motion, Human_Torque, Motor_Torque, Inter_Torque, Gravity_Torque, State, Repeat`
  - 파생 2종: `Human_Torque_Change`(Δτ), `Angle_Change`(각속도)
  - 임상 1종: `MMT` 등급 (3~5 → 0~1 정규화)
- Task one-hot을 **제외**하여, 순수 시계열 패턴만으로 continuum을 예측하도록 강제
- 2-stage 학습(backbone 사전학습 → 상위 레이어 fine-tune) + motor torque 마스킹
- **LOSO (Leave-One-Subject-Out)** 교차검증, 환자 10명

### 결과 (`results/ordinal_learning_summary.csv`)

| 지표 | 값 (mean ± std) |
|---|---|
| C-index | **0.813 ± 0.127** |
| Cumulative AUC | **0.820 ± 0.134** |
| Spearman ρ | 0.652 ± 0.246 |
| Kendall τ | 0.582 ± 0.232 |
| R² | 0.257 ± 0.486 |
| RMSE | 0.663 ± 0.236 |

- **10명 중 9명에서 Task 순서를 올바르게 복원** (`Order_Correct`, `results/patient_results_ordinal_learning.csv`)
- 순위 지표(C-index 0.81, AUC 0.82)가 회귀 지표(R² 0.26)보다 높음 → 절대값보다 **순서 판별에 강함**

## B. 환자별 적합성 평가 (4 metrics)

환자마다 어떤 제어 모드가 맞는지를 4개 지표로 정량화합니다.

| 지표 | 정의 |
|---|---|
| Torque Stability | Hilbert 변환 포락선 기반 토크 안정성 |
| Angle Smoothness | jerk(3차 미분) 기반 각도 부드러움 |
| ROM Suitability | 가동범위 적합도 |
| ST Suitability | Human torque 안정성 |

가중치는 고정값이 아니라 **환자별 적응형(adaptive)** 으로 계산합니다.
Task 간 분산이 큰(변별력 있는) 지표에 높은 가중치를 주고, 서로 상관이 높은 지표에는
페널티를 부여해 중복을 억제합니다.

### 결과 (`results/suitability_results_4metrics_adaptive.csv`)

| Task | 평균 적합도 |
|---|---|
| Task 1 (Stiff) | 0.200 |
| **Task 2 (Compliant)** | **0.905** |
| Task 3 (Resistant) | 0.229 |

환자군 전반에서 Compliant 모드가 가장 적합하게 산출되며, 개인별로는 편차가 존재합니다
(예: 환자 6은 Task 3 적합도 0.789로 저항 모드에도 적응 가능).

---

## 실행 방법

```bash
cd 02_ai_enhanced_evaluation

# 0. 센서 이상치 제거 (권장 선행 단계)
python clean_data.py           # 리포트만
python clean_data.py --write   # data/*_clean.csv 생성

# A. Ordinal regression (GPU 필요)
python ordinal_regression_success.py
#  -> results/patient_results_ordinal_learning.csv
#  -> results/ordinal_learning_summary.csv

# B. 환자별 적합성 (CPU로 충분)
python suitability_calculator_4metrics.py
#  -> results/suitability_results_4metrics_adaptive.csv
```

## 데이터

| 경로 | 내용 |
|---|---|
| `data/result_normal.csv` | 정상인 10명 (Number 1,2,3,7,10,17,21,22,23,24 / Repeat 1–20) |
| `data/result_patient.csv` | 환자 10명 |
| `data/user_info_patient.csv` | 환자 체중·팔길이 (MMT 등급 매핑에 사용) |
| `data/patient_data/` | 환자별 task별 원본 30개 CSV (`{ID}_Task{N}.csv`) |

> `data/result_normal.csv`는 원본 전체(635,883행, 128MB)에서 스크립트가 실제로 사용하는
> 대상자·반복 구간만 추출한 것입니다(100,365행). 스크립트가 동일 조건으로 다시 필터링하므로
> **결과는 원본과 완전히 동일**하며, GitHub 파일 크기 제한을 피하기 위한 조치입니다.

### 센서 이상치

두 코호트의 오염원이 다릅니다. `clean_data.py`로 제거합니다(보간하지 않고 샘플 삭제만).

| 코호트 | 오염원 | 제거량 |
|---|---|---|
| 정상인 | 각도 이상치 (max 1007°) | 1샘플 (0.001%) |
| 환자 | **모터 전류 포화** (\|I\| ≈ 174,433 고정값) | **1,526샘플 (3.38%)** |

환자 전류는 정상 관측 최대 1,853과 포화값 174,433 사이에 **샘플이 하나도 없어** 경계가 명확합니다.
전류 오염은 Eq.2(`τ_motor = −0.00186·I + 0.26813`)를 거쳐 Motor_Torque로, Eq.4를 거쳐
Human_Torque로 전파됩니다. 제거하면 환자 Task 3의 `|Human_Torque|` 평균이 **18.66 → 4.47**로
4배 감소하므로, 위 표의 성능 수치는 **재측정이 필요합니다**.

세그먼트는 하나도 소실되지 않습니다(정상 1,200 / 환자 1,173 전부 유지).

## 의존성

```
torch (CUDA), pandas, numpy, scipy, scikit-learn
```

# 엑소스켈레톤-사람 상호작용성 평가 (Interactivity Score)

`wde/interactivity/extend` 의 **점수 산출 알고리즘만** 교체한 버전입니다.
구조(파일/그래프/서비스)는 원본 그대로 두고, 점수 산출부만
**결정론적·참여(engagement) 기반 점수**로 바꿉니다.

---

## 왜 바꿨나

1. **1차 (SVR → 토크 min-max)**: 기존 SVR 모델(`svr_y_score_model.joblib`, 학습
   24샘플, `gamma=100/C=1000` 과적합)은 입력이 조금만 달라져도 점수가 5~8점씩
   출렁였습니다(재현 불가). 이를 결정론적 토크 min-max 정규화로 바꿨습니다.

2. **2차 (토크 min-max → 참여 기반)**: min-max 방식은 M2/M6 실측 시험에서
   **점수가 4~5점밖에 안 벌어졌습니다**. 원인 두 가지:
   - **포화**: 정규화 범위(`±0.8 N·m`)가 시뮬 augmented 데이터 기준이라, 스케일이
     다른 M2/M6(`human_torque ~1–10 N·m`)는 거의 전 샘플이 상한 1.0에 붙어버림.
   - **신호 낭비**: `human_torque` 하나만 봄. 정작 두 조건을 가르는 신호
     (모터 협조도, 상호작용 파워)를 무시.

   이 버전은 **보조 파워 비율(Assisted Power Ratio)** 로 교체합니다.

---

## 점수 산출 방법 — 보조 파워 비율 (Assisted Power Ratio)

사람이 전달한 기계적 파워 중, **모터가 같은 방향으로 보조하는 동안** 발생한 비율.

```
power_i     = |human_torque_i · angular_velocity_i|          # 사람이 낸 순간 파워
assisted_i  = 1  if sign(human_torque_i) == sign(motor_torque_i)  else 0   # 모터 협조 여부
Interactivity Score = 100 × Σ(assisted_i · power_i) / Σ(power_i)     # 0~100
```

하나의 **파라미터 없는(parameter-free)** 지표가 세 요소를 동시에 보상합니다.

| 요소 | 어떻게 반영되나 |
|---|---|
| **참여** (사람이 힘을 씀) | `human_torque` 가 클수록 그 샘플의 가중치(power)↑ |
| **파워 전달** (빠른 동작) | `angular_velocity` 가 클수록 가중치↑ |
| **협조** (모터 보조) | 모터 토크가 사람 토크와 같은 부호일 때만 assisted=1 |

- **결정론적**: 모델·난수·캘리브레이션 파일 없이 순수 산술 → 같은 입력 → 항상 같은 점수.
- **포화 없음**: 비율이라 스케일에 무관, 어떤 토크 스케일에서도 [0,100]을 자연스럽게 씀.

### 실측 결과 (M2 vs M6)

| 시험 | 점수 |
|---|---|
| **M2** | **68.1 / 100** |
| **M6** | **96.9 / 100** |
| 격차 | **+28.8** (기존 min-max 방식은 46.5 vs 51.7, 격차 5.2) |

(M2/M6은 근육 모델 파라미터만 다른 두 조건: M6가 사람-모터 협조율 0.89·사람 파워 7.8,
M2는 0.63·0.35 로 참여/협조가 낮음 → M6가 더 높은 상호작용성으로 채점됨.)

---

## 무엇이 바뀌고, 무엇이 그대로인가

| 항목 | 상태 |
|---|---|
| `extend/inter_score.py` 의 `infer_y_score()` | **변경** — 참여(보조 파워 비율) 기반 |
| `extend/inter_score.py` 의 `inter_graph()` | **그대로** (각도-시간 그래프 동일) |
| `service_inter.py` (`/score`, `/graph` 엔드포인트) | **그대로** (한 줄도 안 바뀜) |
| `extend/calibration.json`, `extend/build_calibration.py` | **삭제** (min-max 전용, 더 이상 불필요) |

> `infer_y_score()` 는 `model_path`, `n_steps`, `calibration_path` 인자를 **받기만 하고
> 무시**하므로 `service_inter.py` 의 호출부는 수정 없이 그대로 동작합니다.

---

## 사용법

```bash
# 단일 시험 채점 (0706_new/ 안에서)
python inter_score.py data/M2_상호작용성/result.csv
# -> Interactivity Score: 68.1 / 100
python inter_score.py data/M6_상호작용성/result.csv
# -> Interactivity Score: 96.9 / 100
```

웹 서비스(`service_inter.py`)는 기존과 동일하게 `/api/interactivity/score`,
`/api/interactivity/graph` 를 제공합니다.

### 기존 프로젝트에 적용 (drop-in)

1. `wde/interactivity/extend/inter_score.py` 를 이 버전으로 교체
2. `service_inter.py` 는 그대로 (수정 불필요)

> **입력 요건**: `result.csv` 에 `human_torque`, `motor_torque`, `angular_velocity`
> 컬럼이 있어야 합니다(시뮬레이터 출력).

---

## 파일 구성

| 파일 | 설명 |
|---|---|
| `inter_score.py` | `infer_y_score()`(보조 파워 비율) + `inter_graph()`(그래프, 원본 유지) |
| `Evaluation_Table2.csv` | 원본 평가표 (참고) |
| `data/M2_상호작용성/`, `data/M6_상호작용성/` | 실측 시험 데이터 |
| `data/augmented_01.csv` | 예시 입력 |

## 의존성

```
pandas, numpy, matplotlib          # (scikit-learn, joblib 불필요)
```

# New — 결정론적 토크 기반 상호작용성 점수

`wde/interactivity/extend/` 의 **SVR 점수 알고리즘을 대체**하는 새 버전입니다.
기존 파일은 건드리지 않고 이 `New/` 폴더로만 추가했습니다.

## 왜

기존 `extend/inter_score.py` 의 `infer_y_score()` 는 SVR 모델
(`svr_y_score_model.joblib`, 학습 24샘플, gamma=100/C=1000 과적합)을 써서
**입력이 조금만 달라져도 점수가 5~8점 출렁이는**(재현 불가) 문제가 있었습니다.

이 버전은 원래의 **결정론적 토크 기반 평가**
([donghee/wde-interactivity](https://github.com/donghee/wde-interactivity), Module A)를
이식했습니다. **같은 입력 → 항상 같은 점수.**

## 점수 산출

시뮬레이터의 `human_torque` 를 그대로 사용합니다.

1. State: `angular_velocity < 0` → Flexion, 아니면 Extension (`U_Status` 있으면 우선)
2. 방향 (원본 `Evaluation_Table2.csv` 로 검증, State로 완전 분리):
   - **Flexion**: human_torque 클수록 高점 (73/73)
   - **Extension**: human_torque 작을수록 高점 (55/55)
3. 정규화 → 점수:
   ```
   Flexion  : clip( (human_torque − lo) / (hi − lo), 0, 1 )
   Extension: clip( (hi − human_torque) / (hi − lo), 0, 1 )
   Interactivity Score = mean(Filtered_Score) × 100      # 0~100
   ```
   `lo, hi` 는 State별 고정 범위(`calibration.json`). 한 번 산출해 고정 → 결정론적.

예시: `data/augmented_01.csv` → **56.0 / 100** (반복 실행 시 항상 55.955848).

## 사용법

```bash
python inter_score.py data/augmented_01.csv
# -> Interactivity Score: 56.0 / 100   (augmented_01.csv)
```

## 기존 코드에 적용 (drop-in)

`infer_y_score()` 는 `model_path`, `n_steps` 인자를 받기만 하고 무시하므로
`service_inter.py` 는 **수정 불필요**합니다.

1. `New/inter_score.py` → `extend/inter_score.py` 로 교체
2. `New/calibration.json` → `extend/calibration.json` 추가
3. `extend/Regression.py`, `extend/svr_y_score_model.joblib` 삭제
4. `service_inter.py` 그대로

`inter_graph()`(각도-시간 그래프)도 이 파일에 원본 그대로 포함되어 있습니다.

## 캘리브레이션 재생성

```bash
python build_calibration.py "<augmented CSV glob>" --lo 5 --hi 95 --out calibration.json
```

> Extension 범위는 현재 기준 데이터(전부 Flexion)가 없어 Flexion 크기를 잠정 사용합니다(`provisional`).

## 파일

| 파일 | 설명 |
|---|---|
| `inter_score.py` | `infer_y_score()`(토크 기반) + `inter_graph()`(원본 그래프) |
| `calibration.json` | State별 정규화 범위 + 방향 |
| `build_calibration.py` | calibration 재생성 |
| `Evaluation_Table2.csv` | State별 방향 근거 (원본 평가표) |
| `data/augmented_01.csv` | 예시 입력 |

## 의존성

```
pandas, numpy, matplotlib       # (scikit-learn, joblib 불필요)
```

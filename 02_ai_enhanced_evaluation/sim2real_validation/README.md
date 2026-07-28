# Sim2Real Validation — 시뮬레이터가 실제 환자 움직임을 재현하는가

가상(시뮬레이터) 데이터로 학습·증강한 결과가 유효하려면, 먼저 시뮬레이터가 실제 환자의
움직임 특성을 재현한다는 근거가 있어야 합니다. 이 폴더는 **실측(physical) 12명**과
**가상(virtual) 36 trial**의 움직임 지표를 대조해 그 근거를 제공합니다.

## 비교 지표와 결과 (Task 1)

| 지표 | 의미 | R² |
|---|---|---|
| Sign Sum | 움직임 방향 | **1.0000** |
| Hilbert Envelope | 진동(포락선) 특성 | **0.9999** |
| RMS (Weight) | 힘 크기 | **0.9999** |
| Jerk | 각도 부드러움 | **0.9998** |
| ROM | 가동범위 | **0.9709** |

5개 지표 모두 R² ≥ 0.97로, 시뮬레이터가 실제 환자 움직임의 통계적 특성을 재현합니다.

> 단위는 서로 다릅니다(physical=degree, virtual=radian). 상관 기반 지표이므로 단위 차이는
> 결과에 영향을 주지 않으며, 회귀 기울기(slope)에 스케일 차이로 반영됩니다.

## 실행

```bash
cd 02_ai_enhanced_evaluation/sim2real_validation
python physical_virtual_comparison.py     # task 번호(1) 입력 → results/ 에 저장
```

## 데이터

| 경로 | 내용 |
|---|---|
| `data/physical/task{1,2,3}/` | 실측 환자 12명 (`W40xx_Task{N}.csv`) |
| `data/virtual/task1/` | 시뮬레이터 증강 데이터 36 trial + 원본 |
| `results/` | 지표별 R² 요약, trial별 상세, 산점도 |

> 실측 파일명은 익명화되어 있습니다(원본의 피험자 이니셜 제거). 분석에는 `W40xx` ID만
> 사용하므로 결과는 원본과 동일합니다.

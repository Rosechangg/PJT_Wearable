# PJT_Wearable — Upper-Limb Exoskeleton Interaction Evaluation

착용형 상지 엑소스켈레톤의 **사람-로봇 상호작용(pHRI) 평가** 연구 저장소입니다.
평가 방식에 따라 두 갈래로 나누어 관리합니다.

| 폴더 | 내용 | 용도 |
|---|---|---|
| [`01_rule_based_evaluation/`](01_rule_based_evaluation/) | 규칙(수식) 기반 결정론적 상호작용성 점수 | 과제 납품 / 웹 서비스 연동 |
| [`02_ai_enhanced_evaluation/`](02_ai_enhanced_evaluation/) | Transformer 기반 ordinal regression + 환자 적합성 평가 | 논문(ICRA) 투고용 |

---

## 01_rule_based_evaluation — 규칙 기반 평가

시뮬레이터/실장비의 `result.csv`(토크·각속도)만으로 0~100 점의 상호작용성 점수를
**결정론적으로** 산출합니다. 학습 모델이 없어 같은 입력이면 항상 같은 점수가 나오며,
Flask 서비스(`/api/interactivity/score`)에 그대로 연동됩니다.

- `wde/interactivity/0706_new/` — **최신본**. 보조 파워 비율(Assisted Power Ratio) 방식
- `wde/interactivity/New/` — 이전본. 토크 min-max 정규화 방식
- 자세한 내용은 [01_rule_based_evaluation/wde/interactivity/0706_new/README.md](01_rule_based_evaluation/wde/interactivity/0706_new/README.md)

## 02_ai_enhanced_evaluation — AI 고도화 평가

정상인/환자 데이터를 함께 학습해, 상호작용을 **Resistance–Assistance 연속선(-1 ~ +1)**
위의 순서형(ordinal) 값으로 예측합니다. 나아가 환자별로 어떤 제어 모드(task)가
적합한지 4개 지표로 산출합니다.

- 대상: 정상인 10명 + 환자 10명(MMT 3~5등급)
- 자세한 내용은 [02_ai_enhanced_evaluation/README.md](02_ai_enhanced_evaluation/README.md)

---

## 의존성

```
# 01_rule_based_evaluation
pandas, numpy, matplotlib

# 02_ai_enhanced_evaluation
torch (CUDA), pandas, numpy, scipy, scikit-learn
```

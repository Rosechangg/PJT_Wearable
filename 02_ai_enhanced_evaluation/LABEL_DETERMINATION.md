# 라벨 확정 문서 — (도메인, Task) → Continuum 매핑

작성일: 2026-07-28
결론: **기존 라벨이 맞다.** 두 코호트의 Task 번호가 반대인 것도 의도된 설계이며 문서로 확인됨.

---

## 0. 최종 라벨 (확정)

```
 Resistance(−1) ── Compliant(0) ── VeryCompliant(+0.5) ── Stiff(+1)
```

| Task | 정상인 | 환자 |
|---|---|---|
| **Task 1** | Highly compliant → **+0.5** | Stiff → **+1** |
| **Task 2** | Compliant → **0** | Compliant → **0** |
| **Task 3** | Stiff → **+1** | Resistant → **−1** |

**두 실험은 별도로 수행되었고 Task 번호 방향이 반대다.**
정상인은 T1(약)→T3(강), 환자는 T1(강)→T3(저항). 이는 오류가 아니라 서로 다른 프로토콜이다.

---

## 1. 근거 A — 정상인: 원 논문 본문에 명시

`journal_writing/Manuscript_Converted.md` (Experiment 2, 속도제어)

> **Task 1 (Highly compliant mode)** was configured with *reduced current limits and minimal
> impedance gains*, allowing the exoskeleton to follow the wearer's voluntary motion with
> **minimal robotic involvement**.
> **Task 2 (Compliant mode)** applied *moderate current and damping parameters*, introducing
> stabilizing counter-torque that provided **partial assistance**.
> **Task 3 (Stiff mode)** increased *current and velocity-tracking gains*, causing the
> exoskeleton to **rigidly enforce the reference trajectory**, maximizing robotic support
> while **minimizing user effort**.

→ 정상인 라벨 {+0.5, 0, +1} 과 정확히 일치.

## 2. 근거 B — 환자: 제어 코드로 확정

`interactivity_evaluation/data/patients_m2_m6_task1/M2_상호작용성/task.py`

**Task 1 = Stiff** (고정 속도, 힘 피드백 없음)
```python
if upperlimb.get_target_angle() > 90:
    upperlimb.set_velocity(-1.2)   # flexion
else:
    upperlimb.set_velocity(1.2)    # extension
```
`get_force()`를 호출하지만 제어에 쓰지 않고 로깅만 한다 → 사람의 힘과 무관하게 로봇이 구동.

**Task 2 = Compliant** (임피던스 제어)
```python
velocity = -0.6 + (delta_force - m*acceleration - k*(current_position - target_angle)) / c
```
`delta_force`(사람 힘), 관성 m, 스프링 k, 댐핑 c 를 사용 → 사람 힘에 반응.

→ 환자 라벨 T1=+1, T2=0 과 일치.

## 3. 근거 C — 궤적 반복성 (부호 규약과 무관한 독립 검증)

Stiff 모드는 로봇이 궤적·타이밍을 강제하므로 **반복 간 편차가 작아야** 한다.
정제 데이터(`*_clean.csv`)로 계산한 변동계수(CV):

**정상인** — 세그먼트 길이 CV가 라벨과 완벽히 단조 대응

| Task | ROM CV | 속도 CV | **세그먼트 길이 CV** | 라벨 |
|---|---|---|---|---|
| T1 | 0.239 | 0.307 | **0.537** ← 자유로움 | +0.5 Highly compliant |
| T2 | 0.226 | 0.342 | 0.171 | 0 Compliant |
| T3 | **0.188** | 0.299 | **0.056** ← 타이밍 강제 | **+1 Stiff** |

세그먼트 길이 CV 0.537 → 0.171 → **0.056**. Task 3에서 반복 지속시간이 거의 동일해지며,
이는 **로봇이 고정 궤적을 강제**한다는 직접 증거다. 라벨과 일치.

**환자** — ROM 변동성이 저항 모드를 지목

| Task | **ROM CV** | 속도 CV | 세그먼트 길이 CV | 라벨 |
|---|---|---|---|---|
| T1 | **0.176** ← 가장 일관 | 0.386 | **0.450** ← 가장 낮음 | **+1 Stiff** |
| T2 | 0.175 | 0.347 | 0.697 | 0 Compliant |
| T3 | **0.288** ← 가장 불규칙 | **0.448** | 0.461 | **−1 Resistant** |

T3에서 ROM·속도 변동이 모두 최대 → 사람이 저항에 맞서 싸우며 움직임이 불규칙해짐. 라벨과 일치.

---

## 4. ⚠️ 철회: "모터 토크가 작으니 Stiff가 아니다" 는 오판이었다

조사 중 나는 다음과 같이 판단했다가 철회한다.

> (오판) "정상 T3의 모터 토크가 0.37로 가장 작으니 Stiff일 수 없다"
> (오판) "환자 T3의 모터 토크가 3.66으로 가장 크니 모터 OFF일 수 없다"

**원인**: 모터 토크 산출식의 계수가 음수임을 몰랐다.

`Manuscript_Converted.md` 식 (2):
```
τ_motor ≒ −(0.00186) × I + 0.26813
```

데이터 회귀로 검증: `τ = −0.00198·I + 0.133` (보고서 값과 거의 일치)

| 전류 I | τ_motor |
|---|---|
| 0 | **+0.268** |
| +500 | −0.662 |
| −500 | +1.198 |

- **계수가 음수** → 전류 부호와 토크 부호가 반대. 따라서
  `sign(Motor_Torque × 각속도)` 로 계산한 "모터 보조율"은 **해석이 뒤집힌다.**
  환자 T3의 75.3%는 보조가 아니라 **저항**을 의미할 수 있다.
- **절편이 양수** → 전류가 0이어도 τ_motor = 0.268. 즉 "모터 OFF인데 토크가 있다"는 것은
  물리 현상이 아니라 **회귀식의 절편**이다.

원 논문 자체 점검 보고서(`journal_writing/Manuscript_Review_Report.md`)도 이 식을 문제 삼았다.
> "The negative coefficient is **counterintuitive**" / "positive constant term implies
> non-zero torque at zero current, which is **unusual**"

**교훈**: 파생 신호(motor/human torque)의 크기·부호로 제어 모드를 추론하지 말 것.
산출식의 규약에 의존하므로 오판 위험이 크다. **궤적 반복성처럼 규약과 무관한 운동학 지표**를 쓸 것.

---

## 5. 미해결 사항

| # | 항목 | 상태 |
|---|---|---|
| 1 | **환자 Task 3(저항)의 제어 설정값** | ❌ 어느 폴더에도 없음. `보고서/`, `1차시스템통합/`, `previous_work/`, `기술이전/` 전수 확인. 부산대병원 실장비 코드로 추정 |
| 2 | 시뮬레이터에 Task 3 미구현 | ✅ 확인 — `virtual/task3/` 빈 폴더. 시뮬레이터는 Task 1만 구현(공인인증이 Task1 한정이었기 때문) |
| 3 | 저항 모드가 능동(모터 ON, 역방향 토크)인지 수동(모터 OFF, 마찰만)인지 | ❌ 데이터는 전류가 높음(중앙값 713) → 능동 저항 시사. Notion 메모는 "모터 off" |

**단, 1·3이 미해결이어도 라벨은 사용 가능하다.** 라벨은 "T3가 저항 조건이었다"는 실험 설계
사실에 근거하며, 궤적 변동성(근거 C)이 이를 독립적으로 지지한다.

---

## 6. 논문 서사에 미치는 영향 (중요)

두 실험의 Task 번호가 반대이고 프로토콜이 다르므로, **다음 주장은 쓸 수 없다.**

❌ **"같은 제어(same control)가 사람에 따라 극성이 뒤집힌다"**
   → 정상인과 환자는 서로 다른 제어 조건을 받았다. 겹치는 모드는 {Compliant, Stiff} 뿐이고
     'Highly compliant'는 정상인에만, 'Resistant'는 환자에만 존재한다.

✅ **대신 다음은 쓸 수 있다.**

1. **축의 완성**: 정상인 라벨은 전부 0 이상(+0.5, 0, +1)이므로 정상인만으로는 continuum의
   **보조측 절반**만 채워진다. 저항 영역(−1)은 **환자 Task 3 단독 출처(393 세그먼트)**다.
   → *"저항 영역은 환자에게서만 나타난다. 정상인 데이터만으로는 이 축을 만들 수 없다."*
   이 주장은 프로토콜이 다르다는 사실과 **무관하게 성립**한다.

2. **동일 모드의 상이한 효과**: 두 코호트가 공유하는 Stiff와 Compliant에서,
   같은 명목 모드가 대상군에 따라 다른 상호작용 특성을 보이는지는 **검증 가능한 가설**이다.
   (아직 미검증 — 실험 목록에 추가)

---

## 7. 재학습 가능 여부

**가능하다.** 라벨은 확정이고 데이터는 정제되었다.

전제 조건:
- 정제본 사용: `result_normal_clean.csv`, `result_patient_clean.csv` (`clean_data.py` 로 생성)
- 환자 전류 포화 제거로 환자 T3 사람토크 |평균|이 **18.66 → 4.47** 로 변하므로
  기존 C-index 0.813 은 **재측정 필요**

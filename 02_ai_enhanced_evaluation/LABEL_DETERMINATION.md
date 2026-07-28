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

## 2. 근거 B — 환자: 제어 코드로 세 모드 전부 확정 ★

출처: [`controller_reference/wearable_robot_mujoco/elbow_vel_cmd.py`](controller_reference/)
(원본 https://github.com/donghee/wearable_robot_mujoco, MIT, commit `f4c4566`)

`control_logic()` 에 세 모드가 모두 정의되어 있다.

```python
m = 0; c = 100; k = 0

# Task 1 (Stiff)
velocity = -1.2                                                    # flexion

# Task 2 (Compliant)
velocity = -0.6 + (delta_force - m*acceleration - k*(pos-target)) / c

# Task 3 (Resistive)
velocity = +0.2 + (delta_force - m*acceleration - k*(pos-target)) / c
```

extension 은 세 모드 모두 `velocity = 1.2`.

### 모드 차이는 flexion 기저 속도(bias) 하나뿐이다

| Task | Mode | flexion bias | 성질 |
|---|---|---|---|
| 1 | Stiff | **−1.2** | 힘 피드백 **없음**. 사람과 무관하게 고정 속도 구동 |
| 2 | Compliant | **−0.6** | 굽히는 방향으로 절반만 보조 |
| 3 | Resistive | **+0.2** | **부호 반전** — 굽히려는 방향과 반대로 밀어냄 |

`m=0, k=0` 이므로 Task 2·3 은 실질적으로 `velocity = bias + delta_force/100` 인
순수 힘 추종 제어다(관성·스프링 항은 계수가 0이라 비활성).

### ★★ continuum 라벨이 제어식에서 직접 도출된다

```
제어 bias:   −1.2       −0.6       +0.2
              │          │          │
Continuum:    +1          0         −1
```

bias 가 단조 증가하고 continuum 이 단조 감소하는 **일대일 대응**이다.
즉 라벨 −1/0/+1 은 설계자가 임의로 붙인 값이 아니라 **제어 파라미터의 물리적 순서**를 반영한다.
그리고 **bias = 0 이 보조와 저항의 경계**이므로 continuum 의 0점에도 물리적 의미가 있다.

이는 UIST R1 이 제기한 척도 타당성 문제(*"fundamental issues with the score calculation"*)에
대한 직접적인 답이 된다. 논문에서 "우리 축은 제어 파라미터의 연속선과 대응한다"고 주장할 수 있다.

### 저항은 모터 OFF 가 아니라 능동 저항이다

Notion 메모의 "모터 off" 는 초기 구상이었고, 실제 구현은 **모터를 켠 채 반대 방향으로 미는**
능동 저항이다. 따라서 저항 조건에서 모터 전류가 높게 관측되는 것이 **정상**이며,
데이터(환자 T3 전류 중앙값 713)와 코드가 일치한다.

→ 환자 라벨 T1=+1, T2=0, T3=−1 이 세 모드 모두 코드로 확정.

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
| 1 | 환자 Task 3(저항)의 제어식 | ✅ **해결** — `elbow_vel_cmd.py` 에 `bias = +0.2` 로 정의 (근거 B) |
| 2 | 저항이 능동인지 수동인지 | ✅ **해결** — 능동 저항. 모터를 켠 채 역방향으로 밀어냄 |
| 3 | **Task 3 의 댐핑 계수 `c`** | ❌ `task.py` 에 `#c = 10000  # task 3` 주석이 남아 있음. 실제 실험이 `c=100` 인지 `c=10000` 인지 미확인. **c 가 커지면 저항이 강해지므로 논문에 정확히 기술 필요** |
| 4 | 정상인 3모드의 제어 파라미터 실수치 | ❌ 원 논문은 "reduced/moderate/increased current limits, impedance gains" 로만 서술. 구체적 수치 없음. UIST R1 이 지적한 항목 |

`c` 값(3번)은 저항 강도에만 영향을 주고 **모드의 순서(Stiff > Compliant > Resistive)는
바뀌지 않으므로 라벨은 확정 상태로 사용 가능하다.**

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

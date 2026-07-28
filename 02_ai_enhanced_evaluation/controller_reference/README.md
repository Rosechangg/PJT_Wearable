# controller_reference — 제어 모드 정의의 출처

라벨(`Stiff / Compliant / Resistive`)이 무엇을 뜻하는지는 이 제어 코드가 정의합니다.
분석 코드에서 참조할 일이 잦아 사본을 두었습니다.

## 출처

| | |
|---|---|
| 원 저장소 | https://github.com/donghee/wearable_robot_mujoco |
| 저자 | Donghee Park (dongheepark@gmail.com) |
| 라이선스 | **MIT** (`package.xml`, `setup.py`에 명시) |
| 가져온 커밋 | `f4c456698a49c09efd4a136a4b525546e2398079` (2026-07-07) |
| 가져온 날짜 | 2026-07-28 |

**이 폴더는 수정하지 않습니다.** 원본 그대로이며, 갱신이 필요하면 위 저장소에서 다시 가져옵니다.
메쉬·에셋 등 대용량 파일은 제외하고 제어 정의에 직접 관련된 소스만 포함했습니다.

---

## ★ 세 모드의 정의 — `wearable_robot_mujoco/elbow_vel_cmd.py`

`control_logic()` 안에 세 모드가 모두 들어 있고, 실험 시 해당 블록만 활성화합니다.

```python
m = 0        # 관성 계수
c = 100      # 댐핑 계수
k = 0        # 스프링 계수

# ── Task 1 (Stiff) ──
velocity = -1.2                                                   # flexion

# ── Task 2 (Compliant) ──
velocity = -0.6 + (delta_force - m*acceleration - k*(pos - target)) / c

# ── Task 3 (Resistive) ──
velocity = +0.2 + (delta_force - m*acceleration - k*(pos - target)) / c
```

extension 구간은 세 모드 모두 `velocity = 1.2` 로 같습니다.
`delta_force = sensor_force * 1000 / 9.8` (단위 변환).

### 모드 차이는 flexion 기저 속도(bias) 하나

| Task | Mode | flexion bias | extension | m | k | c |
|---|---|---|---|---|---|---|
| 1 | Stiff | **−1.2** | 1.2 | — | — | — |
| 2 | Compliant | **−0.6** | 1.2 | 0 | 0 | 100 |
| 3 | Resistive | **+0.2** | 1.2 | 0 | 0 | 100 |

- Task 1은 **힘 피드백이 없습니다.** 사람의 힘과 무관하게 고정 속도로 구동 → 로봇 주도.
- Task 2·3은 `m=0, k=0` 이므로 실질적으로 **`velocity = bias + delta_force/100`** 인
  순수 힘 추종 제어입니다. 관성·스프링 항은 코드에 있으나 계수가 0이라 비활성입니다.
- **Task 3에서 bias 부호가 음수에서 양수로 뒤집힙니다.** 사용자가 굽히려는(flexion) 방향과
  반대로 로봇이 밀어냅니다. 이것이 저항이며, **모터를 끄는 방식이 아니라 능동 저항**입니다.
  따라서 저항 조건에서 모터 전류가 높게 관측되는 것이 정상입니다.

### continuum 라벨과의 대응

```
제어 bias:   −1.2       −0.6       +0.2
              │          │          │
Continuum:    +1          0         −1
             Stiff    Compliant  Resistive
```

bias가 단조 증가하고 continuum이 단조 감소하는 일대일 대응입니다.
즉 라벨은 임의로 부여한 값이 아니라 **제어 파라미터의 물리적 순서를 반영**합니다.
bias = 0 이 보조와 저항이 갈리는 경계이므로, continuum의 0점에도 물리적 의미가 있습니다.

---

## ⚠️ 미확정: Task 3 의 댐핑 계수

`wearable_robot_mujoco/task.py` 에 다음 주석이 남아 있습니다.

```python
c = upperlimb.get_damping()
#c = 10000   # task 3
```

Task 3에서 `c` 를 100 → 10000 으로 올리는 변형이 있었던 것으로 보입니다.
`c` 가 커지면 `delta_force/c` 항이 작아져 사람 힘의 영향이 줄고 기저 속도(+0.2)가 지배하므로
**저항이 더 강해집니다.**

**환자 실험에 실제로 쓴 값이 `c=100` 인지 `c=10000` 인지 확인이 필요합니다.**
저항 강도가 달라지므로 논문에 정확히 기술해야 합니다.

---

## 관련 문서

- [../LABEL_DETERMINATION.md](../LABEL_DETERMINATION.md) — 라벨 확정 근거와 검증
- 규칙 기반 평가의 원본: https://github.com/donghee/wde-interactivity

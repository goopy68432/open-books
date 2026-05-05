---
title: "08. Newton's Method = 2차 근사 최소화 — 완전 유도"
slug: newton-2nd-order
order: 9
---

# 08. Newton's Method = 2차 근사 최소화 — 완전 유도

> **출제 근거**: 8주차 ★9 \"Newton = 2차 근사 minimum\", 퀴즈 25번 \"$L$에 대해 무엇을 하는가\"
> **시험 출제 방식**: \"Show that Newton's method on $L(\theta)$ corresponds to minimizing the second-order Taylor approximation of $L$ at the current point.\"

---

## 1. 왜 시험에 나오는가

- \"Newton 이 root-finding 인 줄만 알았는데 사실은 **loss 의 2차 근사를 minimize 하는 것**\" 이라는 통찰.
- GD, SGD, Adam 등 모든 optimizer 의 비교 기준.
- 8주차 핵심 통찰. 퀴즈로 직접 출제.

---

## 2. 사전 수학

### 2.1 [고2] Taylor 전개 (1변수)

$f$ 를 $\theta_0$ 근처에서 다항식으로 근사:

$$
f(\theta) \approx f(\theta_0) + f'(\theta_0)(\theta - \theta_0) + \tfrac{1}{2}f''(\theta_0)(\theta - \theta_0)^2 + O((\theta-\theta_0)^3)
$$

**Line-by-Line:**

| 항 | 의미 |
|------|------|
| $f(\theta_0)$ | 0차 (값) |
| $f'(\theta_0)(\theta-\theta_0)$ | 1차 (선형 변화) |
| $\frac{1}{2}f''(\theta_0)(\theta-\theta_0)^2$ | 2차 (곡률) |
| $O((\theta-\theta_0)^3)$ | 3차 이상 잔여항 (작음) |

### 2.2 [대학원] 다변수 Taylor 전개

$\theta \in \mathbb{R}^d$, $\theta_0$ 근처:

$$
L(\theta) \approx L(\theta_0) + \mathbf{g}^\top (\theta - \theta_0) + \tfrac{1}{2}(\theta - \theta_0)^\top H (\theta - \theta_0)
$$

**기호 해체:**

| 기호 | 정의 | 의미 |
|------|------|------|
| $\mathbf{g} = \nabla L(\theta_0)$ | gradient (벡터) | 1차 미분 |
| $H = \nabla^2 L(\theta_0)$ | Hessian (행렬, $d\times d$) | 2차 미분, $H_{ij} = \partial^2 L/(\partial\theta_i\partial\theta_j)$ |

### 2.3 [참고] 1차원 Newton (root-finding)

$f(\theta) = 0$ 을 푸는 Newton 갱신:

$$
\theta_{t+1} = \theta_t - \frac{f(\theta_t)}{f'(\theta_t)}
$$

**Optimization 에 적용**: $L$ 의 최솟값 = $L'(\theta) = 0$ 의 해. 위 식에 $f \leftarrow L'$:

$$
\theta_{t+1} = \theta_t - \frac{L'(\theta_t)}{L''(\theta_t)}
$$

이게 1차원 Newton-for-optimization.

---

## 3. 유도 — 2차 근사 최소화

### Step 1 — $\theta_0$ 에서 2차 Taylor

$$
\tilde{L}(\theta) \;:=\; L(\theta_0) + \mathbf{g}^\top(\theta-\theta_0) + \tfrac{1}{2}(\theta-\theta_0)^\top H (\theta-\theta_0)
\tag{1}
$$

$\tilde L$ 은 $\theta$ 의 **이차 함수** — 정확한 minimum 을 closed form 으로 구할 수 있음.

### Step 2 — $\tilde L$ 의 gradient = 0

$\Delta := \theta - \theta_0$ 로 치환:

$$
\tilde L(\theta_0 + \Delta) = L(\theta_0) + \mathbf{g}^\top \Delta + \tfrac{1}{2}\Delta^\top H \Delta
$$

$\Delta$ 로 미분 (07 토픽 vector gradient 공식):

$$
\nabla_\Delta \tilde L = \mathbf{g} + H \Delta
$$

(대칭 $H$ 가정. $H$ 가 비대칭이라도 Hessian 정의상 일반적으로 대칭 — Schwarz)

극값 조건:

$$
\mathbf{g} + H \Delta^* = 0 \;\Longrightarrow\; \Delta^* = -H^{-1}\mathbf{g}
\tag{2}
$$

### Step 3 — Newton Update Rule

$\theta^* = \theta_0 + \Delta^*$ 이므로:

$$
\boxed{\; \theta_{t+1} \;=\; \theta_t - H_t^{-1} \mathbf{g}_t \;}
\tag{3}
$$

**Line-by-Line:**

| 항 | 의미 |
|------|------|
| $\theta_t$ | 현재 파라미터 |
| $\mathbf{g}_t = \nabla L(\theta_t)$ | 현재 gradient |
| $H_t = \nabla^2 L(\theta_t)$ | 현재 Hessian |
| $-H_t^{-1}\mathbf{g}_t$ | \"2차 근사 함수의 정확한 minimizer 까지의 변위\" |

### Step 4 — 최솟값 검증 ($H \succ 0$)

$\tilde L$ 이 strictly convex 이려면 $H$ 가 **positive definite** 필요. 이때만 $\Delta^*$ 가 진짜 **최솟값**. $H$ 가 nonconvex 영역에서는 $-H^{-1}\mathbf{g}$ 가 saddle 또는 max 로 갈 수도 있음 (NN의 핵심 어려움).

---

## 4. \"무엇을 하는가\" 한 줄 요약 (퀴즈 25 핵심 답)

> **Newton's method는 매 iteration 마다 $L$ 을 현재 점에서 2차 함수로 근사하고, 그 2차 함수의 정확한 최솟값으로 점프한다.**
>
> 즉, **\"locally quadratic\" 이라는 가정 하에 한 번에 최적점으로 가는 방법**.

### 4.1 GD vs Newton 비교

| 방법 | Update | 사용 정보 | 의미 |
|------|--------|----------|------|
| GD | $\theta_{t+1} = \theta_t - \eta \mathbf{g}_t$ | 1차 (gradient) | 1차 근사를 \"학습률만큼\" 따라감 |
| Newton | $\theta_{t+1} = \theta_t - H_t^{-1}\mathbf{g}_t$ | 1차 + 2차 (Hessian) | 2차 근사의 minimizer 로 정확히 점프 |

> 💡 GD = $H = \frac{1}{\eta}I$ 라고 가정한 Newton — 곡률이 모든 방향에서 같다고 보는 것.

---

## 5. NN 에서 왜 Newton 을 못 쓰나 (8주차 \"한계\" 포인트)

### 5.1 차원의 저주

$d$ 차원 파라미터 → Hessian은 $d \times d$. NN의 $d$ 는 수백만~수십억.

| 문제 | 비용 |
|------|------|
| Hessian 저장 | $O(d^2)$ 메모리 |
| Hessian 계산 | $O(d^2)$ 또는 $O(d^3)$ |
| Hessian 역행렬 | $O(d^3)$ 시간 |

→ **불가능**. 따라서 NN 에서는 Newton 못 씀. \"Hessian 계산 불가 ($d^2$ 차원)\" — 8주차 ★8.

### 5.2 우회: Quasi-Newton, Adam

- L-BFGS: Hessian 의 low-rank 근사
- Adam: \"per-parameter adaptive learning rate\" — diagonal Hessian 근사라고 볼 수 있음
- → [12 GD/SGD/Adam](12_GD_SGD_Adam.md)

---

## 6. 모범 답안 템플릿

```
[Setup]
Given a loss L: R^d → R, current iterate θ_t, gradient g_t = ∇L(θ_t),
Hessian H_t = ∇²L(θ_t).
Goal: show Newton's update θ_{t+1} = θ_t - H_t^{-1} g_t comes from
minimizing the 2nd-order Taylor approximation of L at θ_t.

[Step 1 — 2nd-order Taylor approximation]
For Δ := θ - θ_t,
  L̃(θ_t + Δ) = L(θ_t) + g_tᵀ Δ + (1/2) Δᵀ H_t Δ.

[Step 2 — Minimize the quadratic approximation]
∇_Δ L̃ = g_t + H_t Δ.
Setting to zero,
  H_t Δ* = -g_t  ⇒  Δ* = -H_t^{-1} g_t.

[Step 3 — Update rule]
θ_{t+1} = θ_t + Δ* = θ_t - H_t^{-1} g_t.       ∎

[Optimality — strict convexity locally]
If H_t ≻ 0, the quadratic L̃ is strictly convex and Δ* is its unique
minimum. So Newton "jumps to the exact minimizer of the local
quadratic model." Far from convex regimes (e.g., neural nets),
H_t may not be PD, in which case Newton can step toward a saddle
or maximum — this is one reason Newton is rarely used in deep
learning, alongside the O(d²) memory and O(d³) inverse cost.
```

---

## 7. 자주 틀리는 함정

1. **\"Newton = root finding\" 으로만 답함**: 시험은 \"$L$에 무엇을 하는가\"를 물음 → \"$L$의 2차 근사를 minimize 한다\" 명확히.
2. **Hessian PD 조건 누락**: $H$ 가 PD 일 때만 Newton 이 \"최솟값으로 간다\". NN의 saddle 문제 한 줄 추가.
3. **\"왜 NN에서 안 쓰나\" 누락**: $O(d^2)$, $O(d^3)$ 비용 한 줄.
4. **GD와의 관계**: \"GD = $H \approx \frac{1}{\eta}I$ 가정의 Newton\" 한 줄 적으면 통합 시각 점수.
5. **Symbolic vs numerical**: $H$ 가 dense 면 inverse 가 비쌈 → quasi-Newton 언급 가능.

---

## 8. 연결 개념

- → [09 Backpropagation](09_Backprop_ChainRule.md): NN gradient 계산 (Hessian 못 쓰는 대안)
- → [12 GD/SGD/Adam](12_GD_SGD_Adam.md): Newton 의 실용적 근사들
- ← [07 Linear Regression](07_LinearReg_ClosedForm.md): convex 라 1번에 minimum (Newton 1step)

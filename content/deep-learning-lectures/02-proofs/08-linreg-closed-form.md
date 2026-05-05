---
title: "07. Linear Regression Closed Form — 완전 유도"
slug: linreg-closed-form
order: 8
---

# 07. Linear Regression Closed Form — 완전 유도

> **출제 근거**: 6주차 ★9 (Linear Regression as Linear-Restricted MAP), 8주차 퀴즈 27 직접
> **시험 출제 방식**: \"Derive the closed-form solution $\theta^* = (X^\top X)^{-1} X^\top y$ for linear regression. Justify each step.\"

---

## 1. 왜 시험에 나오는가

- **Convex 손실의 closed-form 해**가 가능한 거의 유일한 모델.
- $\nabla L = 0$ 을 끝까지 풀어보는 표준 패턴 (vector calculus).
- 6주차 \"Hypothesis Space Restriction\" + 8주차 \"Optimization\" 의 교차점.

---

## 2. 사전 수학

### 2.1 [고1] 행렬·벡터 곱

$X \in \mathbb{R}^{n\times d}$, $\theta \in \mathbb{R}^d$:

$$
X\theta \in \mathbb{R}^n, \quad (X\theta)_i = \sum_{k=1}^d X_{ik}\theta_k
$$

각 행 $\mathbf{x}_i \in \mathbb{R}^d$ 와 $\theta$ 의 inner product.

### 2.2 [대1] 노름의 제곱

$$
\|\mathbf{v}\|^2 = \mathbf{v}^\top \mathbf{v} = \sum_i v_i^2
$$

### 2.3 [대학원] Vector Gradient 핵심 공식 두 개

이 두 개만 외우면 됨:

$$
\nabla_\theta (\mathbf{a}^\top \theta) = \mathbf{a}, \qquad \nabla_\theta (\theta^\top A \theta) = (A + A^\top)\theta
$$

$A$ 가 대칭이면 $\nabla = 2A\theta$.

**왜 이 공식?**
- 첫 번째: $\mathbf{a}^\top \theta = \sum_i a_i \theta_i$ 는 $\theta_i$ 에 대해 $a_i$. 모은 벡터가 $\mathbf{a}$.
- 두 번째: $\theta^\top A\theta = \sum_{i,j} A_{ij}\theta_i\theta_j$. $\theta_k$ 에 대해 $\partial/\partial\theta_k = \sum_j A_{kj}\theta_j + \sum_i A_{ik}\theta_i = (A\theta)_k + (A^\top \theta)_k$.

### 2.4 [대학원] 행렬 미분 분배

$$
\|X\theta - y\|^2 = (X\theta - y)^\top (X\theta - y) = \theta^\top X^\top X \theta - 2 y^\top X \theta + y^\top y
$$

전개 디테일:

$$
\begin{aligned}
& (X\theta - y)^\top (X\theta - y) \\
&= (X\theta)^\top(X\theta) - (X\theta)^\top y - y^\top(X\theta) + y^\top y \\
&= \theta^\top X^\top X \theta - 2 (X\theta)^\top y + y^\top y \quad (\text{스칼라라 transpose 무관}) \\
&= \theta^\top X^\top X \theta - 2 y^\top X \theta + y^\top y
\end{aligned}
$$

---

## 3. 문제 설정

$D = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, $\mathbf{x}_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}$.

🟦 Hypothesis space: linear functions $h(\mathbf{x}) = \theta^\top \mathbf{x}$ (또는 $\mathbf{x}^\top\theta$).

🟢 Likelihood: Gaussian (→ MSE). [04 토픽](04_NLL_MSE_Gaussian_유도.md) 결과 사용.

행렬 표기:

$$
X = \begin{pmatrix} \mathbf{x}_1^\top \\ \vdots \\ \mathbf{x}_n^\top \end{pmatrix} \in \mathbb{R}^{n\times d}, \quad \mathbf{y} = \begin{pmatrix} y_1 \\ \vdots \\ y_n \end{pmatrix} \in \mathbb{R}^n
$$

---

## 4. 유도 체인

### Step 1 — Loss 정의 (MSE)

$$
L(\theta) \;=\; \frac{1}{2}\sum_{i=1}^n (y_i - \theta^\top \mathbf{x}_i)^2 \;=\; \frac{1}{2}\|\mathbf{y} - X\theta\|^2
\tag{1}
$$

> 💡 $\frac{1}{2}$ 는 미분 시 2가 떨어져 깔끔해지기 위한 관습 — argmin 보존.

### Step 2 — 전개 (2.4절 활용)

$$
L(\theta) = \frac{1}{2}\left[\theta^\top X^\top X \theta - 2 \mathbf{y}^\top X \theta + \mathbf{y}^\top \mathbf{y}\right]
\tag{2}
$$

### Step 3 — Gradient 계산

$\theta$ 에 대한 gradient:

$$
\nabla_\theta L \;=\; \frac{1}{2}\left[2 X^\top X \theta - 2 X^\top \mathbf{y}\right] \;=\; X^\top X \theta - X^\top \mathbf{y}
\tag{3}
$$

**Line-by-Line:**

| 항 | gradient | 사용 공식 |
|------|---------|----------|
| $\theta^\top X^\top X \theta$ | $2 X^\top X \theta$ | 2.3절 (대칭 $A = X^\top X$) |
| $-2\mathbf{y}^\top X \theta$ | $-2 X^\top \mathbf{y}$ | 2.3절 ($\mathbf{a} = X^\top \mathbf{y}$, $\mathbf{a}^\top\theta$) |
| $\mathbf{y}^\top \mathbf{y}$ | $0$ | $\theta$ 무관 |

> ⚠️ \"$\mathbf{y}^\top X \theta$\" 를 $\theta$ 로 미분 시 결과가 $X^\top \mathbf{y}$ (transpose 주의). 행벡터/열벡터 차원 맞추기.

### Step 4 — 1차 조건 ($\nabla = 0$)

$$
X^\top X \theta = X^\top \mathbf{y}
\tag{4}
$$

이것이 **Normal Equation**.

### Step 5 — 풀이

$X^\top X$ 가 invertible 이라 가정 ($X$ 의 columns 가 linearly independent, 즉 full column rank):

$$
\boxed{\; \theta^* \;=\; (X^\top X)^{-1} X^\top \mathbf{y} \;}
\tag{5}
$$

### Step 6 — 최솟값 검증

Hessian:

$$
\nabla^2_\theta L = X^\top X
$$

$X^\top X$ 는 항상 **positive semi-definite** (PSD): 임의 $\mathbf{v}$ 에 대해

$$
\mathbf{v}^\top X^\top X \mathbf{v} = \|X\mathbf{v}\|^2 \geq 0
$$

Full column rank 가정 하에 **positive definite** → $L$ 은 strictly convex → $\theta^*$ 는 **유일한 전역 최솟값**.

---

## 5. 의미 해석

### 5.1 기하학적 — 정사영 (Projection)

$X\theta^* = X(X^\top X)^{-1} X^\top \mathbf{y} = P_X \mathbf{y}$.

여기서 $P_X = X(X^\top X)^{-1}X^\top$ 는 **column space of $X$** 위로의 정사영 행렬.

> 💡 \"$\mathbf{y}$ 를 가능한 hypothesis 공간 (column space of $X$) 위로 가장 가까이 떨어뜨린 점\" — 이게 회귀의 본질.

### 5.2 통합 시각 — Hypothesis Space Restriction = MAP

- Hypothesis space를 \"linear function\"으로 제한 = strong prior on hypothesis class.
- 안에서 MLE = MSE minimization.
- 이것이 6주차 \"Linear Regression as Linear-Restricted MAP\".

→ [14 Hypothesis Space Restriction = MAP](14_Hypothesis_MAP.md)

---

## 6. $X^\top X$ 가 비가역일 때 (실전)

### 6.1 원인

- 데이터 < feature 수 ($n < d$)
- Feature 들이 collinear

### 6.2 해결: Ridge Regression (L2 정규화)

$$
L_{\text{ridge}}(\theta) = \frac{1}{2}\|\mathbf{y} - X\theta\|^2 + \frac{\lambda}{2}\|\theta\|^2
$$

같은 절차로:

$$
\theta^*_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}
$$

$\lambda > 0$ 이면 항상 invertible. \"Ridge = Gaussian prior on θ → MAP\" 시각.

> 💡 시험에서 Ridge 가 나오면 \"Gaussian prior $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$ 의 MAP\"이라고 답 가능.

---

## 7. 모범 답안 템플릿

```
[Setup]
X ∈ R^{n×d}, y ∈ R^n.  Hypothesis: h(x) = θᵀx, θ ∈ R^d.
Loss (MSE / Gaussian-NLL): L(θ) = (1/2) ‖y - Xθ‖².
Goal: closed-form θ* = argmin_θ L(θ).

[Step 1 — Expand L]
‖y - Xθ‖² = θᵀ XᵀX θ - 2 yᵀ X θ + yᵀy.

[Step 2 — Gradient]
Using ∇_θ(θᵀAθ) = 2Aθ for symmetric A, and ∇_θ(aᵀθ) = a:
∇_θ L = XᵀX θ - Xᵀy.

[Step 3 — First-order condition]
∇_θ L = 0  ⇒  XᵀX θ = Xᵀy   (the normal equation)

[Step 4 — Invertibility & solution]
Assume X has full column rank, so XᵀX is positive definite (and invertible).
θ* = (XᵀX)^{-1} Xᵀ y.

[Step 5 — Optimality]
Hessian = XᵀX, PSD in general, PD under full rank ⇒ L strictly convex,
so θ* is the unique global minimum.

[Geometric meaning]
The fitted vector Xθ* = X(XᵀX)^{-1}Xᵀ y projects y onto the column
space of X. Linear regression chooses the parameter whose prediction
is the closest point to y in the linear hypothesis class.

[Connection]
This corresponds to: Gaussian likelihood (→ MSE)
                  + linear hypothesis restriction (→ MAP-style strong inductive bias).
```

---

## 8. 자주 틀리는 함정

1. **Bias 항 누락**: $h(\mathbf{x}) = \theta^\top\mathbf{x} + b$ 인 경우 $\mathbf{x}_i$ 에 1을 augment 하거나 별도 처리 — 답안에서 \"absorb bias by augmenting x with 1\" 한 줄.
2. **$X^\top X$ 가역성 가정 누락** → 식이 정의 안 됨.
3. **Hessian = $X^\top X$ 가 항상 PD인 것처럼 적기** — full column rank 조건 명시.
4. **전개 시 transpose 실수**: $\mathbf{y}^\top X \theta$ 와 $\theta^\top X^\top \mathbf{y}$ 가 같다는 점 (스칼라라 transpose 무관) — 한 줄 명시.
5. **Closed form 가능 이유 (convex)**: NN과 달리 닫힌 형태가 가능한 유일한 이유는 **convex** 손실. 이걸 마지막 줄에 적으면 점수 ↑.

---

## 9. 연결 개념

- ← [04 NLL→MSE](04_NLL_MSE_Gaussian_유도.md): MSE 의 정당화
- → [08 Newton Method](08_Newton_2차근사.md): NN 에서는 closed form 없음 → iterative
- → [14 Hypothesis Space Restriction](14_Hypothesis_MAP.md): linear 제한의 의미
- → [16 Jacobian](16_Jacobian.md): vector 미분 도구

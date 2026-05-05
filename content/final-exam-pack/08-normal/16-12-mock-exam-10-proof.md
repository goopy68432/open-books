---
title: "딥러닝 이론 모의고사 #10 — 증명 문제 집중"
slug: 12-mock-exam-10-proof
order: 16
---

# 딥러닝 이론 모의고사 #10 — 증명 문제 집중

> 배점 100점 / 8문제 / A+ 결정하는 증명 능력 측정

---

## 문제 1. [12점] Rank-Nullity 정리 증명

$A \in \mathbb{R}^{m \times n}$일 때 $n = \text{rank}(A) + \text{nullity}(A)$를 증명하시오.

**(a)** [8점] Null Space의 기저를 잡고, 이를 $\mathbb{R}^n$의 기저로 확장한 뒤, 확장 부분의 상(image)이 Range의 기저가 됨을 보이시오.
**(b)** [4점] 이 정리가 "입력 차원 = 살아남는 차원 + 죽는 차원"이라는 직관적 의미와 어떻게 대응하는지 설명하시오.

---

## 문제 2. [12점] 대칭행렬의 고유값 성질

**(a)** [6점] 대칭행렬 $A = A^\top$의 고유값이 실수임을 증명하시오.
(힌트: $\lambda\|v\|^2 = v^*Av$와 켤레를 취하여 비교)

**(b)** [6점] 서로 다른 고유값에 대응하는 고유벡터가 직교함을 증명하시오.

---

## 문제 3. [15점] Gaussian → MSE 완전 증명

$y_i = h_\theta(x_i) + \epsilon_i$, $\epsilon_i \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \sigma^2)$에서 MLE가 MSE 최소화와 동치임을 증명하시오.

**요구사항**: 5단계 각각에 "왜 이 단계를 밟는가"를 반드시 서술 [15점]

---

## 문제 4. [12점] KL ≥ 0 증명 (Gibbs' Inequality)

**(a)** [4점] Jensen 부등식을 서술하시오 (볼록 함수에 대해).
**(b)** [8점] $-\log$가 볼록임을 보이고, Jensen을 적용하여 $KL(p\|q) \geq 0$을 증명하시오.

---

## 문제 5. [12점] Softmax Jacobian 완전 유도

**(a)** [4점] 왜 $i=j$와 $i\neq j$를 나누어야 하는지 설명하시오.
**(b)** [4점] $i=j$: 몫의 미분법으로 $\frac{\partial p_i}{\partial z_i} = p_i(1-p_i)$ 유도
**(c)** [4점] $i\neq j$: $\frac{\partial p_i}{\partial z_j} = -p_ip_j$ 유도 및 행렬 통합

---

## 문제 6. [12점] MAP → L2 Regularization 증명

Gaussian Prior $P(\theta) = \mathcal{N}(0, \sigma_p^2 I)$일 때,

**(a)** [6점] MAP 추정이 $\arg\min[\text{NLL} + \lambda\|\theta\|^2]$와 동치임을 증명하시오.
**(b)** [3점] $\lambda$와 $\sigma_p$의 관계를 명시하시오.
**(c)** [3점] Laplace Prior의 경우 어떤 정규화가 되는지 증명하시오.

---

## 문제 7. [12점] 정규방정식 유도

**(a)** [6점] $\min_w \|Xw - y\|^2$에서 $\nabla_w = 0$으로 정규방정식을 유도하시오.
**(b)** [3점] $\|Xw-y\|^2$이 $w$에 대해 볼록임을 보이시오.
**(c)** [3점] 볼록이므로 "미분=0 → 전역 최솟값"이 보장되는 이유를 설명하시오.

---

## 문제 8. [13점] Softmax via Lagrangian 유도

$\max_p \sum_i p_i z_i + \tau H(p)$ s.t. $\sum_i p_i = 1$에서 softmax가 유일한 해임을 증명하시오.

**(a)** [4점] 라그랑지안 구성 (왜 라그랑주: 등식 제약)
**(b)** [4점] ∂L/∂p_i = 0에서 p_i 풀기 (왜 미분=0: Fermat)
**(c)** [3점] 정규화 조건으로 Z 결정 (왜: 확률 공리)
**(d)** [2점] τ의 의미 (τ→0, τ→∞)

---
---

# 모범답안

## 답 1.
### (a)
1. $k = \text{nullity}(A)$로 놓고, $\mathscr{N}(A)$의 기저 $\{v_1,...,v_k\}$를 잡는다.
2. 기저 확장 정리에 의해 $\{v_1,...,v_k,u_1,...,u_r\}$을 $\mathbb{R}^n$의 기저로 확장 ($r=n-k$).
3. **$\{Au_1,...,Au_r\}$이 $\mathscr{R}(A)$를 생성함을 보인다**:
   임의의 $y \in \mathscr{R}(A)$에 대해 $y = Ax = A(\sum a_i v_i + \sum b_j u_j) = \sum b_j Au_j$ ($Av_i=0$이므로).
4. **$\{Au_1,...,Au_r\}$이 선형독립임을 보인다**:
   $\sum c_j Au_j = 0$ → $A(\sum c_j u_j) = 0$ → $\sum c_j u_j \in \mathscr{N}(A)$
   → $\sum c_j u_j = \sum d_i v_i$ (Null Space의 원소)
   → $\sum c_j u_j - \sum d_i v_i = 0$
   → 기저의 선형독립성에 의해 모든 $c_j = d_i = 0$.
5. $\text{rank}(A) = r = n-k = n - \text{nullity}(A)$ $\square$

## 답 2.
### (a)
$Av = \lambda v$ ($v \neq 0$). 양변에 $v^*$ (켤레전치) 곱:
$v^* Av = \lambda v^* v = \lambda\|v\|^2$

켤레를 취하면: $\overline{v^* Av} = \bar{\lambda}\|v\|^2$

$A = A^\top$ (실수 대칭)이면 $v^* Av$는 실수:
$(v^* Av)^* = v^\top A^\top \bar{v} = v^\top A\bar{v}$...
더 간단하게: $A$ 실수 대칭이면 $v^* Av = \overline{v^* Av}$ → $\lambda\|v\|^2 = \bar{\lambda}\|v\|^2$
$\|v\|^2 > 0$이므로 $\lambda = \bar{\lambda}$ → $\lambda$는 실수. $\square$

### (b)
$Au_1 = \lambda_1 u_1$, $Au_2 = \lambda_2 u_2$, $\lambda_1 \neq \lambda_2$

$\lambda_1 u_1^\top u_2 = (Au_1)^\top u_2 = u_1^\top A^\top u_2 = u_1^\top Au_2 = \lambda_2 u_1^\top u_2$
(왜 $A^\top = A$: 대칭 가정)

$(\lambda_1 - \lambda_2)u_1^\top u_2 = 0$

$\lambda_1 \neq \lambda_2$이므로 $u_1^\top u_2 = 0$ → 직교. $\square$

## 답 3.
(유도 체인 시트의 유도 #1과 동일 — 5단계 + 5개 "왜" 포함)

## 답 4.
### (b)
$(-\log x)'' = 1/x^2 > 0$ for $x > 0$ → $-\log$는 볼록.

$KL(p\|q) = \sum_x p(x)(-\log\frac{q(x)}{p(x)})$
$= \mathbb{E}_p[-\log(q(X)/p(X))]$

Jensen (f 볼록 → $E[f(X)] \geq f(E[X])$):
$\geq -\log\mathbb{E}_p[q(X)/p(X)]$
$= -\log\sum_x p(x) \cdot q(x)/p(x)$
$= -\log\sum_x q(x) = -\log 1 = 0$ $\square$

## 답 8.
### (a)
$\mathcal{L} = \sum_i p_i z_i - \tau\sum_i p_i\log p_i + \mu(1-\sum_i p_i)$
왜 라그랑주: $\sum p_i = 1$이라는 등식 제약 처리

### (b)
$\partial\mathcal{L}/\partial p_i = z_i - \tau\log p_i - \tau - \mu = 0$
왜 미분=0: Fermat, 최적점의 필요조건
$p_i = \exp((z_i-\mu-\tau)/\tau) = \exp(z_i/\tau)/Z$

### (c)
$\sum p_i = 1$: $Z = \sum_j \exp(z_j/\tau)$
왜: 확률의 정규화 공리

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)} \quad \square$$

### (d)
$\tau \to 0$: 최대 $z_i$만 $\exp \to \infty$, 나머지 $\to 0$ → one-hot (argmax)
$\tau \to \infty$: $z_i/\tau \to 0$, 모든 $\exp \approx 1$ → uniform (최대 불확실성)

---
title: "딥러닝 이론 모의고사 #12 — 실전 Final 시뮬레이션"
slug: 14-mock-exam-12-final-sim
order: 18
---

# 딥러닝 이론 모의고사 #12 — 실전 Final 시뮬레이션

> **이것이 진짜 시험이다** — 90분, 100점, 출제 비중 반영
> 유도 35% + 계산 25% + 개념 25% + 통합 15%

---

## 문제 1. [15점] ★★★★★ 킬러 유도: Gaussian → MSE → MAP → Ridge

**(a)** [8점] $y_i \sim \mathcal{N}(h_\theta(x_i), \sigma^2)$, i.i.d.일 때 MLE가 MSE 최소화와 동치임을 5단계로 증명하시오. 각 단계에 "왜"를 서술하시오.

**(b)** [7점] 여기에 Prior $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$를 추가하면 MAP가 Ridge Regression이 됨을 보이시오. $\lambda$와 $\sigma_p$의 관계를 명시하시오.

---

## 문제 2. [10점] ★★★ CE Loss 유도

$K$-클래스 분류에서 $y_i \sim \text{Cat}(h_\theta(x_i))$일 때:

**(a)** [7점] MLE를 통해 CE Loss를 유도하시오 (i.i.d. → log → NLL).
**(b)** [3점] 이진 분류($K=2$)일 때 BCE를 구체적으로 전개하시오.

---

## 문제 3. [10점] ★★★ Softmax

**(a)** [6점] 라그랑주 승수법으로 softmax를 유도하시오. (목적함수: $\sum p_iz_i + \tau H(p)$, 제약: $\sum p_i=1$)
**(b)** [4점] Softmax Jacobian $\partial p/\partial z$를 유도하고 행렬 형태로 쓰시오.

---

## 문제 4. [10점] ★★ 고유값과 SVD

**(a)** [4점] $A = \begin{bmatrix}5&1\\1&3\end{bmatrix}$의 고유값과 고유벡터를 구하시오.

**(b)** [3점] 이 행렬이 PD(양정치)인지 판별하시오.

**(c)** [3점] 스펙트럼 분해 $A = U\Lambda U^\top$를 쓰시오.

---

## 문제 5. [10점] ★★ 벡터 미분

**(a)** [4점] 정규방정식을 $\nabla_w\|Xw-y\|^2 = 0$으로부터 유도하시오.
**(b)** [3점] $X = \begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix}$, $y = (2,3,5)^\top$일 때 $\hat{w}$를 구하시오.
**(c)** [3점] 이 해가 전역 최솟값인 이유를 볼록성으로 설명하시오.

---

## 문제 6. [10점] ★★★ 베이즈 정리와 MLE/MAP

**(a)** [3점] 베이즈 정리를 유도하시오.
**(b)** [4점] 동전 15번 중 12번 앞면: MLE와 MAP(Beta(2,2))를 각각 구하시오.
**(c)** [3점] MLE vs MAP의 차이를 "과적합 방지" 관점에서 설명하시오.

---

## 문제 7. [10점] ★★★ 정보이론

**(a)** [3점] $H(p)$, $CE(p,q)$, $KL(p\|q)$를 정의하고 관계식을 쓰시오.
**(b)** [4점] $KL \geq 0$을 Jensen 부등식으로 증명하시오.
**(c)** [3점] $CE$ 최소화 = $KL$ 최소화 = $NLL$ 최소화 등가 체인을 설명하시오.

---

## 문제 8. [10점] ★★ Rank-Nullity + 가역성

**(a)** [5점] Rank-Nullity 정리를 증명하시오.
**(b)** [5점] 가역행렬의 동치 조건 5가지를 나열하고, rank=n ↔ 가역을 증명하시오.

---

## 문제 9. [8점] ★ 개념 문제

**(a)** [2점] CLT가 가우시안 노이즈 가정을 정당화하는 논거를 1문장으로 쓰시오.
**(b)** [2점] 내적 $e_{y_i}^\top \log h(x_i)$의 역할을 1문장으로 쓰시오.
**(c)** [2점] VJP가 역전파에 적합한 이유를 1문장으로 쓰시오.
**(d)** [2점] $\sqrt{D_Q}$ 스케일링의 이유를 1문장으로 쓰시오.

---

## 문제 10. [7점] ★ 빈칸 채우기

| 확률 가정 | Loss | 정규화 (Gaussian Prior) |
|---------|------|----------------------|
| Gaussian | (a) | (b) |
| Bernoulli | (c) | (d) |
| Categorical | (e) | (f) |

**(g)** [1점] 이 모든 것을 하나로 묶는 프레임워크의 이름은?

---
---

# 모범답안

## 답 1.
### (a)
**Step 1** (모델 설정, 왜: 가우시안 노이즈 가정):
$p(y_i|x_i,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-h_\theta(x_i))^2}{2\sigma^2}\right)$

**Step 2** (i.i.d. → 곱, 왜: 독립이므로 결합확률=개별곱):
$P(\mathbf{y}|X,\theta) = \prod_i p(y_i|x_i,\theta)$

**Step 3** (log, 왜: 곱→합, 수치안정, 단조증가):
$\log P = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i(y_i-h_\theta(x_i))^2$

**Step 4** (NLL, 왜: 최대화→최소화 변환):
$\text{NLL} = \frac{1}{2\sigma^2}\sum_i(y_i-h_\theta(x_i))^2 + C$

**Step 5** (상수 제거, 왜: θ에 무관): $\theta_{ML} = \arg\min \text{MSE}$ $\square$

### (b)
$\log P(\theta) = -\frac{\|\theta\|^2}{2\sigma_p^2} + C'$ (Gaussian Prior의 log)

MAP = $\arg\min[\text{NLL} - \log P(\theta)]$
$= \arg\min\left[\frac{1}{2\sigma^2}\sum(y_i-h(x_i))^2 + \frac{1}{2\sigma_p^2}\|\theta\|^2\right]$

$\lambda = \sigma^2/\sigma_p^2$로 놓으면: $\arg\min[\text{MSE} + \lambda\|\theta\|^2]$ = Ridge $\square$

## 답 4.
(a) $\det(A-\lambda I) = (5-\lambda)(3-\lambda)-1 = \lambda^2-8\lambda+14 = 0$
$\lambda = \frac{8 \pm \sqrt{64-56}}{2} = \frac{8 \pm 2\sqrt{2}}{2} = 4 \pm \sqrt{2}$
$\lambda_1 = 4+\sqrt{2} \approx 5.41$, $\lambda_2 = 4-\sqrt{2} \approx 2.59$

(b) 두 고유값 모두 양수 → **PD** ✓

(c) 고유벡터 계산 후: $A = U\begin{bmatrix}4+\sqrt{2}&0\\0&4-\sqrt{2}\end{bmatrix}U^\top$

## 답 5.
### (b)
$X^\top X = \begin{bmatrix}3&6\\6&14\end{bmatrix}$, $X^\top y = \begin{bmatrix}10\\23\end{bmatrix}$
$(X^\top X)^{-1} = \frac{1}{6}\begin{bmatrix}14&-6\\-6&3\end{bmatrix}$
$\hat{w} = \frac{1}{6}\begin{bmatrix}140-138\\-60+69\end{bmatrix} = \frac{1}{6}\begin{bmatrix}2\\9\end{bmatrix} = (1/3, 3/2)^\top$

## 답 6.
### (b)
$\theta_{ML} = 12/15 = 0.8$
$\theta_{MAP} = \frac{12+2-1}{15+2+2-2} = 13/17 \approx 0.765$
MAP이 0.5쪽으로 약간 당겨짐.

## 답 9.
(a) 수많은 독립 원인의 합이 가우시안으로 수렴하므로 노이즈가 정규분포를 따른다고 가정하는 것은 합리적이다.
(b) 정답 클래스의 로그 확률만 선택하는 인덱싱 연산이다.
(c) Loss가 스칼라 1개이므로 1회 backward로 모든 파라미터 그래디언트를 동시에 계산할 수 있기 때문이다.
(d) 내적 분산이 $D_Q$에 비례하여 커지므로 softmax 포화를 방지하기 위해 정규화한다.

## 답 10.
(a) MSE, (b) MSE + λ‖θ‖², (c) BCE, (d) BCE + λ‖θ‖², (e) CE, (f) CE + λ‖θ‖²
(g) **NLL (Negative Log-Likelihood) 프레임워크** (또는 MLE/MAP 프레임워크)

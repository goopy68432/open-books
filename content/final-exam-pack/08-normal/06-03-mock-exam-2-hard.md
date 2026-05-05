---
title: "딥러닝 이론 기말 모의고사 #2 (심화)"
slug: 03-mock-exam-2-hard
order: 6
---

# 딥러닝 이론 기말 모의고사 #2 (심화)

> **난이도**: 상 (A+ 타겟 학생용)
> **시험 형식**: 주관식, 논리 과정 서술 필수
> **배점**: 100점 (8문제)

---

## 문제 1. [15점] 통합 유도: 확률 가정에서 Loss까지

아래 표를 완성하시오. 각 칸에 대해 **수식과 유도 근거**를 서술하시오.

| | 가정하는 분포 | MLE 목적함수 (최소화) | MAP 목적함수 (Gaussian Prior) |
|---|---|---|---|
| 회귀 | (a) | (b) | (c) |
| 이진 분류 | (d) | (e) | (f) |
| 다중 분류 | (g) | (h) | (i) |

**(j)** [3점] (b), (e), (h) 모두 $-\log P(\text{data}|\theta)$의 특수한 형태임을 보이고, 이 통합적 관점의 의미를 서술하시오.

---

## 문제 2. [12점] 행렬 미분의 연쇄

2층 신경망 $f(x) = W_2 \sigma(W_1 x)$에서 Loss $L = \|f(x) - y\|^2$일 때,

**(a)** [4점] $\frac{\partial L}{\partial W_2}$를 유도하시오.

**(b)** [4점] $\frac{\partial L}{\partial W_1}$를 체인룰로 유도하시오. 이때 $\sigma'$의 야코비안이 어떤 형태인지 명시하시오.

**(c)** [4점] 이 유도가 역전파 알고리즘의 한 단계와 어떻게 대응하는지 설명하시오. "업스트림 그래디언트"와 "로컬 그래디언트"의 개념을 사용하시오.

---

## 문제 3. [12점] 베이즈 정리의 깊은 이해

동전 던지기에서 앞면 확률 $\theta$를 추정한다. 10번 던져 7번 앞면이 나왔다.

**(a)** [3점] MLE로 $\theta_{ML}$을 구하시오 (유도 과정 포함).

**(b)** [5점] Prior $P(\theta) = \text{Beta}(\alpha, \beta)$일 때 MAP를 유도하시오. Beta 분포의 형태 $P(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$을 사용하시오.

**(c)** [4점] $\alpha = \beta = 2$일 때 $\theta_{MAP}$를 구하고, MLE 결과와 비교하시오. 데이터가 100번 중 70번 앞면이면 $\theta_{MAP}$는 어떻게 변하는가? 이로부터 "데이터가 많으면 Prior의 영향이 줄어든다"는 명제를 정량적으로 설명하시오.

---

## 문제 4. [12점] KL Divergence와 정보이론

**(a)** [3점] $KL(p\|q) \neq KL(q\|p)$임을 간단한 예시로 보이시오 (KL은 대칭이 아님).

**(b)** [5점] 평균 $\mu$, 분산 $\sigma^2$가 고정되었을 때, 엔트로피를 최대화하는 분포가 가우시안임을 라그랑주 승수법으로 유도하시오. (제약: $\int p = 1$, $\int xp = \mu$, $\int(x-\mu)^2 p = \sigma^2$)

**(c)** [4점] 이 결과가 "왜 가우시안 노이즈를 가정하는가"에 대한 또 다른 정당화인 이유를 CLT와 비교하여 설명하시오.

---

## 문제 5. [12점] SVD와 실전 응용

행렬 $A \in \mathbb{R}^{m \times n}$의 SVD가 $A = U\Sigma V^\top$일 때,

**(a)** [4점] $A^\top A$와 $AA^\top$의 고유값 분해가 SVD와 어떻게 관계되는지 유도하시오.

**(b)** [4점] Eckart-Young 정리를 서술하고, 이것이 이미지 압축에서 어떻게 활용되는지 설명하시오.

**(c)** [4점] PCA(주성분 분석)에서 데이터 공분산 행렬의 고유값 분해가 SVD와 어떻게 연결되는지 서술하시오.

---

## 문제 6. [12점] 최적화의 수학

**(a)** [4점] 볼록 함수의 정의를 쓰고, $f(x) = x^\top Ax$ ($A$ 양정치)가 볼록임을 증명하시오.

**(b)** [4점] 정규방정식 $X^\top X\hat{w} = X^\top y$를 유도하고, $X^\top X$가 비가역일 때의 해결책 두 가지를 제시하시오.

**(c)** [4점] Newton's Method $x_{n+1} = x_n - f(x_n)/f'(x_n)$을 사용하여 $\sqrt{7}$을 구하시오. $f(x) = x^2 - 7$로 놓고 $x_0 = 3$에서 2회 반복하시오.

---

## 문제 7. [13점] Attention 메커니즘의 수학

**(a)** [5점] Scaled Dot-Product Attention의 수식을 쓰고, $\sqrt{D_Q}$로 나누는 이유를 분산 분석으로 설명하시오. (힌트: $Q_i, K_j$의 각 원소가 평균 0, 분산 1이면 내적 $Q_i \cdot K_j$의 분산은?)

**(b)** [4점] Self-Attention이 순서를 인식하지 못하는 이유를 수식으로 설명하고, Positional Encoding이 이를 어떻게 해결하는지 서술하시오.

**(c)** [4점] Multi-Head Attention의 수식을 쓰고, 여러 Head를 사용하는 이점을 설명하시오.

---

## 문제 8. [12점] 가역 행렬의 동치 조건과 연결

정방행렬 $A \in \mathbb{R}^{n \times n}$에 대해 다음 조건들이 모두 동치임을 보이시오.

> (i) $A^{-1}$ 존재 ↔ (ii) $\text{rank}(A) = n$ ↔ (iii) $\det(A) \neq 0$ ↔ (iv) 모든 고유값 $\neq 0$ ↔ (v) $\mathscr{N}(A) = \{0\}$

**(a)** [8점] (i)→(ii)→(v)→(i) 순환 증명과, (ii)↔(iii), (iii)↔(iv)의 증명을 서술하시오.

**(b)** [4점] 이 동치 조건이 딥러닝에서 왜 중요한지, 구체적 예시(정규방정식, 야코비안 비특이성 등)를 들어 설명하시오.

---
---

# 모의고사 #2 — 핵심 답안 스케치

---

## 답 1. 통합 유도

| | 분포 | MLE (최소화) | MAP (Gaussian Prior) |
|---|---|---|---|
| 회귀 | $y \sim \mathcal{N}(h(x), \sigma^2)$ | $\sum(y_i-h(x_i))^2$ (MSE) | MSE + $\lambda\|\theta\|^2$ (Ridge) |
| 이진 분류 | $y \sim \text{Ber}(\sigma(w^\top x))$ | $-\sum[y\log\hat{y}+(1-y)\log(1-\hat{y})]$ (BCE) | BCE + $\lambda\|\theta\|^2$ |
| 다중 분류 | $y \sim \text{Cat}(h(x))$ | $-\sum\log[h(x_i)]_{y_i}$ (CE) | CE + $\lambda\|\theta\|^2$ |

(j) 모두 NLL = $-\log P(\text{data}|\theta)$의 형태. 확률 가정만 다르고 **프레임워크는 동일**: 데이터 분포 가정 → i.i.d. → 로그 → NLL. "Loss 함수의 선택은 확률 가정의 선택이다."

---

## 답 3. 베이즈 정리

### (a)
$\text{loglik} = 7\log\theta + 3\log(1-\theta)$, $\partial/\partial\theta = 7/\theta - 3/(1-\theta) = 0$ → $\theta_{ML} = 7/10 = 0.7$

### (b)
Posterior $\propto$ Likelihood × Prior $= \theta^7(1-\theta)^3 \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{7+\alpha-1}(1-\theta)^{3+\beta-1}$
$\partial/\partial\theta[\log] = \frac{7+\alpha-1}{\theta} - \frac{3+\beta-1}{1-\theta} = 0$
$$\theta_{MAP} = \frac{7+\alpha-1}{10+\alpha+\beta-2} = \frac{6+\alpha}{8+\alpha+\beta}$$

### (c)
$\alpha=\beta=2$: $\theta_{MAP} = 8/12 = 2/3 \approx 0.667$ (MLE 0.7보다 0.5쪽으로 당겨짐)

$n=100, k=70$: $\theta_{MAP} = 71/103 \approx 0.689$ (MLE 0.7에 매우 가까움)

정량적 설명: $\theta_{MAP} = \frac{k+\alpha-1}{n+\alpha+\beta-2}$에서 $n$이 커지면 $\alpha, \beta$의 영향이 $O(1/n)$으로 줄어든다. 즉 **데이터가 충분하면 MAP → MLE 수렴**.

---

## 답 4. KL과 최대 엔트로피

### (a) 비대칭 예시
$p = (0.9, 0.1)$, $q = (0.5, 0.5)$
$KL(p\|q) = 0.9\log(0.9/0.5) + 0.1\log(0.1/0.5) \approx 0.368$
$KL(q\|p) = 0.5\log(0.5/0.9) + 0.5\log(0.5/0.1) \approx 0.510$
$\neq$ → KL은 비대칭.

### (b) 최대 엔트로피 → 가우시안
$$\mathcal{L} = -\int p\log p + \lambda_0(\int p - 1) + \lambda_1(\int xp - \mu) + \lambda_2(\int(x-\mu)^2 p - \sigma^2)$$
$\delta\mathcal{L}/\delta p = 0$: $-\log p - 1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2 = 0$
$$\log p = \lambda_0 - 1 + \lambda_1 x + \lambda_2(x-\mu)^2$$
$$p(x) \propto \exp(\lambda_2(x-\mu)^2) = \text{Gaussian} \quad (\lambda_2 < 0)$$

### (c)
CLT: "많은 독립 원인의 합 → 가우시안" (생성 관점)
최대 엔트로피: "평균/분산만 알고 나머지를 모를 때, 가장 보수적(불확실성 최대) 선택 = 가우시안" (정보 관점)
→ 두 가지 독립적 논거가 모두 가우시안을 지지한다.

---

## 답 6. 최적화

### (c) Newton's Method로 √7
$f(x) = x^2 - 7$, $f'(x) = 2x$, $x_{n+1} = x_n - (x_n^2-7)/(2x_n) = (x_n + 7/x_n)/2$

$x_0 = 3$: $x_1 = (3 + 7/3)/2 = (3 + 2.333)/2 = 2.6\overline{6}$
$x_1 = 8/3$: $x_2 = (8/3 + 7/(8/3))/2 = (8/3 + 21/8)/2 = (64+63)/(24 \cdot 2) = 127/48 \approx 2.6458$

$\sqrt{7} = 2.6457...$이므로 2회 반복만에 소수점 4자리까지 정확.

---

## 답 7. Attention 수학

### (a) √D_Q 스케일링
$Q_i, K_j \in \mathbb{R}^{D_Q}$의 각 원소가 평균 0, 분산 1일 때:
$Q_i \cdot K_j = \sum_{d=1}^{D_Q} q_d k_d$

각 항 $q_d k_d$의 분산 = $\text{Var}(q_d)\text{Var}(k_d) = 1$ (독립)
합의 분산 = $D_Q \cdot 1 = D_Q$

→ 내적값의 분산이 $D_Q$에 비례 → $D_Q$가 크면 내적 크기 ↑ → softmax 입력이 극단적 → 그래디언트 소실
→ $\sqrt{D_Q}$로 나누면 분산이 1로 정규화 → softmax가 적절히 분포

### (b)
Self-Attention: $Y = \text{softmax}(QK^\top/\sqrt{D_Q})V$

입력 $X$의 행을 뒤섞어도 (치환 행렬 $P$ 적용: $PX$) → $Q, K, V$ 모두 같은 $P$ 적용 → 결과도 $PY$ → **순서가 아닌 집합(set)으로 취급**.

Positional Encoding: $X' = X + E$ ($E$는 위치별 고유 벡터) → 같은 토큰이라도 위치가 다르면 다른 입력 → 순서 정보 주입.

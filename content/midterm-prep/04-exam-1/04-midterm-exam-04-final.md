---
title: "Deep Learning 중간고사 모의시험 — 4회 (시험 범위 최종 반영)"
slug: midterm-exam-04-final
order: 4
---

# Deep Learning 중간고사 모의시험 — 4회 (시험 범위 최종 반영)

> **시험 범위**: Learning → MAP → ML → Loss(KL/CE) → Prior(Regularization) → Inductive Bias → Supervised Learning (Classification/Regression) → Gaussian → MSE
> **평가 기준**: 답이 아니라 **논리적 도달 과정**. 매 단계마다 가정과 이유를 글로 설명. 수식만 나열 시 0점.
> **문제는 영어** | **풀이는 한국어** | **총 200점** | **120분**

---

# Problem 1 (15점) — The Big Picture: MAP → NLL → MSE

**[EN]** This is the most important question. Derive the complete logical chain from Bayesian inference to MSE loss.

(a) Write Bayes' theorem: $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$. Define each of the four terms using the notation $H$ (hypothesis) and $E$ (experience). (3점)

(b) Show that $\hat{\theta}_{\text{MAP}} = \arg\min_\theta [-\log p(D|\theta) - \log p(\theta)]$. At each step, state what you drop and WHY it can be dropped. (4점)

(c) Assuming the data $\{(x_i, y_i)\}_{i=1}^n$ are i.i.d., expand $-\log p(D|\theta)$ into a sum. **Explicitly state where the independence assumption is used and where the identical distribution assumption is used.** (4점)

(d) Under Gaussian noise $y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$, show that NLL minimization equals MSE minimization. **At each line, write a sentence explaining what mathematical operation you performed and why.** (4점)

---

## 풀이

**(a)** 베이즈 정리는 새로운 데이터(경험 $E$)를 관찰한 후 모델(가설 $H$)에 대한 믿음을 갱신하는 공식이다.

$$p(H|E) = \frac{p(E|H) \cdot p(H)}{p(E)}$$

| 기호 | 이름 | 슬라이드 표기 | 의미 |
|------|------|-------------|------|
| $p(H\|E)$ | Posterior (사후확률) | $\log p(H\|E)$ → **MAP** | 데이터를 본 후 갱신된 가설에 대한 믿음 |
| $p(E\|H)$ | Likelihood (우도) | $\log p(E\|H)$ → **ML** | 가설이 맞다면 이 데이터가 나올 확률 |
| $p(H)$ | Prior (사전확률) | $\log p(H)$ → **Prior** | 데이터를 보기 전의 가설에 대한 믿음 |
| $p(E)$ | Evidence (증거) | — | $\theta$에 무관한 정규화 상수 |

슬라이드의 핵심 등식: $\log p(H|E) = \log p(E|H) + \log p(H) + \text{const}$, 즉 **MAP = ML + Prior**.

**(b)** MAP 추정은 사후확률을 최대화하는 $\theta$를 찾는 것이다.

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta|D)$$

**[Step 1: Evidence 제거]** $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$에서, $p(D)$는 $\theta$에 의존하지 않는 **상수**이다. $\arg\max$는 양의 상수배에 영향받지 않으므로 제거할 수 있다.

$$= \arg\max_\theta \; p(D|\theta) \cdot p(\theta)$$

**[Step 2: 로그 변환]** 곱을 합으로 바꾸기 위해 로그를 취한다. 로그는 **단조 증가 함수**이므로, 최대/최소의 위치를 바꾸지 않는다. 또한 확률값은 0~1 사이의 매우 작은 수일 수 있어 곱하면 underflow 위험이 있지만, 로그를 취하면 합산이 되어 수치적으로 안정하다.

$$= \arg\max_\theta \; [\log p(D|\theta) + \log p(\theta)]$$

**[Step 3: 부호 반전]** 최대화 문제를 최소화 문제로 바꾸기 위해 $-1$을 곱한다. 관례적으로 손실을 "최소화"하는 것으로 표현하기 때문이다.

$$= \arg\min_\theta \; [-\log p(D|\theta) - \log p(\theta)]$$

$$= \arg\min_\theta \; [\underbrace{\text{NLL}(\theta)}_{-\log p(D|\theta) = \text{ML 항}} + \underbrace{\text{Regularization}}_{-\log p(\theta) = \text{Prior 항}}]$$

**(c)** 데이터 $D = \{(x_1,y_1), \ldots, (x_n,y_n)\}$에 대해 NLL을 전개한다.

**[i.i.d. 가정의 두 부분을 각각 명시한다]**

**(i) Independent (독립)**: 각 데이터 포인트가 서로 영향을 주지 않고 독립적으로 생성되었다. 이 가정에 의해 **결합 확률이 개별 확률의 곱**으로 분해된다:

$$p(D|\theta) = p(y_1, \ldots, y_n | x_1, \ldots, x_n, \theta) = \prod_{i=1}^{n} p(y_i | x_i, \theta)$$

만약 데이터가 독립이 아니라면 (예: 시계열에서 $y_t$가 $y_{t-1}$에 영향받는 경우), 이 곱 분해는 성립하지 않는다.

**(ii) Identically Distributed (동일 분포)**: 모든 데이터가 같은 확률 모델 $p(y|x, \theta)$에서 생성되었다. 이 가정 덕분에 **같은 $\theta$를 모든 데이터에 공유**할 수 있다. 만약 각 데이터의 생성 분포가 다르다면 ($\theta_1, \theta_2, \ldots$ 각각 다른 파라미터), 하나의 $\theta$로 전체를 설명하는 모델이 불가능하다.

로그를 취하면 곱이 합으로 변환된다 ($\log(ab) = \log a + \log b$):

$$-\log p(D|\theta) = -\sum_{i=1}^{n} \log p(y_i | x_i, \theta) = \text{NLL}$$

이것이 슬라이드에서 $p_E$ (경험적 분포, i.i.d.)가 강조되는 이유다.

**(d)** 가우시안 노이즈를 가정한다. 왜 가우시안인가? 슬라이드에서 $p(y|x,h) \propto e^{-c(y-h(x))^2}$ 형태로 제시된다. 이 가정의 이론적 근거는 **중심극한정리(CLT)**이다: 예측값과 실제값의 차이(노이즈)는 수많은 미지 요인의 합이며, CLT에 의해 가우시안에 수렴한다.

$$p(y_i|x_i,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

**[Line 1: 로그를 취한다]** 지수 함수 안의 제곱항을 꺼내기 위함이다.

$$-\log p(y_i|x_i,\theta) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}$$

**[Line 2: n개 데이터에 대해 합산한다]** i.i.d.의 독립 가정에 의해 곱→합으로 분해된 것을 사용한다.

$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

**[Line 3: θ에 무관한 항을 제거한다]** 첫째 항 $\frac{n}{2}\log(2\pi\sigma^2)$는 $\theta$를 포함하지 않는 상수이다. $\arg\min$을 구할 때 상수는 최솟값의 위치를 바꾸지 않으므로 제거한다.

**[Line 4: 양의 상수 계수를 제거한다]** $\frac{1}{2\sigma^2}$은 양의 상수이다. 양의 상수를 곱해도 $\arg\min$은 변하지 않으므로 제거한다.

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 = \arg\min_\theta \; n \cdot \underbrace{\frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2}_{\text{MSE}}$$

$$\boxed{\text{가우시안 노이즈 하의 NLL 최소화} = \text{MSE 최소화}}$$

인과관계 체인: **베이즈 → MAP → 로그 → i.i.d.로 합 분해 → 가우시안 대입 → -log(exp) = 제곱 → MSE** $\blacksquare$

---

# Problem 2 (12점) — MAP Estimation: Coin Flip with Prior

**[EN]** You observe $n$ coin flips with $k$ heads. The likelihood is $p(D|\theta) = \theta^k(1-\theta)^{n-k}$ and the prior is $\text{Beta}(\alpha, \beta)$: $p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$.

(a) Write the log-posterior. Clearly label which terms come from the likelihood and which from the prior. (3점)

(b) Derive $\hat{\theta}_{\text{MAP}}$ by setting the derivative to zero. **At each differentiation step, state which rule you use (log derivative, chain rule, etc.) and WHY you set the derivative to zero.** (5점)

(c) With $n=10, k=7, \alpha=2, \beta=2$: compute $\hat{\theta}_{\text{MAP}}$ and $\hat{\theta}_{\text{MLE}}$. Explain why they differ and which direction the prior "pulls" the estimate. What happens as $n \to \infty$? (4점)

---

## 풀이

**(a)** 사후확률 $\propto$ 우도 $\times$ 사전확률 (베이즈 정리에서 $p(D)$ 상수 제거).

**우도에서 오는 항** ($\log p(D|\theta)$):
$$k\log\theta + (n-k)\log(1-\theta)$$

**사전확률에서 오는 항** ($\log p(\theta)$):
$$(\alpha-1)\log\theta + (\beta-1)\log(1-\theta) + \text{const}$$

**로그 사후확률 (합산)**:
$$\log p(\theta|D) = \underbrace{k\log\theta + (n-k)\log(1-\theta)}_{\text{Likelihood (ML)}} + \underbrace{(\alpha-1)\log\theta + (\beta-1)\log(1-\theta)}_{\text{Prior}} + C$$

$$= (k+\alpha-1)\log\theta + (n-k+\beta-1)\log(1-\theta) + C$$

**(b)** MAP 추정은 로그 사후확률을 최대화하는 $\theta$를 찾는 것이다. 미분가능한 함수의 극값에서는 도함수가 0이 된다 (Fermat의 정리). 로그 사후확률은 $\theta \in (0,1)$에서 위로 볼록(concave)하므로, 도함수 = 0인 점이 전역 최대이다.

$\theta$에 대해 미분한다:

$\frac{d}{d\theta}[(k+\alpha-1)\log\theta]$: $\log\theta$의 미분은 $1/\theta$ (로그 미분 규칙). 상수 $(k+\alpha-1)$은 그대로 남는다. 결과: $\frac{k+\alpha-1}{\theta}$.

$\frac{d}{d\theta}[(n-k+\beta-1)\log(1-\theta)]$: 체인룰을 적용한다. 외부 함수 $\log(u)$의 미분은 $1/u$, 내부 함수 $u = 1-\theta$의 미분은 $-1$. 곱하면 $\frac{-1}{1-\theta}$. 상수를 곱하면: $\frac{-(n-k+\beta-1)}{1-\theta}$.

합산하여 0으로 놓는다:

$$\frac{k+\alpha-1}{\theta} - \frac{n-k+\beta-1}{1-\theta} = 0$$

이것을 0으로 놓는 이유: MAP은 로그 사후확률의 최대점을 찾는 것이고, 최대점에서 기울기는 0이기 때문이다.

교차 곱으로 정리한다:

$$(k+\alpha-1)(1-\theta) = (n-k+\beta-1)\theta$$

좌변 전개: $(k+\alpha-1) - (k+\alpha-1)\theta = (n-k+\beta-1)\theta$

$(k+\alpha-1)$을 우변으로 이항: $(k+\alpha-1) = [(k+\alpha-1) + (n-k+\beta-1)]\theta = (n+\alpha+\beta-2)\theta$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+\alpha-1}{n+\alpha+\beta-2}}$$

**(c)** $n=10, k=7, \alpha=2, \beta=2$:

$$\hat{\theta}_{\text{MAP}} = \frac{7+2-1}{10+2+2-2} = \frac{8}{12} = \frac{2}{3} \approx 0.667$$

$$\hat{\theta}_{\text{MLE}} = \frac{k}{n} = \frac{7}{10} = 0.700$$

MAP(0.667) < MLE(0.700)인 이유: $\text{Beta}(2,2)$는 $\theta = 0.5$에서 최대인 대칭 사전분포이다. "동전은 공정할 가능성이 높다"는 사전 믿음이 추정을 0.5 방향으로 끌어당긴다. 이것이 **Prior의 역할 = Regularization**.

$n \to \infty$: $\hat{\theta}_{\text{MAP}} = \frac{k+1}{n+2} \to \frac{k}{n} = \hat{\theta}_{\text{MLE}}$. 분자·분모에서 $\alpha, \beta$의 영향이 데이터 양 $n$에 비해 무시할 만큼 작아진다. **데이터가 충분하면 MAP ≈ MLE, 즉 prior의 영향이 사라진다.** $\blacksquare$

---

# Problem 3 (10점) — Prior = Regularization = Generalization

**[EN]**
(a) If the prior is $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$, show that $-\log p(\theta)$ becomes $\frac{1}{2\sigma_p^2}\|\theta\|^2$ plus a constant. (3점)

(b) Combining with the Gaussian NLL from Problem 1(d), show that the MAP objective is $\text{MSE} + \lambda\|\theta\|^2$ and derive $\lambda = \frac{\sigma^2}{n\sigma_p^2}$. (4점)

(c) Explain the following relationships using this result:
- Large $\sigma_p^2$ (weak prior) → small $\lambda$ → less regularization
- Small $\sigma_p^2$ (strong prior) → large $\lambda$ → more regularization
- Large $n$ (more data) → small $\lambda$ → prior matters less

Why does regularization help generalization? Answer in terms of weight magnitudes and sensitivity to input changes. (3점)

---

## 풀이

**(a)** $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$의 확률밀도:

$$p(\theta) = \frac{1}{(2\pi\sigma_p^2)^{d/2}}\exp\left(-\frac{\|\theta\|^2}{2\sigma_p^2}\right)$$

음의 로그를 취한다:

$$-\log p(\theta) = \frac{d}{2}\log(2\pi\sigma_p^2) + \frac{\|\theta\|^2}{2\sigma_p^2}$$

첫째 항은 $\theta$에 무관한 상수이므로:

$$-\log p(\theta) = \frac{1}{2\sigma_p^2}\|\theta\|^2 + \text{const} \quad \blacksquare$$

**(b)** Problem 1(d)에서 NLL = $\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - f_\theta(x_i))^2 + \text{const}$.

MAP = NLL + Prior 항:

$$\arg\min_\theta\left[\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-f_\theta(x_i))^2 + \frac{1}{2\sigma_p^2}\|\theta\|^2\right]$$

$\frac{1}{2\sigma^2}$으로 나누면 (양의 상수이므로 $\arg\min$ 불변):

$$= \arg\min_\theta\left[\frac{1}{n}\sum_{i=1}^n(y_i-f_\theta(x_i))^2 + \frac{\sigma^2}{n\sigma_p^2}\|\theta\|^2\right]$$

$$= \arg\min_\theta\left[\text{MSE} + \lambda\|\theta\|^2\right], \quad \boxed{\lambda = \frac{\sigma^2}{n\sigma_p^2}}$$

**(c)** $\lambda = \frac{\sigma^2}{n\sigma_p^2}$에서:

- $\sigma_p^2$ 큼 (약한 prior = "θ가 어떤 값이든 괜찮다") → $\lambda$ 작음 → 정규화 약함 → 모델이 자유롭게 학습
- $\sigma_p^2$ 작음 (강한 prior = "θ는 0 근처여야 한다") → $\lambda$ 큼 → 정규화 강함 → 가중치가 작게 유지
- $n$ 큼 (데이터 많음) → $\lambda$ 작음 → 데이터의 힘이 prior를 압도

정규화가 일반화를 돕는 이유: 가중치($\theta$)가 크면, 입력의 작은 변화에 출력이 과민 반응한다 ($\Delta\text{output} = W \cdot \Delta\text{input}$, $W$가 크면 $\Delta\text{output}$이 큼). 이는 훈련 데이터의 노이즈까지 학습(과적합)하는 것을 의미한다. 정규화로 $\|\theta\|$를 작게 유지하면, 모델이 입력 변화에 부드럽게 반응하여 **안 본 데이터에서도 안정적** = 일반화 좋음. $\blacksquare$

---

# Problem 4 (10점) — Inductive Bias and Architecture

**[EN]**
(a) Define inductive bias. Explain how it relates to the prior $p(H)$ in the MAP framework. (3점)

(b) For each architecture, describe its inductive bias in one sentence:
- Linear model: $h(x) = w^\top x + b$
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- Transformer
(4점)

(c) A practitioner has 50 labeled medical images and wants to classify tumors. Another has 100 million internet images. For each, recommend an architecture and justify your choice in terms of inductive bias strength vs. data availability. (3점)

---

## 풀이

**(a)** Inductive bias(귀납적 편향)는 모델이 데이터를 보기 전에 가지고 있는 **"어떤 종류의 함수가 정답일 가능성이 높다"는 사전 가정**이다. 이것은 아키텍처 자체에 내장되어 있다.

MAP 프레임워크와의 관계: inductive bias는 **prior $p(H)$의 역할**과 정확히 대응한다. "CNN은 국소적이고 이동 불변인 패턴이 중요하다고 가정"하는 것은, 가설 공간에서 그런 성질을 가진 함수에 높은 사전확률을 부여하는 것과 같다. 강한 inductive bias = 강한 prior = 가설 공간을 좁게 제한.

**(b)**
- **선형 모델**: "입출력 관계가 직선(초평면)이다." — 가장 강한 inductive bias. 비선형 관계 표현 불가.
- **MLP**: "충분히 넓으면 어떤 연속 함수든 근사 가능하다(Universal Approximation)." — 구조적 가정이 거의 없으므로 inductive bias가 약함.
- **CNN**: "특징은 국소적(local)이고, 위치에 관계없이 동일한 필터로 감지 가능하다(translation equivariance)." — 이미지에 특화된 강한 inductive bias.
- **Transformer**: "모든 위치 간의 관계가 중요할 수 있다(global attention). 위치 정보만 별도로 제공." — 최소한의 inductive bias. 데이터에서 관계를 자유롭게 학습.

**(c)**

**50장 의료 이미지 → CNN 추천**: 데이터가 극도로 적으므로, 이미지 도메인에 대한 사전 지식(국소 패턴, 이동 불변)이 내장된 CNN이 적합하다. 강한 inductive bias가 적은 데이터로도 합리적 표현을 학습하게 해준다. Transformer는 이 수준의 데이터로는 의미 있는 패턴을 학습하기 어렵다 (자유도가 너무 높아 과적합).

**1억장 인터넷 이미지 → Transformer (ViT) 추천**: 데이터가 충분하므로 CNN의 강한 가정(국소성)이 오히려 제약이 된다. Transformer는 약한 inductive bias 덕분에 데이터에서 직접 최적의 표현을 자유롭게 학습할 수 있다. 실제로 ViT는 대규모 데이터에서 CNN을 능가한다. $\blacksquare$

---

# Problem 5 (10점) — KL Divergence, Cross-Entropy, and Loss

**[EN]** The slide states: $KL(p_E \| p_\theta) = CE(p_E, p_\theta) - Ent(p_E)$.

(a) Define each term: $KL$, $CE$, $Ent$. Write the formula for each. (3점)

(b) Prove that $D_{KL}(p \| q) \geq 0$ (Gibbs' inequality). State clearly which mathematical inequality you use. (4점)

(c) Explain why minimizing CE loss is equivalent to minimizing KL divergence, and therefore equivalent to making $p_\theta$ as close as possible to $p_E$. What is the role of $Ent(p_E)$ in this argument? (3점)

---

## 풀이

**(a)** 각 항의 정의:

**엔트로피** (분포 $p$의 불확실성):
$$Ent(p) = H(p) = -\sum_x p(x)\log p(x)$$

**교차 엔트로피** (분포 $p$에서 데이터가 올 때 $q$로 인코딩하는 비용):
$$CE(p, q) = H(p, q) = -\sum_x p(x)\log q(x)$$

**KL 발산** (두 분포 $p$, $q$의 차이):
$$KL(p \| q) = D_{KL}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$$

세 항의 관계: $KL(p \| q) = CE(p, q) - H(p)$

**(b)** Jensen 부등식 또는 $\ln t \leq t - 1$ (모든 $t > 0$, 등호 $t=1$)을 사용한다.

$$-D_{KL}(p\|q) = \sum_x p(x)\log\frac{q(x)}{p(x)}$$

$\log t \leq t - 1$을 $t = q(x)/p(x)$에 적용한다:

$$\sum_x p(x)\log\frac{q(x)}{p(x)} \leq \sum_x p(x)\left(\frac{q(x)}{p(x)} - 1\right) = \sum_x q(x) - \sum_x p(x) = 1 - 1 = 0$$

따라서 $-D_{KL}(p\|q) \leq 0$, 즉 $\boxed{D_{KL}(p\|q) \geq 0}$.

등호 조건: $q(x)/p(x) = 1$ 모든 $x$에서, 즉 $p = q$일 때만. $\blacksquare$

**(c)** $KL(p_E \| p_\theta) = CE(p_E, p_\theta) - Ent(p_E)$에서:

$Ent(p_E)$는 **데이터 분포 자체의 엔트로피**이다. 이것은 $\theta$에 의존하지 않는 **상수**이다 (데이터 분포는 우리가 바꿀 수 없으므로).

따라서:
$$\arg\min_\theta CE(p_E, p_\theta) = \arg\min_\theta [KL(p_E\|p_\theta) + \underbrace{Ent(p_E)}_{\text{상수}}] = \arg\min_\theta KL(p_E\|p_\theta)$$

**CE를 최소화하면 자동으로 KL도 최소화**된다. KL = 0이면 $p_\theta = p_E$ (모델이 데이터를 완벽히 설명). 이것이 딥러닝 학습의 목표: **모델 분포를 데이터 분포에 가능한 한 가깝게** 만드는 것. $\blacksquare$

---

# Problem 6 (10점) — Classification: Categorical Distribution → CE Loss

**[EN]** The slide states: $p(y|x,h) = h(x)_y$ (Categorical Distribution).

(a) Explain what $h(x)_y$ means. If $h(x) = (0.7, 0.2, 0.1)$ and $y = 2$ (0-indexed), what is $p(y|x,h)$? (2점)

(b) For a dataset of $n$ i.i.d. samples, derive the NLL and show it equals $-\frac{1}{n}\sum_{i=1}^n \log h(x_i)_{y_i}$. State where i.i.d. is used. (4점)

(c) Rewrite this as $-\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C \mathbf{1}[y_i = c]\log h(x_i)_c$ using indicator notation, and show this is equivalent to the cross-entropy loss $-\frac{1}{n}\sum_i e_{y_i}^\top \log h(x_i)$ (as in the slide). (4점)

---

## 풀이

**(a)** $h(x)$는 모델의 출력으로, $C$개 클래스에 대한 확률 벡터이다 (softmax 출력). $h(x)_y$는 이 벡터의 $y$번째 원소 = **정답 클래스에 모델이 부여한 확률**.

$h(x) = (0.7, 0.2, 0.1)$이고 $y = 2$ (0-indexed)이면: $p(y|x,h) = h(x)_2 = 0.1$.

**(b)** **독립 가정**에 의해 전체 데이터의 우도는 개별 우도의 곱이다:

$$p(D|\theta) = \prod_{i=1}^{n} p(y_i|x_i,\theta) = \prod_{i=1}^n h(x_i)_{y_i}$$

**동일 분포 가정**에 의해 모든 데이터에 같은 모델 $h$ (= 같은 $\theta$)를 적용한다.

음의 로그를 취하면 (곱 → 합):

$$\text{NLL} = -\log\prod_{i=1}^n h(x_i)_{y_i} = -\sum_{i=1}^n \log h(x_i)_{y_i}$$

$\frac{1}{n}$을 붙여 평균으로 만들면:

$$L = \frac{1}{n}\text{NLL} = -\frac{1}{n}\sum_{i=1}^n \log h(x_i)_{y_i} \quad \blacksquare$$

**(c)** 지시 함수(indicator function) $\mathbf{1}[y_i = c]$를 사용하면:

$$-\log h(x_i)_{y_i} = -\sum_{c=1}^{C}\mathbf{1}[y_i = c]\log h(x_i)_c$$

왜? $\mathbf{1}[y_i = c]$는 $c = y_i$일 때만 1이고 나머지는 0이므로, 합에서 정답 클래스 항만 살아남아 $\log h(x_i)_{y_i}$가 된다.

원-핫 벡터 $e_{y_i}$는 $y_i$ 위치만 1인 벡터이므로, $\mathbf{1}[y_i = c] = (e_{y_i})_c$. 따라서:

$$-\sum_c (e_{y_i})_c \log h(x_i)_c = -e_{y_i}^\top \log h(x_i)$$

이것은 슬라이드의 **CE Loss 수식**: $-\frac{1}{|E|}\sum_i e_{y_i}^\top \log h(x_i)$과 정확히 일치한다. $\blacksquare$

---

# Problem 7 (10점) — Regression: Gaussian → MSE

**[EN]** The slide states: $p(y|x,h) \propto e^{-c(y-h(x))^2}$, and the loss is $\frac{1}{|E|}\sum_i (y_i - h(x_i))^2$.

(a) Show that $p(y|x,h) \propto e^{-c(y-h(x))^2}$ is a Gaussian distribution. Identify the mean and variance in terms of $c$. (3점)

(b) Starting from this Gaussian likelihood, derive the MSE loss through NLL. At EACH step, write one sentence explaining what you did. (4점)

(c) Why is $h(x)$ the MEAN of the Gaussian and not, say, the mode or median? Are they the same for Gaussian? Explain. (3점)

---

## 풀이

**(a)** $p(y|x,h) \propto e^{-c(y-h(x))^2}$를 가우시안 PDF와 비교한다.

가우시안 $\mathcal{N}(\mu, \sigma^2)$의 PDF: $p(y) \propto \exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)$

지수부를 비교하면: $c = \frac{1}{2\sigma^2}$

따라서 **평균 = $h(x)$**, **분산 = $\sigma^2 = \frac{1}{2c}$**.

슬라이드의 $\propto$ 표기는 정규화 상수 $\frac{1}{\sqrt{2\pi\sigma^2}}$를 생략한 것이다.

**(b)**

**[Step 1]** 가우시안 우도를 쓴다: $p(y_i|x_i,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-h(x_i))^2}{2\sigma^2}\right)$

여기서 "슬라이드의 비례식에 정규화 상수를 복원했다."

**[Step 2]** i.i.d. 가정(독립)으로 전체 우도를 곱으로 분해하고, 로그를 취해 합으로 변환한다.

$$-\log p(D|\theta) = \sum_{i=1}^n\left[\frac{1}{2}\log(2\pi\sigma^2) + \frac{(y_i-h(x_i))^2}{2\sigma^2}\right]$$

여기서 "곱 → 합 변환을 위해 로그를 취했고, 독립 가정이 곱 분해를 가능하게 했다."

**[Step 3]** $\theta$에 무관한 상수 $\frac{n}{2}\log(2\pi\sigma^2)$를 제거한다.

여기서 "$\arg\min$을 구할 때 $\theta$를 포함하지 않는 항은 최소점 위치에 영향을 주지 않으므로 제거한다."

**[Step 4]** 양의 상수 $\frac{1}{2\sigma^2}$를 제거한다.

여기서 "양의 상수 곱은 $\arg\min$에 영향을 주지 않으므로 제거한다."

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n(y_i - h(x_i))^2 = \arg\min_\theta \text{MSE} \quad \blacksquare$$

**(c)** 가우시안 분포는 완전히 대칭인 종 모양이므로, **평균 = 최빈값(mode) = 중앙값(median)**이다. 세 값이 전부 같다.

$h(x)$가 가우시안의 **평균**인 이유: MLE는 우도를 최대화하는 파라미터를 찾는데, 가우시안에서 우도가 최대인 지점은 $(y - \mu)^2$이 최소인 지점, 즉 $\mu = h(x)$가 $y$에 가까울 때이다. $\mu$가 데이터의 평균에 해당하므로 $h(x)$는 자연스럽게 조건부 평균이 된다. $\blacksquare$

---

# Problem 8 (10점) — MLE: Gaussian Mean and Variance

**[EN]** Given i.i.d. data $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$:

(a) Write the log-likelihood. (2점)
(b) Derive $\hat{\mu}_{\text{MLE}} = \bar{x}$ by differentiation. State which differentiation rule you use. (3점)
(c) Derive $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(x_i - \bar{x})^2$. (3점)
(d) Show that $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. **State where independence and identical distribution are used separately.** (2점)

---

## 풀이

**(a)** 각 $x_i$의 PDF: $p(x_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$

**i.i.d.의 독립 가정**에 의해 결합 우도 = 개별 우도의 곱. 로그를 취하면:

$$\ell(\mu,\sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2$$

**(b)** $\mu$에 대해 미분한다. $\sigma^2$은 상수 취급 (편미분).

$$\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(x_i - \mu)$$

미분 과정: $-\frac{1}{2\sigma^2} \cdot \frac{d}{d\mu}(x_i-\mu)^2 = -\frac{1}{2\sigma^2} \cdot 2(x_i-\mu) \cdot (-1) = \frac{x_i-\mu}{\sigma^2}$

체인룰을 사용했다: 외부 함수 $u^2$의 미분은 $2u$, 내부 함수 $u = x_i - \mu$의 $\mu$에 대한 미분은 $-1$.

0으로 놓는다 (MLE는 로그 우도를 최대화하는 $\mu$를 찾으며, 극대에서 기울기 = 0):

$$\sum_{i=1}^n(x_i - \mu) = 0 \implies \sum x_i = n\mu \implies \boxed{\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}}$$

**(c)** $\sigma^2$에 대해 미분한다. $s = \sigma^2$로 놓으면:

$$\frac{\partial\ell}{\partial s} = -\frac{n}{2s} + \frac{1}{2s^2}\sum(x_i-\mu)^2$$

0으로 놓으면: $ns = \sum(x_i-\mu)^2$

$$\boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2}$$

**(d)** 핵심 항등식: $\sum(x_i - \bar{x})^2 = \sum(x_i - \mu)^2 - n(\bar{x}-\mu)^2$

$\mathbb{E}[\sum(x_i-\mu)^2] = n\sigma^2$. 여기서 **동일 분포 가정** 사용: 모든 $x_i$가 같은 분산 $\sigma^2$을 가짐.

$\mathbb{E}[n(\bar{x}-\mu)^2] = n\text{Var}(\bar{x}) = n \cdot \frac{\sigma^2}{n} = \sigma^2$. 여기서 **독립 가정** 사용: $\text{Var}(\bar{x}) = \frac{1}{n^2}\sum\text{Var}(x_i) = \frac{\sigma^2}{n}$. 독립이어야 분산의 합이 합의 분산이 된다.

$$\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{1}{n}(n\sigma^2 - \sigma^2) = \frac{n-1}{n}\sigma^2 \neq \sigma^2 \quad \text{(편향)} \quad \blacksquare$$

---

# Problem 9 (10점) — CLT and the Gaussian Assumption

**[EN]**
(a) State the Central Limit Theorem precisely. What are the three conditions required? (3점)

(b) In regression, we model $y = f(x) + \epsilon$. The noise $\epsilon$ comes from many unknown factors. Using CLT, justify why $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is a reasonable assumption. (3점)

(c) Trace the full chain: CLT → Gaussian noise → Gaussian likelihood → $-\log$ → squared error → MSE. Explain what happens at the "$-\log$" step that turns an exponential into a square. (4점)

---

## 풀이

**(a)** CLT: $X_1, X_2, \ldots, X_n$이 세 가지 조건을 만족하면:

1. **독립(Independent)**: 각 $X_i$가 서로 독립
2. **동일 분포(Identically Distributed)**: 모든 $X_i$가 같은 분포
3. **유한 분산**: $\text{Var}(X_i) = \sigma^2 < \infty$

표본 평균의 분포가 정규분포에 수렴한다:

$$\sqrt{n}\frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0,1) \quad (n \to \infty)$$

**(b)** 노이즈 $\epsilon$은 측정 오차, 누락 변수, 환경 변동 등 **수많은 미지 요인**의 합산 효과이다:

$$\epsilon = \delta_1 + \delta_2 + \cdots + \delta_m$$

각 $\delta_j$가 (근사적으로) 독립이고, 개별 영향이 크지 않으며, 유한한 분산을 가지면, CLT에 의해 $\epsilon$은 가우시안에 수렴한다. 따라서 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 가정은 이론적 근거가 있다.

단, 극단적 이상치가 빈번하거나 요인들이 강하게 상관되어 있으면 CLT 적용이 부적절할 수 있다.

**(c)** 전체 체인을 각 단계의 연결과 함께:

**[CLT → 가우시안 노이즈]**: 많은 독립 요인의 합 → 정규분포 수렴 (CLT 정리)

**[가우시안 노이즈 → 가우시안 우도]**: $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0,\sigma^2)$ → $p(y|x) = \mathcal{N}(f(x), \sigma^2)$ (조건부 분포 대입)

**[가우시안 우도 → -log]**: $p(y|x) \propto \exp(-\frac{(y-f(x))^2}{2\sigma^2})$에 $-\log$를 취한다.

**핵심 단계**: $-\log(\exp(-z)) = z$. 로그와 지수는 **역함수**이므로 상쇄된다. 가우시안의 지수 안에 있던 **제곱항**이 밖으로 나온다:

$$-\log\exp\left(-\frac{(y-f(x))^2}{2\sigma^2}\right) = \frac{(y-f(x))^2}{2\sigma^2}$$

**[-log → MSE]**: 상수 $\frac{1}{2\sigma^2}$를 제거하고 $n$개 합산하면 MSE.

$$\boxed{\text{CLT} \to \text{Gaussian} \to \exp(-z^2) \xrightarrow{-\log} z^2 \to \text{MSE}}$$

"$-\log$ 단계에서 지수와 로그가 상쇄되어 제곱항만 남는다"가 가우시안→MSE 연결의 수학적 핵심이다. $\blacksquare$

---

# Problem 10 (10점) — Supervised Learning: Two Frameworks United

**[EN]** The slide shows two cases of supervised learning side by side:

| | Classification | Regression |
|--|---------------|-----------|
| Distribution | Categorical: $p(y\|x,h) = h(x)_y$ | Gaussian: $p(y\|x,h) \propto e^{-c(y-h(x))^2}$ |
| Loss | CE: $-\frac{1}{\|E\|}\sum e_y^\top \log h(x)$ | MSE: $\frac{1}{\|E\|}\sum(y-h(x))^2$ |

(a) Both losses come from the same principle: NLL minimization. Explain this unifying principle in 3-4 sentences. (3점)

(b) Show mathematically that Categorical + NLL → CE Loss. (3점)

(c) Show mathematically that Gaussian + NLL → MSE Loss. (You may reference Problem 1 but still show the key steps.) (2점)

(d) A student asks: "Why don't we use MSE for classification?" Explain what goes wrong when you use $L = \sum(y_i - \hat{y}_i)^2$ with one-hot $y$ and softmax $\hat{y}$. (2점)

---

## 풀이

**(a)** 통합 원리: 지도 학습에서 **어떤 분포를 가정하느냐**에 따라 NLL(음의 로그 우도)의 구체적 형태가 달라지고, 그것이 곧 손실 함수가 된다. 분류에서는 출력이 이산적(카테고리)이므로 카테고리컬 분포를 가정하면 NLL = CE Loss가 나온다. 회귀에서는 출력이 연속적이므로 가우시안 분포를 가정하면 NLL = MSE Loss가 나온다. **손실 함수 = 분포 가정 + NLL 최소화**라는 하나의 프레임워크가 두 경우를 통합한다.

**(b)** Categorical + NLL → CE:

$$p(y_i|x_i,\theta) = h(x_i)_{y_i} = \prod_{c=1}^C h(x_i)_c^{\mathbf{1}[y_i=c]}$$

i.i.d. 독립 가정으로 곱 분해 후, $-\log$를 취한다:

$$\text{NLL} = -\sum_{i=1}^n \log h(x_i)_{y_i} = -\sum_{i=1}^n \sum_c \mathbf{1}[y_i=c]\log h(x_i)_c = -\sum_i e_{y_i}^\top \log h(x_i)$$

$\frac{1}{n}$으로 나누면 슬라이드의 CE Loss. $\blacksquare$

**(c)** Gaussian + NLL → MSE:

$$p(y_i|x_i,\theta) \propto \exp\left(-\frac{(y_i-h(x_i))^2}{2\sigma^2}\right)$$

$-\log$를 취하면 $\frac{(y_i-h(x_i))^2}{2\sigma^2} + \text{const}$. i.i.d.로 합산하고, $\theta$-무관 상수와 양의 상수 제거하면 $\sum(y_i - h(x_i))^2$ = MSE. $\blacksquare$

**(d)** 분류에 MSE를 쓰면 두 가지 문제가 발생한다:

**문제 1 — 기울기 포화**: 소프트맥스 출력이 0이나 1 근처일 때, MSE의 그래디언트에는 $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 항이 남는다. 이 값은 포화 영역에서 거의 0이 되어 **학습이 정체**된다. CE Loss에서는 이 항이 소거되어 깔끔한 그래디언트 $(\hat{y}-y)x$가 나온다 (canonical link function 효과).

**문제 2 — 확률적 비정합**: MSE는 가우시안 노이즈를 가정한 NLL이다. 분류의 출력은 이산적(카테고리컬)이지, 가우시안이 아니다. 잘못된 분포를 가정하면 최적화 landscape가 부적절해져 수렴이 느리고 성능이 나빠진다. $\blacksquare$

---

# Problem 11 (8점) — Learning의 정의와 종류

**[EN]**
(a) Using Tom Mitchell's definition, define "learning" in terms of Task ($T$), Performance measure ($P$), and Experience ($E$). Connect this to the slide's notation $H$ and $E$. (3점)

(b) List and briefly describe three types of learning: supervised, unsupervised, reinforcement. For supervised learning, explain why we need both $x$ (input) and $y$ (label). (3점)

(c) Why is the i.i.d. assumption important for learning? What breaks if the data is NOT i.i.d.? (2점)

---

## 풀이

**(a)** Tom Mitchell (1997): "프로그램이 경험 $E$를 통해 과제 $T$에 대한 성능 $P$가 개선되면, 이 프로그램은 학습했다고 한다."

슬라이드 표기와의 대응:
- $H$ (hypothesis, model) = 학습의 결과물. 파라미터 $\theta$로 정의되는 함수 $h$.
- $E$ (experience, data) = 학습에 사용하는 데이터 $S$ 또는 $D$.
- $T$ = 분류, 회귀, 생성 등의 과제.
- $P$ = 손실 함수 (CE, MSE 등). 이것이 작을수록 성능이 좋다.

학습 = 경험 $E$를 사용하여, 성능 $P$ (손실)을 최소화하는 가설 $H$를 찾는 과정.

**(b)**
- **지도학습(Supervised)**: 입력 $x$와 정답 레이블 $y$의 쌍 $(x,y)$로 학습. $x$만으로는 "무엇이 정답인지" 알 수 없으므로 $y$가 필요하다. $y$가 이산이면 분류, 연속이면 회귀.
- **비지도학습(Unsupervised)**: $x$만 있고 $y$가 없다. 데이터의 숨겨진 구조(군집, 분포)를 발견. 예: 클러스터링, 생성 모델.
- **강화학습(Reinforcement)**: 에이전트가 환경과 상호작용하며 보상 신호를 최대화하는 정책 학습. 예: 게임, 로봇.

**(c)** i.i.d. 가정이 중요한 이유:

**독립(i)**: 결합 우도를 곱으로 분해 가능 → NLL이 합으로 쓰임 → SGD로 효율적 학습 가능.
**동일 분포(d)**: 하나의 $\theta$로 모든 데이터를 설명 가능 → 모델이 의미 있음.

i.i.d.가 깨지면: (1) 곱 분해 불가 → 로그 우도가 깔끔한 합이 아님, (2) 분포가 시간에 따라 변하면(concept drift) 과거 데이터로 학습한 모델이 미래에 무용지물, (3) 일반화 보장의 이론적 근거(호에프딩 등)가 성립하지 않음. $\blacksquare$

---

# Problem 12 (10점) — Empirical Distribution and Optimization

**[EN]** The slide mentions the empirical distribution $p_E$ and connects it to optimization (loss).

(a) Define the empirical distribution $\hat{p}(x, y)$ for a dataset $\{(x_i, y_i)\}_{i=1}^n$. How does it differ from the true data distribution $p^*(x,y)$? (3점)

(b) Show that minimizing CE loss w.r.t. $\theta$ is equivalent to minimizing $D_{KL}(\hat{p} \| p_\theta)$. What does this tell us about the goal of training? (4점)

(c) The loss $\frac{1}{|E|}\sum_i \ell(h(x_i), y_i)$ is called Empirical Risk. How does it relate to the true risk $\mathbb{E}_{(x,y) \sim p^*}[\ell(h(x), y)]$? What role does the number of samples $n$ play? (3점)

---

## 풀이

**(a)** 경험적 분포(empirical distribution):

$$\hat{p}(x, y) = \frac{1}{n}\sum_{i=1}^{n}\delta(x - x_i, y - y_i)$$

이것은 각 데이터 포인트에 확률 $1/n$을 균등하게 부여한 이산 분포이다.

진짜 분포 $p^*(x,y)$와의 차이: $\hat{p}$은 유한한 $n$개 데이터에서 구성된 **근사**이다. $n \to \infty$이면 대수의 법칙(LLN)에 의해 $\hat{p} \to p^*$.

**(b)** CE 손실:

$$L(\theta) = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(y_i|x_i) = -\mathbb{E}_{(x,y)\sim\hat{p}}[\log p_\theta(y|x)]$$

이것은 경험적 분포 $\hat{p}$와 모델 분포 $p_\theta$ 사이의 **교차 엔트로피** $H(\hat{p}, p_\theta)$이다.

$H(\hat{p}, p_\theta) = H(\hat{p}) + D_{KL}(\hat{p} \| p_\theta)$

$H(\hat{p})$는 $\theta$에 무관한 상수이므로:

$$\arg\min_\theta L(\theta) = \arg\min_\theta D_{KL}(\hat{p} \| p_\theta)$$

학습의 목표: **모델 분포 $p_\theta$를 경험적 분포 $\hat{p}$ (= 데이터)에 가능한 한 가깝게 만드는 것**. KL = 0이면 모델이 데이터를 완벽히 설명.

**(c)** 경험적 위험(Empirical Risk): $\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)$

진짜 위험(True Risk): $R(h) = \mathbb{E}_{(x,y) \sim p^*}[\ell(h(x), y)]$

LLN에 의해 $n \to \infty$이면 $\hat{R}(h) \to R(h)$.

유한 $n$에서는 **일반화 오차**(generalization gap) = $R(h) - \hat{R}(h) > 0$. 호에프딩 부등식에 의해 이 갭은 $O(1/\sqrt{n})$으로 줄어든다. $n$이 클수록 경험적 위험이 진짜 위험에 가까워지므로, 훈련 성능이 실전 성능을 더 잘 반영한다. $\blacksquare$

---

# Problem 13 (10점) — Ridge Regression = MAP with Gaussian Prior

**[EN]** Consider linear regression with L2 regularization (Ridge Regression):

$$\hat{\theta} = \arg\min_\theta \frac{1}{2n}\|X\theta - y\|^2 + \frac{\lambda}{2}\|\theta\|^2$$

(a) Derive the closed-form solution $\hat{\theta} = (X^\top X + n\lambda I)^{-1}X^\top y$. Show all steps using matrix calculus. (5점)

(b) Explain why $(X^\top X + n\lambda I)$ is always invertible when $\lambda > 0$, even if $X$ does not have full column rank. (3점)

(c) Interpret this result through the MAP lens: what prior does $\lambda\|\theta\|^2$ correspond to? What is $\sigma_p^2$ in terms of $\lambda$ and $\sigma^2$? (2점)

---

## 풀이

**(a)** $L(\theta) = \frac{1}{2n}(X\theta-y)^\top(X\theta-y) + \frac{\lambda}{2}\theta^\top\theta$

전개:
$$= \frac{1}{2n}(\theta^\top X^\top X\theta - 2y^\top X\theta + y^\top y) + \frac{\lambda}{2}\theta^\top\theta$$

$\theta$에 대해 미분한다:

$\frac{\partial}{\partial\theta}(\theta^\top X^\top X\theta)$: $X^\top X$는 대칭이므로 이차형식의 미분 공식 $\frac{\partial}{\partial\theta}(\theta^\top S\theta) = 2S\theta$를 적용한다. 결과: $2X^\top X\theta$.

$\frac{\partial}{\partial\theta}(2y^\top X\theta)$: 선형 함수의 미분. 결과: $2X^\top y$.

$\frac{\partial}{\partial\theta}(\theta^\top\theta) = 2\theta$ (단위행렬이 대칭이므로 같은 공식 적용).

합산:

$$\nabla_\theta L = \frac{1}{2n}(2X^\top X\theta - 2X^\top y) + \frac{\lambda}{2} \cdot 2\theta = \frac{1}{n}X^\top X\theta - \frac{1}{n}X^\top y + \lambda\theta$$

0으로 놓는 이유: 볼록 함수의 최소는 기울기 = 0에서 달성된다.

$$\frac{1}{n}X^\top X\theta + \lambda\theta = \frac{1}{n}X^\top y$$

$$\left(\frac{1}{n}X^\top X + \lambda I\right)\theta = \frac{1}{n}X^\top y$$

양변에 $n$을 곱하면:

$$(X^\top X + n\lambda I)\theta = X^\top y$$

$$\boxed{\hat{\theta} = (X^\top X + n\lambda I)^{-1}X^\top y}$$

**(b)** $X^\top X$는 PSD (고유값 $\mu_i \geq 0$). $X^\top X + n\lambda I$의 고유값은 $\mu_i + n\lambda$.

$\lambda > 0$이면 $\mu_i + n\lambda > 0$ (모든 $i$). 모든 고유값이 양수 → PD → **가역**. $X$의 rank에 관계없이 항상 역행렬이 존재한다. 이것이 정규화의 수학적 효과.

**(c)** $\lambda\|\theta\|^2$는 가우시안 사전분포 $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$에서 $-\log p(\theta) = \frac{1}{2\sigma_p^2}\|\theta\|^2 + \text{const}$에 대응한다.

Problem 1의 결과에서 $\lambda = \frac{\sigma^2}{n\sigma_p^2}$이므로:

$$\sigma_p^2 = \frac{\sigma^2}{n\lambda}$$

$\lambda$가 크면 → $\sigma_p^2$가 작음 → "θ는 0 근처여야 한다"는 강한 prior. $\blacksquare$

---

# Problem 14 (8점) — Entropy와 확신도

**[EN]**
(a) Compute the entropy $H(p)$ for: (i) one-hot $p = (1, 0, 0)$, (ii) uniform $p = (1/3, 1/3, 1/3)$, (iii) $p = (0.8, 0.1, 0.1)$. Use $\log_2$. (3점)

(b) Explain why entropy is highest for the uniform distribution and zero for the one-hot. Connect this to model confidence in classification. (2점)

(c) In the MAP → CE Loss → MSE chain, where does entropy appear and what role does it play? (3점)

---

## 풀이

**(a)** $H(p) = -\sum p_i \log_2 p_i$ ($0\log 0 := 0$)

**(i)** $H(1,0,0) = -(1\log_2 1 + 0 + 0) = 0$ bits

**(ii)** $H(1/3,1/3,1/3) = -3 \times \frac{1}{3}\log_2\frac{1}{3} = \log_2 3 \approx 1.585$ bits

**(iii)** $H(0.8,0.1,0.1) = -(0.8\log_2 0.8 + 0.1\log_2 0.1 + 0.1\log_2 0.1)$
$= -(0.8 \times (-0.322) + 2 \times 0.1 \times (-3.322))$
$= -(-0.258 - 0.664) = 0.922$ bits

**(b)** 원-핫: 결과가 확정적 → 불확실성 = 0 → 엔트로피 = 0.
균등분포: 모든 결과가 동등하게 가능 → 최대 불확실성 → 최대 엔트로피.

모델 확신도와의 연결: 모델 출력의 엔트로피가 낮으면 → 하나의 클래스에 확률이 집중 → **모델이 확신**. 엔트로피가 높으면 → 확률이 분산 → **모델이 불확실**.

**(c)** $KL(p_E \| p_\theta) = CE(p_E, p_\theta) - Ent(p_E)$.

$Ent(p_E)$는 데이터 자체의 엔트로피이다. CE Loss를 최소화할 때 이 항은 $\theta$에 무관한 **상수**이므로, 학습 과정에서 직접적 역할은 하지 않지만, **CE와 KL의 관계를 성립시키는 다리** 역할을 한다. CE 최소화가 곧 KL 최소화인 이유는 $Ent(p_E)$가 상수이기 때문이다. $\blacksquare$

---

# Problem 15 (10점) — Sigmoid 미분과 Classification 그래디언트

**[EN]**
(a) Prove $\sigma'(z) = \sigma(z)(1-\sigma(z))$ where $\sigma(z) = 1/(1+e^{-z})$. Show every step. (4점)

(b) For binary CE loss $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ with $\hat{y} = \sigma(w^\top x)$, compute $\frac{\partial L}{\partial w}$ using the chain rule. Show the remarkable cancellation. (4점)

(c) At what values of $\hat{y}$ is $\sigma'$ maximized? What is the maximum value? Why does this cause problems in deep networks? (2점)

---

## 풀이

**(a)** $\sigma(z) = (1+e^{-z})^{-1}$

복합 함수의 미분(체인룰)을 적용한다. $u = 1 + e^{-z}$로 놓으면 $\sigma = u^{-1}$.

외부 함수의 미분: $\frac{d}{du}u^{-1} = -u^{-2}$
내부 함수의 미분: $\frac{du}{dz} = -e^{-z}$ ($e^{-z}$를 $z$로 미분하면 체인룰에 의해 $e^{-z} \cdot (-1)$)

곱하면: $\sigma'(z) = (-u^{-2})(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$

$\sigma(z)(1-\sigma(z)) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2}$

두 결과가 동일하다. $\boxed{\sigma'(z) = \sigma(z)(1-\sigma(z))}$ $\blacksquare$

**(b)** $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial\hat{y}} \cdot \frac{\partial\hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$

$\frac{\partial L}{\partial\hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}$

$\frac{\partial\hat{y}}{\partial z} = \hat{y}(1-\hat{y})$ (위에서 증명)

$\frac{\partial z}{\partial w} = x$

곱하면: $\frac{\hat{y}-y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x$

분모 $\hat{y}(1-\hat{y})$가 분자와 **정확히 소거**된다!

$$\boxed{\frac{\partial L}{\partial w} = (\hat{y} - y)x}$$

**(c)** $\sigma'(z) = p(1-p)$ ($p = \sigma(z)$)는 $p = 1/2$ ($z = 0$)에서 최대, 최대값 = $\frac{1}{4}$.

$L$개 시그모이드 층의 그래디언트는 $\prod_l \sigma'(z_l) \leq (1/4)^L$으로 지수적 감소. $L = 10$이면 $\approx 10^{-7}$. 이것이 **vanishing gradient** 문제. $\blacksquare$

---

# Problem 16 (8점) — Why Not MSE for Classification?

**[EN]** Explain mathematically and intuitively why cross-entropy is preferred over MSE for classification.

(a) Compute the gradient of MSE loss $L = \frac{1}{2}(y - \sigma(z))^2$ w.r.t. $z$. Show that $\sigma'(z) = \sigma(z)(1-\sigma(z))$ remains in the gradient. (3점)

(b) Compare this with the CE gradient from Problem 15(b). What term is present in MSE's gradient but absent in CE's? (2점)

(c) When $\sigma(z) \approx 0$ or $\sigma(z) \approx 1$, what happens to learning speed under MSE vs. CE? Which recovers faster from a confident wrong prediction? (3점)

---

## 풀이

**(a)** $L = \frac{1}{2}(y - \sigma(z))^2$

$$\frac{\partial L}{\partial z} = (y - \sigma(z)) \cdot (-1) \cdot \sigma'(z) = -(\text{y} - \hat{y}) \cdot \sigma(z)(1-\sigma(z))$$

$= (\hat{y} - y) \cdot \sigma(z)(1-\sigma(z))$

$\sigma(z)(1-\sigma(z))$ 항이 **소거되지 않고 남아있다**.

**(b)** CE: $\frac{\partial L}{\partial z} = \hat{y} - y$ → 깔끔. $\sigma'$ 없음.

MSE: $\frac{\partial L}{\partial z} = (\hat{y}-y) \cdot \sigma(z)(1-\sigma(z))$ → **$\sigma'(z)$가 추가 곱셈 인자로 남음**.

**(c)** $\sigma(z) \approx 0$ 또는 $\approx 1$이면 $\sigma(z)(1-\sigma(z)) \approx 0$.

**MSE**: 그래디언트 $\approx (\hat{y}-y) \times 0 \approx 0$. 모델이 자신있게 틀린 예측($\hat{y} \approx 0$인데 $y = 1$)을 해도, 기울기가 거의 0이어서 **교정이 극도로 느리다**. 잘못된 확신에서 빠져나오지 못한다.

**CE**: 그래디언트 $= \hat{y} - y$. $\hat{y} \approx 0$이고 $y = 1$이면 그래디언트 $\approx -1$. **큰 교정 신호**가 즉시 전달되어 빠르게 학습한다.

결론: **자신있게 틀린 예측에서 CE가 훨씬 빠르게 복구**한다. 이것이 분류에서 CE를 쓰는 핵심 이유. $\blacksquare$

---

# 채점 기준 총정리

| 문제 | 배점 | 교수 강조 우선순위 | 핵심 채점 포인트 |
|------|------|-----------------|----------------|
| P1 | 15 | **1순위** (MAP→MSE) | 전체 5단계 흐름. 각 단계의 가정(i.i.d., 가우시안, 상수 제거) 명시 |
| P2 | 12 | **2순위** (MAP 직접) | 미분 규칙 명시, 0으로 놓는 이유, prior 영향 |
| P3 | 10 | 1+2순위 통합 | MAP→MSE+L2. λ 유도. prior/data/regularization 관계 |
| P4 | 10 | **3순위** (Inductive Bias) | 4가지 아키텍처 비교. 데이터 양과 관계 |
| P5 | 10 | KL/CE/Ent 관계 | Gibbs 부등식 증명. CE 최소화=KL 최소화 |
| P6 | 10 | 분류 손실 유도 | Categorical→NLL→CE. i.i.d. 명시 |
| P7 | 10 | **1순위** 핵심 | Gaussian→NLL→MSE. 매 줄 설명 |
| P8 | 10 | MLE 직접 유도 | 가우시안 MLE. 독립/동일분포 각각 사용 지점 |
| P9 | 10 | **4순위** (CLT) | CLT→가우시안→MSE 체인. -log(exp)=제곱 설명 |
| P10 | 10 | 통합 이해 | 분류/회귀 통합. NLL 원리. MSE 비적합 이유 |
| P11 | 8 | 학습 정의 | Mitchell 정의, 학습 종류, i.i.d. 역할 |
| P12 | 10 | 경험적 분포 | $\hat{p}$, KL, 경험적 위험 vs 진짜 위험 |
| P13 | 10 | Ridge = MAP | 닫힌 해 유도. 가역성 보장. prior 해석 |
| P14 | 8 | 엔트로피 | 3가지 분포 계산. 확신도 연결. KL 관계에서 역할 |
| P15 | 10 | 시그모이드/그래디언트 | σ' 증명. CE 그래디언트 소거. vanishing gradient |
| P16 | 8 | CE vs MSE 비교 | MSE 그래디언트에 σ' 남음. 포화 문제 |
| **총점** | **169점** | | |

> **최고 우선순위 문제**: P1 + P2 + P7 + P9 (= 47점) — 교수 강조 1~4순위 직접 반영

---
title: "Ch.18 MLE와 MAP 추정"
slug: ch18-mle-map
order: 18
---

# Ch.18 MLE와 MAP 추정

---

## 1. 학습 목표

이 장을 마친 후 다음을 할 수 있어야 한다:

- 추정량(estimator)의 정의를 정확히 진술할 수 있다
- 우도함수(likelihood function)와 로그우도(log-likelihood)의 정의를 서술하고 계산할 수 있다
- 최대우도추정(Maximum Likelihood Estimation, MLE)의 정의를 진술하고, 주어진 데이터에서 MLE를 유도할 수 있다
- 최대사후확률추정(Maximum A Posteriori, MAP)의 정의를 진술하고, 사전분포를 도입하여 MAP를 유도할 수 있다
- Bernoulli 데이터에 대해 MLE와 MAP를 **처음부터 끝까지** 유도할 수 있다
- Gaussian 데이터에 대해 평균과 분산의 MLE를 유도할 수 있다
- MAP 추정이 정규화(regularization)와 어떻게 연결되는지 수식으로 설명할 수 있다
- Gaussian prior가 L2 정규화에 대응됨을 증명할 수 있다
- Uniform prior에서 MAP가 MLE와 일치함을 보일 수 있다
- MLE의 과적합 문제와 MAP의 완화 효과를 구체적 예시로 설명할 수 있다
- MLE와 MAP의 차이를 수식적·직관적으로 비교할 수 있다

---

## 2. 필수 정의

### Definition 18.1 (추정량, Estimator)

**추정량**(estimator) $\hat{\theta}$는 관측된 데이터 $D = \{x_1, x_2, \ldots, x_n\}$의 함수로서, 미지의 모수(parameter) $\theta$의 값을 추정하기 위해 사용되는 통계량이다.

$$\hat{\theta} = g(x_1, x_2, \ldots, x_n)$$

여기서 $g: \mathcal{X}^n \to \Theta$는 표본공간에서 모수공간으로의 함수이다.

> **해석:** 추정량은 데이터로부터 모수의 "최선의 추측"을 만들어내는 규칙이다. 좋은 추정량은 데이터가 많아질수록 참값에 가까워진다(일치성, consistency). 추정량 자체는 확률변수이며, 특정 데이터를 대입한 값을 **추정값**(estimate)이라 부른다.

**예시:**

표본평균 $\hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i$는 모평균 $\mu$에 대한 추정량이다. 데이터 $\{2, 4, 6\}$이 관측되었다면, 추정값은 $\hat{\mu} = 4$이다.

**비예시:**

$\hat{\theta} = 3$ (상수)은 데이터에 의존하지 않으므로 유용한 추정량이 아니다. 형식적으로는 데이터의 함수(상수함수)이지만, 어떤 데이터가 관측되든 같은 값을 반환하므로 데이터의 정보를 전혀 활용하지 못한다.

---

### Definition 18.2 (우도함수, Likelihood Function)

모수 $\theta$를 가지는 확률모형에서, 데이터 $D = \{x_1, x_2, \ldots, x_n\}$이 **독립동일분포**(i.i.d., independent and identically distributed)를 따를 때, **우도함수**(likelihood function)는 다음과 같이 정의된다:

$$L(\theta) = P(D|\theta) = \prod_{i=1}^{n} P(x_i|\theta)$$

> **해석:** 우도함수는 "이 모수값 $\theta$가 참이라면, 관측된 데이터가 나올 가능성이 얼마나 되는가?"를 측정한다. **핵심적 구별**: $P(x_i|\theta)$를 $x_i$의 함수로 보면 확률(분포)이지만, $\theta$의 함수로 보면 우도이다. 우도는 확률이 아니므로 $\int L(\theta)\,d\theta = 1$일 필요가 없다.

**예시:**

동전을 3번 던져 $D = \{H, T, H\}$를 관측했다. 동전의 앞면 확률을 $\theta$라 하면:

$$L(\theta) = P(H|\theta) \cdot P(T|\theta) \cdot P(H|\theta) = \theta \cdot (1-\theta) \cdot \theta = \theta^2(1-\theta)$$

$\theta = 0.5$이면 $L(0.5) = 0.125$, $\theta = 0.7$이면 $L(0.7) = 0.147$이다. 우도가 더 높은 $\theta = 0.7$ 쪽이 관측 데이터를 더 잘 설명한다.

---

### Definition 18.3 (로그우도, Log-Likelihood)

**로그우도**(log-likelihood)는 우도함수에 자연로그를 취한 것이다:

$$\ell(\theta) = \log L(\theta) = \log \prod_{i=1}^{n} P(x_i|\theta) = \sum_{i=1}^{n} \log P(x_i|\theta)$$

> **해석:** 곱을 합으로 변환하여 미분과 계산을 용이하게 한다. $\log$는 단조증가함수이므로, $L(\theta)$를 최대화하는 $\theta$와 $\ell(\theta)$를 최대화하는 $\theta$는 동일하다. 실용적으로도 많은 확률값의 곱은 수치적 언더플로(underflow)를 일으키지만, 로그를 취하면 합으로 바뀌어 안정적 계산이 가능하다.

**예시:**

위의 동전 예시에서:

$$\ell(\theta) = \log[\theta^2(1-\theta)] = 2\log\theta + \log(1-\theta)$$

$\theta = 0.5$이면 $\ell(0.5) = 2\log 0.5 + \log 0.5 = 3\log 0.5 \approx -2.079$

$\theta = 0.7$이면 $\ell(0.7) = 2\log 0.7 + \log 0.3 \approx -1.917$

---

### Definition 18.4 (최대우도추정, Maximum Likelihood Estimation)

**최대우도추정량**(Maximum Likelihood Estimator, MLE) $\hat{\theta}_{MLE}$는 우도함수를 최대화하는 모수값이다:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta) = \arg\max_{\theta} \ell(\theta)$$

> **해석:** MLE는 "관측된 데이터를 가장 그럴듯하게 만드는 모수값"을 선택한다. 오직 데이터만을 사용하며, 모수에 대한 사전 믿음(prior belief)을 고려하지 않는 **빈도주의**(frequentist) 관점의 추정 방법이다.

**예시:**

동전을 10번 던져 7번 앞면이 나왔다면, $L(\theta) = \theta^7(1-\theta)^3$이고:

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \theta^7(1-\theta)^3 = \frac{7}{10} = 0.7$$

(유도는 Section 4에서 상세히 다룬다.)

---

### Definition 18.5 (최대사후확률추정, Maximum A Posteriori Estimation)

모수에 대한 사전분포(prior) $P(\theta)$가 주어졌을 때, **최대사후확률추정량**(MAP estimator) $\hat{\theta}_{MAP}$는 사후분포(posterior)를 최대화하는 모수값이다:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D)$$

베이즈 정리에 의해:

$$P(\theta|D) = \frac{P(D|\theta)\,P(\theta)}{P(D)}$$

$P(D)$는 $\theta$에 의존하지 않으므로:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(D|\theta)\,P(\theta) = \arg\max_{\theta} \left[\ell(\theta) + \log P(\theta)\right]$$

> **해석:** MAP는 MLE에 사전 믿음을 추가한 것이다. 데이터가 적을 때 사전분포가 추정을 안정화하고, 데이터가 많아지면 우도가 지배적이 되어 MLE에 수렴한다. **베이즈주의**(Bayesian) 관점의 점추정 방법이다.

**예시:**

동전을 10번 던져 7번 앞면이 나왔고, 사전분포로 $\theta \sim \text{Beta}(2, 2)$를 설정하면:

$$\hat{\theta}_{MAP} = \frac{7 + 2 - 1}{10 + 2 + 2 - 2} = \frac{8}{12} = \frac{2}{3} \approx 0.667$$

MLE($0.7$)보다 $0.5$ 쪽으로 당겨진다 — 사전분포 $\text{Beta}(2, 2)$가 "동전은 공정할 가능성이 높다"는 믿음을 반영하기 때문이다.

---

## 3. 핵심 정리와 명제

### Theorem 18.1 (MLE의 1차 조건)

$L(\theta) > 0$이고 $\ell(\theta)$가 $\theta$에 대해 미분가능할 때, MLE는 다음의 1차 조건(first-order condition)을 만족한다:

$$\frac{d\ell(\theta)}{d\theta}\bigg|_{\theta = \hat{\theta}_{MLE}} = 0$$

다변수의 경우 (모수 벡터 $\boldsymbol{\theta} \in \mathbb{R}^d$):

$$\nabla_{\boldsymbol{\theta}}\,\ell(\boldsymbol{\theta})\big|_{\boldsymbol{\theta} = \hat{\boldsymbol{\theta}}_{MLE}} = \mathbf{0}$$

> **왜 중요한가:** 이 정리는 MLE를 구하는 실질적 방법을 제공한다. 우도함수를 직접 비교하는 대신, 로그우도를 미분하고 0으로 놓는 방정식을 풀면 된다.

> **어디에 쓰이는가:** 로지스틱 회귀, 신경망의 크로스엔트로피 손실 최소화 등 딥러닝의 거의 모든 학습 과정이 이 원리에 기반한다. 닫힌 해(closed-form)가 없을 때는 경사하강법(gradient descent)으로 수치적으로 푼다.

---

### Theorem 18.2 (MAP = MLE + Log-Prior 정규화)

MAP 추정은 다음과 동치이다:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \left[\ell(\theta) + \log P(\theta)\right]$$

즉, MAP는 로그우도에 **로그사전분포 항** $\log P(\theta)$를 더한 것을 최대화한다.

> **왜 중요한가:** MAP를 "MLE + 정규화 항"으로 볼 수 있음을 보여준다. $\log P(\theta)$는 특정 모수값에 페널티를 부여하여 극단적인 추정을 방지한다.

> **어디에 쓰이는가:** 딥러닝에서 weight decay(가중치 감쇠)가 바로 이 원리의 구현이다. 손실함수에 정규화 항 $\lambda\|\mathbf{w}\|^2$을 추가하는 것은 가중치에 Gaussian prior를 부여한 MAP 추정과 동일하다.

---

### Theorem 18.3 (Uniform Prior에서 MAP = MLE)

사전분포가 균등분포(uniform distribution)일 때, MAP 추정은 MLE와 일치한다:

$$P(\theta) = \text{const} \quad \Longrightarrow \quad \hat{\theta}_{MAP} = \hat{\theta}_{MLE}$$

> **왜 중요한가:** MLE가 MAP의 특수한 경우임을 보여준다. 즉, MLE는 "모수에 대한 사전 정보가 전혀 없다(uninformative prior)"는 가정 하의 MAP이다.

> **어디에 쓰이는가:** 사전분포를 명시하지 않고 학습하는 일반적인 딥러닝 모델(정규화 없는)이 사실상 MLE를 수행하고 있음을 정당화한다.

---

### Theorem 18.4 (Gaussian Prior와 L2 정규화의 동치)

모수에 대한 사전분포가 $\theta_j \overset{iid}{\sim} \mathcal{N}(0, \sigma_0^2)$이면, MAP 추정은 다음과 동치이다:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right] = \arg\min_{\boldsymbol{\theta}} \left[-\ell(\boldsymbol{\theta}) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right]$$

여기서 $\lambda = \frac{1}{\sigma_0^2}$이다.

> **왜 중요한가:** 확률적 추론(MAP)과 최적화 기법(L2 정규화)이 수학적으로 동일한 것임을 밝힌다. 이는 딥러닝에서 정규화를 사용하는 이론적 근거를 제공한다.

> **어디에 쓰이는가:**
> - **Weight decay**: SGD에서 $\mathbf{w} \leftarrow \mathbf{w} - \eta(\nabla L + \lambda \mathbf{w})$는 Gaussian prior 하에서의 MAP gradient step이다.
> - **Ridge regression**: $\min_{\mathbf{w}} \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$는 Gaussian 우도 + Gaussian prior의 MAP이다.

---

## 4. 공식 및 수식 유도

### 4.1 Bernoulli 데이터에 대한 MLE 유도

**설정.** $X_1, X_2, \ldots, X_n \overset{iid}{\sim} \text{Bernoulli}(\theta)$이고, 관측된 데이터에서 $k$개가 1(성공), $n-k$개가 0(실패)이다.

**Step 1: 우도함수 작성.**

$$L(\theta) = \prod_{i=1}^{n} P(x_i|\theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i}$$

$\sum_{i=1}^n x_i = k$이므로:

$$L(\theta) = \theta^k(1-\theta)^{n-k}$$

**Step 2: 로그우도 계산.**

$$\ell(\theta) = \log L(\theta) = k\log\theta + (n-k)\log(1-\theta)$$

**Step 3: 미분.**

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta}$$

**Step 4: 1차 조건 $\frac{d\ell}{d\theta} = 0$.**

$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

$$k(1-\theta) = (n-k)\theta$$

$$k - k\theta = n\theta - k\theta$$

$$k = n\theta$$

$$\boxed{\hat{\theta}_{MLE} = \frac{k}{n}}$$

**Step 5: 2차 조건 확인 (극대 검증).**

$$\frac{d^2\ell}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2} < 0 \quad (\text{for } 0 < \theta < 1, \; 0 < k < n)$$

2차 도함수가 음수이므로 $\hat{\theta}_{MLE} = k/n$은 극대점이다.

---

### 4.2 Bernoulli 데이터에 대한 MAP 유도 (Beta Prior)

**설정.** 데이터는 위와 동일. 사전분포로 $\theta \sim \text{Beta}(\alpha, \beta)$를 부여한다.

**Step 1: Beta 분포의 밀도함수.**

$$P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}, \quad B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

**Step 2: 사후분포의 로그 (상수 무시).**

$$\log P(\theta|D) = \log P(D|\theta) + \log P(\theta) + \text{const}$$

$$= k\log\theta + (n-k)\log(1-\theta) + (\alpha-1)\log\theta + (\beta-1)\log(1-\theta) + \text{const}$$

$$= (k + \alpha - 1)\log\theta + (n - k + \beta - 1)\log(1-\theta) + \text{const}$$

**Step 3: 미분하여 0으로 놓기.**

$$\frac{d}{d\theta}\log P(\theta|D) = \frac{k + \alpha - 1}{\theta} - \frac{n - k + \beta - 1}{1-\theta} = 0$$

**Step 4: 풀기.**

$$(k + \alpha - 1)(1-\theta) = (n - k + \beta - 1)\theta$$

$$k + \alpha - 1 = (n + \alpha + \beta - 2)\theta$$

$$\boxed{\hat{\theta}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}}$$

> **해석:** MLE 결과 $k/n$과 비교하면, MAP는 관측된 성공 횟수 $k$에 $\alpha - 1$을 더하고, 전체 시행 횟수 $n$에 $\alpha + \beta - 2$를 더한 것이다. 이를 **가상 관측**(pseudo-count)이라 부른다. $\alpha = \beta = 1$ (uniform prior)이면 $\hat{\theta}_{MAP} = k/n = \hat{\theta}_{MLE}$임에 주목하라 (Theorem 18.3의 확인).

---

### 4.3 Gaussian 데이터의 MLE 유도

**설정.** $X_1, X_2, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$. 모수 $\theta = (\mu, \sigma^2)$.

**Step 1: 우도함수.**

$$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**Step 2: 로그우도.**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Step 3: $\mu$에 대한 미분.**

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\sum_{i=1}^{n}(x_i - \mu) = 0 \quad \Longrightarrow \quad \sum_{i=1}^n x_i = n\mu$$

$$\boxed{\hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i}$$

**Step 4: $\sigma^2$에 대한 미분.**

$s = \sigma^2$로 치환하면:

$$\frac{\partial \ell}{\partial s} = -\frac{n}{2s} + \frac{1}{2s^2}\sum_{i=1}^{n}(x_i - \mu)^2 = 0$$

$$\frac{n}{2s} = \frac{1}{2s^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

$$ns = \sum_{i=1}^{n}(x_i - \mu)^2$$

$\mu = \hat{\mu}_{MLE} = \bar{x}$를 대입하면:

$$\boxed{\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

> **주의:** $\hat{\sigma}^2_{MLE}$는 $n$으로 나누므로 **편향된(biased)** 추정량이다. 비편향 추정량은 $n-1$로 나눈 표본분산 $S^2$이다. 이는 MLE가 항상 최선의 추정량은 아님을 보여주는 대표적 예시이다.

---

### 4.4 MAP와 L2 정규화 연결 유도

**설정.** 모수 벡터 $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_d)^T$에 독립 Gaussian prior를 부여한다:

$$\theta_j \overset{iid}{\sim} \mathcal{N}(0, \sigma_0^2), \quad j = 1, \ldots, d$$

**Step 1: Prior의 로그.**

$$\log P(\boldsymbol{\theta}) = \log \prod_{j=1}^{d} \frac{1}{\sqrt{2\pi\sigma_0^2}} \exp\left(-\frac{\theta_j^2}{2\sigma_0^2}\right)$$

$$= -\frac{d}{2}\log(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\sum_{j=1}^{d}\theta_j^2$$

$$= \text{const} - \frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|_2^2$$

**Step 2: MAP 목적함수.**

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) + \log P(\boldsymbol{\theta})\right]$$

$$= \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - \frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|_2^2\right]$$

$\lambda = \frac{1}{\sigma_0^2}$로 놓으면:

$$= \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right]$$

부호를 뒤집어 최소화 문제로 변환하면:

$$\boxed{\hat{\boldsymbol{\theta}}_{MAP} = \arg\min_{\boldsymbol{\theta}} \left[-\ell(\boldsymbol{\theta}) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right]}$$

이것은 정확히 **음의 로그우도(negative log-likelihood) + L2 정규화** 형태이다.

> **핵심 통찰:**
> - $\sigma_0^2 \to \infty$ (매우 넓은 prior) $\Rightarrow$ $\lambda \to 0$ $\Rightarrow$ 정규화 없음 $\Rightarrow$ MAP → MLE
> - $\sigma_0^2 \to 0$ (매우 좁은 prior) $\Rightarrow$ $\lambda \to \infty$ $\Rightarrow$ 강한 정규화 $\Rightarrow$ $\hat{\boldsymbol{\theta}} \to \mathbf{0}$

---

## 5. 증명

### Proof 18.1: Bernoulli MLE 유도 (Formal)

**증명 전략:** 로그우도를 $\theta$에 대해 미분하고, 1차 조건을 풀어 극값을 구한 뒤, 2차 조건으로 극대임을 확인한다.

---

**Proof.**

$X_1, \ldots, X_n \overset{iid}{\sim} \text{Bernoulli}(\theta)$, $\theta \in (0, 1)$라 하자. $k = \sum_{i=1}^n x_i$라 두면, 우도함수는:

$$L(\theta) = \prod_{i=1}^n \theta^{x_i}(1-\theta)^{1-x_i} = \theta^k(1-\theta)^{n-k}$$

로그우도:

$$\ell(\theta) = k\log\theta + (n-k)\log(1-\theta) \tag{1}$$

(1)을 $\theta$에 대해 미분:

$$\ell'(\theta) = \frac{k}{\theta} - \frac{n-k}{1-\theta} \tag{2}$$

$\ell'(\theta) = 0$으로 놓으면:

$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

양변에 $\theta(1-\theta)$를 곱하면:

$$k(1-\theta) = (n-k)\theta$$

$$k - k\theta = n\theta - k\theta$$

$$k = n\theta$$

$$\therefore \hat{\theta}_{MLE} = \frac{k}{n} \tag{3}$$

극대 확인. (2)를 한 번 더 미분:

$$\ell''(\theta) = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2} \tag{4}$$

$0 < k < n$이고 $\theta \in (0,1)$이면 (4)의 각 항이 모두 음수이므로 $\ell''(\theta) < 0$이다. 따라서 $\ell(\theta)$는 $(0,1)$에서 오목(concave)하고, $\hat{\theta}_{MLE} = k/n$은 유일한 전역 최대점이다. $\blacksquare$

---

### Proof 18.2: Bernoulli MAP 유도 (Formal)

**증명 전략:** 사후분포의 로그를 구성하고 미분하여 극값을 구한다. 사전분포 $\text{Beta}(\alpha, \beta)$의 효과가 가상 관측으로 작용함을 보인다.

---

**Proof.**

데이터 조건은 Proof 18.1과 동일. 사전분포 $\theta \sim \text{Beta}(\alpha, \beta)$, $\alpha, \beta > 1$로 가정한다.

사후분포의 로그 ($\theta$에 의존하지 않는 항을 상수 $C$로 처리):

$$\log P(\theta|D) = \log P(D|\theta) + \log P(\theta) + C$$

$$= k\log\theta + (n-k)\log(1-\theta) + (\alpha-1)\log\theta + (\beta-1)\log(1-\theta) + C$$

$$= (k+\alpha-1)\log\theta + (n-k+\beta-1)\log(1-\theta) + C \tag{5}$$

$A = k + \alpha - 1$, $B = n - k + \beta - 1$로 놓으면 (5)는 $A\log\theta + B\log(1-\theta) + C$이다.

미분하여 0으로 놓으면:

$$\frac{A}{\theta} - \frac{B}{1-\theta} = 0$$

$$A(1-\theta) = B\theta$$

$$A = (A + B)\theta$$

$$\theta = \frac{A}{A + B} = \frac{k + \alpha - 1}{(k + \alpha - 1) + (n - k + \beta - 1)}$$

$$= \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$

$$\therefore \hat{\theta}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} \tag{6}$$

$\alpha, \beta > 1$이면 $A, B > 0$이므로, 2차 도함수 $-A/\theta^2 - B/(1-\theta)^2 < 0$이 되어 극대가 보장된다. $\blacksquare$

---

### Proof 18.3: MAP = MLE + Prior 관계의 수식적 증명

**증명 전략:** 베이즈 정리를 적용하고, $\arg\max$에서 $\theta$에 무관한 항을 제거하여 MAP가 로그우도 + 로그사전분포의 최대화와 동치임을 보인다.

---

**Proof.**

베이즈 정리에 의해:

$$P(\theta|D) = \frac{P(D|\theta)\,P(\theta)}{P(D)} \tag{7}$$

MAP 추정:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D)$$

(7)에서 $P(D) = \int P(D|\theta')P(\theta')\,d\theta'$는 $\theta$에 의존하지 않는 양의 상수이므로:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(D|\theta)\,P(\theta) \tag{8}$$

$\log$는 단조증가이므로, (8)의 $\arg\max$는 보존된다:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \log\left[P(D|\theta)\,P(\theta)\right]$$

$$= \arg\max_{\theta} \left[\log P(D|\theta) + \log P(\theta)\right]$$

$$= \arg\max_{\theta} \left[\ell(\theta) + \log P(\theta)\right] \tag{9}$$

(9)에서 $P(\theta) = \text{const}$ (uniform prior)이면:

$$\hat{\theta}_{MAP} = \arg\max_{\theta}\,\ell(\theta) = \hat{\theta}_{MLE} \tag{10}$$

따라서 MAP는 MLE의 일반화이며, $\log P(\theta)$ 항이 정규화(regularization) 역할을 한다. $\blacksquare$

---

### Proof 18.4: Gaussian Prior → L2 Regularization 증명

**증명 전략:** 독립 Gaussian prior의 로그를 전개하여 L2 노름 형태를 얻고, MAP 목적함수에 대입하여 L2 정규화와의 동치를 보인다.

---

**Proof.**

모수 벡터 $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_d)^T$에 대해 $\theta_j \overset{iid}{\sim} \mathcal{N}(0, \sigma_0^2)$을 가정한다.

결합 사전분포:

$$P(\boldsymbol{\theta}) = \prod_{j=1}^d \frac{1}{\sqrt{2\pi\sigma_0^2}} \exp\left(-\frac{\theta_j^2}{2\sigma_0^2}\right) = (2\pi\sigma_0^2)^{-d/2} \exp\left(-\frac{\|\boldsymbol{\theta}\|_2^2}{2\sigma_0^2}\right)$$

로그를 취하면:

$$\log P(\boldsymbol{\theta}) = -\frac{d}{2}\log(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|_2^2 \tag{11}$$

(11)에서 첫째 항은 $\boldsymbol{\theta}$에 무관한 상수이다. Proof 18.3의 (9)에 대입하면:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - \frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|_2^2\right]$$

$\lambda = \frac{1}{\sigma_0^2}$로 정의하면:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right]$$

최대화를 최소화로 변환하면 ($\max f = -\min(-f)$):

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\min_{\boldsymbol{\theta}} \left[-\ell(\boldsymbol{\theta}) + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2\right] \tag{12}$$

(12)의 $-\ell(\boldsymbol{\theta})$는 음의 로그우도(negative log-likelihood, NLL)이고, $\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2$는 L2 정규화 항이다.

따라서 Gaussian prior 하에서의 MAP 추정은 NLL + L2 정규화의 최소화와 정확히 동치이다.

또한 $\sigma_0^2$와 $\lambda$의 관계에서:

- $\sigma_0^2$가 크면 (사전분포가 넓으면) $\lambda$가 작아 정규화가 약하다
- $\sigma_0^2$가 작으면 (사전분포가 좁으면) $\lambda$가 커 정규화가 강하다

이는 "모수가 0 근처에 있을 것이라는 확신이 강할수록(= $\sigma_0^2$ 작을수록) 더 강한 정규화를 가한다"는 직관과 일치한다. $\blacksquare$

---

## 6. 계산 예제와 단계별 풀이

### Example 18.1: 동전 던지기 MLE

**문제.** 동전을 10회 던져 7회 앞면(H), 3회 뒷면(T)이 나왔다. 앞면 확률 $\theta$의 MLE를 구하라.

**풀이.**

$n = 10$, $k = 7$ (앞면 횟수)이다.

**Step 1.** 우도함수:

$$L(\theta) = \theta^7(1-\theta)^3$$

**Step 2.** 로그우도:

$$\ell(\theta) = 7\log\theta + 3\log(1-\theta)$$

**Step 3.** 미분:

$$\ell'(\theta) = \frac{7}{\theta} - \frac{3}{1-\theta}$$

**Step 4.** $\ell'(\theta) = 0$:

$$\frac{7}{\theta} = \frac{3}{1-\theta} \implies 7(1-\theta) = 3\theta \implies 7 = 10\theta$$

$$\boxed{\hat{\theta}_{MLE} = 0.7}$$

**Step 5. 검증.** 몇 가지 값에서 $\ell(\theta)$를 비교한다:

| $\theta$ | $\ell(\theta)$ |
|-----------|----------------|
| 0.5 | $7\log 0.5 + 3\log 0.5 = 10\log 0.5 \approx -6.931$ |
| 0.7 | $7\log 0.7 + 3\log 0.3 \approx -2.497 - 3.612 = -6.109$ |
| 0.9 | $7\log 0.9 + 3\log 0.1 \approx -0.737 - 6.908 = -7.645$ |

$\theta = 0.7$에서 로그우도가 최대임을 확인할 수 있다.

---

### Example 18.2: 같은 데이터 + Beta Prior → MAP

**문제.** Example 18.1과 같은 데이터에서, 사전분포를 $\theta \sim \text{Beta}(2, 2)$로 놓았을 때 MAP 추정값을 구하라.

**풀이.**

$n = 10$, $k = 7$, $\alpha = 2$, $\beta = 2$이다.

**Step 1.** 사후분포의 로그 (상수 무시):

$$\log P(\theta|D) \propto (7 + 2 - 1)\log\theta + (3 + 2 - 1)\log(1-\theta)$$

$$= 8\log\theta + 4\log(1-\theta)$$

**Step 2.** 미분하여 0으로 놓기:

$$\frac{8}{\theta} - \frac{4}{1-\theta} = 0$$

$$8(1-\theta) = 4\theta$$

$$8 = 12\theta$$

$$\boxed{\hat{\theta}_{MAP} = \frac{8}{12} = \frac{2}{3} \approx 0.667}$$

**검산.** 공식 (6)을 직접 적용:

$$\hat{\theta}_{MAP} = \frac{7 + 2 - 1}{10 + 2 + 2 - 2} = \frac{8}{12} = \frac{2}{3} \quad \checkmark$$

> **비교:** $\hat{\theta}_{MLE} = 0.7$이지만 $\hat{\theta}_{MAP} = 0.667$이다. $\text{Beta}(2, 2)$ prior는 $\theta = 0.5$에서 최댓값을 가지므로, MAP 추정값이 MLE보다 $0.5$ 쪽으로 끌려갔다. 이를 **사전분포에 의한 축소**(shrinkage toward the prior)라 한다.

---

### Example 18.3: Gaussian 데이터의 MLE

**문제.** 다음 5개의 관측값이 $\mathcal{N}(\mu, \sigma^2)$에서 추출되었다: $D = \{2.1,\; 3.5,\; 2.8,\; 4.0,\; 3.1\}$. $\mu$와 $\sigma^2$의 MLE를 구하라.

**풀이.**

$n = 5$이다.

**Step 1. 평균의 MLE.**

$$\hat{\mu}_{MLE} = \bar{x} = \frac{2.1 + 3.5 + 2.8 + 4.0 + 3.1}{5} = \frac{15.5}{5} = 3.1$$

**Step 2. 분산의 MLE.**

각 편차의 제곱을 구한다:

| $x_i$ | $x_i - \bar{x}$ | $(x_i - \bar{x})^2$ |
|--------|-----------------|---------------------|
| 2.1 | $-1.0$ | $1.00$ |
| 3.5 | $0.4$ | $0.16$ |
| 2.8 | $-0.3$ | $0.09$ |
| 4.0 | $0.9$ | $0.81$ |
| 3.1 | $0.0$ | $0.00$ |

$$\hat{\sigma}^2_{MLE} = \frac{1}{5}(1.00 + 0.16 + 0.09 + 0.81 + 0.00) = \frac{2.06}{5} = 0.412$$

$$\boxed{\hat{\mu}_{MLE} = 3.1, \quad \hat{\sigma}^2_{MLE} = 0.412}$$

> **참고:** 비편향 표본분산은 $S^2 = \frac{2.06}{4} = 0.515$이다. $n$이 작을 때 MLE 분산은 과소추정하는 경향이 있다.

---

### Example 18.4: MLE와 MAP 결과 비교

**문제.** 동전을 3회 던져 3회 모두 앞면이 나왔다. (a) MLE를 구하라. (b) $\text{Beta}(3, 3)$ prior를 사용한 MAP를 구하라. (c) 결과를 비교하라.

**풀이.**

$n = 3$, $k = 3$이다.

**(a) MLE:**

$$\hat{\theta}_{MLE} = \frac{k}{n} = \frac{3}{3} = 1.0$$

MLE는 "앞면이 나올 확률이 100%"라고 추정한다.

**(b) MAP with $\text{Beta}(3, 3)$:**

$$\hat{\theta}_{MAP} = \frac{3 + 3 - 1}{3 + 3 + 3 - 2} = \frac{5}{7} \approx 0.714$$

**(c) 비교:**

| | MLE | MAP |
|---|-----|-----|
| 추정값 | 1.0 | 0.714 |
| 해석 | "절대 뒷면 안 나온다" | "앞면이 좀 더 잘 나온다" |

MLE는 데이터만 충실히 반영하여 극단적 추정($\theta = 1$)을 내놓았다. 이는 3회라는 적은 데이터에 **과적합**(overfitting)한 결과이다. 반면 MAP는 prior $\text{Beta}(3, 3)$이 "$\theta$가 $0.5$ 근처일 가능성이 높다"는 정보를 추가하여, 보다 합리적인 추정을 제공한다.

> **핵심 교훈:** 데이터가 적을수록 MLE는 과적합에 취약하며, 이때 적절한 prior를 도입한 MAP가 더 안정적인 추정을 제공한다.

---

## 7. 자주 나오는 함정과 반례

### 함정 1: MLE의 과적합 문제

**함정:** "MLE는 항상 좋은 추정량이다."

**반례:** Example 18.4에서 보았듯이, 동전을 3번 던져 3번 앞면이면 $\hat{\theta}_{MLE} = 1$이다. 이 추정에 따르면 다음 동전 던지기에서 뒷면이 나올 확률이 0인데, 이는 상식적으로 부적절하다.

극단적 예시: 관측 데이터가 하나도 없으면($n = 0$), $k/n$이 정의되지 않는다. MLE는 데이터가 충분하지 않으면 불안정하거나 정의 자체가 불가능할 수 있다.

**대처:** 사전분포를 도입하여 MAP를 사용하거나, 라플라스 평활(Laplace smoothing) 등의 기법을 적용한다.

---

### 함정 2: Prior 선택이 결과에 미치는 영향

**함정:** "Prior는 결과에 별 영향이 없다."

**반례:** 동일 데이터($n = 10$, $k = 7$)에서 prior에 따른 MAP 변화:

| Prior | $\hat{\theta}_{MAP}$ |
|-------|---------------------|
| $\text{Beta}(1, 1)$ (Uniform) | $7/10 = 0.700$ |
| $\text{Beta}(2, 2)$ | $8/12 = 0.667$ |
| $\text{Beta}(10, 10)$ | $16/28 = 0.571$ |
| $\text{Beta}(1, 5)$ | $7/14 = 0.500$ |

Prior가 강할수록(집중된 분포일수록) MAP는 prior의 최빈값(mode) 쪽으로 강하게 끌린다. 데이터가 많아지면 이 영향은 줄어들지만, 데이터가 적을 때는 prior 선택이 결정적이다.

**대처:** Prior는 도메인 지식에 기반하여 합리적으로 선택한다. 확실한 사전 정보가 없다면 약한 prior (weakly informative prior)를 사용한다.

---

### 함정 3: 로그 변환을 하지 않는 실수

**함정:** "우도함수를 직접 미분하면 된다."

**문제점:** $L(\theta) = \prod_{i=1}^n P(x_i|\theta)$를 곱의 형태로 직접 미분하면 곱의 미분법(product rule)이 필요하여 매우 복잡해진다. $n$개 항의 곱을 미분하면 $n$개의 항이 나온다.

또한 수치적으로도 문제가 된다. 예를 들어 $P(x_i|\theta) = 0.01$인 데이터 100개가 있으면:

$$L(\theta) = 0.01^{100} = 10^{-200}$$

이 값은 부동소수점(floating-point)으로 표현 불가능하다(언더플로). 반면:

$$\ell(\theta) = 100 \times \log 0.01 = -460.5$$

이는 문제없이 표현된다.

**대처:** 항상 로그우도를 사용한다. $\log$는 단조증가이므로 최대화 결과는 동일하다.

---

### 함정 4: MLE의 분산 추정은 편향되어 있다

**함정:** "MLE로 구한 모든 추정량은 비편향(unbiased)이다."

**반례:** Gaussian 분포의 분산 MLE는:

$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

이것의 기댓값은:

$$E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$$

즉, MLE 분산 추정량은 체계적으로 참값보다 작게 추정한다(과소추정). 이러한 편향은 $n$이 클 때는 무시할 만하지만, 소표본에서는 유의미하다.

**대처:** 비편향 추정이 필요하면 $n$ 대신 $n-1$로 나눈 표본분산 $S^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$을 사용한다.

---

### 함정 5: MAP는 완전한 베이즈 추론이 아니다

**함정:** "MAP를 구하면 베이즈 추론을 한 것이다."

**문제점:** MAP는 사후분포 $P(\theta|D)$의 **최빈값**(mode)만 사용하며, 사후분포의 전체 형태(퍼짐, 비대칭성 등)를 무시한다. 완전한 베이즈 추론은 사후분포 전체를 활용한다:

$$P(x_{new}|D) = \int P(x_{new}|\theta)\,P(\theta|D)\,d\theta$$

이 **사후예측분포**(posterior predictive distribution)는 모수의 불확실성을 반영하지만, MAP는 단일 점 $\hat{\theta}_{MAP}$만 사용하므로 불확실성 정보가 사라진다.

예: 사후분포가 이봉(bimodal)이면, mode는 두 봉우리 중 하나만 선택하고 나머지를 완전히 무시한다.

**대처:** 불확실성 추정이 중요한 경우 전체 사후분포를 사용한다 (MCMC, 변분추론 등).

---

## 8. 시험형 서술 문제와 모범답안

### 문제 1

> **Bernoulli 데이터에 대한 MLE를 유도하라.** $X_1, \ldots, X_n \overset{iid}{\sim} \text{Bernoulli}(\theta)$이고, 데이터에서 성공 횟수가 $k$일 때, $\hat{\theta}_{MLE}$를 구하고 극대임을 확인하라.

**모범답안.**

우도함수: $L(\theta) = \theta^k(1-\theta)^{n-k}$

로그우도:

$$\ell(\theta) = k\log\theta + (n-k)\log(1-\theta)$$

1차 조건:

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

$$k(1-\theta) = (n-k)\theta$$

$$k - k\theta = n\theta - k\theta$$

$$k = n\theta \implies \hat{\theta}_{MLE} = \frac{k}{n}$$

극대 확인:

$$\frac{d^2\ell}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2}$$

$0 < k < n$이고 $\theta \in (0,1)$이면, 두 항 모두 음수이므로 $\frac{d^2\ell}{d\theta^2} < 0$이다. 따라서 $\ell(\theta)$는 $(0,1)$에서 순오목(strictly concave)이고, $\hat{\theta}_{MLE} = k/n$은 유일한 전역 최대점이다. $\blacksquare$

---

### 문제 2

> **동일 문제에서 prior를 도입해 MAP를 유도하라.** Bernoulli 데이터에 사전분포 $\theta \sim \text{Beta}(\alpha, \beta)$를 부여했을 때 $\hat{\theta}_{MAP}$를 구하라. $\alpha = \beta = 1$일 때 MLE와의 관계를 설명하라.

**모범답안.**

사후분포의 로그 ($\theta$에 무관한 상수를 $C$로 처리):

$$\log P(\theta|D) = k\log\theta + (n-k)\log(1-\theta) + (\alpha-1)\log\theta + (\beta-1)\log(1-\theta) + C$$

$$= (k+\alpha-1)\log\theta + (n-k+\beta-1)\log(1-\theta) + C$$

미분:

$$\frac{d}{d\theta}\log P(\theta|D) = \frac{k+\alpha-1}{\theta} - \frac{n-k+\beta-1}{1-\theta} = 0$$

$$(k+\alpha-1)(1-\theta) = (n-k+\beta-1)\theta$$

$$k + \alpha - 1 = (n + \alpha + \beta - 2)\theta$$

$$\hat{\theta}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$

$\alpha = \beta = 1$을 대입하면:

$$\hat{\theta}_{MAP} = \frac{k + 1 - 1}{n + 1 + 1 - 2} = \frac{k}{n} = \hat{\theta}_{MLE}$$

$\text{Beta}(1, 1)$은 $[0, 1]$ 위의 균등분포이다. 균등 prior는 모든 $\theta$ 값을 동등하게 취급하므로, 사후분포는 우도함수에만 의존하게 되어 MAP가 MLE와 일치한다. 이는 Theorem 18.3의 구체적 확인이다. $\blacksquare$

---

### 문제 3

> **MLE와 MAP의 차이를 수식으로 설명하라.** 두 추정 방법의 목적함수를 비교하고, MAP에서 prior가 어떤 역할을 하는지 서술하라.

**모범답안.**

MLE는 우도(데이터 적합도)만을 최대화한다:

$$\hat{\theta}_{MLE} = \arg\max_{\theta}\; \ell(\theta) = \arg\max_{\theta} \sum_{i=1}^n \log P(x_i|\theta)$$

MAP는 우도에 사전분포를 추가하여 최대화한다:

$$\hat{\theta}_{MAP} = \arg\max_{\theta} \left[\ell(\theta) + \log P(\theta)\right]$$

두 식의 차이는 오직 $\log P(\theta)$ 항에 있다. 이 항의 역할은 다음과 같다:

1. **정규화(Regularization):** $\log P(\theta)$는 사전 믿음에 부합하지 않는 모수값에 페널티를 부여한다. 예를 들어 $\theta \sim \mathcal{N}(0, \sigma_0^2)$이면 $\log P(\theta) \propto -\theta^2/(2\sigma_0^2)$이므로, $|\theta|$가 클수록 페널티가 커진다.

2. **과적합 방지:** 데이터가 적을 때 MLE는 데이터의 노이즈에 과적합할 수 있다. $\log P(\theta)$는 이를 완화하여 더 안정적인 추정을 제공한다.

3. **데이터 양에 따른 상대적 영향:** $\ell(\theta) = \sum_{i=1}^n \log P(x_i|\theta)$는 $n$에 비례하여 커지지만, $\log P(\theta)$는 $n$과 무관하다. 따라서:
   - $n$이 작을 때: $\log P(\theta)$의 영향이 상대적으로 크다 → MAP $\neq$ MLE
   - $n \to \infty$일 때: $\ell(\theta)$가 지배적 → MAP $\to$ MLE

4. **특수한 경우:** $P(\theta) = \text{const}$ (uniform prior)이면 $\log P(\theta) = \text{const}$이므로 MAP = MLE.

요약하면, MAP는 MLE를 포함하는 더 일반적인 프레임워크이며, prior를 통해 도메인 지식과 정규화를 추정 과정에 자연스럽게 반영한다. $\blacksquare$

---

### 문제 4

> **Gaussian prior가 L2 정규화에 대응됨을 증명하라.** 모수 $\boldsymbol{\theta} \in \mathbb{R}^d$에 $\theta_j \overset{iid}{\sim} \mathcal{N}(0, \sigma_0^2)$의 prior를 부여했을 때, MAP 추정이 NLL + L2 정규화와 동치임을 보이고, 정규화 강도 $\lambda$와 $\sigma_0^2$의 관계를 명시하라.

**모범답안.**

결합 사전분포:

$$P(\boldsymbol{\theta}) = \prod_{j=1}^d \frac{1}{\sqrt{2\pi\sigma_0^2}} \exp\left(-\frac{\theta_j^2}{2\sigma_0^2}\right)$$

로그:

$$\log P(\boldsymbol{\theta}) = -\frac{d}{2}\log(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\sum_{j=1}^d \theta_j^2 = C - \frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|_2^2$$

여기서 $C$는 $\boldsymbol{\theta}$에 무관한 상수이다.

MAP 목적함수:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) + \log P(\boldsymbol{\theta})\right] = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - \frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|_2^2\right]$$

$\lambda = 1/\sigma_0^2$으로 정의하고, 최대화를 최소화로 변환하면:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\min_{\boldsymbol{\theta}} \underbrace{\left[-\ell(\boldsymbol{\theta})\right]}_{\text{NLL}} + \underbrace{\frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2}_{\text{L2 정규화}}$$

이것은 정확히 L2 정규화가 적용된 음의 로그우도 최소화이다.

$\lambda$와 $\sigma_0^2$의 관계: $\lambda = 1/\sigma_0^2$.
- $\sigma_0^2 \to \infty$: 사전분포가 넓어짐 → $\lambda \to 0$ → 정규화 효과 소멸 → MAP → MLE
- $\sigma_0^2 \to 0$: 사전분포가 원점에 집중 → $\lambda \to \infty$ → 강한 정규화 → $\hat{\boldsymbol{\theta}} \to \mathbf{0}$

따라서 딥러닝에서 weight decay 계수를 크게 설정하는 것은, 가중치가 0에 가까울 것이라는 강한 사전 믿음(좁은 Gaussian prior)을 부여하는 것과 수학적으로 동치이다. $\blacksquare$

---

### 문제 5

> **다음 상황에서 MLE와 MAP를 각각 계산하고 비교하라.** 어떤 약의 부작용 발생 확률을 추정하려 한다. 10명의 환자 중 1명에게 부작용이 발생했다. (a) MLE를 구하라. (b) 의학 문헌에 따르면 유사 약물의 부작용 발생률은 보통 5% 근처이다. 이를 반영한 $\text{Beta}(2, 38)$ prior를 사용하여 MAP를 구하라. (c) 두 결과를 비교하고, 어느 추정이 더 합리적인지 논하라.

**모범답안.**

$n = 10$, $k = 1$ (부작용 발생 수).

**(a) MLE:**

$$\hat{\theta}_{MLE} = \frac{k}{n} = \frac{1}{10} = 0.1$$

**(b) MAP with $\text{Beta}(2, 38)$:**

$\text{Beta}(2, 38)$의 최빈값(mode)은 $\frac{\alpha-1}{\alpha+\beta-2} = \frac{1}{38} \approx 0.026$으로, 부작용 발생률이 약 2.6%일 것이라는 사전 믿음을 반영한다.

$$\hat{\theta}_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} = \frac{1 + 2 - 1}{10 + 2 + 38 - 2} = \frac{2}{48} = \frac{1}{24} \approx 0.042$$

**(c) 비교:**

| | MLE | MAP |
|---|-----|-----|
| 추정값 | 0.100 (10%) | 0.042 (4.2%) |

MLE는 "10명 중 1명" 데이터에만 의존하여 10%라는 다소 높은 추정을 내놓았다. 그러나 표본 크기가 10명으로 매우 작아, 이 추정의 불확실성이 크다.

MAP는 의학 문헌의 사전 정보(유사 약물의 부작용 발생률 ~5%)를 반영하여 4.2%로 추정했다. 이는 MLE보다 사전 지식에 가까우면서도, 관측된 데이터(10명 중 1명)의 정보도 반영한 절충적 결과이다.

이 상황에서는 MAP가 더 합리적이다:
1. **소표본**: $n = 10$이므로 데이터만으로는 신뢰할 수 있는 추정이 어렵다
2. **사전 정보의 존재**: 의학 문헌이라는 신뢰할 수 있는 사전 정보가 있다
3. **극단값 회피**: MLE의 10%는 유사 약물 대비 지나치게 높은 추정이며, MAP는 이를 완화한다

만약 환자 수를 1000명으로 늘려 100명에게 부작용이 발생한다면:

$$\hat{\theta}_{MLE} = \frac{100}{1000} = 0.1, \qquad \hat{\theta}_{MAP} = \frac{101}{1038} \approx 0.097$$

데이터가 충분하면 MAP도 MLE에 가까워지며, prior의 영향은 미미해진다. 이는 **데이터가 prior를 압도(overwhelm)**하는 현상이다. $\blacksquare$

---

### 문제 6 (심화)

> **Laplace prior $P(\theta_j) \propto \exp(-b|\theta_j|)$를 사용한 MAP 추정이 L1 정규화에 대응됨을 보이라.**

**모범답안.**

Laplace 분포의 밀도: $P(\theta_j) = \frac{b}{2}\exp(-b|\theta_j|)$

결합 사전분포의 로그:

$$\log P(\boldsymbol{\theta}) = \sum_{j=1}^d \log\left(\frac{b}{2}\right) - b\sum_{j=1}^d |\theta_j| = C - b\|\boldsymbol{\theta}\|_1$$

MAP 목적함수:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\max_{\boldsymbol{\theta}} \left[\ell(\boldsymbol{\theta}) - b\|\boldsymbol{\theta}\|_1\right]$$

최소화로 변환:

$$\hat{\boldsymbol{\theta}}_{MAP} = \arg\min_{\boldsymbol{\theta}} \left[-\ell(\boldsymbol{\theta}) + b\|\boldsymbol{\theta}\|_1\right]$$

이것은 NLL + L1 정규화(Lasso)와 정확히 동치이다.

따라서:
- **Gaussian prior** $\mathcal{N}(0, \sigma_0^2)$ → **L2 정규화** (Ridge) — 모수를 0 근처로 **축소**
- **Laplace prior** $\text{Laplace}(0, 1/b)$ → **L1 정규화** (Lasso) — 모수를 정확히 **0으로** 만듦 (희소성 유도)

L1 정규화가 희소 해(sparse solution)를 만드는 이유는 Laplace 분포가 $\theta = 0$에서 뾰족한 첨점(cusp)을 가지기 때문이다. $\blacksquare$

---

### 문제 7

> **Gaussian 데이터 $X_1, \ldots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2)$에서 평균 $\mu$의 MLE를 유도하라.** ($\sigma^2$는 기지(known)라 가정)

**모범답안.**

우도함수:

$$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

로그우도:

$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

$\mu$에 대해 미분:

$$\frac{d\ell}{d\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = \frac{1}{\sigma^2}\left(\sum_{i=1}^n x_i - n\mu\right)$$

1차 조건 $\frac{d\ell}{d\mu} = 0$:

$$\sum_{i=1}^n x_i - n\mu = 0 \implies \hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

2차 조건 확인:

$$\frac{d^2\ell}{d\mu^2} = -\frac{n}{\sigma^2} < 0$$

$\ell(\mu)$는 $\mu$에 대해 순오목이므로 $\hat{\mu}_{MLE} = \bar{x}$는 유일한 전역 최대점이다.

> **직관:** Gaussian 로그우도에서 $\mu$에 의존하는 부분은 $-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2$이다. 이를 최대화하는 것은 $\sum(x_i - \mu)^2$을 최소화하는 것과 같고, 이는 $\mu$가 데이터의 "중심"인 표본평균일 때 달성된다. 따라서 Gaussian MLE는 최소제곱법(least squares)과 동치이다. $\blacksquare$

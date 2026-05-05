---
title: "Deep Learning 중간고사 모의시험 — 3회 (교수 출제 스타일 반영)"
slug: midterm-exam-03
order: 3
---

# Deep Learning 중간고사 모의시험 — 3회 (교수 출제 스타일 반영)

> **핵심 평가 기준**: 답이 아니라 **논리적 도달 과정**을 평가한다. 수식만 나열하면 0점.
> 매 단계마다 (1) 어떤 가정을 사용했는지, (2) 왜 이 전개를 하는지를 **글로** 설명해야 한다.
> **문제는 영어**, **풀이는 한국어**로 작성.
> **시간**: 120분 | **총점**: 200점

---

# Problem 1 (15점) — MAP → NLL → MSE 전체 흐름

**[EN]** Starting from Bayes' theorem, derive the full chain:

$$\text{MAP} \to \text{Posterior} \to \text{NLL} \to \text{Gaussian assumption} \to \text{MSE Loss}$$

Specifically:
(a) Write Bayes' theorem for the posterior $p(\theta | D)$. Explain each term. (3점)
(b) Show that MAP estimation reduces to minimizing $-\log p(D|\theta) - \log p(\theta)$. State clearly which term you dropped and why. (3점)
(c) Assuming i.i.d. data, expand $-\log p(D|\theta)$ into a sum. State where the i.i.d. assumption is used. (3점)
(d) Assume Gaussian noise: $y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$. Show step by step that minimizing NLL becomes minimizing MSE. At each step, state what you are doing and why. (3점)
(e) If the prior is $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$, show that the full MAP objective becomes MSE + L2 regularization. Derive $\lambda$ in terms of $\sigma^2$ and $\sigma_p^2$. (3점)

---

## 풀이

**(a)** 베이즈 정리를 적용하여 파라미터 $\theta$의 사후 확률을 쓴다.

$$p(\theta | D) = \frac{p(D | \theta) \cdot p(\theta)}{p(D)}$$

각 항의 의미:
- $p(\theta | D)$: **사후 확률(Posterior)**. 데이터 $D$를 관찰한 후 파라미터 $\theta$에 대한 갱신된 믿음.
- $p(D | \theta)$: **우도(Likelihood)**. 파라미터가 $\theta$일 때, 이 데이터 $D$가 관측될 확률.
- $p(\theta)$: **사전 확률(Prior)**. 데이터를 보기 전 $\theta$에 대한 사전 믿음.
- $p(D)$: **증거(Evidence)**. 모든 가능한 $\theta$에 대해 적분한 정규화 상수. $\theta$에 의존하지 않는다.

**(b)** MAP 추정은 사후 확률을 최대화하는 $\theta$를 찾는 것이다.

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta | D) = \arg\max_\theta \frac{p(D|\theta) \cdot p(\theta)}{p(D)}$$

여기서 $p(D)$는 $\theta$에 의존하지 않는 상수이므로, **$\arg\max$에 영향을 주지 않아 제거할 수 있다**. 이것이 핵심적인 단순화 단계다.

$$= \arg\max_\theta \; p(D|\theta) \cdot p(\theta)$$

로그는 단조 증가 함수이므로 $\arg\max$의 위치를 바꾸지 않는다. **계산의 편의를 위해** (곱을 합으로 바꾸기 위해) 양변에 로그를 취한다.

$$= \arg\max_\theta \; [\log p(D|\theta) + \log p(\theta)]$$

최대화를 최소화로 바꾸기 위해 부호를 반전한다.

$$= \arg\min_\theta \; [-\log p(D|\theta) - \log p(\theta)]$$

$$= \arg\min_\theta \; [\underbrace{\text{NLL}(\theta)}_{-\log p(D|\theta)} + \underbrace{\text{Regularization}}_{-\log p(\theta)}]$$

**(c)** 데이터 $D = \{(x_1, y_1), \ldots, (x_n, y_n)\}$에 대해, **i.i.d. 가정**을 사용한다.

i.i.d.(independent and identically distributed)란: 각 데이터 포인트가 (1) 서로 **독립**적으로, (2) **동일한 분포**에서 생성되었다는 가정이다.

**독립 가정에 의해**, 결합 확률이 개별 확률의 곱으로 분해된다:

$$p(D|\theta) = \prod_{i=1}^{n} p(y_i | x_i, \theta)$$

이것이 가능한 이유: 독립이면 $p(A \cap B) = p(A) \cdot p(B)$이고, 이것을 $n$개 데이터에 확장한 것이다. **만약 데이터가 독립이 아니라면 (예: 시계열 데이터) 이 분해는 성립하지 않는다.**

로그를 취하면 곱이 합으로 변환된다:

$$-\log p(D|\theta) = -\sum_{i=1}^{n} \log p(y_i | x_i, \theta)$$

이 변환에서 로그의 성질 $\log(ab) = \log a + \log b$를 사용했다.

**(d)** 이제 각 데이터의 우도에 가우시안 분포를 가정한다.

왜 가우시안인가? 회귀 문제에서 노이즈 $\epsilon_i$는 측정 오차, 누락 변수 등 수많은 미지 요인의 합이다. **중심극한정리(CLT)**에 의해, 많은 독립적 요인의 합은 정규분포에 수렴한다. 따라서 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$로 가정하는 것이 이론적으로 정당화된다.

$y_i = f_\theta(x_i) + \epsilon_i$이므로 $y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$이고:

$$p(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

이것의 음의 로그를 구한다. 로그를 취하는 이유는 지수 함수 안의 제곱항을 꺼내기 위함이다:

$$-\log p(y_i | x_i, \theta) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}$$

$n$개 데이터에 대해 합산하면:

$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

$\theta$에 대해 최소화할 때, 첫째 항 $\frac{n}{2}\log(2\pi\sigma^2)$는 $\theta$에 무관한 상수이므로 제거할 수 있다. $\frac{1}{2\sigma^2}$도 양의 상수이므로 $\arg\min$에 영향을 주지 않는다.

따라서:

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 = \arg\min_\theta \; n \cdot \text{MSE}$$

**가우시안 노이즈 하의 NLL 최소화 = MSE 최소화**. $\blacksquare$

핵심 인과관계:
```
CLT → 노이즈가 가우시안 → 우도가 가우시안 → -log 취하면 제곱항 → MSE
```

**(e)** 사전분포 $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$이면:

$$p(\theta) = \frac{1}{(2\pi\sigma_p^2)^{d/2}} \exp\left(-\frac{\|\theta\|^2}{2\sigma_p^2}\right)$$

음의 로그:

$$-\log p(\theta) = \frac{d}{2}\log(2\pi\sigma_p^2) + \frac{\|\theta\|^2}{2\sigma_p^2}$$

$\theta$에 무관한 상수를 제거하면 $-\log p(\theta) \propto \frac{1}{2\sigma_p^2}\|\theta\|^2$.

MAP 전체 목적함수:

$$\arg\min_\theta \left[\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2 + \frac{1}{2\sigma_p^2}\|\theta\|^2\right]$$

공통 인자 $\frac{1}{2\sigma^2}$으로 나누면 (양의 상수이므로 $\arg\min$ 불변):

$$= \arg\min_\theta \left[\frac{1}{n}\sum_{i=1}^n (y_i - f_\theta(x_i))^2 + \frac{\sigma^2}{n\sigma_p^2}\|\theta\|^2\right]$$

$$= \arg\min_\theta \left[\text{MSE} + \lambda \|\theta\|^2\right], \quad \boxed{\lambda = \frac{\sigma^2}{n\sigma_p^2}}$$

해석: $\sigma_p^2$가 작으면 (prior가 "θ는 0 근처"라고 강하게 믿으면) → $\lambda$가 커짐 → 정규화 강해짐. $n$이 크면 (데이터가 많으면) → $\lambda$가 작아짐 → prior의 영향 약해짐. $\blacksquare$

---

# Problem 2 (12점) — MAP 직접 풀어보기 (동전 던지기)

**[EN]** You flip a coin $n = 10$ times and observe $k = 7$ heads. Assume:
- Likelihood: Binomial, $p(D|\theta) = \theta^k (1-\theta)^{n-k}$
- Prior: $p(\theta) = \text{Beta}(\alpha, \beta)$ with $\alpha = 3, \beta = 3$

(a) Write the log-posterior. Clearly label which part comes from the likelihood and which from the prior. (3점)
(b) Find $\hat{\theta}_{\text{MAP}}$ by differentiating and setting to zero. At each step, explain what mathematical operation you are performing and why. (5점)
(c) Compare $\hat{\theta}_{\text{MAP}}$ with $\hat{\theta}_{\text{MLE}}$. Explain the difference in terms of the prior's influence, and describe what happens as $n \to \infty$. (4점)

---

## 풀이

**(a)** 사후확률은 베이즈 정리에 의해 우도 × 사전확률에 비례한다.

**우도 부분**: 이항분포에서 $\theta$에 의존하는 부분만 남기면 (조합 계수는 $\theta$에 무관하므로 제거):
$$\log p(D|\theta) = k\log\theta + (n-k)\log(1-\theta) = 7\log\theta + 3\log(1-\theta)$$

**사전확률 부분**: $\text{Beta}(3,3)$에서 $p(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^2(1-\theta)^2$이므로:
$$\log p(\theta) = 2\log\theta + 2\log(1-\theta) + \text{const}$$

**로그 사후확률 (합산)**:
$$\log p(\theta|D) = \underbrace{7\log\theta + 3\log(1-\theta)}_{\text{우도(Likelihood)에서}} + \underbrace{2\log\theta + 2\log(1-\theta)}_{\text{사전확률(Prior)에서}} + C$$
$$= 9\log\theta + 5\log(1-\theta) + C$$

**(b)** MAP 추정은 로그 사후확률을 최대화하는 $\theta$를 찾는 것이다. 이를 위해 $\theta$에 대해 미분하고, 기울기가 0인 점을 찾는다. 왜냐하면 (이후 확인할) 이 함수가 위로 볼록(concave)하므로, 기울기 0인 점이 최대가 되기 때문이다.

$$\frac{d}{d\theta}[9\log\theta + 5\log(1-\theta)] = \frac{9}{\theta} - \frac{5}{1-\theta}$$

$\frac{9}{\theta}$는 $\log\theta$를 $\theta$에 대해 미분한 것이고 ($(\log x)' = 1/x$), $\frac{5}{1-\theta}$는 $\log(1-\theta)$를 미분한 것이다 (체인룰: $(\log(1-\theta))' = \frac{-1}{1-\theta}$, 부호가 반대).

이것을 0으로 놓는다:
$$\frac{9}{\theta} = \frac{5}{1-\theta}$$

교차 곱:
$$9(1-\theta) = 5\theta$$
$$9 - 9\theta = 5\theta$$
$$9 = 14\theta$$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{9}{14} \approx 0.643}$$

**일반 공식 확인**: Beta-Binomial 켤레쌍에서 MAP = $\frac{k + \alpha - 1}{n + \alpha + \beta - 2} = \frac{7+2}{10+4} = \frac{9}{14}$ ✓

**(c)** MLE는 사전확률을 무시하고 우도만 최대화한다:

$$\hat{\theta}_{\text{MLE}} = \frac{k}{n} = \frac{7}{10} = 0.700$$

비교:
- $\hat{\theta}_{\text{MAP}} = 0.643$ < $\hat{\theta}_{\text{MLE}} = 0.700$

MAP가 MLE보다 0.5(공정 동전)에 가까운 이유: Beta(3,3) 사전분포는 $\theta = 0.5$에서 최대인 대칭 분포이므로, MAP 추정을 0.5 방향으로 "끌어당기는" 효과가 있다. 이것이 prior의 역할이다.

$n \to \infty$이면:
$$\hat{\theta}_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} \approx \frac{k}{n} = \hat{\theta}_{\text{MLE}}$$

분자·분모에서 $\alpha, \beta$의 영향이 $n, k$에 비해 무시할 만큼 작아지기 때문이다. **데이터가 충분하면 사전 지식의 영향이 사라지고, MAP과 MLE가 수렴한다.** 이것은 베이지안 추론의 중요한 성질이다. $\blacksquare$

---

# Problem 3 (10점) — CLT가 가우시안 가정을 정당화하는 과정

**[EN]** In a regression problem $y = f_\theta(x) + \epsilon$:

(a) The noise $\epsilon$ arises from many unknown factors: measurement error, unobserved variables, environmental fluctuations, etc. Using the Central Limit Theorem (CLT), explain why modeling $\epsilon$ as Gaussian is reasonable. State the CLT precisely. (4점)
(b) Starting from $\epsilon \sim \mathcal{N}(0, \sigma^2)$, show the complete derivation that NLL minimization equals MSE minimization. At each step, clearly state what mathematical property or assumption you are using. (6점)

---

## 풀이

**(a)** 중심극한정리(CLT)를 정확히 서술한다.

**CLT**: $X_1, X_2, \ldots, X_n$이 서로 **독립**이고 **동일한 분포**(i.i.d.)를 따르며, 평균 $\mu$와 유한한 분산 $\sigma^2$을 가지면, 표본 평균 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$의 분포는 $n \to \infty$일 때 정규분포에 수렴한다:

$$\sqrt{n} \frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)$$

노이즈 $\epsilon$에 적용하는 논리: 회귀 문제에서 예측값과 실제값의 차이(노이즈)는 **수많은 미지 요인**의 합산 효과이다. 예: 센서 오차, 날씨 변화, 사용자 행동의 미세한 차이 등. 이 요인들이 각각 독립적이고 개별 영향이 크지 않다면, CLT에 의해 이들의 합인 $\epsilon$은 근사적으로 가우시안 분포를 따른다. 따라서 $\epsilon \sim \mathcal{N}(0, \sigma^2)$라는 가정은 이론적 근거가 있다.

단, CLT가 적용되려면 각 요인이 **독립**이고 **유한한 분산**을 가져야 한다는 전제 조건이 필요하다.

**(b)** $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$이면 $y_i = f_\theta(x_i) + \epsilon_i$이므로, $y_i$의 조건부 분포는:

$$y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$$

**[Step 1: 우도 함수 쓰기]** 가우시안 밀도 함수를 적용한다:

$$p(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

**[Step 2: i.i.d. 가정으로 결합 우도 분해]** 각 데이터 포인트가 독립적으로 생성되었다는 **i.i.d. 가정**에 의해, 전체 데이터의 결합 우도는 개별 우도의 곱이다:

$$p(D|\theta) = \prod_{i=1}^{n} p(y_i | x_i, \theta)$$

여기서 **독립 가정이 사용**되었다. 독립이 아니면 이 곱 분해는 불가능하다.

**[Step 3: 로그 취하기]** 곱을 합으로 바꾸기 위해 로그를 취한다. 로그는 단조 증가 함수이므로 최대/최소 위치를 보존한다:

$$\log p(D|\theta) = \sum_{i=1}^{n} \log p(y_i | x_i, \theta)$$

$$= \sum_{i=1}^{n}\left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right]$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

**[Step 4: NLL로 변환]** 음의 로그 우도(NLL)은 위에 $-1$을 곱한 것이다:

$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

**[Step 5: θ에 무관한 항 제거]** 첫째 항 $\frac{n}{2}\log(2\pi\sigma^2)$는 $\theta$를 포함하지 않는 **상수**이다. $\arg\min$을 구할 때 상수는 최솟값의 위치를 바꾸지 않으므로 제거한다.

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

**[Step 6: 양의 상수 제거]** $\frac{1}{2\sigma^2}$은 양의 상수이므로 $\arg\min$에 영향을 주지 않는다.

$$= \arg\min_\theta \sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 = \arg\min_\theta \; n \cdot \text{MSE}(\theta)$$

$$\boxed{\text{가우시안 노이즈 가정 하에서 NLL 최소화} = \text{MSE 최소화}}$$ $\blacksquare$

---

# Problem 4 (10점) — Inductive Bias와 아키텍처 선택

**[EN]**
(a) Define "inductive bias" in the context of machine learning. Give one concrete example for each: linear model, CNN, and Transformer. (4점)
(b) A researcher has a small dataset (100 labeled images) and a large dataset (10 million images). For each scenario, recommend which architecture (linear model, CNN, or Transformer/ViT) is more suitable and explain why in terms of the strength of inductive bias vs. data requirements. (3점)
(c) Explain why strong inductive bias helps with small data but can hurt with large data. Connect this to the bias-variance tradeoff. (3점)

---

## 풀이

**(a)** Inductive bias(귀납적 편향)란, 모델이 학습 전에 가지고 있는 **"어떤 종류의 함수가 정답일 가능성이 높다"는 사전 가정**이다. 이것은 데이터에서 배우는 것이 아니라, 모델의 아키텍처 자체에 내장되어 있다.

- **선형 모델**: "입출력 관계가 직선(또는 초평면)이다"라는 매우 강한 가정. 가장 강한 inductive bias. 직선이 아닌 관계는 표현 불가.
- **CNN**: "특징 감지기(필터)는 위치에 관계없이 동일하다(translation equivariance)"라는 가정. 이미지에서 고양이가 왼쪽에 있든 오른쪽에 있든 같은 필터로 감지할 수 있다. 이미지에 특화된 강한 inductive bias.
- **Transformer**: 최소한의 inductive bias만 가진다. Positional encoding만 위치 정보를 제공하고, 나머지는 데이터에서 전부 학습한다. Inductive bias가 약함.

**(b)**

**소량 데이터 (100장)**: **CNN** 추천.

이유: 데이터가 적으면, 모델이 데이터에서 배울 수 있는 정보가 제한적이다. CNN의 강한 inductive bias(국소성, 이동 불변성)는 이미지 도메인에 대한 사전 지식을 아키텍처에 이미 담고 있으므로, 적은 데이터로도 합리적인 표현을 학습할 수 있다. Transformer는 이런 사전 지식이 없어서 데이터에서 전부 배워야 하므로, 100장으로는 부족하다.

**대량 데이터 (1000만장)**: **Transformer (ViT)** 추천.

이유: 데이터가 충분하면 강한 inductive bias가 오히려 제약이 된다. CNN의 "국소적 패턴만 보라"는 가정이 전역적 관계 학습을 방해할 수 있다. Transformer는 inductive bias가 약하므로 데이터가 충분할 때 더 유연하게 복잡한 패턴을 학습한다. 실제로 ViT는 대규모 데이터에서 CNN을 능가한다.

**(c)** 강한 inductive bias = 높은 Bias, 낮은 Variance.

소량 데이터에서: 데이터가 적으면 Variance(모델 예측의 흔들림)가 커지기 쉬움. 강한 inductive bias는 모델의 가설 공간을 제한하여 Variance를 줄인다. 이때 약간의 Bias 증가는 감수할 만하다.

대량 데이터에서: 데이터가 충분하면 Variance는 자연스럽게 줄어든다. 이 상황에서 강한 inductive bias는 불필요한 제약을 주어 Bias만 증가시킨다. 약한 inductive bias(Transformer)가 데이터에서 직접 최적의 표현을 학습하므로, Bias도 Variance도 낮출 수 있다.

**핵심**: 데이터 양과 inductive bias 강도의 균형이 최적 모델 선택의 핵심이다. $\blacksquare$

---

# Problem 5 (10점) — MLE의 수학적 유도: 베르누이

**[EN]** Given $n$ i.i.d. coin flips $X_1, \ldots, X_n \sim \text{Bernoulli}(\theta)$ with $k$ heads observed:

(a) Write the likelihood function $L(\theta)$. State where the i.i.d. assumption is used. (3점)
(b) Write the log-likelihood $\ell(\theta)$. Explain why we take the log. (2점)
(c) Derive $\hat{\theta}_{\text{MLE}}$ by differentiation. At each step, state which differentiation rule you use. (3점)
(d) Verify that this is a maximum (not minimum) by checking the second derivative. (2점)

---

## 풀이

**(a)** 각 $X_i$의 PMF: $p(X_i = x_i | \theta) = \theta^{x_i}(1-\theta)^{1-x_i}$ ($x_i \in \{0, 1\}$).

**i.i.d. 가정을 여기서 사용한다**: 각 동전 던지기가 독립이고 동일한 $\theta$를 가지므로, 전체 데이터의 결합 확률은 개별 확률의 **곱**으로 분해된다:

$$L(\theta) = p(D|\theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^{\sum x_i}(1-\theta)^{n - \sum x_i} = \theta^k(1-\theta)^{n-k}$$

여기서 $k = \sum_{i=1}^n x_i$ (앞면 총 수). **동일 분포(identically distributed) 가정** 덕분에 모든 $X_i$가 같은 $\theta$를 공유한다.

**(b)** 로그를 취하는 이유: (1) 곱이 합으로 바뀌어 미분이 쉬워진다. (2) 수치적으로, 매우 작은 확률들의 곱은 컴퓨터에서 underflow 위험이 있지만 로그 합은 안정적이다. (3) 로그는 단조 증가이므로 $\arg\max$가 보존된다.

$$\ell(\theta) = \log L(\theta) = k\log\theta + (n-k)\log(1-\theta)$$

**(c)** $\theta$에 대해 미분한다. $\log\theta$의 미분은 $1/\theta$ (로그 미분 규칙), $\log(1-\theta)$의 미분은 $-1/(1-\theta)$ (체인룰: 합성함수의 미분, 내부 함수 $(1-\theta)$의 미분이 $-1$):

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} + \frac{-(n-k)}{1-\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta}$$

이것을 0으로 놓는 이유: MLE는 로그 우도를 최대화하는 $\theta$를 찾는 것이고, 미분가능한 함수의 극값에서는 미분이 0이 되기 때문이다 (Fermat의 정리).

$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

양변에 $\theta(1-\theta)$를 곱하면 (교차 곱):
$$k(1-\theta) = (n-k)\theta$$
$$k - k\theta = n\theta - k\theta$$
$$k = n\theta$$

$$\boxed{\hat{\theta}_{\text{MLE}} = \frac{k}{n}}$$

**(d)** 이것이 최대인지 최소인지 확인하기 위해 2차 미분을 구한다:

$$\frac{d^2\ell}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2}$$

$k \geq 0$, $n-k \geq 0$, $\theta^2 > 0$, $(1-\theta)^2 > 0$이므로 두 항 모두 음수.

$$\frac{d^2\ell}{d\theta^2} < 0 \quad \text{(모든 } \theta \in (0,1) \text{에서)}$$

2차 미분이 음수이면 함수가 아래로 볼록(concave)하므로, 1차 미분이 0인 점은 **최대**이다. $\blacksquare$

---

# Problem 6 (10점) — Cross-Entropy Loss의 NLL 유도

**[EN]** For multi-class classification with $C$ classes, the model outputs $\hat{y} = \text{softmax}(z)$ and the true label is $y$ (one-hot vector).

(a) Assume $y | x \sim \text{Categorical}(\hat{y})$. Write the likelihood $p(y|x, \theta)$ for a single sample. (2점)
(b) Take the negative log and show it equals the cross-entropy loss $-\sum_{c=1}^C y_c \log \hat{y}_c$. Explain why this simplifies to $-\log \hat{y}_{c^*}$ where $c^*$ is the true class. (4점)
(c) Show that minimizing cross-entropy is equivalent to minimizing $D_{KL}(p_{\text{data}} \| p_{\text{model}})$. State what is constant and what is being optimized. (4점)

---

## 풀이

**(a)** 카테고리컬 분포에서, 정답이 클래스 $c^*$이면 (즉, $y$가 $c^*$ 위치만 1인 원-핫 벡터):

$$p(y | x, \theta) = \prod_{c=1}^{C} \hat{y}_c^{y_c}$$

이 표현의 의미: $y_c = 1$인 클래스만 $\hat{y}_c^1 = \hat{y}_c$가 남고, $y_c = 0$인 클래스는 $\hat{y}_c^0 = 1$이 되어 사라진다. 결과적으로 $p(y|x,\theta) = \hat{y}_{c^*}$.

**(b)** 음의 로그를 취한다. 이유: MLE에서는 우도를 최대화해야 하는데, 최소화 문제로 바꾸기 위해 음의 로그를 사용한다.

$$-\log p(y|x,\theta) = -\log\prod_{c=1}^{C}\hat{y}_c^{y_c}$$

로그의 곱 → 합 성질을 사용한다:

$$= -\sum_{c=1}^{C} y_c \log \hat{y}_c$$

이것이 **교차 엔트로피(Cross-Entropy) 손실**이다.

원-핫 벡터에서 $y_c = 0$인 항은 $0 \times \log\hat{y}_c = 0$이므로 사라진다. 정답 클래스 $c^*$만 살아남아:

$$-\sum_{c=1}^{C} y_c \log\hat{y}_c = -1 \cdot \log\hat{y}_{c^*} = -\log\hat{y}_{c^*}$$

직관적 의미: 모델이 정답 클래스에 높은 확률을 부여할수록 $\hat{y}_{c^*} \to 1$이므로 $-\log\hat{y}_{c^*} \to 0$ (손실 감소). 낮은 확률이면 $-\log\hat{y}_{c^*} \to \infty$ (큰 페널티).

**(c)** 교차 엔트로피와 KL divergence의 관계:

$$H(p, q) = -\sum_c p_c \log q_c = H(p) + D_{KL}(p \| q)$$

여기서 $p = p_{\text{data}}$ (정답 분포), $q = p_{\text{model}}$ (모델 출력).

$H(p) = -\sum_c p_c \log p_c$는 데이터 분포의 엔트로피로, **$\theta$에 의존하지 않는 상수**이다. 데이터의 정답 분포는 학습으로 바꿀 수 있는 것이 아니기 때문이다.

따라서:

$$\arg\min_\theta H(p_{\text{data}}, p_{\text{model}}) = \arg\min_\theta [H(p_{\text{data}}) + D_{KL}(p_{\text{data}} \| p_{\text{model}})]$$

상수 $H(p_{\text{data}})$를 제거하면:

$$= \arg\min_\theta D_{KL}(p_{\text{data}} \| p_{\text{model}})$$

**교차 엔트로피 최소화 = KL divergence 최소화 = 모델 분포를 데이터 분포에 가깝게 만드는 것**. $\blacksquare$

---

# Problem 7 (10점) — 정규방정식의 유도와 해석

**[EN]** For linear regression $\hat{y} = X\theta$ with MSE loss $L(\theta) = \frac{1}{2n}\|X\theta - y\|^2$:

(a) Expand $\|X\theta - y\|^2$ using matrix algebra. (2점)
(b) Compute $\nabla_\theta L$ step by step, stating which matrix calculus identity you use at each step. (4점)
(c) Set the gradient to zero and derive $\theta^* = (X^\top X)^{-1}X^\top y$. Under what condition does $(X^\top X)^{-1}$ exist? If it doesn't exist, what does the pseudoinverse $X^+$ give? (4점)

---

## 풀이

**(a)** $\|X\theta - y\|^2 = (X\theta - y)^\top(X\theta - y)$. 전개한다:

$$= \theta^\top X^\top X\theta - 2y^\top X\theta + y^\top y$$

전개 과정: $(X\theta - y)^\top(X\theta - y) = (X\theta)^\top(X\theta) - (X\theta)^\top y - y^\top(X\theta) + y^\top y$. 스칼라의 전치는 자기 자신이므로 $(X\theta)^\top y = y^\top X\theta$. 따라서 중간 두 항이 합쳐져 $-2y^\top X\theta$.

**(b)** $L = \frac{1}{2n}(\theta^\top X^\top X\theta - 2y^\top X\theta + y^\top y)$

항별로 $\theta$에 대해 미분한다:

- $\frac{\partial}{\partial\theta}(\theta^\top X^\top X\theta)$: $X^\top X$가 대칭 행렬이므로 ($X^\top X = (X^\top X)^\top$), **대칭 이차형식의 미분 공식** $\frac{\partial}{\partial\theta}(\theta^\top S\theta) = 2S\theta$를 적용한다. 결과: $2X^\top X\theta$.

- $\frac{\partial}{\partial\theta}(2y^\top X\theta)$: 이것은 $\theta$에 대한 **선형 함수**이므로, **선형 함수의 미분 공식** $\frac{\partial}{\partial\theta}(a^\top\theta) = a$를 적용한다. $a = 2X^\top y$이므로 결과: $2X^\top y$.

- $\frac{\partial}{\partial\theta}(y^\top y)$: $\theta$를 포함하지 않는 **상수**이므로 미분 결과는 0.

합산:

$$\nabla_\theta L = \frac{1}{2n}(2X^\top X\theta - 2X^\top y) = \frac{1}{n}(X^\top X\theta - X^\top y) = \frac{1}{n}X^\top(X\theta - y)$$

**(c)** $\nabla_\theta L = 0$으로 놓으면:

$$X^\top X\theta = X^\top y$$

이것이 **정규방정식(Normal Equation)**이다. 양변에 $(X^\top X)^{-1}$을 곱하면:

$$\theta^* = (X^\top X)^{-1}X^\top y$$

**존재 조건**: $(X^\top X)^{-1}$이 존재하려면 $X^\top X$가 가역이어야 한다. $X^\top X$는 $d \times d$ 행렬이고, $\text{rank}(X^\top X) = \text{rank}(X)$이므로, $X$가 **full column rank** ($\text{rank}(X) = d$)일 때 가역이다. 이것은 특성(feature)들이 선형독립일 때 성립한다.

**full column rank가 아닌 경우** (특성 간 완전 상관, 또는 데이터 수 < 특성 수): $(X^\top X)^{-1}$이 존재하지 않는다. 이때 의사역행렬 $X^+ = (X^\top X)^+ X^\top$를 사용하면, $Ax = b$의 해 중 **최소 노름 해(minimum-norm solution)**를 준다. 이것은 "가능한 해 중 가장 단순한(파라미터가 작은) 해"로, 과적합을 줄이는 효과가 있다. 이것이 과매개변수화 모델에서 경사 하강법이 자동으로 찾는 해이기도 하다 (implicit regularization). $\blacksquare$

---

# Problem 8 (10점) — 분류에서 MLE = Cross-Entropy 최소화

**[EN]** Consider binary classification where $p(y=1|x,\theta) = \sigma(w^\top x + b)$ with $\sigma(z) = 1/(1+e^{-z})$.

(a) For a single data point $(x_i, y_i)$ where $y_i \in \{0, 1\}$, write $p(y_i | x_i, \theta)$ as a single expression using $\hat{y}_i = \sigma(w^\top x_i + b)$. (2점)
(b) Write the negative log-likelihood for the full i.i.d. dataset. Show that it equals the binary cross-entropy loss. State where i.i.d. is used. (4점)
(c) Compute $\frac{\partial L}{\partial w}$ and show the remarkable cancellation that gives a clean gradient $\frac{1}{n}\sum_i (\hat{y}_i - y_i)x_i$. Explain why this cancellation is NOT a coincidence but a consequence of using the canonical link function. (4점)

---

## 풀이

**(a)** $y_i = 1$이면 $p = \hat{y}_i$, $y_i = 0$이면 $p = 1 - \hat{y}_i$. 하나의 식으로:

$$p(y_i | x_i, \theta) = \hat{y}_i^{y_i}(1 - \hat{y}_i)^{1-y_i}$$

이것은 베르누이 분포의 PMF를 컴팩트하게 쓴 것이다.

**(b)** **i.i.d. 가정**에 의해 전체 데이터의 우도는 개별 우도의 곱이다:

$$p(D|\theta) = \prod_{i=1}^{n} \hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i}$$

**독립 가정이 여기서 사용**되었다. 각 데이터가 독립이므로 곱으로 분해 가능.

음의 로그를 취하면 (로그의 곱 → 합 성질 사용):

$$\text{NLL} = -\sum_{i=1}^{n}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

$\frac{1}{n}$을 붙이면:

$$L = -\frac{1}{n}\sum_{i=1}^{n}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

이것이 **이진 교차 엔트로피(Binary Cross-Entropy) 손실**이다.

**(c)** 체인룰로 $\frac{\partial L}{\partial w}$를 구한다.

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n} \frac{\partial \ell_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w}$$

각 항을 계산한다:

**[1단계]** $\frac{\partial \ell_i}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$

이것은 $-[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$를 $\hat{y}_i$에 대해 미분한 것이다.

**[2단계]** $\frac{\partial \hat{y}_i}{\partial z_i} = \sigma(z_i)(1-\sigma(z_i)) = \hat{y}_i(1-\hat{y}_i)$

이것은 시그모이드의 미분이다.

**[3단계]** $\frac{\partial z_i}{\partial w} = x_i$

$z_i = w^\top x_i + b$이므로.

**곱하면:**

$$\frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)} \times \hat{y}_i(1-\hat{y}_i) \times x_i$$

분모 $\hat{y}_i(1-\hat{y}_i)$와 분자의 같은 항이 **정확히 소거**된다!

$$= (\hat{y}_i - y_i)x_i$$

따라서:

$$\boxed{\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)x_i}$$

이 소거는 우연이 아니라, 시그모이드가 베르누이 분포의 **정준 연결 함수(canonical link function)**이기 때문이다. 지수 가족(exponential family) 분포와 정준 연결 함수의 조합에서는 NLL의 그래디언트가 항상 $(예측 - 정답) \times 입력$ 형태로 깔끔하게 정리된다. 가우시안의 정준 연결은 항등 함수(identity)이고, 이 경우에도 동일한 구조의 그래디언트가 나온다. $\blacksquare$

---

# Problem 9 (10점) — 가우시안의 최대 엔트로피 성질과 MSE 연결

**[EN]**
(a) State precisely: "The Gaussian distribution maximizes entropy among all distributions with fixed mean and variance." What constraints does this impose? (3점)
(b) Explain the logical chain: CLT → Gaussian noise → Gaussian likelihood → NLL → MSE. At each arrow, state what assumption or mathematical step connects them. (4점)
(c) If the noise were NOT Gaussian (e.g., Laplacian: $p(\epsilon) \propto \exp(-|\epsilon|/b)$), what loss function would NLL minimization produce instead of MSE? Derive it. (3점)

---

## 풀이

**(a)** 정확한 서술: 평균 $\mu$와 분산 $\sigma^2$이 고정된 **모든 연속 확률 분포** 중에서, **미분 엔트로피 $h(X) = -\int p(x)\log p(x)\,dx$를 최대화**하는 분포는 $\mathcal{N}(\mu, \sigma^2)$이다.

제약 조건 3가지:
1. 정규화: $\int p(x)\,dx = 1$ (유효한 확률 분포)
2. 평균 고정: $\int xp(x)\,dx = \mu$
3. 분산 고정: $\int (x-\mu)^2 p(x)\,dx = \sigma^2$

이것이 중요한 이유: 우리가 평균과 분산만 알고 다른 정보가 없을 때, 가장 "편향 없는"(최소한의 추가 가정을 하는) 선택이 가우시안이라는 것이다. **정보 이론적으로 가장 보수적인 가정**.

**(b)** 논리 체인:

```
CLT (중심극한정리)
│  가정: 노이즈가 많은 독립 요인의 합
│  수학: n개 독립 확률변수의 합 → N(μ, σ²)
↓
가우시안 노이즈 가정: ε ~ N(0, σ²)
│  가정: y = f(x) + ε
│  수학: 조건부 분포 y|x ~ N(f(x), σ²)
↓
가우시안 우도: p(y|x,θ) = N(y; f_θ(x), σ²)
│  가정: i.i.d. 데이터
│  수학: 곱 → 로그로 합
↓
NLL = Σ[-log p(yi|xi,θ)]
│  수학: 가우시안 PDF 대입, 로그 전개
│  핵심: -log(exp(-z²)) = z² → 제곱 출현
↓
MSE = (1/n)Σ(yi - f_θ(xi))²
```

각 화살표의 핵심:
- CLT → 가우시안: 많은 독립 요인의 합이라는 **물리적 가정**
- 가우시안 → 우도: 노이즈 분포를 직접 우도에 대입하는 **수학적 전개**
- 우도 → NLL: i.i.d. 가정으로 곱을 합으로 분해하고, 로그를 취하는 **계산 단계**
- NLL → MSE: 가우시안의 $\exp(-z^2)$에 $-\log$를 씌우면 $z^2$이 나오는 **로그-지수 상쇄**

**(c)** 라플라시안 노이즈 $p(\epsilon) \propto \exp(-|\epsilon|/b)$이면:

$$p(y_i|x_i,\theta) = \frac{1}{2b}\exp\left(-\frac{|y_i - f_\theta(x_i)|}{b}\right)$$

NLL:

$$-\log p(y_i|x_i,\theta) = \log(2b) + \frac{|y_i - f_\theta(x_i)|}{b}$$

$\theta$에 무관한 항을 제거하면:

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \sum_{i=1}^{n} |y_i - f_\theta(x_i)|$$

이것은 **MAE (Mean Absolute Error)** = L1 손실이다.

즉: **가우시안 노이즈 → MSE (L2), 라플라시안 노이즈 → MAE (L1)**. 노이즈의 분포 가정이 손실 함수를 결정한다. $\blacksquare$

---

# Problem 10 (10점) — Eigenvalues, PCA, 차원 축소의 논리

**[EN]**
(a) Given a data matrix $X \in \mathbb{R}^{n \times d}$ (centered, i.e., mean-subtracted), the sample covariance is $S = \frac{1}{n}X^\top X$. Explain why $S$ is always PSD. (2점)
(b) PCA finds directions of maximum variance by solving $\max_{\|w\|=1} w^\top S w$. Show that the solution is the eigenvector of $S$ with the largest eigenvalue. Use Lagrange multipliers. (5점)
(c) Explain the connection between PCA and SVD: if $X = U\Sigma V^\top$, what are the principal components and how do singular values relate to eigenvalues of $S$? (3점)

---

## 풀이

**(a)** 임의의 벡터 $v \neq 0$에 대해:

$$v^\top S v = v^\top \left(\frac{1}{n}X^\top X\right) v = \frac{1}{n}(Xv)^\top(Xv) = \frac{1}{n}\|Xv\|^2 \geq 0$$

벡터의 노름 제곱은 항상 0 이상이므로 $S$는 PSD이다. 또한 $S$는 대칭이다: $S^\top = \frac{1}{n}(X^\top X)^\top = \frac{1}{n}X^\top X = S$. 대칭 PSD 행렬의 고유값은 모두 0 이상이다.

**(b)** 라그랑주 승수법을 사용하여 제약 최적화 문제를 푼다.

목적함수: $f(w) = w^\top S w$ (분산, 최대화)
제약조건: $g(w) = w^\top w - 1 = 0$ (단위 벡터)

라그랑지안: $\mathcal{L}(w, \lambda) = w^\top S w - \lambda(w^\top w - 1)$

$w$에 대해 미분하고 0으로 놓는다. 이유: 제약 하 극값의 필요 조건이 $\nabla_w \mathcal{L} = 0$이기 때문이다.

$$\frac{\partial \mathcal{L}}{\partial w} = 2Sw - 2\lambda w = 0$$

$S$가 대칭이므로 $\frac{\partial}{\partial w}(w^\top S w) = 2Sw$ 공식을 사용했다.

정리하면:

$$Sw = \lambda w$$

이것은 **고유값 방정식**이다! $w$는 $S$의 고유벡터, $\lambda$는 대응하는 고유값.

어떤 고유벡터를 선택해야 하는가? 제약 $\|w\| = 1$ 하에서:

$$w^\top S w = w^\top(\lambda w) = \lambda w^\top w = \lambda$$

분산 = 고유값. 따라서 **분산을 최대화하려면 가장 큰 고유값에 대응하는 고유벡터를 선택**해야 한다.

첫 번째 주성분 = 최대 고유값의 고유벡터.
$k$번째 주성분 = $k$번째로 큰 고유값의 고유벡터 (이전 주성분들과 직교하는 제약 추가). $\blacksquare$

**(c)** $X = U\Sigma V^\top$이면:

$$S = \frac{1}{n}X^\top X = \frac{1}{n}V\Sigma^\top U^\top U\Sigma V^\top = \frac{1}{n}V\Sigma^2 V^\top$$

$U^\top U = I$ (직교 행렬)이므로 소거된다.

이것은 $S$의 **고유분해**와 정확히 같은 형태다: $S = V\Lambda V^\top$ 여기서 $\Lambda = \frac{1}{n}\Sigma^2$.

따라서:
- **$V$의 열** = $S$의 고유벡터 = **주성분 방향(principal directions)**
- $S$의 고유값 = $\frac{\sigma_i^2}{n}$ (특이값의 제곱 / 데이터 수)
- 큰 특이값 → 큰 분산 → 중요한 주성분

PCA = $V$의 처음 $k$열만 사용한 차원 축소 = SVD의 truncation. $\blacksquare$

---

# Problem 11 (10점) — 소프트맥스 온도와 모델 확신도

**[EN]** The temperature-scaled softmax is $p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$.

(a) For logits $z = (2, 1, 0)$, compute the softmax output for $\tau = 1$, $\tau = 0.5$, and $\tau = 5$. You may leave answers in terms of $e$. (4점)
(b) Describe what happens in the limits $\tau \to 0$ and $\tau \to \infty$, and prove your claims. (3점)
(c) In knowledge distillation, a teacher model's outputs are "softened" using high $\tau$. Explain why softer distributions carry more information than hard (one-hot) labels, using the concept of entropy. (3점)

---

## 풀이

**(a)** $z = (2, 1, 0)$

**$\tau = 1$ (표준)**:

$p = \frac{(e^2, e^1, e^0)}{e^2 + e + 1} = \frac{(7.389, 2.718, 1)}{11.107} \approx (0.665, 0.245, 0.090)$

**$\tau = 0.5$ (날카로움)**:

$z/\tau = (4, 2, 0)$

$p = \frac{(e^4, e^2, 1)}{e^4 + e^2 + 1} = \frac{(54.60, 7.389, 1)}{62.99} \approx (0.867, 0.117, 0.016)$

**$\tau = 5$ (부드러움)**:

$z/\tau = (0.4, 0.2, 0)$

$p = \frac{(e^{0.4}, e^{0.2}, 1)}{e^{0.4} + e^{0.2} + 1} = \frac{(1.492, 1.221, 1)}{3.713} \approx (0.402, 0.329, 0.269)$

관찰: $\tau$가 작을수록 → 가장 큰 logit에 확률 집중. $\tau$가 클수록 → 균등분포에 가까워짐.

**(b)**

$\tau \to 0$: $z_i / \tau \to \pm\infty$. 가장 큰 $z_i$에 대응하는 $\exp(z_i/\tau)$가 나머지를 지수적으로 지배한다.

증명: $z_1 > z_2 \geq \cdots$ 이면 $p_1 = \frac{1}{1 + \sum_{j \geq 2}\exp((z_j - z_1)/\tau)}$. $\tau \to 0$이면 $(z_j - z_1)/\tau \to -\infty$ ($j \geq 2$), 따라서 $\exp \to 0$, $p_1 \to 1$. 즉 **argmax(one-hot)**으로 수렴.

$\tau \to \infty$: $z_i / \tau \to 0$ (모든 $i$). $\exp(z_i/\tau) \to \exp(0) = 1$. 따라서 $p_i \to \frac{1}{C}$. **균등분포**로 수렴.

**(c)** 엔트로피 $H = -\sum p_i \log p_i$는 분포의 불확실성(정보량)을 측정한다.

- 하드 레이블 (one-hot, $\tau \to 0$): $H = 0$. 정보 = "정답은 이 클래스" — **클래스 간 관계에 대한 정보가 전무**.
- 소프트 레이블 (높은 $\tau$): $H > 0$. "고양이일 확률 60%, 호랑이 30%, 개 10%" — **클래스 간 유사도** 정보를 포함. "고양이와 호랑이가 비슷하다"는 것을 교사 모델이 학생에게 전달할 수 있다.

높은 $\tau$에서의 소프트 분포는 엔트로피가 높아서, 각 클래스에 대한 **상대적 관계 정보**를 더 풍부하게 담는다. 이것이 지식 증류에서 소프트 레이블이 하드 레이블보다 효과적인 이유다. $\blacksquare$

---

# Problem 12 (10점) — Sigmoid 미분과 Vanishing Gradient

**[EN]**
(a) Prove that $\sigma'(z) = \sigma(z)(1-\sigma(z))$ where $\sigma(z) = 1/(1+e^{-z})$. Show every step of the derivation. (4점)
(b) Find the maximum value of $\sigma'(z)$ and at what $z$ it occurs. (3점)
(c) In a deep network with $L$ sigmoid layers, the gradient of the loss w.r.t. the first layer's weights involves a product of $L$ terms of $\sigma'(z_l)$. Show that this product is bounded above by $(1/4)^L$ and explain the practical consequence. (3점)

---

## 풀이

**(a)** $\sigma(z) = (1 + e^{-z})^{-1}$

분수 함수의 미분(연쇄 법칙)을 적용한다. 외부 함수: $u^{-1}$, 내부 함수: $u = 1 + e^{-z}$.

$$\sigma'(z) = -(1+e^{-z})^{-2} \cdot \frac{d}{dz}(1+e^{-z})$$

$\frac{d}{dz}(1+e^{-z}) = -e^{-z}$ (지수함수의 미분 + 체인룰: $\frac{d}{dz}e^{-z} = e^{-z} \cdot (-1)$)

$$\sigma'(z) = -(1+e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$$

이것을 $\sigma(z)$와 $1-\sigma(z)$로 다시 쓴다:

$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad 1 - \sigma(z) = 1 - \frac{1}{1+e^{-z}} = \frac{e^{-z}}{1+e^{-z}}$$

곱하면:

$$\sigma(z)(1-\sigma(z)) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2}$$

이것은 위에서 구한 $\sigma'(z)$와 정확히 같다. $\boxed{\sigma'(z) = \sigma(z)(1-\sigma(z))}$ $\blacksquare$

**(b)** $\sigma'(z) = \sigma(z)(1-\sigma(z))$에서 $p = \sigma(z)$로 놓으면 $\sigma' = p(1-p)$.

이것은 아래로 볼록한 이차함수 $f(p) = p - p^2$이고, $p \in (0, 1)$.

$f'(p) = 1 - 2p = 0$에서 $p = 1/2$. $f(1/2) = 1/4$.

$\sigma(z) = 1/2$가 되는 $z$: $1/(1+e^{-z}) = 1/2$ → $z = 0$.

$$\boxed{\max \sigma'(z) = \frac{1}{4}, \quad z = 0\text{에서 달성}}$$

**(c)** $L$개 시그모이드 층을 거친 그래디언트는 체인룰에 의해:

$$\frac{\partial L}{\partial w_1} \propto \prod_{l=1}^{L} \sigma'(z_l) \cdot (\text{기타 항})$$

각 $\sigma'(z_l) \leq 1/4$ (위에서 증명)이므로:

$$\prod_{l=1}^{L} \sigma'(z_l) \leq \left(\frac{1}{4}\right)^L$$

$L = 10$이면: $(1/4)^{10} \approx 9.5 \times 10^{-7}$ — 거의 0.
$L = 20$이면: $(1/4)^{20} \approx 9.1 \times 10^{-13}$ — 사실상 0.

**실전적 결과**: 층이 깊어질수록 앞쪽 층의 그래디언트가 지수적으로 소멸(vanishing gradient)하여, 앞쪽 가중치가 거의 업데이트되지 않는다. 이것이 깊은 시그모이드 네트워크가 학습 불가능했던 이유이다. ReLU($\max(0,z)$)는 양수 영역에서 미분이 항상 1이므로 이 문제를 해결한다. $\blacksquare$

---

# Problem 13 (8점) — SGD의 비편향성과 배치 크기

**[EN]**
(a) For $L(\theta) = \frac{1}{n}\sum_{i=1}^n \ell_i(\theta)$ and mini-batch $B$ sampled uniformly at random, show that $g_B = \frac{1}{|B|}\sum_{i \in B}\nabla\ell_i(\theta)$ is an unbiased estimator of $\nabla L(\theta)$. State clearly what "unbiased" means and what probability is being taken. (4점)
(b) Show that $\text{Var}(g_B) = \frac{\sigma_g^2}{|B|}$ where $\sigma_g^2$ is the per-sample gradient variance. Explain the practical tradeoff of choosing small vs. large batch sizes. (4점)

---

## 풀이

**(a)** "비편향(unbiased)"이란 추정량의 기대값이 추정하려는 참값과 정확히 같다는 의미이다. 즉 $\mathbb{E}[g_B] = \nabla L(\theta)$.

여기서 기대값은 **미니배치 $B$의 랜덤 선택**에 대해 취한다. $\theta$는 고정.

각 $i \in B$가 $\{1, \ldots, n\}$에서 균등 무작위로 선택되므로:

$$\mathbb{E}[\nabla\ell_i(\theta)] = \frac{1}{n}\sum_{j=1}^{n}\nabla\ell_j(\theta) = \nabla L(\theta)$$

이 단계에서 **동일 분포(identically distributed) 가정**이 사용된다: 모든 인덱스가 동일한 확률 $1/n$으로 선택된다.

기대값의 **선형성**에 의해:

$$\mathbb{E}[g_B] = \mathbb{E}\left[\frac{1}{|B|}\sum_{i \in B}\nabla\ell_i(\theta)\right] = \frac{1}{|B|}\sum_{i \in B}\mathbb{E}[\nabla\ell_i(\theta)] = \frac{1}{|B|} \cdot |B| \cdot \nabla L(\theta) = \nabla L(\theta)$$

따라서 $g_B$는 $\nabla L(\theta)$의 비편향 추정량이다. $\blacksquare$

**(b)** 각 $\nabla\ell_i(\theta)$의 분산이 $\sigma_g^2$이고, 배치 내 샘플이 **독립**이면:

$$\text{Var}(g_B) = \text{Var}\left(\frac{1}{|B|}\sum_{i \in B}\nabla\ell_i\right) = \frac{1}{|B|^2}\sum_{i \in B}\text{Var}(\nabla\ell_i) = \frac{1}{|B|^2} \cdot |B| \cdot \sigma_g^2 = \frac{\sigma_g^2}{|B|}$$

두 번째 등호에서 **독립 가정**을 사용했다: 독립인 확률변수의 합의 분산 = 각 분산의 합.

**실전 트레이드오프:**

| 배치 크기 | 장점 | 단점 |
|----------|------|------|
| 작은 $\|B\|$ | 노이즈가 커서 flat minima 탈출 가능 (일반화 좋음), 메모리 적게 사용 | 그래디언트 추정 불안정, 수렴 느림 |
| 큰 $\|B\|$ | 그래디언트 추정 정확 ($\sigma_g^2/\|B\|$ 감소), 병렬 처리 효율 | sharp minima로 수렴 경향, 메모리 많이 사용, 일반화 나빠질 수 있음 |

CLT와 연결: $|B|$가 크면 $g_B$의 분포가 $\mathcal{N}(\nabla L, \sigma_g^2/|B|)$에 근사하며, 분산이 $1/|B|$로 줄어든다. $\blacksquare$

---

# Problem 14 (10점) — 트레이스와 L2 정규화

**[EN]**
(a) Show that $\|\theta\|^2 = \text{Tr}(\theta\theta^\top) = \theta^\top\theta$. Explain why both expressions are equal. (3점)
(b) The ridge regression objective is $L(\theta) = \|X\theta - y\|^2 + \lambda\|\theta\|^2$. Derive the closed-form solution $\theta^* = (X^\top X + \lambda I)^{-1}X^\top y$. (4점)
(c) Explain why adding $\lambda I$ to $X^\top X$ guarantees invertibility even when $X$ does not have full column rank. Use eigenvalue arguments. (3점)

---

## 풀이

**(a)** $\theta \in \mathbb{R}^d$일 때:

$\theta^\top\theta = \sum_{i=1}^d \theta_i^2 = \|\theta\|^2$ (노름의 정의)

$\theta\theta^\top$은 $d \times d$ 행렬이고, $(\theta\theta^\top)_{ii} = \theta_i^2$이다. 따라서:

$\text{Tr}(\theta\theta^\top) = \sum_{i=1}^d (\theta\theta^\top)_{ii} = \sum_{i=1}^d \theta_i^2 = \|\theta\|^2$

두 표현이 같은 이유: **스칼라 = 자기 자신의 trace**이고, $\theta^\top\theta$가 스칼라이므로 $\theta^\top\theta = \text{Tr}(\theta^\top\theta) = \text{Tr}(\theta\theta^\top)$ (trace의 순환 성질). $\blacksquare$

**(b)** $L(\theta) = (X\theta-y)^\top(X\theta-y) + \lambda\theta^\top\theta$

전개: $= \theta^\top X^\top X\theta - 2y^\top X\theta + y^\top y + \lambda\theta^\top\theta$

$= \theta^\top(X^\top X + \lambda I)\theta - 2y^\top X\theta + y^\top y$

$X^\top X + \lambda I$는 대칭이므로, 이차형식의 미분 공식을 적용:

$$\nabla_\theta L = 2(X^\top X + \lambda I)\theta - 2X^\top y$$

0으로 놓으면:

$$(X^\top X + \lambda I)\theta = X^\top y$$

$$\boxed{\theta^* = (X^\top X + \lambda I)^{-1}X^\top y}$$

**(c)** $X^\top X$는 PSD (고유값 $\geq 0$). $X^\top X$의 고유값을 $\mu_1, \ldots, \mu_d$ ($\mu_i \geq 0$)라 하자.

$X^\top X + \lambda I$의 고유값은 $\mu_i + \lambda$이다. **같은 고유벡터**를 가지며 고유값에 $\lambda$가 더해진다.

$\lambda > 0$이면 $\mu_i + \lambda > 0$ (모든 $i$). 따라서 **모든 고유값이 양수** → $X^\top X + \lambda I$는 PD → **항상 가역**이다.

$X$가 full column rank가 아니면 일부 $\mu_i = 0$이 되어 $X^\top X$가 비가역이지만, $\lambda > 0$을 더하면 해당 고유값이 $\lambda > 0$이 되어 가역성이 복원된다. 이것이 L2 정규화의 수학적 효과: **수치적 안정성 보장 + 과적합 방지**. $\blacksquare$

---

# Problem 15 (10점) — Attention에서 sqrt(d_k)의 역할

**[EN]** In scaled dot-product attention, the score is $a(q,k) = q^\top k / \sqrt{d_k}$.

(a) If $q, k \in \mathbb{R}^{d_k}$ with entries i.i.d. $\sim \mathcal{N}(0, 1)$, compute $\mathbb{E}[q^\top k]$ and $\text{Var}(q^\top k)$. Clearly state which properties of expectation and variance you use, and where independence is needed. (5점)
(b) Explain why large $\text{Var}(q^\top k)$ causes softmax saturation, and how dividing by $\sqrt{d_k}$ fixes this. (3점)
(c) After scaling, what is $\text{Var}(q^\top k / \sqrt{d_k})$? Verify that it equals 1 regardless of $d_k$. (2점)

---

## 풀이

**(a)** $q^\top k = \sum_{l=1}^{d_k} q_l k_l$

**기대값:**

$$\mathbb{E}[q^\top k] = \sum_{l=1}^{d_k} \mathbb{E}[q_l k_l]$$

기대값의 **선형성**을 사용했다 (독립 여부와 무관하게 성립).

$q_l$과 $k_l$은 **독립**이고 각각 평균 0이므로:

$$\mathbb{E}[q_l k_l] = \mathbb{E}[q_l] \cdot \mathbb{E}[k_l] = 0 \cdot 0 = 0$$

이 단계에서 **독립 가정이 사용**되었다. 독립이어야 $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$가 성립한다.

$$\boxed{\mathbb{E}[q^\top k] = 0}$$

**분산:**

$$\text{Var}(q^\top k) = \text{Var}\left(\sum_{l=1}^{d_k} q_l k_l\right)$$

각 $q_l k_l$은 서로 **독립**이다 (다른 인덱스의 $q$, $k$ 성분은 독립이므로). **독립인 확률변수의 합의 분산 = 각 분산의 합**:

$$= \sum_{l=1}^{d_k} \text{Var}(q_l k_l)$$

$\text{Var}(q_l k_l)$을 구한다. $q_l, k_l$이 독립이고 $\mathbb{E}[q_l k_l] = 0$이므로:

$$\text{Var}(q_l k_l) = \mathbb{E}[(q_l k_l)^2] - (\mathbb{E}[q_l k_l])^2 = \mathbb{E}[q_l^2 k_l^2] - 0$$

**독립 가정**에 의해 $\mathbb{E}[q_l^2 k_l^2] = \mathbb{E}[q_l^2]\mathbb{E}[k_l^2] = 1 \cdot 1 = 1$

$$\text{Var}(q^\top k) = \sum_{l=1}^{d_k} 1 = \boxed{d_k}$$

**(b)** $\text{Var}(q^\top k) = d_k$이므로, $d_k = 512$이면 점수값의 표준편차가 $\sqrt{512} \approx 22.6$이 된다. 이렇게 큰 값이 소프트맥스에 입력되면:

$\text{softmax}(22.6, -15.3, 3.1, \ldots) \approx (1.0, 0.0, 0.0, \ldots)$

거의 one-hot이 되어 **포화(saturation)** 상태에 빠진다. 포화 영역에서 소프트맥스의 그래디언트가 거의 0이 되어 학습이 정체된다 (시그모이드의 vanishing gradient와 동일 원리).

$\sqrt{d_k}$로 나누면 점수의 분산이 1로 정규화되어, 소프트맥스가 적절한 범위에서 작동한다.

**(c)** $\text{Var}(q^\top k / \sqrt{d_k}) = \frac{1}{d_k}\text{Var}(q^\top k) = \frac{d_k}{d_k} = \boxed{1}$

$d_k$에 무관하게 분산이 항상 1. 차원이 아무리 커져도 소프트맥스 입력의 스케일이 일정하게 유지된다. $\blacksquare$

---

# 채점 기준 요약

| 문제 | 배점 | 핵심 평가 포인트 |
|------|------|----------------|
| P1 | 15 | MAP→NLL→MSE 전체 흐름. 각 단계의 가정(i.i.d., 가우시안, 상수 제거)을 명시했는가 |
| P2 | 12 | MAP 직접 풀기. 미분→0 놓기의 이유, prior 영향 해석 |
| P3 | 10 | CLT→가우시안→MSE 인과관계. 각 화살표의 가정 명시 |
| P4 | 10 | Inductive bias 정의, 데이터 양과 아키텍처 선택의 논리 |
| P5 | 10 | 베르누이 MLE. i.i.d. 사용 지점, 미분 규칙 명시, 2차 미분 최대 확인 |
| P6 | 10 | CE = NLL. 카테고리컬→로그→원핫 소거→KL 동치 |
| P7 | 10 | 정규방정식 유도. 행렬 미분 공식 사용 근거, 가역성 조건 |
| P8 | 10 | 이진 CE 그래디언트. 시그모이드 소거 현상과 canonical link 연결 |
| P9 | 10 | 가우시안 최대 엔트로피. CLT 체인. 라플라시안→MAE |
| P10 | 10 | PCA = 공분산의 고유벡터. 라그랑주로 유도. SVD 연결 |
| P11 | 10 | 소프트맥스 온도. 극한 증명. 지식 증류와 엔트로피 |
| P12 | 10 | 시그모이드 미분 완전 유도. 최대 1/4. (1/4)^L vanishing |
| P13 | 8 | SGD 비편향 증명. 독립 가정 사용 지점. 배치 크기 트레이드오프 |
| P14 | 10 | 릿지 회귀 해. λI가 가역성 보장하는 고유값 논증 |
| P15 | 10 | sqrt(d_k) 유도. 독립 가정 명시. 소프트맥스 포화 연결 |
| **총점** | **155점** | |

> **가장 중요한 채점 기준**: 답이 맞아도 논리 과정이 없으면 0점. 답이 틀려도 논리 과정이 올바르면 부분 점수.

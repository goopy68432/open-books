---
title: "Deep Learning Theory — Midterm Exam (Comprehensive)"
slug: midterm-exam-07-week7-predicted
order: 8
---

# Deep Learning Theory — Midterm Exam (Comprehensive)

**한양대학교 대학원 딥러닝 이론 과목**
**시험 시간: 180분 | 총점: 200점 (20문제 × 10점)**
**작성일: 2026-03-24**

> **시험 유의사항:**
> - 모든 유도 과정에서 가정(i.i.d., Gaussian 등)을 명시적으로 서술할 것
> - "왜 이 단계를 수행하는가"를 반드시 설명할 것
> - 수식만 나열하지 말고, 각 줄마다 논리적 근거를 제시할 것

---

# Part I: Basic Concept Questions (서술형, Problems 1–5)

---

## Problem 1. (10pts)

**Question (English):**
Define Prior, Likelihood, Posterior, and Evidence in the context of Bayesian probability. Write Bayes' theorem and explain each term's role. Then compare the Frequentist and Bayesian interpretations of probability using a coin-flip example: specifically, explain what "the probability of heads is 0.7" means under each interpretation.

**출제 의도:**
베이지안 확률론은 이 과목의 모든 이론적 토대이다. Prior → Likelihood → Posterior의 흐름을 정확히 이해하지 못하면 MAP, MLE, 정규화의 연결 고리를 파악할 수 없다. 또한 Frequentist vs Bayesian 해석의 차이를 명확히 구분하는 것은 "왜 우리가 prior를 도입하는가"를 이해하는 핵심이다.

**정답 작성의 핵심 포인트:**
- Bayes' theorem의 4개 항(Prior, Likelihood, Posterior, Evidence)을 각각 수식과 함께 정의
- Evidence가 normalization constant임을 명시
- Frequentist: 확률은 반복 실험의 빈도 극한 → θ는 고정된 상수
- Bayesian: 확률은 믿음(belief)의 정도 → θ는 불확실성을 가진 확률변수
- 동전 예시에서 두 해석의 차이를 구체적으로 대조

**모범 풀이:**

베이즈 정리(Bayes' Theorem)는 다음과 같다:

$$P(\theta | D) = \frac{P(D | \theta) \, P(\theta)}{P(D)}$$

각 항의 정의와 역할:

1. **Prior (사전 확률)** $P(\theta)$:
   데이터를 관측하기 전에 파라미터 $\theta$에 대해 가지고 있는 사전 믿음(belief)이다. 예를 들어, 동전이 공정하다고 믿으면 $P(\theta = 0.5)$에 높은 확률을 부여한다. Prior는 우리의 사전 지식 또는 가정을 수학적으로 인코딩하는 역할을 한다.

2. **Likelihood (우도)** $P(D | \theta)$:
   파라미터 $\theta$가 주어졌을 때, 관측된 데이터 $D$가 나올 확률이다. "이 파라미터 값이 데이터를 얼마나 잘 설명하는가"를 측정한다. 주의: Likelihood는 $\theta$의 함수이지, $D$의 확률분포가 아니다.

3. **Posterior (사후 확률)** $P(\theta | D)$:
   데이터를 관측한 후 업데이트된 $\theta$에 대한 믿음이다. Bayes' theorem의 핵심 산출물로서, prior와 likelihood를 결합하여 얻는다. 이것이 우리가 궁극적으로 원하는 것이다.

4. **Evidence (증거)** $P(D)$:
   $$P(D) = \int P(D|\theta) P(\theta) \, d\theta$$
   모든 가능한 $\theta$ 값에 대해 likelihood와 prior를 적분한 값이다. 이 항은 posterior가 합법적인 확률분포(적분하면 1)가 되도록 하는 **정규화 상수(normalization constant)** 역할을 한다. MAP나 MLE에서는 $\theta$에 대해 최적화하므로, $\theta$에 의존하지 않는 이 항을 무시할 수 있다.

**Frequentist vs Bayesian 해석 비교 (동전 예시):**

"동전의 앞면 확률이 0.7이다"라는 문장의 해석:

| 관점 | 해석 |
|------|------|
| **Frequentist** | $\theta = 0.7$은 고정된 상수이다. 이 동전을 무한히 던지면 앞면의 비율이 0.7에 수렴한다. 확률은 반복 실험의 빈도 극한(limiting frequency)이다. $\theta$는 확률변수가 아니므로 $P(\theta)$라는 개념 자체가 의미 없다. |
| **Bayesian** | $\theta$는 확률변수이다. 0.7은 "내가 관측한 데이터와 사전 지식을 종합했을 때, $\theta = 0.7$에 대한 나의 믿음의 정도"이다. 새로운 데이터가 관측되면 이 믿음을 업데이트한다. |

핵심적 차이: Frequentist에서는 $\theta$가 고정값이므로 $P(\theta)$(prior)를 사용하지 않고, 오직 likelihood $P(D|\theta)$만으로 추정한다 → **MLE**. Bayesian에서는 $\theta$가 확률변수이므로 $P(\theta)$(prior)를 도입하여 posterior를 계산한다 → **MAP** 또는 Full Bayesian Inference.

**자주 하는 실수:**
- Likelihood $P(D|\theta)$를 "$\theta$의 확률"로 오해하는 것. Likelihood는 $D$가 고정된 상태에서 $\theta$의 함수이다.
- Evidence $P(D)$가 최적화에서 무시 가능하다는 점을 설명하지 않는 것
- Frequentist에서도 "확률"을 사용하지만, 그 확률의 **대상이 데이터**이지 파라미터가 아니라는 점을 놓치는 것

**채점 기준:**
- Bayes' theorem 정확한 수식 작성: 2pts
- 4개 항(Prior, Likelihood, Posterior, Evidence) 각각의 정의와 역할: 4pts (각 1pt)
- Frequentist 해석 정확한 서술: 1.5pts
- Bayesian 해석 정확한 서술: 1.5pts
- 동전 예시를 통한 구체적 대조: 1pt

---

## Problem 2. (10pts)

**Question (English):**
Explain inductive reasoning and contrast it with deductive reasoning. Why is machine learning fundamentally an inductive process? Connect this to the concept of "learning from data." Finally, explain the meaning and implications of George Box's quote: "All models are wrong, but some are useful."

**출제 의도:**
머신러닝이 왜 귀납적(inductive) 과정인지 이해하는 것은 ML의 철학적 토대이다. 유한한 데이터로부터 일반적인 규칙을 추론하는 것의 본질적 한계와 가능성을 동시에 이해해야 한다. 이것은 overfitting, generalization, inductive bias 등 핵심 개념으로 직결된다.

**정답 작성의 핵심 포인트:**
- Deduction: 일반 → 특수 (보장된 결론), Induction: 특수 → 일반 (보장되지 않는 결론)
- ML은 유한한 관측 데이터(특수)에서 일반 법칙(모델)을 추론 → 귀납적
- "Learning from data" = 데이터에서 패턴을 추출하여 미래를 예측
- "All models are wrong": 모델은 현실의 근사일 뿐, 완벽한 표현이 아님
- "But some are useful": 완벽하지 않아도 예측에 유용할 수 있음 → 실용주의적 관점

**모범 풀이:**

**1. 연역적 추론 (Deductive Reasoning):**

일반적인 전제에서 특수한 결론을 도출하는 추론이다. 전제가 참이면 결론이 반드시 참이다.

- 전제: "모든 포유류는 폐로 숨을 쉰다" + "고래는 포유류이다"
- 결론: "고래는 폐로 숨을 쉰다" (100% 보장)

수학적 증명이 대표적인 연역적 추론이다.

**2. 귀납적 추론 (Inductive Reasoning):**

개별적인 관측(특수)에서 일반적인 규칙을 추론한다. 결론이 보장되지 않는다.

- 관측: "내가 본 백조 1000마리는 모두 흰색이었다"
- 추론: "모든 백조는 흰색이다" (보장되지 않음 — 검은 백조 존재)

**3. 머신러닝은 근본적으로 귀납적 과정이다:**

ML의 과정을 분석하면:
- **입력**: 유한한 훈련 데이터 $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ (개별 관측)
- **목표**: 모든 가능한 입력에 대해 잘 작동하는 함수 $f: X \to Y$ (일반 법칙)
- **방법**: 데이터에서 패턴을 추출하여 $f$를 학습

이것은 정확히 "특수 → 일반"의 귀납적 구조이다.

핵심적 한계: 훈련 데이터에 없는 입력에 대한 예측은 **보장되지 않는다**. 이것이 바로 generalization의 문제이며, 이 한계를 극복하기 위해 우리는 **inductive bias**(가정)를 도입한다.

**4. "Learning from Data"와의 연결:**

"데이터로부터 학습한다"는 것은:
$$\text{유한한 } D = \{(x_i, y_i)\}_{i=1}^{n} \xrightarrow{\text{학습 알고리즘}} \hat{f} \approx f_{\text{true}}$$

여기서 $\hat{f}$는 $f_{\text{true}}$의 근사이지, 완벽한 복원이 아니다. 유한한 데이터에서 무한한 입력 공간에 대한 규칙을 추론하므로 본질적으로 불완전하다.

**5. "All models are wrong, but some are useful" (George Box):**

- **"All models are wrong"**: 어떤 모델도 현실을 완벽하게 표현하지 못한다. 선형 모델은 비선형 관계를 놓치고, 신경망도 무한히 복잡한 현실의 근사일 뿐이다. 이것은 귀납적 추론의 본질적 한계에서 비롯된다.

- **"But some are useful"**: 완벽하지 않더라도 실용적으로 충분히 좋은 예측을 할 수 있다. $y = \beta_0 + \beta_1 x$가 현실을 완벽히 설명하지 못하더라도, 키와 몸무게의 관계를 예측하는 데는 유용하다.

- **ML에 대한 함의**: 우리의 목표는 "진짜 함수를 찾는 것"이 아니라, "충분히 유용한 근사를 찾는 것"이다. 이것이 ERM(Empirical Risk Minimization)의 철학적 근거이다.

**자주 하는 실수:**
- 귀납과 연역을 뒤바꾸는 것 (귀납 = 일반→특수로 잘못 기술)
- "All models are wrong"를 단순히 "모델이 부정확하다"로만 쓰고, 귀납적 추론의 본질적 한계와 연결하지 않는 것

**채점 기준:**
- 연역/귀납 정확한 정의 및 예시: 2pts
- ML이 귀납적인 이유 (유한 데이터 → 일반 규칙): 3pts
- "Learning from data"와의 연결: 2pts
- Box 인용구 해석 및 ML과의 연결: 3pts

---

## Problem 3. (10pts)

**Question (English):**
Define the i.i.d. (independent and identically distributed) assumption. Separately explain:
(a) What "independent" enables mathematically (joint → product),
(b) What "identically distributed" enables (shared parameter θ),
(c) Give a concrete real-world example where the i.i.d. assumption fails, and explain which part (independence, identical distribution, or both) is violated and why.

**출제 의도:**
i.i.d. 가정은 MLE/MAP 유도의 핵심 전제이다. 이 가정이 없으면 log-likelihood를 합으로 분해할 수 없고, 하나의 θ로 전체 데이터를 모델링할 수 없다. 학생들이 i.i.d.를 단순히 "독립이고 같은 분포"로 외우는 것을 넘어, 각 조건이 수학적으로 무엇을 가능하게 하는지 정확히 이해해야 한다.

**정답 작성의 핵심 포인트:**
- i.i.d.의 정확한 정의: 각 데이터 포인트가 서로 독립이고, 동일한 확률분포에서 추출됨
- Independent: $P(x_1, \ldots, x_n | \theta) = \prod_{i=1}^n P(x_i | \theta)$ 분해 가능
- Identically distributed: 모든 $x_i$가 동일한 파라미터 $\theta$를 공유
- 두 조건이 결합되어야 log-likelihood가 sum으로 분해됨
- 반례에서 어떤 조건이 위반되는지 구체적으로 명시

**모범 풀이:**

**정의:**
확률변수 $X_1, X_2, \ldots, X_n$이 i.i.d.라 함은:
1. **Independent (독립)**: 임의의 $i \neq j$에 대해 $X_i$와 $X_j$가 통계적으로 독립
2. **Identically Distributed (동일 분포)**: 모든 $X_i$가 동일한 확률분포 $P(X|\theta)$를 따름

**(a) "Independent"가 가능하게 하는 것:**

독립 가정에 의해, 결합확률(joint probability)을 각 개별 확률의 곱(product)으로 분해할 수 있다:

$$P(x_1, x_2, \ldots, x_n | \theta) = \prod_{i=1}^{n} P(x_i | \theta)$$

**왜 이것이 중요한가?** log를 취하면 곱이 합으로 변환된다:

$$\log P(x_1, \ldots, x_n | \theta) = \sum_{i=1}^{n} \log P(x_i | \theta)$$

이 분해가 없으면, $n$개 데이터의 결합 분포를 직접 다뤄야 하므로 계산이 사실상 불가능하다. 독립 가정 덕분에 MLE/MAP에서 log-likelihood를 각 데이터 포인트별 기여의 합으로 쓸 수 있다.

**(b) "Identically Distributed"가 가능하게 하는 것:**

동일 분포 가정에 의해, 모든 데이터 포인트가 **같은 파라미터 $\theta$**를 공유한다:

$$X_i \sim P(X | \theta), \quad \forall i = 1, \ldots, n$$

**왜 이것이 중요한가?** 만약 각 $X_i$가 서로 다른 분포 $P(X|\theta_i)$를 따른다면, $n$개의 서로 다른 파라미터 $\theta_1, \ldots, \theta_n$을 추정해야 한다. 이 경우 각 파라미터에 대해 데이터가 1개뿐이므로 의미 있는 추정이 불가능하다. 동일 분포 가정 덕분에 하나의 $\theta$에 $n$개의 데이터를 모두 사용할 수 있어, $n$이 클수록 추정이 정확해진다.

**(c) i.i.d. 가정이 실패하는 예시:**

**예시: 주식 가격의 일일 수익률 (Stock Daily Returns)**

- **독립성 위반**: 오늘의 주가 변동은 어제의 변동에 영향을 받는다 (자기상관, autocorrelation). 주가가 급락한 다음 날 또 급락할 확률이 높다. 따라서 $P(x_t | x_{t-1}) \neq P(x_t)$이므로 독립이 아니다.

- **동일 분포 위반**: 시장의 변동성(volatility)은 시간에 따라 변한다. 금융 위기 시기의 수익률 분포와 안정기의 수익률 분포는 같지 않다. 따라서 $P(X_t | \theta_t)$에서 $\theta_t$가 시간에 따라 변하므로 동일 분포가 아니다.

이 예시에서는 독립성과 동일 분포 **모두** 위반된다. 시계열 데이터는 일반적으로 i.i.d.가 아니며, 이를 다루기 위해 RNN, LSTM, Transformer 등 시계열 모델이 필요하다.

**자주 하는 실수:**
- Independent와 Identically Distributed의 수학적 역할을 구분하지 않고 뭉뚱그려 설명하는 것
- 반례에서 "어떤 조건이 위반되는지" 명시하지 않는 것
- "joint → product" 분해에서 독립 가정이 사용된다는 점을 간과하는 것

**채점 기준:**
- i.i.d. 정확한 정의: 1pt
- Independent의 수학적 역할 (joint → product → log sum): 3pts
- Identically Distributed의 수학적 역할 (shared θ): 3pts
- 반례 제시 및 어떤 조건이 위반되는지 명시: 3pts

---

## Problem 4. (10pts)

**Question (English):**
The course slides present two supervised learning frameworks side by side:
- **Classification**: $p(y|x,h) = h(x)_y$ (Categorical distribution), Loss = Cross-Entropy
- **Regression**: $p(y|x,h) \propto \exp(-c(y - h(x))^2)$ (Gaussian distribution), Loss = MSE

Explain the unifying principle — Negative Log-Likelihood (NLL) — that connects both. Show explicitly how NLL reduces to CE for classification and MSE for regression. Then explain why we do NOT use MSE for classification tasks.

**출제 의도:**
교수님이 가장 강조하는 핵심 주제 중 하나이다. Classification과 Regression이 겉으로는 다른 손실 함수를 사용하는 것처럼 보이지만, NLL이라는 하나의 원리에서 통합된다는 것을 이해해야 한다. 또한 "왜 MSE를 classification에 쓰지 않는가?"는 실용적으로도 중요한 질문이다.

**정답 작성의 핵심 포인트:**
- NLL: $\mathcal{L} = -\log P(y|x,\theta)$가 통합 원리
- Categorical → NLL → CE 유도
- Gaussian → NLL → MSE 유도
- MSE for classification이 나쁜 이유: gradient 문제 (σ' 항이 남아서 vanishing gradient)

**모범 풀이:**

**통합 원리: Negative Log-Likelihood (NLL)**

확률적 관점에서, 모델이 데이터를 잘 설명하려면 $P(y|x,\theta)$가 커야 한다. 이를 최대화하는 것이 Maximum Likelihood Estimation이다. 최적화의 편의를 위해 (왜 log를 취하는가? → 곱을 합으로 변환하고, 지수함수를 제거하며, 수치적 안정성을 확보하기 위해) log를 취하고 부호를 뒤집어 최소화 문제로 변환한다:

$$\hat{\theta} = \arg\min_\theta \left[ -\log P(y|x,\theta) \right]$$

**1. Classification → Cross-Entropy:**

Categorical 분포를 가정한다. 모델 출력 $\hat{y} = h(x)$가 softmax를 거쳐 각 클래스의 확률을 나타내면:

$$P(y = c | x, \theta) = \hat{y}_c$$

여기서 $y$를 one-hot 벡터 $[y_1, \ldots, y_K]$로 표현하면 ($y_c = 1$이고 나머지는 0):

$$P(y|x,\theta) = \prod_{k=1}^{K} \hat{y}_k^{y_k}$$

NLL을 계산한다:

$$-\log P(y|x,\theta) = -\log \prod_{k=1}^{K} \hat{y}_k^{y_k} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$

이것이 바로 **Cross-Entropy Loss**이다:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=1}^{K} y_k \log \hat{y}_k$$

**2. Regression → MSE:**

Gaussian 분포를 가정한다. $y_i = f_\theta(x_i) + \epsilon_i$에서 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$이면:

$$P(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

NLL을 계산한다:

$$-\log P(y_i | x_i, \theta) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}$$

$\theta$에 대해 최적화할 때, 첫 번째 항은 $\theta$에 무관한 상수이므로:

$$\arg\min_\theta \left[-\log P\right] = \arg\min_\theta \sum_{i=1}^{n} (y_i - f_\theta(x_i))^2$$

이것이 바로 **MSE Loss**이다.

**3. 왜 MSE를 Classification에 사용하지 않는가?**

Binary classification에서 $\hat{y} = \sigma(z)$ (sigmoid)이고 MSE를 사용한다고 하자:

$$\mathcal{L}_{\text{MSE}} = (y - \sigma(z))^2$$

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = 2(y - \sigma(z)) \cdot (-\sigma'(z)) = -2(y - \sigma(z)) \cdot \sigma(z)(1 - \sigma(z))$$

여기서 $\sigma'(z) = \sigma(z)(1-\sigma(z))$ 항이 남는다.

문제: $\sigma(z) \approx 0$ 또는 $\sigma(z) \approx 1$일 때 $\sigma'(z) \approx 0$이 되어 gradient가 소실된다. 즉, **모델의 예측이 매우 틀렸을 때조차** gradient가 작아서 학습이 느리다.

반면 CE loss를 사용하면:

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = \hat{y} - y$$

$\sigma'$가 깔끔하게 소거되어, 예측이 틀릴수록 gradient가 커진다. 이것이 CE loss가 classification에서 MSE보다 훨씬 우수한 이유이다.

**자주 하는 실수:**
- NLL 유도 과정에서 "왜 log를 취하는가"를 설명하지 않는 것
- MSE for classification의 문제를 "gradient가 작다"로만 쓰고, σ' 항의 소거/잔존을 수식으로 보이지 않는 것
- Categorical → CE 유도에서 one-hot 표현을 사용하지 않아 수식이 불명확한 것

**채점 기준:**
- NLL을 통합 원리로 서술: 2pts
- Categorical → NLL → CE 유도: 3pts
- Gaussian → NLL → MSE 유도: 3pts
- MSE for classification의 문제점 (σ' gradient 분석): 2pts

---

## Problem 5. (10pts)

**Question (English):**
Define inductive bias. For each of the following architectures — {Linear model, CNN, Transformer} — describe its inductive bias explicitly. Explain how the size of available training data affects architecture choice. Finally, connect the concept of inductive bias to the prior $p(H)$ in the MAP framework.

**출제 의도:**
Inductive bias는 "모델이 사전에 가정하는 것"이며, 이것이 MAP의 prior와 개념적으로 동일하다는 연결은 교수님이 반복적으로 강조하는 핵심이다. 데이터 크기에 따라 적절한 bias의 강도가 달라진다는 직관도 중요하다.

**정답 작성의 핵심 포인트:**
- Inductive bias: 학습 알고리즘이 미관측 데이터에 대해 일반화하기 위해 도입하는 가정의 집합
- Linear: 입출력이 선형 관계 (가장 강한 bias, 가장 적은 파라미터)
- CNN: locality (지역 패턴 중요), translation equivariance (위치 불변)
- Transformer: 약한 bias, attention으로 모든 관계 학습 가능 (가장 많은 파라미터)
- 적은 데이터 → 강한 bias 필요, 많은 데이터 → 약한 bias 허용
- Inductive bias ↔ prior p(H)

**모범 풀이:**

**Inductive Bias의 정의:**

Inductive bias란, 학습 알고리즘이 훈련 데이터에서 보지 못한 새로운 데이터에 대해 일반화(generalize)하기 위해 사전에 도입하는 **가정(assumption)의 집합**이다. "No Free Lunch Theorem"에 의해, 어떤 가정도 없이는 일반화가 불가능하다.

**아키텍처별 Inductive Bias:**

| 아키텍처 | Inductive Bias | 파라미터 수 | Bias 강도 |
|----------|---------------|------------|----------|
| **Linear Model** | 입력과 출력이 **선형 관계** $y = w^Tx + b$. 이것은 매우 강한 가정으로, 비선형 패턴을 전혀 포착할 수 없다. | 가장 적음 | 가장 강함 |
| **CNN** | (1) **Locality**: 인접한 픽셀끼리의 관계가 중요하다 (커널의 지역 수용 영역). (2) **Translation equivariance**: 패턴은 이미지 내 위치에 무관하다 (weight sharing). (3) **Hierarchical structure**: 저수준 특징 → 고수준 특징으로 계층적 조합. | 중간 | 중간 |
| **Transformer** | 상대적으로 약한 bias. Self-attention은 시퀀스 내 **모든 위치 간의 관계**를 학습할 수 있다. 사전에 지역성이나 순서를 가정하지 않는다 (positional encoding을 별도로 추가해야 순서 정보를 알 수 있을 정도). | 가장 많음 | 가장 약함 |

**데이터 크기와 아키텍처 선택:**

$$\text{데이터 적음} \xrightarrow{\text{필요}} \text{강한 inductive bias (e.g., Linear, CNN)}$$
$$\text{데이터 많음} \xrightarrow{\text{허용}} \text{약한 inductive bias (e.g., Transformer)}$$

**근거**:
- 데이터가 적으면, 모델이 데이터만으로 올바른 패턴을 찾기 어렵다. 강한 사전 가정(bias)이 탐색 공간을 좁혀주어 올바른 방향으로 학습을 유도한다.
- 데이터가 많으면, 데이터 자체가 충분한 정보를 제공하므로 강한 가정이 오히려 모델의 표현력을 제한한다. 약한 bias를 가진 모델이 데이터의 복잡한 패턴을 더 잘 포착한다.

예시: 100장의 이미지 → CNN이 적합 (locality, equivariance가 도움). 1억 장의 이미지 → ViT(Transformer)가 더 나은 성능 (충분한 데이터로 유연한 패턴 학습).

**Inductive Bias와 MAP의 Prior $p(H)$ 연결:**

MAP에서:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[ \log P(D|\theta) + \log P(\theta) \right]$$

$P(\theta)$는 파라미터에 대한 사전 믿음이다. Inductive bias는 이 prior의 **구조적 구현**이다:

- **강한 inductive bias** (Linear model) = **강한 prior**: 파라미터 공간을 크게 제한. "입출력이 선형이다"라는 매우 집중된 prior.
- **약한 inductive bias** (Transformer) = **약한 prior** (넓은 분포): 파라미터 공간을 거의 제한하지 않음. Uniform에 가까운 prior.
- **데이터가 적을 때 강한 prior가 유리**: posterior가 prior에 크게 의존하므로, 좋은 prior가 도움이 된다.
- **데이터가 많을 때 prior의 영향이 줄어듦**: likelihood가 지배적이 되어 prior에 덜 의존하므로, 약한 prior도 괜찮다.

이것은 정확히 MAP에서 $n \to \infty$일 때 $\hat{\theta}_{\text{MAP}} \to \hat{\theta}_{\text{MLE}}$이 되는 현상과 일치한다.

**자주 하는 실수:**
- CNN의 inductive bias를 "이미지를 잘 처리한다"로만 쓰고, locality/translation equivariance를 명시하지 않는 것
- Transformer의 bias가 "없다"고 쓰는 것 (약할 뿐, 없지는 않다)
- Inductive bias와 prior의 연결을 개념적으로만 쓰고, MAP 수식과 연결하지 않는 것

**채점 기준:**
- Inductive bias 정의: 1pt
- 3개 아키텍처의 bias 각각 설명: 3pts (각 1pt)
- 데이터 크기와 아키텍처 선택 관계: 3pts
- Inductive bias ↔ MAP prior 연결: 3pts

---

# Part II: Core Derivation Questions (수식 유도, Problems 6–10)

---

## Problem 6. (10pts)

**Question (English):**
Given $n$ i.i.d. coin flips with $k$ heads, derive $\hat{\theta}_{\text{MLE}} = k/n$ from scratch. Show the log-likelihood, differentiation, and setting to zero. State explicitly where the i.i.d. assumption is used. Verify that your solution is a maximum (not minimum) using the second derivative test.

**출제 의도:**
MLE 유도는 이 과목에서 가장 기본적인 수식 전개이다. 단순히 "미분해서 0으로 놓으면 k/n이 나온다"가 아니라, 각 단계에서 어떤 가정이 사용되고 왜 그 단계를 수행하는지 설명할 수 있어야 한다. 2차 도함수 검증까지 포함해야 완전한 유도이다.

**정답 작성의 핵심 포인트:**
- Bernoulli 분포 모델 설정
- i.i.d. 가정 → joint → product 분해 (어디서 독립이 사용되는지 명시)
- log 변환의 이유 (곱→합, 수치 안정성, 단조함수)
- 미분 → 0으로 놓는 이유 (극값의 필요조건)
- 2차 도함수 < 0 → 극대 확인

**모범 풀이:**

**Step 1: 모델 설정**

각 동전 던지기의 결과를 $X_i \in \{0, 1\}$로 표현한다. $X_i = 1$이면 앞면(head), $X_i = 0$이면 뒷면(tail)이다.

각 $X_i$는 Bernoulli 분포를 따른다고 가정한다:
$$P(X_i = x_i | \theta) = \theta^{x_i}(1-\theta)^{1-x_i}, \quad x_i \in \{0, 1\}$$

여기서 $\theta \in [0, 1]$은 앞면이 나올 확률이다.

**Step 2: Likelihood 작성 (★ i.i.d. 가정 사용)**

$n$개의 관측 $x_1, \ldots, x_n$에 대한 likelihood를 작성한다.

**[독립 가정 사용]**: $X_1, \ldots, X_n$이 **독립**이므로 결합 확률을 개별 확률의 곱으로 분해할 수 있다:

$$L(\theta) = P(x_1, \ldots, x_n | \theta) = \prod_{i=1}^{n} P(x_i | \theta)$$

**[동일 분포 가정 사용]**: 모든 $X_i$가 **동일한 분포**를 따르므로 각 항에서 **같은 파라미터 $\theta$**를 사용한다:

$$L(\theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i}$$

지수 법칙으로 정리하면 ($k = \sum_{i=1}^n x_i$는 앞면의 총 횟수):

$$L(\theta) = \theta^k (1-\theta)^{n-k}$$

**Step 3: Log-Likelihood 변환**

**왜 log를 취하는가?**
1. 곱(product)을 합(sum)으로 변환하여 미분이 용이하다.
2. $\log$는 단조증가함수이므로 $\arg\max L(\theta) = \arg\max \log L(\theta)$이다.
3. 매우 작은 확률들의 곱에서 발생하는 수치적 언더플로우를 방지한다.

$$\ell(\theta) = \log L(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

**Step 4: 미분하여 0으로 놓기**

**왜 미분해서 0으로 놓는가?** 최대값(또는 최소값)의 필요조건은 도함수가 0인 것이다 (정상점, stationary point).

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta}$$

$\frac{d\ell}{d\theta} = 0$으로 놓으면:

$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

양변에 $\theta(1-\theta)$를 곱한다:

$$k(1-\theta) = (n-k)\theta$$

$$k - k\theta = n\theta - k\theta$$

$$k = n\theta$$

$$\boxed{\hat{\theta}_{\text{MLE}} = \frac{k}{n}}$$

이것은 직관적으로도 맞다: 앞면이 나온 비율이 최대우도추정량이다.

**Step 5: 2차 도함수 검증 (극대 확인)**

**왜 2차 도함수를 확인하는가?** 1차 도함수 = 0은 극대, 극소, 변곡점 모두에서 성립한다. 극대임을 보장하려면 2차 도함수가 음수여야 한다.

$$\frac{d^2\ell}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2}$$

$0 < \theta < 1$이고 $0 < k < n$일 때, 두 항 모두 양수이므로:

$$\frac{d^2\ell}{d\theta^2} < 0 \quad \text{(음의 정부호)}$$

따라서 $\hat{\theta}_{\text{MLE}} = k/n$은 log-likelihood의 **극대점(maximum)**이다. ∎

**자주 하는 실수:**
- i.i.d.의 "independent" 부분과 "identically distributed" 부분이 각각 어디서 사용되는지 구분하지 않는 것
- 왜 log를 취하는지 설명 없이 바로 log-likelihood를 쓰는 것
- 2차 도함수 검증을 생략하는 것 (극대인지 극소인지 확인하지 않음)

**채점 기준:**
- Bernoulli 모델 설정 및 likelihood 작성: 2pts
- i.i.d. 가정이 사용되는 지점 명시 (독립 → product, 동일분포 → shared θ): 2pts
- Log 변환 이유 + log-likelihood 수식: 2pts
- 미분, 0으로 놓기, θ_MLE = k/n 유도: 2pts
- 2차 도함수로 극대 확인: 2pts

---

## Problem 7. (10pts)

**Question (English):**
Given $x_1, \ldots, x_n$ i.i.d. from $\mathcal{N}(\mu, \sigma^2)$. Derive $\hat{\mu}_{\text{MLE}} = \bar{x}$ and $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$. Then show that $\hat{\sigma}^2_{\text{MLE}}$ is biased: $E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. State where independence and identical distribution are separately used.

**출제 의도:**
Gaussian MLE는 Bernoulli MLE와 함께 가장 중요한 유도이다. 특히 분산 추정량의 편향성(biasedness)은 MLE의 한계를 보여주는 고전적 예시이며, n-1로 나누는 Bessel's correction의 이론적 근거를 이해해야 한다.

**정답 작성의 핵심 포인트:**
- Gaussian pdf 작성 → i.i.d.로 joint → product → log → sum
- μ에 대해 미분 → μ_MLE = x̄
- σ²에 대해 미분 → σ²_MLE = (1/n)Σ(x_i - x̄)²
- E[σ²_MLE] 계산에서 E[(x_i - x̄)²]를 전개
- 독립/동일분포가 사용되는 지점 명시

**모범 풀이:**

**Step 1: 모델 설정 및 Likelihood**

각 $x_i$는 Gaussian 분포를 따른다:
$$P(x_i | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**[독립 가정]**: Joint probability를 product로 분해:
$$L(\mu, \sigma^2) = \prod_{i=1}^{n} P(x_i | \mu, \sigma^2)$$

**[동일 분포 가정]**: 모든 $x_i$가 같은 $(\mu, \sigma^2)$를 공유:
$$L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$$

**Step 2: Log-Likelihood**

왜 log를 취하는가? 곱을 합으로 변환하고 exp를 제거하기 위해:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Step 3: μ_MLE 유도**

$\mu$에 대해 편미분한다 (왜? 극값의 필요조건):

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = \frac{1}{\sigma^2}\left(\sum_{i=1}^{n} x_i - n\mu\right)$$

$= 0$으로 놓으면:

$$\sum_{i=1}^{n} x_i = n\mu$$

$$\boxed{\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i}$$

**Step 4: σ²_MLE 유도**

$s = \sigma^2$로 놓고 $s$에 대해 편미분한다:

$$\frac{\partial \ell}{\partial s} = -\frac{n}{2s} + \frac{1}{2s^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

$= 0$으로 놓고 $\mu = \hat{\mu}_{\text{MLE}} = \bar{x}$를 대입:

$$\frac{n}{2s} = \frac{1}{2s^2}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

$$ns = \sum_{i=1}^{n}(x_i - \bar{x})^2$$

$$\boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

**Step 5: σ²_MLE의 편향성 증명**

$E[\hat{\sigma}^2_{\text{MLE}}]$를 계산한다.

핵심 관찰: $x_i - \bar{x}$를 전개한다.

$$x_i - \bar{x} = (x_i - \mu) - (\bar{x} - \mu)$$

$$\sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n \left[(x_i - \mu) - (\bar{x} - \mu)\right]^2$$

전개하면:

$$= \sum_{i=1}^n (x_i - \mu)^2 - 2(\bar{x} - \mu)\sum_{i=1}^n (x_i - \mu) + n(\bar{x} - \mu)^2$$

$\sum_{i=1}^n (x_i - \mu) = n(\bar{x} - \mu)$이므로:

$$= \sum_{i=1}^n (x_i - \mu)^2 - 2n(\bar{x} - \mu)^2 + n(\bar{x} - \mu)^2$$

$$= \sum_{i=1}^n (x_i - \mu)^2 - n(\bar{x} - \mu)^2$$

기댓값을 취한다:

$$E\left[\sum_{i=1}^n (x_i - \bar{x})^2\right] = \sum_{i=1}^n E[(x_i - \mu)^2] - n \cdot E[(\bar{x} - \mu)^2]$$

**[동일 분포 사용]**: 모든 $x_i$가 같은 분포이므로 $E[(x_i - \mu)^2] = \sigma^2$ (각 $i$에 대해 동일):

$$\sum_{i=1}^n E[(x_i - \mu)^2] = n\sigma^2$$

**[독립 가정 사용]**: $\text{Var}(\bar{x}) = \text{Var}\left(\frac{1}{n}\sum x_i\right) = \frac{1}{n^2}\sum \text{Var}(x_i) = \frac{\sigma^2}{n}$

여기서 독립이므로 분산이 합산된다 ($\text{Var}(\sum X_i) = \sum \text{Var}(X_i)$, 공분산 항이 0).

따라서:
$$E[(\bar{x} - \mu)^2] = \text{Var}(\bar{x}) = \frac{\sigma^2}{n}$$

대입하면:
$$E\left[\sum_{i=1}^n (x_i - \bar{x})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$

$$\boxed{E[\hat{\sigma}^2_{\text{MLE}}] = \frac{1}{n} \cdot (n-1)\sigma^2 = \frac{n-1}{n}\sigma^2 \neq \sigma^2}$$

$\frac{n-1}{n} < 1$이므로 $\hat{\sigma}^2_{\text{MLE}}$는 $\sigma^2$를 **과소추정(underestimate)**한다. 이것이 바로 비편향 추정량에서 $n$ 대신 $n-1$로 나누는 **Bessel's correction**의 근거이다:

$$s^2_{\text{unbiased}} = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$$

**자주 하는 실수:**
- $E[(x_i - \bar{x})^2]$를 바로 $\sigma^2$로 쓰는 것 ($\bar{x}$가 $x_i$에 의존하므로 그렇지 않다)
- Var(x̄) 계산에서 독립 가정이 필요하다는 점을 누락하는 것
- 편향의 방향(과소추정)을 명시하지 않는 것

**채점 기준:**
- Likelihood 작성 및 log-likelihood: 1pt
- μ_MLE = x̄ 유도: 2pts
- σ²_MLE = (1/n)Σ(x_i-x̄)² 유도: 2pts
- E[σ²_MLE] = (n-1)/n · σ² 증명: 3pts
- 독립/동일분포 사용 지점 명시: 2pts

---

## Problem 8. (10pts)

**Question (English):**
Coin flip experiment: $n = 5$ flips, $k = 4$ heads.
(a) With a Uniform prior on $\theta \in [0,1]$, show that $\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{MLE}} = 4/5$.
(b) With a Beta(2,2) prior: $p(\theta) \propto \theta(1-\theta)$, derive $\hat{\theta}_{\text{MAP}} = 5/7$.
(c) With a strong prior: $p(\theta) \propto \theta^m(1-\theta)^m$ for large $m$, show that $\hat{\theta}_{\text{MAP}} \to 1/2$ as $m \to \infty$.
For each part, show the full derivation with differentiation.

**출제 의도:**
MAP 추정에서 prior의 강도가 결과에 어떤 영향을 미치는지 직접 계산을 통해 체감하게 하는 문제이다. Uniform prior → MLE와 동일, 약한 prior → MLE 방향으로 이동하되 중심 쪽으로 당겨짐, 강한 prior → prior에 지배됨. 이 progression은 prior-data interaction의 핵심이다.

**정답 작성의 핵심 포인트:**
- MAP = argmax [log-likelihood + log-prior]
- Uniform prior → log-prior = constant → MAP = MLE
- Beta(2,2) → log-prior가 추가 항 기여 → 결과가 1/2 방향으로 이동
- m → ∞에서 prior가 지배 → MAP → prior의 mode = 1/2
- 각 part에서 완전한 미분 과정 제시

**모범 풀이:**

**공통 설정:**

데이터: $n = 5$, $k = 4$ (앞면 4번, 뒷면 1번).

Likelihood (Bernoulli i.i.d.):
$$L(\theta) = \theta^4(1-\theta)^1$$

MAP 목표: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log L(\theta) + \log P(\theta)\right]$

왜 log를 취하는가? MAP의 posterior를 최대화하는 것은 log-posterior를 최대화하는 것과 동치이며, log는 곱을 합으로 변환해준다.

---

**(a) Uniform Prior: $P(\theta) = 1$ for $\theta \in [0,1]$**

Log-posterior:
$$\log P(\theta|D) \propto \log L(\theta) + \log P(\theta) = 4\log\theta + \log(1-\theta) + \underbrace{\log 1}_{= 0}$$

Uniform prior의 log는 상수(0)이므로, log-posterior = log-likelihood이다.

미분하여 0으로 놓기:

$$\frac{d}{d\theta}\left[4\log\theta + \log(1-\theta)\right] = \frac{4}{\theta} - \frac{1}{1-\theta} = 0$$

$$4(1-\theta) = \theta$$

$$4 - 4\theta = \theta$$

$$4 = 5\theta$$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{4}{5} = \hat{\theta}_{\text{MLE}}}$$

**결론**: Uniform prior는 아무런 정보를 추가하지 않으므로 MAP = MLE이다.

---

**(b) Beta(2,2) Prior: $P(\theta) \propto \theta^1(1-\theta)^1 = \theta(1-\theta)$**

Beta(2,2)의 의미: 이 prior는 $\theta = 0.5$ 부근에 가장 높은 확률을 부여하여, "동전이 대략 공정하다"는 약한 믿음을 나타낸다.

Log-posterior:
$$\log P(\theta|D) \propto 4\log\theta + \log(1-\theta) + \log\theta + \log(1-\theta)$$
$$= 5\log\theta + 2\log(1-\theta)$$

미분하여 0으로 놓기:

$$\frac{d}{d\theta}\left[5\log\theta + 2\log(1-\theta)\right] = \frac{5}{\theta} - \frac{2}{1-\theta} = 0$$

$$5(1-\theta) = 2\theta$$

$$5 - 5\theta = 2\theta$$

$$5 = 7\theta$$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{5}{7} \approx 0.714}$$

**해석**: MLE = 4/5 = 0.800이었는데, Beta(2,2) prior의 영향으로 MAP가 $0.5$ 방향으로 당겨져서 0.714가 되었다. Prior가 "공정한 동전 쪽으로" 추정을 조정하는 역할을 했다.

**일반 공식**: Beta(α,β) prior + Bernoulli likelihood(k heads, n-k tails)의 MAP:
$$\hat{\theta}_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$

검증: $\frac{4 + 2 - 1}{5 + 2 + 2 - 2} = \frac{5}{7}$ ✓

---

**(c) Strong Prior: $P(\theta) \propto \theta^m(1-\theta)^m$, i.e., Beta($m+1$, $m+1$)**

Log-posterior:
$$\log P(\theta|D) \propto 4\log\theta + \log(1-\theta) + m\log\theta + m\log(1-\theta)$$
$$= (4+m)\log\theta + (1+m)\log(1-\theta)$$

미분하여 0으로 놓기:

$$\frac{4+m}{\theta} - \frac{1+m}{1-\theta} = 0$$

$$(4+m)(1-\theta) = (1+m)\theta$$

$$4 + m - (4+m)\theta = (1+m)\theta$$

$$4 + m = (4+m)\theta + (1+m)\theta = (5 + 2m)\theta$$

$$\hat{\theta}_{\text{MAP}} = \frac{4+m}{5+2m}$$

$m \to \infty$에서의 극한:

$$\lim_{m \to \infty} \frac{4+m}{5+2m} = \lim_{m \to \infty} \frac{4/m + 1}{5/m + 2} = \frac{1}{2}$$

$$\boxed{\hat{\theta}_{\text{MAP}} \xrightarrow{m \to \infty} \frac{1}{2}}$$

**해석**: Prior가 무한히 강해지면, 데이터(4/5 heads)의 정보는 완전히 무시되고, prior의 최빈값(mode) $\theta = 1/2$로 MAP가 수렴한다. 이것은 "데이터보다 사전 믿음을 더 신뢰한다"는 것의 극단적 사례이다.

**3개 결과 요약:**

| Prior | MAP | 해석 |
|-------|-----|------|
| Uniform | 4/5 = 0.800 | 데이터만 반영 (= MLE) |
| Beta(2,2) | 5/7 ≈ 0.714 | 데이터 + 약한 prior → 0.5 쪽으로 약간 이동 |
| Beta(m+1,m+1), m→∞ | → 1/2 | Prior가 지배 → 데이터 무시 |

이 progression은 **prior의 강도가 MAP에 미치는 영향**을 명확히 보여준다.

**자주 하는 실수:**
- (b)에서 Beta(2,2)의 log가 $\log\theta + \log(1-\theta)$임을 쓰지 않고, 비례 상수를 포함하여 복잡하게 만드는 것
- (c)에서 극한 계산을 하지 않고 "m이 크면 1/2에 가까워진다"로만 쓰는 것
- Prior의 강도 변화에 따른 결과 해석을 빠뜨리는 것

**채점 기준:**
- (a) Uniform prior → MAP = MLE 유도 및 이유: 3pts
- (b) Beta(2,2) → MAP = 5/7 유도: 3pts
- (c) Strong prior → MAP → 1/2 유도 및 극한: 3pts
- 전체 해석 (prior 강도와 MAP의 관계): 1pt

---

## Problem 9. (10pts)

**Question (English):**
Given the softmax function $p_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$, derive:
- Case $i = j$: $\frac{\partial p_i}{\partial z_j} = p_i(1 - p_i)$ using the quotient rule
- Case $i \neq j$: $\frac{\partial p_i}{\partial z_j} = -p_i p_j$
- Write the compact matrix form: $\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$

**출제 의도:**
Softmax의 Jacobian은 backpropagation의 핵심 구성 요소이다. 두 경우(i=j, i≠j)를 구분하여 유도하고, 이를 하나의 행렬 표현으로 통합하는 능력을 평가한다.

**정답 작성의 핵심 포인트:**
- Quotient rule 적용 (i=j 경우)
- 분모만 미분되는 경우 (i≠j)
- 각 경우를 Kronecker delta로 통합하여 행렬 표현

**모범 풀이:**

표기: $S = \sum_k \exp(z_k)$로 놓으면 $p_i = \frac{\exp(z_i)}{S}$이다.

---

**Case 1: $i = j$ (자기 자신에 대한 미분)**

Quotient rule을 적용한다: $\frac{d}{dx}\frac{f}{g} = \frac{f'g - fg'}{g^2}$

여기서 $f = \exp(z_i)$, $g = S = \sum_k \exp(z_k)$이고:
- $\frac{\partial f}{\partial z_i} = \exp(z_i)$ ($z_i$에 대한 $\exp(z_i)$의 미분)
- $\frac{\partial g}{\partial z_i} = \exp(z_i)$ ($S$ 안에서 $z_i$에 해당하는 항만 미분)

$$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot S - \exp(z_i) \cdot \exp(z_i)}{S^2}$$

$$= \frac{\exp(z_i)}{S} \cdot \frac{S - \exp(z_i)}{S}$$

$$= p_i \cdot \frac{S - \exp(z_i)}{S}$$

$$= p_i \cdot \left(1 - \frac{\exp(z_i)}{S}\right)$$

$$\boxed{\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)}$$

---

**Case 2: $i \neq j$ (다른 변수에 대한 미분)**

다시 quotient rule을 적용한다. 이번에는:
- $f = \exp(z_i)$, $\frac{\partial f}{\partial z_j} = 0$ ($z_j$에 대한 $\exp(z_i)$의 미분. $i \neq j$이므로 0)
- $g = S$, $\frac{\partial g}{\partial z_j} = \exp(z_j)$

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - \exp(z_i) \cdot \exp(z_j)}{S^2}$$

$$= -\frac{\exp(z_i)}{S} \cdot \frac{\exp(z_j)}{S}$$

$$\boxed{\frac{\partial p_i}{\partial z_j} = -p_i p_j}$$

---

**Compact Matrix Form:**

두 경우를 Kronecker delta $\delta_{ij}$로 통합할 수 있다:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

검증:
- $i = j$: $p_i(1 - p_i)$ ✓
- $i \neq j$: $p_i(0 - p_j) = -p_ip_j$ ✓

이를 행렬로 쓰면, Jacobian $J \in \mathbb{R}^{K \times K}$에서 $(i,j)$ 원소가 $\frac{\partial p_i}{\partial z_j}$이므로:

$$J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

여기서:
- $\text{diag}(\mathbf{p})$: 대각 원소가 $p_1, p_2, \ldots, p_K$인 대각행렬 (Case 1의 $p_i \cdot 1$ 부분 기여)
- $\mathbf{p}\mathbf{p}^T$: $(i,j)$ 원소가 $p_ip_j$인 rank-1 행렬 (Case 2의 $-p_ip_j$ 부분 기여)

$$\boxed{\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T}$$

**자주 하는 실수:**
- Case $i \neq j$에서 $\frac{\partial \exp(z_i)}{\partial z_j} = 0$을 잊고 분자에도 미분 항을 남기는 것
- 행렬 표현에서 $\text{diag}(\mathbf{p})$와 $\mathbf{p}\mathbf{p}^T$의 차원을 혼동하는 것
- Kronecker delta를 이용한 통합 표현을 쓰지 못하는 것

**채점 기준:**
- Case i=j: quotient rule 적용 및 p_i(1-p_i) 유도: 3pts
- Case i≠j: ∂exp(z_i)/∂z_j = 0 명시 및 -p_ip_j 유도: 3pts
- Kronecker delta 통합 표현: 2pts
- 행렬 compact form diag(p) - pp^T: 2pts

---

## Problem 10. (10pts)

**Question (English):**
(a) Prove that the derivative of the sigmoid function satisfies: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
(b) For binary cross-entropy loss with $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x})$, derive the gradient $\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y)\mathbf{x}$, showing explicitly where the $\sigma'$ cancellation occurs.
(c) Explain why $\max_z \sigma'(z) = 1/4$ and its consequence for gradient-based learning (vanishing gradient problem).

**출제 의도:**
Sigmoid의 미분 성질과 CE loss에서의 gradient 소거(cancellation)는 딥러닝의 근본적인 수학이다. σ'(z)가 CE loss의 gradient에서 깔끔하게 소거되는 것은 "왜 CE loss가 classification에 적합한가"의 핵심 답변이다.

**정답 작성의 핵심 포인트:**
- σ(z) = 1/(1+exp(-z))를 미분하여 σ(z)(1-σ(z)) 유도
- BCE loss에서 chain rule 적용 시 σ' 소거를 명시적으로 보임
- σ'(z)의 최대값 = 1/4 계산 (z=0에서)
- Vanishing gradient: σ'가 최대 1/4이므로 layer가 깊어질수록 gradient가 기하급수적으로 감소

**모범 풀이:**

**(a) σ'(z) = σ(z)(1 - σ(z)) 증명:**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

미분한다. $\sigma(z) = (1 + e^{-z})^{-1}$로 보고 chain rule 적용:

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z})$$

$$= \frac{e^{-z}}{(1 + e^{-z})^2}$$

이것이 $\sigma(z)(1-\sigma(z))$와 같음을 보인다:

$$\sigma(z)(1 - \sigma(z)) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2}$$

따라서:

$$\boxed{\sigma'(z) = \sigma(z)(1 - \sigma(z))}$$ ∎

---

**(b) Binary CE Loss의 Gradient 유도 및 σ' 소거:**

Binary Cross-Entropy loss:
$$L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$$

여기서 $\hat{y} = \sigma(z)$, $z = \mathbf{w}^T\mathbf{x}$이다.

Chain rule을 적용한다: $\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}$

**항 1**: $\frac{\partial L}{\partial \hat{y}}$

$$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = \frac{-y(1-\hat{y}) + (1-y)\hat{y}}{\hat{y}(1-\hat{y})} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$$

**항 2**: $\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1-\sigma(z)) = \hat{y}(1-\hat{y})$

**항 3**: $\frac{\partial z}{\partial \mathbf{w}} = \mathbf{x}$

세 항을 곱한다:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot \mathbf{x}$$

**★ 소거 발생!** $\hat{y}(1-\hat{y})$가 분모와 분자에서 정확히 소거된다:

$$\boxed{\frac{\partial L}{\partial \mathbf{w}} = (\hat{y} - y)\mathbf{x}}$$

이 결과는 놀랍도록 깔끔하다. Gradient는 단순히 (예측 - 정답) × 입력이다. σ'가 소거되었기 때문에, $\hat{y}$가 0이나 1에 가까울 때도 gradient가 소실되지 않는다.

---

**(c) max σ'(z) = 1/4 및 Vanishing Gradient:**

$\sigma'(z) = \sigma(z)(1-\sigma(z))$의 최대값을 구한다.

$p = \sigma(z)$로 놓으면 $f(p) = p(1-p)$이고, $0 < p < 1$에서:

$$f'(p) = 1 - 2p = 0 \implies p = \frac{1}{2}$$

$$f\left(\frac{1}{2}\right) = \frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}$$

$f''(p) = -2 < 0$이므로 이것은 최대값이다. 이는 $z = 0$일 때 달성된다 ($\sigma(0) = 1/2$).

$$\boxed{\max_z \sigma'(z) = \frac{1}{4}}$$

**Vanishing Gradient의 결과:**

깊은 네트워크에서 backpropagation 시, 각 layer에서 $\sigma'$가 곱해진다. 최대값이 1/4이므로:

$$\frac{\partial L}{\partial \mathbf{w}^{(1)}} \propto \prod_{l=1}^{L} \sigma'(z^{(l)}) \leq \left(\frac{1}{4}\right)^L$$

$L$이 큰 깊은 네트워크에서 이 값은 기하급수적으로 0에 가까워진다. 이것이 **vanishing gradient problem**이다.

예: $L = 10$ layers → gradient ≤ $(1/4)^{10} \approx 10^{-6}$

이 문제를 해결하기 위해:
- **ReLU 활성화 함수**: $\text{ReLU}'(z) = 1$ ($z > 0$), gradient 소실 없음
- **ResNet (skip connections)**: gradient가 직접 전달되는 경로 제공
- **Batch Normalization**: 활성화 값을 정규화하여 σ'가 0에 가깝지 않도록 유지

**자주 하는 실수:**
- (b)에서 chain rule의 세 항을 명확히 분리하지 않고 한꺼번에 계산하여 소거 과정이 보이지 않는 것
- (c)에서 max σ' = 1/4를 구한 후 vanishing gradient와의 연결을 서술하지 않는 것
- CE loss에서 σ'가 소거된다는 것의 **실용적 의미**를 설명하지 않는 것

**채점 기준:**
- (a) σ'(z) = σ(z)(1-σ(z)) 증명: 3pts
- (b) Chain rule 3항 분리 및 σ' 소거 명시: 4pts
- (c) max σ' = 1/4 계산 및 vanishing gradient 설명: 3pts

---

# Part III: MAP/MLE/Gaussian/Loss Application (응용, Problems 11–15)

---

## Problem 11. (10pts)

**Question (English):**
Starting from the regression model $y_i = f_\theta(x_i) + \varepsilon_i$ where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d., derive that $\arg\min_\theta \text{NLL} = \arg\min_\theta \text{MSE}$. Every line of the derivation must include a sentence explaining what was done and why.

**출제 의도:**
"Gaussian 노이즈 가정 → NLL → MSE"의 체인은 교수님이 가장 중요시하는 유도 중 하나이다. 단순히 수식만 나열하는 것이 아니라, 각 줄마다 "왜 이 단계를 수행하는가"를 설명해야 한다. 이것은 "수학을 이해한다"와 "수학을 외웠다"를 구분하는 핵심이다.

**정답 작성의 핵심 포인트:**
- 회귀 모델 설정: y = f(x) + ε
- Gaussian 노이즈 가정의 의미와 근거 (CLT)
- i.i.d.로 joint → product → log → sum
- 상수 항 제거 논리
- MSE 형태 도출

**모범 풀이:**

**Line 1: 모델 가정**

$$y_i = f_\theta(x_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2) \text{ i.i.d.}$$

> 회귀 문제에서 타겟 $y_i$는 모델의 예측값 $f_\theta(x_i)$에 노이즈 $\varepsilon_i$가 더해진 것으로 모델링한다. 노이즈가 Gaussian인 이유는 CLT에 의해 많은 독립적인 작은 요인들의 합이 정규분포에 수렴하기 때문이다.

**Line 2: 조건부 분포 도출**

$$y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$$

> $\varepsilon_i = y_i - f_\theta(x_i)$이므로, $y_i$는 평균이 $f_\theta(x_i)$이고 분산이 $\sigma^2$인 Gaussian을 따른다.

**Line 3: 개별 확률밀도 작성**

$$P(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

> Gaussian 확률밀도함수를 명시적으로 작성한다.

**Line 4: Joint likelihood 작성 [★ i.i.d. 가정 사용]**

$$P(\mathbf{y} | \mathbf{X}, \theta) = \prod_{i=1}^{n} P(y_i | x_i, \theta)$$

> **[독립 가정 사용]**: 노이즈 $\varepsilon_i$들이 독립이므로 joint probability를 product로 분해할 수 있다.
> **[동일 분포 가정 사용]**: 모든 $\varepsilon_i$가 같은 $\mathcal{N}(0, \sigma^2)$를 따르므로 동일한 $\sigma^2$를 사용한다.

**Line 5: Log 변환 [왜 log를 취하는가?]**

$$\log P(\mathbf{y}|\mathbf{X},\theta) = \sum_{i=1}^{n} \log P(y_i|x_i,\theta)$$

> **왜 log를 취하는가?** (1) product를 sum으로 변환하여 미분과 계산이 편해진다. (2) log는 단조증가함수이므로 argmax가 보존된다. (3) 매우 작은 확률들의 곱에서 수치적 언더플로우를 방지한다.

**Line 6: Log-likelihood 전개**

$$\log P(\mathbf{y}|\mathbf{X},\theta) = \sum_{i=1}^{n}\left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right]$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

> 각 항의 log를 전개한다. $\log(\text{exp}(\cdot))$에 의해 지수가 제거된다.

**Line 7: NLL (Negative Log-Likelihood) 형성**

$$\text{NLL} = -\log P(\mathbf{y}|\mathbf{X},\theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

> 최대화를 최소화 문제로 변환하기 위해 부호를 뒤집는다. $\arg\max \log P = \arg\min (-\log P)$

**Line 8: θ에 무관한 항 제거**

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

> $\frac{n}{2}\log(2\pi\sigma^2)$는 $\theta$에 의존하지 않는 상수이므로 최적화에 영향을 주지 않아 제거할 수 있다.

**Line 9: 양의 상수 제거**

$$= \arg\min_\theta \sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

> $\frac{1}{2\sigma^2} > 0$은 양의 상수이므로 argmin에 영향을 주지 않는다. 양의 상수를 곱해도 최소점은 변하지 않기 때문이다.

**Line 10: MSE와의 동치**

$$= \arg\min_\theta \frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 = \arg\min_\theta \text{MSE}$$

> $1/n$도 양의 상수이므로 곱하거나 나누어도 argmin은 동일하다.

$$\boxed{\arg\min_\theta \text{NLL}_{\text{Gaussian}} = \arg\min_\theta \text{MSE}}$$

**핵심 요약**: Gaussian 노이즈 가정 + i.i.d. 가정 하에서, Maximum Likelihood Estimation은 MSE 최소화와 정확히 동치이다. MSE는 "단순한 거리 측정"이 아니라, **Gaussian 노이즈를 가정한 확률 모델의 NLL**이라는 깊은 이론적 의미를 가진다.

**자주 하는 실수:**
- "왜 log를 취하는가", "왜 상수를 제거할 수 있는가"를 설명하지 않고 바로 수식만 쓰는 것
- i.i.d.의 독립/동일분포가 각각 어디서 사용되는지 표시하지 않는 것
- MSE가 단순한 거리 측정이 아닌 확률적 의미를 가진다는 점을 결론에서 언급하지 않는 것

**채점 기준:**
- 모델 설정 및 Gaussian 가정: 1pt
- Joint → Product (독립 가정 명시): 2pts
- Log 변환 이유 설명: 2pts
- NLL 전개: 2pts
- 상수 항 제거 논리 및 MSE 도출: 2pts
- "왜" 설명의 충실도: 1pt

---

## Problem 12. (10pts)

**Question (English):**
Show that placing a Gaussian prior $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$ on the parameters in MAP estimation yields: $\hat{\theta}_{\text{MAP}} = \arg\min_\theta \left[\text{MSE} + \lambda\|\theta\|^2\right]$. Derive $\lambda = \frac{\sigma^2}{n\sigma_p^2}$ explicitly. Interpret what happens in the three limiting cases: $\sigma_p^2 \to 0$, $n \to \infty$, and $\sigma^2 \to \infty$.

**출제 의도:**
L2 정규화(Ridge Regression)가 MAP에서 Gaussian prior를 놓은 것과 동치라는 사실은 정규화의 이론적 정당성을 제공하는 핵심 결과이다. λ의 공식을 정확히 유도하고, 극한 해석까지 할 수 있어야 한다.

**정답 작성의 핵심 포인트:**
- MAP = argmax [log-likelihood + log-prior]
- Gaussian likelihood → MSE 항
- Gaussian prior → ||θ||² 항
- λ = σ²/(nσ_p²) 유도
- 3가지 극한의 물리적 해석

**모범 풀이:**

**Step 1: MAP 목적 함수 설정**

MAP 추정:
$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log P(D|\theta) + \log P(\theta)\right]$$

> Bayes' theorem에서 $P(\theta|D) \propto P(D|\theta)P(\theta)$이고, evidence $P(D)$는 $\theta$에 무관하므로 posterior 최대화는 likelihood × prior 최대화와 동치이다.

**Step 2: Log-Likelihood (Gaussian 노이즈 가정)**

Problem 11에서 유도한 결과를 사용한다. $y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$이면:

$$\log P(D|\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

**Step 3: Log-Prior (Gaussian prior)**

$\theta \sim \mathcal{N}(0, \sigma_p^2 I)$에서 $\theta \in \mathbb{R}^d$이면:

$$P(\theta) = \frac{1}{(2\pi\sigma_p^2)^{d/2}} \exp\left(-\frac{\|\theta\|^2}{2\sigma_p^2}\right)$$

$$\log P(\theta) = -\frac{d}{2}\log(2\pi\sigma_p^2) - \frac{\|\theta\|^2}{2\sigma_p^2}$$

**Step 4: Log-Posterior 결합**

$$\log P(\theta|D) \propto -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 - \frac{\|\theta\|^2}{2\sigma_p^2} + \text{const}$$

최대화를 최소화로 변환 (부호 반전):

$$\hat{\theta}_{\text{MAP}} = \arg\min_\theta \left[\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 + \frac{\|\theta\|^2}{2\sigma_p^2}\right]$$

**Step 5: λ 유도**

$\frac{1}{2\sigma^2}$을 밖으로 빼면:

$$= \arg\min_\theta \left[\frac{1}{2\sigma^2}\left(\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 + \frac{\sigma^2}{\sigma_p^2}\|\theta\|^2\right)\right]$$

양의 상수 $\frac{1}{2\sigma^2}$는 argmin에 영향 없으므로:

$$= \arg\min_\theta \left[\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 + \frac{\sigma^2}{\sigma_p^2}\|\theta\|^2\right]$$

$n$으로 나누어 MSE 형태로 만들면:

$$= \arg\min_\theta \left[\frac{1}{n}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 + \frac{\sigma^2}{n\sigma_p^2}\|\theta\|^2\right]$$

$$\boxed{= \arg\min_\theta \left[\text{MSE} + \lambda\|\theta\|^2\right], \quad \text{where } \lambda = \frac{\sigma^2}{n\sigma_p^2}}$$

**Step 6: 극한 해석**

| 극한 | λ의 변화 | 해석 |
|------|---------|------|
| $\sigma_p^2 \to 0$ | $\lambda \to \infty$ | Prior가 극도로 강하다 (θ가 0 근처에 있다고 강하게 믿음). 정규화가 극도로 강해져서 $\theta \to 0$. 모델이 아무것도 학습하지 못한다 (underfitting). |
| $n \to \infty$ | $\lambda \to 0$ | 데이터가 무한히 많아지면 prior의 영향이 사라진다. MAP → MLE. 이것은 Bayesian에서 "데이터가 충분하면 prior가 중요하지 않다"는 원리의 수학적 표현이다. |
| $\sigma^2 \to \infty$ | $\lambda \to \infty$ | 데이터의 노이즈가 극도로 크다. 데이터가 신뢰할 수 없으므로 prior에 더 의존한다. 정규화가 강해져서 $\theta \to 0$. 이것은 "불확실한 데이터보다 사전 믿음을 더 신뢰한다"는 의미이다. |

**자주 하는 실수:**
- λ = σ²/(nσ_p²)에서 n을 빠뜨리는 것 (MSE로 변환할 때 1/n을 고려해야 함)
- 극한 해석에서 λ의 변화 방향은 맞추되, 그것이 모델에 미치는 영향을 서술하지 않는 것
- Gaussian prior → L2인데, Laplacian prior → L1이라는 대응을 혼동하는 것

**채점 기준:**
- MAP 목적 함수 설정: 1pt
- Log-likelihood 및 log-prior 작성: 2pts
- MSE + λ||θ||² 형태 유도: 3pts
- λ = σ²/(nσ_p²) 명시적 유도: 2pts
- 3가지 극한 해석: 2pts (오답 시 부분점수 없음)

---

## Problem 13. (10pts)

**Question (English):**
(a) Define KL divergence, Cross-Entropy, and Entropy precisely.
(b) Prove that $\text{KL}(P \| Q) \geq 0$ (Gibbs' inequality).
(c) Show that minimizing CE equals minimizing KL when $P_{\text{data}}$ is fixed. Connect to empirical distribution and ERM.
(d) Compute $H(P)$, $H(P,Q)$, and $\text{KL}(P\|Q)$ for $P = (1/4, 1/4, 1/4, 1/4)$ and $Q = (1/2, 1/4, 1/8, 1/8)$.

**출제 의도:**
KL divergence, Cross-Entropy, Entropy의 관계를 정확히 이해하고, CE 최소화가 KL 최소화와 동치인 이유를 설명할 수 있어야 한다. 이론적 이해와 구체적 계산을 모두 평가한다.

**정답 작성의 핵심 포인트:**
- 세 개념의 정확한 수학적 정의
- Gibbs' inequality 증명 (Jensen's inequality 이용)
- CE = H(P) + KL(P||Q) → H(P) 고정이면 argmin CE = argmin KL
- 구체적 수치 계산

**모범 풀이:**

**(a) 정의:**

이산 확률분포 $P$와 $Q$가 같은 사건 공간 위에 정의되어 있을 때:

**Entropy (엔트로피):**
$$H(P) = -\sum_{x} P(x) \log P(x)$$
> $P$의 불확실성(정보량)을 측정한다. $P$가 균등분포일수록 높고, 하나에 집중될수록 낮다.

**Cross-Entropy (교차 엔트로피):**
$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$
> 진짜 분포가 $P$일 때, 분포 $Q$를 사용하여 인코딩하는 데 필요한 평균 비트 수이다.

**KL Divergence (쿨백-라이블러 발산):**
$$D_{\text{KL}}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$
> $Q$로 $P$를 근사할 때의 정보 손실을 측정한다. $P = Q$이면 $D_{\text{KL}} = 0$이다.

**세 개의 관계:**
$$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$$

이것은 직접 보일 수 있다:
$$H(P,Q) = -\sum P(x)\log Q(x) = -\sum P(x)\log P(x) + \sum P(x)\log\frac{P(x)}{Q(x)} = H(P) + D_{\text{KL}}(P\|Q)$$

---

**(b) Gibbs' Inequality: $D_{\text{KL}}(P \| Q) \geq 0$ 증명**

**Jensen's inequality**를 사용한다. $\log$는 오목(concave) 함수이므로:

$$E[\log X] \leq \log E[X]$$

따라서 $-\log$는 볼록(convex)이고:

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = -\sum_x P(x) \log \frac{Q(x)}{P(x)}$$

Jensen's inequality ($-\log$가 convex이므로):

$$\geq -\log\left(\sum_x P(x) \cdot \frac{Q(x)}{P(x)}\right) = -\log\left(\sum_x Q(x)\right) = -\log(1) = 0$$

따라서:
$$\boxed{D_{\text{KL}}(P \| Q) \geq 0}$$

등호 조건: $P(x) = Q(x)$ for all $x$ (즉, 두 분포가 동일할 때).

---

**(c) CE 최소화 = KL 최소화:**

$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$에서:

$$\arg\min_Q H(P, Q) = \arg\min_Q \left[H(P) + D_{\text{KL}}(P \| Q)\right]$$

$H(P)$는 $Q$에 의존하지 않는 상수이므로:

$$\boxed{\arg\min_Q H(P, Q) = \arg\min_Q D_{\text{KL}}(P \| Q)}$$

> 진짜 분포 $P_{\text{data}}$가 고정되어 있을 때, Cross-Entropy를 최소화하는 것은 KL divergence를 최소화하는 것과 동치이다.

**ERM과의 연결:**

실제로 $P_{\text{data}}$를 알지 못하므로, 경험적 분포(empirical distribution) $\hat{P}_{\text{data}}$를 사용한다:

$$\hat{P}_{\text{data}}(x, y) = \frac{1}{n}\sum_{i=1}^{n}\delta(x - x_i, y - y_i)$$

이 경험적 분포를 사용하여 CE를 계산하면:

$$H(\hat{P}, Q_\theta) = -\frac{1}{n}\sum_{i=1}^{n}\log Q_\theta(y_i | x_i)$$

이것이 바로 **Empirical Risk Minimization (ERM)** with NLL loss이다. 따라서:

$$\text{ERM with NLL} = \text{CE 최소화} = \text{KL 최소화 (P 고정)}$$

---

**(d) 수치 계산:**

$P = (1/4, 1/4, 1/4, 1/4)$, $Q = (1/2, 1/4, 1/8, 1/8)$. $\log$는 $\log_2$ 사용.

**Entropy $H(P)$:**
$$H(P) = -4 \cdot \frac{1}{4}\log_2\frac{1}{4} = -4 \cdot \frac{1}{4} \cdot (-2) = 2 \text{ bits}$$

> 균등분포이므로 최대 엔트로피. 4개 사건의 최대 엔트로피는 $\log_2 4 = 2$이다.

**Cross-Entropy $H(P, Q)$:**
$$H(P,Q) = -\left[\frac{1}{4}\log_2\frac{1}{2} + \frac{1}{4}\log_2\frac{1}{4} + \frac{1}{4}\log_2\frac{1}{8} + \frac{1}{4}\log_2\frac{1}{8}\right]$$

$$= -\left[\frac{1}{4}(-1) + \frac{1}{4}(-2) + \frac{1}{4}(-3) + \frac{1}{4}(-3)\right]$$

$$= -\left[-\frac{1}{4} - \frac{2}{4} - \frac{3}{4} - \frac{3}{4}\right]$$

$$= -\left[-\frac{9}{4}\right] = \frac{9}{4} = 2.25 \text{ bits}$$

**KL Divergence $D_{\text{KL}}(P \| Q)$:**

방법 1: $D_{\text{KL}} = H(P,Q) - H(P) = 2.25 - 2 = 0.25$ bits

방법 2: 직접 계산:
$$D_{\text{KL}} = \frac{1}{4}\log_2\frac{1/4}{1/2} + \frac{1}{4}\log_2\frac{1/4}{1/4} + \frac{1}{4}\log_2\frac{1/4}{1/8} + \frac{1}{4}\log_2\frac{1/4}{1/8}$$

$$= \frac{1}{4}\log_2\frac{1}{2} + \frac{1}{4}\log_2 1 + \frac{1}{4}\log_2 2 + \frac{1}{4}\log_2 2$$

$$= \frac{1}{4}(-1) + 0 + \frac{1}{4}(1) + \frac{1}{4}(1) = -\frac{1}{4} + \frac{2}{4} = \frac{1}{4} = 0.25 \text{ bits}$$

$$\boxed{H(P) = 2, \quad H(P,Q) = 2.25, \quad D_{\text{KL}}(P\|Q) = 0.25 \text{ bits}}$$

검증: $H(P,Q) = H(P) + D_{\text{KL}}(P\|Q)$: $2.25 = 2 + 0.25$ ✓, $D_{\text{KL}} \geq 0$: $0.25 > 0$ ✓

**자주 하는 실수:**
- KL이 비대칭임을 언급하지 않는 것: $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$
- Gibbs' inequality 증명에서 Jensen's inequality의 방향을 틀리는 것 (convex vs concave)
- 수치 계산에서 log의 밑(base)을 명시하지 않는 것

**채점 기준:**
- (a) 세 개념 정의 + 관계식: 2pts
- (b) Gibbs' inequality 증명 (Jensen 사용): 3pts
- (c) CE 최소화 = KL 최소화 + ERM 연결: 2pts
- (d) 수치 계산: 3pts (H(P) 1pt, H(P,Q) 1pt, KL 1pt)

---

## Problem 14. (10pts)

**Question (English):**
(a) State the Central Limit Theorem (CLT) precisely with its three conditions.
(b) Explain why noise $\varepsilon$ can be modeled as Gaussian via CLT.
(c) Trace the complete chain: CLT → Gaussian noise → NLL → $-\log(\exp(-z^2)) = z^2$ → MSE.
(d) If noise were Laplacian $(\varepsilon \sim \text{Laplace}(0, b))$ instead of Gaussian, what loss function results from NLL?

**출제 의도:**
CLT가 왜 Gaussian 노이즈 가정을 정당화하는지, 그리고 이것이 MSE로 이어지는 전체 체인을 추적할 수 있어야 한다. 또한 Laplacian 노이즈 → MAE라는 대응을 통해 "분포 가정이 손실 함수를 결정한다"는 원리를 깊이 이해해야 한다.

**정답 작성의 핵심 포인트:**
- CLT의 정확한 서술 (3 조건: 독립, 동일 분포, 유한 분산)
- 노이즈 = 많은 독립 요인의 합 → CLT → Gaussian
- Gaussian → NLL → exp 제거 → 제곱 → MSE
- Laplacian → NLL → |y - f(x)| → MAE

**모범 풀이:**

**(a) 중심극한정리 (CLT) 정확한 서술:**

$X_1, X_2, \ldots, X_n$이 다음 조건을 만족하는 확률변수들이라 하자:
1. **독립 (Independent)**: 각 $X_i$가 서로 독립
2. **동일 분포 (Identically Distributed)**: 모든 $X_i$가 같은 분포를 따름
3. **유한 분산 (Finite Variance)**: $E[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2 < \infty$

그러면 표본 평균의 표준화된 합은 $n \to \infty$에서 표준정규분포에 수렴한다:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

또는 동치적으로: $\bar{X}_n \xrightarrow{d} \mathcal{N}(\mu, \sigma^2/n)$

**(b) 노이즈의 Gaussian 모델링 정당화:**

실세계에서 측정 오차나 노이즈 $\varepsilon$는 보통 **많은 독립적인 작은 요인들의 합**으로 발생한다:

$$\varepsilon = \delta_1 + \delta_2 + \cdots + \delta_m$$

여기서 각 $\delta_j$는 개별 오차 요인 (센서 오차, 환경 변동, 양자화 오차 등)이다.

각 $\delta_j$가:
- 서로 독립이고
- 동일한(또는 유사한) 분포를 따르며
- 유한한 분산을 가지면

CLT에 의해 $m$이 충분히 클 때:

$$\varepsilon = \sum_{j=1}^m \delta_j \approx \mathcal{N}\left(\sum \mu_j, \sum \sigma_j^2\right)$$

평균을 빼서 $E[\varepsilon] = 0$으로 정규화하면: $\varepsilon \sim \mathcal{N}(0, \sigma^2)$

**핵심**: CLT는 개별 요인의 분포가 무엇이든 (균등, 지수, 이항 등) 상관없이, 합이 Gaussian에 수렴한다는 것이다. 이것이 Gaussian 노이즈 가정의 이론적 근거이다.

**(c) 전체 체인: CLT → Gaussian → NLL → MSE**

$$\text{CLT} \xrightarrow{\text{step 1}} \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

$$\xrightarrow{\text{step 2}} P(y|x,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-f_\theta(x))^2}{2\sigma^2}\right)$$

$$\xrightarrow{\text{step 3}} -\log P(y|x,\theta) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-f_\theta(x))^2}{2\sigma^2}$$

> **Step 3의 핵심**: $-\log(\exp(-z^2)) = z^2$. log가 exp를 소거하여, Gaussian의 지수 부분이 제곱 오차로 변환된다. 이것이 "왜 MSE인가?"의 핵심 답이다.

$$\xrightarrow{\text{step 4}} \arg\min_\theta \sum_i (y_i - f_\theta(x_i))^2 = \arg\min_\theta \text{MSE}$$

$$\boxed{\text{CLT} \to \text{Gaussian} \to \text{NLL} \to -\log\exp(-z^2) = z^2 \to \text{MSE}}$$

**(d) Laplacian 노이즈 → MAE:**

Laplace 분포: $\varepsilon \sim \text{Laplace}(0, b)$

$$P(\varepsilon) = \frac{1}{2b}\exp\left(-\frac{|\varepsilon|}{b}\right)$$

따라서:

$$P(y|x,\theta) = \frac{1}{2b}\exp\left(-\frac{|y - f_\theta(x)|}{b}\right)$$

NLL:

$$-\log P(y|x,\theta) = \log(2b) + \frac{|y - f_\theta(x)|}{b}$$

$\theta$에 대해 최적화:

$$\arg\min_\theta \sum_{i=1}^{n} |y_i - f_\theta(x_i)| = \arg\min_\theta \text{MAE (Mean Absolute Error)}$$

$$\boxed{\text{Laplacian noise} \xrightarrow{\text{NLL}} \text{MAE (L1 Loss)}}$$

**비교 표:**

| 노이즈 분포 | NLL Loss | 특성 |
|------------|----------|------|
| Gaussian $\mathcal{N}(0, \sigma^2)$ | MSE ($L_2$) | 큰 오차에 민감 (제곱), outlier에 취약 |
| Laplacian $\text{Laplace}(0, b)$ | MAE ($L_1$) | 큰 오차에 덜 민감 (절대값), outlier에 robust |

**자주 하는 실수:**
- CLT의 3 조건 중 "유한 분산"을 빠뜨리는 것
- $-\log\exp(-z^2) = z^2$라는 핵심 단계를 명시하지 않는 것
- Laplacian → MAE 유도에서 절대값이 어디서 나오는지 설명하지 않는 것

**채점 기준:**
- (a) CLT 정확한 서술 (3 조건): 2pts
- (b) 노이즈 → Gaussian 정당화: 2pts
- (c) 전체 체인 추적: 3pts
- (d) Laplacian → MAE 유도: 3pts

---

## Problem 15. (10pts)

**Question (English):**
(a) For the matrix $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$: find the rank, the null space, and verify the Rank-Nullity theorem.
(b) For SVD: explain $A = U\Sigma V^T$, write the rank-$k$ approximation formula, and state the Eckart-Young theorem.
(c) Connect PCA to the eigendecomposition of the covariance matrix: show that maximizing $\mathbf{w}^T S \mathbf{w}$ subject to $\|\mathbf{w}\| = 1$ gives the top eigenvector of $S$, using Lagrange multipliers.

**출제 의도:**
선형대수의 핵심 개념(Rank, Null space, SVD, PCA)이 딥러닝의 차원 축소, 데이터 표현, 근사 이론에 직결된다. 특히 PCA-eigenvector 연결은 Lagrange multiplier를 활용한 제약 최적화 연습이다.

**정답 작성의 핵심 포인트:**
- Rank = 독립 열(행)의 수, Null space = Ax = 0의 해 집합
- Rank-Nullity: rank + nullity = 열의 수
- SVD 분해 및 rank-k 근사
- Eckart-Young: rank-k 근사 중 SVD가 Frobenius norm 기준 최적
- PCA: Lagrange → Sw = λw → top eigenvector

**모범 풀이:**

**(a) $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$의 분석:**

**Rank:**
두 번째 행 = 첫 번째 행 × 2이므로, 행 사다리꼴(row echelon form):
$$A \to \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}$$

피봇이 1개이므로:
$$\text{rank}(A) = 1$$

**Null Space:**
$A\mathbf{x} = \mathbf{0}$을 풀면:
$$\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$x_1 + 2x_2 = 0 \implies x_1 = -2x_2$

$$\text{Null}(A) = \text{span}\left\{\begin{bmatrix} -2 \\ 1 \end{bmatrix}\right\}$$

$\text{nullity}(A) = \dim(\text{Null}(A)) = 1$

**Rank-Nullity Theorem 검증:**
$$\text{rank}(A) + \text{nullity}(A) = 1 + 1 = 2 = \text{(열의 수)} \quad \checkmark$$

---

**(b) SVD (Singular Value Decomposition):**

임의의 $m \times n$ 행렬 $A$는 다음과 같이 분해된다:

$$A = U\Sigma V^T$$

여기서:
- $U \in \mathbb{R}^{m \times m}$: 좌특이벡터(left singular vectors)로 구성된 직교행렬. $A A^T$의 고유벡터.
- $\Sigma \in \mathbb{R}^{m \times n}$: 대각에 특이값(singular values) $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$을 가진 행렬.
- $V \in \mathbb{R}^{n \times n}$: 우특이벡터(right singular vectors)로 구성된 직교행렬. $A^T A$의 고유벡터.

**Rank-$k$ 근사:**

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

이것은 가장 큰 $k$개의 특이값과 대응하는 특이벡터만 사용하여 $A$를 근사한다.

**Eckart-Young Theorem:**

rank가 최대 $k$인 모든 행렬 $B$ 중에서, $A_k$가 $A$와의 거리를 최소화한다:

$$A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$$

여기서 $\|M\|_F = \sqrt{\sum_{i,j} M_{ij}^2}$은 Frobenius norm이다.

근사 오차: $\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$ (버려진 특이값의 제곱합)

---

**(c) PCA와 공분산 행렬의 고유분해 연결:**

**목표**: 데이터의 분산을 최대화하는 방향(주성분)을 찾는다.

데이터 $\mathbf{x}_1, \ldots, \mathbf{x}_n$의 공분산 행렬:
$$S = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$$

단위 벡터 $\mathbf{w}$ 방향으로의 사영 분산:
$$\text{Var}_{\mathbf{w}} = \mathbf{w}^T S \mathbf{w}$$

**제약 최적화 문제:**
$$\max_{\mathbf{w}} \mathbf{w}^T S \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^T\mathbf{w} = 1$$

**Lagrange multiplier** $\lambda$를 도입한다:

$$\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T S \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

왜 Lagrange multiplier를 사용하는가? 제약 조건이 있는 최적화에서 제약을 목적 함수에 통합하기 위해서이다.

$\mathbf{w}$에 대해 미분하고 0으로 놓는다:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2S\mathbf{w} - 2\lambda\mathbf{w} = 0$$

$$\boxed{S\mathbf{w} = \lambda\mathbf{w}}$$

이것은 **고유값 방정식(eigenvalue equation)**이다! $\mathbf{w}$는 $S$의 고유벡터이고, $\lambda$는 대응하는 고유값이다.

최대 분산을 구하려면:
$$\mathbf{w}^T S \mathbf{w} = \mathbf{w}^T \lambda \mathbf{w} = \lambda \underbrace{\mathbf{w}^T\mathbf{w}}_{=1} = \lambda$$

따라서 분산을 최대화하려면 **가장 큰 고유값** $\lambda_1$에 대응하는 **고유벡터** $\mathbf{w}_1$을 선택해야 한다.

$$\boxed{\text{제1주성분 } \mathbf{w}_1 = \text{arg\,eig}(S, \lambda_{\max})}$$

두 번째 주성분은 $\mathbf{w}_1$에 직교하는 조건 하에서 두 번째로 큰 고유값의 고유벡터이다. 이를 반복하면 PCA의 전체 주성분을 얻는다.

**자주 하는 실수:**
- (a)에서 null space의 기저를 구체적으로 쓰지 않는 것
- (b)에서 Eckart-Young theorem을 서술하지 않고 SVD 정의만 쓰는 것
- (c)에서 Lagrange multiplier를 적용하지 않고 결과만 쓰는 것
- PCA 결과가 "가장 큰 고유값의 고유벡터"임을 명시하지 않는 것

**채점 기준:**
- (a) Rank, Null space, Rank-Nullity 검증: 3pts
- (b) SVD 설명, rank-k 근사, Eckart-Young: 3pts
- (c) Lagrange multiplier → eigenvalue equation → top eigenvector: 4pts

---

# Part IV: Integrated Professor-Style Questions (통합형, Problems 16–20)

---

## Problem 16. (10pts)

**Question (English):**
Let $f:\mathbb{R}^d \to \mathbb{R}$ be a differentiable scalar-valued function and let $g:\mathbb{R}^d \to \mathbb{R}^m$ be a differentiable vector-valued function.

(a) Starting from Taylor expansion, derive the first-order linear approximation
$$
f(\mathbf{x}+\Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \Delta\mathbf{x}.
$$

(b) Extend this idea to the vector-valued case and show that
$$
g(\mathbf{x}+\Delta\mathbf{x}) \approx g(\mathbf{x}) + J_g(\mathbf{x})\Delta\mathbf{x},
$$
where $J_g(\mathbf{x})$ is the Jacobian.

(c) Let $\mathbf{p}=\text{softmax}(\mathbf{z})$. Use the Jacobian of softmax to write the local linear approximation of $\mathbf{p}(\mathbf{z}+\Delta\mathbf{z})$. Interpret the result: if one logit $z_j$ is perturbed upward, what happens to the other class probabilities and why?

(d) Explain why this “complex function $\approx$ local linear map” viewpoint is fundamental for backpropagation and optimization.

**출제 의도:**
복잡한 함수도 한 점 근처에서는 선형으로 근사된다는 사실은 미분, Jacobian, backpropagation을 하나의 그림으로 묶어 준다. 교수님 스타일에서는 결과 공식만 쓰는 것이 아니라, 왜 Taylor 1차항이 “가장 좋은 국소 선형 근사”인지와 그것이 딥러닝 계산에서 왜 중요한지를 설명해야 한다.

**정답 작성의 핵심 포인트:**
- Taylor expansion에서 1차항이 선형 근사를 준다는 점
- 스칼라 함수에서는 gradient, 벡터 함수에서는 Jacobian이 선형화 행렬 역할
- Softmax의 Jacobian $J=\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top$ 대입
- 한 클래스 확률이 증가하면 정규화 때문에 다른 클래스 확률이 감소한다는 해석
- Backpropagation은 국소 선형 근사들의 연쇄라는 관점

**모범 풀이:**

**Step 1: 스칼라 함수의 1차 Taylor 전개**

미분가능한 스칼라 함수 $f:\mathbb{R}^d \to \mathbb{R}$에 대해, 점 $\mathbf{x}$ 근처에서의 1차 Taylor 전개는
$$
f(\mathbf{x}+\Delta\mathbf{x})
= f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \Delta\mathbf{x} + o(\|\Delta\mathbf{x}\|).
$$

여기서 $o(\|\Delta\mathbf{x}\|)$는 $\|\Delta\mathbf{x}\|\to 0$일 때 그보다 더 빨리 작아지는 항이다. 따라서 아주 작은 변화에 대해서는 이 항을 무시할 수 있으므로,
$$
\boxed{f(\mathbf{x}+\Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \Delta\mathbf{x}}
$$
가 된다.

왜 이것이 중요한가? 복잡한 비선형 함수도 한 점 근처에서는 “상수항 + 선형항”으로 보이기 때문이다. 즉, gradient는 단순히 편미분의 모음이 아니라, 함수의 국소적 변화율을 가장 잘 설명하는 선형 계수 벡터이다.

**Step 2: 왜 gradient가 선형 근사의 계수인가**

$\Delta\mathbf{x}$의 각 성분이 조금씩 바뀔 때, 함수값의 1차 변화량은
$$
\Delta f \approx \sum_{i=1}^{d}\frac{\partial f}{\partial x_i}\Delta x_i
$$
로 주어진다. 이것을 벡터 형태로 쓴 것이
$$
\Delta f \approx \nabla f(\mathbf{x})^\top \Delta\mathbf{x}
$$
이다. 따라서 gradient는 “입력의 작은 변화가 출력에 어떻게 선형적으로 반영되는가”를 나타내는 객체이다.

**Step 3: 벡터 함수의 경우 Jacobian**

이제 $g(\mathbf{x}) = [g_1(\mathbf{x}),\ldots,g_m(\mathbf{x})]^\top \in \mathbb{R}^m$라 하자. 각 성분 $g_i$에 대해 위 스칼라 결과를 적용하면
$$
g_i(\mathbf{x}+\Delta\mathbf{x})
\approx g_i(\mathbf{x}) + \nabla g_i(\mathbf{x})^\top \Delta\mathbf{x}.
$$

이를 $i=1,\ldots,m$에 대해 쌓아 쓰면
$$
g(\mathbf{x}+\Delta\mathbf{x})
\approx g(\mathbf{x}) + 
\begin{bmatrix}
\nabla g_1(\mathbf{x})^\top\\
\vdots\\
\nabla g_m(\mathbf{x})^\top
\end{bmatrix}
\Delta\mathbf{x}.
$$

대괄호 안의 행렬이 바로 Jacobian이다:
$$
J_g(\mathbf{x}) \in \mathbb{R}^{m\times d},\qquad (J_g)_{ij}=\frac{\partial g_i}{\partial x_j}.
$$
따라서
$$
\boxed{g(\mathbf{x}+\Delta\mathbf{x}) \approx g(\mathbf{x}) + J_g(\mathbf{x})\Delta\mathbf{x}}
$$
가 된다.

즉, Jacobian은 벡터값 함수의 국소 선형 근사를 주는 행렬이다.

**Step 4: Softmax의 국소 선형 근사**

$\mathbf{p}=\text{softmax}(\mathbf{z})$이면 softmax Jacobian은
$$
J_{\text{softmax}}(\mathbf{z}) = \mathrm{diag}(\mathbf p) - \mathbf p\mathbf p^\top
$$
이다. 따라서 $\Delta\mathbf{z}$만큼 logits가 변할 때,
$$
\mathbf{p}(\mathbf{z}+\Delta\mathbf{z})
\approx \mathbf{p}(\mathbf{z}) + \left(\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top\right)\Delta\mathbf{z}.
$$

즉,
$$
\boxed{\mathbf{p}(\mathbf{z}+\Delta\mathbf{z})
\approx \mathbf{p} + J_{\text{softmax}}\,\Delta\mathbf{z}}
$$
이다.

**Step 5: 해석**

만약 $z_j$만 조금 증가시킨다면, $p_j$는 증가하려는 경향이 있다. 실제로
$$
\frac{\partial p_j}{\partial z_j}=p_j(1-p_j)>0.
$$
하지만 다른 클래스 $i\neq j$에 대해서는
$$
\frac{\partial p_i}{\partial z_j}=-p_ip_j<0
$$
이므로, 나머지 확률들은 감소한다.

왜 이런 일이 생기는가? softmax 출력은 확률분포이므로 항상 합이 1이어야 한다. 따라서 한 클래스에 확률 질량이 더 가면, 정규화 제약 때문에 다른 클래스에서 그만큼 빠져나와야 한다. 이것이 softmax Jacobian의 비대각 성분이 음수인 이유이다.

**Step 6: 왜 backpropagation과 optimization에 중요한가**

딥러닝의 각 레이어는 일반적으로 비선형 함수이지만, 한 번의 forward pass에서 현재 활성값 근처에서는 Jacobian으로 선형 근사할 수 있다. Backpropagation은 바로 이 “국소 선형 지도(local linear map)”들을 chain rule로 연쇄 곱하여 전체 gradient를 계산하는 과정이다.

즉,
- 복잡한 전체 함수는 전역적으로 비선형이지만,
- 각 작은 구간에서는 선형 근사 가능하고,
- 그 선형 근사들의 곱이 gradient 전달을 만든다.

따라서 Taylor 1차 근사와 Jacobian은 backpropagation의 수학적 핵심이다.

**자주 하는 실수:**
- Taylor 전개를 단순 암기식으로 쓰고, 왜 선형 근사인지 설명하지 않는 것
- Jacobian의 차원 $m\times d$를 잘못 쓰는 것
- softmax에서 한 logit이 증가할 때 다른 확률이 왜 감소하는지 정규화 관점으로 해석하지 않는 것

**채점 기준:**
- (a) 스칼라 Taylor 1차 근사 유도: 3pts
- (b) Jacobian을 이용한 벡터 선형화: 3pts
- (c) Softmax Jacobian 대입 및 해석: 3pts
- (d) Backpropagation/optimization 연결: 1pt

---

## Problem 17. (10pts)

**Question (English):**
Suppose you observe only two coin flips and both are heads.

(a) Derive the Bernoulli MLE of $\theta$.

(b) Using a Beta$(2,2)$ prior, derive the MAP estimate.

(c) Explain carefully why the MLE obtained in part (a) reveals a limitation of MLE in small-data settings.

(d) Rewrite the MAP objective in the form “data-fit term + prior term,” and explain how this connects to regularization and generalization.

**출제 의도:**
MLE의 한계는 큰 데이터에서는 잘 드러나지 않지만, 작은 데이터에서는 매우 선명하게 드러난다. 교수님 스타일에서는 단순히 “MLE=1, MAP=3/4” 같은 숫자만 맞히는 것이 아니라, 왜 MLE가 과도하게 확신하는지, prior가 어떤 철학적 역할을 하는지, regularization과 generalization이 어떻게 연결되는지 설명해야 한다.

**정답 작성의 핵심 포인트:**
- 두 번 모두 앞면이면 MLE는 $\theta=1$
- Beta$(2,2)$ prior 하에서 MAP는 $(k+\alpha-1)/(n+\alpha+\beta-2)=3/4$
- 작은 데이터에서 MLE는 극단값을 주고, 보지 못한 사건에 확률 0을 줄 수 있음
- MAP = NLL + regularizer 관점으로 해석
- prior가 과적합을 줄이고 일반화를 돕는다는 설명

**모범 풀이:**

**Step 1: 데이터와 likelihood 작성**

관측은 $D=\{1,1\}$이므로 $n=2$, $k=2$이다. Bernoulli 모델에서
$$
P(X_i=x_i\mid \theta)=\theta^{x_i}(1-\theta)^{1-x_i}.
$$

i.i.d. 가정을 쓰면:
- **independence**: joint likelihood를 곱으로 분해할 수 있다.
- **identically distributed**: 두 시행 모두 같은 $\theta$를 공유한다.

따라서
$$
L(\theta)=P(D\mid\theta)=\theta^2.
$$

**Step 2: MLE 도출**

MLE는 likelihood를 최대화하는 값이다:
$$
\hat\theta_{\text{MLE}}=\arg\max_{\theta\in[0,1]}\theta^2.
$$

$\theta^2$는 $[0,1]$에서 증가함수이므로 최대값은 경계점 $\theta=1$에서 달성된다:
$$
\boxed{\hat\theta_{\text{MLE}}=1.}
$$

이 결과는 “관측한 두 번 모두 앞면이었으니, 앞면 확률은 100%라고 추정한다”는 뜻이다.

**Step 3: MAP 도출**

Beta$(2,2)$ prior는
$$
p(\theta)\propto \theta^{2-1}(1-\theta)^{2-1}=\theta(1-\theta)
$$
이다.

posterior는
$$
p(\theta\mid D)\propto p(D\mid\theta)p(\theta)\propto \theta^2\cdot\theta(1-\theta)=\theta^3(1-\theta)
$$
이므로, log-posterior를 최대화하면
$$
\log p(\theta\mid D)=3\log\theta+\log(1-\theta)+\text{const}.
$$

미분하여 0으로 놓는다:
$$
\frac{d}{d\theta}\log p(\theta\mid D)=\frac{3}{\theta}-\frac{1}{1-\theta}=0.
$$
따라서
$$
3(1-\theta)=\theta
\quad\Rightarrow\quad
3=4\theta
\quad\Rightarrow\quad
\boxed{\hat\theta_{\text{MAP}}=\frac34.}
$$

일반 공식으로도
$$
\hat\theta_{\text{MAP}}=\frac{k+\alpha-1}{n+\alpha+\beta-2}
=\frac{2+2-1}{2+2+2-2}=\frac34
$$
를 얻는다.

**Step 4: 왜 MLE가 작은 데이터에서 한계를 드러내는가**

MLE는 오직 데이터만 본다. 지금 데이터가 두 번 모두 앞면이므로 $\theta=1$이 데이터에 가장 잘 맞는다. 하지만 이 값은 매우 위험하다.

왜냐하면 $\hat\theta_{\text{MLE}}=1$은 다음을 뜻하기 때문이다:
- 뒷면이 나올 확률을 정확히 0으로 둔다.
- 아직 보지 못한 사건(tail)에 대해 절대 불가능하다고 단정한다.
- 표본 수가 겨우 2개인데도 지나치게 확신(overconfidence)한다.

이것이 MLE의 작은-데이터 한계이다. 데이터가 적으면 empirical distribution이 매우 불안정한데, MLE는 그 empirical distribution을 그대로 밀어붙여 극단적인 추정을 만들 수 있다.

즉, MLE는 **variance가 크고** 과적합되기 쉽다. 이는 “훈련 데이터에는 완벽히 맞지만, 미래 데이터에는 약할 수 있다”는 일반화 문제와 연결된다.

**Step 5: MAP를 data-fit + prior로 다시 쓰기**

MAP는
$$
\hat\theta_{\text{MAP}}
=\arg\max_\theta \big[\log p(D\mid\theta)+\log p(\theta)\big]
$$
와 같다.

최대화를 최소화 문제로 바꾸면
$$
\hat\theta_{\text{MAP}}
=\arg\min_\theta \big[-\log p(D\mid\theta)-\log p(\theta)\big].
$$

여기서
- $-\log p(D\mid\theta)$는 **data-fit term** 또는 NLL,
- $-\log p(\theta)$는 **prior term** 또는 regularization term
으로 볼 수 있다.

즉,
$$
\boxed{\text{MAP}=\text{MLE}+\text{prior penalty}}
$$
라는 구조가 나온다.

이 prior term의 역할은 “극단적 파라미터를 덜 선호하게 만드는 것”이다. Beta$(2,2)$ prior는 $\theta=0.5$ 근처를 선호하므로, $\theta=1$이라는 극단값을 $\theta=3/4$ 쪽으로 끌어당긴다.

이것이 regularization과 generalization의 연결이다:
- regularization은 파라미터 공간을 제한하고,
- prior는 그 제한을 확률적으로 표현하며,
- 결과적으로 작은 데이터에서도 더 안정적인 추정을 하게 하여 generalization을 개선한다.

**자주 하는 실수:**
- MLE가 왜 문제인지 “작은 데이터라서 안 좋다” 정도로만 쓰고, 확률 0/1의 과도한 확신 문제를 지적하지 않는 것
- MAP 공식을 쓰고도 prior term을 regularization과 연결하지 않는 것
- MLE와 MAP의 차이를 단순히 “MAP는 prior를 더한다”로만 쓰고, 그 의미를 해석하지 않는 것

**채점 기준:**
- (a) MLE 도출: 2pts
- (b) MAP 도출: 3pts
- (c) small-data MLE limitation 해석: 3pts
- (d) regularization/generalization/prior 연결: 2pts

---

## Problem 18. (10pts)

**Question (English):**
Let a dataset contain $n$ i.i.d. class labels from $K$ categories. Let the empirical class distribution be $\hat p_{\text{data}}(k)=\frac{n_k}{n}$ and let the model distribution be $q_\theta(k)$.

(a) Show that the empirical negative log-likelihood can be written as
$$
-\frac1n\sum_{i=1}^n \log q_\theta(y_i)
= -\sum_{k=1}^K \hat p_{\text{data}}(k)\log q_\theta(k).
$$

(b) Show that this is exactly the cross-entropy $H(\hat p_{\text{data}}, q_\theta)$.

(c) Use the identity
$$
H(\hat p_{\text{data}}, q_\theta)=H(\hat p_{\text{data}})+D_{\mathrm{KL}}(\hat p_{\text{data}}\|q_\theta)
$$
to explain why minimizing empirical cross-entropy is equivalent to minimizing KL divergence.

(d) Explain where independence and identical distribution are used, and then explain how the objective changes when we move from MLE to MAP.

**출제 의도:**
Loss, empirical distribution, KL divergence, MLE, MAP를 하나의 수식 체계 안에서 연결하는 문제다. 교수님 스타일에서는 “CE를 쓴다”가 아니라, 왜 empirical NLL이 CE가 되고, 왜 그것이 KL 최소화와 같은지, 그리고 prior를 넣으면 무엇이 추가되는지까지 설명할 수 있어야 한다.

**정답 작성의 핵심 포인트:**
- empirical NLL를 class-count 형태로 재정렬
- 그것이 empirical cross-entropy와 동일함을 명시
- $H(\hat p)$는 $\theta$와 무관한 상수이므로 CE 최소화 = KL 최소화
- independence는 joint→product, identical distribution은 shared $q_\theta$
- MAP에서는 $-(1/n)\log p(\theta)$ 항이 추가됨

**모범 풀이:**

**Step 1: empirical NLL를 쓰기**

데이터가 $y_1,\ldots,y_n$이고 각 $y_i\in\{1,\ldots,K\}$라 하자. i.i.d. 가정 아래 likelihood는
$$
P(D\mid\theta)=\prod_{i=1}^n q_\theta(y_i)
$$
이다.

여기서
- **independence**가 사용된 곳: joint probability를 곱으로 분해한 단계
- **identically distributed**가 사용된 곳: 모든 샘플이 같은 모델 분포 $q_\theta$를 공유하는 단계

따라서 empirical negative log-likelihood는
$$
-\frac1n\log P(D\mid\theta)
= -\frac1n\sum_{i=1}^n \log q_\theta(y_i)
$$
가 된다.

**Step 2: class count로 재정렬**

각 클래스 $k$가 $n_k$번 등장했으므로, 위 합은 같은 항을 묶어서
$$
-\frac1n\sum_{i=1}^n \log q_\theta(y_i)
= -\frac1n\sum_{k=1}^K n_k \log q_\theta(k)
$$
로 쓸 수 있다.

그런데 empirical distribution은
$$
\hat p_{\text{data}}(k)=\frac{n_k}{n}
$$
이므로,
$$
\boxed{
-\frac1n\sum_{i=1}^n \log q_\theta(y_i)
= -\sum_{k=1}^K \hat p_{\text{data}}(k)\log q_\theta(k)
}
$$
이다.

**Step 3: cross-entropy와의 동일성**

cross-entropy의 정의는
$$
H(P,Q)=-\sum_k P(k)\log Q(k)
$$
이다. 따라서
$$
\boxed{
-\frac1n\sum_{i=1}^n \log q_\theta(y_i)
= H(\hat p_{\text{data}}, q_\theta)
}
$$
가 된다.

즉 empirical NLL = empirical CE 이다.

이것이 분류에서 cross-entropy loss를 쓰는 확률적 이유이다. CE는 ad hoc한 함수가 아니라 categorical likelihood의 NLL이다.

**Step 4: KL divergence와의 연결**

항등식
$$
H(\hat p_{\text{data}}, q_\theta)
=H(\hat p_{\text{data}})+D_{\mathrm{KL}}(\hat p_{\text{data}}\|q_\theta)
$$
를 사용하자.

여기서 $H(\hat p_{\text{data}})$는 데이터가 정해지면 고정된 값이며, $\theta$에 의존하지 않는다. 따라서
$$
\arg\min_\theta H(\hat p_{\text{data}}, q_\theta)
=\arg\min_\theta D_{\mathrm{KL}}(\hat p_{\text{data}}\|q_\theta).
$$

즉,
$$
\boxed{\text{empirical CE 최소화}=\text{empirical KL 최소화}}
$$
이다.

해석적으로는, 모델 분포 $q_\theta$를 empirical distribution $\hat p_{\text{data}}$에 최대한 가깝게 만드는 것이 곧 학습이다.

**Step 5: ERM, MLE, MAP의 연결**

empirical NLL를 최소화하는 것은 경험적 위험 최소화(ERM)의 특수한 경우이다:
$$
\hat R_n(\theta)=\frac1n\sum_{i=1}^n \ell(y_i,\theta),\qquad
\ell(y_i,\theta)=-\log q_\theta(y_i).
$$

따라서
$$
\text{ERM with } \ell=-\log q_\theta
\quad\Longleftrightarrow\quad
\text{MLE}.
$$

MAP로 가면 posterior를 최대화하므로 목적 함수가
$$
\hat\theta_{\text{MAP}}
=\arg\min_\theta\left[
-\frac1n\sum_{i=1}^n\log q_\theta(y_i)
-\frac1n\log p(\theta)
\right]
$$
가 된다.

즉,
$$
\boxed{\text{MAP}=\text{empirical CE}+\text{prior penalty}}
$$
이다.

MLE는 데이터를 맞추는 데만 집중하지만, MAP는 prior를 통해 “어떤 모델을 더 선호하는가”를 함께 반영한다. 이것이 regularization의 확률적 해석이다.

**자주 하는 실수:**
- empirical distribution $\hat p_{\text{data}}$를 정의하지 않고 곧바로 CE라고 쓰는 것
- independence와 identical distribution의 역할을 구분하지 않는 것
- $H(\hat p_{\text{data}})$가 상수라는 이유를 설명하지 않고 CE 최소화 = KL 최소화라고만 쓰는 것

**채점 기준:**
- (a) empirical NLL 재정렬: 3pts
- (b) CE와 동일함을 보이기: 2pts
- (c) KL 최소화와의 연결: 3pts
- (d) i.i.d.의 역할 + MAP 확장 설명: 2pts

---

## Problem 19. (10pts)

**Question (English):**
You must choose a model class for each of the following three situations:

1. A tabular regression task with 20 features and 300 samples.
2. An image classification task with 5,000 labeled images and strong local spatial patterns.
3. A large-scale multimodal pretraining task with 500 million image-text pairs and long-range dependencies.

For each case:

(a) Choose the most appropriate architecture among a linear model, CNN, and Transformer.

(b) Explain the choice in terms of inductive bias, prior, model capacity, regularization, and generalization.

(c) Explain why architecture itself can be viewed as an implicit prior, and compare this implicit prior with an explicit probabilistic prior in MAP estimation.

(d) State one situation in which the chosen architecture could still fail.

**출제 의도:**
교수님은 아키텍처를 단순한 공학적 선택이 아니라 “prior가 구현된 형태”로 본다. 이 문제는 inductive bias, prior, regularization, generalization, data size, task difficulty를 실제 모델 선택 문제 안에 통합해서 설명할 수 있는지 평가한다.

**정답 작성의 핵심 포인트:**
- 작은 표본/낮은 복잡도: linear model
- 중간 데이터/공간적 구조 강함: CNN
- 초대규모/장거리 상호작용: Transformer
- architecture = implicit prior, MAP prior = explicit prior
- bias가 강하면 데이터 적을 때 유리하지만 잘못되면 underfitting 가능

**모범 풀이:**

**Step 1: Case 1 — 20 features, 300 samples**

가장 적절한 선택은 **linear model**이다.

이유:
- 데이터 수가 매우 적다.
- 입력 차원도 낮고, 문제 복잡도도 상대적으로 낮다.
- 이 상황에서는 너무 유연한 모델을 쓰면 variance가 커져 과적합 위험이 높다.

linear model의 inductive bias는 “입출력 관계가 대체로 선형이거나, 적어도 선형 근사로 충분하다”는 가정이다. 이는 매우 강한 bias이지만, 작은 데이터에서는 오히려 장점이 된다. 가설 공간이 작기 때문에 일반화가 쉬워진다.

MAP 관점에서 보면 이것은 “선형 함수에 높은 prior를 두는 것”과 비슷하다. explicit prior를 두지 않아도, architecture 자체가 가능한 함수 공간을 강하게 제한하므로 implicit prior로 작동한다.

실패 가능성:
- 실제 관계가 고도로 비선형이면 underfitting이 발생할 수 있다.

**Step 2: Case 2 — 5,000 labeled images, strong local spatial patterns**

가장 적절한 선택은 **CNN**이다.

이유:
- 이미지에는 locality가 있다. 인접 픽셀 간 관계가 중요하다.
- 같은 edge나 texture는 이미지 위치가 달라도 같은 패턴이다.
- CNN의 weight sharing과 local receptive field는 이 구조를 정확히 반영한다.

즉 CNN의 inductive bias는 다음과 같다:
- local pattern이 중요하다
- translation equivariance가 중요하다
- 저수준 특징이 고수준 특징으로 계층적으로 조합된다

5,000장은 아주 큰 데이터는 아니므로, 이러한 구조적 bias가 큰 도움이 된다. 완전히 자유로운 Transformer보다 CNN이 더 적은 데이터로도 좋은 일반화를 얻기 쉽다.

MAP 관점에서는 “이동 공유 필터를 사용하는 함수들”에 높은 prior를 둔 것과 같다. 여기에 explicit regularization(L2, dropout)을 추가하면 prior가 더 강해진다.

실패 가능성:
- 전역적 장거리 상호작용이 매우 중요한 문제에서는 CNN이 비효율적일 수 있다.

**Step 3: Case 3 — 500 million image-text pairs, long-range dependencies**

가장 적절한 선택은 **Transformer**이다.

이유:
- 데이터가 압도적으로 많다.
- 이미지와 텍스트의 장거리 상호작용, cross-modal alignment, flexible context modeling이 중요하다.
- 이 경우 강한 구조적 bias보다 높은 표현력이 더 중요하다.

Transformer는 CNN보다 약한 inductive bias를 갖는다. 이는 적은 데이터에서는 단점이지만, 초대규모 데이터에서는 장점이 된다. 데이터 자체가 prior 역할을 상당 부분 대체하므로, 더 유연한 모델이 복잡한 패턴을 직접 학습할 수 있다.

즉, 데이터가 많아질수록 likelihood가 prior보다 지배적이 된다는 Bayesian 직관과도 맞는다. 큰 데이터에서는 약한 prior여도 posterior가 데이터에 의해 잘 형성된다.

실패 가능성:
- 계산 비용과 데이터 요구량이 매우 크므로, 충분한 자원이 없으면 비효율적이다.

**Step 4: architecture = implicit prior**

architecture는 가능한 함수 공간을 제한한다.
- linear model: 선형 함수만 허용
- CNN: locality와 weight sharing을 만족하는 함수 선호
- Transformer: 전역 상호작용을 허용하는 넓은 함수 공간

따라서 architecture choice 자체가 “어떤 함수가 더 그럴듯한가”에 대한 prior를 내장한다. 이것이 implicit prior이다.

반면 MAP의 explicit prior는
$$
\hat\theta_{\text{MAP}}=\arg\max_\theta[\log p(D\mid\theta)+\log p(\theta)]
$$
에서처럼 파라미터 분포 $p(\theta)$를 직접 지정하는 것이다.

비교하면:
- **implicit prior**: architecture, optimizer, initialization 등으로 간접 구현
- **explicit prior**: Gaussian prior, Laplace prior처럼 수식으로 직접 명시

둘 다 generalization에 영향을 준다. 실제 딥러닝에서는 이 둘이 함께 작동한다.

**Step 5: regularization과 generalization**

regularization은 prior의 효과를 강화하는 도구이다.
- L2 regularization: 작은 가중치를 선호하는 Gaussian prior
- dropout: 특정 co-adaptation을 억제하는 구조적 prior에 가까운 효과
- data augmentation: 불변성(invariance)을 강제하는 prior

핵심은, 좋은 generalization은 “데이터 적합”만으로 얻어지지 않고, 적절한 bias/prior/regularization이 함께 있어야 한다는 점이다.

**자주 하는 실수:**
- architecture 선택을 단순히 “요즘 잘 되는 모델” 수준으로 답하는 것
- CNN의 bias를 locality와 weight sharing으로 구체화하지 않는 것
- architecture prior와 MAP prior를 서로 완전히 다른 개념으로 분리해서 쓰는 것

**채점 기준:**
- 세 상황에 대한 적절한 architecture 선택: 3pts
- inductive bias/data size/generalization 해석: 3pts
- architecture = implicit prior, MAP = explicit prior 비교: 3pts
- 실패 가능성 제시: 1pt

---

## Problem 20. (10pts)

**Question (English):**
Let $X \in \mathbb{R}^{n\times d}$ be a centered data matrix with singular value decomposition
$$
X=U\Sigma V^\top.
$$

(a) Write the rank-$k$ approximation $X_k$ and show that the squared reconstruction error is
$$
\|X-X_k\|_F^2=\sum_{i=k+1}^r \sigma_i^2.
$$

(b) Explain why PCA/SVD can be interpreted as keeping “signal” directions and discarding “noise” directions.

(c) Connect this interpretation to Gaussian noise and the CLT intuition used in the course.

(d) Let $P_k=V_kV_k^\top$ be the projection onto the top-$k$ principal subspace. State the rank and nullity of $P_k$, and explain how this connects to the Rank-Nullity theorem.

(e) Explain why low-rank approximation can improve generalization even if it may increase empirical reconstruction error.

**출제 의도:**
이 문제는 선형대수와 확률적 해석을 함께 묻는다. SVD/PCA를 단순 계산 기법이 아니라, 저차원 구조에 대한 inductive bias이자 노이즈 제거 메커니즘으로 해석할 수 있어야 한다. 또한 rank-nullity를 추상 정리가 아니라 “남기는 방향과 버리는 방향의 차원 보존 법칙”으로 이해하는지 평가한다.

**정답 작성의 핵심 포인트:**
- $X_k=U_k\Sigma_kV_k^\top$
- Frobenius 오차는 버린 특이값 제곱합
- PCA는 큰 분산 방향 유지, 작은 분산 방향 제거
- Gaussian noise/CLT 직관: 많은 작은 요인의 합이 대략 가우시안
- projection의 rank = k, nullity = d-k
- low-rank bias가 overfitting을 줄여 generalization 향상 가능

**모범 풀이:**

**Step 1: rank-$k$ 근사**

SVD가
$$
X=\sum_{i=1}^{r}\sigma_i\mathbf u_i\mathbf v_i^\top
$$
라고 하자. 여기서 $r=\mathrm{rank}(X)$이고 $\sigma_1\ge\cdots\ge\sigma_r>0$이다.

rank-$k$ 근사는 상위 $k$개의 특이값만 남겨서
$$
\boxed{
X_k=\sum_{i=1}^{k}\sigma_i\mathbf u_i\mathbf v_i^\top
=U_k\Sigma_kV_k^\top
}
$$
로 정의한다.

**Step 2: reconstruction error**

남은 잔차는
$$
X-X_k=\sum_{i=k+1}^{r}\sigma_i\mathbf u_i\mathbf v_i^\top
$$
이다.

SVD의 직교성 때문에 서로 다른 특이 성분들은 Frobenius inner product에 대해 직교하므로,
$$
\|X-X_k\|_F^2
=\sum_{i=k+1}^{r}\sigma_i^2.
$$

즉,
$$
\boxed{\|X-X_k\|_F^2=\sum_{i=k+1}^{r}\sigma_i^2}
$$
이다.

이 결과는 Eckart-Young theorem과 연결되며, rank-$k$ 행렬들 중에서 $X_k$가 가장 좋은 근사임을 뜻한다.

**Step 3: 왜 signal/noise 분리로 볼 수 있는가**

PCA/SVD는 데이터의 분산이 큰 방향을 먼저 정렬한다.
- 큰 특이값/고유값 방향: 데이터가 구조적으로 많이 변하는 방향
- 작은 특이값/고유값 방향: 변동이 작고, 종종 noise가 지배하는 방향

따라서 상위 $k$개 방향만 유지하는 것은 데이터의 핵심 구조(signal)를 남기고, 미세하고 불안정한 변동(noise)을 제거하는 것으로 해석할 수 있다.

물론 항상 “작은 분산 = noise”는 아니지만, 강의의 기본적 통계 모델에서는 이것이 유용한 근사이다.

**Step 4: Gaussian noise와 CLT 직관**

강의에서 Gaussian noise를 자주 가정하는 이유는 CLT 때문이다. 관측 오차가 많은 독립적인 작은 요인의 합이라면, 그 합은 대략 Gaussian이 된다.

즉 한 관측은
$$
\text{data}=\text{signal}+\text{noise}
$$
로 보고, noise는 대략 mean 0인 Gaussian perturbation으로 모델링할 수 있다.

이 관점에서 보면:
- 큰 분산 방향은 반복적으로 나타나는 구조적 signal일 가능성이 높고,
- 작은 분산 방향은 Gaussian noise에 의해 생긴 요동일 가능성이 높다.

따라서 PCA/SVD의 저랭크 근사는 Gaussian-like noise를 평균화하여 제거하는 일종의 denoising으로 볼 수 있다.

**Step 5: projection의 rank와 nullity**

$P_k=V_kV_k^\top$는 top-$k$ principal subspace로의 직교투영이다.

$V_k$의 열벡터가 $k$개 직교 기저이므로,
$$
\mathrm{rank}(P_k)=k.
$$

입력 공간이 $\mathbb R^d$이므로, Rank-Nullity theorem에 의해
$$
\mathrm{rank}(P_k)+\mathrm{nullity}(P_k)=d.
$$
따라서
$$
\boxed{\mathrm{nullity}(P_k)=d-k.}
$$

해석하면:
- rank $k$: 남겨지는 정보의 차원
- nullity $d-k$: 완전히 버려지는 방향의 차원

즉 rank-nullity는 “활용하는 방향 + 버리는 방향 = 전체 차원”이라는 보존 법칙이다.

**Step 6: 왜 training error가 조금 늘어도 generalization이 좋아질 수 있는가**

저랭크 근사는 표현력을 줄인다. 따라서 훈련 데이터의 모든 미세한 변동까지 완벽히 맞추지는 못할 수 있다. 즉 empirical reconstruction error는 full-rank보다 커질 수 있다.

하지만 바로 그 점 때문에 generalization이 좋아질 수 있다.
- full-rank 모델은 noise까지 따라가며 overfit할 수 있다.
- low-rank 모델은 핵심 구조만 남기고 noise 적합을 막는다.

이것은 regularization과 같은 논리이다. training error를 조금 희생해서 model complexity를 줄이면 test error가 더 좋아질 수 있다.

따라서 low-rank approximation은 단순 압축이 아니라, “데이터는 본질적으로 저차원 구조를 가진다”는 inductive bias 또는 prior로 해석할 수 있다.

**자주 하는 실수:**
- reconstruction error를 버린 특이값의 “합”으로만 쓰고 제곱합임을 빠뜨리는 것
- PCA를 단순 계산 알고리즘으로만 설명하고, noise 제거 해석을 하지 않는 것
- Rank-Nullity를 projection 행렬의 rank/nullity와 연결하지 못하는 것

**채점 기준:**
- (a) rank-$k$ 근사와 오차 공식: 3pts
- (b) signal/noise 해석: 2pts
- (c) Gaussian noise와 CLT 연결: 2pts
- (d) rank, nullity, Rank-Nullity 해석: 2pts
- (e) generalization 해석: 1pt

---

# Final Summary

## 1. 이번 20문제에서 가장 중요한 5개 주제

1. **Bayes' Theorem and the Bayesian learning view**  
   Prior, likelihood, posterior, evidence의 역할과 Frequentist와의 철학적 차이.

2. **MLE and MAP as optimization principles**  
   i.i.d. 가정, log-likelihood, 왜 미분을 0으로 두는지, 왜 그 해가 최적해인지, prior가 어떻게 regularization이 되는지.

3. **Loss as negative log-likelihood**  
   Categorical $\to$ cross-entropy, Gaussian $\to$ MSE, 그리고 empirical distribution / KL / CE / ERM의 연결.

4. **Inductive bias, prior, architecture, and generalization**  
   linear model, CNN, Transformer가 어떤 prior를 구현하며, 데이터 양과 문제 난이도에 따라 왜 선택이 달라지는지.

5. **Linear algebra + calculus as the language of deep learning**  
   Taylor linear approximation, Jacobian, softmax derivative, SVD/PCA, Rank-Nullity가 모두 학습과 표현의 핵심 도구라는 점.

## 2. 학생들이 가장 많이 틀릴 5개 포인트

1. **independence와 identical distribution을 구분하지 않음**  
   joint를 product로 만드는 것은 independence, 같은 $\theta$를 공유하게 만드는 것은 identical distribution이다.

2. **왜 log를 취하는지 설명하지 못함**  
   곱을 합으로 바꾸고, 수치적으로 안정화하고, exp를 제거하기 위해 log를 취한다.

3. **왜 derivative를 0으로 두는지, 왜 최대인지 설명하지 못함**  
   1차 조건은 필요조건일 뿐이며, 2차 도함수나 concavity/convexity로 극대/극소를 확인해야 한다.

4. **cross-entropy와 KL의 관계를 암기로만 씀**  
   $H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q)$에서 $H(P)$가 상수이기 때문에 CE 최소화 = KL 최소화가 된다.

5. **prior를 단순 보정항 정도로만 이해함**  
   prior는 regularization의 확률적 의미이자, architecture까지 포함한 inductive bias의 더 넓은 개념과 연결된다.

## 3. 시험 직전 반드시 다시 써봐야 하는 유도 5개

1. **Bernoulli MLE 유도**  
   $L(\theta)=\theta^k(1-\theta)^{n-k}$, log, 미분, 2차 도함수 확인.

2. **Gaussian MLE 유도**  
   $\mu_{\mathrm{MLE}}=\bar x$, $\sigma^2_{\mathrm{MLE}}=\frac1n\sum (x_i-\bar x)^2$, 그리고 bias 증명.

3. **Gaussian likelihood $\to$ NLL $\to$ MSE 유도**  
   CLT와 Gaussian noise 가정까지 함께 설명할 것.

4. **MAP + Gaussian prior $\to$ L2 regularization 유도**  
   $\lambda=\frac{\sigma^2}{n\sigma_p^2}$까지 정확히 쓸 것.

5. **Softmax Jacobian 유도**  
   $i=j$, $i\neq j$를 나눠서 구하고 $J=\mathrm{diag}(p)-pp^\top$로 묶을 것.

## 4. 교수님 스타일 대비용 답안 작성 팁 7개

1. **수식 한 줄마다 “왜 이 단계가 가능한지”를 문장으로 붙여라.**

2. **가정을 먼저 선언하라.**  
   i.i.d., Gaussian noise, prior 형태, centered data 같은 가정을 먼저 쓰면 답안이 훨씬 강해진다.

3. **independence와 identical distribution의 사용 위치를 따로 표시하라.**

4. **상수 제거, log 변환, derivative = 0의 이유를 절대 생략하지 마라.**

5. **최종 숫자보다 해석을 써라.**  
   예를 들어 “MAP가 1/2 쪽으로 당겨진다”처럼 prior의 의미를 반드시 해석할 것.

6. **개념 문제에서도 비교 구조로 답하라.**  
   Frequentist vs Bayesian, MLE vs MAP, CNN vs Transformer처럼 대비해서 쓰면 채점 포인트가 명확해진다.

7. **마지막 한두 문장으로 딥러닝과 연결하라.**  
   “이 결과가 CE loss/regularization/backpropagation/generalization과 어떻게 연결되는가”를 적으면 교수님 스타일에 맞는다.

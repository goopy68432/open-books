---
title: "Deep Learning Theory — Midterm Examination"
slug: midterm-exam-05-professor-style
order: 5
---

# Deep Learning Theory — Midterm Examination

> **Instructor Style**: 논리적 전개 과정이 답보다 중요. 수식만 나열하면 0점.
> **All questions in English. All solutions in Korean.**
> **난이도**: 입문 20% (Q1-4) / 중급 40% (Q5-12) / 고급 40% (Q13-20)
> **Time**: 120 min | **Total**: 200 points

---

# PART A: Foundations — Concepts & Definitions (입문, 40점)

---

## Q1. (10pts) Defining Learning

**(a)** (5pts) Define "learning" in the context of machine learning using the formal notation: $H$ (hypothesis/model with parameters $\theta$), $E$ (experience/data $D$). Explain how the two components — a **model** and an **objective** — work together.

**(b)** (5pts) In the Bayesian framework, we write $\log p(H|E) = \log p(E|H) + \log p(H) + \text{const}$. Name each term (MAP, ML, Prior) and explain in 2-3 sentences what each represents and why this decomposition matters for deep learning.

---

### [모범 답안]

**(a)** 머신러닝에서 "학습(Learning)"이란, 경험(Experience) $E$를 통해 가설(Hypothesis) $H$의 성능을 개선하는 과정이다.

이를 구성하는 두 요소는 다음과 같다:

**① 모델(Model)**: 가설 공간 $\mathcal{H}$에서 함수를 선택하는 것이다. 모델은 파라미터 $\theta$로 정의되는 함수 $h_\theta : \mathcal{X} \to \mathcal{Y}$이며, 입력 $x$를 받아 출력(예측) $\hat{y}$를 생성한다.

**② 목적함수(Objective)**: 모델의 예측이 데이터를 얼마나 잘 설명하는지 측정하는 손실 함수 $\mathcal{L}$을 정의한다. 학습은 이 손실을 최소화하는 $\theta^*$를 찾는 **최적화 문제**로 귀결된다:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta; D)$$

두 요소가 함께 작동하는 방식: 모델이 가설의 형태(어떤 종류의 함수를 고려할지)를 결정하고, 목적함수가 "어떤 기준으로 좋은 가설인지"를 정량화한다. 학습 알고리즘(SGD 등)은 이 두 요소를 연결하여 데이터에 맞는 최적 파라미터를 찾는다.

**(b)** 베이즈 정리에서:

$$\underbrace{\log p(H|E)}_{\text{MAP}} = \underbrace{\log p(E|H)}_{\text{ML (Likelihood)}} + \underbrace{\log p(H)}_{\text{Prior}} + \text{const}$$

- **$\log p(H|E)$ = MAP (Maximum A Posteriori)**: 데이터 $E$를 관찰한 후 가설 $H$에 대한 **갱신된 믿음**이다. 이것을 최대화하는 $\theta$를 찾는 것이 MAP 추정이다.

- **$\log p(E|H)$ = ML (Maximum Likelihood)**: 가설 $H$(파라미터 $\theta$)가 맞다고 가정했을 때 이 데이터 $E$가 관측될 **가능성**이다. 데이터만 보고 판단하는 순수한 증거 항이다.

- **$\log p(H)$ = Prior**: 데이터를 보기 **전**에 가설에 대해 가지고 있는 사전 지식이다. 딥러닝에서 이것은 **정규화(Regularization)**에 해당하며, 모델이 너무 복잡해지는 것을 방지하여 **일반화(Generalization)**를 돕는다.

이 분해가 중요한 이유: 딥러닝의 모든 손실 함수(CE, MSE) + 정규화(L2, Dropout 등) 구조가 이 프레임워크의 특수한 경우임을 보여준다. **Loss = NLL (from ML) + Regularization (from Prior)**이라는 통합적 관점을 제공한다.

---

## Q2. (10pts) Three Types of Learning & i.i.d.

**(a)** (4pts) Briefly describe Supervised Learning, Unsupervised Learning, and Reinforcement Learning. For Supervised Learning, explain the roles of $x$ (input) and $y$ (label).

**(b)** (6pts) The i.i.d. (independent and identically distributed) assumption is fundamental. Separately explain what "independent" enables and what "identically distributed" enables in the mathematical formulation of learning. Give a concrete example where i.i.d. fails.

---

### [모범 답안]

**(a)**

- **지도학습(Supervised Learning)**: 입력-정답 쌍 $(x_i, y_i)$로 학습. $x$는 모델에 제공되는 입력(이미지, 텍스트 등), $y$는 그에 대한 정답 레이블(클래스, 수치 등). $x$만으로는 "무엇이 정답인지" 알 수 없으므로 $y$가 반드시 필요하다. 목표: $p(y|x)$를 학습.

- **비지도학습(Unsupervised Learning)**: 정답 $y$ 없이 $x$만으로 데이터의 구조(군집, 분포, 표현)를 발견. 목표: $p(x)$를 학습. 예: 생성 모델, 클러스터링.

- **강화학습(Reinforcement Learning)**: 에이전트가 환경과 상호작용하며, 누적 보상을 최대화하는 정책(policy)을 학습.

**(b)** i.i.d.의 두 요소를 **분리하여** 설명한다:

**"Independent" (독립)이 가능하게 하는 것:**

데이터 $\{(x_i, y_i)\}_{i=1}^n$이 독립이면, 결합 확률이 개별 확률의 **곱**으로 분해된다:

$$p(D|\theta) = \prod_{i=1}^{n} p(y_i | x_i, \theta)$$

이 분해가 가능해야 로그를 취했을 때 곱이 **합**으로 변환되어:

$$-\log p(D|\theta) = -\sum_{i=1}^{n} \log p(y_i | x_i, \theta)$$

이 합 구조가 있어야 SGD가 미니배치 단위로 그래디언트를 추정할 수 있다. **독립이 아니면 곱 분해 불가 → NLL이 깔끔한 합이 아님 → SGD 적용 어려움.**

**"Identically Distributed" (동일 분포)이 가능하게 하는 것:**

모든 데이터가 같은 분포에서 생성되었으므로, **하나의 $\theta$로 모든 데이터를 설명**할 수 있다. 만약 분포가 시간이나 상황에 따라 달라진다면($\theta_1, \theta_2, \ldots$), 단일 모델 $f_\theta$가 전체를 대표할 수 없다.

**i.i.d.가 실패하는 예**: 주식 가격 예측. 오늘의 가격 $y_t$는 어제의 가격 $y_{t-1}$에 강하게 의존(독립 위반). 또한 시장 환경이 시간에 따라 변하므로(동일분포 위반, concept drift), 과거 데이터로 학습한 모델이 미래에 무용지물이 될 수 있다.

---

## Q3. (10pts) Categorical vs. Gaussian — Two Supervised Learning Frameworks

**(a)** (5pts) The slide shows two cases:

| | Classification | Regression |
|---|---|---|
| Distribution | $p(y\|x,h) = h(x)_y$ (Categorical) | $p(y\|x,h) \propto e^{-c(y-h(x))^2}$ (Gaussian) |
| Loss | CE: $-\frac{1}{\|E\|}\sum_i e_{y_i}^\top \log h(x_i)$ | MSE: $\frac{1}{\|E\|}\sum_i (y_i - h(x_i))^2$ |

Explain the unifying principle that connects both: "both losses arise from the same principle of NLL minimization under different distributional assumptions."

**(b)** (5pts) A student proposes using MSE loss for a 10-class classification problem. Explain mathematically why this is problematic. (Hint: consider what happens to the gradient when $\sigma(z) \approx 0$ or $1$.)

---

### [모범 답안]

**(a)** 두 손실 함수는 **동일한 원리 — NLL(음의 로그 우도) 최소화 — 의 서로 다른 분포 가정에서 나온 특수 사례**이다.

통합 원리:

$$\text{Loss}(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \log p(y_i | x_i, \theta)$$

- **분류**: $y$가 이산적(클래스 레이블)이므로 **카테고리컬 분포**를 가정. $p(y_i|x_i,\theta) = h(x_i)_{y_i}$ (모델 출력의 $y_i$번째 원소). NLL에 대입하면: $-\frac{1}{n}\sum_i \log h(x_i)_{y_i}$. 원-핫 벡터 $e_{y_i}$를 사용하면 $-\frac{1}{n}\sum_i e_{y_i}^\top \log h(x_i)$ = **CE Loss**.

- **회귀**: $y$가 연속적이므로 **가우시안 분포**를 가정. $p(y_i|x_i,\theta) \propto \exp(-\frac{(y_i - h(x_i))^2}{2\sigma^2})$. NLL에 대입하면: $-\log$와 $\exp$가 상쇄되어 제곱항만 남고, 상수 제거 후 $\frac{1}{n}\sum_i (y_i - h(x_i))^2$ = **MSE Loss**.

핵심: **분포 가정이 손실 함수를 결정한다.** 같은 NLL 원리에서 출발하지만, 카테고리컬 → CE, 가우시안 → MSE로 갈라진다.

**(b)** 분류에 MSE를 쓰면, 소프트맥스(또는 시그모이드) 출력 $\hat{y} = \sigma(z)$에 대해 MSE 그래디언트는:

$$\frac{\partial L_{\text{MSE}}}{\partial z} = (\hat{y} - y) \cdot \sigma'(z) = (\hat{y} - y) \cdot \sigma(z)(1-\sigma(z))$$

반면 CE의 그래디언트는:

$$\frac{\partial L_{\text{CE}}}{\partial z} = \hat{y} - y$$

MSE에는 $\sigma(z)(1-\sigma(z))$ 항이 **추가 인자로 남는다**. 이 항의 최대값은 $1/4$ (at $\sigma = 0.5$)이며, $\sigma(z) \approx 0$ 또는 $\sigma(z) \approx 1$일 때 거의 0이 된다.

수학적 문제: 모델이 **확신 있게 틀린 예측**(예: $\hat{y} = 0.01$인데 $y = 1$)을 했을 때, $\sigma(z)(1-\sigma(z)) \approx 0.01 \times 0.99 \approx 0.01$. 그래디언트가 $(\hat{y}-y) \times 0.01$로 거의 사라진다. 교정이 극도로 느려진다.

CE에서는 같은 상황에서 그래디언트가 $\hat{y} - y \approx -0.99$로 **강한 교정 신호**를 준다.

결론: MSE는 시그모이드/소프트맥스의 포화 영역에서 학습이 정체되므로 분류에 부적합. CE는 canonical link function 덕분에 $\sigma'$ 항이 소거되어 이 문제가 없다.

---

## Q4. (10pts) Inductive Bias

**(a)** (4pts) Define "inductive bias." Explain its relationship to $p(H)$ (the prior) in the MAP framework.

**(b)** (6pts) For each of the following, describe the inductive bias in one sentence and rank them from strongest to weakest bias: **(i)** Linear Model, **(ii)** CNN, **(iii)** Transformer (ViT), **(iv)** MLP. Then explain: with only 100 labeled images, which would you choose and why? With 100 million images, which would you choose and why?

---

### [모범 답안]

**(a)** Inductive bias(귀납적 편향)란, 모델이 **데이터를 보기 전에** 아키텍처 자체에 내장하고 있는 **"어떤 종류의 함수가 정답일 가능성이 높다"는 사전 가정**이다.

MAP 프레임워크에서 prior $p(H)$는 가설 공간에서 특정 가설에 높은 사전확률을 부여하는 역할을 한다. Inductive bias는 이 prior를 **아키텍처 수준에서 구현**한 것이다. 예를 들어 CNN이 "특징은 국소적이고 이동 불변"이라고 가정하는 것은, 그런 성질을 만족하는 함수에 높은 prior를 부여하는 것과 동일하다.

강한 inductive bias = 강한 prior = 가설 공간을 좁게 제한 = 적은 데이터에서도 합리적 해를 찾을 수 있지만, 가정이 틀리면 성능 한계.

**(b)** 각 아키텍처의 inductive bias:

**(i) Linear Model**: "입출력 관계가 직선(초평면)이다." — **가장 강한 bias**.
**(ii) CNN**: "특징은 국소적(local)이고, 위치에 관계없이 동일한 필터로 감지 가능하다(translation equivariance)."
**(iii) MLP**: "특별한 구조 가정 없이, 충분한 폭이면 어떤 연속 함수든 근사 가능." — bias 약함.
**(iv) Transformer (ViT)**: "모든 위치 간의 관계가 중요할 수 있다. 위치 정보만 별도 제공." — **가장 약한 bias**.

**강도 순서**: Linear > CNN > MLP > Transformer

**100장**: **CNN** 추천. 데이터가 극도로 적으므로 이미지 도메인에 대한 사전 지식(국소성, 이동 불변)이 내장된 아키텍처가 필요하다. CNN의 강한 inductive bias가 적은 데이터에서도 합리적 특징을 학습하게 해준다. Transformer는 자유도가 너무 높아 100장으로는 과적합이 심하다.

**1억장**: **Transformer (ViT)** 추천. 데이터가 충분하면 CNN의 "국소적 패턴만 보라"는 가정이 오히려 제약이 된다. Transformer는 약한 bias 덕분에 데이터에서 최적의 표현을 자유롭게 학습한다. 실제로 ViT는 대규모 데이터에서 CNN을 능가한다.

핵심 원리: **데이터가 적을수록 강한 bias가 유리하고, 많을수록 약한 bias가 유리하다.** 이것은 bias-variance tradeoff의 아키텍처 수준 표현이다.

---

# PART B: Intermediate — Proofs, Derivations, Comparisons (중급, 80점)

---

## Q5. (10pts) From Bayes to MAP — Derivation with Full Justification

**[EN]** Starting from Bayes' theorem $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$:

**(a)** (3pts) Show that $\hat{\theta}_{\text{MAP}} = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]$. **At each step, state explicitly what you are doing and why it is mathematically valid.** (e.g., "We drop $p(D)$ because...")

**(b)** (4pts) Assuming the data $D = \{(x_i, y_i)\}_{i=1}^n$ are i.i.d., expand the log-likelihood into a sum. **State separately where the independence assumption is used and where the identical distribution assumption is used.**

**(c)** (3pts) Explain: in what sense is $-\log p(D|\theta)$ a "loss function"? Why do we minimize it? Connect this to the concept of Empirical Risk Minimization (ERM).

---

### [모범 답안]

**(a)** MAP 추정은 사후확률을 최대화하는 $\theta$를 찾는 것이다.

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta|D)$$

**[Step 1: Evidence 제거]** 베이즈 정리에 의해 $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$이다. 여기서 $p(D) = \int p(D|\theta)p(\theta)d\theta$는 모든 가능한 $\theta$에 대한 적분으로, **$\theta$에 의존하지 않는 상수**이다. $\arg\max$는 양의 상수배에 영향받지 않으므로 $p(D)$를 제거할 수 있다. 이것은 수학적으로 $\arg\max_\theta cf(\theta) = \arg\max_\theta f(\theta)$ (단, $c > 0$)이기 때문에 정당하다.

$$= \arg\max_\theta \; p(D|\theta) \cdot p(\theta)$$

**[Step 2: 로그 변환]** 양변에 로그를 취한다. 이 변환이 유효한 이유: 로그 함수는 **순단조 증가(strictly monotonically increasing)** 함수이므로, 함수값의 대소 관계를 보존한다. 즉, $f(a) > f(b) \iff \log f(a) > \log f(b)$. 따라서 $\arg\max$의 위치가 보존된다. 로그를 취하는 **실용적 이유**: (1) 곱이 합으로 변환되어 미분이 쉽다, (2) 매우 작은 확률값의 곱은 수치적 underflow를 유발하지만, 로그 합은 안정적이다.

$$= \arg\max_\theta \; [\log p(D|\theta) + \log p(\theta)]$$

이것은 슬라이드의 핵심 등식 $\log p(H|E) = \log p(E|H) + \log p(H) + \text{const}$와 정확히 대응한다.

**(b)** i.i.d.의 두 가정을 분리하여 적용한다.

**[Independent 사용]**: 각 데이터 포인트 $(x_i, y_i)$가 서로 독립이므로, 결합 우도를 개별 우도의 **곱**으로 분해할 수 있다:

$$p(D|\theta) = p(y_1, \ldots, y_n | x_1, \ldots, x_n, \theta) = \prod_{i=1}^{n} p(y_i | x_i, \theta)$$

이 단계에서 **독립 가정이 핵심적으로 사용**된다. 만약 $y_1$의 값이 $y_2$에 영향을 준다면 (예: 시계열), 이 곱 분해는 성립하지 않고 $p(y_1, y_2 | \ldots) \neq p(y_1|\ldots) \cdot p(y_2|\ldots)$가 된다.

**[Identically Distributed 사용]**: 모든 데이터가 **동일한 분포** $p(y|x, \theta)$에서 생성되었으므로, 같은 파라미터 $\theta$를 모든 $i$에 대해 공유할 수 있다. 만약 $i$번째 데이터의 생성 분포가 $p(y|x, \theta_i)$로 각각 다르다면, 단일 $\theta$로 전체를 설명하는 모델 자체가 불가능하다.

로그를 취하면 곱이 합으로 변환된다 ($\log\prod a_i = \sum\log a_i$):

$$\log p(D|\theta) = \sum_{i=1}^{n} \log p(y_i | x_i, \theta)$$

**(c)** $-\log p(D|\theta)$를 "손실 함수"라 부르는 이유:

- **우도가 높을수록 좋다**: $p(D|\theta)$가 크면 "이 $\theta$ 하에서 데이터가 그럴듯하다" → 좋은 모델.
- **최대화를 최소화로**: 관례적으로 "손실을 최소화"하므로 $-1$을 곱한다. $\arg\max p = \arg\min(-\log p)$.
- **NLL = 경험적 위험**: $-\frac{1}{n}\sum \log p(y_i|x_i,\theta) = \mathbb{E}_{(x,y)\sim\hat{p}}[-\log p(y|x,\theta)]$. 이것은 손실 $\ell = -\log p$를 경험적 분포 $\hat{p}$에 대해 평균낸 **경험적 위험(Empirical Risk)**이다. ERM 원칙: 가설 공간에서 경험적 위험을 최소화하는 가설 $h^* = \arg\min_{h \in \mathcal{H}} \hat{R}(h)$를 선택. NLL 최소화는 ERM의 특수 사례이다.

---

## Q6. (10pts) MAP with Three Different Priors

**[EN]** You flip a coin $n=5$ times and observe $k=4$ heads. The likelihood is $p(D|\theta) = \theta^4(1-\theta)^1$.

Compute $\hat{\theta}_{\text{MAP}}$ for each prior. **Show all differentiation steps and explicitly state why you set the derivative to zero.**

**(a)** (3pts) Uniform prior: $p(\theta) = 1$ for $\theta \in [0,1]$.
**(b)** (4pts) Beta prior: $p(\theta) \propto \theta(1-\theta)$ (i.e., $\text{Beta}(2,2)$).
**(c)** (3pts) Strong prior: $p(\theta) \propto \theta^m(1-\theta)^m$ as $m \to \infty$. What value does $\hat{\theta}_{\text{MAP}}$ approach?

---

### [모범 답안]

세 경우 모두 동일한 방법론을 따른다: (1) 로그 사후확률 쓰기, (2) 미분, (3) 0으로 놓기, (4) 풀기.

미분을 0으로 놓는 이유: 구간 $(0,1)$에서 로그 사후확률은 **위로 볼록(concave)**한 함수이다 (로그의 오목성). 볼록 함수의 최대점에서는 기울기가 0이 되므로, 1차 조건 $\frac{d}{d\theta}(\log p(\theta|D)) = 0$이 최대의 **필요충분조건**이다.

**(a) Uniform Prior**: $p(\theta) = 1$이므로 $\log p(\theta) = 0$ (상수).

로그 사후: $\log p(\theta|D) = 4\log\theta + 1\log(1-\theta) + C$

미분 ($\log\theta$의 미분은 $1/\theta$, $\log(1-\theta)$의 미분은 $-1/(1-\theta)$ — 체인룰):

$$\frac{d}{d\theta} = \frac{4}{\theta} - \frac{1}{1-\theta} = 0$$

$4(1-\theta) = \theta$ → $4 = 5\theta$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{4}{5} = 0.8 = \hat{\theta}_{\text{MLE}}}$$

Uniform prior는 "아무 사전 지식 없음"에 해당하므로 MAP = MLE가 된다.

**(b) Beta(2,2) Prior**: $p(\theta) \propto \theta^1(1-\theta)^1$이므로 $\log p(\theta) = \log\theta + \log(1-\theta) + C$.

로그 사후: $= (4+1)\log\theta + (1+1)\log(1-\theta) + C = 5\log\theta + 2\log(1-\theta) + C$

미분:

$$\frac{5}{\theta} - \frac{2}{1-\theta} = 0$$

$5(1-\theta) = 2\theta$ → $5 = 7\theta$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{5}{7} \approx 0.714}$$

0.8(MLE)보다 0.5에 가까워졌다. Beta(2,2) prior가 $\theta = 0.5$ (공정 동전) 쪽으로 끌어당기기 때문이다.

일반 공식 확인: $\hat{\theta}_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} = \frac{4+1}{5+2} = \frac{5}{7}$ ✓

**(c) Strong Prior** ($m \to \infty$): $p(\theta) \propto \theta^m(1-\theta)^m$

로그 사후: $(4+m)\log\theta + (1+m)\log(1-\theta) + C$

미분: $\frac{4+m}{\theta} - \frac{1+m}{1-\theta} = 0$

$(4+m)(1-\theta) = (1+m)\theta$ → $4+m = (5+2m)\theta$

$$\hat{\theta}_{\text{MAP}} = \frac{4+m}{5+2m}$$

$m \to \infty$이면:

$$\lim_{m \to \infty} \frac{4+m}{5+2m} = \lim_{m \to \infty} \frac{4/m + 1}{5/m + 2} = \frac{1}{2}$$

$$\boxed{\hat{\theta}_{\text{MAP}} \to 0.5 \quad (m \to \infty)}$$

$m$이 커질수록 prior가 $\theta = 0.5$에 극도로 집중 → 데이터(4/5 = 0.8)를 무시하고 prior의 최빈값 0.5로 수렴. **데이터보다 prior가 압도적으로 강하면, MAP은 prior의 mode를 따른다.** $\blacksquare$

---

## Q7. (10pts) Gaussian Noise → MSE: The Complete Chain

**[EN]** Starting from the regression model $y_i = f_\theta(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d.:

**(a)** (3pts) Using the Central Limit Theorem (CLT), justify why $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is a reasonable assumption. State the CLT precisely, including the three required conditions.

**(b)** (7pts) Derive step by step that $\arg\min_\theta \text{NLL}(\theta) = \arg\min_\theta \text{MSE}(\theta)$. **For every line of your derivation, write one sentence in natural language explaining what mathematical operation you performed and why it is valid.** Do NOT simply list equations.

---

### [모범 답안]

**(a)** **중심극한정리(CLT)**: $X_1, \ldots, X_n$이 (1) **독립**, (2) **동일 분포**, (3) **유한 분산** $\sigma^2 < \infty$을 만족하면:

$$\sqrt{n}\frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad (n \to \infty)$$

노이즈 $\epsilon$에 적용: 예측값 $f_\theta(x)$와 실제 $y$의 차이(노이즈)는 측정 오차, 환경 변동, 누락 변수 등 **수많은 미지 요인** $\delta_1, \delta_2, \ldots, \delta_m$의 합산이다. 각 요인이 (근사적으로) 독립이고 개별 영향이 크지 않으며 유한한 분산을 가지면, CLT에 의해 이들의 합 $\epsilon = \sum_j \delta_j$는 가우시안에 수렴한다. 따라서 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 가정은 이론적 근거가 있다.

**(b)** 가우시안 노이즈에서 NLL → MSE 유도:

**[Line 1]** $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$이고 $y_i = f_\theta(x_i) + \epsilon_i$이므로, $y_i$의 조건부 분포를 가우시안 밀도로 쓴다:

$$p(y_i|x_i,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

이것이 유효한 이유: $\epsilon_i = y_i - f_\theta(x_i)$이므로 $\epsilon_i$의 가우시안 분포를 $y_i$의 조건부 분포로 재해석한 것이다.

**[Line 2]** 데이터가 **i.i.d.**이므로, 전체 우도는 개별 우도의 **곱**으로 분해된다. 이것은 **독립 가정** 덕분에 가능하다:

$$p(D|\theta) = \prod_{i=1}^{n} p(y_i|x_i,\theta)$$

**[Line 3]** 양변에 로그를 취한다. **로그의 곱→합 성질** $\log\prod a_i = \sum\log a_i$를 사용한다. 로그를 취하는 이유는 곱을 합으로 바꿔 미분과 합산을 쉽게 하기 위함이다:

$$\log p(D|\theta) = \sum_{i=1}^{n}\left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right]$$

**[Line 4]** NLL(음의 로그 우도)로 부호를 바꾼다. 최대화 문제를 최소화 문제로 변환하기 위함이다:

$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i-f_\theta(x_i))^2$$

**[Line 5]** $\theta$에 대해 $\arg\min$을 구할 때, 첫째 항 $\frac{n}{2}\log(2\pi\sigma^2)$는 $\theta$를 **전혀 포함하지 않는 상수**이므로 최솟값의 위치에 영향을 주지 않아 제거한다:

$$\arg\min_\theta\text{NLL} = \arg\min_\theta \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i-f_\theta(x_i))^2$$

**[Line 6]** $\frac{1}{2\sigma^2}$는 **양의 상수**이다. 양의 상수를 곱해도 최솟값의 위치는 변하지 않으므로 제거한다:

$$= \arg\min_\theta \sum_{i=1}^{n}(y_i-f_\theta(x_i))^2 = \arg\min_\theta \; n \cdot \frac{1}{n}\sum_{i=1}^{n}(y_i-f_\theta(x_i))^2 = \arg\min_\theta \text{MSE}$$

$$\boxed{\text{가우시안 노이즈 가정 + NLL 최소화} = \text{MSE 최소화}}$$

인과관계 요약: CLT → 가우시안 노이즈 → 가우시안 우도 → $-\log(\exp(-z^2)) = z^2$ → 제곱 오차 → MSE. $\blacksquare$

---

## Q8. (10pts) Prior = L2 Regularization

**[EN]** If the prior is $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$:

**(a)** (4pts) Show that the MAP objective becomes $\text{MSE} + \lambda\|\theta\|^2$. **Derive the exact value of $\lambda$ in terms of $\sigma^2$ (noise variance), $\sigma_p^2$ (prior variance), and $n$ (number of data points).**

**(b)** (3pts) Interpret the formula $\lambda = \frac{\sigma^2}{n\sigma_p^2}$: What happens when the prior is very confident ($\sigma_p^2 \to 0$)? When data is abundant ($n \to \infty$)? When noise is large ($\sigma^2 \to \infty$)?

**(c)** (3pts) Explain why L2 regularization ($\lambda\|\theta\|^2$) helps generalization. Your answer must connect weight magnitudes to sensitivity to input perturbations.

---

### [모범 답안]

**(a)** $p(\theta) = \mathcal{N}(0, \sigma_p^2 I)$이면:

$$-\log p(\theta) = \frac{\|\theta\|^2}{2\sigma_p^2} + \frac{d}{2}\log(2\pi\sigma_p^2)$$

두 번째 항은 $\theta$에 무관한 상수이므로 $\arg\min$에서 제거.

MAP = NLL + Prior 항 (Problem 5에서 유도):

$$\arg\min_\theta\left[\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-f_\theta(x_i))^2 + \frac{1}{2\sigma_p^2}\|\theta\|^2\right]$$

$\frac{1}{2\sigma^2}$으로 양변을 나눈다 (양의 상수, $\arg\min$ 불변):

$$= \arg\min_\theta\left[\frac{1}{n}\sum_{i=1}^n(y_i-f_\theta(x_i))^2 + \frac{\sigma^2}{n\sigma_p^2}\|\theta\|^2\right]$$

$$= \arg\min_\theta\left[\text{MSE} + \lambda\|\theta\|^2\right], \quad \boxed{\lambda = \frac{\sigma^2}{n\sigma_p^2}}$$

**(b)** 해석:

| 상황 | $\lambda$ 변화 | 의미 |
|------|-------------|------|
| $\sigma_p^2 \to 0$ (강한 prior) | $\lambda \to \infty$ | "θ는 반드시 0 근처"라는 강한 믿음 → 정규화 극강 → 데이터 무시 |
| $n \to \infty$ (데이터 풍부) | $\lambda \to 0$ | 데이터가 충분 → prior의 필요성 감소 → MLE에 수렴 |
| $\sigma^2 \to \infty$ (노이즈 큼) | $\lambda \to \infty$ | 데이터가 매우 noisy → 데이터를 신뢰하기 어려움 → prior에 더 의존 |

핵심: $\lambda$는 "데이터를 얼마나 믿을지 vs prior를 얼마나 믿을지"의 균형을 조절한다.

**(c)** 가중치 크기와 입력 민감도의 관계:

선형 모델에서 $\hat{y} = \theta^\top x$이면, 입력 변화 $\Delta x$에 대한 출력 변화는:

$$\Delta\hat{y} = \theta^\top \Delta x \leq \|\theta\| \cdot \|\Delta x\|$$

(코시-슈바르츠 부등식). 따라서 **$\|\theta\|$가 크면 작은 $\Delta x$에도 출력이 크게 변한다** → 입력의 노이즈나 작은 변동에 과민 반응 → 훈련 데이터의 노이즈까지 학습(과적합).

L2 정규화로 $\|\theta\|$를 작게 유지하면 → 출력 변화가 입력 변화에 비례적으로 제한됨 → 부드러운(smooth) 함수 → 훈련/테스트 데이터 간 성능 차이가 작음 → **일반화 향상**. $\blacksquare$

---

## Q9. (10pts) MLE for Gaussian — Full Derivation

**[EN]** Given i.i.d. data $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$:

**(a)** (2pts) Write the log-likelihood $\ell(\mu, \sigma^2)$.
**(b)** (3pts) Derive $\hat{\mu}_{\text{MLE}} = \bar{x}$. State the differentiation rule used.
**(c)** (2pts) Derive $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_i(x_i - \bar{x})^2$.
**(d)** (3pts) Prove that $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$ (biased). **State separately where independence and identical distribution are used.**

---

### [모범 답안]

**(a)** 각 $x_i$는 $\mathcal{N}(\mu, \sigma^2)$를 따르므로:

$$p(x_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

**i.i.d.의 독립 가정**에 의해 곱 분해 후 로그를 취하면:

$$\ell(\mu,\sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2$$

**(b)** $\mu$에 대해 편미분한다 ($\sigma^2$는 상수 취급). $-\frac{1}{2\sigma^2}(x_i-\mu)^2$를 $\mu$에 대해 미분할 때, **체인룰**을 적용: 외부 $u^2$의 미분 $= 2u$, 내부 $u = x_i-\mu$의 $\mu$ 미분 $= -1$. 곱하면 $-\frac{1}{2\sigma^2} \cdot 2(x_i-\mu)(-1) = \frac{x_i-\mu}{\sigma^2}$.

$$\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_{i=1}^n(x_i-\mu)$$

MLE는 로그 우도를 최대화하는 것이므로 기울기 = 0으로 놓는다 (위로 볼록한 함수의 극대 조건):

$$\sum(x_i - \mu) = 0 \implies n\mu = \sum x_i \implies \boxed{\hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}}$$

**(c)** $s = \sigma^2$로 놓고 $s$에 대해 미분:

$$\frac{\partial\ell}{\partial s} = -\frac{n}{2s} + \frac{1}{2s^2}\sum(x_i-\mu)^2 = 0$$

$$ns = \sum(x_i-\mu)^2 \implies \boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(x_i-\bar{x})^2}$$

**(d)** 핵심 항등식을 사용한다:

$$\sum_{i=1}^n(x_i-\bar{x})^2 = \sum_{i=1}^n(x_i-\mu)^2 - n(\bar{x}-\mu)^2$$

기대값을 구한다:

$\mathbb{E}[\sum(x_i-\mu)^2] = \sum\mathbb{E}[(x_i-\mu)^2] = n\sigma^2$. 여기서 **동일 분포(identically distributed)** 가정 사용: 모든 $x_i$가 같은 분산 $\sigma^2$을 가짐.

$\mathbb{E}[n(\bar{x}-\mu)^2] = n\text{Var}(\bar{x})$. $\text{Var}(\bar{x}) = \text{Var}(\frac{1}{n}\sum x_i) = \frac{1}{n^2}\sum\text{Var}(x_i)$. 여기서 **독립(independent)** 가정 사용: 독립인 확률변수의 합의 분산 = 각 분산의 합. 따라서 $= \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$. 그러면 $n\text{Var}(\bar{x}) = \sigma^2$.

$$\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{1}{n}(n\sigma^2 - \sigma^2) = \boxed{\frac{n-1}{n}\sigma^2 \neq \sigma^2} \quad \text{(편향)} \quad \blacksquare$$

---

## Q10. (10pts) KL Divergence Properties

**[EN]**
**(a)** (4pts) Prove $D_{KL}(p\|q) \geq 0$ (Gibbs' inequality). **State the mathematical inequality you use and why it applies.**

**(b)** (3pts) Show that $D_{KL}(p\|q) = 0 \iff p = q$.

**(c)** (3pts) Given $P = (0.25, 0.25, 0.25, 0.25)$ and $Q = (0.5, 0.25, 0.125, 0.125)$, compute $H(P)$, $H(P,Q)$, and $D_{KL}(P\|Q)$. Verify $D_{KL} = H(P,Q) - H(P)$.

---

### [모범 답안]

**(a)** 사용하는 부등식: $\ln t \leq t - 1$ (모든 $t > 0$). 등호 조건: $t = 1$.

이 부등식이 적용 가능한 이유: $\ln$은 오목(concave) 함수이고, $t = 1$에서의 접선이 $y = t-1$이다. 오목 함수는 항상 접선 아래에 있으므로 $\ln t \leq t - 1$.

$$-D_{KL}(p\|q) = \sum_x p(x)\ln\frac{q(x)}{p(x)} \leq \sum_x p(x)\left(\frac{q(x)}{p(x)} - 1\right) = \sum_x q(x) - \sum_x p(x) = 1 - 1 = 0$$

따라서 $-D_{KL} \leq 0$, 즉 $\boxed{D_{KL}(p\|q) \geq 0}$.

**(b)** $D_{KL} = 0 \implies$ 위 부등식에서 등호 성립 $\implies \frac{q(x)}{p(x)} = 1$ 모든 $x$에서 $\implies q(x) = p(x)$ 모든 $x$에서.

역으로 $p = q$이면 $\ln\frac{p}{q} = \ln 1 = 0$이므로 $D_{KL} = 0$. $\blacksquare$

**(c)** $\log_2$ 사용.

$H(P) = -4 \times 0.25\log_2 0.25 = -4 \times 0.25 \times(-2) = 2$ bits

$H(P,Q) = -(0.25\log_2 0.5 + 0.25\log_2 0.25 + 0.25\log_2 0.125 + 0.25\log_2 0.125)$
$= -(0.25(-1) + 0.25(-2) + 0.25(-3) + 0.25(-3)) = -(-0.25-0.5-0.75-0.75) = 2.25$ bits

$D_{KL}(P\|Q) = H(P,Q) - H(P) = 2.25 - 2 = 0.25$ bits ✓

직접 계산 검증: $D_{KL} = 0.25\log_2\frac{0.25}{0.5} + 0.25\log_2 1 + 0.25\log_2\frac{0.25}{0.125} + 0.25\log_2\frac{0.25}{0.125} = 0.25(-1) + 0 + 0.25(1) + 0.25(1) = 0.25$ ✓

---

## Q11. (10pts) Categorical → Cross-Entropy Loss Derivation

**[EN]** The slide states $p(y|x,h) = h(x)_y$ and the CE loss is $-\frac{1}{|E|}\sum_i e_{y_i}^\top \log h(x_i)$.

**(a)** (4pts) Starting from the Categorical distribution, derive the CE loss through NLL. **Clearly state where i.i.d. is used.**

**(b)** (3pts) Show the equivalence: $-\log h(x_i)_{y_i} = -e_{y_i}^\top \log h(x_i)$. Explain why the inner product with the one-hot vector "selects" the correct class.

**(c)** (3pts) Show that minimizing CE is equivalent to minimizing $D_{KL}(\hat{p}_{\text{data}} \| p_\theta)$.

---

### [모범 답안]

**(a)** 카테고리컬 분포: $p(y_i|x_i,\theta) = h_\theta(x_i)_{y_i}$ (모델 출력 벡터의 $y_i$번째 원소).

**[i.i.d.의 독립 가정 사용]**: 각 데이터가 독립이므로 전체 우도 = 개별 우도의 곱:

$$p(D|\theta) = \prod_{i=1}^{n} h(x_i)_{y_i}$$

**[동일 분포 가정 사용]**: 모든 데이터에 같은 모델 $h_\theta$를 적용.

음의 로그를 취한다 (곱→합 변환, 최대화→최소화 변환):

$$\text{NLL} = -\sum_{i=1}^{n}\log h(x_i)_{y_i}$$

$\frac{1}{n}$으로 나누어 평균을 내면:

$$L = -\frac{1}{n}\sum_{i=1}^{n}\log h(x_i)_{y_i} = \text{CE Loss} \quad \blacksquare$$

**(b)** $e_{y_i} \in \mathbb{R}^C$는 $y_i$ 위치만 1이고 나머지 0인 원-핫 벡터이다.

$$e_{y_i}^\top \log h(x_i) = \sum_{c=1}^{C} (e_{y_i})_c \cdot \log h(x_i)_c$$

$(e_{y_i})_c = 1$이면 $c = y_i$ (정답 클래스), 나머지 $(e_{y_i})_c = 0$. 따라서 합에서 $c = y_i$ 항만 살아남는다:

$$= 1 \cdot \log h(x_i)_{y_i} + 0 + \cdots + 0 = \log h(x_i)_{y_i}$$

내적의 "선택(selection)" 효과: 원-핫 벡터와의 내적은 **정답 클래스에 해당하는 원소만 추출**하는 필터 역할을 한다. $\blacksquare$

**(c)** 경험적 분포 $\hat{p}$에 대해:

$$L = -\mathbb{E}_{(x,y)\sim\hat{p}}[\log p_\theta(y|x)] = H(\hat{p}, p_\theta)$$

$H(\hat{p}, p_\theta) = H(\hat{p}) + D_{KL}(\hat{p} \| p_\theta)$

$H(\hat{p})$는 $\theta$에 무관한 상수이므로:

$$\arg\min_\theta L = \arg\min_\theta D_{KL}(\hat{p} \| p_\theta) \quad \blacksquare$$

---

## Q12. (10pts) Sigmoid Derivative and Gradient Cancellation

**[EN]**
**(a)** (4pts) Prove $\sigma'(z) = \sigma(z)(1-\sigma(z))$. Show every step with the chain rule explicitly stated.

**(b)** (3pts) For binary CE loss $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ with $\hat{y} = \sigma(w^\top x)$, compute $\frac{\partial L}{\partial w}$ and show the cancellation of $\sigma'$.

**(c)** (3pts) What is $\max_z \sigma'(z)$? In a network with $L$ sigmoid layers, show that the gradient is bounded by $(1/4)^L$. Explain the practical consequence.

---

### [모범 답안]

**(a)** $\sigma(z) = (1+e^{-z})^{-1}$.

**체인룰 적용**: $u = 1 + e^{-z}$로 놓으면 $\sigma = u^{-1}$.

외부 미분: $\frac{d}{du}u^{-1} = -u^{-2}$

내부 미분: $\frac{du}{dz} = \frac{d}{dz}(1+e^{-z}) = -e^{-z}$ (여기서 다시 체인룰: $\frac{d}{dz}e^{-z} = e^{-z} \cdot (-1)$)

곱하면: $\sigma'(z) = -u^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$

이것을 $\sigma(z)$로 다시 표현: $\sigma(z) = \frac{1}{1+e^{-z}}$, $1-\sigma(z) = \frac{e^{-z}}{1+e^{-z}}$

$$\sigma(z)(1-\sigma(z)) = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma'(z) \quad \blacksquare$$

**(b)** 체인룰: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial\hat{y}} \cdot \frac{\partial\hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$

$$\frac{\partial L}{\partial\hat{y}} = \frac{\hat{y}-y}{\hat{y}(1-\hat{y})}, \quad \frac{\partial\hat{y}}{\partial z} = \hat{y}(1-\hat{y}), \quad \frac{\partial z}{\partial w} = x$$

곱하면: $\frac{\hat{y}-y}{\hat{y}(1-\hat{y})} \cdot \hat{y}(1-\hat{y}) \cdot x$. 분모와 같은 항이 **소거**:

$$\boxed{\frac{\partial L}{\partial w} = (\hat{y}-y)x}$$

**(c)** $\sigma'(z) = p(1-p)$ ($p = \sigma(z)$). 이차함수 $f(p) = p-p^2$의 최대: $p = 1/2$에서 $f = 1/4$.

$$\boxed{\max_z\sigma'(z) = \frac{1}{4}}$$

$L$개 시그모이드 층의 그래디언트: $\prod_{l=1}^L \sigma'(z_l) \leq (1/4)^L$.

$L = 10$: $(1/4)^{10} \approx 10^{-6}$, $L = 20$: $\approx 10^{-12}$.

실전적 결과: 앞쪽 층의 그래디언트가 지수적으로 소멸(vanishing gradient) → 학습 불가. ReLU는 양수 영역에서 미분=1이므로 이 문제 해결. $\blacksquare$

---

# PART C: Advanced — Full Derivations & Architectural Analysis (고급, 80점)

---

## Q13. (10pts) SVD and Low-Rank Approximation

**[EN]** Let $A \in \mathbb{R}^{m \times n}$ with SVD $A = U\Sigma V^\top$, $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$.

**(a)** (3pts) Write the rank-$k$ approximation $A_k$ and state the Eckart-Young theorem.
**(b)** (4pts) Prove that $\|A\|_F^2 = \sum_i \sigma_i^2$ using $\|A\|_F^2 = \text{Tr}(A^\top A)$ and the cyclic property of trace.
**(c)** (3pts) Explain one application of low-rank approximation in deep learning (e.g., LoRA) and connect it to the idea of parameter efficiency.

---

### [모범 답안]

**(a)** $A_k = \sum_{i=1}^{k}\sigma_i u_i v_i^\top = U_k\Sigma_k V_k^\top$ (처음 $k$개의 특이값/벡터만 사용).

**Eckart-Young 정리**: 모든 랭크-$k$ 이하 행렬 $B$ 중에서 $\|A - B\|_F$를 최소화하는 것은 $A_k$이다:

$$A_k = \arg\min_{\text{rank}(B)\leq k}\|A-B\|_F, \quad \|A-A_k\|_F^2 = \sum_{i=k+1}^{r}\sigma_i^2$$

**(b)** $A^\top A = V\Sigma^\top U^\top U\Sigma V^\top$. $U$는 직교이므로 $U^\top U = I$:

$$A^\top A = V\Sigma^2 V^\top$$

$\|A\|_F^2 = \text{Tr}(A^\top A) = \text{Tr}(V\Sigma^2 V^\top)$

**Trace의 순환 성질** $\text{Tr}(ABC) = \text{Tr}(CAB)$을 적용:

$$= \text{Tr}(\Sigma^2 V^\top V) = \text{Tr}(\Sigma^2 \cdot I) = \text{Tr}(\Sigma^2) = \sum_{i=1}^r \sigma_i^2 \quad \blacksquare$$

**(c)** **LoRA (Low-Rank Adaptation)**: 사전학습된 가중치 $W$를 고정하고, 업데이트를 저랭크 행렬로 근사한다: $W' = W + \Delta W$, $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$).

파라미터 효율: 원래 $d^2$개 파라미터 대신 $2dr$개만 학습. $r = 8$, $d = 4096$이면 $\frac{2 \times 8 \times 4096}{4096^2} \approx 0.4\%$만 학습. SVD의 아이디어가 핵심: **가중치 변화는 실제로 저랭크이므로, 전체 행렬을 학습할 필요 없이 소수의 방향만 업데이트하면 충분하다.** $\blacksquare$

---

## Q14. (10pts) Rank-Nullity Theorem and Pseudoinverse

**[EN]** Let $A = \begin{pmatrix}1 & 2\\2 & 4\end{pmatrix}$.

**(a)** (3pts) Find rank$(A)$, $\mathscr{N}(A)$, and verify the Rank-Nullity theorem.
**(b)** (4pts) The system $Ax = \begin{pmatrix}3\\6\end{pmatrix}$ has infinitely many solutions. Find the general solution and identify the minimum-norm solution.
**(c)** (3pts) Explain the connection to implicit regularization: why does gradient descent initialized at $\theta = 0$ converge to the minimum-norm solution in overparameterized settings?

---

### [모범 답안]

**(a)** $A$의 두 번째 행 = 첫 번째 행 × 2 → 선형종속 → $\text{rank}(A) = 1$.

$\mathscr{N}(A)$: $Ax = 0$ → $x_1 + 2x_2 = 0$ → $x_1 = -2x_2$ → $\mathscr{N}(A) = \text{span}\{(-2, 1)^\top\}$, $\dim(\mathscr{N}) = 1$.

Rank-Nullity: $n = \text{rank} + \text{nullity}$ → $2 = 1 + 1$ ✓

**(b)** $x_1 + 2x_2 = 3$에서 $x_1 = 3 - 2t$, $x_2 = t$.

일반해: $x = \begin{pmatrix}3\\0\end{pmatrix} + t\begin{pmatrix}-2\\1\end{pmatrix}$

최소 노름: $\|x\|^2 = (3-2t)^2 + t^2 = 5t^2 - 12t + 9$. 미분: $10t - 12 = 0$ → $t = 6/5$.

$$x^*_{\text{min-norm}} = \begin{pmatrix}3/5\\6/5\end{pmatrix}, \quad \|x^*\| = \sqrt{9/25 + 36/25} = \sqrt{45/25} = 3/\sqrt{5}$$

**(c)** 과매개변수화 설정에서 $Ax = y$의 해가 무한히 많을 때, $\theta(0) = 0$에서 시작한 gradient descent는 **$A$의 행공간(row space) 안에서만** 이동한다. 이유: gradient $A^\top(A\theta - y)$는 $A$의 행공간에 속하므로, 업데이트가 null space 방향으로 이동하지 않는다. 행공간 내의 해 = null space 성분이 0인 해 = **최소 노름 해**.

이것이 **암묵적 정규화(implicit regularization)**: 명시적 L2 정규화 없이도, 0 초기화 + gradient descent 자체가 "가장 단순한 해"를 자동으로 선택한다. $\blacksquare$

---

## Q15. (10pts) Softmax Jacobian Derivation

**[EN]** Given $p_i = \frac{\exp(z_i)}{\sum_k\exp(z_k)}$:

**(a)** (5pts) Derive the Jacobian: show $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$ where $\delta_{ij}$ is the Kronecker delta. **Treat the $i=j$ and $i\neq j$ cases separately, showing the quotient rule application.**

**(b)** (3pts) Express the result compactly as $\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top$.

**(c)** (2pts) This Jacobian is needed for backpropagation through the softmax layer. Explain why, in practice, we often compute the gradient of CE loss w.r.t. logits $z$ directly (bypassing the explicit Jacobian computation) and what simplification occurs.

---

### [모범 답안]

**(a)** $s = \sum_k\exp(z_k)$로 놓으면 $p_i = \exp(z_i)/s$.

**Case $i = j$**: $\frac{\partial p_i}{\partial z_i}$를 구한다. **분수 미분(몫의 미분)** 적용:

$$\frac{\partial}{\partial z_i}\frac{\exp(z_i)}{s} = \frac{\exp(z_i) \cdot s - \exp(z_i) \cdot \exp(z_i)}{s^2}$$

분자: $\exp(z_i)(s - \exp(z_i))$

$$= \frac{\exp(z_i)}{s} \cdot \frac{s-\exp(z_i)}{s} = p_i \cdot (1-p_i) = p_i(1-p_i)$$

Kronecker delta로: $p_i(\delta_{ii} - p_i) = p_i(1 - p_i)$ ✓

**Case $i \neq j$**: $\frac{\partial p_i}{\partial z_j}$를 구한다. 분자 $\exp(z_i)$는 $z_j$에 무관하므로 상수. 분모만 $z_j$에 의존.

$$\frac{\partial}{\partial z_j}\frac{\exp(z_i)}{s} = \exp(z_i) \cdot \frac{-\exp(z_j)}{s^2} = -\frac{\exp(z_i)}{s}\frac{\exp(z_j)}{s} = -p_ip_j$$

Kronecker delta로: $p_i(\delta_{ij} - p_j) = p_i(0 - p_j) = -p_ip_j$ ✓

통합: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$ $\blacksquare$

**(b)** 행렬로 쓰면:

$(i,j)$ 원소: $p_i\delta_{ij} - p_ip_j$

$p_i\delta_{ij}$ = 대각행렬 $\text{diag}(p)$의 $(i,j)$ 원소.
$p_ip_j$ = 외적 $pp^\top$의 $(i,j)$ 원소.

$$\boxed{\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top}$$

**(c)** CE Loss $= -e_y^\top\log p$를 $z$에 대해 미분하면, 체인룰에 의해:

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p}\frac{\partial p}{\partial z}$$

놀랍게도, 이 곱을 계산하면 $\frac{\partial L}{\partial z_j} = p_j - y_j$ (= 소프트맥스 출력 - 원핫 정답)로 **극도로 단순하게 정리**된다. 이것은 카테고리컬 분포의 canonical link function이 소프트맥스이기 때문이다. 실전에서는 야코비안을 명시적으로 구하지 않고, 이 단순화된 $p - y$ 형태를 직접 사용한다.

---

## Q16. (10pts) Linear Approximation and Taylor Expansion

**[EN]**
**(a)** (4pts) Write the first-order Taylor expansion of a scalar function $f(x)$ around $x_0$, and the second-order expansion. Explain what each term represents geometrically.

**(b)** (3pts) Gradient descent uses only the first-order approximation: $x_{t+1} = x_t - \eta\nabla f(x_t)$. Newton's method uses the second order: $x_{t+1} = x_t - [H(x_t)]^{-1}\nabla f(x_t)$. Explain the tradeoff between the two.

**(c)** (3pts) In the context of the softmax Jacobian (Q15), explain how the linear approximation $f(x+\delta) \approx f(x) + J\delta$ is used in backpropagation.

---

### [모범 답안]

**(a)** **1차 테일러 전개** (선형 근사):

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

기하학적 의미: $x_0$에서의 **접선**으로 함수를 근사. $f'(x_0)$은 접선의 기울기.

**2차 테일러 전개** (이차 근사):

$$f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2$$

추가 항: $\frac{1}{2}f''(x_0)(x-x_0)^2$은 함수의 **곡률(curvature)**을 반영. $f'' > 0$이면 위로 볼록, $f'' < 0$이면 아래로 볼록.

다변수로 확장: $f(x) \approx f(x_0) + \nabla f(x_0)^\top(x-x_0) + \frac{1}{2}(x-x_0)^\top H(x_0)(x-x_0)$

여기서 $H$는 헤시안 행렬 (2차 미분 정보).

**(b)** 트레이드오프:

| | Gradient Descent (1차) | Newton's Method (2차) |
|--|-------|------|
| **사용 정보** | 기울기만 | 기울기 + 곡률 |
| **수렴 속도** | 선형 수렴 (느림) | 이차 수렴 (빠름) |
| **계산 비용** | $O(d)$ | $O(d^3)$ (헤시안 역행렬) |
| **조건** | 학습률 $\eta$ 필요 | 학습률 불필요 (자동 결정) |
| **문제** | 학습률 튜닝 필요 | 비볼록에서 불안정, 메모리 폭발 |

딥러닝에서는 파라미터 $d$가 수백만~수조이므로 $d^3$은 불가능 → GD 기반 방법(SGD, Adam)을 사용.

**(c)** 역전파에서 소프트맥스 층을 통과할 때, 출력 $p$의 작은 변화 $\delta z$ (logit 변화)에 대한 $p$의 변화는:

$$\delta p \approx J \cdot \delta z$$

여기서 $J = \text{diag}(p) - pp^\top$ (Q15의 야코비안). 이것은 $f(z + \delta z) \approx f(z) + J\delta z$라는 **1차 선형 근사**이다. 역전파는 이 선형 근사를 **역방향으로** 적용하여, 출력의 그래디언트 $\frac{\partial L}{\partial p}$를 입력의 그래디언트 $\frac{\partial L}{\partial z} = J^\top \frac{\partial L}{\partial p}$로 변환한다. $\blacksquare$

---

## Q17. (10pts) PCA as Eigenvalue Problem

**[EN]** Given centered data matrix $X \in \mathbb{R}^{n \times d}$, sample covariance $S = \frac{1}{n}X^\top X$.

**(a)** (3pts) Show $S$ is PSD.
**(b)** (5pts) Using Lagrange multipliers, show that the direction of maximum variance $w^* = \arg\max_{\|w\|=1} w^\top Sw$ is the eigenvector of $S$ corresponding to its largest eigenvalue. **State the Lagrangian, differentiate, set to zero, and interpret.**
**(c)** (2pts) Connect PCA to SVD: if $X = U\Sigma V^\top$, identify the principal components.

---

### [모범 답안]

**(a)** 임의의 $v \neq 0$: $v^\top Sv = \frac{1}{n}v^\top X^\top Xv = \frac{1}{n}\|Xv\|^2 \geq 0$. 벡터 노름 제곱 ≥ 0이므로 PSD. $\blacksquare$

**(b)** 라그랑지안: $\mathcal{L}(w, \lambda) = w^\top Sw - \lambda(w^\top w - 1)$

$w$에 대해 미분한다. $w^\top Sw$는 대칭 이차형식이므로 미분은 $2Sw$. $w^\top w$의 미분은 $2w$:

$$\nabla_w\mathcal{L} = 2Sw - 2\lambda w = 0$$

이것을 0으로 놓는 이유: 제약 조건 하 극값의 필요조건 (KKT 1차 조건)이 $\nabla_w\mathcal{L} = 0$이기 때문이다.

정리하면: $Sw = \lambda w$ — 이것은 **고유값 방정식**이다.

$\|w\| = 1$ 제약 하에서: $w^\top Sw = w^\top(\lambda w) = \lambda$. 따라서 **분산 = 고유값**.

분산을 최대화하려면 **가장 큰 고유값**에 대응하는 고유벡터를 선택해야 한다. $\blacksquare$

**(c)** $S = \frac{1}{n}V\Sigma^2 V^\top$ (SVD 대입 후 $U^\top U = I$ 소거). 이것은 $S$의 고유분해 형태이다.

$V$의 열 = 주성분 방향. $\sigma_i^2/n$ = 해당 방향의 분산(= $S$의 고유값). $\blacksquare$

---

## Q18. (10pts) Bayesian Belief Update — Full Worked Example

**[EN]** A medical test has sensitivity 99% ($P(\text{positive}|\text{disease}) = 0.99$) and specificity 99% ($P(\text{negative}|\text{healthy}) = 0.99$). The disease prevalence is 0.1% ($P(\text{disease}) = 0.001$).

**(a)** (5pts) If a patient tests positive, what is the probability they actually have the disease? Use Bayes' theorem and show all steps. Explain the surprising result.

**(b)** (5pts) The patient takes a second independent test and it is also positive. Now what is the posterior? **Use the posterior from (a) as the new prior and apply Bayes' theorem again.** This illustrates the Bayesian belief update cycle: Prior → Data → Posterior → (becomes new Prior) → Data → Posterior.

---

### [모범 답안]

**(a)** 베이즈 정리:

$$P(\text{disease}|\text{positive}) = \frac{P(\text{positive}|\text{disease}) \cdot P(\text{disease})}{P(\text{positive})}$$

분모 (전체 확률의 법칙):

$$P(\text{positive}) = P(+|\text{D})P(\text{D}) + P(+|\text{H})P(\text{H})$$
$$= 0.99 \times 0.001 + 0.01 \times 0.999 = 0.00099 + 0.00999 = 0.01098$$

$$P(\text{D}|+) = \frac{0.99 \times 0.001}{0.01098} = \frac{0.00099}{0.01098} \approx 0.0902 \approx \boxed{9\%}$$

놀라운 결과: 99% 정확도의 검사에서 양성이 나왔는데, **실제 질병 확률은 9%**에 불과하다. 이유: **prior(유병률)가 0.1%로 매우 낮기 때문**이다. 양성 결과의 대부분은 건강한 사람의 거짓 양성(false positive)이다. prior가 극히 낮으면 강한 증거(99% 정확도)도 posterior를 크게 올리지 못한다.

**(b)** 첫 번째 검사 후의 posterior $P(\text{D}|+_1) = 0.0902$를 새로운 **prior**로 사용한다.

이것이 **베이지안 믿음 갱신 사이클**이다: $\text{Prior}(0.001) \xrightarrow{\text{1차 검사}} \text{Posterior}_1(0.0902) \xrightarrow{\text{새 Prior}} \text{Prior}_2(0.0902) \xrightarrow{\text{2차 검사}} \text{Posterior}_2(?)$

새 prior: $P(\text{D}) = 0.0902$, $P(\text{H}) = 0.9098$.

두 번째 양성 결과:

$$P(+_2) = 0.99 \times 0.0902 + 0.01 \times 0.9098 = 0.08930 + 0.00910 = 0.09840$$

$$P(\text{D}|+_2) = \frac{0.99 \times 0.0902}{0.09840} = \frac{0.08930}{0.09840} \approx \boxed{90.8\%}$$

1차 검사 후 9% → 2차 검사 후 91%. 증거(데이터)가 누적될수록 posterior가 극적으로 변한다. 이것이 베이지안의 핵심: **데이터가 쌓일수록 prior의 영향이 줄고, 증거가 판단을 지배한다.** 이는 MAP에서 $n \to \infty$이면 MAP → MLE가 되는 것과 같은 원리이다. $\blacksquare$

---

## Q19. (10pts) From Regression to Classification — The Unified View

**[EN]** This question tests the understanding of the complete framework.

**(a)** (5pts) Draw a concept map connecting: Bayes' theorem → MAP → NLL → {Gaussian → MSE, Categorical → CE} → Prior → Regularization → Generalization. For each arrow, write one sentence explaining the logical connection.

**(b)** (5pts) A colleague says: "I don't understand why we need probability at all. Can't we just minimize prediction error directly?" Write a rigorous response explaining what we gain from the probabilistic framework that a naive "minimize error" approach misses. Address at least: (i) principled loss function design, (ii) regularization as prior, (iii) uncertainty quantification.

---

### [모범 답안]

**(a)** 개념 맵:

```
Bayes' theorem: p(θ|D) = p(D|θ)p(θ)/p(D)
    │
    ▼ (Evidence p(D)는 θ에 무관 → 제거)
MAP: argmax [log p(D|θ) + log p(θ)]
    │                          │
    ▼ (부호 반전)               ▼
NLL: -log p(D|θ)          Prior: -log p(θ)
    │                          │
    ├──── Gaussian 가정 ────┐   ├── Gaussian prior ──┐
    │    (CLT 정당화)       │   │   N(0, σ²I)        │
    ▼                      │   ▼                    │
    MSE Loss              │   L2 Regularization    │
    │                      │   │                    │
    ├──── Categorical 가정 ─┘   │                    │
    ▼                          │                    │
    CE Loss                    │                    │
    │                          │                    │
    └──────── + ───────────────┘                    │
              │                                     │
              ▼                                     │
    Total Loss = NLL + λ||θ||²                      │
              │                                     │
              ▼ (weight 작게 → 부드러운 함수)          │
    Generalization ◄────────────────────────────────┘
```

각 화살표의 논리:
- Bayes → MAP: Evidence 제거 (θ에 무관한 상수)
- MAP → NLL + Prior: 로그 변환으로 곱→합, 부호 반전으로 max→min
- NLL + Gaussian → MSE: $-\log\exp(-z^2) = z^2$, CLT가 가우시안 가정을 정당화
- NLL + Categorical → CE: $-\log h(x)_y$가 정답 클래스의 로그 확률
- Prior(Gaussian) → L2: $-\log\mathcal{N}(0,\sigma_p^2) \propto \|\theta\|^2$
- L2 → Generalization: 가중치 작게 → 입력 변화에 둔감 → 과적합 방지

**(b)** 확률적 프레임워크가 "단순 오차 최소화"보다 우월한 이유:

**(i) 원칙적 손실 함수 설계**: "왜 MSE인가? 왜 CE인가?"에 대한 답을 준다. 확률적 프레임워크 없이는 손실 함수 선택이 임의적이다. 하지만 NLL을 통해 **분포 가정이 손실 함수를 결정**한다는 것을 알면, 문제의 성격에 맞는 손실을 **원칙적으로** 선택할 수 있다. 예: 이상치가 많으면 라플라시안 → L1 Loss, 정상적 데이터면 가우시안 → L2 Loss.

**(ii) 정규화의 이론적 근거**: "왜 L2 정규화가 과적합을 방지하는가?"는 확률 없이는 설명하기 어렵다. MAP 프레임워크에서 $\lambda\|\theta\|^2 = -\log p(\theta)$ (가우시안 prior)임을 알면, 정규화는 **"파라미터가 0 근처에 있을 것"이라는 사전 지식을 수학적으로 통합**한 것임을 이해한다. 정규화 강도 $\lambda$도 prior의 분산 $\sigma_p^2$로 해석 가능.

**(iii) 불확실성 정량화**: 점 추정(MLE/MAP)을 넘어, 사후 분포 전체 $p(\theta|D)$를 활용하면 **예측의 불확실성**을 계산할 수 있다. 이것은 의료, 자율주행 등 신뢰성이 중요한 응용에서 필수적이다. "이 예측이 얼마나 확실한가?"는 단순 오차 최소화로는 답할 수 없다.

---

## Q20. (10pts) The Complete Story — A Synthesis Question

**[EN]** A data scientist is building a regression model. She collects $n = 1000$ data points, assumes $y = f_\theta(x) + \epsilon$, and uses Ridge Regression: $\hat{\theta} = \arg\min_\theta [\text{MSE} + \lambda\|\theta\|^2]$.

**(a)** (3pts) Trace the theoretical justification for her choice: CLT → Gaussian noise → NLL → MSE → Gaussian prior → L2 regularization. Write one sentence for each step.

**(b)** (4pts) She chooses $\lambda = 0.01$. If the noise variance is $\sigma^2 = 1$, what is the implied prior variance $\sigma_p^2$? What does this say about her prior belief?

**(c)** (3pts) Her model achieves training MSE = 0.05 and test MSE = 0.30. Diagnose the problem and recommend a solution in terms of the MAP framework (i.e., should she increase or decrease $\lambda$, and why in terms of the prior?).

---

### [모범 답안]

**(a)** 각 단계의 이론적 정당화:

1. **CLT → 가우시안 노이즈**: 예측 오차 $\epsilon$은 수많은 미지 요인의 합산이며, CLT에 의해 가우시안에 수렴한다.

2. **가우시안 노이즈 → 가우시안 우도**: $\epsilon \sim \mathcal{N}(0, \sigma^2)$이므로 $p(y|x,\theta) = \mathcal{N}(f_\theta(x), \sigma^2)$이다.

3. **가우시안 우도 → NLL**: i.i.d. 가정으로 우도를 곱으로 분해하고, 음의 로그를 취하여 최소화 문제로 변환한다.

4. **NLL → MSE**: 가우시안 PDF에서 $-\log\exp(-z^2) = z^2$으로 제곱항이 출현하여 NLL ∝ MSE가 된다.

5. **가우시안 Prior → L2**: $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$를 가정하면 $-\log p(\theta) \propto \|\theta\|^2$이다.

6. **MAP = MSE + L2**: NLL + Prior = MSE + $\lambda\|\theta\|^2$. 이것이 Ridge Regression이다.

**(b)** $\lambda = \frac{\sigma^2}{n\sigma_p^2}$에서:

$$\sigma_p^2 = \frac{\sigma^2}{n\lambda} = \frac{1}{1000 \times 0.01} = \frac{1}{10} = 0.1$$

해석: prior의 표준편차 $\sigma_p = \sqrt{0.1} \approx 0.316$. 이것은 "대부분의 가중치가 $\pm 0.63$ (= $2\sigma_p$) 안에 있을 것"이라는 **적당히 강한 사전 믿음**이다. 완전히 약한 prior($\sigma_p = 100$)도 아니고, 극단적으로 강한 prior($\sigma_p = 0.01$)도 아닌 중간 수준.

**(c)** 진단: 훈련 MSE(0.05) ≪ 테스트 MSE(0.30) → **과적합(Overfitting)**. 모델이 훈련 데이터의 노이즈까지 학습했다.

MAP 관점에서의 해결:

**$\lambda$를 증가**시켜야 한다. 이유:

- $\lambda$ 증가 = prior를 더 강하게 = $\sigma_p^2$ 감소 = "가중치는 더 작아야 한다"는 믿음 강화
- 가중치가 작아지면 → 모델의 출력이 입력 변화에 덜 민감 → 부드러운(smooth) 함수 → 훈련 데이터의 노이즈에 덜 맞춤 → 일반화 개선

구체적으로: $\lambda = 0.01 \to 0.1$ 정도로 10배 증가시키면, $\sigma_p^2 = 0.01$이 되어 "가중치가 $\pm 0.2$ 안에 있을 것"이라는 더 강한 prior가 적용된다. 훈련 MSE는 약간 올라가겠지만(0.05 → 0.1 정도), 테스트 MSE가 크게 줄어들어(0.30 → 0.15 정도) **일반화 갭이 좁아질 것**으로 기대한다. $\blacksquare$

---

# 채점 기준 총정리

| 구분 | 문제 | 배점 합계 | 난이도 |
|------|------|----------|--------|
| **입문** (20%) | Q1, Q2, Q3, Q4 | 40점 | 개념 정의, 프레임워크 이해 |
| **중급** (40%) | Q5~Q12 | 80점 | 유도, 증명, 비교 |
| **고급** (40%) | Q13~Q20 | 80점 | 수식 전개, 아키텍처 분석, 통합 |
| **총점** | 20문제 | **200점** | |

> **핵심 채점 원칙**: 답이 맞아도 논리 과정이 없으면 0점. 매 단계마다 (1) 어떤 가정을 사용했는지 (i.i.d., 가우시안 등), (2) 왜 이 수학적 조작을 했는지 (로그 변환, 상수 제거, 미분=0 등)를 글로 명확히 서술해야 점수를 받을 수 있음.

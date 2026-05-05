---
title: "09. 베이지안 확률 & 정보이론 (Bayesian Probability & Information Theory)"
slug: 09-bayesian-information-theory
order: 9
---

# 09. 베이지안 확률 & 정보이론 (Bayesian Probability & Information Theory)

## 1. 동기부여 및 개요

딥러닝의 학습(training)은 본질적으로 **데이터로부터 불확실한 모델 파라미터를 추정**하는 과정이다. 이 장에서는 두 가지 핵심 이론을 다룬다:

1. **베이지안 추론**: 사전 지식(prior)과 데이터(likelihood)를 결합하여 사후 분포(posterior)를 구하는 체계. MLE의 과적합 문제를 해결하고 정규화의 이론적 근거를 제공한다.
2. **정보이론**: "불확실성을 어떻게 측정하고, 분포 간 차이를 어떻게 정량화하는가?"에 대한 답. 딥러닝의 손실 함수(cross-entropy), VAE의 ELBO, Knowledge Distillation의 KL divergence 등 현대 딥러닝의 거의 모든 학습 목표가 여기에 기반한다.

> **선수 지식**: 확률론 기초 (08장), 미적분, 로그 함수의 성질

---

## 2. 베이즈 정리 (Bayes' Theorem)

### 2.1 정의

**Theorem 2.1 (Bayes' Theorem).**

$$\boxed{P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}}$$

| 기호 | 이름 | 의미 |
|:---:|:---|:---|
| $P(H)$ | **사전 확률 (Prior)** | 데이터를 보기 전 가설에 대한 믿음 |
| $P(E \mid H)$ | **우도 (Likelihood)** | 가설 $H$ 하에서 데이터 $E$가 관측될 확률 |
| $P(E)$ | **주변 우도 (Evidence)** | 정규화 상수, $P(E) = \int P(E \mid H)P(H)\,dH$ |
| $P(H \mid E)$ | **사후 확률 (Posterior)** | 데이터를 관측한 후의 업데이트된 믿음 |

> Sir Harold Jeffreys: "베이즈 정리는 확률론에서 피타고라스 정리가 기하학에서 차지하는 위치와 같다."

### 2.2 빈도주의 vs 베이지안

| 관점 | 확률의 해석 | 대상 |
|:---|:---|:---|
| **베이지안** | 믿음의 정도 (degree of belief) | 가설 (Hypothesis) |
| **빈도주의** | 빈도의 극한 (long-run frequency) | 사건 (Event) |

### 2.3 순차적 업데이트

베이지안 추론의 핵심 사이클:

$$\cdots \to \text{Data}_1 \to \text{Posterior}_1 = \text{Prior}_2 \to \text{Data}_2 \to \text{Posterior}_2 \to \cdots$$

이전 사후 확률이 다음 관측의 사전 확률이 된다. 이것이 **온라인 학습(online learning)**의 이론적 근거이다.

### 2.4 계산적 도전 (심화)

사후 분포를 해석적으로 구하려면 $P(E) = \int P(E \mid H)P(H)\,dH$를 계산해야 하는데, 고차원에서 이 적분은 일반적으로 **난해(intractable)**하다. 근사 방법:

| 방법 | 설명 | 딥러닝 응용 |
|:---|:---|:---|
| **변분 추론 (VI)** | $q(H) \approx P(H \mid E)$로 근사, KL 최소화 | VAE의 ELBO |
| **MCMC** | 사후 분포에서 직접 샘플링 | Bayesian NN 학습 |
| **라플라스 근사** | MAP 근처에서 가우시안 근사 | 불확실성 추정 |

---

## 3. 최대우도추정 (Maximum Likelihood Estimation, MLE)

### 3.1 정의

데이터 $E = \{x_1, \ldots, x_n\}$이 i.i.d.일 때:

$$\theta_{\text{ML}} = \arg\max_\theta \prod_{i=1}^n P(x_i \mid \theta) = \arg\max_\theta \sum_{i=1}^n \log P(x_i \mid \theta)$$

### 3.2 구체적 예시

**베르누이 분포에서의 MLE:**
동전 $n$번 중 앞면 $k$번:

$$\text{loglik} = k\log\theta + (n-k)\log(1-\theta)$$

$$\frac{\partial}{\partial\theta}\text{loglik} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0 \implies \boxed{\theta_{\text{ML}} = \frac{k}{n}}$$

**정규분포에서의 MLE:**

$$\mu_{\text{ML}} = \frac{1}{n}\sum_{i=1}^n x_i, \qquad \sigma^2_{\text{ML}} = \frac{1}{n}\sum_{i=1}^n(x_i - \bar{x})^2$$

> **주의**: $\mathbb{E}[\sigma^2_{\text{ML}}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$ (편향 추정량). 비편향 추정량은 $n$을 $n-1$로 나눈 것 (Bessel's correction).

### 3.3 MLE의 점근적 성질

- **일치성**: $n \to \infty$이면 $\theta_{\text{ML}} \xrightarrow{p} \theta^*$
- **점근적 정규성**: $\sqrt{n}(\theta_{\text{ML}} - \theta^*) \xrightarrow{d} \mathcal{N}(0, I(\theta^*)^{-1})$
- **점근적 효율성**: Cramer-Rao 하한 달성

### 3.4 MLE의 한계: 과적합

$n = 3$번 동전을 던져 $k = 3$번 앞면이면 $\theta_{\text{ML}} = 1.0$. 미래의 모든 동전 던지기가 앞면이라고 예측하게 되는데, 이는 명백히 비합리적이다. 이 문제를 해결하는 것이 MAP 추정이다.

---

## 4. MAP 추정 (Maximum A Posteriori)

### 4.1 정의

$$\theta_{\text{MAP}} = \arg\max_\theta P(\theta \mid E) = \arg\max_\theta \left[\underbrace{\log P(E \mid \theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{log-prior}}\right]$$

### 4.2 Uniform Prior일 때 MAP = MLE

$P(\theta) = \text{const}$이면 $\log P(\theta)$가 상수이므로 $\theta_{\text{MAP}} = \theta_{\text{ML}}$.

### 4.3 Beta Prior를 사용한 예시

$P(\theta) \propto \theta(1 - \theta)$ (Beta(2,2)에 비례)이면:

$$\text{logpost} = (k+1)\log\theta + (n-k+1)\log(1-\theta) + C$$

$$\theta_{\text{MAP}} = \frac{k+1}{n+2}$$

$n = 3, k = 3$이면 $\theta_{\text{MAP}} = 4/5 = 0.8$ (MLE의 1.0보다 합리적).

### 4.4 MAP과 정규화의 관계

이것이 이 장의 **가장 중요한 연결**이다:

| 사전 분포 | $\log P(\theta)$ | 정규화 형태 |
|:---|:---|:---|
| $P(\theta) \sim \mathcal{N}(0, \sigma_p^2 I)$ | $-\frac{1}{2\sigma_p^2}\|\theta\|^2$ | **L2 정규화 (Weight Decay)** |
| $P(\theta) \propto \exp(-\lambda\|\theta\|_1)$ | $-\lambda\|\theta\|_1$ | **L1 정규화 (Sparsity)** |

따라서:
- **Weight decay**는 가우시안 사전분포를 가정한 MAP 추정
- **LASSO**는 라플라스 사전분포를 가정한 MAP 추정

이 관계는 13장 정규화에서 더 깊이 다룬다.

### 4.5 MLE vs MAP 비교

| 속성 | MLE | MAP |
|:---|:---|:---|
| 목적함수 | $\max P(E \mid \theta)$ | $\max P(\theta \mid E)$ |
| 사전분포 | 균등 (implicit) | 비균등 (명시적) |
| 과적합 | 쉬움 | 어려움 (정규화 효과) |
| 데이터 적을 때 | 불안정 | 안정적 |
| 데이터 많을 때 | 성능 좋음 | MLE에 수렴 |

### 4.6 MAP의 한계

MAP는 사후 분포의 **모드(mode)**만 구하고 **분포 전체의 형태(불확실성)**는 무시한다. Full Bayesian 접근은 $P(\theta \mid E)$ 전체를 활용:

$$P(x_{\text{new}} \mid E) = \int P(x_{\text{new}} \mid \theta)P(\theta \mid E)\,d\theta$$

---

## 5. 엔트로피 (Entropy)

### 5.1 정의

**Definition 5.1 (Shannon Entropy).**

$$H(p) := -\mathbb{E}_p[\log p(X)] = -\sum_x p(x)\log p(x)$$

### 5.2 직관: 놀라움의 평균

개별 사건 $x$의 **놀라움(surprisal)**: $-\log p(x)$. 확률이 낮을수록 놀라움이 크다. 엔트로피는 이 놀라움의 **기댓값**이다.

### 5.3 주요 성질

- $H(p) \geq 0$ (항상 비음수)
- 결정론적 분포 (one-hot): $H = 0$ (최소)
- 유한 집합 $\mathcal{S}$ 위의 균등분포: $H = \log|\mathcal{S}|$ (최대)

### 5.4 미분 엔트로피 (연속)

$$H(X) = -\int p(x)\log p(x)\,dx$$

가우시안의 엔트로피: $H(\mathcal{N}(\mu, \Sigma)) = \frac{1}{2}\log\left((2\pi e)^d|\Sigma|\right)$

### 5.5 최대 엔트로피 원리와의 연결

06장에서 유도한 것처럼:
- 제약 없음 (유한 집합) $\to$ 균등분포
- 평균+분산 제약 $\to$ 정규분포

### 5.6 딥러닝에서의 엔트로피

- **Label Smoothing**: one-hot 레이블의 엔트로피를 약간 높여 과적합 방지
- **RL 탐색 보너스**: 정책의 엔트로피에 보너스를 부여하여 exploration 촉진 (SAC 알고리즘)

---

## 6. 교차 엔트로피 (Cross-Entropy)

### 6.1 정의

**Definition 6.1.**

$$CE(p, q) := -\mathbb{E}_p[\log q(X)] = -\sum_x p(x)\log q(x)$$

### 6.2 핵심 관계

$$\boxed{CE(p, q) = H(p) + KL(p \| q)}$$

- $H(p)$: 진짜 분포의 본질적 불확실성 (상수, 바꿀 수 없음)
- $KL(p \| q) \geq 0$: 추가적인 비효율성

**따라서 CE를 최소화하는 것 = KL을 최소화하는 것**

### 6.3 분류 문제에서의 교차 엔트로피 손실

진짜 레이블이 one-hot $p = (0, \ldots, 1, \ldots, 0)$이고 모델 출력이 $q = \text{softmax}(z)$일 때:

$$CE(p, q) = -\log q(y_{\text{true}}) = -\log\frac{e^{z_{y_{\text{true}}}}}{\sum_c e^{z_c}}$$

이것이 **Negative Log-Likelihood (NLL)**이며, 딥러닝 분류의 표준 손실 함수이다.

### 6.4 CE와 MSE의 관계

정규분포 가정 하에서:

$$KL(\mathcal{N}(\mu_1, \sigma^2I) \| \mathcal{N}(\mu_2, \sigma^2I)) = \frac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2$$

따라서 **가우시안 가정 + CE(NLL) 최소화 = MSE 최소화**. 이는 10장 선형회귀에서 상세히 다룬다.

---

## 7. KL 다이버전스 (KL Divergence)

### 7.1 정의

**Definition 7.1.**

$$KL(p \| q) := \mathbb{E}_p\left[\log\frac{p(X)}{q(X)}\right] = \sum_x p(x)\log\frac{p(x)}{q(x)}$$

### 7.2 핵심 성질

- $KL(p \| q) \geq 0$ (**Gibbs' inequality**)
- $KL(p \| q) = 0 \iff p = q$
- $KL(p \| q) \neq KL(q \| p)$ (**비대칭! "거리"가 아님**)

### 7.3 Forward KL vs Reverse KL

이것은 딥러닝에서 매우 중요한 구분이다:

| | Forward KL: $\min_q KL(p \| q)$ | Reverse KL: $\min_q KL(q \| p)$ |
|:---|:---|:---|
| 특성 | **Zero-avoiding (mode-covering)** | **Zero-forcing (mode-seeking)** |
| 행동 | $p > 0$인 모든 곳을 커버하려 함 | $p = 0$인 곳에서 $q = 0$이 되려 함 |
| 다봉 분포 근사 | 모든 모드 사이를 채움 | 하나의 모드에 집중 |
| 딥러닝 사용처 | **MLE / CE 최소화 (학습)** | **변분 추론 (VI), VAE** |

```
                Forward KL                    Reverse KL
    p(x)         q(x)              p(x)         q(x)
    ╱╲   ╱╲     ╱────╲            ╱╲   ╱╲       ╱╲
   ╱  ╲ ╱  ╲   ╱      ╲         ╱  ╲ ╱  ╲     ╱  ╲
  ╱    ╲╱    ╲ ╱        ╲       ╱    ╲╱    ╲   ╱    ╲
 ───────────── ──────────       ───────────── ─────────
  mode-covering                  mode-seeking
  (두 모드 사이 채움)             (한 모드에 집중)
```

### 7.4 가우시안 간의 KL

$$KL(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} + \text{tr}(\Sigma_1\Sigma_2^{-1}) + (\mu_1 - \mu_2)^\top\Sigma_2^{-1}(\mu_1 - \mu_2) - D\right]$$

특히 VAE에서 $q(z \mid x) = \mathcal{N}(\mu, \sigma^2I)$와 prior $p(z) = \mathcal{N}(0, I)$ 사이의 KL이 ELBO의 정규화 항이 된다.

### 7.5 딥러닝에서의 핵심 응용

1. **VAE의 ELBO**: $\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - KL(q(z|x) \| p(z))$
2. **Knowledge Distillation**: Student가 Teacher의 soft label을 따라가도록 $KL(p_T \| p_S)$ 최소화
3. **PPO**: 정책 업데이트를 KL divergence로 제한
4. **Natural Gradient**: Fisher 정보행렬 $F(\theta) \approx$ 국소 KL의 Hessian

---

## 8. 상호정보량 (Mutual Information)

### 8.1 정의

**Definition 8.1.**

$$I(X; Y) := KL(p(x,y) \| p(x)p(y)) = \mathbb{E}_{p(x,y)}\left[\log\frac{p(x,y)}{p(x)p(y)}\right]$$

### 8.2 동치 표현들

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

해석: **$Y$를 알았을 때 $X$에 대한 불확실성이 얼마나 줄어드는가.**

### 8.3 성질

- $I(X; Y) \geq 0$
- $I(X; Y) = 0 \iff X \perp Y$ (독립)
- **상관계수와 달리 비선형 관계도 포착**

### 8.4 딥러닝에서의 응용

- **InfoNCE Loss (Contrastive Learning)**: $I(X; Y)$의 하한을 최대화. SimCLR, CLIP 등의 이론적 기반
- **Information Bottleneck**: $\min_{p(z|x)}[-I(Z;Y) + \beta I(Z;X)]$ --- 표현 $Z$가 입력의 정보를 최소한으로 가지면서 출력 정보를 최대화
- **Data Processing Inequality**: $X \to Y \to Z$이면 $I(X;Z) \leq I(X;Y)$ --- 레이어가 깊어질수록 입력 정보가 줄어들 수밖에 없다는 이론적 한계

---

## 9. 확률분포 적합과 체제 분류

### 9.1 분포 적합 (Distribution Fitting)

데이터가 주어졌을 때 모델의 파라미터를 찾는 과정:

$$\text{MLE} \to \text{MAP} \to \text{Full Bayesian}$$

"가우시안으로 적합"이라는 가정 자체가 **귀납적 편향(inductive bias)**이다. 정당화:
- 최대 엔트로피 원리: 평균과 분산만 알 때 가장 보수적인 선택
- CLT: 많은 독립적 요인의 합은 가우시안에 수렴

### 9.2 현대 딥러닝의 체제 분류

| 체제 | 데이터/모델 관계 | 전략 |
|:---|:---|:---|
| Internet-Scale DL | 극도로 undercapacity | Scaling 중심 (MLE) |
| Classical ML | Undercapacity | 정규화 중심 (MAP) |
| ImageNet-Scale DL | Overcapacity | Double descent 현상 |

LLM 시대에는 데이터가 충분히 많아 강한 prior 없이 MLE만으로도 뛰어난 성능을 달성한다 (Scaling Law의 이론적 기반).

---

## 10. 흔한 오해와 주의점

| 오해 | 실제 | 교정 |
|:---|:---|:---|
| KL divergence는 거리(distance) | **비대칭**이므로 거리 아님. 삼각부등식도 불만족 | "상대 엔트로피" 또는 "정보 이득"으로 이해 |
| MLE는 항상 좋은 추정량 | 소표본에서 극단적 값 가능 ($k/n = 3/3 = 1$) | 점근적으로만 좋은 성질. 소표본에선 MAP 권장 |
| 엔트로피 높음 = 나쁨 | 맥락 의존. RL에서는 높은 엔트로피가 탐색에 유리 | 중립적 불확실성 측정 도구 |
| CE와 KL은 완전히 다른 것 | $CE = H(p) + KL$. $H(p)$ 고정이면 최적화 관점에서 동치 | 같은 최적화 목표의 다른 표현 |
| 강한 prior가 항상 좋음 | prior 너무 강하면 데이터 무시 | 데이터 양과 prior 강도의 균형 필요 |
| Forward KL ≈ Reverse KL | 다봉 분포 근사에서 결과가 극적으로 다름 | mode-covering vs mode-seeking 구분 |

---

## 11. 핵심 요약

| 개념 | 핵심 수식 | 딥러닝 연결 |
|:---|:---|:---|
| 베이즈 정리 | $P(H \mid E) \propto P(E \mid H)P(H)$ | 데이터로 모델 업데이트 |
| MLE | $\arg\max \log P(E \mid \theta)$ | 정규화 없는 학습 |
| MAP | $\arg\max[\text{loglik} + \text{logprior}]$ | Weight decay, Dropout |
| 엔트로피 | $H(p) = -\mathbb{E}_p[\log p]$ | Label smoothing, RL 탐색 |
| 교차 엔트로피 | $CE(p,q) = H(p) + KL(p \| q)$ | **분류 손실함수** |
| KL divergence | $\mathbb{E}_p[\log(p/q)]$ | VAE ELBO, KD, PPO |
| 상호정보량 | $I(X;Y) = H(X) - H(X \mid Y)$ | Contrastive learning |

**핵심 등식 체인**:

$$\text{CE 최소화} = \text{KL 최소화} \quad (\because H(p) \text{는 상수})$$

$$\text{가우시안 가정} + \text{CE} = \text{MSE} \quad (\text{10장에서 상세 다룸})$$

$$\text{MAP} + \text{가우시안 prior} = \text{L2 정규화 (Weight Decay)}$$

**한 문장 요약**: 딥러닝의 학습은 베이지안 추론 관점에서 사후 확률을 최대화하는 것이며, 그 목적함수(CE, KL)는 정보이론이 제공하는 불확실성 측정 도구로 구성된다.

---

## 참고 문헌

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 3 & 5.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapters 4-6.
- Kingma, D. P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR*.
- Hinton, G. et al. (2015). Distilling the Knowledge in a Neural Network.

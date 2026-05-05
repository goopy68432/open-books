---
title: "08. 확률론 (Probability Theory)"
slug: 08-probability
order: 8
---

# 08. 확률론 (Probability Theory)

## 1. 동기부여 및 개요

딥러닝의 거의 모든 핵심 구성 요소가 확률론 위에 세워져 있다:
- **손실 함수**: cross-entropy는 조건부 확률의 음의 로그
- **생성 모델**: VAE, diffusion model은 확률분포를 직접 학습
- **베이지안 추론**: 모델의 불확실성을 정량화

확률을 이해하지 못하면 모델이 "왜" 그렇게 학습하는지, 불확실성을 "어떻게" 정량화하는지 설명할 수 없다.

> **선수 지식**: 집합론 기초, 미적분(적분), 선형대수(행렬, 역행렬)

---

## 2. 표본공간, 사건, 확률의 기초

### 2.1 기본 정의

**Definition 2.1 (표본공간, Sample Space).**
실험에서 나올 수 있는 **모든 결과의 집합**을 표본공간 $S$라 한다.

**Definition 2.2 (사건, Event).**
표본공간 $S$의 **부분집합** $E \subseteq S$를 사건이라 한다.

**Definition 2.3 (확률분포, Probability Distribution).**
함수 $p: S \to [0, 1]$이 다음을 만족하면 확률분포이다:
1. $0 \leq p(s) \leq 1$ (각 결과에 대해)
2. $\sum_{s \in S} p(s) = 1$ (정규화 조건)

사건 $E$의 확률: $P(E) := \sum_{s \in E} p(s)$

### 2.2 확률의 기본 성질

- $P(\emptyset) = 0$, $P(S) = 1$
- 여사건: $P(E^c) = 1 - P(E)$
- 포함-배제: $P(E_1 \cup E_2) = P(E_1) + P(E_2) - P(E_1 \cap E_2)$

### 2.3 측도론적 정의 (심화)

수학적으로 엄밀한 확률은 확률공간 $(\Omega, \mathcal{F}, P)$로 정의된다. $\mathcal{F}$는 $\sigma$-대수, $P$는 가산 가법적 측도이다. 연속 표본공간에서는 확률밀도함수(pdf)를 누적분포함수(cdf) $c_X(a) := P(X \leq a)$의 미분으로 정의한다.

> **주의**: 밀도 $p(x)$는 확률이 아니라 **"단위 길이당 확률 질량"**이다. $p(x)$는 1을 초과할 수 있다.

---

## 3. 조건부 확률과 독립성

### 3.1 조건부 확률

**Definition 3.1 (조건부 확률).**

$$P(E \mid F) = \frac{P(E \cap F)}{P(F)} \quad (P(F) > 0)$$

직관: 전체 표본공간을 $F$로 "축소"한 후, 그 안에서 $E \cap F$의 비율을 계산하는 것.

### 3.2 독립성

**Definition 3.2 (독립, Independence).**
두 사건 $E, F$가 독립이면:

$$P(E \cap F) = P(E) \cdot P(F) \quad \Leftrightarrow \quad P(E \mid F) = P(E)$$

### 3.3 조건부 독립

**Definition 3.3 (조건부 독립, Conditional Independence).**

$$P(E, F \mid G) = P(E \mid G) \cdot P(F \mid G) \quad \text{(표기: } E \perp F \mid G\text{)}$$

**핵심 경고**: 독립과 조건부 독립은 별개의 개념이다:
- 독립 $\not\Rightarrow$ 조건부 독립
- 조건부 독립 $\not\Rightarrow$ 독립

**예시**: 주사위 두 개를 던질 때, 각 주사위 결과 $E, F$는 독립이지만, 합이 짝수라는 조건 $G$가 주어지면 $E \not\perp F \mid G$이다 ("explaining away" 현상).

### 3.4 딥러닝에서의 조건부 독립

- **Chain rule**: $p(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T p(w_t \mid w_{1:t-1})$ (언어 모델의 기초)
- **Markov 가정**: $w_t \perp w_{1:t-k-1} \mid w_{t-k:t-1}$ (n-gram 모델)
- **VAE**: latent variable $z$가 주어지면 관측 변수들이 조건부 독립

---

## 4. 확률변수와 확률분포

### 4.1 확률변수

**Definition 4.1 (확률변수, Random Variable).**
표본공간 $S$에서 실수로의 함수:

$$X: S \to \mathbb{R}, \quad s \mapsto X(s)$$

$P(X = a) = P(\{s \in S : X(s) = a\}) = P(X^{-1}(\{a\}))$

### 4.2 분포의 두 유형

| | 이산 (Discrete) | 연속 (Continuous) |
|:---|:---|:---|
| 분포 함수 | PMF: $p_X(a) = P(X = a)$ | PDF: $p_X(a) = \frac{d}{da}c_X(a)$ |
| 확률 계산 | $P(X \in A) = \sum_{a \in A} p_X(a)$ | $P(X \in A) = \int_A p_X(x)\,dx$ |
| 정규화 | $\sum_a p_X(a) = 1$ | $\int p_X(x)\,dx = 1$ |

---

## 5. 핵심 이산 분포

### 5.1 베르누이 분포

**Definition 5.1.** $X \sim \text{Bern}(\theta)$:

$$P(X = 1) = \theta, \quad P(X = 0) = 1 - \theta$$

### 5.2 이항 분포

독립인 베르누이 시행 $n$번 중 성공 $k$번:

$$P(X = k) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$$

### 5.3 범주형 분포

$C$개 레이블에 대해 $\text{Cat}(y; \theta) = \theta_y$, 여기서 $\sum_{c=1}^C \theta_c = 1$.

### 5.4 분포 간 관계

```
베르누이 (|S|=2)  ──일반화──→  범주형 (|S|=C)
    │                             │
  n번 반복                      n번 반복
    ↓                             ↓
이항 (Binomial)   ──일반화──→  다항 (Multinomial)
```

### 5.5 딥러닝에서의 활용

- 신경망의 **softmax 출력** = 범주형 분포의 파라미터 $\theta$
- **Cross-entropy loss** = $-\log \text{Cat}(y; \hat{\theta})$
- **Next-Token Prediction**: $p(w_t \mid w_{1:t-1})$은 vocabulary 크기 $|V|$에 대한 조건부 범주형 분포

---

## 6. 정규(가우시안) 분포와 중심극한정리

### 6.1 정규분포

**Definition 6.1 (정규분포).**

$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right)$$

$\mu$: 평균 (중심), $\sigma$: 표준편차 (퍼짐 정도)

### 6.2 정규화 상수의 유도

$$I = \int_\mathbb{R} e^{-x^2/2}\,dx \;\Rightarrow\; I^2 = \int\int e^{-(x^2+y^2)/2}\,dx\,dy = 2\pi \;\Rightarrow\; I = \sqrt{2\pi}$$

(극좌표 변환 이용)

### 6.3 중심극한정리 (CLT)

**Theorem 6.2 (Central Limit Theorem).**
$X_1, X_2, \ldots$가 i.i.d.이고 유한 분산 $\sigma^2$을 가지면:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

### 6.4 정규분포가 "정규"인 두 가지 이유

1. **최대 엔트로피**: 평균 $\mu$와 분산 $\sigma^2$만 알려졌을 때 엔트로피를 최대화하는 분포 (06장 변분법/라그랑주 승수법으로 증명)
2. **CLT의 보편성**: 유한 분산을 갖는 거의 모든 분포의 i.i.d. 합이 가우시안으로 수렴

### 6.5 딥러닝에서의 가우시안

가중치 초기화, 노이즈 모델링, diffusion process, VAE의 prior 등 거의 모든 곳에서 기본 가정으로 사용된다.

---

## 7. 기댓값과 분산

### 7.1 정의

**Definition 7.1.**

$$\mathbb{E}[X] = \begin{cases} \sum_x x \cdot p_X(x) & \text{(이산)} \\ \int x \cdot p_X(x)\,dx & \text{(연속)} \end{cases}$$

$$\text{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

### 7.2 기댓값의 핵심 성질

**기댓값의 선형성** (독립 여부와 **무관**하게 항상 성립):

$$\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$$

독립일 때 추가 성질:

$$\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] \quad \text{(역은 불성립!)}$$

### 7.3 분산의 성질

- $\text{Var}[aX + b] = a^2\text{Var}[X]$
- 비상관이면: $\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y]$
- 일반적으로: $\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y] + 2\text{Cov}[X, Y]$

### 7.4 전체 기대/분산 법칙

$$\mathbb{E}[X] = \mathbb{E}_Y[\mathbb{E}[X \mid Y]] \quad \text{(Law of Total Expectation)}$$

$$\text{Var}[X] = \mathbb{E}_Y[\text{Var}[X \mid Y]] + \text{Var}_Y[\mathbb{E}[X \mid Y]] \quad \text{(Law of Total Variance)}$$

### 7.5 딥러닝에서의 기댓값/분산

- **Empirical risk**: $\hat{R} = \frac{1}{n}\sum_i \ell(f(x_i), y_i) \approx \mathbb{E}[\ell(f(X), Y)]$ (표본 근사)
- **미니배치 SGD**: 배치 크기 증가 $\to$ 기울기 추정의 분산 감소 $\to$ 안정적 학습
- **기댓값의 선형성**: 역전파에서 기울기의 기댓값 계산에 핵심

---

## 8. 결합분포, 주변분포, 공분산

### 8.1 결합분포와 주변분포

두 확률변수 $X, Y$에 대해:

- **결합분포**: $p(x, y) = P(X = x, Y = y)$
- **주변분포**: $p(x) = \sum_y p(x, y)$ 또는 $p(x) = \int p(x, y)\,dy$

관계: $p(x, y) = p(x \mid y) \cdot p(y)$

### 8.2 공분산

**Definition 8.1.**

$$\text{Cov}[X, Y] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

### 8.3 독립 vs 비상관

**Theorem 8.2.**
- 독립 $\Rightarrow$ 비상관 ($\text{Cov} = 0$)
- 비상관 $\not\Rightarrow$ 독립 (일반적으로)
- **결합 정규분포**에서만: 비상관 $\Leftrightarrow$ 독립

**반례**: $(X, Y) \in \{(0,1), (1,0), (-1,0), (0,-1)\}$이 균등 확률이면 $\text{Cov}[X,Y] = 0$이지만 $X, Y$는 독립이 아니다. $P(X=0, Y=0) = 0 \neq P(X=0)P(Y=0) = 1/4$이기 때문.

---

## 9. 다변량 정규분포 (MVN)

### 9.1 정의

**Definition 9.1.**

$$\mathcal{N}(\mathbf{y}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{y} - \boldsymbol{\mu})\right)$$

### 9.2 핵심 성질

1. **주변분포도 정규**: $y_i \sim \mathcal{N}(\mu_i, \Sigma_{ii})$
2. **조건부분포도 정규**: $y_1 \mid y_2 \sim \mathcal{N}(\mu_{1|2}, \Sigma_{1|2})$
   - $\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(y_2 - \mu_2)$
   - $\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$
3. **선형 변환에 닫힘**: $\mathbb{E}[A\mathbf{x} + b] = A\boldsymbol{\mu} + b$, $\text{Cov}[A\mathbf{x}+b] = A\boldsymbol{\Sigma}A^\top$

### 9.3 선형 가우시안 모델 (베이지안 추론)

- Prior: $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$
- Likelihood: $\mathbf{y} \mid \mathbf{z} \sim \mathcal{N}(W\mathbf{z} + b, \boldsymbol{\Sigma}_y)$
- Posterior: $\mathbf{z} \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_{z|y}, \boldsymbol{\Sigma}_{z|y})$

이는 **베이즈 정리의 가우시안 버전**이며, Gaussian Process regression과 VAE의 reparameterization trick의 기초이다. 이 내용은 09장 베이지안 확률에서 심화된다.

---

## 10. 디랙 델타와 경험적 분포

### 10.1 디랙 델타

$\delta(x)$: $x = 0$에서만 0이 아니며, $\int \delta(x)\,dx = 1$. 가우시안의 분산을 0으로 보내면 디랙 델타로 수렴.

### 10.2 경험적 분포

데이터 $\{x_1, \ldots, x_N\}$이 주어졌을 때:

$$p_S(x) = \frac{1}{N}\sum_{i=1}^N \delta(x - x_i)$$

각 데이터 포인트에 동일한 가중치 $1/N$을 부여하는 이산 분포. MLE는 경험적 분포와 모델 분포 사이의 KL divergence를 최소화하는 것과 동치이다.

> **Glivenko-Cantelli 정리**: 경험적 CDF는 참 CDF로 균등 수렴한다. 이것이 유한 데이터로 학습이 가능한 근본적 이유.

---

## 11. Next-Token Prediction과 확률의 연결

### 11.1 Chain Rule 분해

문장 $(w_1, w_2, \ldots, w_T)$의 확률:

$$p(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T p(w_t \mid w_{1:t-1})$$

### 11.2 학습 목표

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{l_i}\log p_\theta(w_t^{(i)} \mid w_{1:t-1}^{(i)})$$

이것은 경험적 분포와 모델 분포 사이의 KL divergence 최소화와 동치이다. 이 장의 모든 개념 --- 조건부 확률, 범주형 분포, chain rule, 기댓값 --- 이 NTP에 집약되어 있다.

---

## 12. 흔한 오해와 주의점

| 오해 | 실제 | 교정 |
|:---|:---|:---|
| 확률밀도 $p(x) = 0.7$이면 확률 70% | 밀도는 확률이 아님. $p(x) > 1$ 가능. 확률은 구간의 적분 | $U[0, 0.5]$에서 밀도가 2인 예시 활용 |
| 비상관이면 독립 | 독립 $\Rightarrow$ 비상관이지만 역은 불성립 | 반례 직접 계산; 결합 정규에서만 동치 |
| $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$면 독립 | 이것은 비상관 조건일 뿐 | 모든 사건 조합에 대한 조건 필요 |
| 정규분포는 모든 데이터에 적합 | CLT는 유한 분산 i.i.d.에만 적용. Cauchy 등 heavy-tail에는 불가 | $\alpha$-stable 분포에서 $\alpha < 2$이면 분산 무한 |
| 조건부 독립이면 독립 | 두 개념은 완전히 별개 | 주사위 예시: 독립이지만 합 조건 하에 종속 |
| 기댓값은 실현 가능한 값 | 주사위 $\mathbb{E}[X] = 3.5$이지만 실제로 3.5 불가 | "장기적 평균"이지 실현값이 아님 |

---

## 13. 핵심 요약

```
┌─────────────────────────────────────────────────────────┐
│                    확률론 핵심 정리                       │
├─────────────────────────────────────────────────────────┤
│ 1. 확률의 언어: S → E ⊂ S → P(E) ∈ [0,1] → p: S→[0,1]│
│ 2. 조건부 확률: P(E|F) = P(E∩F)/P(F)                   │
│ 3. 독립: P(E∩F) = P(E)P(F). 조건부 독립과는 별개       │
│ 4. 확률변수: X: S → R. PMF(이산), PDF(연속)             │
│ 5. 핵심 분포: 베르누이 → 이항 → 범주형 → 다항           │
│ 6. 가우시안: MaxEnt + CLT에 의한 보편적 분포              │
│ 7. E[X+Y] = E[X]+E[Y] (항상). E[XY]=E[X]E[Y] (독립시) │
│ 8. 독립 ⊂ 비상관 (일반). 결합정규에서만 동치             │
│ 9. MVN: 주변분포, 조건부분포 모두 정규. 선형 변환에 닫힘 │
│10. NTP = chain rule + 범주형 분포 + CE 최소화            │
└─────────────────────────────────────────────────────────┘
```

**한 문장 요약**: 확률론은 불확실성을 수학적으로 다루는 유일한 일관된 체계이며, 딥러닝은 이 체계 위에서 데이터의 조건부 분포를 학습하는 것이다.

---

## 참고 문헌

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 3: Probability and Information Theory.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 2.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*.

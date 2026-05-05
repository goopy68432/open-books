---
title: "06. 베이즈 추론, MLE, MAP (Bayesian Inference, MLE, MAP)"
slug: 06-mle-map
order: 12
---

# 06. 베이즈 추론, MLE, MAP (Bayesian Inference, MLE, MAP)

> 대학원 딥러닝 중간고사 보충 자료
> 난이도별 3단계 구성: 입문 / 중급 / 고급

---

## 목차

1. [입문 (Beginner)](#입문-beginner)
2. [중급 (Intermediate)](#중급-intermediate)
3. [고급 (Advanced)](#고급-advanced)
4. [핵심 요약 및 시험 대비 체크리스트](#핵심-요약-및-시험-대비-체크리스트)

---

# 입문 (Beginner)

## 1.1 베이즈 정리 (Bayes' Theorem)

### 조건부 확률에서 출발

두 사건 $A$, $B$에 대해 조건부 확률은:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

마찬가지로:

$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

두 식에서 $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$이므로:

$$\boxed{P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}}$$

이것이 **베이즈 정리**이다.

### 각 항의 이름과 의미

$$\underbrace{P(\theta | D)}_{\text{사후확률 (posterior)}} = \frac{\overbrace{P(D | \theta)}^{\text{우도 (likelihood)}} \cdot \overbrace{P(\theta)}^{\text{사전확률 (prior)}}}{\underbrace{P(D)}_{\text{증거 (evidence)}}}$$

| 용어 | 의미 | 비유 |
|------|------|------|
| **사전확률** $P(\theta)$ | 데이터를 보기 전의 믿음 | 선입견, 경험 |
| **우도** $P(D\|\theta)$ | 모수가 $\theta$일 때 데이터가 관측될 확률 | 증거의 힘 |
| **사후확률** $P(\theta\|D)$ | 데이터를 본 후 갱신된 믿음 | 결론 |
| **증거** $P(D)$ | 데이터의 주변 확률 (정규화 상수) | 스케일링 |

### 직관적 공식

$$\text{사후확률} \propto \text{우도} \times \text{사전확률}$$

증거 $P(D)$는 $\theta$에 의존하지 않으므로 정규화 상수 역할만 한다.

## 1.2 의료 진단 예제 — 베이즈 정리의 위력

### 문제 설정

- 질병의 유병률: $P(\text{질병}) = 0.001$ (1000명 중 1명)
- 검사 양성 정확도 (민감도): $P(\text{양성}|\text{질병}) = 0.99$
- 검사 음성 정확도 (특이도): $P(\text{음성}|\text{건강}) = 0.99$, 즉 $P(\text{양성}|\text{건강}) = 0.01$

**질문**: 양성 판정을 받았을 때, 실제로 질병에 걸렸을 확률은?

### 풀이

$$P(\text{질병}|\text{양성}) = \frac{P(\text{양성}|\text{질병}) \cdot P(\text{질병})}{P(\text{양성})}$$

분모 계산 (전체 확률의 법칙):

$$P(\text{양성}) = P(\text{양성}|\text{질병})P(\text{질병}) + P(\text{양성}|\text{건강})P(\text{건강})$$
$$= 0.99 \times 0.001 + 0.01 \times 0.999 = 0.00099 + 0.00999 = 0.01098$$

따라서:

$$P(\text{질병}|\text{양성}) = \frac{0.99 \times 0.001}{0.01098} \approx 0.0902 \approx 9\%$$

### 핵심 교훈

99% 정확도의 검사에서 양성이 나왔는데, 실제 질병 확률은 겨우 9%이다! **사전확률(유병률)이 매우 낮으면**, 양성 결과의 대부분은 거짓 양성(false positive)이다.

이것이 베이즈 추론의 핵심: **사전 지식(prior)을 무시하면 잘못된 결론에 도달한다.**

## 1.3 최대 우도 추정 (Maximum Likelihood Estimation, MLE)

### 기본 아이디어

"데이터를 가장 잘 설명하는 모수를 찾자."

데이터 $D = \{x_1, x_2, \ldots, x_N\}$이 i.i.d.로 $P(x|\theta)$에서 생성되었다고 가정하면:

**우도 함수 (Likelihood)**:

$$L(\theta) = P(D|\theta) = \prod_{i=1}^{N} P(x_i | \theta)$$

**로그 우도 (Log-Likelihood)** (곱 → 합으로 변환):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{N} \log P(x_i | \theta)$$

**MLE**:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)$$

### 예제 1: 동전 던지기 (베르누이)

동전을 10번 던져 앞면 7번, 뒷면 3번이 나왔다.

$$L(\mu) = \mu^7 (1-\mu)^3$$

$$\ell(\mu) = 7\log\mu + 3\log(1-\mu)$$

$$\frac{d\ell}{d\mu} = \frac{7}{\mu} - \frac{3}{1-\mu} = 0$$

$$7(1-\mu) = 3\mu \implies 7 - 7\mu = 3\mu \implies \mu = 0.7$$

$$\boxed{\hat{\mu}_{\text{MLE}} = \frac{7}{10} = 0.7}$$

일반적으로 $N$번 중 $k$번 앞면이면: $\hat{\mu}_{\text{MLE}} = k/N$

### 예제 2: 가우시안 분포

데이터 $\{x_1, \ldots, x_N\}$이 $\mathcal{N}(\mu, \sigma^2)$에서 왔다고 하면:

$$\ell(\mu, \sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^{N}(x_i - \mu)^2$$

$\mu$에 대해 미분:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{N}(x_i - \mu) = 0 \implies \boxed{\hat{\mu}_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^{N} x_i = \bar{x}}$$

$\sigma^2$에 대해 미분:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{N}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{N}(x_i - \mu)^2 = 0$$

$$\boxed{\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2}$$

**주의**: MLE의 분산 추정은 $1/N$으로 나누므로 **편향(biased)**되어 있다. 불편 추정량은 $1/(N-1)$.

### MLE의 한계

- **데이터가 적을 때 과적합(overfitting)**: 동전 3번 던져 3번 다 앞면이면 $\hat{\mu} = 1.0$
- **사전 지식을 반영하지 못함**: 동전이 공정할 가능성이 높다는 상식을 무시

## 1.4 최대 사후 확률 추정 (Maximum A Posteriori, MAP)

### 기본 아이디어

"데이터뿐만 아니라 사전 지식도 반영하자."

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta | D) = \arg\max_\theta P(D|\theta) P(\theta)$$

로그를 취하면:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\sum_{i=1}^{N} \log P(x_i | \theta) + \log P(\theta)\right]$$

### MLE vs MAP

$$\underbrace{\hat{\theta}_{\text{MLE}}}_{\text{데이터만}} = \arg\max_\theta \sum \log P(x_i|\theta)$$

$$\underbrace{\hat{\theta}_{\text{MAP}}}_{\text{데이터 + 사전지식}} = \arg\max_\theta \left[\sum \log P(x_i|\theta) + \log P(\theta)\right]$$

MAP = MLE + **정규화 항(regularization term)**

### 직관적 비교

| | MLE | MAP |
|--|-----|-----|
| 수식 | $\max L(\theta)$ | $\max L(\theta) \cdot P(\theta)$ |
| 사전 지식 | 없음 (균등 사전) | 있음 |
| 데이터 적을 때 | 과적합 위험 | 사전확률이 안정화 |
| 데이터 많을 때 | MAP과 수렴 | MLE과 수렴 |

## 1.5 베타 분포 (Beta Distribution) — 동전 던지기의 사전분포

### 정의

$$\text{Beta}(\mu | a, b) = \frac{\mu^{a-1}(1-\mu)^{b-1}}{B(a, b)}$$

여기서 $B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$는 베타 함수.

### 모수의 직관적 의미

- $a$: "사전에 관찰한 앞면 횟수 + 1" 같은 의미
- $b$: "사전에 관찰한 뒷면 횟수 + 1" 같은 의미
- $a = b = 1$: 균등분포 (아무 정보 없음)
- $a = b = 2$: 0.5 근처에 약간 집중 (약한 사전지식)
- $a = b = 100$: 0.5에 강하게 집중 (강한 사전지식)

### 평균과 분산

$$\mathbb{E}[\mu] = \frac{a}{a+b}, \quad \text{Var}(\mu) = \frac{ab}{(a+b)^2(a+b+1)}$$

---

# 중급 (Intermediate)

## 2.1 켤레 사전분포 (Conjugate Prior)

### 정의

우도 함수 $P(D|\theta)$가 주어졌을 때, 사전분포 $P(\theta)$가 **켤레 사전분포**라 함은 사후분포 $P(\theta|D)$가 사전분포와 **같은 분포족**에 속하는 것이다.

### 왜 중요한가?

1. 사후분포의 해석이 쉽다 (같은 분포족이므로 모수만 갱신)
2. 순차적 갱신이 간단하다 (오늘의 사후 = 내일의 사전)
3. 해석적(closed-form) 풀이 가능

### 주요 켤레 쌍

| 우도 (Likelihood) | 켤레 사전 (Prior) | 사후 (Posterior) |
|-------------------|------------------|-----------------|
| 베르누이/이항 | 베타 Beta$(a, b)$ | Beta$(a', b')$ |
| 가우시안 (분산 기지) | 가우시안 $\mathcal{N}(\mu_0, \sigma_0^2)$ | 가우시안 |
| 가우시안 (평균 기지) | 역감마 Inv-Gamma | 역감마 |
| 포아송 | 감마 Gamma | 감마 |
| 카테고리컬/다항 | 디리클레 Dir | 디리클레 |

## 2.2 동전 던지기: Beta(2,2) 사전의 완전한 유도

### 문제 설정

- **사전분포**: $P(\mu) = \text{Beta}(\mu | 2, 2) = 6\mu(1-\mu)$
  - 이는 "동전이 대략 공정하다"는 약한 믿음을 반영
  - 사전 평균: $\mathbb{E}[\mu] = 2/(2+2) = 0.5$
- **데이터**: $N$번 던져서 앞면 $k$번, 뒷면 $N-k$번
- **우도**: $P(D|\mu) = \mu^k(1-\mu)^{N-k}$

### Step 1: 사후분포 계산

$$P(\mu | D) \propto P(D|\mu) \cdot P(\mu)$$

$$\propto \mu^k(1-\mu)^{N-k} \cdot \mu^{2-1}(1-\mu)^{2-1}$$

$$= \mu^{k+1}(1-\mu)^{N-k+1}$$

이것은 $\text{Beta}(k+2, N-k+2)$의 형태이다!

$$\boxed{P(\mu | D) = \text{Beta}(\mu | k+2, N-k+2)}$$

### Step 2: 구체적 숫자 예시

10번 던져서 7번 앞면:

| 추정 방법 | 결과 | 계산 |
|-----------|------|------|
| MLE | $\hat{\mu} = 0.700$ | $k/N = 7/10$ |
| MAP | $\hat{\mu} = 0.643$ | $\frac{k+a-1}{N+a+b-2} = \frac{7+1}{10+2} = 8/12$ |
| 사후 평균 | $\hat{\mu} = 0.643$ | $\frac{k+a}{N+a+b} = \frac{9}{14}$ |

**MAP 유도:**

$$\hat{\mu}_{\text{MAP}} = \arg\max_\mu \mu^{k+a-1}(1-\mu)^{N-k+b-1}$$

로그 취하고 미분:

$$\frac{k+a-1}{\mu} - \frac{N-k+b-1}{1-\mu} = 0$$

$$\hat{\mu}_{\text{MAP}} = \frac{k+a-1}{N+a+b-2}$$

### Step 3: 데이터 양에 따른 변화

| 데이터 | MLE | MAP (Beta(2,2)) | 차이 |
|--------|-----|------------------|------|
| 3번 중 3번 앞면 | 1.000 | 0.667 | 사전확률이 극단 방지 |
| 10번 중 7번 | 0.700 | 0.643 | 사전확률 영향 줄어듦 |
| 100번 중 70번 | 0.700 | 0.692 | 거의 동일 |
| 1000번 중 700번 | 0.700 | 0.699 | 사실상 동일 |

**핵심**: 데이터가 충분히 많으면 사전분포의 영향은 사라지고 MLE와 MAP이 수렴한다. 이를 **사후 일관성(posterior consistency)**이라 한다.

### Step 4: 순차적 갱신 (Sequential Update)

첫째 날: 5번 던져 3번 앞면
$$\text{사후}_1 = \text{Beta}(3+2, 2+2) = \text{Beta}(5, 4)$$

둘째 날: 5번 더 던져 4번 앞면
$$\text{사전}_2 = \text{사후}_1 = \text{Beta}(5, 4)$$
$$\text{사후}_2 = \text{Beta}(5+4, 4+1) = \text{Beta}(9, 5)$$

10번 중 7번 앞면의 결과와 동일: $\text{Beta}(7+2, 3+2) = \text{Beta}(9, 5)$

**순서에 무관(order-invariant)**하다!

## 2.3 가우시안 우도 + 가우시안 사전의 MAP

### 설정

- 데이터: $x_1, \ldots, x_N \sim \mathcal{N}(\mu, \sigma^2)$ (분산 $\sigma^2$ 기지)
- 사전: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

### 사후분포 유도

$$P(\mu | D) \propto P(D|\mu) P(\mu)$$

$$\propto \exp\left(-\frac{1}{2\sigma^2}\sum_{i}(x_i - \mu)^2\right) \cdot \exp\left(-\frac{(\mu - \mu_0)^2}{2\sigma_0^2}\right)$$

지수 부분을 $\mu$에 대해 정리하면 (완전제곱식):

$$P(\mu | D) = \mathcal{N}(\mu | \mu_N, \sigma_N^2)$$

여기서:

$$\frac{1}{\sigma_N^2} = \frac{N}{\sigma^2} + \frac{1}{\sigma_0^2} \quad \text{(정밀도의 합)}$$

$$\mu_N = \sigma_N^2 \left(\frac{N\bar{x}}{\sigma^2} + \frac{\mu_0}{\sigma_0^2}\right) \quad \text{(정밀도 가중 평균)}$$

### 해석

$$\mu_N = \underbrace{\frac{N/\sigma^2}{N/\sigma^2 + 1/\sigma_0^2}}_{\text{데이터 가중치}} \bar{x} + \underbrace{\frac{1/\sigma_0^2}{N/\sigma^2 + 1/\sigma_0^2}}_{\text{사전 가중치}} \mu_0$$

- $N \to \infty$: $\mu_N \to \bar{x}$ (데이터가 지배)
- $N = 0$: $\mu_N = \mu_0$ (사전분포만)
- $\sigma_0 \to \infty$ (무정보 사전): $\mu_N \to \bar{x}$ (MLE)
- 사후 분산 $\sigma_N^2$는 항상 사전 분산보다 작다 (데이터가 불확실성 감소)

## 2.4 정규화와 MAP의 관계

### L2 정규화 = 가우시안 사전의 MAP

가중치 $\mathbf{w}$에 대한 가우시안 사전:

$$P(\mathbf{w}) = \mathcal{N}(\mathbf{w} | \mathbf{0}, \tau^2 \mathbf{I}) \propto \exp\left(-\frac{\|\mathbf{w}\|^2}{2\tau^2}\right)$$

MAP 추정:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_\mathbf{w} \left[\sum_{i} \log P(y_i | x_i, \mathbf{w}) + \log P(\mathbf{w})\right]$$

$$= \arg\min_\mathbf{w} \left[-\sum_{i} \log P(y_i | x_i, \mathbf{w}) + \frac{1}{2\tau^2}\|\mathbf{w}\|^2\right]$$

$$= \arg\min_\mathbf{w} \left[\mathcal{L}_{\text{data}} + \lambda \|\mathbf{w}\|^2\right]$$

여기서 $\lambda = \frac{1}{2\tau^2}$.

이것이 바로 **L2 정규화(weight decay)**이다!

$$\boxed{\text{L2 정규화} = \text{가우시안 사전의 MAP 추정}}$$

### L1 정규화 = 라플라스 사전의 MAP

라플라스 사전:

$$P(w_j) = \frac{1}{2b}\exp\left(-\frac{|w_j|}{b}\right)$$

$$\log P(\mathbf{w}) = -\frac{1}{b}\sum_j |w_j| + \text{const}$$

MAP 추정:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_\mathbf{w} \left[\mathcal{L}_{\text{data}} + \frac{1}{b}\|\mathbf{w}\|_1\right]$$

$$\boxed{\text{L1 정규화 (LASSO)} = \text{라플라스 사전의 MAP 추정}}$$

### 왜 L1은 희소(sparse) 해를 만드는가?

라플라스 분포는 0에서 뾰족하다 (미분 불가). 이는 많은 가중치를 정확히 0으로 만드는 효과가 있다. 기하학적으로 보면:

- L2: 원형 등고선 → 축 위가 아닌 곳에서 만남 → 0이 되기 어려움
- L1: 다이아몬드형 등고선 → 꼭짓점(축 위)에서 만남 → 정확히 0이 되기 쉬움

### 정규화 강도의 베이즈적 해석

| 정규화 | 사전 분포 | 사전 분산 ($\tau^2$) | $\lambda$ |
|--------|----------|---------------------|-----------|
| 강한 정규화 | 좁은 사전 (작은 $\tau^2$) | 작다 | 크다 |
| 약한 정규화 | 넓은 사전 (큰 $\tau^2$) | 크다 | 작다 |
| 정규화 없음 | 균등 사전 ($\tau^2 \to \infty$) | $\infty$ | 0 |

## 2.5 MLE vs MAP vs Full Bayesian 비교

### 점추정 vs 분포 추정

| 방법 | 수식 | 산출물 |
|------|------|--------|
| **MLE** | $\arg\max_\theta P(D\|\theta)$ | 점추정 $\hat{\theta}$ |
| **MAP** | $\arg\max_\theta P(\theta\|D)$ | 점추정 $\hat{\theta}$ |
| **Full Bayesian** | $P(\theta\|D) = \frac{P(D\|\theta)P(\theta)}{P(D)}$ | 분포 $P(\theta\|D)$ |

### Full Bayesian의 예측

$$P(x_{\text{new}} | D) = \int P(x_{\text{new}} | \theta) P(\theta | D) \, d\theta$$

모든 가능한 $\theta$에 대해 가중 평균을 낸다.

**비교**: MLE/MAP은 하나의 $\hat{\theta}$만 사용:
$$P(x_{\text{new}} | D) \approx P(x_{\text{new}} | \hat{\theta})$$

Full Bayesian은 불확실성을 반영하므로 더 보수적인 예측을 한다.

### 동전 예시로 비교

데이터: 3번 던져 3번 다 앞면. "다음에 앞면이 나올 확률은?"

| 방법 | 추정값 | 다음 앞면 확률 |
|------|--------|---------------|
| MLE | $\hat{\mu} = 1.0$ | 1.0 (과도한 확신) |
| MAP (Beta(2,2)) | $\hat{\mu} = 0.667$ | 0.667 |
| Full Bayesian (Beta(2,2)) | 분포 Beta(5,2) | $\mathbb{E}[\mu] = 5/7 \approx 0.714$ |

Full Bayesian의 **사후 예측 분포(posterior predictive)**:

$$P(x=1|D) = \int_0^1 \mu \cdot \text{Beta}(\mu|5,2) \, d\mu = \frac{5}{7}$$

이것은 **라플라스 보정(Laplace smoothing)**과도 연결된다.

---

# 고급 (Advanced)

## 3.1 지수족에서의 베이즈 추론

### 지수족 우도와 켤레 사전의 일반론

지수족 우도:
$$P(x|\boldsymbol{\eta}) = h(x)\exp(\boldsymbol{\eta}^\top \mathbf{T}(x) - A(\boldsymbol{\eta}))$$

$N$개 i.i.d. 데이터의 우도:
$$P(D|\boldsymbol{\eta}) = \left(\prod_i h(x_i)\right) \exp\left(\boldsymbol{\eta}^\top \sum_i \mathbf{T}(x_i) - NA(\boldsymbol{\eta})\right)$$

**켤레 사전분포의 일반적 형태:**

$$P(\boldsymbol{\eta} | \boldsymbol{\chi}, \nu) \propto \exp\left(\boldsymbol{\eta}^\top \boldsymbol{\chi} - \nu A(\boldsymbol{\eta})\right)$$

여기서 $\boldsymbol{\chi}$는 "가상 충분 통계량", $\nu$는 "가상 관측 수".

**사후분포:**

$$P(\boldsymbol{\eta} | D) \propto \exp\left(\boldsymbol{\eta}^\top \left(\boldsymbol{\chi} + \sum_i \mathbf{T}(x_i)\right) - (\nu + N) A(\boldsymbol{\eta})\right)$$

갱신 규칙:
$$\boldsymbol{\chi}' = \boldsymbol{\chi} + \sum_i \mathbf{T}(x_i), \quad \nu' = \nu + N$$

**해석**: 사전분포의 $\boldsymbol{\chi}, \nu$는 "가상 데이터"에 해당하며, 실제 데이터의 충분 통계량이 단순히 더해진다.

## 3.2 베이즈 모델 선택과 오컴의 면도날

### 주변 우도 (Marginal Likelihood / Evidence)

$$P(D | \mathcal{M}) = \int P(D|\theta, \mathcal{M}) P(\theta | \mathcal{M}) \, d\theta$$

이것은 모델 $\mathcal{M}$의 데이터에 대한 **적합도**를 측정하며, 자동으로 모델 복잡도에 대한 벌칙을 포함한다.

### 베이즈 인수 (Bayes Factor)

두 모델 $\mathcal{M}_1$, $\mathcal{M}_2$ 비교:

$$\text{BF}_{12} = \frac{P(D|\mathcal{M}_1)}{P(D|\mathcal{M}_2)}$$

$$\underbrace{\frac{P(\mathcal{M}_1|D)}{P(\mathcal{M}_2|D)}}_{\text{사후 오즈}} = \text{BF}_{12} \times \underbrace{\frac{P(\mathcal{M}_1)}{P(\mathcal{M}_2)}}_{\text{사전 오즈}}$$

### 오컴의 면도날 (Occam's Razor)

주변 우도는 자동으로 **단순한 모델을 선호**한다.

직관적 이유: 복잡한 모델은 사전분포가 넓은 모수 공간에 퍼져 있어, 어떤 특정 데이터셋에 대한 $P(D|\mathcal{M})$이 오히려 작아진다. 반면 단순한 모델은 좁은 범위의 데이터를 잘 설명하므로, 해당 데이터가 관측되면 $P(D|\mathcal{M})$이 크다.

$$\int P(D|\theta) P(\theta) d\theta \approx P(D|\hat{\theta}) \cdot \underbrace{\Delta\theta}_{\text{사후 폭}} / \underbrace{\sigma_\theta}_{\text{사전 폭}}$$

복잡한 모델($\sigma_\theta$ 큼)은 $\Delta\theta/\sigma_\theta$가 작아서 자동 벌칙을 받는다.

### BIC (Bayesian Information Criterion)

주변 우도의 라플라스 근사에서 유도:

$$\text{BIC} = -2\log P(D|\hat{\theta}) + k\log N$$

여기서 $k$는 모수 수, $N$은 데이터 수. 작을수록 좋은 모델.

## 3.3 라플라스 근사 (Laplace Approximation)

### 아이디어

사후분포를 MAP 해 $\hat{\theta}$에서의 2차 테일러 전개로 가우시안 근사한다.

$$\log P(\theta|D) \approx \log P(\hat{\theta}|D) - \frac{1}{2}(\theta - \hat{\theta})^\top \mathbf{H}(\theta - \hat{\theta})$$

여기서 $\mathbf{H} = -\nabla^2 \log P(\theta|D)\big|_{\theta=\hat{\theta}}$는 MAP 해에서의 음의 헤시안.

따라서:

$$P(\theta|D) \approx \mathcal{N}(\theta | \hat{\theta}, \mathbf{H}^{-1})$$

### 딥러닝에서의 응용

- **Laplace Redux**: 훈련된 네트워크의 가중치를 MAP 해로 보고, 헤시안의 대각/KFAC 근사로 사후분포를 추정
- **불확실성 정량화**: 예측의 불확실성을 추정할 수 있다
- **Bayesian Neural Networks (BNN)**의 실용적 근사

## 3.4 변분 추론 (Variational Inference) 개요

### 문제

사후분포 $P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$에서 $P(D) = \int P(D|\theta)P(\theta)d\theta$가 계산 불가능한 경우가 대부분.

### 핵심 아이디어

다루기 쉬운 분포 $q(\theta)$로 $P(\theta|D)$를 근사한다:

$$q^*(\theta) = \arg\min_{q \in \mathcal{Q}} D_{\text{KL}}(q(\theta) \| P(\theta|D))$$

### ELBO (Evidence Lower Bound) 유도

$$\log P(D) = \log \int P(D, \theta) \, d\theta = \log \int q(\theta) \frac{P(D, \theta)}{q(\theta)} \, d\theta$$

Jensen의 부등식으로:

$$\geq \int q(\theta) \log \frac{P(D, \theta)}{q(\theta)} \, d\theta = \underbrace{\mathbb{E}_q[\log P(D|\theta)]}_{\text{재구성 항}} - \underbrace{D_{\text{KL}}(q(\theta) \| P(\theta))}_{\text{KL 항}} = \text{ELBO}$$

정리하면:

$$\log P(D) = \text{ELBO} + D_{\text{KL}}(q(\theta) \| P(\theta|D))$$

$D_{\text{KL}} \geq 0$이므로 ELBO는 $\log P(D)$의 하한이다. ELBO를 최대화하면 KL 다이버전스가 최소화된다.

### VAE (Variational Autoencoder)에서의 ELBO

$$\text{ELBO} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log P_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| P(\mathbf{z}))$$

- 첫째 항: 재구성 손실 (reconstruction loss)
- 둘째 항: KL 정규화 (잠재 변수가 사전분포에 가깝도록)

$q$와 $P(\mathbf{z})$가 모두 가우시안이면 KL 항을 해석적으로 계산할 수 있다 (05장의 두 가우시안 간 KL 공식 사용).

## 3.5 MCMC (Markov Chain Monte Carlo) 개요

### 동기

사후분포에서 직접 샘플링이 어려울 때, 마르코프 체인을 구성하여 사후분포로 수렴하는 샘플을 생성한다.

### Metropolis-Hastings 알고리즘

1. 현재 상태 $\theta_t$에서 제안 분포 $q(\theta'|\theta_t)$로부터 $\theta'$ 생성
2. 수용 확률 계산:
   $$\alpha = \min\left(1, \frac{P(\theta'|D) \cdot q(\theta_t|\theta')}{P(\theta_t|D) \cdot q(\theta'|\theta_t)}\right)$$
3. $u \sim \text{Uniform}(0,1)$. $u < \alpha$이면 $\theta_{t+1} = \theta'$, 아니면 $\theta_{t+1} = \theta_t$

**장점**: $P(D)$를 몰라도 된다 ($P(\theta|D)$의 비율만 필요)

### Hamiltonian Monte Carlo (HMC)

물리학의 해밀턴 역학을 이용하여 효율적으로 고차원 공간을 탐색:

- 운동량(momentum) 변수 도입
- 에너지 보존 법칙을 이용한 제안
- 기울기 정보를 활용하여 높은 수용률 달성
- NUTS (No-U-Turn Sampler): HMC의 자동 튜닝 버전 (Stan, PyMC3에서 사용)

## 3.6 경험적 베이즈와 하이퍼파라미터 최적화

### 경험적 베이즈 (Empirical Bayes / Type-II MLE)

사전분포의 하이퍼파라미터 $\boldsymbol{\alpha}$를 데이터로부터 추정:

$$\hat{\boldsymbol{\alpha}} = \arg\max_{\boldsymbol{\alpha}} P(D|\boldsymbol{\alpha}) = \arg\max_{\boldsymbol{\alpha}} \int P(D|\theta) P(\theta|\boldsymbol{\alpha}) \, d\theta$$

### 딥러닝에서의 연결

- **정규화 강도 $\lambda$ 선택**: 교차 검증 대신 주변 우도 최대화로 자동 결정 가능
- **Automatic Relevance Determination (ARD)**: 각 입력 차원마다 별도의 정규화 강도를 두고 경험적 베이즈로 최적화
- **Neural Architecture Search (NAS)와의 연결**: 모델 구조도 하이퍼파라미터로 보고 주변 우도로 선택

## 3.7 MLE, MAP, Bayesian의 점근적 행동

### 번스타인-폰 미제스 정리 (Bernstein-von Mises Theorem)

데이터가 충분히 많으면, 사전분포의 선택과 무관하게:

$$P(\theta|D) \xrightarrow{d} \mathcal{N}\left(\hat{\theta}_{\text{MLE}}, \frac{1}{N}\mathcal{I}(\theta^*)^{-1}\right)$$

여기서 $\mathcal{I}(\theta^*)$은 참값에서의 피셔 정보.

**의미**:
1. 사후분포는 점근적으로 가우시안이 된다
2. 사후분포의 중심은 MLE로 수렴한다
3. MAP도 MLE로 수렴한다
4. 사전분포의 영향은 사라진다 ($O(1/N)$)
5. 사후 분산은 크라메르-라오 하한을 달성한다

### 유한 데이터에서의 차이

| 데이터 양 | MLE | MAP | Full Bayesian |
|-----------|-----|-----|---------------|
| 매우 적음 ($N < 10$) | 과적합/불안정 | 사전에 의존 | 불확실성 반영, 가장 안정적 |
| 적당함 ($N \sim 100$) | 합리적 | 약간의 정규화 효과 | 불확실성 정보 제공 |
| 매우 많음 ($N \to \infty$) | 최적 | $\approx$ MLE | $\approx$ MLE, 좁은 사후분포 |

## 3.8 딥러닝에서의 베이즈적 관점 총정리

### 드롭아웃(Dropout)의 베이즈적 해석

Gal & Ghahramani (2016): 드롭아웃을 적용한 훈련은 **변분 추론**의 한 형태이다.

- $q(\mathbf{W})$: 드롭아웃 마스크에 의한 가중치 분포
- 테스트 시 드롭아웃을 여러 번 적용하면 (MC Dropout) → 사후 예측 분포의 근사 → **불확실성 추정** 가능

### 앙상블(Ensemble)의 베이즈적 해석

여러 모델의 예측을 평균내는 것은 사후분포에 대한 **몬테카를로 적분**으로 볼 수 있다:

$$P(y|x, D) = \int P(y|x, \theta) P(\theta|D) d\theta \approx \frac{1}{M}\sum_{m=1}^{M} P(y|x, \hat{\theta}_m)$$

여기서 $\hat{\theta}_m$은 서로 다른 초기화/데이터 부분집합으로 훈련된 모델.

### Stochastic Weight Averaging - Gaussian (SWAG)

SGD 궤적의 1, 2차 모멘트를 추적하여 가우시안 사후 근사를 얻는다:
- 저비용으로 불확실성 추정 가능
- 라플라스 근사보다 실용적

### 정규화 기법들의 통합적 베이즈 관점

| 정규화 기법 | 베이즈적 해석 |
|------------|--------------|
| L2 (weight decay) | 가우시안 사전 $\mathcal{N}(0, \tau^2)$ |
| L1 (LASSO) | 라플라스 사전 |
| Dropout | 변분 추론 (스파이크-앤-슬랩 사전의 근사) |
| Data augmentation | 사전분포에 불변성(invariance) 반영 |
| Early stopping | 사전분포를 초기화 값으로 설정한 MAP의 근사 |
| Batch Normalization | 사후분포의 곡률(curvature) 개선 |
| Label smoothing | 카테고리컬 분포에 디리클레 사전 적용 |

---

# 핵심 요약 및 시험 대비 체크리스트

## 공식 정리표

| 개념 | 공식 | 핵심 포인트 |
|------|------|------------|
| 베이즈 정리 | $P(\theta\|D) \propto P(D\|\theta)P(\theta)$ | 사후 $\propto$ 우도 $\times$ 사전 |
| MLE | $\arg\max_\theta \sum\log P(x_i\|\theta)$ | 데이터만으로 점추정 |
| MAP | $\arg\max_\theta [\sum\log P(x_i\|\theta) + \log P(\theta)]$ | MLE + 정규화 |
| 베르누이 MLE | $\hat{\mu} = k/N$ | 단순 비율 |
| Beta-Bernoulli MAP | $\hat{\mu} = (k+a-1)/(N+a+b-2)$ | 가상 관측치 반영 |
| 가우시안 사후 평균 | 정밀도 가중 평균 | 데이터와 사전의 절충 |
| L2 정규화 | $\lambda\|\mathbf{w}\|^2$ ↔ $\mathcal{N}(0, \frac{1}{2\lambda})$ 사전 | 가우시안 MAP |
| L1 정규화 | $\lambda\|\mathbf{w}\|_1$ ↔ Laplace$(0, 1/\lambda)$ 사전 | 라플라스 MAP |
| ELBO | $\log P(D) \geq \mathbb{E}_q[\log P(D\|\theta)] - D_{\text{KL}}(q\|P)$ | 변분 추론의 핵심 |

## 시험에 자주 나오는 유도/증명

1. **베이즈 정리**를 조건부 확률에서 유도
2. **베르누이 MLE**: 로그 우도 미분하여 $\hat{\mu} = k/N$
3. **가우시안 MLE**: $\hat{\mu}, \hat{\sigma}^2$ 유도
4. **Beta-Bernoulli 켤레**: 사전 Beta(a,b) + 데이터 → 사후 Beta(a+k, b+N-k)
5. **L2 정규화 = 가우시안 MAP** 유도
6. **ELBO 유도** (Jensen 부등식 이용)

## 개념적 이해 체크

- [ ] MLE가 왜 과적합하기 쉬운지 설명할 수 있는가?
- [ ] MAP이 MLE와 어떻게 다르고, 왜 정규화 효과가 있는지?
- [ ] Full Bayesian이 점추정과 무엇이 다른지?
- [ ] 데이터가 많아지면 MLE, MAP, Bayesian이 왜 수렴하는지?
- [ ] 드롭아웃이 왜 베이즈적으로 해석되는지?
- [ ] KL 다이버전스가 ELBO에서 어떤 역할을 하는지?
- [ ] 정규화 강도 $\lambda$와 사전분포 분산의 관계?

## 딥러닝 연결 포인트

- **손실함수 = 음의 로그 우도 (NLL)** → MLE
- **손실함수 + 정규화 = NLL + 로그 사전확률** → MAP
- **VAE = ELBO 최대화** → 변분 추론
- **MC Dropout** → 근사 베이즈 추론으로 불확실성 추정
- **앙상블** → 사후분포에 대한 몬테카를로 적분
- **사전훈련(pre-training)** → 사전분포의 설정
- **전이학습(transfer learning)** → 사전 지식을 사전분포로 인코딩

---

> **참고문헌**
> - Bishop, *Pattern Recognition and Machine Learning*, Ch. 2-3
> - Murphy, *Probabilistic Machine Learning: An Introduction*, Ch. 4-5
> - Murphy, *Probabilistic Machine Learning: Advanced Topics*, Ch. 4-6
> - Goodfellow et al., *Deep Learning*, Ch. 5
> - Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
> - Blundell et al., "Weight Uncertainty in Neural Networks", ICML 2015

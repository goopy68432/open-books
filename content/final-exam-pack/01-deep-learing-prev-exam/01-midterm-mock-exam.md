---
title: "Deep Learning 중간고사 모의시험 (문제 + 풀이 + 필수 개념)"
slug: midterm-mock-exam
order: 1
---

# Deep Learning 중간고사 모의시험 (문제 + 풀이 + 필수 개념)

> **기출 분석 기반**: 학부 기출문제 정리 page_001~003의 중간고사 범위를 100% 커버
> **채점 기준**: "답만 적으면 0점" — 모든 유도 단계에 "왜"를 서술할 것
> **시험 전 TIP**: 시험직전 요약해주는데서 다 나오는 편. Summary 참석(힌트 많음). 증명 식의 전개 과정. 큰 5문제 + 소문항 여러 개.

---

## 기출 분석: 중간고사 출제 영역

| # | 영역 | 세부 출제 포인트 | 예상 배점 |
|---|------|----------------|---------|
| 1 | 확률분포 — 평균·분산 계산 | 균등분포, 정규분포, 포아송, pdf, E(X^n) | 15점 |
| 2 | MLE / MAP / Bayesian | MLE 유도, MAP 유도, prior update, posterior | 20점 |
| 3 | 손실함수 관계 서술 | MSE↔NLL, NLL↔KL, CE↔KL 관계 증명 | 15점 |
| 4 | KL Divergence 계산 | KL ≥ 0 증명, 정규분포 KL 계산 | 10점 |
| 5 | 행렬 미분 / Backpropagation | softmax 야코비안, chain rule, loss 미분값 구하기 | 20점 |
| 6 | 학습률 조건 | η < 2/λ_max(A) 증명 | 10점 |
| 7 | Pooling | Average Pooling을 행렬로 표현 | 5점 |
| 8 | 기타 | Quadratic 관련, 평균 분산 유도 | 5점 |

---

# 문제 1. [15점] 확률분포 — 평균·분산 계산

## 문제 1-(a) [5점]

균등분포(Uniform distribution) U[a, b]의 평균과 분산을 구하시오. pdf를 먼저 쓰고, 정의로부터 유도할 것.

## 문제 1-(b) [5점]

정규분포 X ~ N(0, 1)에 대해 다음을 구하시오:
- E(X), E(X²), E(X^{2n-1}), E(X^{2n})

각각의 결과가 왜 그 값이 되는지 직관적으로도 설명하시오.

## 문제 1-(c) [5점]

구간 [a, b] 내에서 mean과 variance를 구하고, x^{2n-1}과 x^{2n}의 기댓값을 구하시오.
(힌트: U[a,b]의 pdf = 1/(b-a))

---

### 풀이 1-(a): 균등분포의 평균과 분산

**필수 개념**: 균등분포는 구간 [a,b]에서 모든 값이 동일한 확률밀도를 가지는 분포. "아무 정보도 없을 때" 가장 자연스러운 가정 (최대 엔트로피 원리).

**pdf**:
$$f(x) = \begin{cases} \frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$$

**평균**:
$$E[X] = \int_a^b x \cdot \frac{1}{b-a} dx = \frac{1}{b-a}\left[\frac{x^2}{2}\right]_a^b = \frac{b^2 - a^2}{2(b-a)} = \frac{(b+a)(b-a)}{2(b-a)} = \frac{a+b}{2}$$

> **왜 (a+b)/2인가**: 대칭인 구간의 중점. 직관과 일치.

**분산**: Var(X) = E[X²] - (E[X])²

$$E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a} dx = \frac{1}{b-a}\left[\frac{x^3}{3}\right]_a^b = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}$$

$$\text{Var}(X) = \frac{a^2+ab+b^2}{3} - \left(\frac{a+b}{2}\right)^2 = \frac{a^2+ab+b^2}{3} - \frac{a^2+2ab+b^2}{4}$$

통분(분모 12):
$$= \frac{4(a^2+ab+b^2) - 3(a^2+2ab+b^2)}{12} = \frac{4a^2+4ab+4b^2-3a^2-6ab-3b^2}{12} = \frac{a^2-2ab+b^2}{12} = \frac{(b-a)^2}{12}$$

$$\boxed{E[X] = \frac{a+b}{2}, \quad \text{Var}(X) = \frac{(b-a)^2}{12}}$$

---

### 풀이 1-(b): 표준정규분포의 모멘트

**필수 개념**: 정규분포의 대칭성(기함수의 적분=0)과 가우시안 적분 기법.

**E(X) = 0**: N(0,1)은 원점 대칭. x·f(x)는 기함수이므로 적분=0.

**E(X²) = 1**: Var(X) = E[X²] - (E[X])² = E[X²] - 0 = 1 (∵ 분산=1)

또는 직접 계산:
$$E[X^2] = \int_{-\infty}^{\infty} x^2 \cdot \frac{1}{\sqrt{2\pi}}e^{-x^2/2} dx$$

부분적분 또는 감마함수로: = 1 ✓

**E(X^{2n-1}) = 0** (모든 홀수 모멘트):
x^{2n-1} · f(x)는 기함수(홀수 × 짝수 = 홀수). 대칭 구간 적분 = 0.

> **왜 0인가**: 양수 영역의 기여와 음수 영역의 기여가 정확히 상쇄. N(0,1)은 완벽하게 대칭.

**E(X^{2n})** (짝수 모멘트):
$$E[X^{2n}] = \frac{(2n)!}{2^n \cdot n!} = (2n-1)!! = 1 \cdot 3 \cdot 5 \cdots (2n-1)$$

이는 이중계승(double factorial)으로 표현. 예: E[X⁴] = 3, E[X⁶] = 15, E[X⁸] = 105.

> 유도: 적률생성함수 M(t) = exp(t²/2)를 t에 대해 2n번 미분하고 t=0 대입.

$$\boxed{E[X^{2n-1}] = 0, \quad E[X^{2n}] = (2n-1)!! = \frac{(2n)!}{2^n n!}}$$

---

### 풀이 1-(c): U[a,b]에서 x^k의 기댓값

$$E[X^k] = \int_a^b x^k \cdot \frac{1}{b-a} dx = \frac{1}{b-a}\cdot\frac{b^{k+1} - a^{k+1}}{k+1}$$

특히:
- E[X^{2n-1}] = (b^{2n} - a^{2n}) / (2n(b-a))
- E[X^{2n}] = (b^{2n+1} - a^{2n+1}) / ((2n+1)(b-a))

> **주의**: 정규분포와 달리 균등분포는 대칭 구간 [-a,a]가 아닌 한 홀수 모멘트도 0이 아님.

---

# 문제 2. [20점] MLE / MAP / Bayesian Estimation

## 문제 2-(a) [5점]

동전 던지기 실험에서 n = 10회 중 k = 7회 앞면이 나왔다.
앞면 확률 θ에 대해 MLE θ̂_ML을 구하시오. 전체 유도 과정(우도→로그→미분=0)을 빠짐없이 서술하시오.

## 문제 2-(b) [5점]

사전분포 Prior = θ^m(1-θ)^m일 때, MAP 추정량 θ̂_MAP을 구하시오. (m = 2, n = 6, k = 0)

이 결과가 MLE와 어떻게 다른지, 사전분포가 어떤 역할을 하는지 설명하시오.

## 문제 2-(c) [5점]

Prior가 Beta(α, β)이고 데이터가 k번 앞면, n-k번 뒷면일 때:
- Posterior 분포를 구하시오.
- Posterior가 다시 Beta 분포가 되는 이유(conjugacy)를 설명하시오.
- Prior update 과정을 직관적으로 설명하시오.

## 문제 2-(d) [5점]

Likelihood p(D|θ), Posterior estimator p(θ|D)를 각각 쓰시오.
MLE와 MAP의 정의를 수식으로 쓰고, 두 방법의 차이를 "균등 사전분포"를 이용하여 설명하시오.

---

### 풀이 2-(a): 베르누이 MLE 완전 유도

**필수 개념**: MLE는 "관찰된 데이터를 가장 잘 설명하는 파라미터"를 찾는 방법. i.i.d 가정 → 곱 → 로그 → 미분=0.

**Step 1 — 단일 시행의 확률**:
$$p(x_i|\theta) = \theta^{x_i}(1-\theta)^{1-x_i}, \quad x_i \in \{0, 1\}$$

**Step 2 — i.i.d 가정으로 우도 함수** (왜: 각 동전 던지기가 독립이므로 결합확률=곱):
$$L(\theta) = \prod_{i=1}^{10} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^7(1-\theta)^3$$

**Step 3 — 로그 변환** (왜: 곱→합으로 변환하여 미분 계산 용이):
$$\ell(\theta) = 7\log\theta + 3\log(1-\theta)$$

**Step 4 — 미분=0** (왜: ℓ(θ)는 θ∈(0,1)에서 오목함수이므로 극대점=전역최대):
$$\frac{d\ell}{d\theta} = \frac{7}{\theta} - \frac{3}{1-\theta} = 0$$

$$7(1-\theta) = 3\theta \implies 7 - 7\theta = 3\theta \implies 7 = 10\theta$$

$$\boxed{\hat{\theta}_{ML} = \frac{7}{10} = 0.7}$$

> **직관**: "관찰된 빈도 = 추정된 확률". 10번 중 7번 앞면이면 θ̂ = 0.7. 자연스러운 결과.

---

### 풀이 2-(b): MAP 추정 (m=2, n=6, k=0)

**필수 개념**: MAP = MLE + 사전분포. log posterior = log likelihood + log prior + C.

k=0 (6번 모두 뒷면), Prior = θ^2(1-θ)^2

**사후 확률 ∝ 우도 × 사전**:
$$p(\theta|D) \propto \theta^0(1-\theta)^6 \cdot \theta^2(1-\theta)^2 = \theta^2(1-\theta)^8$$

**로그 변환**:
$$\log p(\theta|D) \propto 2\log\theta + 8\log(1-\theta)$$

**미분=0**:
$$\frac{2}{\theta} - \frac{8}{1-\theta} = 0 \implies 2(1-\theta) = 8\theta \implies 2 = 10\theta$$

$$\boxed{\hat{\theta}_{MAP} = \frac{2}{10} = 0.2}$$

**MLE와의 비교**:
- MLE: θ̂_ML = k/n = 0/6 = **0** (극단적! 앞면이 절대 안 나온다고 추정)
- MAP: θ̂_MAP = **0.2** (사전분포가 θ=0에서 벗어나도록 당김)

**사전분포의 역할**: θ^2(1-θ)^2는 θ=0과 θ=1에서 0이 되어 극단값을 억제. 마치 "의사 데이터(pseudo-data)" 2개의 앞면 + 2개의 뒷면을 추가한 효과. 이것이 **정규화(regularization)**.

> **주의**: 기출에서 "이 문제는 이론 불가능 점 체크 필요"라고 언급. θ^m에서 m이 정수가 아니면 Beta 분포와의 관계를 확인해야 함.

---

### 풀이 2-(c): Conjugate Prior와 Prior Update

**필수 개념**: 켤레 사전분포(conjugate prior)는 사후분포가 사전분포와 같은 분포족에 속하게 만드는 사전분포.

Prior: θ ~ Beta(α, β), pdf ∝ θ^{α-1}(1-θ)^{β-1}

Likelihood: L(θ) = θ^k(1-θ)^{n-k}

**Posterior**:
$$p(\theta|D) \propto L(\theta) \cdot p(\theta) = \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}$$
$$= \theta^{k+\alpha-1}(1-\theta)^{n-k+\beta-1}$$

이것은 **Beta(k+α, n-k+β)** 분포의 커널!

$$\boxed{p(\theta|D) = \text{Beta}(\alpha + k, \quad \beta + n - k)}$$

**Conjugacy(켤레성)의 의미**: Beta 사전 + 베르누이 우도 → Beta 사후. 분포의 "형태"가 보존됨.

**Prior Update 직관**:
- Prior: Beta(α, β) → "α-1번 앞면, β-1번 뒷면을 이미 본 것"과 같은 효과
- 데이터: k번 앞면, n-k번 뒷면 관찰
- Posterior: Beta(α+k, β+n-k) → "총 (α-1+k)번 앞면, (β-1+n-k)번 뒷면" 효과

> **백엔드 관점**: 캐시 업데이트와 유사. 기존 캐시(prior)에 새 데이터를 반영(update)하여 갱신된 캐시(posterior)를 얻음. 전체를 재계산하지 않고 증분 업데이트.

---

### 풀이 2-(d): MLE vs MAP 정의

**Likelihood**: $p(D|\theta) = \prod_{i=1}^n p(x_i|\theta)$

**Posterior**: $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$ (베이즈 정리)

**MLE**: $\hat{\theta}_{ML} = \arg\max_\theta p(D|\theta)$ — 우도만 최대화

**MAP**: $\hat{\theta}_{MAP} = \arg\max_\theta p(\theta|D) = \arg\max_\theta p(D|\theta)p(\theta)$ — 사후확률 최대화

**균등 사전분포일 때**: p(θ) = C (상수)이면
$$\arg\max_\theta p(D|\theta) \cdot C = \arg\max_\theta p(D|\theta)$$

따라서 **MAP = MLE**. 균등 사전분포는 "어떤 θ도 선호하지 않는" 무정보 사전분포이므로, 데이터만으로 판단하는 MLE와 동일해진다.

---

# 문제 3. [15점] 손실함수 관계 서술

## 문제 3-(a) [5점]

MSE와 NLL의 관계를 설명하시오. 해당하는 분포 서술을 포함하시오.

## 문제 3-(b) [5점]

NLL과 KL divergence의 관계를 서술하시오. CE(p,q) = KL(p∥q) + H(p)를 유도하시오.

## 문제 3-(c) [5점]

MSE, MLE, NLL, KL의 전체 관계를 설명하고 증명하시오.
"왜 MSE를 최소화하는 것이 KL 발산을 최소화하는 것과 같은가?"에 답하시오.

---

### 풀이 3-(a): MSE ↔ NLL

**필수 개념**: "어떤 확률 분포를 가정하느냐"에 따라 NLL이 특정 손실함수가 됨.

**가우시안 가정**: y_i ~ N(h_w(x_i), σ²)

$$\text{NLL}(w) = -\sum_{i=1}^n \log p(y_i|x_i;w)$$
$$= -\sum_{i=1}^n \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - h_w(x_i))^2}{2\sigma^2}\right]$$
$$= \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - h_w(x_i))^2$$

$$= \frac{n}{2\sigma^2} \cdot \text{MSE}(w) + C$$

**결론**: 가우시안 노이즈 가정 하에서 NLL 최소화 = MSE 최소화.
- **가우시안 → MSE** (회귀)
- **베르누이 → BCE** (이진 분류)
- **범주형 → CE** (다중 분류)

> **왜 이 관계가 중요한가**: MSE를 "그냥 좋은 손실"로 쓰는 것이 아니라, "가우시안 노이즈를 가정했을 때의 MLE"로서 확률론적 정당성을 가진다.

---

### 풀이 3-(b): NLL ↔ KL Divergence

**CE 정의로부터 분해**:
$$CE(p,q) = -\sum_x p(x)\log q(x)$$

$$= -\sum_x p(x)\log q(x) \underbrace{+ \sum_x p(x)\log p(x) - \sum_x p(x)\log p(x)}_{\text{= 0 (더하고 빼기)}}$$

$$= \sum_x p(x)\log\frac{p(x)}{q(x)} + \left(-\sum_x p(x)\log p(x)\right)$$

$$\boxed{CE(p,q) = KL(p\|q) + H(p)}$$

경험적 분포 p_S에서:
$$\text{NLL}(\theta) = n \cdot CE(p_S, q_\theta) = n \cdot [KL(p_S\|q_\theta) + H(p_S)]$$

H(p_S)는 θ에 무관 → **NLL 최소화 = CE 최소화 = KL 최소화**

---

### 풀이 3-(c): 대통일 — MSE = MLE = NLL = KL

**체인**:
```
가우시안 가정 → MLE 수행 → NLL 최소화 → MSE 최소화
     ↕                                    ↕
확률 모델링                            손실 최소화
     ↕                                    ↕
경험적 분포 p_S vs 모델 q_θ  ←→  KL(p_S ∥ q_θ) 최소화
```

**증명 요약**:
1. **MLE = NLL 최소화**: argmax p(D|θ) = argmin(-log p(D|θ)) (로그 단조증가)
2. **NLL = CE** (×n): NLL = -Σlog q_θ(y_i|x_i) = n·CE(p_S, q_θ)
3. **CE = KL + H**: CE(p_S, q_θ) = KL(p_S∥q_θ) + H(p_S)
4. **CE 최소화 = KL 최소화**: H(p_S)는 상수
5. **가우시안 → NLL = MSE**: NLL(w) = n/(2σ²)·MSE(w) + C

∴ **MSE 최소화 = MLE = NLL 최소화 = KL 최소화**

> **시험에서 이렇게 답하라**: "가우시안 노이즈를 가정하면 MLE가 MSE 최소화와 동치이고, NLL은 경험적 분포와 모델 분포 간 CE의 n배이며, CE = KL + H에서 H가 상수이므로 CE 최소화 = KL 최소화. 따라서 MSE를 최소화하는 것은 곧 모델 분포를 데이터 분포에 KL 의미에서 가장 가깝게 만드는 것이다."

---

# 문제 4. [10점] KL Divergence 계산

## 문제 4-(a) [5점]

KL divergence가 항상 non-negative임을 증명하시오. (KL(p∥q) ≥ 0)

## 문제 4-(b) [5점]

P ~ N(μ₁, 1), Q ~ N(μ₂, 1) (분산은 1로 같고 평균만 다름)일 때 KL(P∥Q)를 구하시오.
KL divergence에서 p와 q가 주어졌을 때 값 계산하시오.

---

### 풀이 4-(a): KL ≥ 0 증명 (깁스 부등식)

**필수 개념**: 젠센 부등식 — f가 볼록이면 E[f(X)] ≥ f(E[X]). -log는 볼록함수.

$$KL(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = -\sum_x p(x)\log\frac{q(x)}{p(x)}$$

-log는 볼록함수이므로 젠센 부등식 적용:

$$-\sum_x p(x)\log\frac{q(x)}{p(x)} \geq -\log\left(\sum_x p(x)\cdot\frac{q(x)}{p(x)}\right)$$

$$= -\log\left(\sum_x q(x)\right) = -\log(1) = 0$$

$$\boxed{KL(p\|q) \geq 0}$$

등호 조건: p(x) = q(x) for all x (두 분포가 동일할 때).

> **왜 ≥ 0인가**: KL은 "정보 손실량"을 측정. 같은 분포면 손실 없음(0). 다르면 반드시 손실 발생(>0).

---

### 풀이 4-(b): 정규분포 KL 계산

**일반 공식** (단변량): P = N(μ₁, σ₁²), Q = N(μ₂, σ₂²)

$$KL(P\|Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**σ₁ = σ₂ = 1 대입**:

$$KL(P\|Q) = \log\frac{1}{1} + \frac{1 + (\mu_1-\mu_2)^2}{2} - \frac{1}{2}$$
$$= 0 + \frac{1}{2} + \frac{(\mu_1-\mu_2)^2}{2} - \frac{1}{2}$$

$$\boxed{KL(P\|Q) = \frac{(\mu_1-\mu_2)^2}{2}}$$

> **핵심 통찰**: 분산이 같으면 KL = MSE/2. 이것이 "가우시안 가정 하에서 KL 최소화 = MSE 최소화"의 직접적 증거.

**유도 과정** (정의로부터):

$$KL(P\|Q) = \int p(x)\log\frac{p(x)}{q(x)}dx = \int p(x)\left[\log p(x) - \log q(x)\right]dx$$

$$= -H(P) - \int p(x)\log q(x)dx$$

H(P) = ½log(2πe·1) = ½log(2πe)

$$\int p(x)\log q(x)dx = \int p(x)\left[-\frac{1}{2}\log(2\pi) - \frac{(x-\mu_2)^2}{2}\right]dx$$
$$= -\frac{1}{2}\log(2\pi) - \frac{1}{2}E_P[(X-\mu_2)^2]$$

$$E_P[(X-\mu_2)^2] = \text{Var}_P(X) + (E_P[X]-\mu_2)^2 = 1 + (\mu_1-\mu_2)^2$$

따라서:
$$KL = -\frac{1}{2}\log(2\pi e) + \frac{1}{2}\log(2\pi) + \frac{1+(\mu_1-\mu_2)^2}{2}$$
$$= -\frac{1}{2} + \frac{1+(\mu_1-\mu_2)^2}{2} = \frac{(\mu_1-\mu_2)^2}{2} \quad \checkmark$$

---

# 문제 5. [20점] 행렬 미분 / Backpropagation

## 문제 5-(a) [4점]

-log σ(Az + b)를 A, z, b 각각에 대해 미분하시오. (σ는 시그모이드 함수, 소문항 4개)

## 문제 5-(b) [4점]

[softmax(α)]_bₑ, α = W^T z + b일 때, softmax를 α, W, b에 대해 미분하시오.

## 문제 5-(c) [4점]

주어진 softmax 함수값에 대해 기저 벡터, 행렬 미분하시오.

## 문제 5-(d) [4점]

softmax 함수를 이용해서 행렬을 미분하는 문제. Chain Rule을 이용한 전미분 결과 구하시오.
(softmax 미분 → 스칼라 → 행렬 미분 과정)

## 문제 5-(e) [4점]

2층 신경망 f(x) = W₂σ(W₁x + b₁) + b₂의 역전파. 파라미터 W₁, W₂에 대한 loss ∂L/∂W₁, ∂L/∂W₂를 구하시오.
(L = ½∥f(x) - y∥²)

---

### 풀이 5-(a): -log σ(Az+b) 미분

**필수 개념**: 시그모이드의 미분 σ'(x) = σ(x)(1-σ(x)). 연쇄법칙의 반복 적용.

s = Az + b, σ = σ(s) = 1/(1+e^{-s})로 놓자.

**f = -log σ(s)**

**∂f/∂s 먼저 구하기**:
$$\frac{\partial f}{\partial s} = -\frac{\sigma'(s)}{\sigma(s)} = -\frac{\sigma(s)(1-\sigma(s))}{\sigma(s)} = -(1-\sigma(s)) = \sigma(s) - 1$$

**∂f/∂b**: s = Az + b이므로 ∂s/∂b = I
$$\frac{\partial f}{\partial b} = \frac{\partial f}{\partial s}\frac{\partial s}{\partial b} = (\sigma(s) - 1) \cdot I = \sigma(Az+b) - 1$$

**∂f/∂z**: ∂s/∂z = A
$$\frac{\partial f}{\partial z} = \frac{\partial f}{\partial s}\frac{\partial s}{\partial z} = (\sigma(s) - 1) \cdot A = (\sigma(Az+b) - 1)A$$

**∂f/∂A**: s = Az + b에서 ∂s/∂A는 텐서. 스칼라 f에 대해:
$$\frac{\partial f}{\partial A} = (\sigma(s) - 1) \cdot z^T$$

(외적 형태: 오차 신호 × 입력의 전치)

> **핵심 패턴**: 모든 미분이 "σ(s)-1" (= -sigmoid의 보수)을 포함. 이것이 오차 신호(δ)이고, 뒤에 곱해지는 것은 "해당 파라미터가 s에 기여하는 방식".

---

### 풀이 5-(b): softmax(W^Tz + b) 미분

**필수 개념**: 소프트맥스 야코비안 ∂p/∂α = diag(p) - pp^T

α = W^T z + b, p = softmax(α)

**∂p/∂α** = diag(p) - pp^T (소프트맥스 야코비안)

**∂p/∂b**: α = W^Tz + b이므로 ∂α/∂b = I
$$\frac{\partial p}{\partial b} = \frac{\partial p}{\partial \alpha}\frac{\partial \alpha}{\partial b} = (\text{diag}(p) - pp^T) \cdot I = \text{diag}(p) - pp^T$$

**∂p/∂W**: ∂α/∂W에서 α = W^Tz이므로 이것은 텐서.
특정 클래스 c에 대해 ∂p_c/∂W를 구하면:

$$\frac{\partial p_c}{\partial W} = \sum_j \frac{\partial p_c}{\partial \alpha_j}\frac{\partial \alpha_j}{\partial W}$$

∂α_j/∂W = z · e_j^T (j번째 열 방향) 이므로:

$$\frac{\partial p_c}{\partial W} = z \cdot \left[\frac{\partial p_c}{\partial \alpha}\right]^T = z \cdot [(\text{diag}(p) - pp^T)_{c,:}]$$

---

### 풀이 5-(e): 2층 신경망 역전파

**필수 개념**: 연쇄법칙 + 외적 형태의 가중치 갱신

f(x) = W₂σ(W₁x + b₁) + b₂, L = ½∥f(x)-y∥²

**순전파**:
- z₁ = W₁x + b₁
- a₁ = σ(z₁)
- z₂ = W₂a₁ + b₂ = f(x)

**역전파**:

**Step 1**: 출력 오차
$$\delta_2 = \frac{\partial L}{\partial z_2} = f(x) - y$$

**Step 2**: ∂L/∂W₂ (외적!)
$$\boxed{\frac{\partial L}{\partial W_2} = \delta_2 \cdot a_1^T}$$

**Step 3**: 오차 역전파
$$\delta_1 = W_2^T\delta_2 \odot \sigma'(z_1) = W_2^T(f(x)-y) \odot \sigma(z_1)\odot(1-\sigma(z_1))$$

**Step 4**: ∂L/∂W₁ (외적!)
$$\boxed{\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T}$$

> **패턴**: 모든 가중치 미분 = δ(오차 신호) × input^T (입력의 전치). 외적 형태.

**Matrix 미분 닫힘 문제 4개 (기출 패턴)**:
1. ∂/∂x (a^Tx) = a
2. ∂/∂x (x^TAx) = (A+A^T)x, 대칭이면 2Ax
3. ∂/∂X tr(AX) = A^T
4. ∂/∂X tr(X^TAX) = (A+A^T)X, 대칭이면 2AX

---

# 문제 6. [10점] 학습률 조건

## 문제 6 [10점]

이차 손실함수 L(w) = ½w^T Aw - b^T w (A는 대칭 양의 정부호)에 대한 경사하강법:

$$w_{t+1} = w_t - \eta \nabla L(w_t) = w_t - \eta(Aw_t - b)$$

학습률이 η < 2/λ_max(A)일 때 loss가 감소함을 증명하시오.

---

### 풀이 6: 학습률 상한 증명

**필수 개념**: 고유값 분해를 사용한 좌표 변환. 각 고유값 방향에서 독립적으로 수렴 조건 분석.

**최적해**: ∇L = Aw* - b = 0 → w* = A⁻¹b

**오차 벡터 정의**: e_t = w_t - w*

$$e_{t+1} = w_{t+1} - w^* = (w_t - \eta(Aw_t - b)) - w^*$$
$$= (w_t - w^*) - \eta A(w_t - w^*) \quad (\because Aw^* = b)$$
$$= (I - \eta A)e_t$$

**고유값 분해**: A = UΛU^T, Λ = diag(λ₁,...,λₙ)

고유벡터 좌표로 변환: ẽ_t = U^T e_t

$$\tilde{e}_{t+1} = (I - \eta\Lambda)\tilde{e}_t$$

각 성분별로:
$$\tilde{e}_{t+1,i} = (1 - \eta\lambda_i)\tilde{e}_{t,i}$$

**수렴 조건**: |1 - ηλᵢ| < 1 for all i

$$-1 < 1 - \eta\lambda_i < 1$$
$$0 < \eta\lambda_i < 2$$
$$0 < \eta < \frac{2}{\lambda_i} \quad \text{for all } i$$

가장 제한적인 조건은 λᵢ가 최대일 때:

$$\boxed{\eta < \frac{2}{\lambda_{\max}(A)}}$$

**Loss 감소 증명**:

$$\|e_{t+1}\|^2 = \sum_i (1-\eta\lambda_i)^2 \tilde{e}_{t,i}^2$$

η < 2/λ_max이면 모든 i에서 (1-ηλᵢ)² < 1이므로:

$$\|e_{t+1}\|^2 < \|e_t\|^2$$

L(w) = ½∥e∥²_A + C (이차형식)이므로 오차 감소 → loss 감소. ∎

> **직관**: 학습률이 너무 크면 가장 가파른 방향(λ_max)에서 최적점을 넘어가 발산. η < 2/λ_max는 "가장 가파른 방향에서도 넘어가지 않는" 최대 보폭.

> **백엔드 관점**: 조건수 κ = λ_max/λ_min이 크면 학습이 느림. 이것은 DB 인덱스가 편향되었을 때 쿼리 최적화가 어려운 것과 유사. Preconditioning(Adam 등)은 조건수를 1에 가깝게 만들어 이 문제를 해결.

---

# 문제 7. [5점] Average Pooling

## 문제 7 [5점]

입력 벡터 x = [x₁, x₂, x₃, x₄]^T에 대해 커널 크기 2, 스트라이드 2인 1D Average Pooling 연산을 행렬 곱으로 표현하시오.

---

### 풀이 7: Pooling을 행렬로 표현

**필수 개념**: Pooling은 선형 연산이므로 행렬로 표현 가능. 이것이 역전파에서 야코비안으로 직접 사용됨.

출력: y = [(x₁+x₂)/2, (x₃+x₄)/2]^T

행렬 형태:
$$y = Px, \quad P = \frac{1}{2}\begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}$$

검증:
$$y = \frac{1}{2}\begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix} = \begin{bmatrix}\frac{x_1+x_2}{2}\\\frac{x_3+x_4}{2}\end{bmatrix} \checkmark$$

**2D 확장** (기출 가능): H_in=4, k=2, p=0, s=2이면 H_out = (4-2+0)/2 + 1 = 2

**역전파에서의 의미**: ∂L/∂x = P^T · ∂L/∂y. P^T는 그래디언트를 각 입력 원소에 1/k²씩 균등 분배.

> **출력 크기 공식**: H_out = floor((H_in - k + 2p) / s) + 1

---

# 문제 8. [5점] 기타 — 평균·분산 유도

## 문제 8 [5점]

확률변수 X의 평균과 분산을 정의로부터 유도하시오.
Var(X) = E[X²] - (E[X])²임을 증명하시오.

---

### 풀이 8: 분산 공식 유도

**정의**:
$$E[X] = \sum_x x \cdot p(x) \quad \text{(이산)}, \quad E[X] = \int x \cdot f(x)dx \quad \text{(연속)}$$

$$\text{Var}(X) = E[(X - E[X])^2]$$

**유도**:
$$\text{Var}(X) = E[(X - \mu)^2] \quad (\mu = E[X])$$
$$= E[X^2 - 2\mu X + \mu^2]$$
$$= E[X^2] - 2\mu E[X] + \mu^2 \quad \text{(기댓값의 선형성)}$$
$$= E[X^2] - 2\mu^2 + \mu^2$$
$$= E[X^2] - \mu^2$$

$$\boxed{\text{Var}(X) = E[X^2] - (E[X])^2}$$

> **왜 선형성을 쓸 수 있는가**: E[aX+b] = aE[X]+b는 X, Y의 독립성 없이 항상 성립. 분산 전개에서 E[X²-2μX+μ²]를 세 항으로 나눌 수 있는 이유.

---

# 30분 직전 최종 체크리스트 (중간고사)

| # | 유도/증명 | 핵심 수식 | ☐ |
|---|---------|---------|---|
| 1 | 균등분포 평균·분산 | E=(a+b)/2, V=(b-a)²/12 | ☐ |
| 2 | 정규분포 홀수모멘트=0 | 대칭성 → 기함수 적분=0 | ☐ |
| 3 | 베르누이 MLE | θ̂=k/n (5단계 유도) | ☐ |
| 4 | MAP 유도 | logpost = loglik + logprior | ☐ |
| 5 | Beta-Bernoulli conjugacy | Post = Beta(α+k, β+n-k) | ☐ |
| 6 | MSE = NLL (가우시안) | NLL ∝ MSE + C | ☐ |
| 7 | CE = KL + H | 정의 전개로 증명 | ☐ |
| 8 | MSE=MLE=NLL=KL 체인 | 5단계 연결 | ☐ |
| 9 | KL ≥ 0 (젠센) | -Σp log(q/p) ≥ -log(Σq) = 0 | ☐ |
| 10 | 정규분포 KL | σ같으면 = (μ₁-μ₂)²/2 | ☐ |
| 11 | softmax 야코비안 | diag(p) - pp^T | ☐ |
| 12 | -log σ(s) 미분 | = σ(s) - 1 | ☐ |
| 13 | 역전파 = 외적 | ∂L/∂W = δ · x^T | ☐ |
| 14 | η < 2/λ_max 증명 | |1-ηλᵢ| < 1 | ☐ |
| 15 | Avg Pooling 행렬 | P = ½[1 1 0 0; 0 0 1 1] | ☐ |
| 16 | Var = E[X²]-(E[X])² | 정의→전개→선형성 | ☐ |

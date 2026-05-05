---
title: "02. MLE, MAP, 베이지안 추론"
slug: 02-mle-map
order: 3
---

# 02. MLE, MAP, 베이지안 추론

---

## 한 줄 결론

> **"무조건 MAP입니다. ML도 사실 MAP의 일종"** — 모든 학습은 Likelihood와 Prior의 결합이며, Prior를 의식적으로 선택하는 것이 딥러닝 설계의 핵심이다.

---

## 교수님 핵심 강조

> **"수업 전체에서 가장 중요한 날"** (3주차)

> **"다 이해를 하셔야 됩니다. 이 흐름을 다 이해를 하셔야 됩니다"**

이 챕터가 시험의 60% 이상을 차지할 것으로 예상. 유도 과정을 반드시 단계별로 서술해야 한다.

---

# 입문 (Goal: Bayes 정리가 학습과 어떻게 연결되는지)

## 쉽게 설명하면

> **학습 = Prior(사전 믿음)를 Data(관찰)로 업데이트하여 Posterior(사후 믿음)를 얻는 것**

교수님의 색깔 비유:
- **Prior(파란색)** → 데이터 보기 전의 믿음
- **Data** → 관찰된 증거
- **Posterior(빨간색)** → 데이터를 본 후의 업데이트된 믿음

### 백엔드 비유

```
Prior       = 설정 파일의 기본값 (application.yml의 default)
Data        = 런타임에 들어오는 실제 트래픽 패턴
Posterior   = 오토스케일러가 학습한 최적 설정값
MLE         = 기본값 무시하고 트래픽 데이터만으로 결정 (위험!)
MAP         = 기본값 + 트래픽 데이터를 종합하여 결정 (안전)
```

## Bayes 정리

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

> 교수님: **"P(h|e) ∝ P(e|h)P(h)"**

| 항 | 이름 | 의미 |
|----|------|------|
| P(θ\|D) | **Posterior** | 데이터를 본 후 θ에 대한 믿음 |
| P(D\|θ) | **Likelihood** | θ가 주어졌을 때 데이터가 나올 확률 |
| P(θ) | **Prior** | 데이터 보기 전 θ에 대한 믿음 |
| P(D) | **Evidence** | 정규화 상수 (최적화 시 무시) |

$$\boxed{Posterior \propto Likelihood \times Prior}$$

---

# 중급 (Goal: MLE/MAP 완전 유도)

## 3단계 풀이 워크플로우

> 교수님이 반복 강조한 풀이 순서:

| 단계 | 내용 | 시험 답안에 쓸 것 |
|------|------|------------------|
| **① Likelihood 쓰기** | P(D\|θ) = ∏ P(x_i\|θ) | i.i.d. 가정 명시 |
| **② Prior 정하기** | P(θ) 선택 | 어떤 prior인지 명시 |
| **③ 합쳐서 미분=0** | log(Likelihood × Prior) 미분 | 단계별 전개 |

---

## MLE 완전 유도: Bernoulli Likelihood

### 설정

동전을 n번 던져 k번 앞면(H)이 나옴. θ = P(H).

### Step 1: Likelihood 쓰기

$$P(D|\theta) = \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^k(1-\theta)^{n-k}$$

여기서 $k = \sum x_i$ (앞면 횟수), i.i.d. 가정.

### Step 2: Prior — Uniform (= prior 없음)

$$P(\theta) = 1, \quad \theta \in [0, 1]$$

> 교수님: "Uniform prior일 때 MAP = MLE"

### Step 3: Log-Likelihood → 미분 = 0

$$\ell(\theta) = \log P(D|\theta) = k \log\theta + (n-k)\log(1-\theta)$$

> **왜 log를 취하는가?** product → sum 변환으로 미분이 용이해진다.

$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

$$k(1-\theta) = (n-k)\theta$$

$$k - k\theta = n\theta - k\theta$$

$$k = n\theta$$

$$\boxed{\theta_{MLE}^* = \frac{k}{n}}$$

### 교수님 경고

> **"n=3, k=3이면 θ=1 → 극단적! 데이터에만 의존"**

3번 던져서 3번 앞면이면 "이 동전은 무조건 앞면만 나온다"고 결론. 상식적으로 비합리적.

> 남에게 설명한다면: "3번 API 호출 실패했다고 서버가 영원히 죽었다고 판단하는 것과 같다. 재시도 로직(=Prior)이 필요하다."

---

## MAP 유도 1: Uniform Prior → MLE와 동일

$P(\theta) = 1$ (상수) → $\log P(\theta) = 0$

$$\theta_{MAP}^* = \arg\max [\log P(D|\theta) + \log P(\theta)] = \arg\max [\log P(D|\theta)]$$

$$\therefore \theta_{MAP}^* = \theta_{MLE}^* = \frac{k}{n}$$

> **핵심**: Uniform prior = "나는 아무 선호도 없다" → Prior가 최적화에 영향을 주지 않음

---

## MAP 유도 2: p(θ) ∝ θ(1-θ) → Laplace Smoothing

### Prior 설정

$$P(\theta) \propto \theta(1-\theta)$$

이것은 Beta(2,2) 분포에 해당. 0.5 근처를 약간 선호.

### Log-Posterior

$$\log P(\theta|D) \propto k\log\theta + (n-k)\log(1-\theta) + \log\theta + \log(1-\theta)$$

$$= (k+1)\log\theta + (n-k+1)\log(1-\theta)$$

### 미분 = 0

$$\frac{k+1}{\theta} - \frac{n-k+1}{1-\theta} = 0$$

$$(k+1)(1-\theta) = (n-k+1)\theta$$

$$k+1 = (n+2)\theta$$

$$\boxed{\theta_{MAP}^* = \frac{k+1}{n+2}}$$

### 이것이 바로 Laplace Smoothing!

> n=3, k=3일 때: θ* = 4/5 = 0.8 (MLE의 1.0보다 훨씬 합리적!)

> 백엔드 비유: 카운트 기반 추천 시스템에서 분모에 +2, 분자에 +1 하는 것. "데이터가 부족할 때 극단값 방지"

---

## MAP 유도 3: p(θ) ∝ θ^m(1-θ)^m → 일반화

### Prior 설정

$$P(\theta) \propto \theta^m(1-\theta)^m$$

이것은 Beta(m+1, m+1) 분포. m이 클수록 0.5 근처를 강하게 선호.

### Log-Posterior

$$\log P(\theta|D) \propto (k+m)\log\theta + (n-k+m)\log(1-\theta)$$

### 미분 = 0

$$\frac{k+m}{\theta} = \frac{n-k+m}{1-\theta}$$

$$(k+m)(1-\theta) = (n-k+m)\theta$$

$$k+m = (n+2m)\theta$$

$$\boxed{\theta_{MAP}^* = \frac{k+m}{n+2m}}$$

### m의 의미: Prior 강도

| m | θ* (n=3, k=3) | 해석 |
|---|---------------|------|
| 0 | 3/3 = 1.0 | MLE (prior 없음) |
| 1 | 4/5 = 0.8 | Laplace Smoothing |
| 10 | 13/23 ≈ 0.57 | 강한 prior |
| 100 | 103/203 ≈ 0.507 | 매우 강한 prior |
| ∞ | → 0.5 | Prior가 데이터를 완전 지배 |

> **m→∞ 극한: θ→0.5** (강한 prior = 데이터와 무관하게 0.5에 수렴)

> 남에게 설명한다면: "m은 가상의 추가 데이터 수. m=100이면 '이미 앞면 100번, 뒷면 100번 봤다'고 믿는 것. 실제 3번의 데이터가 이 믿음을 크게 바꿀 수 없다."

---

## MAP 유도 4: 비대칭 Prior p(θ) ∝ θ^m

### Prior 설정

$$P(\theta) \propto \theta^m$$

앞면 쪽을 선호하는 비대칭 prior.

### Log-Posterior

$$\log P(\theta|D) \propto (k+m)\log\theta + (n-k)\log(1-\theta)$$

### 미분 = 0

$$\frac{k+m}{\theta} = \frac{n-k}{1-\theta}$$

$$\boxed{\theta_{MAP}^* = \frac{k+m}{n+m}}$$

> 대칭 vs 비대칭: 대칭 prior θ^m(1-θ)^m은 0.5를 향하고, 비대칭 prior θ^m은 1을 향한다.

---

## Gaussian MLE

### 설정

$x_1, ..., x_n \sim N(\mu, \sigma^2)$ (i.i.d.)

### Likelihood

$$P(D|\mu,\sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

### Log-Likelihood

$$\ell(\mu,\sigma^2) = -\frac{n}{2}\log(2\pi) - n\log\sigma - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$$

### μ에 대해 미분 = 0

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\sum_{i=1}^{n} x_i = n\mu$$

$$\boxed{\mu_{ML} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}}$$

### σ²에 대해 미분 = 0

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^{n}(x_i-\mu)^2 = 0$$

$$\boxed{\sigma^2_{ML} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

### Bessel 보정 (편향 보정)

$\sigma^2_{ML}$은 **편향 추정량** (biased estimator):

$$E[\sigma^2_{ML}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$$

비편향 추정량:

$$\boxed{s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

> 왜 n-1? 표본평균 $\bar{x}$를 사용하면서 자유도가 1 줄었기 때문. n개 데이터에서 $\bar{x}$를 이미 구했으므로 독립적인 정보는 n-1개.

---

# 고급 (Goal: MAP-정규화 연결, NLL-MSE 연결, 아키텍처 진화)

## NLL 정의와 MSE 연결

### NLL (Negative Log-Likelihood) 정의

$$NLL = -\log P(D|\theta) = -\sum_{i=1}^{n} \log P(x_i|\theta)$$

> 교수님: **"nll 줄이는게 likelihood를 키우는거죠 이렇게 다 연결"**

### NLL → MSE 유도 (Gaussian noise 가정)

**가정**: $y_i = f_\theta(x_i) + \epsilon_i$, $\epsilon_i \sim N(0, \sigma^2)$

따라서: $P(y_i|x_i, \theta) = N(y_i; f_\theta(x_i), \sigma^2)$

$$NLL = -\sum_{i=1}^{n} \log \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

$$= \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2$$

θ에 대해 최소화할 때, 첫째 항은 상수:

$$\boxed{\arg\min_\theta NLL = \arg\min_\theta \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - f_\theta(x_i))^2 = \arg\min_\theta MSE}$$

> **핵심 연결**: Gaussian noise 가정 하에서 NLL 최소화 = MSE 최소화. 우리가 MSE를 쓰는 것은 "노이즈가 Gaussian"이라고 암묵적으로 가정하는 것!

---

## MAP → L2 정규화 (Ridge) 연결

### Prior 설정

$$P(w) = N(0, \tau^2 I) = \prod_j \frac{1}{\sqrt{2\pi}\tau} \exp\left(-\frac{w_j^2}{2\tau^2}\right)$$

### MAP 목적함수

$$\theta_{MAP}^* = \arg\max [\log P(D|\theta) + \log P(\theta)]$$

$$= \arg\min \left[\frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 + \frac{1}{2\tau^2}\sum_j w_j^2\right]$$

$\lambda = \sigma^2/\tau^2$으로 놓으면:

$$\boxed{\arg\min_\theta \left[ MSE + \lambda \|w\|_2^2 \right] = \text{Ridge Regression}}$$

> **핵심**: L2 정규화 = "가중치는 0 근처에 있을 것이다"라는 **Gaussian prior**

> 남에게 설명한다면: "Ridge는 '파라미터가 너무 크면 안 된다'는 사전지식을 수학적으로 표현한 것. λ가 크면 prior가 강한 것이고, 작으면 데이터에 더 의존."

---

## MAP → L1 정규화 (LASSO) 연결

### Prior 설정

$$P(w) \propto \prod_j \exp\left(-\frac{|w_j|}{b}\right) \quad \text{(Laplace 분포)}$$

### MAP 목적함수

$$\arg\min_\theta \left[\frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 + \frac{1}{b}\sum_j |w_j|\right]$$

$$\boxed{\arg\min_\theta \left[ MSE + \lambda \|w\|_1 \right] = \text{LASSO}}$$

> **핵심**: L1 정규화 = "대부분의 가중치는 정확히 0일 것이다"라는 **Laplace prior** → 스파스(sparse) 솔루션

| 정규화 | Prior | 효과 | 비유 |
|--------|-------|------|------|
| L2 (Ridge) | Gaussian | 가중치를 작게 유지 | 모든 컬럼을 약간씩 축소 |
| L1 (LASSO) | Laplace | 불필요한 가중치를 0으로 | Feature selection (DROP COLUMN) |

---

## 교수님의 ML vs Strong MAP 비교표

> **"무조건 MAP입니다"**

| | ML (Uniform Prior) | Weak MAP | Strong MAP |
|---|---|---|---|
| **가설공간** | 넓음 (제약 없음) | 중간 | 좁음 (강한 제약) |
| **표현력** | 높음 | 중간 | 낮음 |
| **Inductive Bias** | 약함 | 중간 | 강함 |
| **필요한 Knowledge** | 적음 | 중간 | 많음 |
| **필요한 Data** | 많음 | 중간 | 적음 |
| **과적합 위험** | 높음 | 중간 | 낮음 |

### 아키텍처 진화에 대입

| 아키텍처 | Prior 유형 | Prior 강도 | 필요 데이터 |
|----------|-----------|-----------|------------|
| **Linear Regression** | "세상은 선형" | 강함 | 적음 |
| **CNN** | "공간적 지역성, translation equivariance" | 중간 | 중간 |
| **Transformer** | "가정 최소화" | 약함 | 매우 많음 |

> 교수님: Transformer는 약한 prior로 거대한 데이터를 때려넣는 전략. "증기기관 비유: 먼저 만들고 나서 이론 이해"

> **"아키텍처 자체가 Prior다"** — CNN의 convolution 구조 = "인접 픽셀끼리 관련 있다"는 prior. Transformer의 attention = "어떤 토큰이든 관련될 수 있다"는 약한 prior.

---

## "무조건 MAP입니다"

> 교수님의 최종 선언

왜 MLE가 아니라 MAP인가?

1. **MLE는 MAP의 특수 케이스** (Uniform prior)
2. 모든 실용적 모델은 prior를 사용한다 (정규화 = prior)
3. Prior를 명시하지 않아도 **아키텍처 자체가 implicit prior**
4. 데이터가 유한한 이상, prior 없이는 과적합이 필연적

```
MLE:  arg max P(D|θ)                     ← "prior 없음"이 아니라 "uniform prior"
MAP:  arg max P(D|θ)·P(θ)               ← prior 명시
Full Bayesian: P(θ|D) 전체 분포를 구함    ← 점 추정이 아닌 분포 추정
```

---

## KL Divergence (보충)

$$KL(P\|Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = E_P\left[\log \frac{P(x)}{Q(x)}\right]$$

### 핵심 성질

1. **KL(P||Q) >= 0** (항상, Gibbs' inequality)
2. **KL(P||Q) = 0 iff P = Q**
3. **비대칭**: KL(P||Q) ≠ KL(Q||P) (거리 함수가 아님!)

> 백엔드 비유: KL divergence = 두 API 응답 분포의 차이. "모니터링에서 production 분포와 training 분포의 괴리를 측정하는 것"

---

## 풀이 시 체크리스트

시험 답안 작성 시 반드시 확인:

- [ ] **i.i.d. 명시**: "Assume $x_1, ..., x_n$ are i.i.d."
- [ ] **log 변환 이유**: "Taking log for computational convenience (product → sum)"
- [ ] **Likelihood 명시**: $P(D|\theta) = \prod P(x_i|\theta) = ...$
- [ ] **Prior 명시**: $P(\theta) = ...$ (없으면 Uniform임을 명시)
- [ ] **미분 전개**: 한 줄씩 단계적으로
- [ ] **미분=0 풀이**: 정리하여 θ* 도출
- [ ] **Boundary 체크**: θ가 [0,1] 범위인지, 경계에서 극값은 아닌지
- [ ] **해석**: 결과가 의미하는 바를 한 줄로 (예: "Laplace smoothing")
- [ ] **특수 케이스 검증**: m=0이면 MLE로 환원되는지, n→∞이면 MLE에 수렴하는지

---

## 핵심 킬러 요약

```
[Bayes 공식]
P(θ|D) ∝ P(D|θ) · P(θ)
Posterior ∝ Likelihood × Prior

[MLE]
θ* = arg max P(D|θ)
Bernoulli: θ* = k/n  ← 위험! 극단적!

[MAP with θ(1-θ)]
θ* = (k+1)/(n+2)  ← Laplace Smoothing

[MAP with θ^m(1-θ)^m]
θ* = (k+m)/(n+2m)  ← m→∞이면 0.5

[비대칭 prior θ^m]
θ* = (k+m)/(n+m)  ← m→∞이면 1.0

[NLL → MSE]
Gaussian noise 가정 → NLL 최소화 = MSE 최소화

[정규화 = Prior]
Gaussian prior → L2 (Ridge)
Laplace prior → L1 (LASSO)

[아키텍처 = Prior]
Linear(강) → CNN(중) → Transformer(약)

[3단계 풀이]
① Likelihood 쓰기 ② Prior 정하기 ③ 합쳐서 미분=0
```

---

## 핵심 정리

| 개념 | 공식/키워드 | 시험 출제 확률 |
|------|------------|---------------|
| Bayes 정리 | P(θ\|D) ∝ P(D\|θ)P(θ) | ★★★★★ |
| Bernoulli MLE | θ* = k/n | ★★★★★ |
| MAP (Laplace) | θ* = (k+1)/(n+2) | ★★★★★ |
| MAP (일반) | θ* = (k+m)/(n+2m) | ★★★★★ |
| Gaussian MLE | μ=x̄, σ²=Σ(x-x̄)²/n | ★★★★ |
| Bessel 보정 | n → n-1 | ★★★ |
| NLL → MSE | Gaussian noise 가정 | ★★★★★ |
| L2 = Gaussian prior | Ridge | ★★★★ |
| L1 = Laplace prior | LASSO | ★★★★ |
| KL Divergence | KL>=0, KL=0 iff P=Q | ★★★ |
| Uniform prior → MLE | MAP의 특수 케이스 | ★★★★★ |

> **최종 한 줄**: 딥러닝의 모든 학습은 MAP이다. Loss 함수는 NLL이고, 정규화는 Prior이며, 아키텍처 선택 자체가 가장 큰 Prior 결정이다.

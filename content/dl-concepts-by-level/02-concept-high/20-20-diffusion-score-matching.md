---
title: "20. 확산모델 & Score Matching"
slug: 20-diffusion-score-matching
order: 20
---

# 20. 확산모델 & Score Matching

## 왜 배우는가?

GAN은 고품질 이미지를 빠르게 만들지만 모드 붕괴(mode collapse) 문제가 있고, VAE는 다양한 이미지를 만들지만 품질이 아쉽다. **확산모델(Diffusion Model)**은 이 두 장점을 모두 취한다: 고품질 + 높은 다양성.

DALL-E, Stable Diffusion, Midjourney, Sora --- 현재 이미지와 영상 생성 AI의 핵심 기술이 바로 확산모델이다.

---

## 1. 확산모델의 핵심 아이디어

### 비유: 잉크 퍼지기의 역과정

깨끗한 물에 잉크를 한 방울 떨어뜨리면 점점 퍼져서 균일한 색이 된다. 확산모델은 이 과정을 **거꾸로** 해서, 균일한 노이즈에서 선명한 이미지를 만들어낸다.

### 두 가지 과정

```
[순방향: 노이즈 추가] (고정, 학습 불필요)
깨끗한 이미지 x₀ → z₁ → z₂ → ... → z_T (순수 노이즈)
                    매 단계 가우시안 노이즈를 조금씩 추가

[역방향: 노이즈 제거] (이것만 학습!)
순수 노이즈 z_T → ... → z₂ → z₁ → x₀ (깨끗한 이미지)
                    매 단계 노이즈를 조금씩 제거
```

핵심: 순방향은 고정이고, **역방향만 학습**한다.

---

## 2. VAE와 확산모델의 관계

### VAE 복습

VAE의 ELBO(Evidence Lower Bound):

$$-\log p(x) \leq \underbrace{\mathbb{E}_z[-\log p_\theta(x|z)]}_{\text{복원 오차}} + \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{잠재 분포 정규화}}$$

### 확산모델 = 계층적 VAE

| 특성 | VAE | 확산모델 |
|------|-----|---------|
| 인코더 | $q_\phi(z|x)$ -- **학습** | $q(z_t|z_{t-1})$ -- **고정** |
| 디코더 | $p_\theta(x|z)$ -- 학습 | $p_\theta(z_{t-1}|z_t)$ -- 학습 |
| 잠재 변수 | 1개 ($z$) | **T개** ($z_1, \ldots, z_T$) |
| 잠재 공간 차원 | 데이터보다 작음 (병목) | 데이터와 **같은 차원** |

확산모델의 ELBO:

$$-\text{ELBO} = \underbrace{\mathbb{E}_{z_1}[-\log p_\theta(x|z_1)]}_{\text{복원 오차}} + \underbrace{\text{KL}(\text{순방향} \| \text{역방향})}_{\text{각 단계의 일관성}}$$

복원 오차: $\|x - \hat{x}(z_1; \theta)\|^2$

일관성 항: $\|z_{t-1} - \hat{z}_{t-1}(z_t; \theta)\|^2$ (각 단계의 예측이 정확한가)

---

## 3. DDPM의 학습 목표

### Forward Process (순방향)

각 단계에서 가우시안 노이즈를 추가한다:

$$q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} \cdot z_{t-1}, \;\beta_t I)$$

$\beta_t$는 노이즈 스케줄로, $t$가 커질수록 노이즈가 많아진다.

좋은 점: 임의의 시점 $t$에서의 $z_t$를 한 번에 계산할 수 있다:

$$q(z_t | x_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t} \cdot x_0, \;(1 - \bar{\alpha}_t) I)$$

여기서 $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$이다.

이를 더 간단하게 쓰면:

$$z_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

### Simplified Loss: 노이즈를 맞추기

DDPM의 최종 학습 목표는 놀라울 만큼 간단하다:

$$L = \mathbb{E}_{t, x_0, \varepsilon}\left[\|\varepsilon - \varepsilon_\theta(z_t, t)\|^2\right]$$

의미: 깨끗한 이미지 $x_0$에 노이즈 $\varepsilon$을 섞어 $z_t$를 만들고, 신경망 $\varepsilon_\theta$가 **어떤 노이즈가 추가되었는지** 맞추도록 학습한다.

### 학습 과정 의사코드

```
반복:
  1. 학습 데이터에서 x₀를 뽑는다
  2. 시간 t를 1~T에서 무작위로 뽑는다
  3. 노이즈 ε ~ N(0, I)를 뽑는다
  4. z_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε 를 계산한다
  5. 손실 = ‖ε - ε_θ(z_t, t)‖² 를 최소화한다
```

---

## 4. Score Function: 확률의 나침반

### 에너지 기반 모델에서 출발

EBM의 확률 분포: $p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}$

정규화 상수 $Z_\theta = \int \exp(-E_\theta(x)) dx$는 고차원에서 계산 불가능하다.

### Score Function의 등장

$$s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

$\log p_\theta(x) = -E_\theta(x) - \log Z_\theta$인데, $Z_\theta$는 $x$에 무관한 상수이므로 $x$로 미분하면 사라진다!

**Score function은 $Z_\theta$ 없이 계산할 수 있다.**

### 기하학적 의미

Score function은 데이터 공간의 각 점에서 **확률이 증가하는 방향**을 가리키는 화살표(벡터장)이다.

```
에너지 지형:

높음 ↑   \       /
          \  ↗  /     ← 화살표가 score function
           \/
          ●●●         ← 데이터 (골짜기)
낮음 ↓
```

---

## 5. Score Matching과 Denoising Score Matching

### Score Matching (SM)

목표: 모델의 score가 실제 데이터 분포의 score와 같아지도록 학습

$$L_{SM} = \frac{1}{2}\mathbb{E}_{x \sim p_0}\left[\|s_\theta(x) - \nabla_x \log p_0(x)\|^2\right]$$

문제: $\nabla_x \log p_0(x)$를 모른다! 부분적분으로 변환할 수 있지만, Jacobian 트레이스 계산이 고차원에서 비싸다.

### Denoising Score Matching (DSM): 실용적 해결

데이터에 노이즈를 추가한 후, 노이즈가 추가된 데이터의 score를 학습한다.

$\tilde{x} = x + \sigma\varepsilon$ ($\varepsilon \sim \mathcal{N}(0, I)$)일 때:

$$\nabla_{\tilde{x}} \log p(\tilde{x} | x) = \frac{x - \tilde{x}}{\sigma^2} = -\frac{\varepsilon}{\sigma}$$

이 값은 "원본 데이터를 향하는 방향"이다! 노이즈가 있는 곳에서 깨끗한 곳으로 향하는 화살표.

DSM 손실:

$$L_{DSM} = \frac{1}{2}\mathbb{E}_{x, \varepsilon}\left[\left\|s_\theta(x + \sigma\varepsilon;\; \sigma) + \frac{\varepsilon}{\sigma}\right\|^2\right]$$

**놀라운 사실**: $L_{SM} = L_{DSM} + \text{상수}$. 즉, DSM은 원래 score matching과 동치이면서 계산이 훨씬 쉽다!

---

## 6. NCSN: 다중 노이즈 스케일

### 문제: 단일 노이즈로는 부족하다

노이즈가 작으면 데이터가 없는 영역(저밀도 영역)에서 score 추정이 부정확하다. 노이즈가 크면 세밀한 구조를 놓친다.

### 해결: 여러 노이즈 수준에서 동시에 학습

$$L_{NCSN} = \sum_{i=1}^{L} \lambda(\sigma_i) \cdot L_{DSM}(\theta;\; \sigma_i)$$

$\sigma_1 < \sigma_2 < \cdots < \sigma_L$로, 작은 것부터 큰 것까지 여러 스케일을 사용한다.

| 큰 $\sigma$ | 작은 $\sigma$ |
|-------------|--------------|
| 넓은 영역 커버 | 세밀한 구조 포착 |
| 글로벌 구조 학습 | 로컬 디테일 학습 |

하나의 네트워크 $s_\theta(\tilde{x}; \sigma)$가 $\sigma$를 조건으로 받아 모든 노이즈 수준을 처리한다.

### NCSN과 DDPM의 연결

$$s_\theta(\tilde{x}, \sigma) = -\frac{\varepsilon_\theta(\tilde{x}, t)}{\sigma}$$

DDPM의 "노이즈 예측"과 NCSN의 "score 예측"은 부호와 스케일만 다른 **같은 것**이다! 변분 관점과 score 관점은 같은 목표를 다른 언어로 표현한 것이다.

---

## 7. Langevin Dynamics: Score로 샘플링하기

### 기본 아이디어

Score function(확률이 증가하는 방향)을 따라가면 데이터 분포에서 샘플을 얻을 수 있다.

$$x_{k+1} = x_k + \eta \cdot s_\theta(x_k) + \sqrt{2\eta} \cdot \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I)$$

| 항 | 의미 |
|----|------|
| $x_k$ | 현재 위치 |
| $\eta \cdot s_\theta(x_k)$ | 확률 높은 방향으로 이동 |
| $\sqrt{2\eta} \cdot \varepsilon_k$ | 랜덤 노이즈 (다양성 확보) |

### Annealed Langevin Dynamics (어닐드 랑주뱅 동역학)

큰 노이즈 $\sigma_L$에서 시작하여 점진적으로 작은 노이즈 $\sigma_1$로 줄이며 샘플링한다.

```
x ~ N(0, I)  (순수 노이즈에서 시작)
    │
    ↓ σ_L에서 K번 Langevin step (큰 스케일 구조 형성)
    │
    ↓ σ_{L-1}에서 K번 step
    │
    ↓ ... (점점 세밀해짐)
    │
    ↓ σ_1에서 K번 step (디테일 완성)
    │
    ↓ 최종 샘플 (생성된 이미지)
```

이 과정은 DDPM의 역방향 샘플링과 본질적으로 동일하다.

---

## 8. Guided Diffusion: 조건부 생성

"아무 이미지나 생성"이 아니라 "고양이 이미지를 생성"하고 싶을 때 사용한다.

### Classifier Guidance

별도의 분류기 $p(c|x)$를 사용하여 생성을 안내한다:

$$\text{guided score} = \nabla_x \log p(x) + \lambda \cdot \nabla_x \log p(c|x)$$

- 첫째 항: 그럴듯한 이미지 방향 (비조건부)
- 둘째 항: 클래스 $c$일 확률이 높아지는 방향
- $\lambda$: 가이던스 강도 (클수록 조건에 충실)

### Classifier-Free Guidance

**별도 분류기 없이** 하나의 모델로 조건부/비조건부를 모두 처리한다:

$$\text{guided score} = (1 - \lambda) \cdot \nabla_x \log p(x) + \lambda \cdot \nabla_x \log p(x|c)$$

학습 시: 조건 $c$를 확률적으로 빈 값($\emptyset$)으로 바꿔서, 모델이 조건부와 비조건부 생성을 모두 학습하게 한다.

추론 시: $\lambda > 1$로 설정하면 조건에 더 충실한 고품질 이미지가 생성된다.

### 수학적으로 보면

$$(1-\lambda)\nabla_x \log p(x) + \lambda \nabla_x \log p(x|c) = \nabla_x \log p(x) + \lambda \nabla_x \log p(c|x) + \text{상수}$$

베이즈 정리에 의해, classifier-free guidance는 사실상 **분류기 gradient를 암묵적으로 계산**하는 것이다! 분류기를 따로 만들지 않아도 되는 것이지, 분류 정보를 안 쓰는 것이 아니다.

이것이 DALL-E 2, Stable Diffusion, Imagen 등 현대 text-to-image 모델의 핵심 기법이다.

---

## 9. 두 관점의 통합

확산모델을 이해하는 두 가지 시각은 결국 같은 곳에 도달한다:

| 변분 관점 (DDPM) | Score 기반 관점 (NCSN) |
|-----------------|---------------------|
| 순방향: 노이즈 추가 | 데이터에 노이즈 추가 |
| 역방향: $p_\theta(z_{t-1}|z_t)$ 학습 | score $s_\theta(\tilde{x}, \sigma)$ 학습 |
| 노이즈 예측: $\varepsilon_\theta$ | Score 예측: $s_\theta = -\varepsilon_\theta / \sigma$ |
| ELBO 최소화 | DSM 손실 최소화 |
| 역방향 sampling | Annealed Langevin dynamics |
| **결과: 같은 학습 목표!** | |

Song et al. (2021)은 연속 시간 확률적 미분방정식(SDE) 프레임워크로 이 두 관점을 완전히 통합했다:

- 순방향 = SDE: $dx = f(x,t)dt + g(t)dw$
- 역방향 = reverse SDE: $dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$

역방향에서 학습이 필요한 것은 오직 **score function** $\nabla_x \log p_t(x)$ 하나뿐이다.

---

## 오해하기 쉬운 포인트

### 1. "확산모델의 순방향도 학습해야 한다"

순방향 $q(z_t|z_{t-1})$은 미리 정해진 가우시안 노이즈 추가이며, 학습 파라미터가 **없다**. 학습하는 것은 오직 역방향뿐이다.

### 2. "Score function은 확률값 자체를 알려준다"

Score는 **벡터**(gradient)이지 스칼라(확률)가 아니다. "확률이 증가하는 방향"을 알려줄 뿐, 확률 값 자체는 모른다.

### 3. "DDPM과 NCSN은 완전히 다른 모델이다"

$s_\theta = -\varepsilon_\theta / \sigma$ 관계로 연결되며, 수학적으로 동치인 학습 목표를 가진다. "노이즈 예측"과 "score 예측"이라는 다른 언어를 쓸 뿐이다.

### 4. "Classifier-free guidance에는 분류 정보가 없다"

"별도의 분류기 네트워크가 필요 없다"가 정확한 표현이지, 분류 정보를 사용하지 않는다는 뜻이 아니다. 수학적으로 암묵적 분류기 gradient를 계산하고 있다.

### 5. "확산모델의 잠재 공간은 VAE처럼 저차원이다"

확산모델의 $z_t$는 원본 데이터와 **같은 차원**이다. "잠재"는 차원 축소가 아니라 노이즈 수준의 계층을 의미한다.

### 6. "큰 노이즈 스케일은 불필요하다"

작은 $\sigma$만 쓰면 데이터가 없는 영역의 score가 부정확하다. 큰 $\sigma$는 글로벌 구조를, 작은 $\sigma$는 로컬 디테일을 담당한다. 둘 다 필수적이다.

---

## 정리/요약

| 개념 | 핵심 한 줄 |
|------|-----------|
| 확산모델 | 데이터에 노이즈를 점진적으로 추가한 뒤, 역과정을 학습하여 노이즈에서 데이터 생성 |
| VAE와의 관계 | 확산모델 = 순방향 고정 + 같은 차원 잠재공간을 가진 계층적 VAE |
| DDPM 손실 | $L = \mathbb{E}[\|\varepsilon - \varepsilon_\theta(z_t, t)\|^2]$ --- 추가된 노이즈를 맞추기 |
| Score function | $s(x) = \nabla_x \log p(x) = -\nabla_x E(x)$. 정규화 상수 $Z$ 없이 계산 가능 |
| DSM | 노이즈를 추가한 뒤 "원본 방향"의 score를 학습. 원래 SM과 동치 |
| NCSN | 여러 노이즈 스케일에서 동시 학습. $s_\theta = -\varepsilon_\theta / \sigma$로 DDPM과 동치 |
| Langevin dynamics | Score를 따라 이동하며 샘플 생성. Annealed 버전은 DDPM 샘플링과 동일 |
| Classifier-free guidance | $(1-\lambda) \cdot \text{비조건부} + \lambda \cdot \text{조건부}$. 현대 이미지 생성의 핵심 |
| 두 관점 통합 | 변분(DDPM)과 score(NCSN)는 같은 목표의 다른 표현. SDE로 완전히 통합 |

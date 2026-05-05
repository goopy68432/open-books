---
title: "19. 생성모델 - EBM & GAN"
slug: 19-generative-models-ebm-gan
order: 19
---

# 19. 생성모델 - EBM & GAN

## 왜 배우는가?

지금까지 배운 모델(분류, 회귀 등)은 "이것이 무엇인가?"에 답하는 **판별모델**이다. 생성모델은 한 단계 더 나아가 "이것과 비슷한 새로운 것을 만들어라"에 답한다.

GAN(2014)에서 시작하여 StyleGAN, DALL-E, Stable Diffusion, Sora까지 --- 이미지, 영상, 텍스트를 만들어내는 AI의 핵심 기술이 모두 생성모델이다. 생성모델을 이해하면 현대 AI의 가장 활발한 연구 영역을 이해할 수 있다.

---

## 1. 생성모델의 큰 그림

### 목표

데이터의 확률 분포 $p_{\text{data}}(x)$를 근사하는 모델 $p_\theta(x)$를 학습하여, $p_\theta$에서 샘플링하면 진짜 같은 새로운 데이터가 나오게 하는 것.

### 주요 접근법 비교

| 모델 | 핵심 아이디어 | 장점 | 단점 |
|------|-------------|------|------|
| **GAN** | 생성자 vs 판별자 대결 | 고품질, 빠른 생성 | 학습 불안정, 모드 붕괴 |
| **VAE** | 압축 + 복원 + 확률적 구조 | 안정적 학습, 다양성 | 흐릿한 출력 |
| **Flow** | 가역적 변환의 연쇄 | 정확한 확률 계산 | 설계 제약 많음 |
| **EBM** | 에너지 높낮이로 확률 표현 | 유연한 모델링 | 샘플링 어려움 |
| **Diffusion** | 노이즈 추가/제거의 역과정 | 고품질 + 다양성 | 느린 생성 |
| **ARM** | 순서대로 하나씩 생성 | 정확한 확률 계산 | 느린 생성 |

### 생성모델 삼각형 (Trade-off)

```
        고품질 샘플
           /\
          /  \
    GAN  /    \ Diffusion
        /      \
       /________\
  빠른 샘플링  높은 다양성
      VAE
```

세 마리 토끼를 동시에 잡기 어렵다. GAN은 빠르고 고품질이지만 다양성이 낮고, VAE는 빠르고 다양하지만 품질이 낮고, Diffusion은 고품질이고 다양하지만 느리다.

---

## 2. VAE (Variational Autoencoder)

### 핵심 구조

```
입력 x ──→ [인코더] ──→ μ, σ ──→ z = μ + σ·ε ──→ [디코더] ──→ x' (복원)
                                    ↑
                              ε ~ N(0, I)
                         (Reparameterization Trick)
```

- **인코더**: 입력 $x$를 평균 $\mu$와 표준편차 $\sigma$로 변환
- **Reparameterization Trick**: $z = \mu + \sigma \odot \varepsilon$로 샘플링을 미분 가능하게 만듦
- **디코더**: $z$에서 $x'$를 복원

### ELBO (Evidence Lower Bound)

직접 $\log p(x)$를 최대화하기 어려우므로, 하한(lower bound)인 ELBO를 대신 최대화한다:

$$-\text{ELBO} = \underbrace{\mathbb{E}_z[-\log p_\theta(x|z)]}_{\text{복원 오차 (원본과 얼마나 다른가)}} + \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{정규화 (잠재 공간이 정리되었는가)}}$$

### KL Divergence의 닫힌 형태

인코더 출력이 $\mathcal{N}(\mu, \text{diag}(\sigma^2))$이고 사전분포가 $\mathcal{N}(0, I)$일 때:

$$\text{KL} = \frac{1}{2}\left[-2\sum_j \log \sigma_j + \|\sigma\|^2 + \|\mu\|^2 - D\right]$$

$\sigma = 1, \mu = 0$이면 KL = 0이 되어, 인코더가 표준정규분포에 가까울수록 패널티가 작아진다.

### Reparameterization Trick이 왜 필요한가?

$z$를 $q_\phi(z|x)$에서 직접 샘플링하면 $\phi$에 대해 미분할 수 없다 (확률적 노드). $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$으로 바꾸면, 확률은 $\varepsilon$에 들어가고 $\phi$에 대한 미분이 가능해진다.

---

## 3. GAN (Generative Adversarial Network)

### 위조범과 경찰의 게임

- **생성자(Generator) $G$**: 랜덤 노이즈 $z$에서 가짜 이미지 $G(z)$를 만든다
- **판별자(Discriminator) $D$**: 입력이 진짜(1)인지 가짜(0)인지 판별한다

$$z \sim \mathcal{N}(0, I), \quad x_{\text{fake}} = G_\theta(z)$$

### 목적함수: Minimax Game

판별자는 진짜를 진짜로, 가짜를 가짜로 잘 분류하도록:

$$\max_D \;\; \mathbb{E}_{x \sim \text{data}}[\log D(x)] + \mathbb{E}_{z \sim \mathcal{N}}[\log(1 - D(G(z)))]$$

생성자는 판별자를 속이도록:

$$\min_G \;\; \mathbb{E}_{z \sim \mathcal{N}}[\log(1 - D(G(z)))]$$

### GAN은 암시적(implicit) 생성모델

VAE는 $p_\theta(x)$를 명시적으로 정의하지만, GAN은 **확률 밀도를 직접 정의하지 않는다**. 대신 $z \to G(z)$라는 샘플링 과정으로 분포를 표현한다. 그래서 GAN에서는 likelihood를 계산할 수 없다.

### 발전사

2014년 GAN(흐릿한 얼굴) → 2015년 DCGAN → 2017년 PGGAN → 2018년 StyleGAN(진짜 같은 얼굴)

---

## 4. WGAN (Wasserstein GAN)

### 기존 GAN의 문제

KL divergence와 TV distance는 두 분포가 전혀 겹치지 않으면 무한대가 되거나 불연속적으로 점프한다. 이러면 gradient가 제대로 전달되지 않아 학습이 불안정하다.

### 예시로 이해하기

두 분포가 직선 위에 있고, 하나를 $\theta$만큼 이동시키는 경우:

| 거리 척도 | $\theta = 0$ | $\theta \neq 0$ | 연속성 |
|----------|-------------|-----------------|--------|
| KL divergence | 0 | $\infty$ | 불연속 |
| TV distance | 0 | 1 | 불연속 |
| **Wasserstein 거리** | 0 | $|\theta|$ | **연속** |

### Wasserstein 거리: "흙 옮기기" 비용

두 분포 사이의 "최적 수송 비용"으로 해석한다. 한 분포(흙 무더기)를 다른 분포 모양으로 옮기는 데 드는 최소 노동량이다.

$$W(P_S, P_\theta) = \sup_{\|f\|_L \leq 1} \left[\mathbb{E}_{x \sim P_S}[f(x)] - \mathbb{E}_{x \sim P_\theta}[f(x)]\right]$$

여기서 $\|f\|_L \leq 1$은 **1-Lipschitz 조건**: 함수의 변화율이 1 이하여야 한다는 제약이다.

### WGAN의 장점

- Wasserstein 거리가 연속이므로 항상 유의미한 gradient를 제공
- 학습이 더 안정적
- Lipschitz 제약은 weight clipping이나 gradient penalty로 구현

---

## 5. EBM (Energy-Based Model)

### 핵심 아이디어: 에너지 지도

- 진짜 데이터가 있는 곳: 에너지 낮음 (골짜기)
- 가짜/이상한 데이터: 에너지 높음 (산꼭대기)

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \quad Z_\theta = \int \exp(-E_\theta(x)) dx$$

에너지 함수 $E_\theta(x)$는 신경망으로 모델링한다. $Z_\theta$는 정규화 상수(partition function)로, 고차원에서 직접 계산이 불가능하다.

### 학습: $Z_\theta$를 계산하지 않고도 학습할 수 있다!

MLE의 기울기를 구하면:

$$-\nabla_\theta \ell = -\underbrace{\mathbb{E}_{x \sim p_S}[\nabla_\theta E_\theta(x)]}_{\text{실제 데이터의 에너지를 낮춤}} + \underbrace{\mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)]}_{\text{모델 샘플의 에너지를 높임}}$$

핵심: $\nabla_\theta \log Z_\theta = -\mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)]$로, $Z_\theta$의 값이 아닌 $p_\theta$에서의 **샘플**만 필요하다.

### 직관적 이해

```
에너지 ↑  ___         ___
         /   \       /   \
        /     \_____/     \     ← 초기 에너지 함수
       /                   \
에너지 ↓

         ●●● (실제 데이터)  ○○○ (모델 샘플)

학습 후:
에너지 ↑  ___   ↑높임   ___
         / ○ \_____  / ○ \
        /           \/     \
       / ●●● ↓낮춤  ●●●    \
에너지 ↓
```

실제 데이터 위치의 에너지는 낮추고(push down), 모델이 생성한 샘플의 에너지는 높이는(push up) 동역학이다.

### 샘플링: Langevin MCMC

$p_\theta(x)$에서 샘플을 뽑기 위해 Langevin dynamics를 사용한다:

$$x_{t+1} = x_t + \lambda \nabla_x \log p_\theta(x_t) + \sqrt{2\lambda} \cdot \varepsilon_t$$

여기서 $\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$이다. 이 **스코어 함수(score function)**는 확산모델의 핵심 개념으로 이어진다.

---

## 6. Normalizing Flow

### 핵심 아이디어: 가역적 변환의 연쇄

단순한 분포(정규분포)를 **가역 변환** 여러 개로 변형하여 복잡한 분포를 만든다.

$$u \sim \mathcal{N}(0, I) \xrightarrow{f_1} \xrightarrow{f_2} \cdots \xrightarrow{f_K} x$$

### Change-of-Variables Formula

$f$가 가역이고 $g = f^{-1}$이면:

$$\log p_X(x) = \log p_U(g(x)) - \log |\det f'(u)|$$

합성 변환의 경우:

$$\log p_X(x) = \log p_U(g(x)) - \sum_{k=1}^{K} \log |\det f_k'(u_{k-1})|$$

### 장점과 한계

- **장점**: 정확한 likelihood 계산이 가능 (VAE의 ELBO가 아닌 정확한 값)
- **한계**: Jacobian 행렬식 계산이 비싸고, 입출력 차원이 같아야 하는 제약

---

## 7. Auto-Regressive Models (ARM)

### 확률의 연쇄법칙을 직접 사용

$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})$$

각 조건부 분포를 신경망으로 모델링한다. 순서대로 하나씩 생성한다.

- **텍스트**: GPT (한 단어씩)
- **이미지**: PixelCNN (한 픽셀씩)
- **오디오**: WaveNet (한 샘플씩)

**장점**: 정확한 likelihood 계산 가능
**단점**: 순차적 생성이므로 느리다

---

## 8. Diffusion Model (맛보기)

확산모델은 다음 장(20장)에서 자세히 다루지만, 생성모델의 큰 그림에서 위치를 잡아두자.

- **순방향**: 원본 이미지에 노이즈를 점진적으로 추가 → 순수 노이즈
- **역방향**: 노이즈에서 시작하여 점진적으로 제거 → 이미지 생성
- **관점**: Hierarchical VAE(계층적 VAE)의 특수한 경우로 볼 수 있다
- **연결**: EBM의 score function과 깊이 연결된다

---

## 9. 생성모델 목적함수 통합 비교

모든 생성모델은 결국 데이터 분포와 모델 분포 사이의 "거리"를 줄이는 것이 목표이다:

| 모델 | 목적함수 핵심 |
|------|-------------|
| **VAE** | 복원 오차(reconstruction) + KL 정규화 |
| **GAN** | $\min_G \max_D$ minimax game |
| **Flow** | 정확한 log-likelihood 최대화 (change-of-variables) |
| **EBM** | 실제 데이터 에너지 ↓, 모델 샘플 에너지 ↑ |

그리고 이들은 서로 연결되어 있다:
- GAN의 판별자를 에너지 함수로 보면 → EBM
- Flow의 가역 변환을 인코더/디코더로 보면 → VAE
- EBM의 score function을 학습하면 → Diffusion
- VAE의 계층을 무한히 쌓으면 → Diffusion

---

## 오해하기 쉬운 포인트

### 1. "GAN의 생성자는 확률분포를 직접 정의한다"

GAN은 **암시적(implicit)** 모델로, $z \to G(z)$ 과정만 정의한다. $p_\theta(x)$를 계산할 방법이 없다.

### 2. "VAE의 KL 항은 불필요한 제약이다"

KL 항을 제거하면 복원은 잘 되지만, 잠재공간이 불규칙해져서 새로운 샘플 생성이나 보간(interpolation)이 불가능해진다. KL 항이 잠재공간을 구조화하는 핵심이다.

### 3. "Wasserstein 거리와 KL divergence는 같은 역할이다"

KL은 두 분포가 안 겹치면 $\infty$가 되어 gradient가 없다. Wasserstein은 항상 연속적이고 유의미한 gradient를 제공한다. 이것이 WGAN이 안정적인 이유이다.

### 4. "EBM에서 정규화 상수 $Z_\theta$를 계산해야 한다"

기울기 계산에는 $Z_\theta$의 값이 아닌 $p_\theta$에서의 **샘플만** 필요하다. Langevin MCMC로 근사 샘플링한다.

### 5. "Reparameterization Trick은 코딩 편의용이다"

수학적 필수 조건이다. 이것 없이는 VAE의 인코더 파라미터에 대한 gradient를 효율적으로 계산할 수 없다.

### 6. "Diffusion은 GAN과 완전히 다른 새로운 것이다"

Diffusion model은 hierarchical VAE의 한 형태이며, EBM의 score matching과도 깊이 연결된다. 기존 생성모델의 아이디어가 결합된 것이다.

---

## 정리/요약

| 개념 | 핵심 한 줄 |
|------|-----------|
| 생성모델 분류 | GAN(적대적), VAE(변분), Flow(가역), EBM(에너지), Diffusion(노이즈역전), ARM(자기회귀) |
| VAE | ELBO = Reconstruction + KL. Reparameterization trick으로 미분 가능하게. 안정적이지만 흐릿함 |
| GAN | 생성자 vs 판별자 minimax game. 고품질이지만 mode collapse 위험 |
| WGAN | KL/TV의 불연속성을 Wasserstein 거리로 해결. 1-Lipschitz 제약 |
| EBM | $p(x) \propto \exp(-E(x))$. 기울기에 $Z$가 불필요. Score function = $-\nabla_x E(x)$ |
| Flow | 가역변환 + change-of-variables. 정확한 likelihood 가능 |
| ARM | $p(x) = \prod p(x_t \mid x_{<t})$. 정확하지만 순차 생성이 느림 |
| Trade-off | GAN = 빠름+고품질, VAE = 빠름+다양, Diffusion = 고품질+다양(느림) |

---
title: "19. 생성모델 -- EBM과 GAN"
slug: 19-generative-models-ebm-gan
order: 19
---

# 19. 생성모델 -- EBM과 GAN

## 1. 동기부여 및 개요

생성모델은 데이터의 확률분포 $p_{\text{data}}(x)$를 학습하여 **새로운 데이터를 생성**하는 딥러닝의 핵심 패러다임이다. 판별모델이 $p(y|x)$를 학습한다면, 생성모델은 $p(x)$ 자체를 학습한다.

2014년 GAN의 등장 이후, StyleGAN, DALL-E, Stable Diffusion, Sora에 이르기까지 이미지/영상/텍스트 생성이 폭발적으로 발전하였다.

```
확률분포 p(x)
    |
    +-----> MLE (최대우도추정) -----> VAE (ELBO)
    |                                   |
    +-----> KL Divergence               +-----> Diffusion Model
    |           |
    +-----> f-divergence -----> GAN (minimax)
    |           |
    +-----> Wasserstein 거리 --> WGAN
    |
    +-----> Energy Function --> EBM --> Score Function --> Diffusion/Score Matching
    |
    +-----> Chain Rule -------> ARM (PixelCNN, GPT)
    |
    +-----> Change of Vars ---> Normalizing Flow
```

---

## 2. 생성모델의 분류와 학습 목표

### 2.1 일반적 학습 목표: MLE

$$\max_\theta \; \mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)]$$

각 모델은 $\log p_\theta(x)$를 다루는 방식이 다르다:

| 모델 | $\log p_\theta(x)$ 처리 방식 | 밀도 계산 | 샘플링 |
|------|----------------------------|----------|--------|
| **VAE** | ELBO (하한) 최대화 | 근사 | 빠름 |
| **GAN** | 직접 계산 안 함 (implicit) | 불가 | 빠름 |
| **Flow** | Change of variables로 정확 계산 | 정확 | 빠름 |
| **EBM** | 에너지 함수 + 분배함수 | 정규화 상수 필요 | MCMC |
| **ARM** | 연쇄법칙으로 분해 | 정확 | 느림 (순차적) |
| **Diffusion** | 계층적 ELBO | 근사 | 느림 (반복적) |

### 2.2 생성모델 Trade-off 삼각형

```
              High Quality Samples
                    /    \
                   /      \
                GAN        Diffusion
               /              \
              /                \
   Fast Sampling ---- VAE ---- Mode Coverage (Diversity)
```

세 가지 좋은 성질을 동시에 달성하기 어렵다: GAN은 빠르고 고품질이나 mode collapse, VAE는 빠르고 다양하나 blurry, Diffusion은 고품질+다양하나 느리다.

---

## 3. VAE (Variational Autoencoder)

### 3.1 ELBO 유도

**Theorem 3.1 (Evidence Lower Bound).**

$$-\log p(x) \leq \underbrace{\mathbb{E}_{z \sim q_\phi(\cdot|x)}[-\log p_\theta(x|z)]}_{\text{reconstruction}} + \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{consistency}}$$

*증명 스케치.* Jensen 부등식을 적용한다:

$$-\log p(x) = -\log \int p(x,z) dz = -\log \mathbb{E}_{q(z|x)}\!\left[\frac{p(x,z)}{q(z|x)}\right] \leq \mathbb{E}_{q(z|x)}\!\left[-\log \frac{p(x,z)}{q(z|x)}\right]$$

등호 조건: $q_\phi(z|x) = p(z|x)$ (변분 사후분포 = 진짜 사후분포). $\square$

### 3.2 Reparameterization Trick

**인코더:** $q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma_\phi(x))^2)$

Naive Monte Carlo estimator $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi(z)]$는 분산이 극도로 높다.

**해결:** $z = \mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$로 재매개변수화하면:

$$\mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{p(\varepsilon)}[f(\mu_\phi(x) + \sigma_\phi(x) \odot \varepsilon)]$$

$\phi$에 대한 미분이 $f$ 내부로 들어가므로 저분산 기울기 추정이 가능하다.

### 3.3 KL Divergence의 닫힌 형태

$$\text{KL}(\mathcal{N}(\mu, \text{diag}(\sigma)^2) \| \mathcal{N}(0, I)) = \frac{1}{2}\left[-2\sum_{j=1}^{D} \log \sigma_j + \|\sigma\|^2 + \|\mu\|^2 - D\right]$$

*증명 스케치.* 일반 가우시안 KL 공식에 $\mu_2 = 0$, $\Sigma_2 = I$, $\Sigma_1 = \text{diag}(\sigma)^2$를 대입하면 된다. $\square$

| 항 | 의미 |
|----|------|
| $-2\sum_j \log \sigma_j$ | 엔트로피 항 ($\sigma$가 1에서 벗어나는 페널티) |
| $\|\sigma\|^2 + \|\mu\|^2$ | 평균/분산이 $\mathcal{N}(0,I)$에서 벗어나는 비용 |
| $-D$ | 상수 보정 |

### 3.4 복원 분포 가정에 따른 손실

- $p(x|z) = \mathcal{N}(x; \text{NN}_\theta(z), I)$ $\Rightarrow$ NLL $\propto \|x - \text{NN}_\theta(z)\|^2$ (MSE)
- $p(x|z) = \text{Ber}(x; \text{NN}_\theta(z))$ $\Rightarrow$ NLL = BCE (binary cross-entropy)

---

## 4. GAN (Generative Adversarial Network)

### 4.1 정의

**Definition 4.1 (GAN, Goodfellow et al., 2014).** GAN은 생성자 $G_\theta$와 판별자 $D_\phi$의 minimax 게임이다:

$$\min_\theta \max_\phi \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D_\phi(x)] + \mathbb{E}_{z \sim q(z)}[\log(1 - D_\phi(G_\theta(z)))]$$

- **생성자:** $z \sim \mathcal{N}(0, I) \mapsto x = G_\theta(z)$ -- 노이즈를 데이터로 변환
- **판별자:** 입력이 진짜(1)인지 가짜(0)인지 판별

### 4.2 암시적 생성모델

GAN은 $p_\theta(x)$를 명시적으로 정의하지 않는 **implicit generative model**이다.
샘플링 과정 $z \to G_\theta(z)$만 정의하며, likelihood 평가가 불가능하다.

### 4.3 최적 판별자와 이론적 의미

**Theorem 4.1.** 고정된 $G$에 대해 최적 판별자는 $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$이다.

이 최적 판별자 하에서 생성자의 목적함수는 $p_{\text{data}}$와 $p_G$ 사이의 **Jensen-Shannon divergence**를 최소화하는 것과 동치이다.

### 4.4 실제 학습의 어려움

- **Mode collapse:** 생성자가 소수의 모드만 생성
- **Training instability:** 생성자와 판별자의 균형 유지가 어려움
- **실용적 트릭:** $\min_\theta \log(1 - D(G(z)))$ 대신 $\max_\theta \log D(G(z))$ 사용 (초기 기울기 소실 회피)

---

## 5. WGAN (Wasserstein GAN)

### 5.1 f-divergence의 한계

**Definition 5.1 (f-divergence).**

$$D_f(p \| q) = \int q(x) \, f\!\left(\frac{p(x)}{q(x)}\right) d\mu(x)$$

$f$에 따라 KL divergence, TV distance 등을 포함하는 일반적 분류이다.

**문제:** 두 분포의 support가 저차원 매니폴드 위에 있고 교차하지 않으면:
- $\text{KL}(P_0 \| P_\theta) = \infty$ ($\theta \neq 0$일 때)
- $D_{TV}(P_\theta, P_0) = 1$ ($\theta \neq 0$일 때)

두 경우 모두 기울기가 0이어서 학습이 불가능하다.

### 5.2 Wasserstein 거리

**Definition 5.2 (Wasserstein-1 Distance / Earth Mover's Distance).**

$$W(P_S, P_\theta) = \sup_{\|f\|_L \leq 1} \; \mathbb{E}_{x \sim P_S}[f(x)] - \mathbb{E}_{x \sim P_\theta}[f(x)]$$

여기서 $\|f\|_L \leq 1$은 1-Lipschitz 조건이다.

**핵심 장점:** 분포의 support가 떨어져 있어도 $W(P_\theta, P_0) = |\theta|$로 **연속적이고 미분 가능**하다.

| 거리 | $P_\theta = P_0$ ($\theta=0$) | $P_\theta \neq P_0$ ($\theta \neq 0$) |
|------|------------------------------|---------------------------------------|
| KL | 0 | $\infty$ |
| TV | 0 | 1 (불연속 점프) |
| Wasserstein | 0 | $|\theta|$ (연속, 미분 가능) |

### 5.3 WGAN 목적함수

$$\max_\phi \; \mathbb{E}_{x \sim P_S}[f_\phi(x)] - \mathbb{E}_{z \sim q(z)}[f_\phi(G_\theta(z))] \quad \text{s.t. } \|f_\phi\|_L \leq 1$$

$$\min_\theta \; \mathbb{E}_{z \sim q(z)}[-f_\phi(G_\theta(z))]$$

Lipschitz 제약 구현: weight clipping (Arjovsky et al., 2017) 또는 gradient penalty (Gulrajani et al., 2017).

---

## 6. Energy-Based Model (EBM)

### 6.1 정의

**Definition 6.1 (Energy-Based Model).**

$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}, \quad Z_\theta = \int \exp(-E_\theta(x)) dx$$

에너지 함수 $E_\theta(x) \geq 0$은 신경망으로 모델링하며, $Z_\theta$는 정규화 상수(partition function)이다.

**직관:** 데이터가 있는 곳은 에너지가 낮고(확률이 높고), 데이터가 없는 곳은 에너지가 높다(확률이 낮다).

### 6.2 MLE 기울기 유도

**Theorem 6.1 (EBM의 MLE 기울기).**

$$-\nabla_\theta \ell(\theta) = -\underbrace{\mathbb{E}_{x \sim p_{\text{data}}}[\nabla_\theta E_\theta(x)]}_{\text{실제 데이터: 에너지 감소}} + \underbrace{\mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)]}_{\text{모델 샘플: 에너지 증가}}$$

*증명 스케치.*

$$\nabla_\theta \log Z_\theta = \frac{1}{Z_\theta} \nabla_\theta \int \exp(-E_\theta(x)) dx = -\mathbb{E}_{x \sim p_\theta}[\nabla_\theta E_\theta(x)]$$

$\nabla_\theta \ell = \nabla_\theta \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)] = -\mathbb{E}_{p_{\text{data}}}[\nabla_\theta E_\theta(x)] + \nabla_\theta \log Z_\theta$에 대입하면 결과를 얻는다. $\square$

**핵심:** $Z_\theta$의 값 자체가 아닌, $p_\theta$에서의 **샘플**만 있으면 기울기를 계산할 수 있다.

### 6.3 Langevin MCMC 샘플링

$p_\theta(x)$에서 샘플링하기 위해 Langevin 동역학을 사용한다:

$$x_{t+1} = x_t + \lambda \nabla_x \log p_\theta(x_t) + \sqrt{2\lambda} \, \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, I)$$

여기서 **score function** $s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$이다.
$Z_\theta$는 $x$에 무관하므로 score에서 사라진다. 이 score function이 [20장]의 Score-Based Generative Models로 이어지는 핵심 연결고리이다.

---

## 7. Normalizing Flow

### 7.1 정의

가역 변환 $f: \mathbb{R}^d \to \mathbb{R}^d$ ($g = f^{-1}$)을 통해 단순 분포를 복잡한 분포로 변환한다.

### 7.2 Change of Variables Formula

**Theorem 7.1.**

$$\log p_X(x) = \log p_U(g(x)) - \log |\det f'(u)|$$

합성 변환 $f = f_K \circ \cdots \circ f_1$에 대해:

$$\log p_X(x) = \log p_U(g(x)) - \sum_{k=1}^{K} \log |\det f'_k(u_{k-1})|$$

**장점:** 정확한 likelihood 계산 가능
**제약:** 차원 보존 ($d_{\text{input}} = d_{\text{output}}$), Jacobian determinant의 효율적 계산 필요

실용적 접근: coupling layer (RealNVP), autoregressive flow (MAF/IAF) 등에서 삼각 Jacobian을 사용하여 $O(d)$로 determinant 계산.

---

## 8. Auto-Regressive Models (ARM)

### 8.1 정의

확률의 연쇄법칙을 직접 사용한다:

$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})$$

각 조건부 분포를 신경망으로 모델링하고, masked convolution(causal convolution)으로 미래 정보를 차단한다.

### 8.2 대표 구현

| 구현 | 도메인 | 특징 |
|------|--------|------|
| PixelCNN | 이미지 | Raster scan order 마스킹 |
| WaveNet | 오디오 | Dilated causal convolution |
| GPT | 텍스트 | Transformer 기반 |

**장점:** 정확한 $\log p(x)$ 계산 (MLE 학습 직접적)
**단점:** 순차적 생성으로 샘플링이 느림

---

## 9. 생성모델 목적함수 통합 비교

| 모델 | 목적함수 |
|------|---------|
| **VAE** | $\min_{\theta,\phi} \; \mathbb{E}_x[\text{KL}(q_\phi(z|x) \| p(z))] + \mathbb{E}_{x,z}[-\log p_\theta(x|z)]$ |
| **GAN** | $\min_\theta \max_\phi \; \mathbb{E}_x[\log D_\phi(x)] + \mathbb{E}_z[\log(1-D_\phi(G_\theta(z)))]$ |
| **Flow** | $\max_\theta \; \mathbb{E}_x[\log p_U(g_\theta(x)) - \log |\det f'(u)|]$ |
| **EBM** | $\min_\theta \; \mathbb{E}_{p_{\text{data}}}[E_\theta(x)] - \mathbb{E}_{p_\theta}[E_\theta(x)]$ |

이 네 프레임워크는 상호 연결된다:
- GAN의 판별자 $\to$ EBM의 에너지 함수
- Flow의 가역 변환 $\to$ VAE의 인코더/디코더
- EBM의 score function $\to$ Diffusion model
- VAE의 계층을 무한히 $\to$ Diffusion model

---

## 10. 흔한 오해와 주의점

| # | 오해 | 올바른 이해 |
|---|------|-------------|
| 1 | GAN의 생성자가 $p_\theta(x)$를 직접 학습 | GAN은 **implicit** 모델. 샘플링만 정의하며 likelihood 계산 불가 |
| 2 | VAE의 KL 항은 불필요한 제약 | KL 없으면 잠재공간이 불규칙해져 생성/보간 불가. 핵심 정규화 |
| 3 | Wasserstein 거리와 KL은 교체 가능 | KL은 support 불일치 시 $\infty$, Wasserstein은 연속적. 학습 안정성에 본질적 차이 |
| 4 | EBM에서 $Z_\theta$를 직접 계산해야 한다 | 기울기에는 $Z_\theta$ 불필요. $p_\theta$에서의 **샘플**만 필요 |
| 5 | Reparameterization trick은 구현 편의 | VAE 학습의 **수학적 필수 조건**. Naive estimator는 분산이 너무 높음 |
| 6 | Diffusion model은 GAN과 완전히 다르다 | Diffusion은 hierarchical VAE의 특수 형태. EBM의 score matching과도 연결 |

---

## 11. 핵심 요약

| 개념 | 핵심 |
|------|------|
| **VAE** | ELBO = Reconstruction + KL. Reparameterization trick으로 역전파 가능 |
| **GAN** | $\min_G \max_D$의 minimax game. Implicit 모델. High quality but mode collapse |
| **WGAN** | f-divergence의 불연속 문제를 Wasserstein 거리로 해결. Lipschitz 제약 필요 |
| **EBM** | $p(x) \propto \exp(-E(x))$. Score function $s(x) = -\nabla_x E(x)$이 핵심 |
| **Flow** | 가역 변환 + change of variables. 정확한 likelihood, Jacobian이 병목 |
| **ARM** | $p(x) = \prod p(x_t|x_{<t})$. 정확한 likelihood, 순차 생성이 느림 |
| **Trade-off** | Quality vs Diversity vs Speed. 세 마리 토끼를 동시에 잡기 어렵다 |

**관련 개념 연결:** EBM의 Score function $\to$ [20장] Score matching과 Diffusion model, VAE의 ELBO $\to$ [20장] Diffusion의 계층적 ELBO, GAN의 implicit 모델 $\to$ [18장] 대조학습의 representation

**참고 문헌:**
- Goodfellow et al., "Generative Adversarial Nets," NeurIPS 2014
- Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR 2014
- Arjovsky et al., "Wasserstein Generative Adversarial Networks," ICML 2017
- LeCun et al., "A Tutorial on Energy-Based Learning," 2006
- Rezende & Mohamed, "Variational Inference with Normalizing Flows," ICML 2015

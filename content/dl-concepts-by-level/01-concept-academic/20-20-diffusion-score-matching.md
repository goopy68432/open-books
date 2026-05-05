---
title: "20. 확산모델과 Score Matching"
slug: 20-diffusion-score-matching
order: 20
---

# 20. 확산모델과 Score Matching

## 1. 동기부여 및 개요

GAN은 고품질 샘플을 빠르게 생성하지만 모드 붕괴(mode collapse) 문제가 있고,
VAE는 다양성은 좋지만 샘플 품질이 낮다. **확산모델(Diffusion Model)**은 이 두 장점을
모두 취한다: 고품질 샘플 + 높은 다양성(mode coverage).

데이터에 점진적으로 노이즈를 추가한 뒤, 그 역과정을 학습하여 노이즈로부터 데이터를 생성한다.
DALL-E 2, Stable Diffusion, Sora 등 현재 이미지/오디오/비디오 생성의 SOTA를 달성하는 핵심 기술이다.

확산모델을 이해하는 두 가지 동등한 관점이 존재한다:
1. **변분 관점 (Variational View):** VAE를 계층적으로 확장 $\to$ DDPM
2. **스코어 기반 관점 (Score-Based View):** 스코어 함수 학습 $\to$ NCSN

```
EBM: p(x) = exp(-E(x))/Z
       |
  Score Function: s(x) = nabla_x log p(x) = -nabla_x E(x)   [Z 불필요!]
       |
  Score Matching ──> Denoising Score Matching (DSM)
       |                        |
  Langevin Dynamics     NCSN (다중 노이즈 스케일)
       |                        |
       +----------- 통합 ---------+
       |                        |
  Annealed Langevin      <===>      DDPM (역방향 샘플링)
       |
  Guided Diffusion (Classifier / Classifier-Free Guidance)
```

---

## 2. VAE 복습: ELBO와 확산모델의 출발점

### 2.1 ELBO 유도

[19장]에서 유도한 ELBO를 간략히 복습한다:

$$-\log p(x) \leq \underbrace{\mathbb{E}_{z \sim q_\phi(\cdot|x)}[-\log p_\theta(x|z)]}_{\text{reconstruction}} + \underbrace{\text{KL}(q_\phi(z|x) \| p(z))}_{\text{consistency}}$$

등호 조건: $q_\phi(z|x) = p(z|x)$일 때. 실제로는 gap이 존재한다:

$$\log p(x) = \text{ELBO}(x; \theta, \phi) + \text{KL}(q_\phi(z|x) \| p(z|x))$$

### 2.2 VAE에서 확산모델로

확산모델은 VAE의 구조를 **계층적(hierarchical)**으로 확장한다:
- VAE: 잠재 변수 1개 ($z$), 인코더를 **학습**
- 확산모델: 잠재 변수 $T$개 ($z_1, \ldots, z_T$), 순방향 과정을 **고정**

| | VAE | Diffusion Model |
|---|---|---|
| 순방향 (인코더) | $q_\phi(z|x)$ -- **학습** | $q(z_t|z_{t-1})$ -- **고정** |
| 역방향 (디코더) | $p_\theta(x|z)$ -- 학습 | $p_\theta(z_{t-1}|z_t)$ -- 학습 |
| 잠재 변수 | 1개, 저차원 | $T$개, 원본과 **같은 차원** |
| 병목 구조 | 있음 | 없음 |

---

## 3. 확산모델: Forward & Reverse Process

### 3.1 순방향 과정 (Forward Process)

**Definition 3.1.** 순방향 과정은 고정된 마르코프 체인이다:

$$x_0 \to z_1 \to z_2 \to \cdots \to z_T \sim \mathcal{N}(0, I)$$

$$q(z_t \mid z_{t-1}) = \mathcal{N}(z_t;\; \sqrt{1-\beta_t}\, z_{t-1},\; \beta_t I)$$

여기서 $\{\beta_t\}_{t=1}^{T}$는 미리 정해진 노이즈 스케줄이다.

**Theorem 3.1 (임의 시점 Marginal).**
$\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$로 정의하면:

$$q(z_t \mid x_0) = \mathcal{N}(z_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t) I)$$

*증명 스케치.* 가우시안의 선형 변환 성질을 반복 적용한다. $z_t = \sqrt{1-\beta_t}\, z_{t-1} + \sqrt{\beta_t}\, \varepsilon_t$를 재귀적으로 전개하면, $z_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$을 얻는다. $\square$

이 결과 덕분에 학습 시 중간 단계를 거치지 않고 임의의 $t$에서 직접 $z_t$를 샘플링할 수 있다.

### 3.2 역방향 과정 (Reverse Process)

역방향에서 학습하는 것은 $p_\theta(z_{t-1} \mid z_t)$이다:

$$p_\theta(z_{t-1} \mid z_t) = \mathcal{N}(z_{t-1};\; \mu_\theta(z_t, t),\; \sigma_t^2 I)$$

---

## 4. DDPM의 ELBO와 학습 목표

### 4.1 ELBO 분해

확산모델의 ELBO를 시간 단계별로 분해하면:

$$L = \mathbb{E}_q\!\left[-\log p_\theta(x|z_1) + \sum_{t=2}^{T} \text{KL}(q(z_{t-1}|z_t, x_0) \| p_\theta(z_{t-1}|z_t)) + \text{KL}(q(z_T|x_0) \| p(z_T))\right]$$

여기서 $q(z_{t-1}|z_t, x_0)$는 순방향 과정의 사후분포(posterior)로, 닫힌 형태의 가우시안이다:

$$q(z_{t-1} \mid z_t, x_0) = \mathcal{N}(z_{t-1};\; \tilde{\mu}_t(z_t, x_0),\; \tilde{\beta}_t I)$$

두 가우시안 사이의 KL은 **평균의 차이의 제곱**에 비례하므로, 각 시간 단계의 손실은 결국 $\|\tilde{\mu}_t - \mu_\theta\|^2$ 형태가 된다.

### 4.2 Simplified Loss (노이즈 예측)

**Theorem 4.1 (Ho et al., 2020).** $\tilde{\mu}_t(z_t, x_0)$를 $\varepsilon$-예측으로 재매개변수화하면, simplified loss는:

$$L_{\text{simple}} = \mathbb{E}_{t \sim U[1,T],\; x_0,\; \varepsilon \sim \mathcal{N}(0,I)}\!\left[\|\varepsilon - \varepsilon_\theta(\underbrace{\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \varepsilon}_{z_t},\; t)\|^2\right]$$

**직관:** 깨끗한 이미지에 노이즈를 섞은 후, 신경망 $\varepsilon_\theta$가 **어떤 노이즈가 추가되었는지** 예측한다.

```python
# DDPM 학습 의사코드
def ddpm_train_step(model, x_0):
    t = randint(1, T)                            # 랜덤 시간 선택
    eps = randn_like(x_0)                         # 노이즈 샘플링
    alpha_bar_t = alpha_bar_schedule[t]
    z_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps  # 노이즈 추가
    eps_pred = model(z_t, t)                      # 노이즈 예측
    loss = mse(eps_pred, eps)                     # 단순 MSE!
    return loss
```

---

## 5. EBM에서 Score 함수로

### 5.1 Score 함수의 정의와 의의

**Definition 5.1 (Score Function).**

$$s(x) \equiv \nabla_x \log p(x)$$

EBM에서 $p_\theta(x) = \exp(-E_\theta(x)) / Z_\theta$이면:

$$s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$

**핵심:** $Z_\theta$는 $x$에 무관하므로 미분 시 사라진다. 정규화 상수를 모르고도 score를 계산할 수 있다.

**기하학적 의미:** Score는 데이터 공간의 각 점에서 **확률 밀도가 증가하는 방향**을 가리키는 벡터장(vector field)이다.

### 5.2 Score Matching (Hyvarinen, 2005)

$$L_{SM}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p_0}\!\left[\|s_\theta(x) - \nabla_x \log p_0(x)\|^2\right]$$

문제: $\nabla_x \log p_0(x)$를 모른다. 부분적분을 통해 동치 변환하면:

$$L_{SM}(\theta) = \frac{1}{2}\mathbb{E}_{x \sim p_0}\!\left[\|s_\theta(x)\|^2 + 2\,\text{Tr}(\nabla_x s_\theta(x))\right] + C$$

$p_0$의 score 없이 학습 가능하지만, **야코비안의 트레이스** $\text{Tr}(\nabla_x s_\theta(x))$는 고차원에서 계산 비용이 크다.

---

## 6. Denoising Score Matching (DSM)과 NCSN

### 6.1 DSM의 핵심 아이디어

데이터에 알려진 노이즈를 추가하면 score가 닫힌 형태로 구해진다:

$$p_\sigma(\tilde{x} \mid x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I) \implies \nabla_{\tilde{x}} \log p_\sigma(\tilde{x} \mid x) = -\frac{\tilde{x} - x}{\sigma^2} = -\frac{\varepsilon}{\sigma}$$

**DSM 손실:**

$$L_{DSM}(\theta; \sigma) = \frac{1}{2}\mathbb{E}_{x, \varepsilon}\!\left[\left\|s_\theta(x + \sigma\varepsilon;\, \sigma) + \frac{\varepsilon}{\sigma}\right\|^2\right]$$

**Theorem 6.1.** $L_{SM}(\theta; \sigma) = L_{DSM}(\theta; \sigma) + C$ (상수 차이만 존재).

즉, DSM은 원래 score matching과 **동치**이면서 계산이 훨씬 간단하다.

### 6.2 NCSN (Noise-Conditioned Score Network)

Song & Ermon (2019): 단일 노이즈 수준에서는 저밀도 영역의 score 추정이 부정확하다.

**해결:** 여러 노이즈 수준에서 동시에 학습한다.

$$L_{NCSN}(\theta) = \sum_{i=1}^{L} \lambda(\sigma_i) \, L_{DSM}(\theta; \sigma_i), \quad 0 < \sigma_1 < \sigma_2 < \cdots < \sigma_L$$

| 노이즈 수준 | 역할 |
|------------|------|
| 큰 $\sigma$ | 넓은 영역 커버, 저밀도까지 도달. **글로벌 구조** 포착 |
| 작은 $\sigma$ | 세밀한 구조 포착. **로컬 디테일** |

하나의 네트워크 $s_\theta(\tilde{x}; \sigma)$가 $\sigma$를 조건으로 받아 모든 노이즈 수준을 처리한다.

### 6.3 DDPM과 NCSN의 관계

$$s_\theta(\tilde{x}, \sigma) = -\frac{\varepsilon_\theta(\tilde{x}, t)}{\sigma}$$

DDPM의 노이즈 예측 = NCSN의 스코어 예측. 수학적으로 동치인 최적화 문제를 서로 다른 언어로 표현한 것이다.

---

## 7. Langevin 동역학과 Annealed Langevin Dynamics

### 7.1 Langevin 동역학

**Definition 7.1.** Score 함수를 이용한 MCMC 샘플링:

$$x_{k+1} = x_k + \eta \cdot s_\theta(x_k) + \sqrt{2\eta} \cdot \varepsilon_k, \quad \varepsilon_k \sim \mathcal{N}(0, I)$$

- $\eta \cdot s_\theta(x_k)$: 확률 높은 방향으로 이동 (deterministic drift)
- $\sqrt{2\eta} \cdot \varepsilon_k$: 랜덤 노이즈 (stochastic diffusion) -- 다양한 샘플 생성

$\eta \to 0$, 스텝 수 $\to \infty$이면 $p_\theta(x)$의 정확한 샘플에 수렴한다.

### 7.2 Annealed Langevin Dynamics

큰 노이즈에서 시작하여 점진적으로 작은 노이즈로 줄이며 샘플링한다:

```
1. x ~ N(0, I)                         # 순수 노이즈에서 시작
2. for l = L, ..., 1 do                 # 노이즈 레벨을 줄여가며
3.     for k = 1, ..., K do             # 각 레벨에서 K번 Langevin 스텝
4.         eps ~ N(0, I)
5.         x <- x + eta_l * s(x, sigma_l) + sqrt(2*eta_l) * eps
6.     end for
7. end for
8. return x                             # 생성된 데이터
```

스텝 크기 $\eta_l \propto \sigma_l^2$로 노이즈 수준에 비례하여 설정한다.

이 과정은 DDPM의 역방향 샘플링과 본질적으로 동일하다.

---

## 8. 두 관점의 통합

### 8.1 대응 관계

| 변분 관점 (DDPM) | 스코어 기반 관점 (NCSN) |
|---|---|
| 순방향: 노이즈 추가 ($q(z_t|z_{t-1})$ 고정) | 데이터에 다중 스케일 노이즈 추가 |
| 역방향: $p_\theta(z_{t-1}|z_t)$ 학습 | 스코어 $s_\theta(\tilde{x}, \sigma)$ 학습 |
| 노이즈 예측: $\varepsilon_\theta$ | 스코어 예측: $s_\theta = -\varepsilon_\theta / \sigma$ |
| ELBO 최소화 | DSM 손실 최소화 |
| 역방향 sampling | Annealed Langevin dynamics |

### 8.2 DDPM 손실의 DSM 해석

$s_\theta(z_t, t) = -\varepsilon_\theta(z_t, t) / \sqrt{1-\bar{\alpha}_t}$로 치환하면:

$$L_{\text{DDPM}} = \mathbb{E}\!\left[(1-\bar{\alpha}_t) \cdot \left\|s_\theta(z_t, t) + \frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}}\right\|^2\right]$$

이는 $\sigma_t = \sqrt{1-\bar{\alpha}_t}$에서의 DSM 손실에 가중치를 곱한 것이다.

### 8.3 SDE 프레임워크 (Song et al., 2021)

연속 시간으로 통합하면:

**순방향 SDE:** $dx = f(x,t)\,dt + g(t)\,dw$

**역방향 SDE:** $dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]\,dt + g(t)\,d\bar{w}$

| 모델 | $f(x,t)$ | $g(t)$ | SDE 유형 |
|------|----------|--------|---------|
| DDPM | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ | VP-SDE |
| NCSN | $0$ | $\sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$ | VE-SDE |

역방향 SDE에서 유일하게 학습이 필요한 항은 $\nabla_x \log p_t(x)$, 즉 **score 함수**이다. 이로써 변분 관점과 스코어 관점이 연속 시간 SDE 아래 완전히 통합된다.

---

## 9. 가이디드 확산 (Guided Diffusion)

### 9.1 Classifier Guidance

조건부 생성을 위해 베이즈 정리를 score에 적용한다:

$$\nabla_x \log p(x \mid c) = \nabla_x \log p(x) + \nabla_x \log p(c \mid x)$$

가이던스 강도 $\lambda$로 증폭:

$$\text{guided score} = \nabla_x \log p(x) + \lambda \cdot \nabla_x \log p(c \mid x)$$

**요구사항:** 노이즈에 강건한 별도의 분류기 $p(c|x_t)$를 학습해야 한다.

### 9.2 Classifier-Free Guidance (Ho & Salimans, 2022)

**핵심 아이디어:** 분류기 없이, 하나의 모델이 조건부/비조건부를 모두 처리한다.

$$\tilde{\varepsilon}_\theta(x_t, c) = (1-\lambda)\,\varepsilon_\theta(x_t, \varnothing) + \lambda\,\varepsilon_\theta(x_t, c)$$

- 학습 시: 조건 $c$를 확률적으로 드롭 ($c \to \varnothing$)
- 추론 시: 비조건부와 조건부 출력을 선형 결합

### 9.3 수학적 해석

**Theorem 9.1.**

$$(1-\lambda)\nabla_x \log p(x) + \lambda \nabla_x \log p(x|c) = \nabla_x \log p(x) + \lambda \nabla_x \log p(c|x) + \text{const}$$

*증명.* $\nabla_x \log p(x|c) = \nabla_x \log p(x) + \nabla_x \log p(c|x) - \nabla_x \log p(c)$에서, $\nabla_x \log p(c)$는 $x$에 무관하므로 const. 이를 대입하면 결과를 얻는다. $\square$

따라서 classifier-free guidance는 **암묵적으로 분류기 그래디언트를 계산**하는 것이다. "별도의 분류기 네트워크가 필요 없다"가 정확한 표현이지, 분류 정보를 사용하지 않는다는 뜻이 아니다.

$\lambda > 1$이면 조건에 과도하게 충실하여 품질은 향상되지만 다양성이 감소한다. 이것이 DALL-E 2, Stable Diffusion, Imagen 등의 핵심 기법이다.

---

## 10. 흔한 오해와 주의점

| # | 오해 | 올바른 이해 |
|---|------|-------------|
| 1 | 순방향 과정도 학습해야 한다 | 순방향은 고정된 가우시안 노이즈 추가. 학습 파라미터 **없음** |
| 2 | Score 함수는 확률값을 알려준다 | Score는 **벡터**(그래디언트). 확률이 증가하는 **방향**을 알려줄 뿐, 값 자체는 알 수 없음 |
| 3 | DDPM과 NCSN은 완전히 다르다 | $s_\theta = -\varepsilon_\theta / \sigma$로 연결. 동일한 학습 목표를 다른 언어로 표현 |
| 4 | Classifier-free guidance에 분류 정보가 없다 | 암묵적으로 $\nabla_x \log p(c|x)$를 계산. 별도 분류기 **네트워크**가 불필요할 뿐 |
| 5 | 확산모델의 잠재 공간은 VAE처럼 저차원 | $z_t$는 $x$와 **같은 차원**. 병목 없음. "잠재"는 노이즈 수준의 계층 구조 |
| 6 | 큰 노이즈 스케일은 불필요 | 작은 $\sigma$만으로는 저밀도 영역의 score가 부정확. 큰 $\sigma$가 글로벌 구조 포착에 필수 |

---

## 11. 핵심 요약

| 개념 | 핵심 |
|------|------|
| **순방향** | $q(z_t|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t) I)$. 고정, 학습 불필요 |
| **역방향** | $p_\theta(z_{t-1}|z_t)$를 학습. 노이즈 예측 $\varepsilon_\theta$로 매개변수화 |
| **DDPM 손실** | $L = \mathbb{E}[\|\varepsilon - \varepsilon_\theta(z_t, t)\|^2]$. 단순 MSE |
| **Score 함수** | $s(x) = \nabla_x \log p(x) = -\nabla_x E(x)$. 정규화 상수 불필요 |
| **DSM** | $L = \mathbb{E}[\|s_\theta(\tilde{x}; \sigma) + \varepsilon/\sigma\|^2]$. Score matching과 동치 |
| **NCSN** | 다중 노이즈 스케일 DSM. 글로벌+로컬 구조 포착 |
| **두 관점 통합** | $s_\theta = -\varepsilon_\theta / \sigma$. 변분 $\equiv$ 스코어 관점 |
| **Langevin** | $x_{k+1} = x_k + \eta \cdot s(x_k) + \sqrt{2\eta} \cdot \varepsilon$. Score 기반 샘플링 |
| **Classifier-Free** | $(1-\lambda)\varepsilon(\varnothing) + \lambda\varepsilon(c)$. 암묵적 분류기 그래디언트 |
| **Trade-off** | Diffusion = 고품질 + 다양성, 대가는 느린 샘플링 |

**관련 개념 연결:** EBM의 에너지 함수/Langevin MCMC $\to$ [19장], VAE의 ELBO 구조 $\to$ [19장], Transformer 기반 노이즈 예측 네트워크 $\to$ [16장]

**참고 문헌:**
- Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020 (DDPM)
- Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution," NeurIPS 2019 (NCSN)
- Song et al., "Score-Based Generative Modeling through SDEs," ICLR 2021
- Ho & Salimans, "Classifier-Free Diffusion Guidance," NeurIPS Workshop 2022
- Hyvarinen, "Estimation of Non-Normalized Statistical Models," JMLR 2005 (Score Matching)

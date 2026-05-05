---
title: "12. 딥러닝 최적화"
slug: 12-deep-learning-optimization
order: 12
---

# 12. 딥러닝 최적화

## 12.1 동기부여 및 개요

딥러닝 모델의 학습은 본질적으로 **최적화 문제**이다. 수백만에서 수십억 개의 파라미터 $\theta$에 대해 손실 함수 $L_S(\theta)$를 최소화하는 것이 목표이다. 선형 회귀처럼 닫힌 형태의 해가 존재하는 볼록 문제와 달리, 심층 신경망의 손실은 일반적으로 **비볼록(nonconvex)**이므로 경사 기반 반복 최적화(gradient-based iterative optimization)가 필수적이다.

본 장에서는 볼록 최적화의 이론적 기반에서 출발하여, 경사 하강법(GD), 확률적 경사 하강법(SGD), 모멘텀, Adam 등의 최적화 알고리즘을 수학적으로 다루고, 역전파(backpropagation)의 메커니즘, 그리고 수렴 이론과 학습률 스케줄링을 논의한다.

**연결**: 11장의 신경망 구조를 **어떻게 학습시키는가**에 대한 답이다. 13장의 일반화/정규화와 밀접하게 관련된다.

---

## 12.2 볼록 최적화의 기초

### 12.2.1 볼록 집합과 볼록 함수

**정의 12.1 (볼록 집합).** 집합 $X \subseteq \mathbb{R}^n$이 볼록이란, 임의의 $x, y \in X$와 $\lambda \in [0,1]$에 대해 $\lambda x + (1-\lambda)y \in X$가 성립함을 의미한다.

**정의 12.2 (볼록 함수).** $f: X \to \mathbb{R}$이 볼록이란, 임의의 $x, y \in X$와 $\lambda \in [0,1]$에 대해:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**정리 12.1 (볼록 함수의 동치 조건).** 미분 가능한 함수 $f$에 대해 다음은 동치이다:

1. $f$가 볼록
2. **(접선 부등식)**: $f(x) \geq f(y) + \nabla f(y)^\top(x - y), \quad \forall x, y$
3. **(단조 기울기)**: $\langle \nabla f(x) - \nabla f(y), x - y \rangle \geq 0, \quad \forall x, y$
4. **(헤시안 PSD)**: $\nabla^2 f(x) \succeq 0, \quad \forall x$ (2회 미분 가능 시)

*직관적 해석*: 접선 부등식은 볼록 함수의 접선이 항상 함수 아래에 위치함을 의미한다. 즉, 1차 근사가 항상 과소추정한다.

**정리 12.2 (Jensen 부등식).** $f$가 볼록이면:

$$f\!\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i), \quad \lambda_i \geq 0, \; \sum_i \lambda_i = 1$$

확률 변수 버전: $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$

*응용 예시 (KL-divergence의 비음수성)*: $-\log$가 볼록이므로:

$$D_{\text{KL}}(p \| q) = \mathbb{E}_p\!\left[-\log \frac{q(X)}{p(X)}\right] \geq -\log \mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = 0$$

### 12.2.2 강볼록성과 매끄러움

**정의 12.3 ($m$-강볼록성).** $f$가 $m$-강볼록 ($m > 0$)이란:

$$f(x) \geq f(y) + \nabla f(y)^\top(x-y) + \frac{m}{2}\|x-y\|^2, \quad \forall x, y$$

동치 조건: $\nabla^2 f(x) \succeq mI$ (2회 미분 가능 시)

**정의 12.4 ($\beta$-매끄러움).** $\nabla f$가 $\beta$-Lipschitz 연속이란:

$$\|\nabla f(x) - \nabla f(y)\| \leq \beta\|x - y\|, \quad \forall x, y$$

동치 조건: $\nabla^2 f(x) \preceq \beta I$. 이는 1차 근사의 오차가 2차로 제한됨을 의미:

$$\left|f(x) - f(y) - \nabla f(y)^\top(x-y)\right| \leq \frac{\beta}{2}\|x-y\|^2$$

*직관적 비유*: 강볼록성은 함수가 "최소한 이 정도는 휘어있다"는 하한, 매끄러움은 "최대한 이 정도만 휘어있다"는 상한이다.

**정의 12.5 (Polyak-Lojasiewicz 조건).** $f$가 $c$-PL 조건을 만족한다는 것은:

$$\frac{1}{2}\|\nabla f(x)\|^2 \geq c\left(f(x) - f(x^*)\right), \quad \forall x$$

$m$-강볼록이면 PL 조건을 만족하지만, 역은 성립하지 않는다. PL 조건은 **비볼록 함수에도 적용 가능**하여 DNN 분석에 유용하다.

---

## 12.3 경사 하강법

### 12.3.1 기본 경사 하강법 (GD)

**알고리즘 12.1 (Gradient Descent).**

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

여기서 $\eta > 0$는 학습률(learning rate)이다.

**정리 12.3 (2차 함수에서의 수렴 조건).** 2차 손실 $L(\theta) = \frac{1}{2}\theta^\top A\theta + a^\top\theta + b$ ($A \succ 0$)에서 GD가 수렴하려면:

$$\eta < \frac{2}{\lambda_{\max}(A)}$$

*증명 스케치.* $\delta = -\eta \nabla L$를 대입하면 $L(\theta_{t+1}) \leq L(\theta_t) - \eta(1 - \frac{\eta}{2}\lambda_{\max}(A))\|\nabla L\|^2$이다. $\eta < 2/\lambda_{\max}$이면 괄호 안이 양수이므로 매 스텝 손실이 감소한다.

### 12.3.2 수렴 속도 분류

| 수렴 속도 | 표현 | 조건 |
|----------|------|------|
| Sublinear | $L_t - L^* \leq O(1/t)$ | 볼록 + smooth |
| Linear | $L_t - L^* \leq O(\rho^t)$, $\rho < 1$ | 강볼록 + smooth |
| Quadratic | $\|\theta_{t+1} - \theta^*\| \leq c\|\theta_t - \theta^*\|^2$ | Newton's method |

**정리 12.4 (PL + Smooth에서의 선형 수렴).** $\beta$-smooth + $c$-PL 조건 하에서 $\eta = 1/\beta$로 GD를 수행하면:

$$L_t - L^* \leq \left(1 - \frac{c}{\beta}\right)^t (L_0 - L^*)$$

*증명.* $\beta$-smoothness에서 $L_{t+1} - L_t \leq -\frac{1}{2\beta}\|\nabla L_t\|^2$, PL 조건에서 $\|\nabla L_t\|^2 \geq 2c(L_t - L^*)$를 결합하면 $L_{t+1} - L^* \leq (1 - c/\beta)(L_t - L^*)$이다. 이를 재귀 적용하면 결과를 얻는다. $\square$

**조건수(condition number)** $\kappa = \beta/m$: 수렴 속도가 $(1 - 1/\kappa)^t$이므로, 조건수가 클수록 수렴이 느리다.

---

## 12.4 확률적 경사 하강법 (SGD)

### 12.4.1 미니배치 SGD

전체 경험적 위험 $L_S(\theta) = \frac{1}{n}\sum_{i=1}^{n}\ell(y_i, f_\theta(x_i))$의 기울기 대신, 미니배치 $B \subset S$ ($|B| = b$)의 기울기를 사용:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L_B(\theta_t)$$

**보조정리 12.1 (불편 추정).** $\mathbb{E}_B[\nabla L_B(\theta)] = \nabla L_S(\theta)$. 즉, 미니배치 기울기는 전체 기울기의 불편 추정량이다.

### 12.4.2 SGD의 이중적 역할

SGD의 확률적 잡음(noise)은 단점만이 아니라 여러 이점을 제공한다:

1. **계산 효율성**: 한 스텝의 비용 $O(b)$ vs 전체 GD의 $O(n)$
2. **암묵적 정규화**: 잡음이 sharp minima에서 탈출시키고 flat minima로 유도
3. **안장점 탈출**: 잡음이 saddle point에서 벗어나는 데 도움

**배치 크기-학습률 관계**: 선형 스케일링 규칙에 따르면, 배치 크기를 $k$배 하면 학습률도 $k$배 한다 (Goyal et al., 2017).

---

## 12.5 모멘텀 방법

### 12.5.1 Heavy Ball Momentum (Polyak, 1964)

$$m_t = \beta m_{t-1} + \nabla L(\theta_t), \quad \theta_{t+1} = \theta_t - \eta m_t$$

$m_t = \sum_{i=0}^{t} \beta^{t-i} \nabla L(\theta_i)$로, 기울기의 **지수 이동 평균(EMA)**이다.

### 12.5.2 Nesterov Accelerated Gradient (NAG, 1983)

$$\tilde{\theta}_t = \theta_t + \beta(\theta_t - \theta_{t-1}), \quad \theta_{t+1} = \tilde{\theta}_t - \eta \nabla L(\tilde{\theta}_t)$$

NAG는 "앞을 내다보고(look-ahead)" 기울기를 계산하여 오버슈팅을 사전에 보정한다.

**수렴 비교 (2차 볼록 문제, $\kappa = \lambda_{\max}/\lambda_{\min}$)**:

| 방법 | 수렴 속도 |
|------|---------|
| GD | $O\!\left(\left(\frac{\kappa-1}{\kappa+1}\right)^{2t}\right)$ |
| Heavy Ball | $O\!\left(\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^{2t}\right)$ |
| NAG | $O(1/t^2)$ (가속, accelerated) |

NAG는 1차 방법 중 이론적으로 최적의 수렴 속도를 달성한다 (Nesterov, 1983).

---

## 12.6 적응적 학습률: AdaGrad, RMSProp, Adam

### 12.6.1 Preconditioned Gradient Descent

기본 아이디어: 기울기 $g$ 대신 $M^{-1}g$를 사용하여 각 파라미터 방향에 다른 학습률을 적용:

$$\theta_{t+1} = \theta_t - \eta M_t^{-1} g_t, \quad M_t = \text{diag}(\sqrt{s_t + \varepsilon})$$

### 12.6.2 알고리즘 비교

**AdaGrad** (Duchi et al., 2011): $s_t = \sum_{i=1}^{t} g_i^2$ (누적합). 문제: $s_t$가 단조 증가하여 학습률이 0으로 수렴.

**RMSProp** (Hinton, 2012): $s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2$ (EMA로 해결).

**알고리즘 12.2 (Adam, Kingma & Ba 2015).**

```
초기화: m₀ = 0, s₀ = 0
for t = 1, 2, ... do:
    g_t = ∇L(θ_t)
    m_t = β₁ m_{t-1} + (1-β₁) g_t          # 1차 모멘트 (momentum)
    s_t = β₂ s_{t-1} + (1-β₂) g_t²          # 2차 모멘트 (RMSProp)
    m̂_t = m_t / (1 - β₁ᵗ)                   # 편향 보정
    ŝ_t = s_t / (1 - β₂ᵗ)                   # 편향 보정
    θ_{t+1} = θ_t - η · m̂_t / (√ŝ_t + ε)
```

기본 하이퍼파라미터: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

### 12.6.3 편향 보정의 필요성

$s_0 = 0$으로 초기화하면 $\mathbb{E}[s_t] = \mathbb{E}[g^2](1-\beta_2^t)$으로 편향된다. 따라서 $\hat{s}_t = s_t/(1-\beta_2^t)$로 보정하여 $\mathbb{E}[\hat{s}_t] \approx \mathbb{E}[g^2]$을 달성한다. 특히 $t$가 작을 때 보정 효과가 크다.

**기하학적 해석**: $M$이 Hessian의 대각 근사 역할을 하므로, 적응적 방법은 Newton's method의 $O(p)$ 비용 근사이다.

### 12.6.4 Adam의 한계와 변형

- **AdamW** (Loshchilov & Hutter, 2019): weight decay를 기울기가 아닌 파라미터에 직접 적용하여 정규화 효과 개선
- Adam은 일부 볼록 문제에서 수렴하지 않을 수 있음 (Reddi et al., 2018)
- CV에서는 SGD+momentum이, NLP에서는 Adam이 표준적으로 사용됨

---

## 12.7 역전파와 자동 미분

### 12.7.1 계산 그래프와 연쇄 법칙

역전파의 본질은 **역방향 모드 자동 미분(reverse-mode autodiff)**이다. 스칼라 손실 $L$에 대해 모든 파라미터의 기울기를 한 번의 역방향 패스로 계산한다.

**벡터-야코비안 곱(VJP) 체인**:

$$\frac{\partial L}{\partial W_1} = \underbrace{\frac{\partial L}{\partial z_2}}_{1 \times d_2} \underbrace{\frac{\partial z_2}{\partial x_1}}_{d_2 \times d_1} \underbrace{\frac{\partial x_1}{\partial z_1}}_{d_1 \times d_1} \underbrace{\frac{\partial z_1}{\partial W_1}}_{d_1 \times |W_1|}$$

### 12.7.2 주요 미분 공식

| 연산 | 미분 |
|------|------|
| Softmax + CE: $L = -\log p_y$ | $\frac{\partial L}{\partial z} = p - e_y$ |
| MSE: $L = \frac{1}{2}\|z-z_0\|^2$ | $\frac{\partial L}{\partial z} = z - z_0$ |
| ReLU$(z)$ | $\text{diag}(\mathbf{1}(z > 0))$ |
| 선형 $Wx$ | $\frac{\partial(Wx)}{\partial x} = W$ |
| 스칼라 $a^\top W x$ | $\frac{\partial}{\partial W} = xa^\top$ |

**시간 복잡도**: $L$개 층, 각 층의 차원이 $n$일 때 역방향 패스는 $O(Ln^2)$으로, 순방향 패스와 같은 차수이다.

### 12.7.3 자동 미분 vs 수치 미분 vs 기호 미분

| 방법 | 정확도 | 비용 | 문제점 |
|------|--------|------|--------|
| 수치 미분 | 근사 | $O(p)$ 순전파 | 수치 오차, 느림 |
| 기호 미분 | 정확 | 가변 | 표현식 폭발(expression swell) |
| 자동 미분 | 정확 | ~순전파 비용 | 없음 (실용적 최선) |

```python
# PyTorch 자동 미분 예시
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad)  # tensor([7.]) = 2*x + 3 = 2*2 + 3
```

---

## 12.8 학습률 스케줄

### 12.8.1 주요 스케줄 전략

| 스케줄 | 수식 | 특징 |
|--------|------|------|
| Step decay | $\eta_t = \eta_0 \gamma^{\lfloor t/T_s \rfloor}$ | 단순, 널리 사용 |
| Square-root | $\eta_t = \eta_0 / \sqrt{t+1}$ | 이론적 보장 |
| Cosine annealing | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos(\pi t/T))$ | 실전 인기 |
| Warmup + decay | 초기 선형 증가 후 감소 | Adam 초기 불안정 해소 |

### 12.8.2 이론적 요건

**보조정리 12.2 (Robbins-Monro 조건).** SGD의 수렴을 위해 학습률 스케줄은 다음을 만족해야 한다:

$$\sum_{t=1}^{\infty} \eta_t = \infty \quad (\text{도달 가능성}), \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty \quad (\text{잡음 소멸})$$

$\eta_t = O(1/\sqrt{t})$가 이를 만족한다.

---

## 12.9 기울기 문제와 해결책

### 12.9.1 기울기 폭발과 소실

깊은 네트워크에서 역전파 시 각 층의 야코비안이 곱해지므로:

- 각 층의 "배율"이 1보다 크면 기울기가 지수적으로 **폭발**
- 1보다 작으면 기울기가 지수적으로 **소실**

### 12.9.2 해결 전략

| 문제 | 해결책 | 원리 |
|------|--------|------|
| 폭발 | Gradient Clipping: $g' = \min(1, c/\|g\|) \cdot g$ | 기울기 노름 제한 |
| 소실 | ReLU (비포화 활성화) | $\text{ReLU}'(x) = 1$ ($x > 0$) |
| 소실 | Residual Connection: $y = F(x) + x$ | $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$ |
| 양쪽 | 정규화 층 (BN, LN) | 활성화 분포 안정화 |
| 양쪽 | 적절한 초기화 (Xavier, He) | 분산 유지 |

### 12.9.3 Batch Normalization

**정의 12.6 (Batch Normalization, Ioffe & Szegedy 2015).** 미니배치 $B = \{x^{(1)}, \ldots, x^{(m)}\}$에 대해:

$$\hat{x}_k^{(i)} = \frac{x_k^{(i)} - \mu_{B,k}}{\sqrt{\sigma_{B,k}^2 + \epsilon}}, \quad y^{(i)} = \gamma \odot \hat{x}^{(i)} + \beta$$

$\gamma, \beta$는 학습 가능한 스케일/시프트 파라미터. 추론 시에는 EMA로 추적한 통계량을 사용한다.

**BN vs LN**: BN은 배치 + 공간 축으로 정규화(CV 표준), LN은 채널 + 공간 축으로 정규화(NLP/Transformer 표준). Santurkar et al. (2018)에 따르면 BN의 실제 효과는 **손실 landscape을 매끄럽게 만드는 것**이다.

---

## 12.10 비볼록 최적화와 DNN

DNN의 손실은 일반적으로 비볼록이지만, SGD가 놀랍게도 좋은 해를 찾는 이유에 대해 여러 가설이 있다:

1. **Overparameterized DNN에서 대부분의 국소 최솟값은 전역 최솟값과 비슷한 손실을 가진다**
2. **SGD의 잡음이 안장점 탈출과 flat minima 선호에 도움**을 준다
3. **PL 조건이 국소적으로 성립**할 수 있어 선형 수렴이 가능하다

---

## 12.11 흔한 오해와 주의점

| 오해 | 올바른 이해 |
|------|-----------|
| "학습률이 작을수록 항상 좋다" | 너무 작으면 수렴이 느리고 sharp minima에 갇힐 수 있다 |
| "Adam은 항상 SGD보다 낫다" | CV에서는 SGD+momentum이 더 나은 일반화를 보이기도 한다 |
| "비볼록이면 GD는 무용지물이다" | Overparameterized DNN에서 SGD는 실전적으로 잘 작동한다 |
| "역전파가 순전파보다 훨씬 비싸다" | VJP 체인에서 역전파의 시간 복잡도는 순전파의 상수배 |
| "BN은 internal covariate shift 해결용이다" | 후속 연구에 따르면 loss landscape을 smooth하게 만드는 효과가 핵심 |

---

## 12.12 핵심 요약

```
최적화 알고리즘 진화:

GD ──> SGD ──> SGD+Momentum ──> Adam (= Momentum + RMSProp)
  │               │  Heavy Ball     │  1차 모멘트 m (EMA of g)
  │               │  NAG            │  2차 모멘트 s (EMA of g²)
  │               │                 │  편향 보정
  │
  └── Preconditioned GD: θ ← θ - ηM⁻¹g
        AdaGrad → RMSProp → Adam → AdamW

핵심 수렴 결과:
  β-smooth + c-PL + η=1/β  →  L_t - L* ≤ (1-c/β)^t (L₀-L*)

실전 체크리스트:
  [1] 역전파 (VJP chain) + PyTorch autograd
  [2] 학습률 스케줄: warmup + cosine/step decay
  [3] Gradient clipping (폭발 방지)
  [4] Residual connection + BN/LN (소실 방지)
  [5] 하이퍼파라미터: η, batch size, β₁, β₂, ε
```

**참고문헌**: Polyak (1964), Nesterov (1983), Duchi et al. (2011), Kingma & Ba (2015), Ioffe & Szegedy (2015), Loshchilov & Hutter (2019), Santurkar et al. (2018)

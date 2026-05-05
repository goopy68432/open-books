---
title: "11. 기계학습 모델과 신경망 기초"
slug: 11-ml-models-neural-networks
order: 11
---

# 11. 기계학습 모델과 신경망 기초

## 11.1 동기부여 및 개요

선형 모델은 결정 경계(decision boundary)가 초평면(hyperplane)으로 제한되어 XOR과 같은 간단한 비선형 문제조차 해결할 수 없다. SVM은 커널 트릭을 통해 비선형 분류를 달성하였으나, 적절한 커널 함수를 사람이 설계해야 한다는 근본적 한계가 있다. 신경망(neural network)은 **특징 추출기 자체를 데이터로부터 학습**함으로써 이 한계를 돌파한다.

본 장에서는 퍼셉트론에서 시작하여 SVM, 커널 메서드를 거쳐 다층 퍼셉트론(MLP)에 이르는 모델의 진화 과정을 추적하고, 신경망의 이론적 기반인 보편 근사 정리와 귀납적 편향의 역할을 살펴본다.

**학습 목표**: 선형 모델 $\to$ SVM $\to$ 커널 메서드 $\to$ 신경망으로의 전환이 왜 필요했는지 이해하고, 각 모델의 수학적 구조와 한계를 정확히 파악한다.

---

## 11.2 퍼셉트론과 선형 분류의 한계

### 11.2.1 퍼셉트론의 정의

**정의 11.1 (퍼셉트론).** 입력 $x \in \mathbb{R}^d$에 대한 퍼셉트론의 결정 함수는 다음과 같다:

$$f(x; w) = \mathbf{1}(w^\top x \geq 0)$$

여기서 $\mathbf{1}(\cdot)$은 지시 함수(indicator function)이며, 결정 경계는 $w^\top x = 0$인 초평면이다.

Rosenblatt(1957)의 원래 구조는 $S$(감각 유닛) $\to$ $A$(연합 유닛, 고정 가중치) $\to$ $R$(출력 유닛, 학습 가능 가중치)의 토폴로지였다. 결정 함수는 $l(x) = \text{sign}\left(\sum_i \alpha_i z_i(x)\right)$이며, $z_i(x)$는 비선형 변환된 특징이고 $\alpha_i$만 학습한다. 즉, 퍼셉트론도 이미 **변환된 공간에서의 선형 분류**라는 아이디어를 내포하고 있었다.

### 11.2.2 XOR 문제와 선형 분리 불가능성

**정리 11.1 (Minsky & Papert, 1969).** 단일 퍼셉트론은 XOR 함수를 실현할 수 없다.

*증명 스케치.* XOR의 입출력 관계는 $(0,0)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$, $(1,1)\to 0$이다. $w_1 x_1 + w_2 x_2 + b = 0$인 직선이 이 네 점을 양쪽으로 올바르게 분리하려면 모순이 발생한다. $(0,1)$과 $(1,0)$이 양의 영역에, $(0,0)$과 $(1,1)$이 음의 영역에 있어야 하므로 $w_2 + b > 0$, $w_1 + b > 0$이지만 $w_1 + w_2 + b < 0$이고 $b < 0$이어야 한다. 처음 두 조건에서 $w_1 + w_2 + 2b > 0$이므로 $w_1 + w_2 > -2b > 0$인데, 이는 $w_1 + w_2 + b < 0$과 결합하면 $b < -(w_1+w_2) < 0$, 즉 $b < 0$이다. 하지만 $w_1 + w_2 > 0$과 $w_1 + w_2 + b < 0$에서 $b < -(w_1 + w_2)$이고, 동시에 $w_1 + b > 0$에서 $b > -w_1$이므로, $-w_1 < b < -(w_1 + w_2)$, 즉 $w_2 < 0$. 유사하게 $w_1 < 0$이 되어 $w_1 + w_2 > 0$에 모순된다. $\square$

이 결과는 1970년대 "AI 겨울"의 한 원인이 되었으나, 다층 구조가 이를 해결할 수 있음은 이미 인식되고 있었다.

---

## 11.3 서포트 벡터 머신 (SVM)

### 11.3.1 최대 마진 분류기

**정의 11.2 (기하학적 마진).** 결정 초평면 $w^\top x + b = 0$에 대한 마진은 다음과 같다:

$$\rho(w) = \min_{x_\pm \in \mathcal{X}_\pm} \frac{w^\top(x_+ - x_-)}{\|w\|} = \frac{2}{\|w\|}$$

여기서 $y_i(w^\top x_i + b) \geq 1$의 정규화 조건을 사용한다.

**정의 11.3 (Hard-Margin SVM, Primal).**

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1, \quad \forall i \in [N]$$

이는 이차 목적함수(quadratic objective)와 선형 제약(linear constraints)을 가진 볼록 최적화 문제이다.

### 11.3.2 라그랑주 쌍대 문제

라그랑지안을 구성하면:

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 + \sum_{i=1}^{N} \alpha_i\left(1 - y_i(w^\top x_i + b)\right)$$

KKT 조건 $\nabla_w \mathcal{L} = 0$으로부터 $w = \sum_i \alpha_i y_i x_i$, $\nabla_b \mathcal{L} = 0$으로부터 $\sum_i \alpha_i y_i = 0$을 얻는다.

**정리 11.2 (SVM Dual).** Primal 변수를 소거하면 다음 쌍대 문제를 얻는다:

$$\max_\alpha \quad -\frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i$$
$$\text{s.t.} \quad \sum_i \alpha_i y_i = 0, \quad \alpha_i \geq 0$$

**핵심 관찰**: 쌍대 문제에서 데이터는 **내적** $x_i^\top x_j$으로만 나타난다. 이것이 커널 트릭의 이론적 전제 조건이다.

### 11.3.3 Soft-Margin SVM

선형 분리 불가능한 경우, 슬랙 변수 $\zeta_i \geq 0$을 도입한다:

$$\min_{w,b,\zeta} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{N} \zeta_i \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \zeta_i$$

하이퍼파라미터 $C$는 마진 폭과 오분류 허용 사이의 트레이드오프를 제어한다. Hinge loss $\ell_{\text{hinge}}(z) = \max(0, 1 - z)$와의 관계로, soft-margin SVM은 정규화된 hinge loss 최소화와 동치이다.

| $C$ 값 | 서포트 벡터 수 | 마진 폭 | 특성 |
|--------|--------------|---------|------|
| 작음 (0.01) | 많음 (~260) | 넓음 | 높은 정규화 |
| 중간 (1) | 중간 (~78) | 중간 | 균형 |
| 큼 (100) | 적음 (~45) | 좁음 | 낮은 정규화 |

---

## 11.4 커널 메서드

### 11.4.1 커널 트릭의 원리

**정의 11.4 (커널 함수).** 커널 함수 $K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$는 어떤 특징 공간 $\mathcal{F}$와 특징 맵 $\phi: \mathcal{X} \to \mathcal{F}$에 대해 다음을 만족한다:

$$K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{F}$$

SVM의 결정 함수에 커널을 적용하면:

$$f(x) = \sum_{i=1}^{N} \alpha_i y_i K(x_i, x) + b$$

**정리 11.3 (Mercer's Theorem, 비형식적).** 양의 준정부호(positive semi-definite) 커널은 어떤 특징 공간에서의 내적에 대응한다. 즉, 커널 행렬 $[K]_{ij} = K(x_i, x_j)$가 PSD이면 유효한 커널이다.

대표적 커널 함수:

| 커널 | 수식 | 특징 공간 차원 |
|------|------|--------------|
| 선형 | $K(x,x') = x^\top x'$ | $d$ |
| 다항식 | $K(x,x') = (x^\top x' + c)^p$ | $\binom{d+p}{p}$ |
| RBF(가우시안) | $K(x,x') = \exp(-\gamma\|x-x'\|^2)$ | 무한 |

RBF 커널은 무한 차원 특징 공간에 대응하므로, SVM+RBF는 사실상 무한 차원에서 선형 분류를 수행한다.

### 11.4.2 비모수적 커널 방법

**커널 밀도 추정 (KDE)**:

$$\hat{p}(x) = \frac{1}{N}\sum_{n=1}^{N} K_h(x - x_n), \quad K_h(u) = \frac{1}{\sqrt{2\pi h^2}}\exp\left(-\frac{u^2}{2h^2}\right)$$

**Nadaraya-Watson 커널 회귀**:

$$\hat{f}(x) = \sum_{i=1}^{N} y_i \cdot \frac{K(x, x_i)}{\sum_{j=1}^{N} K(x, x_j)}$$

이들은 **비모수적(non-parametric)** 방법으로, 모델 복잡도가 데이터 크기에 따라 증가한다. 그러나 모든 커널 방법의 한계는 **커널 함수를 사람이 선택해야 한다**는 것이다.

---

## 11.5 학습 가능한 특징 추출기: 신경망의 탄생

### 11.5.1 고정 기저함수에서 학습 가능한 기저함수로

모델의 진화 경로:

$$\underbrace{w^\top x}_{\text{선형 모델}} \to \underbrace{w^\top \phi(x)}_{\text{비선형 기저함수}} \to \underbrace{w^\top \phi(x; \theta')}_{\text{학습 가능한 특징 추출기}}$$

고정 기저함수 모델 $f(x;\theta) = W\phi(x) + b$에서 $\phi$는 사람이 설계한 비선형 변환이다. **신경망의 핵심 혁신**은 $\phi$를 파라미터 $\theta'$로 매개변수화하여 데이터로부터 학습하는 것이다:

$$f(x; \theta, \theta') = W\phi(x; \theta') + b$$

### 11.5.2 비선형 활성화 함수의 필요성

**정리 11.4 (선형 합성의 축소).** 비선형 활성화 함수가 없는 다층 네트워크는 단일 선형 변환과 동치이다:

$$W_{L-1} W_{L-2} \cdots W_0 x = W' x$$

*직관적 해석*: 아무리 층을 깊게 쌓아도, 비선형 활성화 없이는 표현력이 증가하지 않는다.

### 11.5.3 주요 활성화 함수

| 함수 | 수식 | 범위 | 미분 연속성 | 주요 특징 |
|------|------|------|-----------|---------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $(0,1)$ | $C^\infty$ | 포화(saturation) 문제 |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1,1)$ | $C^\infty$ | 영점 대칭 |
| ReLU | $\max(0, x)$ | $[0,\infty)$ | $C^0$ | 계산 효율, dead neuron 문제 |
| LeakyReLU | $\max(\alpha x, x)$ | $(-\infty,\infty)$ | $C^0$ | dead neuron 완화 |
| GELU | $x \cdot \Phi(x)$ | $[-0.17,\infty)$ | $C^\infty$ | Transformer 표준 |
| SiLU/Swish | $x \cdot \sigma(x)$ | $[-0.278,\infty)$ | $C^\infty$ | smooth ReLU 변형 |

여기서 $\Phi$는 표준 정규분포의 CDF이다.

**Vanishing Gradient 문제**: Sigmoid의 도함수 $\sigma'(x) = \sigma(x)(1-\sigma(x))$는 $|x|$가 크면 0에 수렴하여, 깊은 네트워크에서 역전파 시 기울기가 소실된다. ReLU는 양수 영역에서 도함수가 1이므로 이 문제를 해결한다.

**Dead ReLU 문제**: $\text{ReLU}'(x) = 0$ ($x < 0$)이므로, 음수 영역에 진입한 뉴런은 기울기를 전혀 받지 못해 영구적으로 비활성화된다. LeakyReLU, ELU 등이 이를 해결한다.

---

## 11.6 다층 퍼셉트론과 보편 근사 정리

### 11.6.1 2층 신경망의 구조

**정의 11.5 (2층 신경망).** 입력 $x \in \mathbb{R}^{d_0}$에 대해:

$$\phi(x; W_0) = \sigma(W_0 x) \in \mathbb{R}^{d_1}, \quad f(x; w, W_0) = w^\top \sigma(W_0 x)$$

여기서 $W_0 \in \mathbb{R}^{d_1 \times d_0}$은 은닉층 가중치, $w \in \mathbb{R}^{d_1}$은 출력층 가중치이다.

**XOR 해결 예시**: Heaviside step function $\sigma$를 사용하면:

$$h_1 = \sigma(x_1 + x_2 - 1.5) = \text{AND}(x_1, x_2)$$
$$h_2 = \sigma(x_1 + x_2 - 0.5) = \text{OR}(x_1, x_2)$$
$$y = \sigma(-h_1 + h_2 - 0.5) = \text{XOR}(x_1, x_2)$$

은닉층이 입력 공간을 **선형 분리 가능한 공간으로 재매핑**한다. 이것이 표현 학습(representation learning)의 원형이다.

### 11.6.2 보편 근사 정리

**정리 11.5 (Universal Approximation Theorem, Cybenko 1989; Hornik et al. 1989).** $\sigma$가 비상수, 유계, 단조증가인 연속 함수이면, 임의의 $\epsilon > 0$와 $f \in C([0,1]^n)$에 대해, 적절한 $N$과 파라미터 $\{w_i, b_i, \alpha_i\}_{i=1}^{N}$이 존재하여:

$$\sup_{x \in [0,1]^n} \left| f(x) - \sum_{i=1}^{N} \alpha_i \sigma(w_i^\top x + b_i) \right| < \epsilon$$

*증명 스케치 (계단 함수 근사)*: 각 은닉 뉴런이 하나의 bump(단위 구간 함수)를 생성하고, 이들의 가중합으로 목표 함수를 계단 근사한다. 두 임계값 $T_1, T_2$에서 활성화하는 뉴런 쌍으로 폭이 $T_2 - T_1$인 bump를 만들고, $N$개의 bump를 적절한 높이 $h_i$로 합산하면 $\sum_i h_i \cdot \text{bump}_i(x) \approx f(x)$이다.

**UAT의 한계** (중요):
1. 필요한 뉴런 수 $N$에 대한 상한을 제공하지 않는다 (기하급수적일 수 있음)
2. 파라미터를 **찾는 방법**(최적화 가능성)을 보장하지 않는다
3. **일반화 성능**을 보장하지 않는다

### 11.6.3 MLP의 일반 구조

**정의 11.6 ($L$층 MLP).** 순전파(forward pass)는 다음과 같이 정의된다:

$$x_0 = x, \quad x_{k+1} = \sigma(W_k x_k + b_k) \quad (k = 0, \ldots, L-2), \quad z = W_{L-1} x_{L-1} + b_{L-1}$$

여기서 $W_k \in \mathbb{R}^{d_{k+1} \times d_k}$, 전체 파라미터 수는 $\sum_{k=0}^{L-1} d_{k+1}(d_k + 1)$이다.

```
입력          은닉층 1       은닉층 2       출력층
x ∈ R^d₀ → σ(W₀x) ∈ R^d₁ → σ(W₁x₁) ∈ R^d₂ → W₂x₂ ∈ R^d₃
```

---

## 11.7 깊이의 중요성과 귀납적 편향

### 11.7.1 깊은 네트워크의 효율성

**명제 11.1 (비형식적, Telgarsky 2016).** 특정 함수 클래스에 대해 깊이 $L$의 네트워크는 다항식 크기로 표현 가능하지만, 깊이 $O(1)$의 네트워크는 지수적 너비가 필요하다.

ImageNet 분류 오류율의 역사가 이를 실증적으로 보여준다:

| 연도 | 모델 | 깊이 | Top-5 오류율 |
|------|------|------|------------|
| 2010 | Shallow | 얕음 | 28.2% |
| 2012 | AlexNet | 8 | 16.4% |
| 2014 | VGG | 19 | 7.3% |
| 2014 | GoogLeNet | 22 | 6.7% |
| 2015 | ResNet | 152 | 3.57% |

### 11.7.2 귀납적 편향

**정의 11.7 (귀납적 편향, Inductive Bias).** 모델이 새로운 입력에 일반화하기 위해 사전에 내재한 가정의 집합이다.

| 아키텍처 | 귀납적 편향 |
|----------|-----------|
| MLP (Fully Connected) | 제한 적음 |
| CNN | 지역성(locality), 이동 불변성(stationarity) |
| RNN | 시간적 파라미터 공유 |
| GNN | 순열 불변성(permutation invariance) |

**No Free Lunch Theorem**: 모든 문제에 대해 최적으로 작동하는 단일 모델은 존재하지 않는다. 따라서 데이터의 구조에 맞는 귀납적 편향을 인코딩하는 아키텍처 선택이 중요하다.

### 11.7.3 Fisher의 선형 판별분석 (LDA)

LDA는 사영(projection) 후 클래스 분리를 최대화하는 방향을 찾는다:

$$\max_w S(w) = \frac{(w^\top(m_1 - m_2))^2}{w^\top(\Sigma_1 + \Sigma_2)w}$$

**정리 11.6.** 공유 공분산 $\Sigma_1 = \Sigma_2 = \Sigma$ 가정 하에서 최적 사영 방향은:

$$w^* \propto \Sigma^{-1}(m_1 - m_2)$$

LDA는 각 클래스가 공유 공분산을 가진 가우시안이라는 가정 하에서 Bayes-optimal 분류기와 동치이다. SVM과의 핵심 차이는: LDA는 전체 데이터의 통계량(평균, 공분산)을 사용하지만, SVM은 경계 근처의 서포트 벡터만 사용한다.

---

## 11.8 흔한 오해와 주의점

| 오해 | 올바른 이해 |
|------|-----------|
| "SVM은 항상 선형 분류기이다" | 커널 트릭으로 원래 입력 공간에서 비선형 결정 경계를 만든다 |
| "활성화 없이 층을 쌓으면 더 강력해진다" | $W_L \cdots W_0 = W'$: 선형 합성은 여전히 선형 |
| "UAT가 있으니 2층이면 충분하다" | 존재성만 보장하며, 효율적 구현/최적화/일반화는 보장하지 않음 |
| "ReLU는 $x=0$에서 미분 불가능하므로 문제" | 확률적으로 $x=0$은 거의 발생하지 않으며 subgradient 사용 가능. 진짜 문제는 Dead ReLU |
| "SVM Dual은 Primal과 다른 해를 준다" | 볼록 최적화에서 강한 쌍대성이 성립하여 최적값 동일 |

---

## 11.9 핵심 요약

**모델 진화의 핵심 흐름**:

```
선형 모델 ──[XOR 한계]──> SVM+커널 ──[커널 설계 한계]──> 학습 가능한 특징
    ──[비선형 활성화]──> 2층 NN ──[효율성 한계]──> 깊은 NN ──[귀납적 편향]──> CNN/RNN/Transformer
```

**기억해야 할 핵심 수식 3가지**:

1. **SVM Dual**: $\max_\alpha -\frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i, x_j) + \sum_i \alpha_i$
2. **2층 NN**: $f(x;\theta) = w^\top\sigma(W_0 x)$
3. **MLP 순전파**: $x_{k+1} = \sigma(W_k x_k)$, 출력 $z = W_{L-1}x_{L-1}$

**참고문헌**: Rosenblatt (1957), Minsky & Papert (1969), Vapnik (1995), Cybenko (1989), Hornik et al. (1989), Telgarsky (2016), Sutton (2019, "The Bitter Lesson")

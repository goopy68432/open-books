---
title: "06. 최적화와 제약 최적화 (Optimization & Constrained Optimization)"
slug: 06-optimization-constrained
order: 6
---

# 06. 최적화와 제약 최적화 (Optimization & Constrained Optimization)

## 1. 동기부여 및 개요

딥러닝은 본질적으로 **손실 함수를 최소화하는 최적화 문제**이다. 신경망의 학습이란 파라미터 공간에서 손실을 줄이는 방향으로 이동하는 과정이며, 이를 수학적으로 정당화하려면 최적화 이론이 필요하다.

이 장에서는 다음 질문들에 답한다:
- 함수의 최솟값은 어디에 존재하는가? (임계점, 극값 판정)
- 미분 불가능한 함수는 어떻게 다루는가? (열미분/Subgradient)
- 제약 조건이 있을 때 최적화를 어떻게 수행하는가? (라그랑주 승수법, KKT 조건)
- Softmax 함수는 어떤 원리에서 유도되는가? (최대 엔트로피 원리)
- 함수 자체를 최적화하려면 어떻게 하는가? (변분법)

> **선수 지식**: 다변수 미적분(편미분, 그래디언트), 선형대수(행렬, 고유값), 기초 확률론

---

## 2. 임계점과 극값 판정

### 2.1 정의와 기본 개념

**Definition 2.1 (임계점, Critical Point).**
$f: \mathbb{R}^n \to \mathbb{R}$이 미분 가능할 때, $\nabla f(x_0) = 0$을 만족하는 점 $x_0$를 **임계점(critical point)** 또는 **정류점(stationary point)**이라 한다.

**Theorem 2.2 (페르마 정리, Fermat's Theorem).**
$f$가 개구간 $(a, b)$에서 미분 가능하고 $x_0$에서 극값을 가지면, $f'(x_0) = 0$이다.

이 정리의 핵심은 **필요조건**이라는 점이다. 역은 성립하지 않는다:

$$x_0 \text{가 극값} \implies \nabla f(x_0) = 0 \quad (\text{역은 불성립})$$

**반례**: $f(x) = x^3$에서 $x = 0$은 임계점이지만 극값이 아니다 (변곡점).

### 2.2 극값의 존재 위치

**Theorem 2.3.**
함수 $f$의 정의역 위에서 극값은 오직 다음 세 위치에서만 존재한다:
1. **경계(boundary)**
2. **미분 불가능점**
3. **임계점** ($\nabla f = 0$)

### 2.3 이계 도함수 판정법 (Second Derivative Test)

1변수의 경우, $f'(x_0) = 0$일 때:
- $f''(x_0) > 0$: **극소(local minimum)**
- $f''(x_0) < 0$: **극대(local maximum)**
- $f''(x_0) = 0$: 판정 불가

다변수에서는 **Hessian 행렬** $H = \nabla^2 f(x_0)$의 고유값으로 판정한다:

| Hessian 조건 | 판정 결과 |
|:---:|:---:|
| 모든 고유값 $> 0$ (양정치) | 극소 |
| 모든 고유값 $< 0$ (음정치) | 극대 |
| 부호 혼합 (부정치) | **안장점(saddle point)** |

### 2.4 딥러닝에서의 적용

고차원 손실 함수의 landscape에서는 **안장점이 극소점보다 훨씬 많다**. $n$차원에서 Hessian의 모든 고유값이 양수일 확률은 기하급수적으로 감소하기 때문이다 (Dauphin et al., 2014). SGD가 "나쁜 극소"에 갇히는 것보다 안장점 근처에서 학습이 느려지는 것이 실제로 더 큰 문제이며, momentum이나 Adam이 이를 완화한다.

---

## 3. 열미분 (Subgradient)

### 3.1 정의

**Definition 3.1 (Subgradient).**
$f: A \subset \mathbb{R}^n \to \mathbb{R}$에 대해, 점 $x$에서의 **열미분 집합(subdifferential)**은:

$$\partial f(x) := \{g \in \mathbb{R}^n : f(z) \geq f(x) + g^\top(z - x), \; \forall z \in A\}$$

기하학적으로, $g$는 점 $x$에서 함수 그래프 아래에 놓이는 **지지 초평면(supporting hyperplane)**의 기울기이다.

### 3.2 주요 성질

- **볼록 함수**에 대해 subdifferential은 항상 비공집합이다.
- $f$가 $x$에서 미분 가능하면 $\partial f(x) = \{\nabla f(x)\}$ (유일).

### 3.3 ReLU의 Subgradient

$$\partial \text{ReLU}(x) = \begin{cases} \{0\} & x < 0 \\ [0, 1] & x = 0 \\ \{1\} & x > 0 \end{cases}$$

실무에서 PyTorch/TensorFlow는 $x = 0$에서 $\text{ReLU}'(0) = 0$으로 설정한다. 이는 Clarke의 일반화 미분과 일치한다.

> **주의점**: Subgradient 방법은 비볼록 최적화에서 수렴 보장이 약하다. 이를 개선한 **proximal gradient method** (ISTA/FISTA)는 L1 정규화 문제에서 핵심적으로 사용된다. 이 내용은 13장 정규화에서 다시 등장한다.

---

## 4. 제약 최적화 (Constrained Optimization)

### 4.1 문제 정의

**Definition 4.1 (제약 최적화 문제의 표준형).**

$$\min_{x} f(x) \quad \text{subject to} \quad g_i(x) = 0 \; (i = 1, \ldots, m), \quad h_j(x) \leq 0 \; (j = 1, \ldots, k)$$

- $f(x)$: 목적 함수 (objective function)
- $g_i(x) = 0$: 등식 제약 (equality constraint)
- $h_j(x) \leq 0$: 부등식 제약 (inequality constraint)
- **허용 집합(feasible set)**: 모든 제약을 만족하는 $x$의 집합

### 4.2 핵심 아이디어

제약이 없으면 $\nabla f(x) = 0$을 풀면 되지만, 제약이 있으면 최적점이 $\nabla f(x) \neq 0$인 경계 위에 존재할 수 있다. 이를 해결하기 위해 제약 조건을 목적 함수에 "흡수"시키는 방법이 라그랑주 승수법이다.

### 4.3 딥러닝에서의 제약 최적화

| 기법 | 제약 형태 | 설명 |
|:---|:---|:---|
| Weight decay / L2 정규화 | $\|\theta\| \leq C$ | 라그랑주 형태: $\mathcal{L} + \lambda\|\theta\|^2$ |
| Spectral Normalization | $\sigma(W) \leq 1$ | 가중치 행렬의 스펙트럼 노름 제약 |
| Wasserstein GAN | $\|f\|_L \leq 1$ | Lipschitz 제약 |

---

## 5. 라그랑주 승수법 (Lagrange Multipliers)

### 5.1 기하학적 직관

등식 제약 $g(x) = 0$ 하에서 $f(x)$를 최소화할 때, 최적점에서는 **$\nabla f$와 $\nabla g$가 평행**하다:

$$\nabla f(x^*) = -\lambda \nabla g(x^*)$$

직관적으로, 제약 곡면 위에서 $f$의 등고선이 제약 곡선에 **접하는** 점이 최적점이다.

```
         ┌─ 등고선 f = c₃
    ╭────┤  등고선 f = c₂
    │    └─ 등고선 f = c₁
    │
    │   ∇f ↑  ← 최적점에서 ∇f ∥ ∇g
    │       ● ← 접점 = 최적점
    │      /
    └────/──── 제약 곡선 g(x) = 0
```

### 5.2 라그랑지안

**Definition 5.1 (라그랑지안, Lagrangian).**

$$\mathcal{L}(x, \lambda) = f(x) + \lambda \cdot g(x)$$

**Theorem 5.2.**
$x^*$가 등식 제약 $g(x) = 0$ 하에서 $f(x)$의 해라면, 유일한 $\lambda^*$가 존재하여:

$$\nabla_{x, \lambda} \mathcal{L}(x^*, \lambda^*) = 0$$

> **주의**: 이것은 **필요조건**이다. 라그랑지안의 정류점이 모두 원래 문제의 해는 아니다. 모든 정류점을 찾은 후 $f$ 값을 비교하여 최적해를 결정해야 한다.

### 5.3 예제

**예제 1**: $f(x,y) = x^2 + y^2$을 $g(x,y) = x + y - 1 = 0$ 하에서 최소화

$$\mathcal{L} = x^2 + y^2 + \lambda(x + y - 1)$$

$$\nabla \mathcal{L} = 0 \;\Leftrightarrow\; \begin{cases} 2x + \lambda = 0 \\ 2y + \lambda = 0 \\ x + y - 1 = 0 \end{cases} \;\Rightarrow\; (x, y, \lambda) = \left(\frac{1}{2}, \frac{1}{2}, -1\right)$$

**예제 2**: $f(x,y) = x + y$를 $g(x,y) = x^2 + y^2 - 1 = 0$ (단위원) 위에서 최대/최소화

풀면 두 정류점: $\left(\frac{\sqrt{2}}{2}, \frac{\sqrt{2}}{2}\right)$ (최대), $\left(-\frac{\sqrt{2}}{2}, -\frac{\sqrt{2}}{2}\right)$ (최소)

### 5.4 라그랑주 승수의 해석

$\lambda^*$는 **한계 비용(shadow price)**으로 해석된다: 제약을 $g(x) = c$로 약간 완화했을 때, $\frac{\partial f^*}{\partial c} = -\lambda^*$이다. 딥러닝에서 정규화 강도 $\lambda$를 조절하면 성능이 어떻게 변하는지와 직결되는 개념이다.

---

## 6. KKT 조건 (Karush-Kuhn-Tucker Conditions)

### 6.1 문제 설정

부등식 제약이 포함된 일반 문제:

$$\min_x f(x) \quad \text{s.t.} \quad g(x) = 0, \; h(x) \leq 0$$

라그랑지안:

$$\mathcal{L}(x, \lambda, \mu) = f(x) + \lambda \cdot g(x) + \mu \cdot h(x)$$

### 6.2 KKT 조건

**Theorem 6.1 (KKT 필요조건).**
$x^*$가 해이면, $\exists \lambda^*, \mu^*$ s.t.:

| 조건 | 수식 | 의미 |
|:---|:---|:---|
| **정류성** (Stationarity) | $\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = 0$ | 라그랑지안의 기울기가 0 |
| **원시 실행가능성** (Primal Feasibility) | $g(x^*) = 0,\; h(x^*) \leq 0$ | 제약 조건 만족 |
| **쌍대 실행가능성** (Dual Feasibility) | $\mu^* \geq 0$ | 부등식 제약의 승수가 비음수 |
| **상보 이완성** (Complementary Slackness) | $\mu^* \cdot h(x^*) = 0$ | 비활성 제약의 승수는 0 |

### 6.3 상보 이완성의 의미

- $h_j(x^*) < 0$ (비활성 제약): $\mu_j = 0$ --- 이 제약은 최적해에 영향 없음
- $\mu_j > 0$ (양의 승수): $h_j(x^*) = 0$ --- 이 제약은 등식으로 활성화됨

### 6.4 볼록성과 충분성

KKT 조건은 일반적으로 **필요조건**이다. 그러나 $f$가 볼록이고, $g_i$가 아핀이며, $h_j$가 볼록이면 KKT는 **필요충분조건**이 된다 (Slater 조건 하에서 strong duality 성립).

### 6.5 딥러닝에서의 KKT

- **SVM**: $\min \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$에서 상보 이완성에 의해 서포트 벡터만 $\alpha_i > 0$
- **Projected Gradient Descent**: 매 스텝 feasible set으로 프로젝션하는 것은 KKT를 반복 적용하는 과정

---

## 7. 최대 엔트로피 원리 --- 이산 (Maximum Entropy, Discrete)

### 7.1 동기

$n$개의 사건에 대해 확률분포를 정해야 하는데, 아는 정보가 $\sum_i p_i = 1$뿐이라면? 가장 **편향 없는** 선택은 **엔트로피를 최대화**하는 것이다.

### 7.2 기본 MaxEnt: 균등분포의 유도

$$\max_p H(p) = -\sum_{i=1}^n p_i \log p_i \quad \text{s.t.} \quad \sum_i p_i = 1, \; p_i \geq 0$$

라그랑지안을 세우고 정류점 조건을 풀면:

$$-\log p_i - 1 + \lambda = 0 \quad \Rightarrow \quad p_i = e^{\lambda - 1} = \frac{1}{n}$$

결과: **균등분포(uniform distribution)**

### 7.3 에너지 제약이 있는 MaxEnt: Softmax의 유도

에너지 값 $z_i$와 온도 매개변수 $\tau$가 주어진 경우:

$$\max_p \sum_i p_i z_i + \tau H(p) \quad \text{s.t.} \quad \sum_i p_i = 1$$

정류점 조건을 풀면:

$$\boxed{p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)} = \text{softmax}(z/\tau)_i}$$

이것이 **온도 매개변수를 가진 softmax 함수**이다.

### 7.4 온도(Temperature)의 효과

| $\tau$ | 분포 | 딥러닝 응용 |
|:---:|:---|:---|
| $\tau \to 0$ | argmax (one-hot) | 가장 확신 있는 선택 |
| $\tau = 1$ | 표준 softmax | 일반적인 분류 |
| $\tau \to \infty$ | 균등분포 | 최대 불확실성 |

이는 LLM의 텍스트 생성 전략, 강화학습의 exploration-exploitation, Knowledge Distillation (Hinton et al., 2015)에서 핵심적으로 사용된다.

> **연결**: Softmax = MaxEnt의 이산 해 = 통계역학의 Gibbs/Boltzmann 분포 = 지수족 분포의 자연스러운 귀결. 이 개념은 09장 정보이론에서 교차 엔트로피와 연결된다.

---

## 8. 변분법 (Calculus of Variations)

### 8.1 동기

지금까지의 최적화는 "최적의 **숫자(벡터)**"를 찾았다. 변분법은 "최적의 **함수**"를 찾는다.

**Definition 8.1 (범함수, Functional).**
함수를 입력받아 실수를 출력하는 대응: $J[y] = \int_a^b L(x, y(x), y'(x))\,dx$

### 8.2 Euler-Lagrange 방정식

범함수 $J[y]$를 최소화하는 함수 $y$는 다음 미분방정식을 만족한다:

$$\boxed{\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0}$$

**유도 (스케치)**: $y$를 $y + \epsilon \delta y$로 변분하고 $\delta J = 0$ 조건에 부분적분을 적용하면, 경계 조건 $\delta y(a) = \delta y(b) = 0$ 하에서 위 방정식이 도출된다.

### 8.3 대표 예시

| 문제 | $L$ | 해 |
|:---|:---|:---|
| 최단 거리 | $\sqrt{1 + [y']^2}$ | 직선 |
| 최속 강하선 (Brachistochrone) | $\sqrt{(1 + [y']^2) / y}$ | 사이클로이드 |

### 8.4 딥러닝에서의 변분법

- **Variational Inference**: ELBO를 최대화하는 근사 사후분포 $q(\theta)$는 함수 공간의 최적화
- **Neural ODE**: 연속 시간 신경망의 학습은 Euler-Lagrange의 연속 시간 버전인 Pontryagin의 최대 원리를 사용
- **Score Matching / Diffusion Models**: score function $\nabla_x \log p(x)$의 추정은 범함수 최적화
- **Physics-Informed Neural Networks (PINNs)**: 물리 법칙(PDE)을 Euler-Lagrange에서 유도

---

## 9. 최대 엔트로피 원리 --- 연속 (MaxEnt Continuous)

### 9.1 연속 엔트로피와 변분법의 결합

연속 확률분포 $p(z)$에 대해:

$$\max_p H(p) = -\int p(z) \log p(z)\,dz \quad \text{s.t.} \quad \int p(z)\,dz = 1$$

변분법 + 라그랑주 승수법으로 풀면, $[a,b]$ 위에서 해는 **균등분포**이다.

### 9.2 제약에 따른 MaxEnt 분포

| 제약 조건 | 정의역 | MaxEnt 분포 |
|:---|:---|:---|
| $\int p = 1$ | $[a, b]$ | 균등분포 (Uniform) |
| $\int p = 1$, $E[x] = \mu$ | $[0, \infty)$ | 지수분포 (Exponential) |
| $\int p = 1$, $E[x] = \mu$, $\text{Var}(x) = \sigma^2$ | $\mathbb{R}$ | **정규분포 (Gaussian)** |

### 9.3 딥러닝과의 연결

- **VAE의 사전분포**: 잠재 공간에 $\mathcal{N}(0, I)$를 사용하는 것은 평균과 분산만 알 때의 MaxEnt 선택
- **Energy-Based Models**: $p(x) \propto e^{-E(x)/T}$는 에너지 기댓값 제약 하의 MaxEnt이며, Boltzmann Machine의 이론적 기반

---

## 10. 흔한 오해와 주의점

| 오해 | 실제 | 교정 |
|:---|:---|:---|
| $f'(x)=0$이면 반드시 극값 | 임계점이 극값은 아닐 수 있다 ($x^3$의 $x=0$) | 이계 도함수 판정 또는 고차 도함수 확인 |
| 라그랑주 승수법이 최솟값을 직접 찾아줌 | **정류점**만 찾아주며, 최소/최대 구분은 별도 비교 필요 | 모든 정류점의 $f$ 값을 비교 |
| KKT 조건은 항상 충분조건 | 일반적으로 필요조건; 볼록성 필요 시 충분 | 볼록성 확인: $f$ 볼록, $g_i$ 아핀, $h_j$ 볼록 |
| 엔트로피 최대화 = 항상 균등분포 | 추가 제약(평균, 분산 등)이 있으면 다른 분포 | 제약 조건에 따라 MaxEnt 해가 달라짐 |
| Softmax는 단순한 정규화 함수 | MaxEnt 원리에서 유도된 최대 엔트로피 분포 | 이론적 기반 = Gibbs/Boltzmann 분포 |
| 변분법은 딥러닝과 무관 | VAE, Neural ODE, Diffusion Model의 핵심 | Euler-Lagrange와 최적 제어의 연결 |

---

## 11. 핵심 요약

```
┌─────────────────────────────────────────────────────────┐
│              최적화 & 제약 최적화 핵심 정리               │
├─────────────────────────────────────────────────────────┤
│ 1. 임계점: ∇f = 0 (필요조건, 충분조건 아님)              │
│ 2. 극값 판정: Hessian 고유값 부호로 결정                  │
│ 3. Subgradient: 비미분 볼록함수의 gradient 일반화         │
│ 4. 라그랑주 승수법: L = f + λg, 기하학적으로 ∇f ∥ ∇g    │
│ 5. KKT: 정류성 + 원시실행가능 + 쌍대실행가능 + 상보이완   │
│ 6. MaxEnt(이산): 에너지 제약 → softmax 유도              │
│ 7. 변분법: Euler-Lagrange 방정식 (함수 공간 최적화)      │
│ 8. MaxEnt(연속): 제약별 균등/지수/정규분포 유도           │
└─────────────────────────────────────────────────────────┘
```

**한 문장 요약**: 딥러닝의 학습 과정은 임계점 탐색(SGD), 제약 흡수(정규화 = 라그랑주 쌍대), softmax 유도(MaxEnt), 함수 공간 최적화(변분추론)로 이루어지며, 이 모든 것이 최적화 이론의 직접적인 응용이다.

---

## 참고 문헌

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 4: Numerical Computation.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*.
- Dauphin, Y. et al. (2014). Identifying and attacking the saddle point problem in high-dimensional non-convex optimization.
- Hinton, G. et al. (2015). Distilling the Knowledge in a Neural Network.

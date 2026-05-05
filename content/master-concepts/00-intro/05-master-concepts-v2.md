---
title: "딥러닝 이론 마스터 컨셉 가이드 v2"
slug: master-concepts-v2
order: 5
---

# 딥러닝 이론 마스터 컨셉 가이드 v2

> **Version 2** — 7주차 강의 스크립트(`수업_스크립트/딥러닝이론-1~8주차.md`) 분석을 통해 **교수가 직접 강조한 메시지**를 반영하여 **중요도를 재가중치**한 버전.
>
> **출처:** 한양대 이성윤 교수 「딥러닝」 강의 (742장 슬라이드 + 7주차 강의 스크립트)
>
> **v1 대비 변경점:** 강의 인용구 추가, 교수 강조 항목 ★ 표시, 교수의 직관 설명 보존, 채점 철학 명시.

---

## 0. 강의의 메타 메시지 (교수가 직접 한 말)

### 0.1 ★★★★★ "Deduction vs Induction" — 1주차 첫 강의 핵심
> *"두 개의 차이를 이해하는 것이 중간고사 전까지 목표 중 굉장히 중요합니다. AI 역사에서는 인덕션이 이겼습니다."*

| | Deduction (연역) | Induction (귀납) |
|---|---|---|
| 키워드 | Logic, Theorem, Rule, Symbolic AI | Data, Pattern, Statistical AI, ML |
| 흐름 | 정리(Theory) → 가설(Hypothesis) → 검증(Observation) | 관측(Observation) → 가설 → (검증) |
| 예시 | "수학적 귀납법" (사실은 deduction!) | n²+n+41 소수 추정 (40에서 깨짐, Black Swan) |
| AI 역사 | Expert System (옛날) | Neural Network (현재) |
| 문제점 | 너무 엄격, 발전 더딤 | "성급한 일반화" 가능 |

**시험 답안 강조:**
> *"문제를 인덕션으로 풀면 안 됩니다. 논리적으로 해야 됩니다."*

### 0.2 ★★★★★ "It Takes Two to Tango" — 1주차 강의 목표
강의 전체의 목표를 한 그림으로:
- **Maximum Likelihood (MLE)** — 데이터로부터 학습
- **Prior (사전 지식)** — 사람의 지식·인덕티브 바이어스

> *"이 두 가지를 이해하는 게 이 수업의 목표입니다. 이 둘을 이해하는 게 끝입니다."*

### 0.3 ★★★★★ "이론 vs 공학적 산출물 — 둘 다 중요"
> *"공학적 산출물은 거의 항상 이론적 이해보다 앞서왔다. 비행기도 비행 동력학이 나오기 전에 나왔습니다."*

- 알케미(연금술) → 화학(이론) → 더 좋은 결과 (Schmidhuber 비판)
- 증기기관(공학) → 열역학(이론) → 내연기관·컴퓨터 (LeCun 반박)
- 두 시각이 다 맞다. **딥러닝 레볼루션 = 증기기관 시기 같은 단계**.

### 0.4 ★★★★★ "답만 적으면 0점" — 채점 철학
> *"답만 적으면 점수가 없습니다. 답은 보지 않을 거고요. 어떤 과정을 설명하라 이런 식으로 되어 있을 겁니다. Explain how to obtain ─ 그냥 obtain 하라는 게 아니라 같이 하라는 뜻입니다."*

**채점 기준:**
- 답 자체 → 점수 없음
- 풀이 과정에서 i.i.d 가정·로그 이유·미분=0 이유 명시 → 점수 부여
- 논리적 서술 (수식만 나열 ≠ 답안)

### 0.5 ★★★★★ "행렬은 마음의 고향" — 이인석 교수 인용
> *"행렬은 마음의 고향이라는 뜻이 있고, 어떤 수학적 오브젝트를 만나더라도 행렬을 생각해야 한다."*
>
> *"우리가 아는 것은 행렬뿐이라고까지 강하게 얘기를 하십니다."*

→ **Linear Algebra와 Matrix가 모든 것의 출발**. 함수도 매트릭스, 미분도 매트릭스.

---

## 1. 수학적 기초

### 1.1 ★★★★★ 선형대수 (1주차 후반 + 2주차)

#### 핵심 메시지
> *"저희가 다룰 대상은 Vector이고, 다룬다는 것은 Function이다."*

- **벡터**: ordered tuple (∈ ℝⁿ)
- **함수**: Vector → Vector
- **선형 함수**: $f(\mathbf{u}+\mathbf{v}) = f(\mathbf{u})+f(\mathbf{v})$, $f(a\mathbf{v}) = af(\mathbf{v})$
- **행렬 = 선형변환**: 둘이 동치 (대학원 면접 단골)

#### Inner Product (내적) — 면접 단골
> *"대학원 면접 가면 항상 물어봅니다. 모르면 탈락."*

$$\langle \mathbf{v}, \mathbf{w} \rangle = \sum_i v_i w_i = \|\mathbf{v}\|\|\mathbf{w}\|\cos\theta$$

- 같은 방향 → cos=1 → 큰 값
- 직교 → cos=0
- 반대 → cos=-1 → 음수 (얼마나 similar한지의 측정)

#### Matrix Multiplication 시간 복잡도
> *"매트릭스 멀티플리케이션이 굉장히 무거운 연산입니다."*

| 양 | 복잡도 |
|---|---|
| 일반 (M×N × N×P) | O(MNP) |
| Square (N×N) | O(N³) |
| Strassen | O(N^2.8) |

#### Range, Null, Rank-Nullity ★★★★★
> *"Fundamental Theorem of Linear Algebra ─ 이것을 이해하면 많은 개념을 다 이해하게 됩니다."*

$$\boxed{\dim(\mathcal{R}(A)) + \dim(\mathcal{N}(A)) = n}$$

- $\mathcal{R}(A)$ = Image = "출력 쪽이 어디까지 가나"
- $\mathcal{N}(A)$ = Kernel = "입력 중 0으로 가는 것"
- **Rank** = dim(Range), **Nullity** = dim(Null)

#### Eigenvalue/Eigenvector ★★★★ (시험 1번 출제)
$A\mathbf{v} = \lambda\mathbf{v}$. 큰 λ가 더 중요.

**응용:**
- **이미지 압축**: 큰 eigenvalue 50개만으로 거의 원본 복원
- **Page Rank** (구글 알고리즘): 페이지 중요도 = 인접행렬의 최대 고유벡터
- **PCA**: 공분산 행렬의 최대 고유벡터 = 분산 최대 보존 방향

#### SVD (Singular Value Decomposition)
$$A = U\Sigma V^T$$
- 임의의 m×n 행렬에 대해
- 큰 특이값(σ₁ ≥ σ₂ ≥ …)부터 살리면 rank-k 근사
- **PCA가 SVD의 응용**

### 1.2 미적분과 Linear Approximation ★★★★★

#### "복잡한 함수를 1차로 근사한다"
> *"복잡한 함수가 너무 알기 어렵기 때문에 1차로 근사할 수밖에 없습니다. 미분을 한다는 것은 매트릭스를 얻는다는 거예요."*

**Newton's Method (루트7 계산 예시):**
$$x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)}$$

- f(x) = x²-7의 0을 찾고 싶음
- 1차 근사 → 0 찾기 → 반복
- **모든 최적화의 근본**

#### Newton's method = "L에 대해 2차 근사" (8주차)
> *"f가 L의 미분이라고 하면, L에 대해서는 2차 근사를 하는 거예요."*

L(θ) 최소화 ⇔ $\nabla L = 0$ 찾기 = "L에 대해 quadratic 근사 후 0 그래디언트 위치"

#### 가우스 적분 ★★★★ (시험 2번)
$$\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$$

극좌표 변환 증명. 모든 정규분포 모멘트의 기초.

#### Jacobian — 벡터를 벡터로 미분
$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Softmax Jacobian (시험 8번):**
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j), \quad J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

> *"Attention에서도 Softmax가 쓰이고, image classification에서도 Softmax가 쓰이는데, 그 함수가 다 미분이 됩니다. 이게 중요합니다."*

### 1.3 ★★★★★ 확률·통계

#### 핵심 분포: Bernoulli vs Gaussian
> *"이 두 개를 좀 포커스에서 봐주시길 바랍니다."*

**Bernoulli** — 가장 간단한 분포
> *"샘플 스페이스가 2 ─ 가능한 outcome이 두 개. θ 하나의 값으로 완전히 결정."*
$$p(y|\theta) = \theta^y(1-\theta)^{1-y}$$

**Gaussian — 3가지 핵심 (반복 강조!)**
> *"이게 왜 중요한지 다음 주에 이해하실 수 있습니다. 스포일러 ─ Exponential을 log로 치고 마이너스를 붙이면 안에 있는 것 사라집니다. 남은 게 뭐죠? 제곱이 남습니다. 이게 카우시안에서 가져와 보자."*

| 핵심 | 의미 |
|------|-----|
| **Exponential** | $e^{-(\cdot)}$ |
| **Minus** | 음수 부호 |
| **Square** | $(x-\mu)^2$ |

→ **-log를 취하면 제곱만 남음** → MSE 손실의 출처!

$$p(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**표준정규 모멘트 (외워라):** $E[X^{2n}] = (2n-1)!!$, 홀수=0
- $E[X^2]=1, E[X^4]=3, E[X^6]=15$

#### Bayesian vs Frequentist ★★★★★
> *"통계학자들 입장에서 정의하는 데부터 다르게 시작하니까 다 다른 거예요. 사이가 좋지 않은 두 진영입니다."*

| | Bayesian | Frequentist |
|---|---|---|
| 확률 의미 | Belief (믿음의 정도) | Frequency (상대 빈도) |
| 다루는 것 | Hypothesis $p(H)$ | Event $p(E)$ |
| θ | 확률변수 | 고정 미지수 |
| 정의 예 | "이 동전이 공정하다는 믿음" | "100만번 던지면 50만번" |

> *"베이지안적인 것을 처음 소개해드리면 뭔가 이질감이 들 겁니다. 같이 해보면서 익숙해지는 게 중요합니다."*

#### Independence
> *"독립의 정의는 곱이라는 거 강조를 드립니다. 이 곱이 오늘 또 나오게 될 거고요."*

$$P(E \cap F) = P(E)P(F) \quad \Leftrightarrow \quad p(y_1, \ldots, y_n|\theta) = \prod_i p(y_i|\theta)$$

#### Inclusion-Exclusion
$$P(E_1 \cup E_2) = P(E_1) + P(E_2) - P(E_1 \cap E_2)$$

#### Chebyshev / Hoeffding (3주차)
대수의 법칙·표본평균 수렴 속도 정량화.

---

## 2. ★★★★★ 통계적 추정 — 강의의 심장

### 2.1 베이즈 정리 (3주차 — "오늘이 가장 중요한 날")
> *"베이지안 Probability 자체는 가장 중요한 토픽을 다루고 있다고 생각하실 수 있는데, 가장 중요함에도 가장 간단한 걸로 시작합니다. 코인 토싱."*

$$p(H|E) = \frac{p(E|H)p(H)}{p(E)}$$

**용어 (반드시 외울 것):**
- **Prior** $p(H)$ ─ 데이터 보기 전 hypothesis에 대한 믿음 (파란색 곡선)
- **Posterior** $p(H|E)$ ─ 데이터 보고 난 후 믿음 (빨간색 곡선)
- **Likelihood** $p(E|H)$ ─ "라이클리우드"
- **Evidence** $p(E)$ ─ θ 무관 (MAP에서 무시)

#### Belief Update = Learning
> *"이 Belief Update를 하는 게 저희가 Learning이라고 하는 것과 연결됩니다."*

→ **베이지안 관점에서 학습 = 데이터로 prior를 posterior로 업데이트하는 과정**

### 2.2 ★★★★★ MLE — Maximum Likelihood Estimation

#### 7단계 체인 (모든 분포에 동일!)

```
   y_i ~ Distribution(θ) i.i.d
            ↓ (i.i.d → 곱)         "독립이 곱이다"
   L(θ) = ∏ p(y_i|θ)
            ↓ (log)                 "곱→합, argmax 보존, 수치 안정"
   ℓ(θ) = ∑ log p(y_i|θ)
            ↓ (× -1)                "최소화 표준"
   NLL(θ) = -ℓ(θ)
            ↓ (페르마)               "미분 가능 함수의 내부 극값"
   dℓ/dθ = 0
            ↓ (풀이)
   θ̂_MLE
            ↓ (2계 미분 검증)
   최댓값 확정
```

**베르누이 결과:** $\hat{\theta}_{\text{MLE}} = k/n$

> *"3번 던져서 3번 안면 나왔어요. MLE는 1이라고 답합니다. 이거에 대해서 어떻게 생각하시나요? 데이터에만 의존하고 있어요. 시행 횟수가 적을 때 안 좋습니다."*

### 2.3 ★★★★★ MAP — 데이터 + 지식의 균형

#### 핵심 통찰
> *"MLE = MAP under uniform prior. 즉, prior가 없을(uniform) 때 MAP가 MLE다. 거꾸로 이해하라."*

#### Prior 강도에 따른 변화
n=3, k=3 (3번 다 앞면)일 때 prior $p(\theta) \propto \theta^m(1-\theta)^m$:

| m | 의미 | 결과 |
|---|------|-----|
| 0 | uniform = 지식 없음 = MLE | $\hat{\theta} = 1$ (극단적!) |
| 1 | 약한 prior | $\hat{\theta} = 4/5 = 0.8$ |
| ∞ | 강한 prior (0.5에 디랙 델타) | $\hat{\theta} = 0.5$ (데이터 무시!) |

**교수의 정리:**
> *"MLE: 데이터에만 의존, knowledge 없음. Strong MAP: knowledge만 의존, 데이터 무시. 적절한 균형이 best."*

| 방법 | 데이터가 적을 때 | 데이터가 많을 때 |
|------|----------------|-----------------|
| MLE | 안 좋음 (overfitting) | 좋음 |
| Strong MAP | 좋음 (prior로 보완) | 안 좋음 (데이터 무시) |

### 2.4 ★★★★★ NLL = 손실 함수의 정체 (7주차)

#### 핵심 통찰
> *"Loss Function이 먼저가 아닙니다. NLL이 먼저예요. Squared Loss를 왜 쓰는지, Cross Entropy를 왜 쓰는지에 대한 답은 ─ 어떤 distribution을 hypothesis로 주느냐에 따라 결정됩니다."*

**ERM (Empirical Risk Minimization)** = NLL과 동치:
$$\hat{R}(h) = \frac{1}{n}\sum_i \ell(h(x_i), y_i) \equiv -\frac{1}{n}\sum_i \log p(y_i|h)$$

#### 분포 → 손실 매핑

| 분포 가정 | NLL → Loss | 어디 쓰이나 |
|---------|----------|-----------|
| **Gaussian noise** | NLL → MSE | 회귀 |
| **Bernoulli** | NLL → BCE | 이진 분류 |
| **Categorical** | NLL → CE | 다중 분류 |
| Laplace noise | NLL → MAE | (드물게) |

**가우시안 → MSE 유도 (강의에 직접):**
> *"Negative log를 취하면 가우시안의 3가지 핵심 중 'exponential'과 'minus'가 사라지고 'square'만 남습니다. 그게 MSE입니다."*

$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i(y_i-f_i)^2 \xrightarrow{\arg\min} \text{MSE}$$

### 2.5 MAP → 정규화 (Prior가 정규화의 베이지안 해석)

| Prior | -log prior | 정규화 | 강도 |
|-------|-----------|-------|------|
| Uniform | 상수 | 없음 (= MLE) | 약함 |
| Gauss $N(0, \tau^2 I)$ | $\|\mathbf{w}\|^2/(2\tau^2)$ | **L2** | 균형 |
| Laplace | $\|\mathbf{w}\|_1/b$ | **L1** | 희소성 |
| Dirac at 0.5 | -∞ except 0.5 | Strong MAP | 매우 강함 |

> *"Hypothesis Space를 제약하는 것이 prior knowledge를 주는 것이고, 이것이 Inductive Bias 입니다."*

### 2.6 KL Divergence와 정보이론

$$\text{KL}(p\|q) = E_p[\log(p/q)] \geq 0$$

**핵심 분해:**
$$\boxed{H(p, q) = H(p) + \text{KL}(p\|q)}$$

> *"진짜 분포 p가 고정 → H(p) 상수 → CE 최소화 ⇔ KL 최소화."*

---

## 3. 고전 ML — Inductive Bias의 시작

### 3.1 ★★★★★ Hypothesis Space 제약 = Inductive Bias (7주차)
> *"전체 함수 공간 F에서 좋은 함수를 찾는 건 굉장히 어렵습니다. 그래서 제약을 시키는 방법으로 이 방법을 쓰게 되는 거죠."*

전체 함수 → 부분집합 (parametric family) → 최적화

> *"Linear function만 생각하는 게 prior knowledge를 주는 것이고, 이건 굉장히 강한 prior입니다. '동전 던지기 예시에서의 Strong Prior' 같은 거예요."*

### 3.2 Function Space → Parameter Space 전환
> *"함수에서 뭔가를 찾는다는 게 어려우니까, 파라미터라는 개념을 도입해서 점으로 표현해서 찾는 거예요."*

예: 선형함수 $f(x) = ax + b$ → 점 $(a, b) \in \mathbb{R}^2$

→ **신경망 가중치 = 함수를 표현하는 파라미터**

### 3.3 Perceptron (Rosenblatt 1958)
$$f(\mathbf{x}) = \begin{cases} 1 & \mathbf{w}^T\mathbf{x} + b \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

> *"AI라는 표현이 처음 나왔을 때(1956)와 거의 같은 시기. 머신러닝의 시작."*

- **Decision Boundary**: $\mathbf{w}^T\mathbf{x} + b = 0$
- 두 영역으로 나누는 분류기

### 3.4 LDA vs QDA
- LDA: 공분산 동일 가정 → 선형 결정경계
- QDA: 클래스별 공분산 → 2차 결정경계 (page_350)

### 3.5 SVM과 Margin
**Hard-margin SVM:**
$$\min \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.}\; y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$

Margin = $\gamma = 1/\|\mathbf{w}\|$ — Lagrange multiplier 필요.

---

## 4. 최적화

### 4.1 Newton's Method = 모든 최적화의 근본
> *"이거(Newton's method)는 옛날 옛적부터 쓰던 방법입니다. 그래서 결국 다루는 대상은 Vector고, 다룬다는 거는 Function 알고리즘이고, 함수가 너무 복잡해서 그걸 이해하기 위해서는 Linear Approximation을 하는 것 ─ 그러니까 Calculus를 한다라고 말씀드립니다."*

### 4.2 Gradient Descent
$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla L(\theta^{(t)})$$

수렴 조건 (L-smooth + convex): $\eta < 1/L \Rightarrow O(1/t)$.

### 4.3 SGD, Momentum, Adam
미니배치, 잡음으로 local minima 탈출, 적응적 학습률.

### 4.4 ★ Lagrange Multiplier (page_120)
$$\nabla_\mathbf{x} \mathcal{L} = \nabla f - \lambda \nabla g = 0$$

PCA, SVM 유도의 핵심 도구.

### 4.5 PyTorch 표준 학습 루프
```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

---

## 5. 신경망 기초

### 5.1 Universal Approximation Theorem (UAT)
1개 은닉층 + 비선형 활성화 + 충분한 뉴런 → 임의 연속함수 근사.

**한계:** 존재성만 보장. 효율성·학습 가능성은 별개.

### 5.2 ★★★★★ 활성화 함수 (시험 단골)

| 함수 | 식 | 미분 |
|------|-----|-----|
| Sigmoid | $1/(1+e^{-x})$ | $\sigma(1-\sigma)$ |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $1-\tanh^2$ |
| ReLU | $\max(0,x)$ | $\mathbb{1}_{x>0}$ |
| GELU | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ |
| Softplus | $\log(1+e^x)$ | $\sigma(x)$ |

**Sigmoid 미분 유도:** $e^{-x} = (1+e^{-x})-1$ 분해 트릭.

### 5.3 Backpropagation 4식
1. **출력층:** $\delta^{(L)} = \nabla L \odot \sigma'(z^{(L)})$
2. **역전파:** $\delta^{(l)} = (W^{(l+1)})^T\delta^{(l+1)} \odot \sigma'(z^{(l)})$
3. **가중치 그래디언트:** $\partial L/\partial W^{(l)} = \delta^{(l)}(a^{(l-1)})^T$
4. **편향 그래디언트:** $\partial L/\partial b^{(l)} = \delta^{(l)}$

**Softmax+CE 그래디언트 = $p - y$** (놀라운 단순성).

### 5.4 ★★★★★ Softmax 자코비안 (시험 8번)

> *"Softmax는 Attention에서도 쓰이고, 분류 출력층에서도 쓰입니다. 미분이 된다 ─ 이게 중요합니다."*

#### 두 케이스 분리
- **i = j:** $\partial p_i/\partial z_i = p_i(1-p_i)$
- **i ≠ j:** $\partial p_i/\partial z_j = -p_i p_j$

#### 통합 (Kronecker Delta)
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

#### 행렬 형태
$$J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### 5.5 초기화
- **Xavier** (sigmoid/tanh): $\text{Var}(w) = 2/(n_{\text{in}}+n_{\text{out}})$
- **He** (ReLU): $\text{Var}(w) = 2/n_{\text{in}}$ (절반 죽임 보정)

### 5.6 Vanishing Gradient
$\sigma' < 1$이 곱셈으로 누적되어 깊은 망에서 0. 해결: ReLU, BN, ResNet.

### 5.7 Grandmother Cell (Jennifer Aniston 뉴런)
> *"한 셀이 한 개념을 표현한다는 가설. 분산 표현 논의의 출발점."*

---

## 6. 일반화 이론

### 6.1 ★★★★★ Bias-Variance Tradeoff
$$E[(y_0-\hat{f})^2] = \underbrace{(\bar{f}-f)^2}_{\text{Bias}^2} + \underbrace{E[(\bar{f}-\hat{f})^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

**유도 트릭:** $\bar{f}$ 더하고 빼기, 교차항 3개 모두 0.

### 6.2 Double Descent
> *"In statistics and ML textbooks, an estimate that fits every training example perfectly is often presented as overfitting. However, this is not true."*

over-parameterized 영역에서 test error가 다시 감소.

### 6.3 Sharpness 논쟁
> *"Sharpness는 generalization과 잘 상관되지 않는다."*

Flat vs Sharp minima 논쟁 — 단순 직관 못 믿음.

### 6.4 Regularization 종류
- **L2** = Gauss prior
- **L1** = Laplace prior (희소성)
- **Dropout**: 학습 시 뉴런 랜덤 제거
- **Batch Norm**: 미니배치 정규화

---

## 7. CNN

### 7.1 합성곱
$$(I*K)(i,j) = \sum_{m,n} I(i+m, j+n)K(m,n)$$

**Inductive Bias:** parameter sharing, translation equivariance.

### 7.2 Pooling, Stride, Padding
- Max Pooling → 공간 축소 + 작은 이동에 강건
- Receptive Field → 깊어질수록 넓어짐

### 7.3 응용
ImageNet 분류, YOLO 검출, U-Net 분할.

---

## 8. RNN / LSTM

### 8.1 RNN 기본
$$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

### 8.2 LSTM 게이트
$$f_t = \sigma(W_f[h_{t-1}, x_t]) \quad (\text{forget})$$
$$i_t = \sigma(W_i[h_{t-1}, x_t]) \quad (\text{input})$$
$$o_t = \sigma(W_o[h_{t-1}, x_t]) \quad (\text{output})$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

**해결:** Vanishing gradient (cell state는 곱이 아닌 덧셈 전파).

### 8.3 BPTT
시간 펼친 후 backprop.

---

## 9. Attention과 Transformer

### 9.1 Self-Attention
$$Q=XW_Q, K=XW_K, V=XW_V$$
$$\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$$

> *"Softmax가 미분된다는 게 Attention의 핵심입니다."*

### 9.2 Multi-Head + Masked + Transformer
- Multi-Head: $h$개 head를 concat
- Masked: 자동회귀 (j > i → -∞)
- Transformer 블록: PE → MHA → Add&Norm → FF → Add&Norm

---

## 10. 생성 모델

### 10.1 GAN (2014)
$$\min_G \max_D V(D,G) = E_x[\log D(x)] + E_z[\log(1-D(G(z)))]$$

균형: $D^* = p_d/(p_d+p_g)$.

### 10.2 VAE — ELBO
$$\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\|p(z))$$

**Reparameterization:** $z = \mu + \sigma \odot \epsilon$.

### 10.3 Diffusion
- Forward: data → noise (고정)
- Reverse: noise → data (학습)
- Noise prediction $\epsilon_\theta(z_t, t)$

---

## 11. 모던 응용

### 11.1 Transfer Learning vs Domain Adaptation
- Transfer: $\mathcal{X}_s \approx \mathcal{X}_t$, $\mathcal{Y}_s \neq \mathcal{Y}_t$
- Domain Adapt: $\mathcal{X}_s \neq \mathcal{X}_t$, $\mathcal{Y}_s = \mathcal{Y}_t$

### 11.2 CLIP — Contrastive Language-Image
$$L_{ij} = \cos(I_i, T_j), \quad \mathcal{L} = -\sum -\log\text{softmax}(L)$$

400M 이미지-텍스트 쌍 → zero-shot 분류.

### 11.3 LLM 동향
CoT, VLM, Scaling Laws, RLHF.

---

## 12. ★★★★★ 시험 답안 작성법 (교수의 채점 철학)

### 12.1 답안 첫 줄 (반드시!)
1. **모델 명시:** "$y_i \sim \text{Bern}(\theta)$ i.i.d, $i=1,\ldots,n$"
2. **가정 명시:** "i.i.d, $\theta \in (0,1)$, 미분 가능"

### 12.2 단계마다 "왜?" 한 줄
- **곱:** "i.i.d 가정에 의해 결합확률 = 곱"
- **로그:** "단조성으로 argmax 보존, 곱→합으로 미분 단순, underflow 방지"
- **음수:** "ML 표준은 손실 최소화"
- **미분=0:** "페르마 정리"
- **검증:** "2계 미분 < 0이므로 오목, 임계점이 최댓값"

### 12.3 정리 인용
- "Rank-Nullity 정리"
- "Spectral 정리"
- "베이즈 정리"
- "Jensen 부등식"
- "Hoeffding 부등식"

---

## 13. 강의의 8대 통합 메시지 (최종 정리)

1. **딥러닝 = 선형대수 + 미적분 + 확률의 조합**
2. **모든 손실 함수는 분포 가정에서 유도된다 (NLL이 통일 도구)**
3. **모든 정규화는 Prior에서 유도된다 (MAP 관점)**
4. **모든 학습은 i.i.d → 로그 → 미분=0 체인이다**
5. **Hypothesis Space 제약 = Inductive Bias = Prior Knowledge**
6. **Linear Approximation = 모든 최적화의 근본 (Newton's method)**
7. **Function Space → Parameter Space 전환이 신경망의 정체**
8. **MLE (데이터만) ↔ Strong MAP (지식만), 균형이 best**

---

## 14. 약어·기호 사전

[v1과 동일, MASTER-CONCEPTS.md §13 참조]

---

## 15. 학습 자료 매핑 (final-fire/와의 대응)

| 본 가이드 § | final-fire 위치 |
|----------|---------------|
| §1.1 선형대수 | `00-prerequisites/06-07/`, `01-eigen/` |
| §1.2 미적분 | `00-prerequisites/03-05/` |
| §1.3 확률 | `00-prerequisites/09-10/`, `02-gaussian/`, `03-uniform/` |
| §2.2 MLE | `04-mle-bernoulli/` |
| §2.3 MAP | `05-07-map-*/` |
| §2.4 NLL | `09-killer-chains/07-nll-master.md` |
| §2.5 정규화 | `09-killer-chains/06-map-to-l2.md` |
| §5.4 Softmax | `08-softmax/`, `10-ten-proofs/04` |
| §6 일반화 | `11-extra-topics/03-bias-variance.md` |
| §10 생성모델 | (final-fire 미수록 — 본 가이드 보충) |

---

**작성일:** 2026-04-26 (v2)
**기반:** 742장 슬라이드 + 1,2,3,4,6,7,8주차 강의 스크립트 직접 분석
**다음:** [`LEARNING-MAP_v2.md`](./LEARNING-MAP_v2.md) — 학습 순서 + Mermaid 관계도 v2

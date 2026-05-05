---
title: "Deep Learning Theory — Midterm Mock Exam (Part 3: Q21–Q30)"
slug: midterm-mock-exam-part3
order: 4
---

# Deep Learning Theory — Midterm Mock Exam (Part 3: Q21–Q30)

> **범위**: MAP Estimation, Softmax Jacobian, Norms & Regularization, Computation Graphs, Pseudo-inverse, Bias-Variance, Hoeffding's Inequality, Activation Functions, Batch Normalization, Comprehensive Forward/Backward Pass
> **출제 스타일**: 증명·유도 중심, 논리적 과정 > 정답
> **풀이 3단계**: 🟢 중학생 · 🟡 고등학생 · 🔴 대학생

---

## 문제 21 (MAP Estimation)

### [EN] Problem Statement

Given a prior $p(\theta) = \mathcal{N}(0, \tau^2)$ and a likelihood $p(D|\theta) = \prod_{i=1}^{n} \mathcal{N}(y_i;\, \theta x_i,\, \sigma^2)$:

**(a)** Write the MAP objective: show that maximizing the posterior is equivalent to maximizing $\log p(D|\theta) + \log p(\theta) + \text{const}$.

**(b)** Show that MAP estimation is equivalent to minimizing $\text{MSE} + \lambda\|\theta\|^2$ (MSE loss with L2 regularization).

**(c)** Identify what $\lambda$ equals in terms of $\sigma^2$ and $\tau^2$.

**(d)** Explain conceptually why the prior acts as regularization.

### [KR] 문제

사전 분포 $p(\theta) = \mathcal{N}(0, \tau^2)$, 가능도 $p(D|\theta) = \prod_{i=1}^{n} \mathcal{N}(y_i;\, \theta x_i,\, \sigma^2)$가 주어졌을 때:

**(a)** MAP 목적함수를 작성하라: 사후 분포를 최대화하는 것이 $\log p(D|\theta) + \log p(\theta) + \text{const}$를 최대화하는 것과 동치임을 보여라.

**(b)** MAP 추정이 $\text{MSE} + \lambda\|\theta\|^2$를 최소화하는 것과 동치임을 보여라.

**(c)** $\lambda$가 $\sigma^2$, $\tau^2$로 어떻게 표현되는지 구하라.

**(d)** 사전 분포가 왜 정규화(regularization) 역할을 하는지 개념적으로 설명하라.

### 출제 의도

MAP 추정은 **베이지안 관점과 최적화 관점을 연결**하는 핵심 개념이다. 이 문제는 "L2 정규화가 왜 등장하는가"를 확률론적으로 유도할 수 있는지 확인한다. 딥러닝에서 weight decay의 이론적 근거를 이해하는 데 필수적이다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**배경 지식부터 설명할게요.**

**확률이란?** 어떤 일이 일어날 가능성을 숫자로 나타낸 것이에요. 0이면 절대 안 일어나고, 1이면 반드시 일어나요.

**정규분포(가우시안 분포)란?** 종 모양의 곡선이에요. 평균값 근처에 데이터가 많고, 멀어질수록 적어요. 키를 예로 들면, 평균 키(170cm) 근처에 사람이 가장 많고, 아주 크거나 아주 작은 사람은 적어요.

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

여기서:
- $\mu$ = 평균 (종의 꼭대기 위치)
- $\sigma^2$ = 분산 (종이 얼마나 넓게 퍼지는지)
- $\exp(\cdot)$ = $e$의 거듭제곱 ($e \approx 2.718$)

**모델이란?** 입력 $x$를 받아서 출력 $y$를 예측하는 규칙이에요. 여기서는 가장 간단한 규칙: $y = \theta \cdot x$ (기울기가 $\theta$인 직선)를 사용해요.

**$\theta$란?** 우리가 찾고 싶은 "가장 좋은 기울기"예요.

---

**(a) MAP 목적함수 작성**

**베이즈 정리**부터 시작해요. "데이터 $D$를 보고 나서 $\theta$가 얼마일 가능성이 높은가?"를 구하는 공식이에요:

$$p(\theta|D) = \frac{p(D|\theta) \cdot p(\theta)}{p(D)}$$

- $p(\theta|D)$: **사후 분포** — 데이터를 본 후 $\theta$에 대한 믿음
- $p(D|\theta)$: **가능도** — $\theta$가 이 값일 때 데이터가 나올 확률
- $p(\theta)$: **사전 분포** — 데이터를 보기 전 $\theta$에 대한 믿음
- $p(D)$: **증거** — 데이터 자체의 확률 ($\theta$와 무관한 상수!)

MAP는 $p(\theta|D)$를 최대로 만드는 $\theta$를 찾는 것이에요. $p(D)$는 $\theta$에 의존하지 않으므로:

$$\theta_{\text{MAP}} = \arg\max_\theta \; p(D|\theta) \cdot p(\theta)$$

양변에 $\log$를 취하면 (log는 단조증가이므로 최대점이 변하지 않아요):

$$\theta_{\text{MAP}} = \arg\max_\theta \; \left[\log p(D|\theta) + \log p(\theta)\right]$$

$\log p(D)$는 $\theta$와 무관한 상수이므로 $\text{const}$로 표기할 수 있어요. 따라서:

$$\theta_{\text{MAP}} = \arg\max_\theta \; \left[\log p(D|\theta) + \log p(\theta) + \text{const}\right] \quad \checkmark$$

---

**(b) MSE + L2 정규화와의 동치**

**Step 1: 가능도의 로그 계산**

가능도는 각 데이터 포인트 $(x_i, y_i)$에 대해:

$$p(D|\theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \theta x_i)^2}{2\sigma^2}\right)$$

이것은 "실제 값 $y_i$와 예측 값 $\theta x_i$의 차이가 작을수록 확률이 높다"는 뜻이에요.

로그를 취하면 ($\prod$가 $\sum$이 되어요):

$$\log p(D|\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \theta x_i)^2$$

첫 번째 항은 $\theta$와 무관한 상수예요. 중요한 부분은:

$$\log p(D|\theta) = -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \theta x_i)^2 + \text{const}_1$$

**Step 2: 사전 분포의 로그 계산**

$$p(\theta) = \frac{1}{\sqrt{2\pi\tau^2}} \exp\left(-\frac{\theta^2}{2\tau^2}\right)$$

로그를 취하면:

$$\log p(\theta) = -\frac{\theta^2}{2\tau^2} + \text{const}_2$$

**Step 3: MAP 목적함수 합치기**

$$\log p(D|\theta) + \log p(\theta) = -\frac{1}{2\sigma^2}\sum_{i}(y_i - \theta x_i)^2 - \frac{\theta^2}{2\tau^2} + \text{const}$$

최대화 → 부호를 뒤집으면 **최소화**:

$$\theta_{\text{MAP}} = \arg\min_\theta \left[\frac{1}{2\sigma^2}\sum_{i}(y_i - \theta x_i)^2 + \frac{\theta^2}{2\tau^2}\right]$$

$\frac{1}{2\sigma^2}$를 앞으로 빼면:

$$= \arg\min_\theta \left[\sum_{i}(y_i - \theta x_i)^2 + \frac{\sigma^2}{\tau^2}\theta^2\right]$$

$\sum_{i}(y_i - \theta x_i)^2$는 바로 **MSE**(의 $n$배)이고, $\frac{\sigma^2}{\tau^2}\theta^2$는 **L2 정규화**예요!

따라서: $\text{MSE} + \lambda\|\theta\|^2$ 형태와 동치 $\checkmark$

---

**(c) $\lambda$ 구하기**

위에서 바로 보이듯:

$$\boxed{\lambda = \frac{\sigma^2}{\tau^2}}$$

- $\sigma^2$: 데이터의 노이즈 크기
- $\tau^2$: 사전 분포의 분산 (θ에 대한 사전 믿음의 폭)

**직관**: 데이터 노이즈($\sigma^2$)가 크면 → $\lambda$가 크다 → 정규화를 세게 건다 (데이터를 덜 믿으니까 사전 믿음에 더 의존).
사전 분포가 넓으면($\tau^2$가 크면) → $\lambda$가 작다 → 정규화를 약하게 건다 (θ가 크든 작든 상관없다고 미리 믿었으니까).

---

**(d) 사전 분포가 정규화 역할을 하는 이유**

비유로 설명할게요.

시험공부를 할 때, 선생님이 "범위는 1장부터 10장까지"라고 했어요(= 사전 분포). 근데 친구가 "시험에 11장도 나온대!"라고 했어요(= 데이터).

- **사전 분포 없이** 친구 말만 믿으면 → 11장까지 공부 (과적합 위험)
- **사전 분포가 있으면** → "선생님이 10장까지라 했으니, 11장은 좀 의심스럽다" → 극단적 행동을 자제

수학적으로: $p(\theta) = \mathcal{N}(0, \tau^2)$는 "$\theta$는 0 근처에 있을 가능성이 높다"는 믿음이에요. 이것이 $\theta$가 너무 커지는 것을 막아줘요 = **정규화**!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 베이즈 정리로부터:

$$p(\theta|D) \propto p(D|\theta)p(\theta)$$

양변에 $\log$를 취하면:

$$\log p(\theta|D) = \log p(D|\theta) + \log p(\theta) + \text{const}$$

여기서 const $= -\log p(D)$는 $\theta$에 무관. MAP는 이를 $\theta$에 대해 최대화.

**(b)**

$$\log p(D|\theta) = -\frac{1}{2\sigma^2}\sum_i(y_i - \theta x_i)^2 + C_1$$

$$\log p(\theta) = -\frac{\theta^2}{2\tau^2} + C_2$$

합산 후 부호 반전하여 최소화 문제로 변환:

$$\min_\theta \left[\sum_i(y_i - \theta x_i)^2 + \frac{\sigma^2}{\tau^2}\theta^2\right]$$

이는 MSE + $\lambda\|\theta\|^2$ 꼴.

**(c)** $\lambda = \sigma^2/\tau^2$

**(d)** Gaussian prior는 $\theta = 0$을 중심으로 한 "소프트 제약"이다. $\tau^2$가 작을수록 $\theta$를 0 가까이 강하게 끌어당기므로, 모델 복잡도를 제한하는 정규화 효과를 낸다. 이는 가능도만 최대화할 때 발생하는 과적합을 방지한다.

#### 🔴 Level 3: 대학생 눈높이

**(a)** $\theta_{\text{MAP}} = \arg\max_\theta \log p(\theta|D) = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]$. Bayes' rule에서 evidence $p(D)$는 $\theta$-independent constant.

**(b)** Gaussian likelihood와 Gaussian prior의 log를 전개하면:

$$\mathcal{L}(\theta) = \frac{1}{2\sigma^2}\sum_i(y_i - \theta x_i)^2 + \frac{1}{2\tau^2}\|\theta\|^2$$

공통 인수 $\frac{1}{2\sigma^2}$를 제거하면 $\text{MSE} + \frac{\sigma^2}{\tau^2}\|\theta\|^2$.

**(c)** $\lambda = \sigma^2/\tau^2$. 이는 signal-to-noise 비율의 역수와 관련된다.

**(d)** Gaussian prior는 weight space에서의 soft constraint로, posterior를 likelihood 단독 대비 prior mode 방향으로 shrink시킨다. 이는 Tikhonov regularization과 정확히 동치이며, Bayesian Occam's razor의 한 형태다. $\tau^2 \to \infty$이면 uninformative prior → MLE로 수렴, $\tau^2 \to 0$이면 $\theta \to 0$으로 수렴.

---

## 문제 22 (Softmax Jacobian)

### [EN] Problem Statement

For the softmax output $p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$, derive $\frac{\partial p_i}{\partial z_j}$ for both cases:

- When $i = j$: show that $\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$
- When $i \neq j$: show that $\frac{\partial p_i}{\partial z_j} = -p_i p_j$

Explain why this result is needed for backpropagation through the softmax layer.

### [KR] 문제

소프트맥스 출력 $p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$에 대해, $\frac{\partial p_i}{\partial z_j}$를 다음 두 경우로 나누어 유도하라:

- $i = j$일 때: $\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)$임을 보여라
- $i \neq j$일 때: $\frac{\partial p_i}{\partial z_j} = -p_i p_j$임을 보여라

이 결과가 소프트맥스 층의 역전파에 왜 필요한지 설명하라.

### 출제 의도

소프트맥스는 분류 문제의 출력층에 거의 항상 사용된다. 그 **야코비안(Jacobian)**을 직접 유도하는 것은 역전파의 수학적 기초를 이해하는 데 핵심이다. 몫의 미분법(quotient rule) 활용 능력도 함께 평가한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**소프트맥스(Softmax)란 무엇인가?**

여러 개의 숫자가 있을 때, 그것들을 "확률"로 변환하는 함수예요.

예시: 시험 점수가 국어=3, 수학=1, 영어=2 이면:
- $\exp(3) \approx 20.09$, $\exp(1) \approx 2.72$, $\exp(2) \approx 7.39$
- 합계 = $20.09 + 2.72 + 7.39 = 30.20$
- 국어 확률 = $20.09/30.20 \approx 0.665$ (66.5%)
- 수학 확률 = $2.72/30.20 \approx 0.090$ (9.0%)
- 영어 확률 = $7.39/30.20 \approx 0.245$ (24.5%)

모든 확률을 더하면 1이 돼요!

**$\exp$란?** $e^x$를 줄여 쓴 것이에요. $e \approx 2.718$이고, $\exp(x)$는 항상 양수예요.

**미분이란?** "아주 조금 바꿨을 때 결과가 얼마나 바뀌는가"를 나타내는 숫자예요.

**몫의 미분법(Quotient Rule)**: $\frac{f(x)}{g(x)}$를 미분하면 $\frac{f'g - fg'}{g^2}$

---

**유도 시작**

$p_i = \frac{\exp(z_i)}{S}$, 여기서 $S = \sum_j \exp(z_j)$라고 놓을게요.

**경우 1: $i = j$ (같은 인덱스)**

$p_i = \frac{\exp(z_i)}{S}$에서 $z_i$로 미분해요.

분자: $f = \exp(z_i)$, 분모: $g = S$

- $f' = \frac{\partial \exp(z_i)}{\partial z_i} = \exp(z_i)$ ($e^x$를 미분하면 $e^x$ 자기 자신!)
- $g' = \frac{\partial S}{\partial z_i} = \exp(z_i)$ ($S$에서 $z_i$에 관련된 항만 남음)

몫의 미분법 적용:

$$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot S - \exp(z_i) \cdot \exp(z_i)}{S^2}$$

$$= \frac{\exp(z_i)}{S} \cdot \frac{S - \exp(z_i)}{S}$$

$$= p_i \cdot \left(1 - \frac{\exp(z_i)}{S}\right) = p_i(1 - p_i) \quad \checkmark$$

**경우 2: $i \neq j$ (다른 인덱스)**

$p_i = \frac{\exp(z_i)}{S}$에서 $z_j$($j \neq i$)로 미분해요.

- $f' = \frac{\partial \exp(z_i)}{\partial z_j} = 0$ (분자에는 $z_j$가 없으므로!)
- $g' = \frac{\partial S}{\partial z_j} = \exp(z_j)$

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - \exp(z_i) \cdot \exp(z_j)}{S^2}$$

$$= -\frac{\exp(z_i)}{S} \cdot \frac{\exp(z_j)}{S} = -p_i \cdot p_j \quad \checkmark$$

**결과를 하나로 정리하면:**

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

여기서 $\delta_{ij}$는 크로네커 델타: $i = j$이면 1, 아니면 0.

---

**왜 역전파에 필요한가?**

신경망을 학습시키려면 "손실(Loss)의 변화 → 각 가중치의 변화"를 거꾸로 추적해야 해요 (역전파). 소프트맥스 층을 통과할 때, 손실에서 입력 $z_j$까지의 기울기를 구하려면:

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}$$

이때 $\frac{\partial p_i}{\partial z_j}$를 알아야 계산할 수 있어요. 이것이 바로 위에서 유도한 소프트맥스 야코비안이에요!

#### 🟡 Level 2: 고등학생 눈높이

$S = \sum_k \exp(z_k)$로 놓자.

**Case $i = j$:**

$$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot S - \exp(z_i)^2}{S^2} = \frac{\exp(z_i)}{S}\left(1 - \frac{\exp(z_i)}{S}\right) = p_i(1 - p_i)$$

**Case $i \neq j$:**

$$\frac{\partial p_i}{\partial z_j} = \frac{0 - \exp(z_i)\exp(z_j)}{S^2} = -p_i p_j$$

**통합 표현**: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$

야코비안 행렬로 쓰면: $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$

역전파에서 $\frac{\partial L}{\partial \mathbf{z}} = J^\top \frac{\partial L}{\partial \mathbf{p}}$로 기울기를 전파한다.

#### 🔴 Level 3: 대학생 눈높이

Softmax의 Jacobian: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$, 행렬 형태로 $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$.

유도는 quotient rule의 직접 적용이다. Cross-entropy loss $L = -\sum_i y_i \log p_i$와 결합하면:

$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j} = -\sum_i \frac{y_i}{p_i} \cdot p_i(\delta_{ij} - p_j) = -y_j + p_j\sum_i y_i = p_j - y_j$$

이 깔끔한 결과 ($\mathbf{p} - \mathbf{y}$)는 softmax + cross-entropy를 하나의 층으로 결합할 때의 큰 장점이다. 수치 안정성(log-sum-exp trick)과 계산 효율 모두 향상된다.

---

## 문제 23 (Norm and Regularization)

### [EN] Problem Statement

**(a)** Define the L1 norm $\|w\|_1$ and L2 norm $\|w\|_2$ for vector $w = [w_1, w_2, \ldots, w_n]$.

**(b)** For $w = [3, -4, 0, 1]$, compute both norms.

**(c)** Explain why L1 regularization promotes sparsity (some weights become exactly 0) while L2 does not. Use a geometric argument with the diamond vs circle constraint regions.

**(d)** In the MAP framework, what prior distribution corresponds to L1 regularization? L2?

### [KR] 문제

**(a)** 벡터 $w = [w_1, w_2, \ldots, w_n]$에 대해 L1 노름 $\|w\|_1$과 L2 노름 $\|w\|_2$를 정의하라.

**(b)** $w = [3, -4, 0, 1]$에 대해 두 노름을 각각 계산하라.

**(c)** L1 정규화는 희소성(일부 가중치가 정확히 0이 됨)을 촉진하는 반면 L2는 그렇지 않은 이유를 설명하라. 다이아몬드 vs 원형 제약 영역의 기하학적 논거를 사용하라.

**(d)** MAP 프레임워크에서 L1 정규화에 대응하는 사전 분포는? L2는?

### 출제 의도

노름(norm)은 벡터의 "크기"를 측정하는 기본 개념이다. L1과 L2 정규화는 딥러닝에서 과적합 방지를 위한 가장 대표적인 기법이며, **기하학적 직관**과 **확률론적 해석** 두 가지 관점에서 이해해야 한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**노름(Norm)이란?** 벡터의 "길이" 또는 "크기"를 재는 방법이에요.

**(a) 정의**

**L1 노름** (맨해튼 거리): 각 원소의 **절댓값**을 모두 더해요.

$$\|w\|_1 = |w_1| + |w_2| + \cdots + |w_n|$$

비유: 바둑판 모양의 도시(맨해튼)에서 가로·세로로만 이동할 때의 거리.

**L2 노름** (유클리드 거리): 각 원소를 **제곱**해서 더한 뒤 **제곱근**을 씌워요.

$$\|w\|_2 = \sqrt{w_1^2 + w_2^2 + \cdots + w_n^2}$$

비유: 직선으로 잰 거리. 피타고라스 정리와 같아요!

---

**(b) 계산 ($w = [3, -4, 0, 1]$)**

**L1 노름:**
$$\|w\|_1 = |3| + |-4| + |0| + |1| = 3 + 4 + 0 + 1 = \boxed{8}$$

**L2 노름:**
$$\|w\|_2 = \sqrt{3^2 + (-4)^2 + 0^2 + 1^2} = \sqrt{9 + 16 + 0 + 1} = \sqrt{26} \approx \boxed{5.099}$$

---

**(c) L1은 왜 가중치를 0으로 만들까?**

2차원에서 생각해볼게요. $w = [w_1, w_2]$

**L1 제약 영역**: $|w_1| + |w_2| \leq c$ → **다이아몬드(마름모)** 모양
```
        ·
       / \
      /   \
     ·     ·
      \   /
       \ /
        ·
```

**L2 제약 영역**: $w_1^2 + w_2^2 \leq c^2$ → **원** 모양

이제 손실 함수의 등고선(타원 모양)이 이 제약 영역에 처음 닿는 점을 찾아야 해요.

- **다이아몬드(L1)**: 꼭짓점이 축 위에 있어요! 타원이 꼭짓점에 닿을 확률이 높아요. 꼭짓점에서는 $w_1 = 0$ 또는 $w_2 = 0$이에요. → **희소성(sparsity)**!
- **원(L2)**: 어디든 닿을 수 있어요. 축 위에 닿을 특별한 이유가 없어요. → 가중치가 작아지지만 정확히 0이 되진 않아요.

비유: L1은 "가장 중요한 과목만 공부"하는 전략이고, L2는 "모든 과목을 조금씩 공부"하는 전략이에요.

---

**(d) 대응하는 사전 분포**

- **L2 정규화** ↔ **가우시안(정규분포) 사전 분포**: $p(w) = \mathcal{N}(0, \tau^2)$
  - 문제 21에서 이미 유도했어요!

- **L1 정규화** ↔ **라플라스(Laplace) 분포 사전 분포**: $p(w) = \frac{1}{2b}\exp\left(-\frac{|w|}{b}\right)$
  - 이 분포는 0에서 뾰족한 모양이에요 (정규분포보다 0 근처에 더 집중).
  - 로그를 취하면: $\log p(w) = -\frac{|w|}{b} + \text{const}$ → 절댓값 = L1 노름!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $\|w\|_1 = \sum_i |w_i|$, $\|w\|_2 = \sqrt{\sum_i w_i^2}$

**(b)** $\|w\|_1 = 8$, $\|w\|_2 = \sqrt{26}$

**(c)** 제약 최적화 관점: $\min_w L(w)$ s.t. $\|w\|_p \leq c$의 해는 손실 등고선이 제약 영역에 접하는 점이다.

- L1: 제약 영역($\ell_1$ ball)은 축에 꼭짓점을 가지는 다면체. 등고선이 꼭짓점에서 접할 확률이 높으므로 좌표값 = 0이 되는 해를 유도.
- L2: 제약 영역($\ell_2$ ball)은 초구(hypersphere). 등고선이 어디서든 접할 수 있어 좌표가 정확히 0이 될 이유가 없다.

또한 subgradient 관점: L1의 $|w_i|$의 미분은 $w_i = 0$에서 불연속(subgradient $\in [-1, 1]$)이므로 0으로 수렴 후 거기 "고정"될 수 있다. L2의 $w_i^2$의 미분은 $2w_i$로 0에서 기울기도 0이므로 0을 향해 점근적으로만 접근한다.

**(d)** L1 → Laplace prior: $p(w) \propto \exp(-\|w\|_1/b)$. L2 → Gaussian prior: $p(w) \propto \exp(-\|w\|_2^2/(2\tau^2))$. MAP estimation과 정규화 손실 함수의 동치 관계에서 직접 유도된다.

#### 🔴 Level 3: 대학생 눈높이

**(a-b)** 생략 (기본 정의 및 산술).

**(c)** KKT 조건으로 엄밀하게 설명: Lagrangian $\mathcal{L} = L(w) + \lambda(\|w\|_p - c)$에서, L1의 경우 $\partial|w_i|/\partial w_i$가 $w_i = 0$에서 subdifferential $[-1,1]$을 가지므로 KKT의 stationarity condition $\partial L/\partial w_i \in \lambda[-1,1]$이 만족될 수 있다. 즉 gradient가 충분히 작으면 $w_i = 0$이 해가 된다. L2는 미분이 연속이므로 해가 정확히 축 위에 놓일 measure-zero 사건이다.

**(d)** Exponential family conjugacy와 MAP의 관계: Gaussian prior → L2 (ridge), Laplace prior → L1 (lasso). Elastic net은 이 둘의 혼합으로, Gaussian-Laplace mixture prior에 대응한다.

---

## 문제 24 (Computation Graph and Backpropagation)

### [EN] Problem Statement

Given the computation: $L = (wx + b - y)^2$ where $w=2, x=3, b=1, y=10$.

**(a)** Draw the computation graph with intermediate variables.

**(b)** Compute the forward pass (find $L$).

**(c)** Compute $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$ using the chain rule, showing each intermediate derivative.

**(d)** If learning rate $\eta = 0.01$, what are the updated $w$ and $b$ after one gradient descent step?

### [KR] 문제

다음 계산이 주어졌다: $L = (wx + b - y)^2$, $w=2, x=3, b=1, y=10$.

**(a)** 중간 변수를 포함한 계산 그래프를 그려라.

**(b)** 순전파(forward pass)를 계산하여 $L$을 구하라.

**(c)** 연쇄 법칙을 사용하여 $\frac{\partial L}{\partial w}$와 $\frac{\partial L}{\partial b}$를 구하라. 각 중간 미분값을 모두 보여라.

**(d)** 학습률 $\eta = 0.01$일 때, 경사하강법 1회 후 업데이트된 $w$와 $b$를 구하라.

### 출제 의도

역전파(backpropagation)의 핵심 원리인 **계산 그래프 + 연쇄 법칙**을 직접 손으로 계산하는 문제다. 추상적 공식이 아닌 **구체적 수치**로 각 단계를 추적하여, 경사하강법의 전체 과정을 이해하고 있는지 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**계산 그래프란?** 복잡한 계산을 작은 단계로 쪼개서 그림으로 나타낸 것이에요. 각 동그라미(노드)는 하나의 간단한 연산을 해요.

---

**(a) 계산 그래프**

$L = (wx + b - y)^2$을 분해할게요:

1. $u_1 = w \times x$ (곱셈)
2. $u_2 = u_1 + b$ (덧셈)
3. $u_3 = u_2 - y$ (뺄셈)
4. $L = u_3^2$ (제곱)

```
w ──→ [×] ──→ u₁ ──→ [+] ──→ u₂ ──→ [−] ──→ u₃ ──→ [²] ──→ L
x ──↗              b ──↗           y ──↗
```

---

**(b) 순전파 (왼쪽 → 오른쪽으로 계산)**

$$u_1 = w \times x = 2 \times 3 = 6$$
$$u_2 = u_1 + b = 6 + 1 = 7$$
$$u_3 = u_2 - y = 7 - 10 = -3$$
$$L = u_3^2 = (-3)^2 = \boxed{9}$$

직관: 예측값은 $wx + b = 7$이고, 실제값은 $y = 10$이므로, 오차가 $-3$이고, 제곱 오차는 $9$.

---

**(c) 역전파 (오른쪽 → 왼쪽으로 미분 전파)**

**연쇄 법칙(Chain Rule)이란?** 여러 함수가 연결되어 있을 때, 전체 미분 = 각 단계 미분의 곱.

비유: 자전거 기어처럼 — 페달을 1바퀴 돌리면 첫 번째 기어가 3바퀴, 두 번째 기어가 2바퀴 돌면, 총 $1 \times 3 \times 2 = 6$바퀴.

**Step 1**: $\frac{\partial L}{\partial u_3}$

$L = u_3^2$이므로 $\frac{\partial L}{\partial u_3} = 2u_3 = 2 \times (-3) = -6$

**Step 2**: $\frac{\partial u_3}{\partial u_2}$

$u_3 = u_2 - y$이므로 $\frac{\partial u_3}{\partial u_2} = 1$

**Step 3**: $\frac{\partial u_2}{\partial u_1}$ 및 $\frac{\partial u_2}{\partial b}$

$u_2 = u_1 + b$이므로 $\frac{\partial u_2}{\partial u_1} = 1$, $\frac{\partial u_2}{\partial b} = 1$

**Step 4**: $\frac{\partial u_1}{\partial w}$

$u_1 = w \times x$이므로 $\frac{\partial u_1}{\partial w} = x = 3$

---

**$\frac{\partial L}{\partial w}$ 구하기** (연쇄 법칙으로 곱하기):

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial u_3} \cdot \frac{\partial u_3}{\partial u_2} \cdot \frac{\partial u_2}{\partial u_1} \cdot \frac{\partial u_1}{\partial w}$$

$$= (-6) \times 1 \times 1 \times 3 = \boxed{-18}$$

**$\frac{\partial L}{\partial b}$ 구하기**:

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial u_3} \cdot \frac{\partial u_3}{\partial u_2} \cdot \frac{\partial u_2}{\partial b}$$

$$= (-6) \times 1 \times 1 = \boxed{-6}$$

**검산**: $L = (2 \cdot 3 + 1 - 10)^2$에서 직접 미분해도 $\frac{\partial L}{\partial w} = 2(wx+b-y) \cdot x = 2(-3)(3) = -18$ ✓

---

**(d) 경사하강법 업데이트**

$$w_{\text{new}} = w - \eta \cdot \frac{\partial L}{\partial w} = 2 - 0.01 \times (-18) = 2 + 0.18 = \boxed{2.18}$$

$$b_{\text{new}} = b - \eta \cdot \frac{\partial L}{\partial b} = 1 - 0.01 \times (-6) = 1 + 0.06 = \boxed{1.06}$$

직관: 기울기가 음수 = "이 방향으로 가면 손실이 줄어든다" → $w$와 $b$를 증가시켜요. 예측값 $7$이 실제값 $10$보다 작으므로, $w$와 $b$를 키우는 게 맞아요!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 계산 그래프: $w, x \to u_1=wx \to u_2=u_1+b \to u_3=u_2-y \to L=u_3^2$

**(b)** $L = (2 \cdot 3 + 1 - 10)^2 = (-3)^2 = 9$

**(c)** $\frac{\partial L}{\partial w} = 2(wx+b-y) \cdot x = 2(-3)(3) = -18$, $\frac{\partial L}{\partial b} = 2(wx+b-y) = 2(-3) = -6$

**(d)** $w \leftarrow 2 - 0.01(-18) = 2.18$, $b \leftarrow 1 - 0.01(-6) = 1.06$

#### 🔴 Level 3: 대학생 눈높이

$L = \|f_\theta(x) - y\|^2$, $f_\theta(x) = wx + b$. Forward: residual $r = -3$, $L = 9$.

Backward: $\nabla_\theta L = 2r \cdot \nabla_\theta f = 2(-3)[x, 1]^\top = [-18, -6]^\top$.

GD update: $\theta \leftarrow \theta - \eta \nabla L$, 즉 $[w, b] \leftarrow [2.18, 1.06]$.

이는 단일 데이터 포인트에 대한 SGD 1-step이며, 여러 데이터에 대해 배치로 확장하면 $\nabla L = \frac{2}{n}\sum_i r_i [x_i, 1]^\top$가 된다.

---

## 문제 25 (Pseudo-inverse and Least Squares)

### [EN] Problem Statement

When the system $Ax = b$ has no exact solution (overdetermined system):

**(a)** Explain why $Ax = b$ might have no solution geometrically.

**(b)** Show that the least-squares solution $x^* = \arg\min \|Ax - b\|^2$ satisfies the normal equation $A^\top A x = A^\top b$.

**(c)** For $A = \begin{bmatrix}1\\2\\3\end{bmatrix}$, $b = \begin{bmatrix}1\\3\\4\end{bmatrix}^\top$, compute $x^*$ numerically.

**(d)** What is the pseudo-inverse $A^+$ and how does it relate to $x^*$?

### [KR] 문제

연립방정식 $Ax = b$가 정확한 해를 갖지 않을 때 (과결정 시스템):

**(a)** $Ax = b$가 기하학적으로 왜 해가 없을 수 있는지 설명하라.

**(b)** 최소제곱해 $x^* = \arg\min \|Ax - b\|^2$가 정규 방정식 $A^\top Ax = A^\top b$를 만족함을 보여라.

**(c)** $A = \begin{bmatrix}1\\2\\3\end{bmatrix}$, $b = \begin{bmatrix}1\\3\\4\end{bmatrix}^\top$에 대해 $x^*$를 수치적으로 계산하라.

**(d)** 의사역행렬(pseudo-inverse) $A^+$란 무엇이며 $x^*$와 어떤 관계인지 설명하라.

### 출제 의도

선형 회귀의 수학적 기초인 **최소제곱법**과 **정규 방정식**의 유도를 확인한다. 딥러닝의 가장 단순한 형태인 선형 모델의 최적해를 닫힌 형태(closed-form)로 구하는 능력을 평가하며, 의사역행렬이라는 일반화된 역행렬 개념까지 연결한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**행렬이란?** 숫자를 직사각형 모양으로 배열한 것이에요. 연립방정식을 간결하게 쓰는 방법이에요.

$Ax = b$는 여러 개의 방정식을 한 번에 쓴 것이에요.

---

**(a) 왜 해가 없을 수 있나?**

$A = \begin{bmatrix}1\\2\\3\end{bmatrix}$, $x$는 스칼라, $b = \begin{bmatrix}1\\3\\4\end{bmatrix}$

이걸 풀어쓰면:
- $1 \cdot x = 1$ → $x = 1$
- $2 \cdot x = 3$ → $x = 1.5$
- $3 \cdot x = 4$ → $x = 1.333...$

세 방정식이 모두 다른 $x$를 원해요! **세 조건을 동시에 만족하는 $x$가 없어요.**

**기하학적 해석**: $Ax$는 벡터 $A$의 스칼라 배, 즉 $A$ 방향의 직선 위 점이에요. $b$가 이 직선 위에 있지 않으면 $Ax = b$를 정확히 만족하는 $x$가 없어요.

비유: 한 줄로만 서야 하는데(직선), 목적지($b$)가 그 줄 위에 없으면 정확히 도착할 수 없어요. 가장 가까운 점으로 가는 수밖에!

---

**(b) 정규 방정식 유도**

"가장 가까운 점"을 찾는다는 것은 $\|Ax - b\|^2$를 최소화하는 것이에요.

$$f(x) = \|Ax - b\|^2 = (Ax - b)^\top(Ax - b)$$

전개할게요:

$$f(x) = x^\top A^\top A x - 2x^\top A^\top b + b^\top b$$

최솟값에서는 미분이 0이에요:

$$\frac{\partial f}{\partial x} = 2A^\top Ax - 2A^\top b = 0$$

$$\boxed{A^\top Ax = A^\top b}$$

이것이 **정규 방정식(Normal Equation)**이에요!

**직관적 해석**: $Ax^*$은 $b$를 $A$의 열공간에 **수직으로 내린 정사영(projection)**이에요. "수직"이라는 조건이 $A^\top(b - Ax) = 0$, 즉 잔차가 $A$의 열공간에 수직이라는 뜻이에요.

---

**(c) 수치 계산**

$A = \begin{bmatrix}1\\2\\3\end{bmatrix}$, $b = \begin{bmatrix}1\\3\\4\end{bmatrix}$

**Step 1**: $A^\top A$ 계산

$$A^\top A = [1\; 2\; 3] \begin{bmatrix}1\\2\\3\end{bmatrix} = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14$$

**Step 2**: $A^\top b$ 계산

$$A^\top b = [1\; 2\; 3] \begin{bmatrix}1\\3\\4\end{bmatrix} = 1 \cdot 1 + 2 \cdot 3 + 3 \cdot 4 = 1 + 6 + 12 = 19$$

**Step 3**: 정규 방정식 풀기

$$14x = 19$$
$$x^* = \frac{19}{14} \approx \boxed{1.357}$$

**검증**: $Ax^* = \begin{bmatrix}1.357\\2.714\\4.071\end{bmatrix}$, 이것과 $b = \begin{bmatrix}1\\3\\4\end{bmatrix}$의 차이가 최소!

---

**(d) 의사역행렬(Pseudo-inverse)**

보통 $Ax = b$의 해는 $x = A^{-1}b$이지만, $A$가 정사각 행렬이 아니면 역행렬이 없어요!

**의사역행렬** $A^+$는 이런 경우에도 "역행렬 비슷한 것"을 정의한 것이에요.

$$A^+ = (A^\top A)^{-1} A^\top$$

이를 사용하면:

$$x^* = A^+ b = (A^\top A)^{-1} A^\top b$$

우리 문제에서: $A^+ = \frac{1}{14}[1\; 2\; 3] = [\frac{1}{14}\; \frac{2}{14}\; \frac{3}{14}]$

$x^* = A^+ b = \frac{1}{14}(1 + 6 + 12) = \frac{19}{14}$ ✓

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $\text{Col}(A)$는 $\mathbb{R}^3$의 1차원 부분공간(직선). $b$가 이 직선 위에 없으면 해가 없다.

**(b)** $f(x) = \|Ax-b\|^2$를 전개하고 $x$에 대해 미분하면 $\nabla_x f = 2A^\top(Ax - b) = 0$, 즉 $A^\top Ax = A^\top b$.

기하학적으로: 최적 잔차 $r^* = b - Ax^*$는 $\text{Col}(A)$에 직교해야 하므로 $A^\top r^* = 0$.

**(c)** $A^\top A = 14$, $A^\top b = 19$, $x^* = 19/14$.

**(d)** $A^+ = (A^\top A)^{-1}A^\top$는 Moore-Penrose pseudo-inverse의 좌측 역(left inverse) 형태. $x^* = A^+ b$. 일반적으로 $A$의 SVD $A = U\Sigma V^\top$일 때 $A^+ = V\Sigma^+ U^\top$으로 정의된다.

#### 🔴 Level 3: 대학생 눈높이

**(a)** $b \notin \text{Col}(A)$이면 해 부재. Overdetermined ($m > n$)일 때 일반적.

**(b)** $\nabla_x \|Ax-b\|^2 = 2A^\top(Ax-b) = 0$에서 정규방정식 유도. 이는 orthogonal projection $\hat{b} = A(A^\top A)^{-1}A^\top b$로 이해 가능.

**(c)** $x^* = 19/14$.

**(d)** Moore-Penrose pseudo-inverse: $A^+ = V\Sigma^+ U^\top$ (SVD 기반). Full column rank일 때 $A^+ = (A^\top A)^{-1}A^\top$ (left inverse). 4가지 조건 만족: $AA^+A = A$, $A^+AA^+ = A^+$, $(AA^+)^\top = AA^+$, $(A^+A)^\top = A^+A$. 딥러닝에서 linear layer의 closed-form solution이 바로 이것이며, gradient descent는 이를 iterative하게 근사한다.

---

## 문제 26 (Bias-Variance Tradeoff)

### [EN] Problem Statement

**(a)** Decompose the expected prediction error into bias², variance, and irreducible noise:

$$E[(y - \hat{h}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

**(b)** Explain with a dartboard analogy what high bias and high variance mean.

**(c)** A linear model underfits while a degree-20 polynomial overfits the same data. Which has higher bias? Higher variance? Explain.

**(d)** How does regularization strength affect the bias-variance tradeoff?

### [KR] 문제

**(a)** 기대 예측 오차를 편향², 분산, 환원불가능 노이즈로 분해하라:

$$E[(y - \hat{h}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

**(b)** 다트판 비유를 사용하여 높은 편향과 높은 분산이 무엇을 의미하는지 설명하라.

**(c)** 선형 모델은 과소적합하고 20차 다항식은 과적합한다면, 어느 것이 편향이 더 높은가? 분산이 더 높은가? 설명하라.

**(d)** 정규화 강도가 편향-분산 트레이드오프에 어떤 영향을 미치는지 설명하라.

### 출제 의도

**편향-분산 분해**는 모델 성능을 이해하는 가장 근본적인 프레임워크다. 과적합(overfitting)과 과소적합(underfitting)을 단순히 외우는 것이 아니라, **수학적으로 왜** 발생하는지를 이 분해를 통해 설명할 수 있는지 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**핵심 개념 설명**

모델을 학습시킬 때, 매번 다른 데이터 세트를 사용하면 다른 모델이 나와요. 같은 방법으로 100번 학습시키면 100개의 서로 다른 모델이 생겨요.

- **편향(Bias)**: 100개 모델의 **평균 예측**이 정답과 얼마나 떨어져 있는가
- **분산(Variance)**: 100개 모델의 **예측이 서로 얼마나 다른가**
- **노이즈**: 데이터 자체에 포함된 랜덤한 오차 (어떤 모델도 줄일 수 없음)

---

**(a) 분해 유도**

실제 데이터: $y = f(x) + \epsilon$, 여기서 $\epsilon$은 노이즈, $E[\epsilon] = 0$, $\text{Var}(\epsilon) = \sigma^2$.

$\hat{h}(x)$는 우리 모델의 예측이에요. $\bar{h}(x) = E[\hat{h}(x)]$는 "평균 모델"이에요.

기대 오차를 분해할게요:

$$E[(y - \hat{h}(x))^2]$$

$y = f(x) + \epsilon$을 대입하고, $\hat{h}$를 평균 $\bar{h}$를 중심으로 분해:

$$= E[(f(x) + \epsilon - \hat{h}(x))^2]$$

$$= E[((f(x) - \bar{h}(x)) + (\bar{h}(x) - \hat{h}(x)) + \epsilon)^2]$$

제곱을 전개할게요. 세 항을 $A$, $B$, $C$로 놓으면:
- $A = f(x) - \bar{h}(x)$ (상수)
- $B = \bar{h}(x) - \hat{h}(x)$ ($E[B] = 0$)
- $C = \epsilon$ ($E[C] = 0$)

$$E[(A + B + C)^2] = E[A^2 + B^2 + C^2 + 2AB + 2AC + 2BC]$$

- $E[A^2] = (f(x) - \bar{h}(x))^2$ → 이것이 $\text{Bias}^2$!
- $E[B^2] = E[(\hat{h}(x) - \bar{h}(x))^2]$ → 이것이 $\text{Variance}$!
- $E[C^2] = E[\epsilon^2] = \sigma^2$ → 이것이 환원불가능 노이즈!
- $E[2AB] = 2A \cdot E[B] = 0$ (∵ $E[B] = 0$)
- $E[2AC] = 2A \cdot E[C] = 0$ (∵ $E[\epsilon] = 0$)
- $E[2BC] = 2E[B] \cdot E[C] = 0$ (∵ $B$와 $C$는 독립, 각각 평균 0)

따라서:

$$\boxed{E[(y - \hat{h}(x))^2] = \underbrace{(f(x) - \bar{h}(x))^2}_{\text{Bias}^2} + \underbrace{E[(\hat{h}(x) - \bar{h}(x))^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}}$$

---

**(b) 다트판 비유**

다트를 과녁에 던진다고 상상해요.

| | 낮은 분산 (모여있음) | 높은 분산 (흩어져있음) |
|---|---|---|
| **낮은 편향** (중심 근처) | 🎯 잘 모아서 중앙에! (이상적) | 중앙 근처지만 흩어져 있음 |
| **높은 편향** (중심에서 멀리) | 한쪽에 모여있지만 중심에서 멀리 | 아무데나 흩어져 있고 중심에서도 멀리 (최악) |

- **높은 편향**: 다트가 체계적으로 한쪽으로 치우침 → 조준 자체가 잘못됨
- **높은 분산**: 다트가 여기저기 흩어짐 → 손이 불안정함

---

**(c) 선형 모델 vs 20차 다항식**

| | 선형 모델 | 20차 다항식 |
|---|---|---|
| **편향** | **높음** ↑ (곡선 데이터를 직선으로 못 따라감) | **낮음** ↓ (어떤 곡선이든 따라갈 수 있음) |
| **분산** | **낮음** ↓ (데이터가 바뀌어도 직선은 비슷) | **높음** ↑ (데이터가 조금만 바뀌어도 곡선이 완전히 바뀜) |

**왜?** 선형 모델은 자유도가 2개(기울기, 절편)뿐이라 데이터를 충분히 표현 못하지만(과소적합), 안정적이에요. 20차 다항식은 자유도가 21개여서 데이터를 완벽히 표현하지만(과적합), 새 데이터에 민감해요.

---

**(d) 정규화 강도의 영향**

정규화 강도 $\lambda$를 키우면:
- **편향 증가** ↑: 모델이 더 단순해지므로 데이터의 진짜 패턴도 놓칠 수 있음
- **분산 감소** ↓: 모델이 데이터 변동에 덜 민감해짐

$\lambda$가 너무 작으면 → 과적합 (높은 분산)
$\lambda$가 너무 크면 → 과소적합 (높은 편향)
**적절한 $\lambda$** → 총 오차 = Bias² + Variance가 최소!

이것이 바로 **편향-분산 트레이드오프**예요. 한쪽을 줄이면 다른 쪽이 커지는 관계!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $y = f(x) + \epsilon$에서 $E[(y-\hat{h})^2]$를 $\bar{h} = E[\hat{h}]$를 이용해 분해하면, 교차항이 독립성과 $E[\epsilon]=0$으로 소거되어 Bias² + Variance + $\sigma^2$ 세 항으로 분리된다.

**(b)** Bias = 체계적 오차 (조준 방향), Variance = 랜덤 오차 (손 떨림). 총 오차 = 두 오차의 합.

**(c)** 모델 복잡도 ↑ → Bias ↓, Variance ↑. 선형 모델: 높은 Bias, 낮은 Variance. 고차 다항식: 낮은 Bias, 높은 Variance.

**(d)** $\lambda$ ↑ → effective model complexity ↓ → Bias ↑, Variance ↓. 최적 $\lambda$는 총 오차를 최소화하는 지점. Cross-validation으로 탐색.

#### 🔴 Level 3: 대학생 눈높이

**(a)** Standard bias-variance decomposition. $\hat{h}$가 training set $D$에 대한 random variable일 때 expectation은 $E_D$로 취한다.

**(b)** Bias: systematic error (model misspecification). Variance: estimation error (finite sample).

**(c)** VC dimension / model complexity 관점: 선형 모델의 VC dim은 2, 20차 다항식은 21. Approximation error(bias)와 estimation error(variance)의 tradeoff.

**(d)** Regularization은 hypothesis class를 효과적으로 축소한다. $\lambda$는 effective degrees of freedom을 조절. Ridge regression에서 $\text{df}(\lambda) = \text{tr}(X(X^\top X + \lambda I)^{-1}X^\top)$로 정량화 가능. Double descent 현상에서는 이 고전적 tradeoff가 깨질 수 있으나, 그 경우에도 implicit regularization이 작용한다.

---

## 문제 27 (Hoeffding's Inequality and Generalization)

### [EN] Problem Statement

State Hoeffding's inequality. Then:

**(a)** We flip a coin 100 times and get 60 heads. Use Hoeffding's inequality to bound the probability that the true $p$ differs from $0.6$ by more than $0.1$.

**(b)** Explain how Hoeffding's inequality connects to the generalization gap in machine learning (empirical risk vs population risk).

**(c)** Why does having more training data reduce the generalization gap?

### [KR] 문제

Hoeffding 부등식을 기술하라. 그 후:

**(a)** 동전을 100번 던져서 60번 앞면이 나왔다. Hoeffding 부등식을 사용하여 실제 $p$가 $0.6$에서 $0.1$ 이상 차이날 확률의 상한을 구하라.

**(b)** Hoeffding 부등식이 머신러닝의 일반화 갭(경험적 위험 vs 모집단 위험)과 어떻게 연결되는지 설명하라.

**(c)** 훈련 데이터가 많아지면 왜 일반화 갭이 줄어드는지 설명하라.

### 출제 의도

Hoeffding 부등식은 **학습 이론(learning theory)**의 기초이다. "훈련 데이터에서 잘 되면 실제 데이터에서도 잘 될까?"라는 근본 질문에 대한 **수학적 보장**을 제공한다. 경험적 위험과 모집단 위험의 관계를 정량적으로 이해하는지 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**Hoeffding 부등식이란?**

"표본 평균은 진짜 평균에서 크게 벗어나기 어렵다"를 수학으로 표현한 것이에요.

**정식 진술**: $X_1, X_2, \ldots, X_n$이 $[a, b]$ 범위의 독립 확률변수이고, 표본 평균이 $\bar{X} = \frac{1}{n}\sum X_i$, 진짜 평균이 $\mu = E[\bar{X}]$일 때:

$$P(|\bar{X} - \mu| \geq t) \leq 2\exp\left(-\frac{2n t^2}{(b-a)^2}\right)$$

풀어서 말하면: 표본 평균이 진짜 평균에서 $t$ 이상 떨어질 확률은 **지수적으로** 작아요. $n$이 클수록, $t$가 클수록 그 확률이 급격히 줄어요.

---

**(a) 동전 문제**

- $n = 100$, 표본 평균 $\bar{X} = 0.6$ (60번 앞면)
- 각 $X_i \in [0, 1]$이므로 $a = 0, b = 1$
- $t = 0.1$ (진짜 $p$와의 차이)

Hoeffding 부등식에 대입:

$$P(|\bar{X} - p| \geq 0.1) \leq 2\exp\left(-\frac{2 \times 100 \times 0.1^2}{(1-0)^2}\right)$$

$$= 2\exp\left(-\frac{2 \times 100 \times 0.01}{1}\right) = 2\exp(-2)$$

$$= 2 \times e^{-2} \approx 2 \times 0.1353 = \boxed{0.2707}$$

즉: 진짜 확률 $p$가 $0.5$에서 $0.7$ 범위 밖에 있을 확률은 **최대 약 27%**예요.

(참고: 이것은 상한(upper bound)이에요. 실제 확률은 이보다 훨씬 작을 수 있어요.)

---

**(b) 머신러닝과의 연결**

머신러닝에서:
- **경험적 위험(Empirical Risk)** $\hat{R}$: 훈련 데이터에서의 평균 오차 (시험 문제를 미리 풀어본 점수)
- **모집단 위험(Population Risk)** $R$: 전체 데이터에서의 평균 오차 (실제 시험 점수)
- **일반화 갭**: $|R - \hat{R}|$ (연습 점수와 실전 점수의 차이)

Hoeffding 부등식이 말하는 것:

$$P(|R - \hat{R}| \geq t) \leq 2\exp(-2nt^2)$$

이것은 "훈련 데이터가 충분하면, 훈련 성능과 실제 성능이 비슷하다"는 보장이에요!

비유: 시험 범위에서 100문제를 풀어봤는데 80점이었다면, 실제 시험에서도 대략 80점 근처일 가능성이 높아요. 하지만 3문제만 풀어봤다면? 실제 시험 점수는 예측하기 어려워요.

---

**(c) 데이터가 많으면 왜 일반화 갭이 줄어드나?**

Hoeffding 부등식에서 핵심은 지수 부분: $\exp(-2nt^2)$

- $n$이 커지면 → $\exp$ 안의 값이 더 음수가 됨 → 전체 확률이 급격히 0에 가까워짐
- 즉, **같은 갭** $t$에 대해 확률 상한이 줄어듦
- 또는 **같은 확률**에 대해 보장할 수 있는 갭 $t$가 줄어듦

구체적으로: 확률 상한을 $\delta$로 고정하면:

$$t \leq \sqrt{\frac{\log(2/\delta)}{2n}}$$

$n$이 4배가 되면 $t$는 절반으로! (제곱근 관계)

비유: 여론조사를 100명에게 하면 오차 ±10%, 10,000명에게 하면 오차 ±1%. 더 많이 물어볼수록 진짜 의견에 가까워져요!

#### 🟡 Level 2: 고등학생 눈높이

**Hoeffding**: $P(|\bar{X} - \mu| \geq t) \leq 2\exp(-2nt^2/(b-a)^2)$

**(a)** $2\exp(-2 \cdot 100 \cdot 0.01) = 2e^{-2} \approx 0.271$.

**(b)** 고정된 가설 $h$에 대해 $\hat{R}(h) = \frac{1}{n}\sum L(h(x_i), y_i)$는 $R(h) = E[L(h(x), y)]$의 sample mean이다. Hoeffding을 적용하면 $P(|R(h) - \hat{R}(h)| \geq \epsilon) \leq 2e^{-2n\epsilon^2}$ (loss가 $[0,1]$일 때).

단, 실제 학습에서는 가설 $h$를 데이터로 선택하므로 union bound를 적용해야 한다: $|\mathcal{H}|$개의 가설에 대해 $P(\exists h: |R-\hat{R}| \geq \epsilon) \leq 2|\mathcal{H}|e^{-2n\epsilon^2}$.

**(c)** $\epsilon \leq \sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}$이므로 $n \to \infty$이면 $\epsilon \to 0$. 수렴 속도는 $O(1/\sqrt{n})$.

#### 🔴 Level 3: 대학생 눈높이

**(a)** $2e^{-2} \approx 0.271$.

**(b)** 단일 가설에 대한 Hoeffding은 uniform convergence의 출발점이다. 유한 가설 공간 $|\mathcal{H}| < \infty$에서 union bound로 확장: PAC learning의 sample complexity $n \geq \frac{\log(2|\mathcal{H}|/\delta)}{2\epsilon^2}$. 무한 가설 공간에서는 VC dimension이나 Rademacher complexity로 대체한다.

**(c)** Generalization bound의 $n$ 의존성: $O(\sqrt{(\text{VC dim} \cdot \log n)/n})$. 데이터 증가 → 경험적 과정의 집중 부등식이 더 tight해짐. 이는 law of large numbers의 non-asymptotic 버전이다. 현대 딥러닝에서는 over-parameterization에도 불구하고 일반화가 잘 되는 현상이 classical bound로 설명되지 않아, PAC-Bayes, compression, implicit regularization 등의 새로운 이론이 필요하다.

---

## 문제 28 (Activation Functions)

### [EN] Problem Statement

**(a)** Write the formulas for ReLU, sigmoid, and tanh activation functions.

**(b)** Compute their derivatives.

**(c)** Explain the vanishing gradient problem: why does sigmoid cause it while ReLU does not?

**(d)** For a 10-layer network with sigmoid activations, if each layer's gradient is at most $0.25$, estimate the gradient at the first layer. What does this mean for learning?

### [KR] 문제

**(a)** ReLU, sigmoid, tanh 활성화 함수의 공식을 쓰라.

**(b)** 각 함수의 도함수를 구하라.

**(c)** 기울기 소실 문제를 설명하라: sigmoid는 왜 이 문제를 일으키고 ReLU는 왜 일으키지 않는가?

**(d)** sigmoid 활성화를 사용하는 10층 네트워크에서, 각 층의 기울기 최댓값이 $0.25$이면, 첫 번째 층에서의 기울기를 추정하라. 이것이 학습에 어떤 의미인가?

### 출제 의도

활성화 함수는 신경망에 **비선형성**을 부여하는 핵심 요소다. 그 도함수의 성질이 **기울기 소실(vanishing gradient)**과 직결되며, 이는 딥러닝 역사에서 가장 중요한 문제 중 하나였다. 수학적으로 왜 발생하는지를 유도할 수 있는지 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**활성화 함수란?** 신경망의 각 뉴런에서 입력을 변환하는 함수예요. 이것 없으면 아무리 층을 쌓아도 그냥 직선(선형)이에요!

---

**(a) 공식**

**ReLU** (Rectified Linear Unit):

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

비유: 양수면 그대로 통과, 음수면 0으로 차단. "관문"처럼 작동!

**Sigmoid**:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

출력이 항상 0과 1 사이. 비유: 볼륨 조절기 — 입력이 아무리 크거나 작아도 출력은 0~1 사이.

**Tanh** (Hyperbolic Tangent):

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

출력이 항상 -1과 1 사이. Sigmoid와 비슷하지만 중심이 0.

---

**(b) 도함수**

**ReLU의 미분**:

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

양수면 기울기 1, 음수면 기울기 0. 매우 간단!

**Sigmoid의 미분**:

유도: $\sigma(x) = (1 + e^{-x})^{-1}$

$$\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)(1 - \sigma(x))$$

$\sigma(x)$가 0~1 사이이므로, $\sigma'(x)$의 최댓값은 $x=0$일 때:

$$\sigma'(0) = 0.5 \times 0.5 = 0.25$$

즉 **기울기가 항상 0.25 이하**!

**Tanh의 미분**:

$$\tanh'(x) = 1 - \tanh^2(x)$$

최댓값은 $x=0$일 때: $1 - 0 = 1$. Sigmoid보다는 낫지만, $|x|$가 커지면 여전히 0에 가까워짐.

---

**(c) 기울기 소실 문제 (Vanishing Gradient)**

역전파에서 기울기는 연쇄 법칙에 의해 **곱해져** 전파돼요:

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h_{10}} \cdot \frac{\partial h_{10}}{\partial h_9} \cdot \frac{\partial h_9}{\partial h_8} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}$$

각 $\frac{\partial h_{k+1}}{\partial h_k}$에 활성화 함수의 미분이 포함돼요.

**Sigmoid**: 미분의 최댓값이 0.25. 10개를 곱하면?

$$0.25^{10} = 0.25 \times 0.25 \times \cdots = 0.00000095 \approx 10^{-6}$$

기울기가 **백만 분의 1**로 줄어요! 첫 번째 층은 거의 아무것도 학습할 수 없어요.

**ReLU**: 양수 영역에서 미분이 **정확히 1**. 10개를 곱해도:

$$1^{10} = 1$$

기울기가 줄어들지 않아요! 이것이 ReLU가 혁명적인 이유예요.

비유:
- Sigmoid = 10명이 전화로 메시지를 전달하는데, 각자 볼륨을 25%로 줄임 → 마지막 사람은 거의 안 들림
- ReLU = 각자 볼륨을 100% 유지 → 마지막 사람도 또렷하게 들림

---

**(d) 10층 Sigmoid 네트워크의 기울기 추정**

각 층의 기울기 최댓값이 0.25이면:

$$\text{첫 번째 층의 기울기} \leq 0.25^{10-1} = 0.25^9$$

$$0.25^9 = \frac{1}{4^9} = \frac{1}{262144} \approx \boxed{3.81 \times 10^{-6}}$$

**학습에 미치는 영향:**
- 학습률 $\eta = 0.01$이면, 실제 업데이트량 $\approx 0.01 \times 3.81 \times 10^{-6} = 3.81 \times 10^{-8}$
- 가중치가 $10^{-8}$ 단위로 바뀌므로, 사실상 **학습이 멈춤**
- 이것이 2000년대 이전에 "딥 네트워크는 학습시킬 수 없다"고 여겨졌던 이유!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** ReLU: $\max(0,x)$, Sigmoid: $1/(1+e^{-x})$, Tanh: $(e^x-e^{-x})/(e^x+e^{-x})$.

**(b)** ReLU': $\mathbf{1}_{x>0}$, Sigmoid': $\sigma(1-\sigma) \leq 0.25$, Tanh': $1-\tanh^2 \leq 1$.

**(c)** 역전파 시 기울기는 층별 Jacobian의 곱이다. $\prod_{k=1}^{L}\sigma'(z_k) \leq 0.25^L$이므로 $L$이 클 때 기울기가 지수적으로 감쇠. ReLU는 활성 영역에서 미분이 1이므로 감쇠 없음. 단, dead neuron 문제 존재.

**(d)** $0.25^9 \approx 3.81 \times 10^{-6}$. Gradient가 numerical precision 수준으로 떨어져 실질적 학습 불가. 이것이 ResNet의 skip connection, He initialization, batch normalization 등의 동기.

#### 🔴 Level 3: 대학생 눈높이

**(a-b)** 생략.

**(c)** Jacobian의 spectral norm 관점: $\|\frac{\partial h_{k+1}}{\partial h_k}\| = \|\text{diag}(\sigma'(z_k)) W_k\|$. Sigmoid에서 $\sigma' \leq 0.25$이므로 Jacobian의 spectral norm이 1 미만이 되기 쉬워 gradient가 기하급수적으로 감쇠. ReLU는 saturation이 없어 이를 완화하나, dying ReLU 문제가 있어 Leaky ReLU, ELU, GELU 등이 제안됨.

**(d)** $0.25^9 \approx 4 \times 10^{-6}$. Xavier/He initialization은 각 층의 Jacobian spectral norm을 1 근처로 유지하도록 설계. Residual connection은 gradient highway를 제공하여 이 문제를 근본적으로 해결: $\frac{\partial}{\partial x_l} = I + \frac{\partial F}{\partial x_l}$로 identity 경로가 gradient 직통 전달.

---

## 문제 29 (Batch Normalization)

### [EN] Problem Statement

**(a)** Write the batch normalization formula: given a mini-batch $\{x_1, \ldots, x_m\}$, compute $\mu_B$, $\sigma_B^2$, $\hat{x}_i$, and $y_i = \gamma\hat{x}_i + \beta$.

**(b)** Why does normalizing help training? (internal covariate shift argument)

**(c)** During inference, batch statistics are replaced by running averages. Explain why.

**(d)** What role do learnable parameters $\gamma$ and $\beta$ play?

### [KR] 문제

**(a)** 배치 정규화 공식을 작성하라: 미니배치 $\{x_1, \ldots, x_m\}$이 주어졌을 때, $\mu_B$, $\sigma_B^2$, $\hat{x}_i$, $y_i = \gamma\hat{x}_i + \beta$를 구하라.

**(b)** 정규화가 학습에 왜 도움이 되는가? (내부 공변량 이동(internal covariate shift) 논거)

**(c)** 추론(inference) 시에는 배치 통계가 이동 평균으로 대체된다. 그 이유를 설명하라.

**(d)** 학습 가능한 매개변수 $\gamma$와 $\beta$는 어떤 역할을 하는가?

### 출제 의도

배치 정규화는 현대 딥러닝의 **필수 기법**이다. 단순히 "정규화하면 좋다"가 아니라, **왜** 좋은지, **추론 시 왜 다르게 동작**하는지, $\gamma$와 $\beta$가 **왜 필요**한지까지 논리적으로 설명할 수 있는지를 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**배치 정규화(Batch Normalization, BN)란?** 신경망의 각 층에서 데이터를 "표준화"하는 기법이에요.

**표준화란?** 평균을 0으로, 표준편차를 1로 만드는 것. 시험 점수를 "편차값"으로 바꾸는 것과 같아요.

---

**(a) 공식**

미니배치 $\{x_1, x_2, \ldots, x_m\}$ ($m$개의 데이터)이 주어졌을 때:

**Step 1: 평균 구하기**

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

예: $x = [2, 4, 6]$이면 $\mu_B = (2+4+6)/3 = 4$

**Step 2: 분산 구하기**

$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

예: $\sigma_B^2 = ((2-4)^2 + (4-4)^2 + (6-4)^2)/3 = (4+0+4)/3 = 8/3$

**Step 3: 정규화하기**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$\epsilon \approx 10^{-5}$는 0으로 나누는 것을 방지하는 아주 작은 수예요.

예: $\hat{x}_1 = \frac{2 - 4}{\sqrt{8/3}} = \frac{-2}{1.633} \approx -1.22$

이제 $\hat{x}$의 평균은 약 0, 분산은 약 1이 됐어요!

**Step 4: 스케일과 이동 (Scale and Shift)**

$$y_i = \gamma \hat{x}_i + \beta$$

$\gamma$와 $\beta$는 **학습되는 매개변수**예요. 네트워크가 필요하면 정규화를 "되돌릴" 수 있게 해줘요.

---

**(b) 왜 도움이 되나?**

**내부 공변량 이동(Internal Covariate Shift)**:

비유로 설명할게요. 매일 다른 과목 시험을 보는데:
- 월요일: 국어 (평균 60, 표준편차 10)
- 화요일: 수학 (평균 30, 표준편차 20)

매일 점수의 "스케일"이 바뀌면, 공부 전략을 계속 바꿔야 해서 효율이 떨어져요.

신경망에서도 마찬가지예요. 앞 층의 가중치가 바뀌면, 뒷 층이 받는 입력의 분포가 매번 달라져요. 이것을 **내부 공변량 이동**이라 해요.

BN은 각 층의 입력을 항상 평균 0, 분산 1로 맞춰서, 뒷 층이 "안정적인" 입력을 받게 해요. → 학습이 더 빠르고 안정적!

(참고: 최근 연구에서는 BN의 효과가 ICS 완화보다는 loss landscape의 smoothing에 더 관련된다는 주장도 있어요.)

---

**(c) 추론 시 이동 평균을 사용하는 이유**

**문제**: 추론(inference) 시에는 데이터가 **한 개씩** 올 수 있어요. 한 개로는 평균과 분산을 의미있게 계산할 수 없어요!

**해결**: 학습 과정에서 각 미니배치의 $\mu_B$와 $\sigma_B^2$를 **이동 평균(running average)**으로 기록해둬요:

$$\mu_{\text{running}} \leftarrow \alpha \cdot \mu_{\text{running}} + (1-\alpha) \cdot \mu_B$$
$$\sigma^2_{\text{running}} \leftarrow \alpha \cdot \sigma^2_{\text{running}} + (1-\alpha) \cdot \sigma^2_B$$

여기서 $\alpha$는 보통 0.9 또는 0.99.

추론 시에는 이 이동 평균을 사용해요. 이렇게 하면:
- 입력이 1개여도 정규화 가능
- 같은 입력에 항상 **같은 출력** (결정론적, 배치 구성에 무관)

---

**(d) $\gamma$와 $\beta$의 역할**

"정규화하면 좋다"고 했는데, 왜 다시 $\gamma$와 $\beta$로 변환하나요?

$\hat{x}$는 항상 평균 0, 분산 1이에요. 하지만 **네트워크가 원하는 분포가 항상 평균 0, 분산 1은 아닐 수 있어요!**

예: sigmoid 활성화 함수는 입력이 0 근처일 때 거의 직선이에요. 정규화하면 모든 입력이 0 근처가 되어 sigmoid의 비선형성을 활용하지 못해요.

$\gamma$와 $\beta$가 있으면:
- $\gamma = \sigma_B$, $\beta = \mu_B$로 학습하면 → **정규화를 완전히 되돌릴 수 있음** (항등 변환)
- 네트워크가 스스로 "얼마나 정규화할지"를 결정할 수 있음
- 표현력(representational power)을 보존하면서도 정규화의 이점을 누릴 수 있음

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $\mu_B = \frac{1}{m}\sum x_i$, $\sigma_B^2 = \frac{1}{m}\sum(x_i-\mu_B)^2$, $\hat{x}_i = (x_i-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, $y_i = \gamma\hat{x}_i + \beta$.

**(b)** 층 간 전달되는 활성화의 분포가 학습 중 변동하면(internal covariate shift), 후속 층의 최적화가 불안정해진다. BN은 각 층 입력을 정규화하여 분포를 안정시킨다. 또한 loss landscape을 smooth하게 만들어 larger learning rate 사용이 가능해진다.

**(c)** 추론 시 단일 샘플 또는 가변 배치 크기에서 배치 통계가 불안정하므로, 학습 중 exponential moving average로 축적한 전체 데이터의 통계를 사용한다.

**(d)** $\gamma, \beta$는 affine transformation을 복원하여 정규화가 네트워크의 표현력을 제한하지 않도록 한다. $\gamma = \sigma_B, \beta = \mu_B$이면 identity mapping이 가능.

#### 🔴 Level 3: 대학생 눈높이

**(a)** 표준 BN 공식. $\epsilon$은 numerical stability.

**(b)** 원래 논문(Ioffe & Szegedy, 2015)의 ICS 가설은 이후 비판을 받았다(Santurkar et al., 2018). 실제 효과는 loss landscape의 Lipschitz smoothness 개선이 주요 원인으로, 이는 더 공격적인 learning rate와 더 빠른 수렴을 가능하게 한다. BN은 또한 implicit regularization 효과를 제공(배치 간 노이즈가 일종의 noise injection 역할).

**(c)** 추론 시 결정론적 출력이 필요하고, 단일 샘플에서 배치 통계 추정이 불가능하다. Running statistics는 전체 훈련 데이터 분포의 unbiased estimator에 수렴.

**(d)** $\gamma, \beta$가 없으면 BN이 각 층 활성화를 unit Gaussian으로 강제하여, 예를 들어 sigmoid의 선형 영역에만 활성화를 제한하게 된다. Learnable affine parameters는 네트워크가 최적의 활성화 분포를 자율적으로 선택하게 하면서도 정규화의 optimization benefit을 유지한다.

---

## 문제 30 (Comprehensive — From Data to Prediction)

### [EN] Problem Statement

A simple neural network for binary classification has: input $x \in \mathbb{R}$, one hidden layer with 2 neurons (ReLU activation), output layer with sigmoid. Weights: $w_1 = 1, w_2 = -1$ (hidden layer), $v_1 = 2, v_2 = 1$ (output layer), all biases $= 0$.

**(a)** For input $x = 2$, compute the forward pass step by step to get the prediction $p$.

**(b)** If the true label is $y = 1$, compute the binary cross-entropy loss: $L = -[y\log(p) + (1-y)\log(1-p)]$.

**(c)** Compute $\frac{\partial L}{\partial v_1}$ using the chain rule (show the full chain).

**(d)** Explain why this trained model can learn non-linear decision boundaries even though each operation is simple.

### [KR] 문제

이진 분류를 위한 간단한 신경망: 입력 $x \in \mathbb{R}$, 2개 뉴런의 은닉층(ReLU 활성화), sigmoid 출력층. 가중치: $w_1 = 1, w_2 = -1$ (은닉층), $v_1 = 2, v_2 = 1$ (출력층), 모든 편향 $= 0$.

**(a)** 입력 $x = 2$에 대해 순전파를 단계별로 계산하여 예측값 $p$를 구하라.

**(b)** 실제 라벨이 $y = 1$일 때, 이진 교차 엔트로피 손실 $L = -[y\log(p) + (1-y)\log(1-p)]$을 계산하라.

**(c)** 연쇄 법칙을 사용하여 $\frac{\partial L}{\partial v_1}$을 구하라 (전체 체인을 보여라).

**(d)** 각 연산이 단순한데도 이 모델이 비선형 결정 경계를 학습할 수 있는 이유를 설명하라.

### 출제 의도

이 문제는 **신경망의 전체 파이프라인**을 하나로 통합한다: 순전파 → 손실 계산 → 역전파. 개별 개념(ReLU, sigmoid, cross-entropy, chain rule)을 종합적으로 적용하는 능력, 그리고 신경망의 **표현력**에 대한 근본적 이해를 평가한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**신경망의 구조를 먼저 이해해요:**

```
입력 x=2
    ├── [×w₁=1] ──→ z₁ ──→ [ReLU] ──→ h₁ ──┐
    │                                          ├── [v₁h₁ + v₂h₂] ──→ a ──→ [sigmoid] ──→ p
    └── [×w₂=-1] ──→ z₂ ──→ [ReLU] ──→ h₂ ──┘
```

- 은닉층: 2개의 뉴런이 각각 $w_1$, $w_2$로 입력을 받고 ReLU를 적용
- 출력층: 은닉층 출력을 $v_1$, $v_2$로 가중합한 뒤 sigmoid 적용

---

**(a) 순전파 (Forward Pass)**

**Step 1: 은닉층 입력 계산**

$$z_1 = w_1 \cdot x = 1 \times 2 = 2$$
$$z_2 = w_2 \cdot x = (-1) \times 2 = -2$$

**Step 2: ReLU 적용**

$$h_1 = \text{ReLU}(z_1) = \text{ReLU}(2) = \max(0, 2) = 2$$
$$h_2 = \text{ReLU}(z_2) = \text{ReLU}(-2) = \max(0, -2) = 0$$

두 번째 뉴런은 "꺼졌어요" (음수 입력 → ReLU가 0 출력)!

**Step 3: 출력층 가중합**

$$a = v_1 \cdot h_1 + v_2 \cdot h_2 = 2 \times 2 + 1 \times 0 = 4 + 0 = 4$$

**Step 4: Sigmoid 적용**

$$p = \sigma(a) = \frac{1}{1 + e^{-4}} = \frac{1}{1 + e^{-4}}$$

$e^{-4} \approx 0.0183$이므로:

$$p = \frac{1}{1 + 0.0183} = \frac{1}{1.0183} \approx \boxed{0.9820}$$

모델이 98.2%의 확률로 "양성(1)"이라고 예측해요!

---

**(b) 이진 교차 엔트로피 손실 (Binary Cross-Entropy Loss)**

$y = 1$이므로:

$$L = -[y\log(p) + (1-y)\log(1-p)]$$
$$= -[1 \cdot \log(0.9820) + 0 \cdot \log(0.0180)]$$
$$= -\log(0.9820)$$

$\log$는 자연로그(밑이 $e$)예요.

$$\log(0.9820) \approx -0.01815$$

$$L = -(-0.01815) = \boxed{0.01815}$$

손실이 매우 작아요! 모델이 거의 맞게 예측했으니까 당연해요.

직관: 교차 엔트로피는 "정답이 1인데 0.982로 예측"하면 거의 0에 가깝고, "정답이 1인데 0.01로 예측"하면 매우 커요.

---

**(c) $\frac{\partial L}{\partial v_1}$ 계산 (역전파)**

연쇄 법칙: $v_1$이 $a$에 영향 → $a$가 $p$에 영향 → $p$가 $L$에 영향

$$\frac{\partial L}{\partial v_1} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial a} \cdot \frac{\partial a}{\partial v_1}$$

**Step 1: $\frac{\partial L}{\partial p}$**

$L = -\log(p)$ (∵ $y=1$)

$$\frac{\partial L}{\partial p} = -\frac{1}{p} = -\frac{1}{0.9820} \approx -1.0183$$

**Step 2: $\frac{\partial p}{\partial a}$**

$p = \sigma(a)$이고, sigmoid의 미분은 $\sigma(a)(1 - \sigma(a))$:

$$\frac{\partial p}{\partial a} = p(1-p) = 0.9820 \times 0.0180 \approx 0.01767$$

**Step 3: $\frac{\partial a}{\partial v_1}$**

$a = v_1 h_1 + v_2 h_2$이므로:

$$\frac{\partial a}{\partial v_1} = h_1 = 2$$

**전체 체인:**

$$\frac{\partial L}{\partial v_1} = (-1.0183) \times (0.01767) \times (2)$$

$$= -1.0183 \times 0.03534 \approx \boxed{-0.0360}$$

**검증**: 더 깔끔하게, $y=1$일 때 softmax+CE의 gradient는:

$$\frac{\partial L}{\partial v_1} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial a} \cdot h_1 = \left(-\frac{1}{p}\right) \cdot p(1-p) \cdot h_1 = -(1-p) \cdot h_1$$

$$= -(1 - 0.9820) \times 2 = -0.0180 \times 2 = -0.0360 \quad \checkmark$$

직관: 기울기가 음수 → $v_1$을 키우면 손실이 줄어든다. 모델이 이미 거의 맞게 예측하고 있으므로 업데이트 크기가 작다 (0.036).

---

**(d) 비선형 결정 경계를 학습할 수 있는 이유**

각 연산은 단순해요:
- 곱셈: 선형
- 덧셈: 선형
- ReLU: 조각적 선형 (piecewise linear)
- Sigmoid: 비선형이지만 단순한 곡선

**핵심**: ReLU가 입력 공간을 **영역별로 분할**해요!

우리 예시에서:
- $x > 0$이면: $h_1 = x$, $h_2 = 0$ → $a = 2x$ → $p = \sigma(2x)$
- $x < 0$이면: $h_1 = 0$, $h_2 = -x$ → $a = -x$ → $p = \sigma(-x)$

**입력 $x$의 부호에 따라 완전히 다른 함수가 적용돼요!** 이것은 더 이상 하나의 직선이 아니에요.

비유: 레고 블록 하나하나는 단순한 직사각형이지만, 조합하면 복잡한 성을 만들 수 있어요!

수학적으로: $n$개의 ReLU 뉴런이 있으면 입력 공간을 최대 $2^n$개의 선형 영역으로 분할할 수 있어요. 각 영역에서는 선형이지만, 전체적으로는 매우 복잡한 비선형 함수를 표현할 수 있어요. 이것이 **Universal Approximation Theorem**의 핵심 아이디어예요: 충분히 넓은 은닉층이 있으면 어떤 연속 함수든 근사할 수 있다!

#### 🟡 Level 2: 고등학생 눈높이

**(a)**
- $z_1 = 2, z_2 = -2$
- $h_1 = \text{ReLU}(2) = 2, h_2 = \text{ReLU}(-2) = 0$
- $a = 2(2) + 1(0) = 4$
- $p = \sigma(4) = 1/(1+e^{-4}) \approx 0.9820$

**(b)** $L = -\log(0.9820) \approx 0.01815$

**(c)** $\frac{\partial L}{\partial v_1} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial a} \cdot \frac{\partial a}{\partial v_1} = (-1/p) \cdot p(1-p) \cdot h_1 = -(1-p)h_1 = -0.018 \times 2 = -0.036$.

여기서 sigmoid + BCE의 gradient가 $p - y$로 단순화되는 것을 활용: $\frac{\partial L}{\partial a} = p - y = 0.982 - 1 = -0.018$, $\frac{\partial L}{\partial v_1} = (p-y) \cdot h_1 = -0.036$.

**(d)** ReLU의 piecewise linearity가 입력 공간을 다수의 선형 영역으로 분할한다. 각 영역에서의 affine 함수 조합으로 전체적인 비선형 함수를 구성. Hidden neurons의 수가 많을수록 더 복잡한 decision boundary 표현 가능.

#### 🔴 Level 3: 대학생 눈높이

**(a)** $p = \sigma(v^\top \text{ReLU}(wx)) = \sigma(4) \approx 0.982$.

**(b)** $L = -\log\sigma(4) = \log(1+e^{-4}) \approx 0.018$.

**(c)** $\frac{\partial L}{\partial v_1} = (\sigma(a) - y) \cdot h_1 = (p-1) \cdot 2 = -0.036$. BCE + sigmoid의 gradient가 $p - y$로 단순화되는 것은 exponential family의 canonical form에서 유도된다.

**(d)** Piecewise linear activation(ReLU)의 composition은 exponentially many linear regions를 생성한다(Montufar et al., 2014). 깊이 $L$, 폭 $n$의 ReLU network는 최대 $O\left(\left(\frac{n}{d}\right)^{(L-1)d} n^d\right)$개의 선형 영역을 가진다($d$는 입력 차원). 이는 깊은 네트워크가 얕은 네트워크보다 지수적으로 더 표현력이 강함을 의미한다. Universal Approximation Theorem(Cybenko, 1989; Hornik, 1991)은 단일 은닉층으로 임의의 연속함수 근사가 가능함을 보장하지만, 효율적 근사를 위해서는 depth가 필수적이다.

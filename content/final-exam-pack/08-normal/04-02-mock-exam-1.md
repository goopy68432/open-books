---
title: "딥러닝 이론 기말 모의고사 #1"
slug: 02-mock-exam-1
order: 4
---

# 딥러닝 이론 기말 모의고사 #1

> **시험 형식**: 주관식 (퀴즈 유사), 답 도출 논리 과정 서술
> **시험 시간**: 90분
> **배점**: 100점 (10문제)
> **주의**: 답만 적으면 0점. 모든 유도의 각 단계에 "왜"를 글로 설명해야 점수

---

## 문제 1. [10점] Gaussian → MSE 유도

다음을 증명하시오.

> 관측 모델 $y_i = h_\theta(x_i) + \epsilon_i$에서 $\epsilon_i \stackrel{i.i.d.}{\sim} \mathcal{N}(0, \sigma^2)$일 때, Maximum Likelihood Estimation은 MSE 최소화와 동치이다.

**(a)** [6점] i.i.d. 가정에서 출발하여 $\theta_{ML} = \arg\min_\theta \text{MSE}$를 유도하시오. 각 단계에서 "왜 이 조작을 하는가"를 반드시 서술하시오.

**(b)** [4점] 이 유도에서 Central Limit Theorem(CLT)은 어떤 역할을 하는가? "가우시안 노이즈 가정"을 정당화하는 논리를 서술하시오.

---

## 문제 2. [10점] Bernoulli → Cross-Entropy 유도

**(a)** [7점] K-클래스 분류에서 $y_i \sim \text{Cat}(h_\theta(x_i))$로 모델링할 때, MLE를 통해 Cross-Entropy Loss가 유도되는 과정을 완전히 서술하시오.

**(b)** [3점] 이진 분류($K=2$)일 때 CE Loss를 $\theta = \sigma(w^\top x)$를 사용하여 구체적으로 전개하시오. (여기서 $\sigma$는 시그모이드 함수)

---

## 문제 3. [10점] MAP과 Regularization

**(a)** [5점] MAP 추정의 정의를 베이즈 정리로부터 유도하고, MLE와의 차이를 수식으로 설명하시오.

**(b)** [5점] Prior $P(\theta) = \mathcal{N}(0, \sigma_p^2 I)$일 때, MAP가 L2 Regularization과 동치임을 유도하시오. 정규화 상수 $\lambda$와 $\sigma_p$의 관계를 명시하시오.

---

## 문제 4. [10점] Softmax 함수

**(a)** [6점] 라그랑주 승수법을 사용하여 softmax 함수를 유도하시오.
- 목적함수: $\max_p \sum_i p_i z_i + \tau H(p)$
- 제약조건: $\sum_i p_i = 1$, $p_i \geq 0$

**(b)** [4점] temperature $\tau$가 $\tau \to 0$과 $\tau \to \infty$일 때 softmax의 동작을 각각 설명하시오.

---

## 문제 5. [10점] Softmax Jacobian

$p_i = \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$일 때,

**(a)** [7점] $\frac{\partial p_i}{\partial z_j}$를 $i=j$인 경우와 $i \neq j$인 경우로 나누어 유도하시오. 왜 두 경우를 나누어야 하는지 설명하시오.

**(b)** [3점] 결과를 행렬 형태 $\frac{\partial p}{\partial z} = ?$로 통합하여 쓰시오.

---

## 문제 6. [10점] 선형대수 기본정리

**(a)** [4점] 행렬 $A \in \mathbb{R}^{m \times n}$의 Image(Range) $\mathscr{R}(A)$와 Kernel(Null Space) $\mathscr{N}(A)$를 정의하고, 각각의 딥러닝에서의 의미를 간략히 설명하시오.

**(b)** [6점] Rank-Nullity 정리 $n = \text{rank}(A) + \text{nullity}(A)$를 증명하시오.

---

## 문제 7. [10점] 고유값과 SVD

**(a)** [3점] 정방행렬 $A$의 고유값/고유벡터 정의를 쓰고, 대칭행렬에서 서로 다른 고유값에 대응하는 고유벡터가 직교함을 증명하시오.

**(b)** [4점] SVD $A = U\Sigma V^\top$에서 $U, \Sigma, V$의 의미를 설명하고, 저랭크 근사(Low-rank Approximation)와의 관계를 서술하시오.

**(c)** [3점] Google PageRank에서 고유벡터가 어떻게 활용되는지 간략히 서술하시오.

---

## 문제 8. [10점] 벡터 미분과 체인룰

**(a)** [3점] 다음의 결과를 구하시오 (풀이 과정 포함):
- (i) $\frac{\partial}{\partial x}(a^\top x)$
- (ii) $\frac{\partial}{\partial x}(x^\top S x)$ (S는 대칭행렬)
- (iii) $\nabla_W \|Wx - y\|^2$

**(b)** [4점] 체인룰을 행렬 형태로 서술하고, 이것이 역전파(Backpropagation)의 수학적 기초인 이유를 설명하시오.

**(c)** [3점] VJP(Vector-Jacobian Product)가 JVP(Jacobian-Vector Product)보다 역전파에 적합한 이유를 차원 분석으로 설명하시오.

---

## 문제 9. [10점] 정보이론

**(a)** [3점] 엔트로피 $H(p)$, 교차 엔트로피 $CE(p,q)$, KL 발산 $KL(p\|q)$를 각각 정의하고, 세 가지의 관계식을 쓰시오.

**(b)** [4점] Jensen 부등식을 사용하여 $KL(p\|q) \geq 0$을 증명하시오. (왜 $-\log$에 Jensen을 적용할 수 있는지 명시)

**(c)** [3점] $H(p)$가 상수일 때, "CE 최소화 = KL 최소화 = NLL 최소화"가 성립하는 이유를 설명하시오.

---

## 문제 10. [10점] 통합 문제

다음의 등가 관계 체인을 수학적으로 설명하시오.

$$\text{Gaussian Prior의 MAP} = \text{NLL} + \text{L2 Reg} = \text{MSE} + \lambda\|\theta\|^2 = \text{Ridge Regression}$$

**(a)** [4점] 왼쪽에서 오른쪽으로 유도하는 과정을 빠짐없이 서술하시오.

**(b)** [3점] 만약 Prior를 Laplace 분포로 바꾸면 어떤 Regularization이 되는가? 이 경우 L1과 L2의 차이점(특히 sparsity 관점에서)을 설명하시오.

**(c)** [3점] Linear Model, Neural Network, CNN, Transformer를 inductive bias의 강도 순으로 나열하고, 데이터 양에 따른 적합성을 설명하시오.

---
---

# 모의고사 #1 — 모범답안

---

## 답 1. Gaussian → MSE 유도

### (a)

**Step 1 — 확률 모델**: $\epsilon_i \sim \mathcal{N}(0,\sigma^2)$이므로
$$p(y_i|x_i,\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - h_\theta(x_i))^2}{2\sigma^2}\right)$$

**Step 2 — i.i.d. 가정** (왜: 각 관측이 독립동일분포이므로 결합확률이 곱으로 분해됨)
$$P(\mathbf{y}|X,\theta) = \prod_{i=1}^n p(y_i|x_i,\theta)$$

**Step 3 — 로그 변환** (왜: (1) 곱→합으로 계산 편의, (2) 컴퓨터 수치 언더플로 방지, (3) log는 단조증가이므로 argmax 불변)
$$\log P = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - h_\theta(x_i))^2$$

**Step 4 — NLL 최소화** (왜: MLE는 로그우도 최대화 = NLL 최소화)
$$\text{NLL} = \frac{1}{2\sigma^2}\sum_i(y_i - h_\theta(x_i))^2 + C$$

**Step 5 — θ에 무관한 항 제거** (왜: $\sigma^2$와 상수는 $\theta$의 함수가 아니므로 최적화에 영향 없음)
$$\theta_{ML} = \arg\min_\theta \sum_i(y_i - h_\theta(x_i))^2 = \arg\min_\theta \text{MSE} \quad \square$$

### (b)
CLT에 의하면, 수많은 독립적 요인의 합은 정규분포에 수렴한다. 현실의 관측 노이즈는 센서 오차, 환경 변동 등 무수히 많은 독립 원인의 합이므로, CLT가 $\epsilon_i \sim \mathcal{N}(0,\sigma^2)$ 가정의 이론적 근거가 된다. 따라서 가우시안 노이즈 가정 → MSE 사용은 CLT에 의해 정당화된다.

---

## 답 2. Bernoulli → CE 유도

### (a)
**Step 1**: $y_i \in \{1,...,K\}$이고 모델 출력 $h_\theta(x_i) \in \Delta^K$ (확률 심플렉스).
$$p(y_i|x_i,\theta) = [h_\theta(x_i)]_{y_i} = \prod_{c=1}^K [h_\theta(x_i)]_c^{\mathbb{1}[y_i=c]}$$

**Step 2**: i.i.d. → $P(\mathbf{y}|X,\theta) = \prod_{i=1}^n [h_\theta(x_i)]_{y_i}$
(왜: 각 데이터 포인트가 독립이므로 결합확률 = 개별 확률의 곱)

**Step 3**: 로그 (왜: 곱→합, 수치안정성, 단조함수)
$$\log P = \sum_{i=1}^n \log [h_\theta(x_i)]_{y_i}$$

**Step 4**: NLL (왜: 부호 반전으로 최소화 문제로 변환)
$$\text{NLL} = -\sum_{i=1}^n \log [h_\theta(x_i)]_{y_i}$$

내적 표현: $= -\sum_i e_{y_i}^\top \log h_\theta(x_i) = \text{CE}(p_{\text{data}}, h_\theta)$ $\square$

### (b)
이진 분류: $K=2$, $\theta = \sigma(w^\top x)$
$$\text{CE} = -\sum_i [y_i \log \sigma(w^\top x_i) + (1-y_i)\log(1-\sigma(w^\top x_i))]$$
이것은 Binary Cross-Entropy (BCE)이며, Bernoulli 분포 $\text{Ber}(y;\theta) = \theta^y(1-\theta)^{1-y}$의 NLL과 정확히 동일하다.

---

## 답 3. MAP과 Regularization

### (a)
베이즈 정리: $P(\theta|\text{data}) = \frac{P(\text{data}|\theta)P(\theta)}{P(\text{data})}$

MAP는 사후확률의 최대화:
$$\theta_{MAP} = \arg\max_\theta P(\theta|\text{data}) = \arg\max_\theta P(\text{data}|\theta)P(\theta)$$
(왜 $P(\text{data})$ 무시: θ에 무관한 상수)

로그: $\theta_{MAP} = \arg\max_\theta [\log P(\text{data}|\theta) + \log P(\theta)]$

**MLE와의 차이**: MLE는 $\log P(\theta)$ 항이 없다 (= Uniform Prior, $P(\theta) = \text{const}$).
$$\theta_{ML} = \arg\max_\theta \log P(\text{data}|\theta)$$

### (b)
$P(\theta) = \mathcal{N}(0, \sigma_p^2 I)$이면:
$$\log P(\theta) = -\frac{1}{2\sigma_p^2}\|\theta\|_2^2 + C'$$

MAP를 최소화로:
$$\theta_{MAP} = \arg\min_\theta \left[\text{NLL}(\theta) + \frac{1}{2\sigma_p^2}\|\theta\|_2^2\right]$$

$\lambda = \frac{1}{2\sigma_p^2}$로 정의하면:
$$\theta_{MAP} = \arg\min_\theta [\text{Loss}(\theta) + \lambda\|\theta\|_2^2] \quad \square$$

$\sigma_p$가 작을수록 → $\lambda$가 클수록 → 정규화가 강함 → 가중치를 0에 가깝게 억제.

---

## 답 4. Softmax 함수

### (a)
라그랑지안: $\mathcal{L} = \sum_i p_i z_i - \tau \sum_i p_i \log p_i + \mu(1 - \sum_i p_i)$
(왜 라그랑주: $\sum p_i = 1$ 등식 제약)

$\frac{\partial \mathcal{L}}{\partial p_i} = z_i - \tau(\log p_i + 1) - \mu = 0$ (왜 미분=0: Fermat, 내부 극값 필요조건)

$\log p_i = (z_i - \mu - \tau)/\tau$ → $p_i = \exp(z_i/\tau) \cdot \exp((-\mu-\tau)/\tau)$

$\sum p_i = 1$ 적용 (왜: 확률 공리):
$$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)} \quad \square$$

### (b)
- $\tau \to 0$: $\exp(z_i/\tau)$에서 최대 $z_i$만 살아남 → one-hot (argmax)
- $\tau \to \infty$: $z_i/\tau \to 0$ → 모든 $\exp \approx 1$ → uniform 분포

---

## 답 5. Softmax Jacobian

### (a)
$p_i = e^{z_i}/s$, $s = \sum_k e^{z_k}$

**왜 두 경우를 나누는가**: $i=j$일 때 분자 $e^{z_i}$가 $z_j(=z_i)$에 의존하므로 분자+분모 모두 미분해야 하고(몫의 미분법), $i \neq j$일 때는 분자 미분이 0이므로 더 단순하다.

**$i=j$**: $\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} s - e^{z_i} e^{z_i}}{s^2} = \frac{e^{z_i}}{s}(1 - \frac{e^{z_i}}{s}) = p_i(1-p_i)$

**$i \neq j$**: $\frac{\partial p_i}{\partial z_j} = \frac{0 - e^{z_i} e^{z_j}}{s^2} = -p_i p_j$

### (b)
$$\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top$$

---

## 답 6. 선형대수 기본정리

### (a)
- $\mathscr{R}(A) = \{Av : v \in \mathbb{R}^n\}$: A의 출력이 도달할 수 있는 공간 (= 가능한 출력 집합)
- $\mathscr{N}(A) = \{v : Av = 0\}$: A에 의해 0으로 보내지는 입력 방향 (= 정보 손실 방향)

DL 의미: Rank = 네트워크가 표현할 수 있는 차원의 수, Nullity = 구별할 수 없는 입력 방향의 수

### (b)
(유도 #6 참조 — 전체 증명 생략 없이 서술)

---

## 답 7. 고유값과 SVD

### (a)
정의: $Av = \lambda v$ ($v \neq 0$). λ가 고유값, v가 고유벡터.

직교 증명: $\lambda_1 u_1^\top u_2 = (Au_1)^\top u_2 = u_1^\top A^\top u_2 = u_1^\top Au_2 = \lambda_2 u_1^\top u_2$
(왜 $A^\top = A$: 대칭 가정)
$(\lambda_1 - \lambda_2)u_1^\top u_2 = 0$, $\lambda_1 \neq \lambda_2 \Rightarrow u_1^\top u_2 = 0$ $\square$

### (b)
$A = U\Sigma V^\top$: U는 출력 공간 직교기저, V는 입력 공간 직교기저, Σ는 특이값(중요도).
저랭크 근사: 상위 r개 특이값만 유지 → $A_r = \sum_{i=1}^r \sigma_i u_i v_i^\top$ → 최적 랭크-r 근사 (Eckart-Young 정리).

### (c)
웹을 거대한 전이 행렬로 모델링 → Power Method로 주요 고유벡터 추출 → 이 고유벡터의 각 원소가 해당 페이지의 중요도(PageRank).

---

## 답 8. 벡터 미분과 체인룰

### (a)
(i) $\frac{\partial}{\partial x}(a^\top x) = a^\top$ (선형함수의 미분 = 계수)
(ii) $\frac{\partial}{\partial x}(x^\top Sx) = 2x^\top S$ (이차형식의 미분, S 대칭)
(iii) $\nabla_W\|Wx-y\|^2 = 2(Wx-y)x^\top$ (체인룰 + 외적)

### (b)
$$\frac{\partial(f \circ g)}{\partial x} = \frac{\partial f}{\partial y}\bigg|_{y=g(x)} \cdot \frac{\partial g}{\partial x}$$
딥러닝의 순전파: $x \to h_1 \to h_2 \to ... \to L$ (합성함수 체인)
역전파: 끝($\partial L/\partial h_n$)에서 시작하여 체인룰을 **역순으로** 곱해나감.

### (c)
출력 $u \in \mathbb{R}^m$, 입력 $v \in \mathbb{R}^n$일 때 Jacobian $J \in \mathbb{R}^{m \times n}$.
- JVP: $J \cdot \Delta v$ → $\mathbb{R}^m$ (입력 변화 → 출력 변화, forward)
- VJP: $\Delta u^\top \cdot J$ → $\mathbb{R}^n$ (출력 그래디언트 → 입력 그래디언트, backward)

최종 loss는 **스칼라 1개** → VJP로 한 번의 backward pass에 **모든 파라미터 그래디언트** 계산 가능. JVP는 파라미터마다 forward pass 필요 → 비효율.

---

## 답 9. 정보이론

### (a)
- $H(p) = -\sum_x p(x)\log p(x)$ (분포 p의 불확실성)
- $CE(p,q) = -\sum_x p(x)\log q(x)$ (p 기준으로 q를 사용하는 비용)
- $KL(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$ (p와 q의 차이)

관계: $\boxed{KL(p\|q) = CE(p,q) - H(p)}$

### (b)
$$KL(p\|q) = -\sum_x p(x)\log\frac{q(x)}{p(x)}$$
$-\log$는 볼록(convex)이므로 Jensen 부등식 적용 (왜 적용 가능: $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$, f 볼록):
$$\geq -\log\left(\sum_x p(x)\frac{q(x)}{p(x)}\right) = -\log\left(\sum_x q(x)\right) = -\log 1 = 0 \quad \square$$

### (c)
$KL(p\|q) = CE(p,q) - H(p)$에서 p가 데이터 분포(고정)이면 $H(p)$는 상수.
따라서 $\min_q CE(p,q) = \min_q KL(p\|q)$.
또한 CE = NLL (유도 #2에서 CE가 NLL임을 보였으므로).
결론: $\min \text{CE} = \min \text{KL} = \min \text{NLL}$.

---

## 답 10. 통합 문제

### (a)
1. MAP: $\theta_{MAP} = \arg\max_\theta [\log P(D|\theta) + \log P(\theta)]$
2. Gaussian Prior: $\log P(\theta) = -\frac{1}{2\sigma_p^2}\|\theta\|^2 + C$
3. Gaussian Likelihood (회귀): $-\log P(D|\theta) = \frac{1}{2\sigma^2}\sum(y_i-h(x_i))^2 + C'$
4. 부호 반전 → 최소화: $\arg\min[\text{MSE} + \frac{1}{2\sigma_p^2}\|\theta\|^2]$
5. $\lambda = 1/(2\sigma_p^2)$: $\arg\min[\text{MSE} + \lambda\|\theta\|^2] = \text{Ridge Regression}$ $\square$

### (b)
Laplace Prior: $P(\theta) \propto \exp(-\|\theta\|_1/b)$ → $\log P(\theta) = -\|\theta\|_1/b + C$ → **L1 Regularization = LASSO**

L1 vs L2:
- L1: 미분 불연속점(0)이 존재 → 가중치를 정확히 0으로 만듦 → **sparsity** 유도
- L2: 미분이 연속 → 가중치를 0 근처로 줄이지만 정확히 0이 되진 않음 → 전체적 축소

### (c)
Inductive bias 강도: **Linear > CNN > NN(MLP) > Transformer**
- Linear: 가장 강한 bias (선형 가정) → 데이터 적을 때 유리, 표현력 제한
- CNN: 이미지 특화 prior (locality, translation equivariance) → 중간 데이터
- NN(MLP): 약한 bias, 높은 표현력 → 중간~많은 데이터
- Transformer: 가장 약한 prior → **대량 데이터 필수** (ViT의 교훈: JFT-300M 필요)

핵심 트레이드오프: bias 강하면 적은 데이터에서 유리하지만 표현력 제한, 약하면 표현력은 높지만 과적합 위험. 데이터가 충분하면 약한 bias가 승리한다.

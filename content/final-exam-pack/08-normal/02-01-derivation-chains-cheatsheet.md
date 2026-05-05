---
title: "딥러닝 이론 기말 — 10대 핵심 유도 체인 완전 재현 시트"
slug: 01-derivation-chains-cheatsheet
order: 2
---

# 딥러닝 이론 기말 — 10대 핵심 유도 체인 완전 재현 시트

> **사용법**: 시험 직전에 이 유도들을 **백지에서 처음부터 끝까지 재현**할 수 있으면 A+
> **채점 핵심**: 각 단계마다 **"왜 이 단계를 밟는가"**를 반드시 서술

---

## 유도 #1: Gaussian 가정 → MSE Loss

### 스토리 한 줄
"회귀에서 노이즈가 정규분포를 따른다고 가정하면, MLE는 자연스럽게 MSE 최소화가 된다."

### 완전 유도

**출발**: 관측 모델 $y_i = h_\theta(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

**Step 1 — 확률 모델 설정** (왜: 노이즈가 가우시안이므로)
$$p(y_i | x_i, \theta) = \mathcal{N}(y_i; h_\theta(x_i), \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - h_\theta(x_i))^2}{2\sigma^2}\right)$$

**Step 2 — i.i.d. 가정으로 결합확률을 곱으로 분해** (왜: 각 데이터가 독립동일분포)
$$P(\mathbf{y}|X,\theta) = \prod_{i=1}^n p(y_i|x_i,\theta)$$

**Step 3 — 로그를 취한다** (왜: (1) 곱→합 변환, (2) 수치 언더플로 방지, (3) 단조증가이므로 최적값 불변)
$$\log P = \sum_{i=1}^n \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - h_\theta(x_i))^2}{2\sigma^2}\right]$$

**Step 4 — NLL(Negative Log-Likelihood) 최소화** (왜: 최대화 → 부호 반전하여 최소화로 변환)
$$\text{NLL} = \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - h_\theta(x_i))^2 + \text{const}$$

**Step 5 — σ와 상수는 θ와 무관하므로 제거** (왜: 최적화 변수 θ에 영향 없음)
$$\boxed{\theta_{ML} = \arg\min_\theta \sum_{i=1}^n (y_i - h_\theta(x_i))^2 = \arg\min_\theta \text{MSE}}$$

### 결론
가우시안 가정 하의 MLE = MSE 최소화. **MSE를 쓰는 것은 노이즈가 정규분포라는 암묵적 가정이다.**

---

## 유도 #2: Bernoulli/Categorical 가정 → Cross-Entropy Loss

### 스토리 한 줄
"분류에서 출력이 카테고리 분포를 따른다고 가정하면, MLE는 Cross-Entropy 최소화가 된다."

### 완전 유도

**출발**: 분류 모델 $p(y_i = c | x_i) = [h_\theta(x_i)]_c$ (softmax 출력의 c번째 원소)

**Step 1 — Categorical 분포 설정** (왜: 분류는 이산 확률 분포)
$$p(y_i|x_i,\theta) = \prod_{c=1}^C [h_\theta(x_i)]_c^{\mathbb{1}[y_i=c]} = [h_\theta(x_i)]_{y_i}$$

**Step 2 — i.i.d. → 곱** (왜: 독립 관측)
$$P(\mathbf{y}|X,\theta) = \prod_{i=1}^n [h_\theta(x_i)]_{y_i}$$

**Step 3 — 로그** (왜: 곱→합, 수치안정, 단조증가)
$$\log P = \sum_{i=1}^n \log [h_\theta(x_i)]_{y_i}$$

**Step 4 — NLL 최소화** (왜: 최대화의 부호 반전)
$$\boxed{\text{NLL} = -\sum_{i=1}^n \log [h_\theta(x_i)]_{y_i} = \text{Cross-Entropy Loss}}$$

### 내적 표현
$e_{y_i}$를 정답 클래스의 one-hot 벡터라 하면:
$$\text{CE} = -\sum_i e_{y_i}^\top \log h_\theta(x_i)$$

### 결론
CE를 쓰는 것은 출력이 카테고리 분포라는 가정이다. **MSE vs CE의 차이 = 가우시안 vs 카테고리 가정의 차이.**

---

## 유도 #3: MAP + Gaussian Prior → L2 Regularization

### 스토리 한 줄
"가중치에 가우시안 사전분포를 가정한 MAP 추정은 L2 정규화(Weight Decay)와 수학적으로 동치다."

### 완전 유도

**출발**: MAP = Likelihood × Prior 최대화

**Step 1 — MAP 정의** (왜: 사후확률 최대화 = Bayes 정리의 분자 최대화)
$$\theta_{MAP} = \arg\max_\theta \log P(\theta|\text{data}) = \arg\max_\theta [\log P(\text{data}|\theta) + \log P(\theta)]$$
(왜 Evidence $P(\text{data})$ 무시: θ에 독립인 상수)

**Step 2 — Gaussian Prior 설정** (왜: 가중치가 0 근처에 있을 것이라는 사전 믿음)
$$P(\theta) = \mathcal{N}(0, \sigma_p^2 I) \quad \Rightarrow \quad \log P(\theta) = -\frac{1}{2\sigma_p^2}\|\theta\|_2^2 + C$$

**Step 3 — MAP를 최소화 문제로 변환** (왜: 부호 반전)
$$\theta_{MAP} = \arg\min_\theta \left[\underbrace{-\log P(\text{data}|\theta)}_{\text{NLL (= Loss)}} + \underbrace{\frac{1}{2\sigma_p^2}\|\theta\|_2^2}_{\text{L2 Regularization}}\right]$$

**Step 4 — λ = 1/(2σ_p²)로 정의** (왜: 하이퍼파라미터로 통합)
$$\boxed{\theta_{MAP} = \arg\min_\theta [\text{Loss}(\theta) + \lambda\|\theta\|_2^2]}$$

### 등가 관계 체인
$$\text{MAP (Gaussian Prior)} = \text{L2 Reg} = \text{Weight Decay} = \text{Ridge}$$
$$\text{MAP (Laplace Prior)} = \text{L1 Reg} = \text{LASSO} = \text{Sparsity}$$

---

## 유도 #4: Lagrangian → Softmax 함수

### 스토리 한 줄
"확률의 합=1 제약 하에서 기대값+엔트로피를 최대화하면 softmax가 유일한 해다."

### 완전 유도

**문제**: $\max_p \sum_i p_i z_i + \tau H(p)$ s.t. $\sum_i p_i = 1$, $p_i \geq 0$
(왜 이 문제: logit $z_i$를 확률로 변환하되, 불확실성(엔트로피)도 적당히 유지)

**Step 1 — 라그랑지안** (왜: 등식 제약이 있으므로 라그랑주 승수법)
$$\mathcal{L} = \sum_i p_i z_i - \tau \sum_i p_i \log p_i + \lambda(1 - \sum_i p_i)$$

**Step 2 — ∂L/∂p_i = 0** (왜: Fermat 정리, 내부 극값의 필요조건)
$$z_i - \tau(\log p_i + 1) - \lambda = 0$$

**Step 3 — p_i에 대해 풀기**
$$\log p_i = \frac{z_i - \lambda - \tau}{\tau} \quad \Rightarrow \quad p_i = \frac{1}{Z}\exp\left(\frac{z_i}{\tau}\right)$$

**Step 4 — 정규화** (왜: $\sum p_i = 1$은 확률의 공리)
$$Z = \sum_j \exp(z_j/\tau)$$

$$\boxed{p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)} = \text{softmax}(z/\tau)_i}$$

**τ의 의미**: $\tau \to 0$ → argmax (one-hot), $\tau \to \infty$ → uniform

---

## 유도 #5: Softmax Jacobian

### 완전 유도

$p_i = e^{z_i} / s$, 여기서 $s = \sum_k e^{z_k}$

**Case 1: i = j** (왜 구분: 분자 $e^{z_i}$가 $z_i$에 의존하므로)
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \cdot s - e^{z_i} \cdot e^{z_i}}{s^2} = p_i - p_i^2 = p_i(1-p_i)$$

**Case 2: i ≠ j** (왜 구분: 분자 $e^{z_i}$가 $z_j$에 무관 → 분자 미분 = 0)
$$\frac{\partial p_i}{\partial z_j} = \frac{0 - e^{z_i} \cdot e^{z_j}}{s^2} = -p_i p_j$$

**통합**: $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$

$$\boxed{\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top}$$

---

## 유도 #6: Rank-Nullity 정리

$A \in \mathbb{R}^{m \times n}$일 때: $\boxed{n = \text{rank}(A) + \text{nullity}(A)}$

**Step 1**: $\mathscr{N}(A)$의 기저 $\{v_1,...,v_k\}$를 잡는다 ($k$ = nullity)
**Step 2**: 이를 확장하여 $\mathbb{R}^n$의 기저 $\{v_1,...,v_k, u_1,...,u_r\}$ ($r = n-k$)
**Step 3**: $\{Au_1,...,Au_r\}$이 $\mathscr{R}(A)$의 기저임을 보인다
- 생성: 임의의 $Ax = \sum a_i Av_i + \sum b_j Au_j = \sum b_j Au_j$ ($Av_i = 0$이므로)
- 선형독립: $\sum c_j Au_j = 0 \Rightarrow A(\sum c_j u_j) = 0 \Rightarrow \sum c_j u_j \in \mathscr{N}(A)$
  → $v_i$들의 선형결합이어야 하는데, $u_j$들과 $v_i$들은 독립 → 모든 $c_j = 0$
**Step 4**: $\text{rank} = r = n - k$. $\square$

---

## 유도 #7: 대칭행렬의 스펙트럼 분해

**핵심 정리**: 대칭행렬 $A = A^\top$이면 $A = U\Lambda U^\top = \sum_i \lambda_i u_i u_i^\top$

**핵심 성질 (증명 키포인트)**:
1. 대칭행렬의 고유값은 실수 (왜: $\lambda\|v\|^2 = v^*Av = \overline{v^*Av} = \bar{\lambda}\|v\|^2$)
2. 서로 다른 고유값의 고유벡터는 직교 (왜: $\lambda_1 u_1^\top u_2 = (Au_1)^\top u_2 = u_1^\top Au_2 = \lambda_2 u_1^\top u_2$, $\lambda_1 \neq \lambda_2 \Rightarrow u_1^\top u_2 = 0$)

---

## 유도 #8: 베이즈 정리

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$$

**Step 1**: 조건부 확률 정의: $P(H|E) = P(H \cap E) / P(E)$
**Step 2**: 대칭: $P(H \cap E) = P(E|H) \cdot P(H)$
**Step 3**: 대입: $P(H|E) = P(E|H) \cdot P(H) / P(E)$ $\square$

---

## 유도 #9: KL Divergence ≥ 0 (Gibbs' inequality)

$$KL(p\|q) = -\sum_x p(x) \log \frac{q(x)}{p(x)} \geq -\log \sum_x p(x) \frac{q(x)}{p(x)} = -\log 1 = 0$$

**왜 Jensen**: $-\log$는 **볼록(convex)**함수이므로 Jensen 부등식 $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$ 적용 가능

---

## 유도 #10: 정규방정식 (Normal Equation)

**문제**: $\min_w \|Xw - y\|^2$

**Step 1 — 전개**:
$$\|Xw-y\|^2 = (Xw-y)^\top(Xw-y) = w^\top X^\top X w - 2y^\top X w + y^\top y$$

**Step 2 — ∇_w = 0** (왜: quadratic이므로 볼록, 미분=0이 최솟값의 충분조건):
$$\nabla_w = 2X^\top X w - 2X^\top y = 0$$

**Step 3**:
$$\boxed{X^\top X \hat{w} = X^\top y \quad \Rightarrow \quad \hat{w} = (X^\top X)^{-1} X^\top y}$$

(왜 역행렬 존재 필요: $X^\top X$가 가역이 아니면 해가 유일하지 않음 → 정규화 필요)

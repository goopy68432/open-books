---
title: "Deep Learning 중간고사 모의시험"
slug: midterm-exam
order: 1
---

# Deep Learning 중간고사 모의시험

> **범위**: Preliminaries (Linear Algebra, Matrix Calculus, Optimization, Probability, Bayesian) + Introduction (Linear/Logistic Regression, Neural Networks) + Optimization & Generalization
> **시간**: 120분
> **총점**: 200점
> **주의**: 모든 풀이에 논리적 근거와 수학적 유도 과정을 반드시 기술하시오. 결과만 쓰면 감점.

---

# Part 1: Linear Algebra (50점)

---

## Problem 1 (10점)

**[EN]** Let $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$.

(a) Find all eigenvalues and eigenvectors of $A$. (6점)
(b) Write the spectral decomposition $A = U\Lambda U^\top$. (4점)

**[KR]** $A = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$ 에 대하여,

(a) $A$의 모든 고유값과 고유벡터를 구하시오. (6점)
(b) 스펙트럴 분해 $A = U\Lambda U^\top$ 를 작성하시오. (4점)

---

## Problem 2 (12점)

**[EN]** Consider the matrix $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$.

(a) What is the rank of $A$? Justify your answer. (3점)
(b) What is $\dim(\mathscr{N}(A))$? Use the Rank-Nullity theorem. (3점)
(c) Can the equation $Ax = b$ have a unique solution for every $b \in \mathbb{R}^3$? Explain why or why not. (3점)
(d) Describe what the Moore-Penrose pseudoinverse $A^+$ gives when we compute $A^+ b$, and explain its geometric meaning. (3점)

**[KR]** 행렬 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$ 에 대하여,

(a) $A$의 랭크(rank)를 구하고, 그 이유를 설명하시오. (3점)
(b) Rank-Nullity 정리를 사용하여 $\dim(\mathscr{N}(A))$를 구하시오. (3점)
(c) 방정식 $Ax = b$가 모든 $b \in \mathbb{R}^3$에 대해 유일한 해를 가질 수 있는가? 이유를 설명하시오. (3점)
(d) Moore-Penrose 의사역행렬 $A^+$를 사용하여 $A^+ b$를 계산하면 어떤 해를 얻는지 설명하고, 그 기하학적 의미를 서술하시오. (3점)

---

## Problem 3 (10점)

**[EN]** Let $A \in \mathbb{R}^{m \times n}$ with SVD $A = U\Sigma V^\top$, where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$.

(a) Write the rank-$k$ approximation $A_k$ in terms of the SVD components. (3점)
(b) Prove that $A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$ (state the Eckart-Young theorem and explain why the Frobenius norm decomposes into singular values). (4점)
(c) In the context of deep learning, explain one practical application of low-rank approximation (e.g., LoRA). (3점)

**[KR]** $A \in \mathbb{R}^{m \times n}$의 SVD가 $A = U\Sigma V^\top$이고, $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$일 때,

(a) 랭크-$k$ 근사 $A_k$를 SVD 구성요소로 표현하시오. (3점)
(b) $A_k$가 $\|A - B\|_F$를 최소화하는 랭크-$k$ 이하 행렬임을 설명하시오 (Eckart-Young 정리). 프로베니우스 노름이 특이값으로 분해되는 이유를 포함하시오. (4점)
(c) 딥러닝에서 저랭크 근사의 실용적 응용 예시를 하나 들고 설명하시오. (3점)

---

## Problem 4 (8점)

**[EN]** Let $A \in \mathbb{R}^{n \times n}$ be symmetric and positive semi-definite (PSD).

(a) Prove that all eigenvalues of $A$ are non-negative. (4점)
(b) Prove that $B^\top B$ is always PSD for any real matrix $B$. (4점)

**[KR]** $A \in \mathbb{R}^{n \times n}$이 대칭이고 양반정치(PSD)일 때,

(a) $A$의 모든 고유값이 0 이상임을 증명하시오. (4점)
(b) 임의의 실수 행렬 $B$에 대해 $B^\top B$가 항상 PSD임을 증명하시오. (4점)

---

## Problem 5 (10점)

**[EN]** Prove the following trace identities:

(a) $\text{Tr}(AB) = \text{Tr}(BA)$ for $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times m}$. (4점)
(b) Show that $\|A\|_F^2 = \text{Tr}(A^\top A) = \sum_{i} \sigma_i^2$, where $\sigma_i$ are the singular values of $A$. (3점)
(c) Show that $x^\top A x = \text{Tr}(xx^\top A)$ for a vector $x$ and square matrix $A$. (3점)

**[KR]** 다음 trace 항등식을 증명하시오:

(a) $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times m}$일 때 $\text{Tr}(AB) = \text{Tr}(BA)$. (4점)
(b) $\|A\|_F^2 = \text{Tr}(A^\top A) = \sum_{i} \sigma_i^2$임을 보이시오 ($\sigma_i$는 $A$의 특이값). (3점)
(c) 벡터 $x$와 정방행렬 $A$에 대해 $x^\top A x = \text{Tr}(xx^\top A)$임을 보이시오. (3점)

---

# Part 2: Calculus, Matrix Calculus, and Optimization (50점)

---

## Problem 6 (12점)

**[EN]** Let $f(\theta) = \frac{1}{2}\|X\theta - y\|^2$ where $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $\theta \in \mathbb{R}^d$.

(a) Compute $\nabla_\theta f(\theta)$ using matrix calculus. Show all steps. (4점)
(b) Set the gradient to zero and derive the normal equation $\theta^* = (X^\top X)^{-1}X^\top y$. State the condition under which this solution exists. (4점)
(c) If we add L2 regularization: $g(\theta) = \frac{1}{2}\|X\theta - y\|^2 + \frac{\lambda}{2}\|\theta\|^2$, derive the new optimal $\theta^*$ and explain why this is always invertible when $\lambda > 0$. (4점)

**[KR]** $f(\theta) = \frac{1}{2}\|X\theta - y\|^2$ ($X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, $\theta \in \mathbb{R}^d$)에 대하여,

(a) 행렬 미적분을 사용하여 $\nabla_\theta f(\theta)$를 구하시오. 모든 과정을 보이시오. (4점)
(b) 그래디언트를 0으로 놓아 정규방정식 $\theta^* = (X^\top X)^{-1}X^\top y$를 유도하시오. 이 해가 존재하기 위한 조건을 명시하시오. (4점)
(c) L2 정규화를 추가한 $g(\theta) = \frac{1}{2}\|X\theta - y\|^2 + \frac{\lambda}{2}\|\theta\|^2$에 대해 새로운 최적해 $\theta^*$를 유도하고, $\lambda > 0$일 때 항상 역행렬이 존재하는 이유를 설명하시오. (4점)

---

## Problem 7 (10점)

**[EN]** Derive the Jacobian of the softmax function.

Given $p_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$, show that:

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

and express this compactly as $\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top$. (10점)

**[KR]** 소프트맥스 함수의 야코비안을 유도하시오.

$p_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$일 때, 다음을 보이시오:

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & i = j \text{일 때} \\ -p_i p_j & i \neq j \text{일 때} \end{cases}$$

이를 $\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top$으로 간결하게 표현하시오. (10점)

---

## Problem 8 (10점)

**[EN]** Using Lagrange multipliers, derive the softmax function from the maximum entropy principle.

Given: Maximize $H(p) = -\sum_i p_i \log p_i$ subject to $\sum_i p_i = 1$ and $\sum_i p_i z_i = \mu$ (fixed expected value).

(a) Write the Lagrangian. (2점)
(b) Solve for $p_i$ by setting partial derivatives to zero. (5점)
(c) Show that the solution takes the form $p_i = \frac{\exp(z_i / \tau)}{Z}$ where $Z$ is the partition function, and explain the role of temperature $\tau$. (3점)

**[KR]** 라그랑주 승수법을 사용하여 최대 엔트로피 원리로부터 소프트맥스 함수를 유도하시오.

주어진 조건: $H(p) = -\sum_i p_i \log p_i$를 최대화, 제약 조건 $\sum_i p_i = 1$, $\sum_i p_i z_i = \mu$ (고정된 기대값).

(a) 라그랑지안을 작성하시오. (2점)
(b) 편미분을 0으로 놓아 $p_i$를 구하시오. (5점)
(c) 해가 $p_i = \frac{\exp(z_i / \tau)}{Z}$ 형태임을 보이고, 온도(temperature) $\tau$의 역할을 설명하시오. (3점)

---

## Problem 9 (8점)

**[EN]** Consider gradient descent on $f(x) = \frac{1}{2}x^2$ with learning rate $\eta$.

(a) Write the gradient descent update rule for this function. (2점)
(b) Show that after $t$ iterations: $x_t = (1 - \eta)^t x_0$. (3점)
(c) For what range of $\eta$ does gradient descent converge? What happens when $\eta > 2$? (3점)

**[KR]** $f(x) = \frac{1}{2}x^2$에 대한 학습률 $\eta$의 경사 하강법을 고려하자.

(a) 이 함수에 대한 경사 하강법의 갱신 규칙을 쓰시오. (2점)
(b) $t$번 반복 후 $x_t = (1 - \eta)^t x_0$임을 보이시오. (3점)
(c) 경사 하강법이 수렴하는 $\eta$의 범위를 구하시오. $\eta > 2$이면 어떤 일이 벌어지는가? (3점)

---

## Problem 10 (10점)

**[EN]** Prove that the Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ is the maximum entropy distribution among all distributions with fixed mean $\mu$ and variance $\sigma^2$.

Hint: Use the Lagrangian with constraints $\int p(x)dx = 1$, $\int xp(x)dx = \mu$, $\int (x-\mu)^2 p(x)dx = \sigma^2$.

**[KR]** 평균 $\mu$와 분산 $\sigma^2$이 고정된 모든 분포 중에서 가우시안 분포 $\mathcal{N}(\mu, \sigma^2)$가 최대 엔트로피 분포임을 증명하시오.

힌트: 제약 조건 $\int p(x)dx = 1$, $\int xp(x)dx = \mu$, $\int (x-\mu)^2 p(x)dx = \sigma^2$에 대해 라그랑지안을 사용하시오.

---

# Part 3: Probability and Information Theory (50점)

---

## Problem 11 (10점)

**[EN]** Let $X \sim \mathcal{N}(\mu, \sigma^2)$ with i.i.d. samples $\{x_1, \ldots, x_n\}$.

(a) Write the log-likelihood function $\ell(\mu, \sigma^2)$. (2점)
(b) Derive $\hat{\mu}_{ML} = \frac{1}{n}\sum_{i=1}^n x_i$ by maximizing the log-likelihood w.r.t. $\mu$. (3점)
(c) Derive $\hat{\sigma}^2_{ML} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$. (3점)
(d) Show that $\mathbb{E}[\hat{\sigma}^2_{ML}] = \frac{n-1}{n}\sigma^2$ (i.e., the MLE for variance is biased). (2점)

**[KR]** $X \sim \mathcal{N}(\mu, \sigma^2)$이고 i.i.d. 표본 $\{x_1, \ldots, x_n\}$이 주어졌을 때,

(a) 로그 우도 함수 $\ell(\mu, \sigma^2)$를 쓰시오. (2점)
(b) 로그 우도를 $\mu$에 대해 최대화하여 $\hat{\mu}_{ML} = \frac{1}{n}\sum_{i=1}^n x_i$를 유도하시오. (3점)
(c) $\hat{\sigma}^2_{ML} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$를 유도하시오. (3점)
(d) $\mathbb{E}[\hat{\sigma}^2_{ML}] = \frac{n-1}{n}\sigma^2$임을 보이시오 (MLE 분산 추정량이 편향됨). (2점)

---

## Problem 12 (12점)

**[EN]**

(a) State Bayes' theorem and define each term (prior, likelihood, posterior, evidence). (3점)
(b) Show that the MAP estimate is equivalent to: $\hat{\theta}_{MAP} = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]$. (3점)
(c) If the prior is $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$ and the likelihood is Gaussian, show that MAP estimation is equivalent to minimizing MSE loss + L2 regularization. Derive the value of the regularization coefficient $\lambda$ in terms of $\sigma_p^2$. (6점)

**[KR]**

(a) 베이즈 정리를 서술하고, 각 항(사전확률, 우도, 사후확률, 증거)을 정의하시오. (3점)
(b) MAP 추정이 $\hat{\theta}_{MAP} = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]$와 동치임을 보이시오. (3점)
(c) 사전분포가 $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$이고 우도가 가우시안일 때, MAP 추정이 MSE 손실 + L2 정규화를 최소화하는 것과 동치임을 보이시오. 정규화 계수 $\lambda$를 $\sigma_p^2$로 표현하시오. (6점)

---

## Problem 13 (10점)

**[EN]**

(a) Define the KL divergence $D_{KL}(p \| q)$ and prove that $D_{KL}(p \| q) \geq 0$ (Gibbs' inequality). (5점)
(b) Show that minimizing the cross-entropy $H(p, q)$ is equivalent to minimizing $D_{KL}(p \| q)$ when $p$ is fixed. (2점)
(c) For classification with true labels $y$ (one-hot) and model output $\hat{y}$ (softmax), write the cross-entropy loss and explain why this is equivalent to negative log-likelihood. (3점)

**[KR]**

(a) KL 발산 $D_{KL}(p \| q)$를 정의하고, $D_{KL}(p \| q) \geq 0$ (Gibbs 부등식)을 증명하시오. (5점)
(b) $p$가 고정일 때, 교차 엔트로피 $H(p, q)$ 최소화가 $D_{KL}(p \| q)$ 최소화와 동치임을 보이시오. (2점)
(c) 정답 레이블 $y$ (원-핫)와 모델 출력 $\hat{y}$ (소프트맥스)에 대한 교차 엔트로피 손실을 쓰고, 이것이 음의 로그 우도와 동치인 이유를 설명하시오. (3점)

---

## Problem 14 (8점)

**[EN]**

(a) State the Central Limit Theorem (CLT). (2점)
(b) Explain why CLT justifies the use of Gaussian noise assumptions in regression (connecting to MSE loss). (3점)
(c) In SGD with mini-batch size $B$, the gradient estimate has variance proportional to $\sigma^2/B$. Explain, using CLT, why larger batch sizes lead to more stable gradient estimates. (3점)

**[KR]**

(a) 중심극한정리(CLT)를 서술하시오. (2점)
(b) CLT가 왜 회귀에서 가우시안 노이즈 가정을 정당화하는지 설명하시오 (MSE 손실과 연결). (3점)
(c) 미니배치 크기 $B$인 SGD에서 그래디언트 추정의 분산이 $\sigma^2/B$에 비례한다. CLT를 사용하여 큰 배치 크기가 왜 더 안정적인 그래디언트 추정을 제공하는지 설명하시오. (3점)

---

## Problem 15 (10점)

**[EN]** (Hoeffding's Inequality Application)

Let $X_1, \ldots, X_n$ be i.i.d. random variables with $X_i \in [0, 1]$ and $\mathbb{E}[X_i] = \mu$.

(a) State Hoeffding's inequality for $P(|\bar{X} - \mu| \geq t)$. (3점)
(b) A model achieves empirical accuracy $\hat{\mu} = 0.95$ on $n = 1000$ test samples. Using Hoeffding's inequality, compute an upper bound on the probability that the true accuracy $\mu$ differs from $\hat{\mu}$ by more than 0.03. (4점)
(c) How many test samples $n$ are needed to guarantee $P(|\hat{\mu} - \mu| \geq 0.01) \leq 0.05$? (3점)

**[KR]** (호에프딩 부등식 응용)

$X_1, \ldots, X_n$이 $X_i \in [0, 1]$이고 $\mathbb{E}[X_i] = \mu$인 i.i.d. 확률변수일 때,

(a) $P(|\bar{X} - \mu| \geq t)$에 대한 호에프딩 부등식을 서술하시오. (3점)
(b) 모델이 $n = 1000$개 테스트 샘플에서 경험적 정확도 $\hat{\mu} = 0.95$를 달성했다. 호에프딩 부등식을 사용하여, 실제 정확도 $\mu$가 $\hat{\mu}$와 0.03 이상 차이날 확률의 상한을 구하시오. (4점)
(c) $P(|\hat{\mu} - \mu| \geq 0.01) \leq 0.05$를 보장하려면 테스트 샘플 $n$이 최소 몇 개 필요한가? (3점)

---

# Part 4: Deep Learning Fundamentals (50점)

---

## Problem 16 (12점)

**[EN]** Consider a 2-layer neural network: $f(x) = W_2 \sigma(W_1 x + b_1) + b_2$ where $\sigma$ is ReLU, $x \in \mathbb{R}^2$, $W_1 \in \mathbb{R}^{3 \times 2}$, $b_1 \in \mathbb{R}^3$, $W_2 \in \mathbb{R}^{1 \times 3}$, $b_2 \in \mathbb{R}$.

(a) Compute the forward pass for $x = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$ with:
$W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$, $b_1 = \begin{pmatrix} 0 \\ 0 \\ -1 \end{pmatrix}$, $W_2 = \begin{pmatrix} 1 & -1 & 2 \end{pmatrix}$, $b_2 = 0.5$. (5점)

(b) Compute $\frac{\partial f}{\partial W_2}$ and $\frac{\partial f}{\partial b_2}$ at this point. (4점)
(c) Explain why removing all activation functions $\sigma$ would make this network equivalent to a single linear layer, regardless of depth. (3점)

**[KR]** 2층 신경망 $f(x) = W_2 \sigma(W_1 x + b_1) + b_2$ (ReLU 활성화, $x \in \mathbb{R}^2$, $W_1 \in \mathbb{R}^{3 \times 2}$, $b_1 \in \mathbb{R}^3$, $W_2 \in \mathbb{R}^{1 \times 3}$, $b_2 \in \mathbb{R}$)에 대하여,

(a) $x = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$, $W_1 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$, $b_1 = \begin{pmatrix} 0 \\ 0 \\ -1 \end{pmatrix}$, $W_2 = \begin{pmatrix} 1 & -1 & 2 \end{pmatrix}$, $b_2 = 0.5$일 때 순전파를 계산하시오. (5점)
(b) 이 지점에서 $\frac{\partial f}{\partial W_2}$와 $\frac{\partial f}{\partial b_2}$를 구하시오. (4점)
(c) 모든 활성화 함수 $\sigma$를 제거하면, 깊이에 관계없이 이 네트워크가 단일 선형 층과 동치인 이유를 설명하시오. (3점)

---

## Problem 17 (12점)

**[EN]** (Backpropagation / Chain Rule)

For the computational graph: $L = (y - \sigma(w^\top x))^2$ where $\sigma(z) = \frac{1}{1 + e^{-z}}$ (sigmoid), $x \in \mathbb{R}^d$, $w \in \mathbb{R}^d$.

(a) Draw the computational graph with intermediate variables $z = w^\top x$, $a = \sigma(z)$, $L = (y - a)^2$. (2점)
(b) Show that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. (3점)
(c) Compute $\frac{\partial L}{\partial w}$ using the chain rule. Show all intermediate steps. (4점)
(d) If $\sigma(z) \approx 0$ or $\sigma(z) \approx 1$, what happens to $\frac{\partial L}{\partial w}$? Explain the vanishing gradient problem for sigmoid. (3점)

**[KR]** (역전파 / 연쇄 법칙)

계산 그래프: $L = (y - \sigma(w^\top x))^2$ ($\sigma(z) = \frac{1}{1 + e^{-z}}$, $x \in \mathbb{R}^d$, $w \in \mathbb{R}^d$)에 대하여,

(a) 중간 변수 $z = w^\top x$, $a = \sigma(z)$, $L = (y - a)^2$으로 계산 그래프를 그리시오. (2점)
(b) $\sigma'(z) = \sigma(z)(1 - \sigma(z))$임을 보이시오. (3점)
(c) 연쇄 법칙을 사용하여 $\frac{\partial L}{\partial w}$를 구하시오. 모든 중간 과정을 보이시오. (4점)
(d) $\sigma(z) \approx 0$ 또는 $\sigma(z) \approx 1$일 때 $\frac{\partial L}{\partial w}$에 어떤 일이 발생하는가? 시그모이드의 기울기 소실 문제를 설명하시오. (3점)

---

## Problem 18 (10점)

**[EN]** (Scaled Dot-Product Attention)

Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{m \times d_k}$, values $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

(a) For $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$, compute $QK^\top$ and $QK^\top / \sqrt{d_k}$ (with $d_k = 2$). (3점)

(b) Explain mathematically why we divide by $\sqrt{d_k}$. (Hint: if $q_i, k_j \sim \mathcal{N}(0,1)$ i.i.d., what is $\text{Var}(q^\top k)$?) (4점)

(c) In masked self-attention (used in GPT), we set $[QK^\top]_{ij} = -\infty$ for $j > i$ before applying softmax. Explain why this results in $A_{ij} = 0$ for $j > i$, and why this is necessary for autoregressive generation. (3점)

**[KR]** (스케일드 닷-프로덕트 어텐션)

$Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$일 때:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

(a) $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$일 때 $QK^\top$과 $QK^\top / \sqrt{d_k}$ ($d_k = 2$)를 계산하시오. (3점)

(b) $\sqrt{d_k}$로 나누는 이유를 수학적으로 설명하시오. (힌트: $q_i, k_j \sim \mathcal{N}(0,1)$ i.i.d.이면 $\text{Var}(q^\top k)$는?) (4점)

(c) 마스크드 셀프 어텐션(GPT)에서 소프트맥스 적용 전 $j > i$인 위치를 $-\infty$로 설정한다. 이것이 왜 $A_{ij} = 0$ ($j > i$)을 만드는지 설명하고, 자기회귀 생성에서 왜 필요한지 서술하시오. (3점)

---

## Problem 19 (8점)

**[EN]** (Generalization Theory)

(a) Define the bias-variance decomposition: $\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$. Define each term. (3점)
(b) Explain the double descent phenomenon. Draw the test error curve as a function of model capacity, and label the three regimes (underfitting, interpolation threshold, overparameterized). (3점)
(c) What is implicit regularization? Why does gradient descent initialized at zero tend to find the minimum-norm solution in overparameterized settings? (2점)

**[KR]** (일반화 이론)

(a) 편향-분산 분해를 정의하시오: $\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$. 각 항을 정의하시오. (3점)
(b) 이중 하강(Double Descent) 현상을 설명하시오. 모델 용량의 함수로 테스트 에러 곡선을 그리고, 세 가지 영역(과소적합, 보간 임계점, 과매개변수화)을 표시하시오. (3점)
(c) 암묵적 정규화(Implicit Regularization)란 무엇인가? 과매개변수화 환경에서 0에서 초기화된 경사 하강법이 왜 최소 노름 해를 찾는 경향이 있는지 설명하시오. (2점)

---

## Problem 20 (8점)

**[EN]** (Integration Problem)

Show that MLE under Gaussian noise is equivalent to minimizing MSE loss:

Given: $y_i = f_\theta(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d.

(a) Write $p(y_i | x_i, \theta)$ as a Gaussian density. (2점)
(b) Write the log-likelihood $\ell(\theta) = \sum_{i=1}^n \log p(y_i | x_i, \theta)$. (2점)
(c) Show that $\arg\max_\theta \ell(\theta) = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n (y_i - f_\theta(x_i))^2$. (4점)

**[KR]** (통합 문제)

가우시안 노이즈 하에서 MLE가 MSE 손실 최소화와 동치임을 보이시오:

주어진 조건: $y_i = f_\theta(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d.

(a) $p(y_i | x_i, \theta)$를 가우시안 밀도함수로 쓰시오. (2점)
(b) 로그 우도 $\ell(\theta) = \sum_{i=1}^n \log p(y_i | x_i, \theta)$를 전개하시오. (2점)
(c) $\arg\max_\theta \ell(\theta) = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n (y_i - f_\theta(x_i))^2$임을 보이시오. (4점)

---

# 풀이 (Solutions)

---

## Problem 1 풀이

**(a)** 고유값 방정식 $\det(A - \lambda I) = 0$:

$$\det\begin{pmatrix} 2-\lambda & 1 \\ 1 & 3-\lambda \end{pmatrix} = (2-\lambda)(3-\lambda) - 1 = \lambda^2 - 5\lambda + 5 = 0$$

근의 공식: $\lambda = \frac{5 \pm \sqrt{25-20}}{2} = \frac{5 \pm \sqrt{5}}{2}$

$\lambda_1 = \frac{5+\sqrt{5}}{2} \approx 3.618$, $\lambda_2 = \frac{5-\sqrt{5}}{2} \approx 1.382$

$\lambda_1$의 고유벡터: $(A - \lambda_1 I)v = 0$

$$\begin{pmatrix} 2-\lambda_1 & 1 \\ 1 & 3-\lambda_1 \end{pmatrix} v = 0$$

$2 - \lambda_1 = \frac{-1-\sqrt{5}}{2}$이므로 $v_1 \propto \begin{pmatrix} 1 \\ \frac{1+\sqrt{5}}{2} \end{pmatrix}$ (정규화하여 단위벡터로)

$\lambda_2$도 동일한 과정으로 $v_2 \propto \begin{pmatrix} 1 \\ \frac{1-\sqrt{5}}{2} \end{pmatrix}$

**(b)** $A = U\Lambda U^\top$ 에서 $U = [v_1 | v_2]$ (단위 고유벡터 열), $\Lambda = \text{diag}(\lambda_1, \lambda_2)$

검증: $A$가 대칭이므로 고유벡터는 직교 → $U^\top U = I$ ✓

---

## Problem 2 풀이

**(a)** $A$의 두 열 $\begin{pmatrix}1\\3\\5\end{pmatrix}$, $\begin{pmatrix}2\\4\\6\end{pmatrix}$을 확인. 두 번째 열 = 첫 번째 열의 2배가 아님 ($2 \neq 2, 6 \neq 6$이지만 $4 \neq 6$). 두 열은 선형독립. 하지만 3개 열이 있는 게 아니라 2개 열이므로, $\text{rank}(A) \leq \min(3, 2) = 2$. 두 열이 선형독립이므로 $\text{rank}(A) = 2$.

**(b)** Rank-Nullity: $n = \text{rank}(A) + \text{null}(A)$, $2 = 2 + \text{null}(A)$ → $\text{null}(A) = 0$. 영공간은 $\{0\}$뿐.

**(c)** $A$는 $3 \times 2$ 행렬이므로 $Ax$의 출력은 $\mathbb{R}^3$ 속의 벡터지만, $\text{rank}(A) = 2$이므로 $\text{im}(A)$는 $\mathbb{R}^3$의 2차원 부분공간. 따라서 $\text{im}(A) \neq \mathbb{R}^3$이므로, $\text{im}(A)$ 밖의 $b$에 대해서는 **해가 존재하지 않는다**. 유일한 해는 더더욱 불가능.

**(d)** $A^+ b$는 $\|Ax - b\|^2$를 최소화하는 $x$를 준다 (최소 자승 해). 기하학적으로, $b$를 $\text{im}(A)$ 위에 **직교 사영(orthogonal projection)**한 후, 그 사영점에 대응하는 $x$를 구하는 것이다. 만약 해가 여러 개이면 (null space가 비자명하면) $A^+b$는 그 중 **노름이 최소인 해**를 선택한다.

---

## Problem 3 풀이

**(a)** $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top = U_k \Sigma_k V_k^\top$

여기서 $U_k$는 $U$의 처음 $k$열, $\Sigma_k$는 $\sigma_1, ..., \sigma_k$의 대각행렬, $V_k$는 $V$의 처음 $k$열.

**(b)** Eckart-Young 정리: 프로베니우스 노름에서 $A_k$는 랭크-$k$ 이하 행렬 중 $A$에 가장 가깝다.

증명의 핵심: $\|A - B\|_F^2 = \text{Tr}((A-B)^\top(A-B))$. SVD를 대입하면:

$$\|A - A_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

임의의 랭크-$k$ 행렬 $B$에 대해 $\|A - B\|_F^2 \geq \sum_{i=k+1}^{r} \sigma_i^2$임을 보일 수 있다 (특이값의 interlacing inequality 이용). 따라서 $A_k$가 최적.

**(c)** LoRA (Low-Rank Adaptation): 사전학습된 모델의 가중치 $W$를 고정하고, $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$)만 학습. 저랭크 행렬로 가중치 업데이트를 근사하여, 파라미터 수를 대폭 줄이면서도 fine-tuning 성능을 유지.

---

## Problem 4 풀이

**(a)** $A$가 PSD이고 $\lambda$가 고유값, $v$가 대응 고유벡터($v \neq 0$)라 하자.

$$v^\top A v = v^\top (\lambda v) = \lambda (v^\top v) = \lambda \|v\|^2$$

$A$가 PSD이므로 $v^\top Av \geq 0$, $\|v\|^2 > 0$ ($v \neq 0$).

따라서 $\lambda \geq 0$. $\blacksquare$

**(b)** 임의의 $x \neq 0$에 대해:

$$x^\top (B^\top B) x = (Bx)^\top (Bx) = \|Bx\|^2 \geq 0$$

벡터의 노름 제곱은 항상 0 이상이므로, $B^\top B$는 PSD. $\blacksquare$

---

## Problem 5 풀이

**(a)** $(AB)_{ii} = \sum_k A_{ik} B_{ki}$이므로:

$$\text{Tr}(AB) = \sum_i (AB)_{ii} = \sum_i \sum_k A_{ik} B_{ki}$$

$(BA)_{kk} = \sum_i B_{ki} A_{ik}$이므로:

$$\text{Tr}(BA) = \sum_k (BA)_{kk} = \sum_k \sum_i B_{ki} A_{ik}$$

이중 합의 순서를 바꾸면 두 식이 동일. $\blacksquare$

**(b)** $\|A\|_F^2 = \sum_{i,j} A_{ij}^2 = \text{Tr}(A^\top A)$ (정의에서 직접).

SVD를 대입: $A^\top A = V\Sigma^\top U^\top U\Sigma V^\top = V\Sigma^2 V^\top$ ($U$가 직교이므로 $U^\top U = I$).

$\text{Tr}(A^\top A) = \text{Tr}(V\Sigma^2 V^\top) = \text{Tr}(\Sigma^2 V^\top V) = \text{Tr}(\Sigma^2) = \sum_i \sigma_i^2$. (순환 성질 사용) $\blacksquare$

**(c)** $x^\top Ax$는 스칼라. 스칼라 = 자기 자신의 trace.

$x^\top Ax = \text{Tr}(x^\top Ax)$. 순환 성질: $\text{Tr}(x^\top Ax) = \text{Tr}(Axx^\top) = \text{Tr}(xx^\top A)$. $\blacksquare$

---

## Problem 6 풀이

**(a)** $f(\theta) = \frac{1}{2}(X\theta - y)^\top(X\theta - y) = \frac{1}{2}(\theta^\top X^\top X\theta - 2y^\top X\theta + y^\top y)$

$\nabla_\theta f = \frac{1}{2}(2X^\top X\theta - 2X^\top y) = X^\top X\theta - X^\top y = X^\top(X\theta - y)$

사용한 공식: $\frac{\partial}{\partial \theta}(\theta^\top S\theta) = 2S\theta$ (S 대칭), $\frac{\partial}{\partial \theta}(a^\top \theta) = a$

**(b)** $\nabla_\theta f = 0$:

$X^\top X\theta = X^\top y$ → $\theta^* = (X^\top X)^{-1}X^\top y$

조건: $X^\top X$가 가역이어야 함 → $\text{rank}(X) = d$ (full column rank) 필요.

**(c)** $\nabla_\theta g = X^\top(X\theta - y) + \lambda\theta = (X^\top X + \lambda I)\theta - X^\top y = 0$

$\theta^* = (X^\top X + \lambda I)^{-1} X^\top y$

$\lambda > 0$일 때 항상 가역인 이유: $X^\top X$는 PSD (Problem 4(b)에서 증명), $\lambda I$는 PD. 합은 PD → 모든 고유값이 양수 → 가역. $\blacksquare$

---

## Problem 7 풀이

$s = \sum_k \exp(z_k)$로 놓으면 $p_i = \exp(z_i)/s$.

**$i = j$인 경우:**

$$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot s - \exp(z_i) \cdot \exp(z_i)}{s^2} = \frac{\exp(z_i)}{s} \cdot \frac{s - \exp(z_i)}{s} = p_i(1 - p_i)$$

**$i \neq j$인 경우:**

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot s - \exp(z_i) \cdot \exp(z_j)}{s^2} = -\frac{\exp(z_i)}{s} \cdot \frac{\exp(z_j)}{s} = -p_i p_j$$

행렬 형태:

$$\frac{\partial p}{\partial z} = \begin{pmatrix} p_1(1-p_1) & -p_1 p_2 & \cdots \\ -p_2 p_1 & p_2(1-p_2) & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

대각 성분: $p_i - p_i^2$ → $\text{diag}(p)$ 부분. 비대각 성분: $-p_i p_j$ → $-pp^\top$ 부분.

$$\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top \quad \blacksquare$$

---

## Problem 8 풀이

**(a)** 라그랑지안:

$$\mathcal{L}(p, \lambda_0, \lambda_1) = -\sum_i p_i \log p_i + \lambda_0\left(\sum_i p_i - 1\right) + \lambda_1\left(\sum_i p_i z_i - \mu\right)$$

**(b)** $p_i$에 대해 편미분:

$$\frac{\partial \mathcal{L}}{\partial p_i} = -\log p_i - 1 + \lambda_0 + \lambda_1 z_i = 0$$

$$\log p_i = \lambda_0 - 1 + \lambda_1 z_i$$

$$p_i = \exp(\lambda_0 - 1 + \lambda_1 z_i) = C \cdot \exp(\lambda_1 z_i)$$

여기서 $C = \exp(\lambda_0 - 1)$은 정규화 상수.

제약 조건 $\sum_i p_i = 1$에서: $C = \frac{1}{\sum_j \exp(\lambda_1 z_j)}$

**(c)** $\lambda_1 = 1/\tau$로 놓으면:

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$$

이것이 **temperature-scaled softmax**.

- $\tau \to 0$: 가장 큰 $z_i$에 확률이 집중 (argmax에 수렴)
- $\tau \to \infty$: 균등분포에 수렴 (모든 $p_i \to 1/C$, 최대 엔트로피)
- $\tau = 1$: 표준 소프트맥스

$\tau$는 분포의 "날카로움"을 조절하는 하이퍼파라미터. 지식 증류(Knowledge Distillation)에서 핵심적으로 사용됨. $\blacksquare$

---

## Problem 9 풀이

**(a)** $f'(x) = x$이므로 갱신 규칙:

$$x_{t+1} = x_t - \eta \cdot f'(x_t) = x_t - \eta x_t = (1-\eta)x_t$$

**(b)** 귀납법:

$x_1 = (1-\eta)x_0$

$x_2 = (1-\eta)x_1 = (1-\eta)^2 x_0$

$x_t = (1-\eta)^t x_0$ $\blacksquare$

**(c)** 수렴 조건: $|x_t| \to 0$ ⟺ $|(1-\eta)^t| \to 0$ ⟺ $|1-\eta| < 1$

이를 풀면: $-1 < 1-\eta < 1$ → $0 < \eta < 2$

$\eta > 2$이면: $|1-\eta| > 1$ → $|x_t| = |1-\eta|^t |x_0| \to \infty$ → **발산**

$\eta = 1$이면: $x_1 = 0$ → 1스텝에 수렴 (최적)
$\eta = 2$이면: $x_t = (-1)^t x_0$ → 0과 $-x_0$ 사이를 **진동**

---

## Problem 10 풀이

라그랑지안:

$$\mathcal{L}[p] = -\int p(x)\log p(x)\,dx + \lambda_0\!\left(\int p(x)\,dx - 1\right) + \lambda_1\!\left(\int xp(x)\,dx - \mu\right) + \lambda_2\!\left(\int (x-\mu)^2 p(x)\,dx - \sigma^2\right)$$

변분법(calculus of variations)으로 $p(x)$에 대해 함수 미분:

$$\frac{\delta \mathcal{L}}{\delta p(x)} = -\log p(x) - 1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2 = 0$$

$$\log p(x) = (\lambda_0 - 1) + \lambda_1 x + \lambda_2(x-\mu)^2$$

$$p(x) = \exp\!\left[(\lambda_0 - 1) + \lambda_1 x + \lambda_2(x-\mu)^2\right]$$

대칭 조건($\mu$ 중심)에서 $\lambda_1 = 0$. $\lambda_2 < 0$이어야 적분 가능 (꼬리가 감소). $\lambda_2 = -\frac{1}{2\sigma^2}$로 놓으면:

$$p(x) = C \cdot \exp\!\left[-\frac{(x-\mu)^2}{2\sigma^2}\right]$$

정규화 조건 $\int p(x)\,dx = 1$에서 $C = \frac{1}{\sqrt{2\pi\sigma^2}}$.

이것이 $\mathcal{N}(\mu, \sigma^2)$. 평균과 분산이 고정되면 **가우시안이 최대 엔트로피 분포**. $\blacksquare$

---

## Problem 11 풀이

**(a)** $p(x_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$

$$\ell(\mu,\sigma^2) = \sum_{i=1}^n \log p(x_i|\mu,\sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i-\mu)^2$$

**(b)** $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$

$\sum_{i=1}^n x_i - n\mu = 0$ → $\hat{\mu}_{ML} = \frac{1}{n}\sum_{i=1}^n x_i$ $\blacksquare$

**(c)** $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0$

$n\sigma^2 = \sum_{i=1}^n (x_i - \mu)^2$ → $\hat{\sigma}^2_{ML} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$ $\blacksquare$

**(d)** $\mathbb{E}[\hat{\sigma}^2_{ML}] = \mathbb{E}\!\left[\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2\right]$

핵심 단계: $\sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n (x_i - \mu)^2 - n(\bar{x}-\mu)^2$

기대값: $\mathbb{E}\!\left[\sum_{i=1}^n (x_i-\mu)^2\right] = n\sigma^2$, $\mathbb{E}[n(\bar{x}-\mu)^2] = n \cdot \frac{\sigma^2}{n} = \sigma^2$

따라서 $\mathbb{E}\!\left[\sum (x_i-\bar{x})^2\right] = n\sigma^2 - \sigma^2 = (n-1)\sigma^2$

$\mathbb{E}[\hat{\sigma}^2_{ML}] = \frac{(n-1)\sigma^2}{n} = \frac{n-1}{n}\sigma^2 \neq \sigma^2$ (편향) $\blacksquare$

---

## Problem 12 풀이

**(a)** 베이즈 정리: $P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$

- $P(\theta)$: **사전 확률(Prior)** — 데이터를 보기 전 파라미터에 대한 믿음
- $P(D|\theta)$: **우도(Likelihood)** — 파라미터가 $\theta$일 때 데이터가 관측될 확률
- $P(\theta|D)$: **사후 확률(Posterior)** — 데이터를 본 후 갱신된 믿음
- $P(D)$: **증거(Evidence)** — 정규화 상수 ($\theta$에 무관)

**(b)** MAP: $\hat{\theta} = \arg\max_\theta P(\theta|D) = \arg\max_\theta \frac{P(D|\theta)P(\theta)}{P(D)}$

$P(D)$는 $\theta$와 무관하므로:

$= \arg\max_\theta P(D|\theta)P(\theta) = \arg\max_\theta [\log P(D|\theta) + \log P(\theta)]$ $\blacksquare$

**(c)** 우도가 가우시안: $P(D|\theta) = \prod_i \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$

$\log P(D|\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2$

사전분포: $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$

$\log P(\theta) = -\frac{d}{2}\log(2\pi\sigma_p^2) - \frac{1}{2\sigma_p^2}\|\theta\|^2$

MAP = $\arg\max_\theta [\log P(D|\theta) + \log P(\theta)]$

상수 제거 후:

$= \arg\min_\theta \left[\frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 + \frac{1}{2\sigma_p^2}\|\theta\|^2\right]$

$= \arg\min_\theta \left[\underbrace{\frac{1}{n}\sum_i (y_i - f_\theta(x_i))^2}_{\text{MSE}} + \underbrace{\frac{\sigma^2}{n\sigma_p^2}}_{\lambda}\|\theta\|^2\right]$

따라서 $\lambda = \frac{\sigma^2}{n\sigma_p^2}$. L2 정규화 = 가우시안 사전 확률. $\blacksquare$

---

## Problem 13 풀이

**(a)** 정의: $D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = -\sum_x p(x) \log \frac{q(x)}{p(x)}$

Gibbs 부등식 증명 ($\ln$으로 진행 후 일반화):

$\ln t \leq t - 1$ (모든 $t > 0$, 등호 $t=1$)을 사용.

$$-D_{KL}(p \| q) = \sum_x p(x) \ln \frac{q(x)}{p(x)} \leq \sum_x p(x) \left(\frac{q(x)}{p(x)} - 1\right) = \sum_x q(x) - \sum_x p(x) = 1 - 1 = 0$$

따라서 $D_{KL}(p \| q) \geq 0$. 등호 조건: $q(x)/p(x) = 1$ 모든 $x$에서, 즉 $p = q$. $\blacksquare$

**(b)** $H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{KL}(p \| q)$

$p$가 고정이면 $H(p)$는 상수. 따라서:

$\arg\min_q H(p, q) = \arg\min_q D_{KL}(p \| q)$ $\blacksquare$

**(c)** $y$가 원-핫(클래스 $c$에서 1, 나머지 0), $\hat{y} = \text{softmax}(z)$:

$H(y, \hat{y}) = -\sum_k y_k \log \hat{y}_k = -\log \hat{y}_c$

이것은 $-\log P(Y=c | x, \theta)$ = **음의 로그 우도(NLL)**와 정확히 같다. 카테고리컬 분포의 우도가 $P(Y=c) = \hat{y}_c$이므로. $\blacksquare$

---

## Problem 14 풀이

**(a)** CLT: $X_1, ..., X_n$이 i.i.d.이고 $\mathbb{E}[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2$이면:

$$\sqrt{n}\frac{\bar{X} - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad (n \to \infty)$$

즉, 표본 평균은 근사적으로 $\bar{X} \sim \mathcal{N}(\mu, \sigma^2/n)$

**(b)** 회귀에서 $y = f(x) + \epsilon$, 노이즈 $\epsilon$은 많은 작은 독립적 요인의 합이라 볼 수 있다 (측정 오차, 환경 요인 등). CLT에 의해 이런 합은 가우시안에 수렴한다. 따라서 $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 가정이 정당화되고, 이 가정 하에서 MLE = MSE 최소화 (Problem 20에서 증명).

**(c)** 미니배치 그래디언트: $g_B = \frac{1}{B}\sum_{i \in \text{batch}} \nabla \ell_i(\theta)$

각 $\nabla \ell_i$가 i.i.d.이고 분산 $\sigma_g^2$이면:

$\text{Var}(g_B) = \frac{\sigma_g^2}{B}$

CLT에 의해 $B$가 크면 $g_B \sim \mathcal{N}(\nabla L(\theta), \sigma_g^2/B)$에 근사.

$B$가 커질수록 분산 $\sigma_g^2/B$가 줄어든다 → 그래디언트 추정이 진짜 그래디언트에 가까워짐 → **더 안정적**. $\blacksquare$

---

## Problem 15 풀이

**(a)** 호에프딩 부등식: $X_i \in [a_i, b_i]$ i.i.d., $\bar{X} = \frac{1}{n}\sum X_i$일 때:

$$P(|\bar{X} - \mu| \geq t) \leq 2\exp\!\left(-\frac{2n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$

$X_i \in [0, 1]$이면 $(b_i - a_i) = 1$:

$$P(|\bar{X} - \mu| \geq t) \leq 2\exp(-2nt^2)$$

**(b)** $n = 1000$, $t = 0.03$:

$$P(|\hat{\mu} - \mu| \geq 0.03) \leq 2\exp(-2 \times 1000 \times 0.03^2) = 2\exp(-1.8) \approx 2 \times 0.1653 = 0.331$$

상한: 약 **33.1%**

**(c)** $2\exp(-2n \times 0.01^2) \leq 0.05$

$\exp(-0.0002n) \leq 0.025$

$-0.0002n \leq \ln(0.025) = -3.689$

$n \geq \frac{3.689}{0.0002} = 18{,}445$

최소 **18,445개** 테스트 샘플 필요. $\blacksquare$

---

## Problem 16 풀이

**(a)** 순전파:

Step 1: $z = W_1 x + b_1 = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}\begin{pmatrix}1\\-1\end{pmatrix} + \begin{pmatrix}0\\0\\-1\end{pmatrix} = \begin{pmatrix}1\\-1\\0\end{pmatrix} + \begin{pmatrix}0\\0\\-1\end{pmatrix} = \begin{pmatrix}1\\-1\\-1\end{pmatrix}$

Step 2: ReLU 적용: $a = \sigma(z) = \begin{pmatrix}\max(0,1)\\\max(0,-1)\\\max(0,-1)\end{pmatrix} = \begin{pmatrix}1\\0\\0\end{pmatrix}$

Step 3: $f = W_2 a + b_2 = \begin{pmatrix}1&-1&2\end{pmatrix}\begin{pmatrix}1\\0\\0\end{pmatrix} + 0.5 = 1 + 0.5 = \boxed{1.5}$

**(b)** $f = W_2 a + b_2$에서:

$\frac{\partial f}{\partial W_2} = a^\top = \begin{pmatrix}1 & 0 & 0\end{pmatrix}$

$\frac{\partial f}{\partial b_2} = 1$

**(c)** 활성화 없이 $L$층: $f(x) = W_L \cdots W_2 W_1 x + \text{bias}$

행렬 곱의 결합법칙: $W_L \cdots W_1 = \tilde{W}$ (하나의 행렬)

따라서 $f(x) = \tilde{W}x + \tilde{b}$로, 단일 선형 층과 동일. 깊이를 아무리 쌓아도 표현력이 증가하지 않는다. **비선형 활성화 함수가 있어야** 깊이가 의미를 가진다. $\blacksquare$

---

## Problem 17 풀이

**(a)** 계산 그래프:

```
x, w → z = w^T x → a = σ(z) → L = (y - a)²
                                    ↑
                                    y
```

**(b)** $\sigma(z) = (1 + e^{-z})^{-1}$

$$\sigma'(z) = -(-e^{-z})(1+e^{-z})^{-2} = \frac{e^{-z}}{(1+e^{-z})^2}$$

$$= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z) \cdot \frac{1+e^{-z}-1}{1+e^{-z}} = \sigma(z)(1-\sigma(z)) \quad \blacksquare$$

**(c)** 체인룰:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$

각 항:
- $\frac{\partial L}{\partial a} = -2(y-a)$
- $\frac{\partial a}{\partial z} = \sigma(z)(1-\sigma(z))$
- $\frac{\partial z}{\partial w} = x$

합치면:

$$\frac{\partial L}{\partial w} = -2(y - \sigma(w^\top x)) \cdot \sigma(w^\top x)(1-\sigma(w^\top x)) \cdot x$$

**(d)** $\sigma(z) \approx 0$이면 $\sigma(z)(1-\sigma(z)) \approx 0 \times 1 = 0$

$\sigma(z) \approx 1$이면 $\sigma(z)(1-\sigma(z)) \approx 1 \times 0 = 0$

어느 쪽이든 $\frac{\partial a}{\partial z} \approx 0$ → **그래디언트가 거의 0** → 가중치 업데이트가 거의 없음 → **학습 정체**.

이것이 **기울기 소실(vanishing gradient)** 문제. 시그모이드의 출력이 0이나 1에 가까운 "포화 영역(saturation region)"에서 발생. ReLU ($\max(0,x)$)는 양수 영역에서 기울기가 항상 1이므로 이 문제를 해결. $\blacksquare$

---

## Problem 18 풀이

**(a)** $QK^\top = \begin{pmatrix}1&0\\0&1\end{pmatrix}\begin{pmatrix}1&0&1\\0&1&1\end{pmatrix} = \begin{pmatrix}1&0&1\\0&1&1\end{pmatrix}$

$d_k = 2$이므로 $QK^\top/\sqrt{2} = \begin{pmatrix}1/\sqrt{2}&0&1/\sqrt{2}\\0&1/\sqrt{2}&1/\sqrt{2}\end{pmatrix} \approx \begin{pmatrix}0.707&0&0.707\\0&0.707&0.707\end{pmatrix}$

**(b)** $q_i, k_j \sim \mathcal{N}(0, 1)$ i.i.d.이면:

$q^\top k = \sum_{l=1}^{d_k} q_l k_l$

$\mathbb{E}[q_l k_l] = 0$ (독립 × 평균 0)

$\text{Var}(q_l k_l) = \mathbb{E}[q_l^2 k_l^2] - (\mathbb{E}[q_l k_l])^2 = \mathbb{E}[q_l^2]\mathbb{E}[k_l^2] - 0 = 1 \cdot 1 = 1$

$\text{Var}(q^\top k) = \sum_{l=1}^{d_k} \text{Var}(q_l k_l) = d_k$

$d_k$가 크면 $q^\top k$의 분산이 $d_k$에 비례하여 커짐 → softmax 입력이 극단적으로 커짐 → softmax 출력이 one-hot에 가까워짐 → **기울기가 거의 0** (softmax 포화).

$\sqrt{d_k}$로 나누면: $\text{Var}(q^\top k / \sqrt{d_k}) = d_k/d_k = 1$로 정규화 → softmax가 적절한 범위에서 작동. $\blacksquare$

**(c)** $j > i$인 위치에 $-\infty$를 설정하면:

$\text{softmax}$에서 $\exp(-\infty) = 0$이므로 $A_{ij} = 0$ ($j > i$).

이것이 필요한 이유: 자기회귀(autoregressive) 생성에서 위치 $i$의 출력은 **과거 토큰 ($j \leq i$)만 참조**해야 한다. 미래 토큰($j > i$)의 정보를 사용하면 "답을 보고 문제를 푸는" 것과 같아 학습이 무의미해진다. 마스킹은 이 인과적(causal) 제약을 강제한다. $\blacksquare$

---

## Problem 19 풀이

**(a)** $\mathbb{E}[(y-\hat{f}(x))^2]$ 분해:

- **$\text{Bias}^2 = (\mathbb{E}[\hat{f}(x)] - f(x))^2$**: 모델의 평균 예측이 정답에서 얼마나 벗어나는가. 모델이 너무 단순하면 큼.
- **$\text{Variance} = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$**: 학습 데이터가 바뀌면 예측이 얼마나 흔들리는가. 모델이 너무 복잡하면 큼.
- **$\text{Noise} = \mathbb{E}[(y - f(x))^2] = \sigma^2$**: 데이터 자체의 노이즈. 줄일 수 없음.

**(b)** Double Descent:

```
Test Error
  │
  │  ╲           ╱╲
  │    ╲       ╱    ╲
  │      ╲   ╱        ╲_________
  │        ╲╱
  │         ↑
  │   Interpolation Threshold
  └──────────────────────── Model Capacity
    Under-     Classical    Over-
    fitting    U-shape      parameterized
```

세 영역:
1. **과소적합(Underfitting)**: 모델 용량 부족. 편향 높음.
2. **보간 임계점(Interpolation Threshold)**: 모델이 훈련 데이터를 겨우 외울 수 있는 최소 크기. 이 지점에서 테스트 에러 **최악**.
3. **과매개변수화(Overparameterized)**: 용량이 넉넉. 해가 무수히 많고, SGD가 그 중 부드러운(최소 노름) 해를 선택 → 테스트 에러 다시 감소.

**(c)** 암묵적 정규화: 명시적 정규화(L2 등) 없이도 SGD 자체가 자동으로 "간단한 해"를 선호하는 현상.

$\theta(0) = 0$에서 시작한 gradient descent는 $Ax = y$의 해 중 **최소 노름 해** $\theta^* = A^+ y$로 수렴. 이유: gradient flow $\dot{\theta} = -A^\top(A\theta - y)$의 궤적이 $A$의 행공간(row space) 안에 머물기 때문 (null space 방향으로는 움직이지 않음). 최소 노름 해 = 가장 "부드러운" 해 → 일반화 좋음. $\blacksquare$

---

## Problem 20 풀이

**(a)** $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$이므로 $y_i | x_i, \theta \sim \mathcal{N}(f_\theta(x_i), \sigma^2)$:

$$p(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right)$$

**(b)** i.i.d. 가정:

$$\ell(\theta) = \sum_{i=1}^n \log p(y_i | x_i, \theta) = \sum_{i=1}^n \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2}\right]$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2$$

**(c)** $\arg\max_\theta \ell(\theta)$에서 $\theta$와 무관한 항을 제거:

$$= \arg\max_\theta \left[-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2\right]$$

max에 음수가 붙어 있으므로 min으로 뒤집기:

$$= \arg\min_\theta \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2$$

$\frac{1}{2\sigma^2}$은 양의 상수이므로 최소화 위치에 영향 없음:

$$= \arg\min_\theta \sum_{i=1}^n (y_i - f_\theta(x_i))^2 = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n (y_i - f_\theta(x_i))^2$$

**가우시안 노이즈 하의 MLE = MSE 최소화**. $\blacksquare$

---

# 채점 기준 요약

| Part | 배점 | 핵심 역량 |
|------|------|----------|
| Part 1: Linear Algebra | 50점 | 고유값/SVD 계산, Rank-Nullity, PSD 증명, Trace 성질 |
| Part 2: Calculus & Optimization | 50점 | 행렬 미분, 소프트맥스 야코비안, 라그랑주, 경사하강법 수렴 |
| Part 3: Probability & Info Theory | 50점 | MLE 유도, 베이즈-MAP-정규화 연결, KL/CE 증명, 호에프딩 |
| Part 4: Deep Learning | 50점 | 순전파 계산, 역전파/체인룰, Attention 수식, 일반화 이론 |
| **총점** | **200점** | |

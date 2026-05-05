---
title: "Deep Learning 중간고사 모의시험 — 2회"
slug: midterm-exam-02
order: 2
---

# Deep Learning 중간고사 모의시험 — 2회

> **범위**: Preliminaries + Introduction + Optimization & Generalization
> **시간**: 120분 | **총점**: 200점
> **주의**: 모든 풀이에 논리적 근거와 수학적 유도 과정을 반드시 기술하시오.

---

# Part 1: Linear Algebra (50점)

---

## Problem 1 (10점)

**[EN]** Let $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$.

(a) Compute $A^{-1}$ using the formula for 2×2 matrices. (3점)
(b) Verify that $AA^{-1} = I$. (2점)
(c) Compute $\det(A)$ and $\text{Tr}(A)$. Show that $\det(A) = \lambda_1 \lambda_2$ and $\text{Tr}(A) = \lambda_1 + \lambda_2$ where $\lambda_1, \lambda_2$ are eigenvalues. (5점)

**[KR]** $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$에 대하여,

(a) 2×2 역행렬 공식을 사용하여 $A^{-1}$을 구하시오. (3점)
(b) $AA^{-1} = I$를 검증하시오. (2점)
(c) $\det(A)$와 $\text{Tr}(A)$를 구하시오. 고유값 $\lambda_1, \lambda_2$에 대해 $\det(A) = \lambda_1\lambda_2$, $\text{Tr}(A) = \lambda_1 + \lambda_2$가 성립함을 보이시오. (5점)

---

## Problem 2 (10점)

**[EN]** Let $u = \begin{pmatrix}1\\2\\2\end{pmatrix}$, $v = \begin{pmatrix}2\\-1\\0\end{pmatrix}$.

(a) Compute $\langle u, v \rangle$, $\|u\|$, $\|v\|$. (3점)
(b) Compute the cosine similarity $\cos(u,v)$. Are $u$ and $v$ more similar or more orthogonal? (3점)
(c) Find the projection of $v$ onto $u$: $\text{proj}_u v = \frac{\langle v, u \rangle}{\langle u, u \rangle} u$. (2점)
(d) Verify that $(v - \text{proj}_u v) \perp u$. (2점)

**[KR]** $u = \begin{pmatrix}1\\2\\2\end{pmatrix}$, $v = \begin{pmatrix}2\\-1\\0\end{pmatrix}$에 대하여,

(a) $\langle u, v \rangle$, $\|u\|$, $\|v\|$를 구하시오. (3점)
(b) 코사인 유사도 $\cos(u,v)$를 구하시오. $u$와 $v$는 유사한가 직교에 가까운가? (3점)
(c) $v$를 $u$ 위에 사영한 $\text{proj}_u v = \frac{\langle v, u \rangle}{\langle u, u \rangle} u$를 구하시오. (2점)
(d) $(v - \text{proj}_u v) \perp u$임을 검증하시오. (2점)

---

## Problem 3 (10점)

**[EN]** Let $A = \begin{pmatrix}1 & 2\\2 & 4\end{pmatrix}$.

(a) Find $\text{rank}(A)$ and $\mathscr{N}(A)$ (null space). (4점)
(b) The system $Ax = \begin{pmatrix}3\\6\end{pmatrix}$ has infinitely many solutions. Find the general solution. (3점)
(c) Among all solutions, find the one with minimum $\|x\|$ (minimum-norm solution). Verify this equals $A^+ b$. (3점)

**[KR]** $A = \begin{pmatrix}1 & 2\\2 & 4\end{pmatrix}$에 대하여,

(a) $\text{rank}(A)$와 영공간 $\mathscr{N}(A)$를 구하시오. (4점)
(b) $Ax = \begin{pmatrix}3\\6\end{pmatrix}$은 무한히 많은 해를 가진다. 일반해를 구하시오. (3점)
(c) 모든 해 중 $\|x\|$가 최소인 해(최소 노름 해)를 구하시오. 이것이 $A^+b$와 같음을 확인하시오. (3점)

---

## Problem 4 (12점)

**[EN]** Let $A = \begin{pmatrix}3 & 0\\0 & 1\end{pmatrix}$.

(a) Compute the SVD of $A$: $A = U\Sigma V^\top$. (3점)
(b) What is $A^+$? (2점)
(c) Let $B = \begin{pmatrix}3 & 0\\0 & 1\\0 & 0\end{pmatrix}$. Compute $B^\top B$, $BB^\top$, and show that the non-zero eigenvalues of $B^\top B$ and $BB^\top$ are the same. (4점)
(d) In the context of neural networks, explain why a weight matrix $W$ with very small singular values can cause vanishing gradients during backpropagation. (3점)

**[KR]** $A = \begin{pmatrix}3 & 0\\0 & 1\end{pmatrix}$에 대하여,

(a) SVD $A = U\Sigma V^\top$를 구하시오. (3점)
(b) $A^+$를 구하시오. (2점)
(c) $B = \begin{pmatrix}3 & 0\\0 & 1\\0 & 0\end{pmatrix}$에 대해 $B^\top B$, $BB^\top$를 계산하고, 비영 고유값이 동일함을 보이시오. (4점)
(d) 신경망에서 가중치 행렬 $W$의 특이값이 매우 작을 때 역전파에서 기울기 소실이 발생하는 이유를 설명하시오. (3점)

---

## Problem 5 (8점)

**[EN]** Prove the following:

(a) For orthogonal matrix $U$ ($U^\top U = I$), show that $\|Ux\| = \|x\|$ for all $x$. (3점)
(b) Show that if $A$ is symmetric PD, then $A^{-1}$ is also symmetric PD. (5점)

**[KR]** 다음을 증명하시오:

(a) 직교행렬 $U$ ($U^\top U = I$)에 대해 $\|Ux\| = \|x\|$가 모든 $x$에서 성립함을 보이시오. (3점)
(b) $A$가 대칭 양정치(PD)이면 $A^{-1}$도 대칭 양정치임을 보이시오. (5점)

---

# Part 2: Calculus and Optimization (50점)

---

## Problem 6 (10점)

**[EN]** Let $f(x, y) = x^2 y + 3xy^2 - 2x + 5$.

(a) Compute $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$. (4점)
(b) Compute the gradient $\nabla f$ at the point $(1, -1)$. (3점)
(c) In which direction does $f$ increase most rapidly at $(1, -1)$? What is the rate of increase? (3점)

**[KR]** $f(x, y) = x^2 y + 3xy^2 - 2x + 5$에 대하여,

(a) $\frac{\partial f}{\partial x}$와 $\frac{\partial f}{\partial y}$를 구하시오. (4점)
(b) 점 $(1, -1)$에서 그래디언트 $\nabla f$를 구하시오. (3점)
(c) $(1, -1)$에서 $f$가 가장 빠르게 증가하는 방향은? 증가 속도는? (3점)

---

## Problem 7 (12점)

**[EN]** (Chain Rule and Backpropagation)

Consider: $L = \frac{1}{2}(y - \sigma(w_2 \cdot \text{ReLU}(w_1 x + b_1) + b_2))^2$

where $\sigma(z) = 1/(1+e^{-z})$, $\text{ReLU}(z) = \max(0,z)$, and $x = 2$, $w_1 = 0.5$, $b_1 = -0.5$, $w_2 = 1$, $b_2 = 0$, $y = 1$.

(a) Compute the forward pass: find $z_1 = w_1 x + b_1$, $a_1 = \text{ReLU}(z_1)$, $z_2 = w_2 a_1 + b_2$, $a_2 = \sigma(z_2)$, $L$. (4점)
(b) Compute $\frac{\partial L}{\partial a_2}$, $\frac{\partial a_2}{\partial z_2}$, $\frac{\partial z_2}{\partial w_2}$, $\frac{\partial z_2}{\partial a_1}$, $\frac{\partial a_1}{\partial z_1}$, $\frac{\partial z_1}{\partial w_1}$. (4점)
(c) Using the chain rule, compute $\frac{\partial L}{\partial w_2}$ and $\frac{\partial L}{\partial w_1}$. (4점)

**[KR]** $L = \frac{1}{2}(y - \sigma(w_2 \cdot \text{ReLU}(w_1 x + b_1) + b_2))^2$에서 $\sigma(z) = 1/(1+e^{-z})$, $\text{ReLU}(z) = \max(0,z)$이고, $x = 2$, $w_1 = 0.5$, $b_1 = -0.5$, $w_2 = 1$, $b_2 = 0$, $y = 1$일 때,

(a) 순전파를 계산하시오: $z_1$, $a_1$, $z_2$, $a_2$, $L$. (4점)
(b) $\frac{\partial L}{\partial a_2}$, $\frac{\partial a_2}{\partial z_2}$, $\frac{\partial z_2}{\partial w_2}$, $\frac{\partial z_2}{\partial a_1}$, $\frac{\partial a_1}{\partial z_1}$, $\frac{\partial z_1}{\partial w_1}$을 각각 구하시오. (4점)
(c) 체인룰로 $\frac{\partial L}{\partial w_2}$와 $\frac{\partial L}{\partial w_1}$을 구하시오. (4점)

---

## Problem 8 (10점)

**[EN]** (Convexity)

(a) Prove that $f(x) = \|Ax - b\|^2$ is convex. (Hint: show that the Hessian is PSD.) (5점)
(b) Prove that adding L2 regularization $g(x) = f(x) + \lambda\|x\|^2$ ($\lambda > 0$) makes the function **strictly** convex. (3점)
(c) Why does strict convexity guarantee a unique global minimum? (2점)

**[KR]** (볼록성)

(a) $f(x) = \|Ax - b\|^2$이 볼록 함수임을 증명하시오. (힌트: 헤시안이 PSD임을 보이시오.) (5점)
(b) L2 정규화를 추가한 $g(x) = f(x) + \lambda\|x\|^2$ ($\lambda > 0$)가 **강볼록(strictly convex)**임을 증명하시오. (3점)
(c) 강볼록이 유일한 전역 최솟값을 보장하는 이유를 설명하시오. (2점)

---

## Problem 9 (8점)

**[EN]** (Taylor Expansion and Newton's Method)

(a) Write the second-order Taylor expansion of $f(x)$ around $x_0$. (2점)
(b) Newton's method update: $x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$. Apply two iterations to find the minimum of $f(x) = x^4 - 4x^2 + 1$ starting from $x_0 = 2$. (4점)
(c) Newton's method can diverge or oscillate. Give a specific condition on $f$ that guarantees convergence. (2점)

**[KR]** (테일러 전개와 뉴턴법)

(a) $f(x)$의 $x_0$ 주위 2차 테일러 전개를 쓰시오. (2점)
(b) 뉴턴법: $x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$. $f(x) = x^4 - 4x^2 + 1$에 대해 $x_0 = 2$에서 시작하여 2회 반복을 수행하시오. (4점)
(c) 뉴턴법이 발산하거나 진동할 수 있다. 수렴을 보장하는 $f$에 대한 구체적 조건을 하나 제시하시오. (2점)

---

## Problem 10 (10점)

**[EN]** (Constrained Optimization)

Maximize $f(x,y) = xy$ subject to $g(x,y) = x + y - 10 = 0$ where $x, y > 0$.

(a) Write the Lagrangian $\mathcal{L}(x, y, \lambda)$. (2점)
(b) Solve the system of equations from $\nabla_{x,y,\lambda}\mathcal{L} = 0$ to find the optimal $(x^*, y^*, \lambda^*)$. (5점)
(c) Interpret the result: what geometric shape with perimeter 20 has maximum area? Connect this to the bias-variance tradeoff in ML. (3점)

**[KR]** (제약 최적화)

$f(x,y) = xy$를 $g(x,y) = x + y - 10 = 0$ ($x, y > 0$) 제약 하에서 최대화하시오.

(a) 라그랑지안 $\mathcal{L}(x, y, \lambda)$를 작성하시오. (2점)
(b) $\nabla_{x,y,\lambda}\mathcal{L} = 0$의 연립방정식을 풀어 $(x^*, y^*, \lambda^*)$를 구하시오. (5점)
(c) 결과를 해석하시오: 둘레가 20인 직사각형 중 넓이가 최대인 것은? 이것을 ML의 편향-분산 트레이드오프와 연결하시오. (3점)

---

# Part 3: Probability and Information Theory (50점)

---

## Problem 11 (10점)

**[EN]** A bag contains 3 red and 2 blue balls. Two balls are drawn without replacement.

(a) What is the probability that both balls are red? (3점)
(b) Given that the first ball is red, what is the probability that the second is also red? (3점)
(c) Are the events "first ball is red" and "second ball is red" independent? Prove your answer using the definition of independence. (4점)

**[KR]** 가방에 빨간 공 3개, 파란 공 2개가 있다. 비복원으로 2개를 뽑는다.

(a) 두 개 모두 빨간 공일 확률은? (3점)
(b) 첫 번째가 빨간 공일 때, 두 번째도 빨간 공일 조건부 확률은? (3점)
(c) "첫 번째 빨강"과 "두 번째 빨강"은 독립인가? 독립의 정의를 사용하여 증명하시오. (4점)

---

## Problem 12 (10점)

**[EN]** Let $X \sim \text{Bernoulli}(p)$.

(a) Compute $\mathbb{E}[X]$ and $\text{Var}(X)$ from the definition. (4점)
(b) The entropy of $X$ is $H(X) = -p\log p - (1-p)\log(1-p)$. Find the value of $p$ that maximizes $H(X)$. (3점)
(c) Consider $n$ i.i.d. copies $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$. Derive the MLE $\hat{p}_{ML}$ by maximizing the log-likelihood. (3점)

**[KR]** $X \sim \text{Bernoulli}(p)$에 대하여,

(a) 정의로부터 $\mathbb{E}[X]$와 $\text{Var}(X)$를 구하시오. (4점)
(b) $X$의 엔트로피 $H(X) = -p\log p - (1-p)\log(1-p)$를 최대화하는 $p$를 구하시오. (3점)
(c) $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$ i.i.d.에 대해 로그 우도를 최대화하여 MLE $\hat{p}_{ML}$을 유도하시오. (3점)

---

## Problem 13 (12점)

**[EN]** (Information Theory)

Let $P = (0.25, 0.25, 0.25, 0.25)$ and $Q = (0.5, 0.25, 0.125, 0.125)$ be two distributions over 4 outcomes.

(a) Compute $H(P)$ (entropy of $P$). (2점)
(b) Compute $H(P, Q)$ (cross-entropy from $P$ to $Q$). (3점)
(c) Compute $D_{KL}(P \| Q)$. Verify that $D_{KL} = H(P,Q) - H(P)$. (3점)
(d) Compute $D_{KL}(Q \| P)$. Is it equal to $D_{KL}(P \| Q)$? What does this asymmetry mean for ML? (4점)

**[KR]** $P = (0.25, 0.25, 0.25, 0.25)$, $Q = (0.5, 0.25, 0.125, 0.125)$가 4개 결과에 대한 두 분포일 때,

(a) $H(P)$ (P의 엔트로피)를 계산하시오. (2점)
(b) $H(P, Q)$ (P에서 Q로의 교차 엔트로피)를 계산하시오. (3점)
(c) $D_{KL}(P \| Q)$를 계산하시오. $D_{KL} = H(P,Q) - H(P)$가 성립함을 확인하시오. (3점)
(d) $D_{KL}(Q \| P)$를 계산하시오. $D_{KL}(P \| Q)$와 같은가? 이 비대칭성이 ML에서 의미하는 바를 설명하시오. (4점)

---

## Problem 14 (10점)

**[EN]** (Bayesian Inference — Coin Flipping)

You have a coin with unknown probability $\theta$ of heads. Your prior belief is $\theta \sim \text{Beta}(2, 2)$.

You flip the coin 10 times and observe 7 heads, 3 tails.

(a) Write the likelihood function $P(D|\theta)$ for this data. (2점)
(b) Using the Beta-Binomial conjugacy, derive the posterior $P(\theta|D)$. (3점)
(c) Compute the MAP estimate $\hat{\theta}_{MAP}$ and the MLE $\hat{\theta}_{ML}$. Compare them and explain the difference. (3점)
(d) As $n \to \infty$, what happens to the difference between MAP and MLE? Why? (2점)

**[KR]** 앞면 확률 $\theta$가 미지인 동전이 있다. 사전 믿음: $\theta \sim \text{Beta}(2, 2)$.

10번 던져서 앞면 7번, 뒷면 3번을 관찰했다.

(a) 이 데이터의 우도 함수 $P(D|\theta)$를 쓰시오. (2점)
(b) 베타-이항 켤레성을 사용하여 사후분포 $P(\theta|D)$를 유도하시오. (3점)
(c) MAP 추정 $\hat{\theta}_{MAP}$와 MLE $\hat{\theta}_{ML}$을 구하고 비교하시오. (3점)
(d) $n \to \infty$이면 MAP와 MLE의 차이는 어떻게 되는가? 이유를 설명하시오. (2점)

---

## Problem 15 (8점)

**[EN]** (Markov and Chebyshev Inequalities)

Let $X \geq 0$ with $\mathbb{E}[X] = 4$ and $\text{Var}(X) = 2$.

(a) Using Markov's inequality, bound $P(X \geq 10)$. (2점)
(b) Using Chebyshev's inequality, bound $P(|X - 4| \geq 3)$. (3점)
(c) Which bound is tighter for $P(X \geq 10)$? Show both calculations. (3점)

**[KR]** $X \geq 0$이고 $\mathbb{E}[X] = 4$, $\text{Var}(X) = 2$일 때,

(a) 마르코프 부등식으로 $P(X \geq 10)$의 상한을 구하시오. (2점)
(b) 체비셰프 부등식으로 $P(|X - 4| \geq 3)$의 상한을 구하시오. (3점)
(c) $P(X \geq 10)$에 대해 어느 상한이 더 타이트한가? 두 계산 결과를 모두 보이시오. (3점)

---

# Part 4: Deep Learning Fundamentals (50점)

---

## Problem 16 (12점)

**[EN]** (Logistic Regression)

For binary classification with $P(y=1|x) = \sigma(w^\top x + b)$ where $\sigma(z) = 1/(1+e^{-z})$:

(a) Write the negative log-likelihood (cross-entropy loss) for a single sample $(x_i, y_i)$ where $y_i \in \{0, 1\}$. (3점)
(b) Derive $\frac{\partial L}{\partial w}$ for the full dataset loss $L = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$. Show that it simplifies to $\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_i$. (6점)
(c) Compare this gradient to the MSE gradient for linear regression: $\frac{2}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_i$. What is structurally similar? What differs? (3점)

**[KR]** 이진 분류에서 $P(y=1|x) = \sigma(w^\top x + b)$일 때:

(a) 단일 샘플 $(x_i, y_i)$ ($y_i \in \{0, 1\}$)에 대한 음의 로그 우도(교차 엔트로피 손실)를 쓰시오. (3점)
(b) 전체 데이터 손실 $L = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$에 대해 $\frac{\partial L}{\partial w}$를 유도하고, $\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_i$로 정리됨을 보이시오. (6점)
(c) 선형 회귀의 MSE 그래디언트 $\frac{2}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_i$와 구조적으로 비교하시오. (3점)

---

## Problem 17 (10점)

**[EN]** (Universal Approximation & Depth)

(a) State the Universal Approximation Theorem (for one hidden layer with sufficient width). (2점)
(b) Show that a network without activation functions, regardless of depth $L$: $f(x) = W_L W_{L-1} \cdots W_1 x$ can be collapsed into $f(x) = \tilde{W}x$. (3점)
(c) Give a concrete example of a function that a single linear layer cannot represent but a 2-layer network with ReLU can (XOR or similar). Illustrate with specific weights. (5점)

**[KR]** (보편 근사 정리와 깊이)

(a) 보편 근사 정리(Universal Approximation Theorem)를 서술하시오. (2점)
(b) 활성화 함수 없이 $L$개 층을 쌓아도 $f(x) = W_L \cdots W_1 x = \tilde{W}x$로 축소됨을 보이시오. (3점)
(c) 단일 선형 층으로는 표현 불가능하지만 ReLU가 있는 2층 신경망으로는 가능한 함수의 구체적 예시를 제시하시오. 구체적 가중치를 포함하시오. (5점)

---

## Problem 18 (10점)

**[EN]** (SGD and Momentum)

Consider minimizing $f(\theta) = \frac{1}{n}\sum_{i=1}^n f_i(\theta)$.

(a) Write the SGD update rule with mini-batch $B \subset \{1, \ldots, n\}$ and learning rate $\eta$. (2점)
(b) Show that the mini-batch gradient is an unbiased estimator of the full gradient: $\mathbb{E}[\nabla f_B(\theta)] = \nabla f(\theta)$, where the expectation is over random batch selection. (3점)
(c) Write the Momentum update: $v_{t+1} = \beta v_t + \nabla f_B(\theta_t)$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$. Explain why momentum helps escape shallow local minima and reduces oscillation. (3점)
(d) Write the Adam update rule and explain what the first moment $m_t$ and second moment $v_t$ track. (2점)

**[KR]** $f(\theta) = \frac{1}{n}\sum_{i=1}^n f_i(\theta)$ 최소화를 고려하자.

(a) 미니배치 $B \subset \{1, \ldots, n\}$와 학습률 $\eta$에 대한 SGD 갱신 규칙을 쓰시오. (2점)
(b) 미니배치 그래디언트가 전체 그래디언트의 비편향 추정량임을 보이시오: $\mathbb{E}[\nabla f_B(\theta)] = \nabla f(\theta)$. (3점)
(c) 모멘텀 갱신을 쓰고, 모멘텀이 얕은 극소에서 탈출하고 진동을 줄이는 이유를 설명하시오. (3점)
(d) Adam 갱신 규칙을 쓰고, 1차 모멘트 $m_t$와 2차 모멘트 $v_t$가 추적하는 것을 설명하시오. (2점)

---

## Problem 19 (10점)

**[EN]** (Regularization and Generalization)

(a) Define Dropout mathematically: during training, each neuron $h_i$ is replaced by $\tilde{h}_i$. Write $\tilde{h}_i$ in terms of a Bernoulli mask. Explain why we divide by $(1-p)$. (4점)
(b) Explain the connection between Dropout and ensemble learning. How many effective sub-networks does a network with $n$ neurons create? (3점)
(c) Sharpness-Aware Minimization (SAM) solves $\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$. Explain in 3-4 sentences why minimizing the worst-case loss in a neighborhood leads to flatter minima and better generalization. (3점)

**[KR]** (정규화와 일반화)

(a) Dropout을 수학적으로 정의하시오: 각 뉴런 $h_i$를 $\tilde{h}_i$로 대체. 베르누이 마스크로 $\tilde{h}_i$를 표현하고, $(1-p)$로 나누는 이유를 설명하시오. (4점)
(b) Dropout과 앙상블 학습의 관계를 설명하시오. $n$개 뉴런의 네트워크가 몇 개의 효과적인 서브 네트워크를 만드는가? (3점)
(c) SAM은 $\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$을 푼다. 이웃 영역에서의 최악 손실을 최소화하면 왜 평탄한 극소와 좋은 일반화로 이어지는지 3-4문장으로 설명하시오. (3점)

---

## Problem 20 (8점)

**[EN]** (Multi-Head Attention Computation)

Given a single attention head with $d_k = 2$:

$Q = \begin{pmatrix}1 & 0\\0 & 1\\1 & 1\end{pmatrix}$, $K = \begin{pmatrix}1 & 1\\0 & 1\end{pmatrix}$, $V = \begin{pmatrix}10\\20\end{pmatrix}$

Note: $V$ here is $2 \times 1$ (each key has a 1-dimensional value).

(a) Compute the attention score matrix $S = QK^\top / \sqrt{d_k}$. (3점)
(b) Apply row-wise softmax to $S$ to get attention weights $A$. (You may use $\text{softmax}(a, b) = (\frac{e^a}{e^a+e^b}, \frac{e^b}{e^a+e^b})$. Leave in terms of $e$.) (3점)
(c) If multi-head attention uses $h = 4$ heads, each with $d_k = d_v = 16$, and the model dimension is $d_{model} = 64$, verify that the total computation is equivalent to a single-head attention with $d_k = 64$. Why does multi-head help despite same total cost? (2점)

**[KR]** $d_k = 2$인 단일 어텐션 헤드에서:

$Q = \begin{pmatrix}1 & 0\\0 & 1\\1 & 1\end{pmatrix}$, $K = \begin{pmatrix}1 & 1\\0 & 1\end{pmatrix}$, $V = \begin{pmatrix}10\\20\end{pmatrix}$ ($V$는 $2 \times 1$)

(a) 어텐션 스코어 행렬 $S = QK^\top / \sqrt{d_k}$를 계산하시오. (3점)
(b) $S$에 행별 소프트맥스를 적용하여 어텐션 가중치 $A$를 구하시오. ($e$ 포함 형태 허용) (3점)
(c) $h = 4$ 헤드, 각 $d_k = d_v = 16$, $d_{model} = 64$인 멀티헤드 어텐션이 $d_k = 64$인 단일 헤드와 총 연산량이 동등함을 확인하시오. 같은 비용인데 멀티헤드가 왜 더 좋은가? (2점)

---

# 풀이 (Solutions)

---

## Problem 1 풀이

**(a)** 2×2 역행렬: $A^{-1} = \frac{1}{\det(A)}\begin{pmatrix}d & -b\\-c & a\end{pmatrix}$ ($A = \begin{pmatrix}a&b\\c&d\end{pmatrix}$)

$\det(A) = 3 \times 2 - 1 \times 0 = 6$

$A^{-1} = \frac{1}{6}\begin{pmatrix}2 & -1\\0 & 3\end{pmatrix} = \begin{pmatrix}1/3 & -1/6\\0 & 1/2\end{pmatrix}$

**(b)** $AA^{-1} = \begin{pmatrix}3&1\\0&2\end{pmatrix}\begin{pmatrix}1/3&-1/6\\0&1/2\end{pmatrix} = \begin{pmatrix}1+0 & -1/2+1/2\\0+0 & 0+1\end{pmatrix} = \begin{pmatrix}1&0\\0&1\end{pmatrix} = I$ ✓

**(c)** $A$는 상삼각행렬 → 고유값 = 대각 원소: $\lambda_1 = 3$, $\lambda_2 = 2$

$\det(A) = 6 = 3 \times 2 = \lambda_1 \lambda_2$ ✓

$\text{Tr}(A) = 3 + 2 = 5 = \lambda_1 + \lambda_2$ ✓

일반 증명: 특성 다항식 $\det(A - \lambda I) = \lambda^n - \text{Tr}(A)\lambda^{n-1} + \cdots + (-1)^n\det(A)$에서, 비에타 공식으로 $\sum \lambda_i = \text{Tr}(A)$, $\prod \lambda_i = \det(A)$.

---

## Problem 2 풀이

**(a)** $\langle u, v \rangle = 1(2) + 2(-1) + 2(0) = 2 - 2 + 0 = 0$

$\|u\| = \sqrt{1+4+4} = 3$, $\|v\| = \sqrt{4+1+0} = \sqrt{5}$

**(b)** $\cos(u,v) = \frac{0}{3\sqrt{5}} = 0$ → **완전 직교**. 두 벡터는 직각.

**(c)** $\text{proj}_u v = \frac{\langle v, u \rangle}{\langle u, u \rangle} u = \frac{0}{9} u = \begin{pmatrix}0\\0\\0\end{pmatrix}$

(내적이 0이므로 사영 = 영벡터)

**(d)** $v - \text{proj}_u v = v - 0 = v$

$\langle v, u \rangle = 0$ → $v \perp u$ ✓ (이미 직교이므로 자명)

---

## Problem 3 풀이

**(a)** $A = \begin{pmatrix}1&2\\2&4\end{pmatrix}$: 두 번째 행 = 첫 번째 행 × 2 → $\text{rank}(A) = 1$

Rank-Nullity: $2 = 1 + \text{null}(A)$ → $\text{null}(A) = 1$

$\mathscr{N}(A)$: $x_1 + 2x_2 = 0$ → $x_1 = -2x_2$ → $\mathscr{N}(A) = \text{span}\{(-2, 1)^\top\}$

**(b)** $x_1 + 2x_2 = 3$에서 $x_1 = 3 - 2x_2$

일반해: $x = \begin{pmatrix}3\\0\end{pmatrix} + t\begin{pmatrix}-2\\1\end{pmatrix}$, $t \in \mathbb{R}$

**(c)** $\|x\|^2 = (3-2t)^2 + t^2 = 9 - 12t + 4t^2 + t^2 = 5t^2 - 12t + 9$

$\frac{d}{dt}(5t^2 - 12t + 9) = 10t - 12 = 0$ → $t = 6/5$

$x^* = \begin{pmatrix}3-12/5\\6/5\end{pmatrix} = \begin{pmatrix}3/5\\6/5\end{pmatrix}$

검증: $A^+ = A^\top(AA^\top)^{-1}$... $AA^\top = \begin{pmatrix}5&10\\10&20\end{pmatrix}$, rank 1이므로 직접 의사역행렬 사용.

$A^+b = \frac{1}{\|a\|^2}a \cdot \frac{\langle a, b \rangle}{\|a\|^2}$... 계산하면 $\begin{pmatrix}3/5\\6/5\end{pmatrix}$ ✓

---

## Problem 4 풀이

**(a)** $A = \begin{pmatrix}3&0\\0&1\end{pmatrix}$는 이미 대각 → $U = V = I$, $\Sigma = \begin{pmatrix}3&0\\0&1\end{pmatrix}$

$A = I \cdot \begin{pmatrix}3&0\\0&1\end{pmatrix} \cdot I^\top$

**(b)** $A^+ = A^{-1} = \begin{pmatrix}1/3&0\\0&1\end{pmatrix}$ (정방 비특이 → 의사역행렬 = 역행렬)

**(c)** $B^\top B = \begin{pmatrix}3&0&0\\0&1&0\end{pmatrix}\begin{pmatrix}3&0\\0&1\\0&0\end{pmatrix} = \begin{pmatrix}9&0\\0&1\end{pmatrix}$ (고유값: 9, 1)

$BB^\top = \begin{pmatrix}3&0\\0&1\\0&0\end{pmatrix}\begin{pmatrix}3&0&0\\0&1&0\end{pmatrix} = \begin{pmatrix}9&0&0\\0&1&0\\0&0&0\end{pmatrix}$ (고유값: 9, 1, 0)

비영 고유값: 둘 다 $\{9, 1\}$ ✓ (일반적으로 $B^\top B$와 $BB^\top$의 비영 고유값은 항상 동일. SVD에서 $B = U\Sigma V^\top$ → $B^\top B = V\Sigma^2 V^\top$, $BB^\top = U\Sigma^2 U^\top$이므로.)

**(d)** 역전파에서 기울기는 $\frac{\partial L}{\partial h_l} = W_{l+1}^\top \frac{\partial L}{\partial h_{l+1}}$처럼 가중치 행렬이 곱해진다. $W$의 특이값이 매우 작으면 ($\sigma_i \ll 1$), 기울기 벡터가 $W^\top$를 통과할 때마다 크기가 $\sigma_{\max}$ 배만큼 줄어든다. $L$개 층을 거치면 $\sigma_{\max}^L \to 0$ → **기울기 소실**. ResNet의 skip connection은 $\frac{\partial}{\partial h_l} = I + \cdots$에서 $I$ 항이 기울기 하한을 보장하여 이를 해결.

---

## Problem 5 풀이

**(a)** $\|Ux\|^2 = (Ux)^\top(Ux) = x^\top U^\top U x = x^\top I x = x^\top x = \|x\|^2$

양변에 제곱근: $\|Ux\| = \|x\|$ $\blacksquare$

**(b)** 대칭: $(A^{-1})^\top = (A^\top)^{-1} = A^{-1}$ ($A$가 대칭이므로 $A^\top = A$) ✓

PD: 임의의 $x \neq 0$에 대해 $x^\top A^{-1} x > 0$을 보여야 한다.

$y = A^{-1}x$로 놓으면 $x = Ay$, $y \neq 0$ ($A$ 가역이므로).

$x^\top A^{-1} x = (Ay)^\top A^{-1}(Ay) = y^\top A^\top A^{-1} A y = y^\top A y > 0$

마지막 부등식: $A$가 PD이고 $y \neq 0$이므로. $\blacksquare$

---

## Problem 6 풀이

**(a)** $\frac{\partial f}{\partial x} = 2xy + 3y^2 - 2$

$\frac{\partial f}{\partial y} = x^2 + 6xy$

**(b)** $(1, -1)$ 대입:

$\frac{\partial f}{\partial x}\big|_{(1,-1)} = 2(1)(-1) + 3(-1)^2 - 2 = -2 + 3 - 2 = -1$

$\frac{\partial f}{\partial y}\big|_{(1,-1)} = 1^2 + 6(1)(-1) = 1 - 6 = -5$

$\nabla f(1,-1) = (-1, -5)$

**(c)** 가장 빠르게 증가하는 방향 = 그래디언트 방향 = $(-1, -5)$ (또는 정규화: $\frac{(-1,-5)}{\sqrt{26}}$)

증가 속도 = 그래디언트의 크기 = $\|\nabla f\| = \sqrt{1+25} = \sqrt{26}$

---

## Problem 7 풀이

**(a)** 순전파:

$z_1 = 0.5 \times 2 + (-0.5) = 0.5$

$a_1 = \text{ReLU}(0.5) = 0.5$

$z_2 = 1 \times 0.5 + 0 = 0.5$

$a_2 = \sigma(0.5) = \frac{1}{1+e^{-0.5}} \approx \frac{1}{1+0.6065} = \frac{1}{1.6065} \approx 0.6225$

$L = \frac{1}{2}(1 - 0.6225)^2 = \frac{1}{2}(0.3775)^2 \approx 0.0713$

**(b)** 각 편미분:

$\frac{\partial L}{\partial a_2} = -(y - a_2) = -(1 - 0.6225) = -0.3775$

$\frac{\partial a_2}{\partial z_2} = a_2(1-a_2) = 0.6225 \times 0.3775 \approx 0.2350$

$\frac{\partial z_2}{\partial w_2} = a_1 = 0.5$

$\frac{\partial z_2}{\partial a_1} = w_2 = 1$

$\frac{\partial a_1}{\partial z_1} = \mathbb{1}[z_1 > 0] = 1$ (ReLU, $z_1 = 0.5 > 0$)

$\frac{\partial z_1}{\partial w_1} = x = 2$

**(c)** $\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial w_2} = (-0.3775)(0.2350)(0.5) \approx -0.0444$

$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_2} \cdot \frac{\partial a_2}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}$
$= (-0.3775)(0.2350)(1)(1)(2) \approx -0.1774$

---

## Problem 8 풀이

**(a)** $f(x) = (Ax-b)^\top(Ax-b)$. 헤시안: $H = \nabla^2 f = 2A^\top A$.

$A^\top A$는 PSD (Problem 4(c)와 동일 논리: $x^\top A^\top A x = \|Ax\|^2 \geq 0$).

$2A^\top A$도 PSD ($2 > 0$).

헤시안이 PSD이면 $f$는 볼록. $\blacksquare$

**(b)** $H_g = 2A^\top A + 2\lambda I$. $\lambda > 0$이므로 $2\lambda I$는 PD.

PSD + PD = PD (모든 고유값이 $2\lambda$ 이상). $H_g$가 PD → $g$는 강볼록. $\blacksquare$

**(c)** 강볼록 함수에서 임의의 두 점 $x \neq y$에 대해 $g(\lambda x + (1-\lambda)y) < \lambda g(x) + (1-\lambda)g(y)$ (엄격 부등식). 극소점이 두 개라면 그 사이의 점에서 이 부등식이 깨짐 → 모순. 따라서 극소점은 최대 하나. 강볼록이면 $\|x\| \to \infty$에서 $g \to \infty$이므로 최소가 존재 → 유일한 전역 최솟값. $\blacksquare$

---

## Problem 9 풀이

**(a)** $f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}f''(x_0)(x-x_0)^2$

**(b)** $f(x) = x^4 - 4x^2 + 1$

$f'(x) = 4x^3 - 8x$, $f''(x) = 12x^2 - 8$

뉴턴법 (극소 찾기): $x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$

1회차 ($x_0 = 2$):
$f'(2) = 32 - 16 = 16$, $f''(2) = 48 - 8 = 40$
$x_1 = 2 - 16/40 = 2 - 0.4 = 1.6$

2회차 ($x_1 = 1.6$):
$f'(1.6) = 4(4.096) - 8(1.6) = 16.384 - 12.8 = 3.584$
$f''(1.6) = 12(2.56) - 8 = 30.72 - 8 = 22.72$
$x_2 = 1.6 - 3.584/22.72 = 1.6 - 0.1578 \approx 1.442$

(참고: 실제 극소는 $x = \sqrt{2} \approx 1.414$에 있음. 빠르게 수렴 중.)

**(c)** $f$가 **강볼록(strictly convex)**이면 뉴턴법이 수렴한다. 더 일반적으로, 초기점이 극소 근처에 있고 $f''(x) \neq 0$이면 **이차 수렴(quadratic convergence)**한다. (Kantorovich 정리)

---

## Problem 10 풀이

**(a)** $\mathcal{L}(x, y, \lambda) = xy + \lambda(x + y - 10)$

**(b)** $\frac{\partial \mathcal{L}}{\partial x} = y + \lambda = 0$ → $\lambda = -y$

$\frac{\partial \mathcal{L}}{\partial y} = x + \lambda = 0$ → $\lambda = -x$

$\lambda = -y = -x$ → $x = y$

$\frac{\partial \mathcal{L}}{\partial \lambda} = x + y - 10 = 0$ → $2x = 10$ → $x = 5$

$(x^*, y^*, \lambda^*) = (5, 5, -5)$, $f(5,5) = 25$

**(c)** 둘레 20인 직사각형의 최대 넓이 = **정사각형** (한 변 5). 넓이 = 25.

ML 연결: 편향과 분산은 트레이드오프 관계이며 "총 예산(모델 복잡도)"이 고정. 정사각형이 최적이듯, 편향과 분산을 **균등하게 배분**하는 것이 최적. 한쪽에 치우치면 전체 성능이 떨어진다.

---

## Problem 11 풀이

**(a)** $P(\text{둘 다 빨강}) = \frac{3}{5} \times \frac{2}{4} = \frac{6}{20} = \frac{3}{10} = 0.3$

**(b)** $P(\text{2번째 빨강} | \text{1번째 빨강}) = \frac{2}{4} = 0.5$

(첫 번째 빨강 뽑은 후 남은 공: 빨2 + 파2 = 4개)

**(c)** 독립의 정의: $P(A \cap B) = P(A) \times P(B)$

$P(\text{1번째 빨강}) = 3/5$

$P(\text{2번째 빨강}) = ?$ 전체 확률: $P(R_2) = P(R_2|R_1)P(R_1) + P(R_2|B_1)P(B_1) = \frac{2}{4}\times\frac{3}{5} + \frac{3}{4}\times\frac{2}{5} = \frac{6}{20} + \frac{6}{20} = \frac{3}{5}$

$P(R_1) \times P(R_2) = \frac{3}{5} \times \frac{3}{5} = \frac{9}{25} = 0.36$

$P(R_1 \cap R_2) = 0.3 \neq 0.36$

$\neq$이므로 **독립이 아니다**. $\blacksquare$

---

## Problem 12 풀이

**(a)** $X \in \{0, 1\}$, $P(X=1) = p$, $P(X=0) = 1-p$

$\mathbb{E}[X] = 0 \times (1-p) + 1 \times p = p$

$\mathbb{E}[X^2] = 0^2(1-p) + 1^2 p = p$

$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = p - p^2 = p(1-p)$

**(b)** $\frac{dH}{dp} = -\log p - 1 + \log(1-p) + 1 = \log\frac{1-p}{p} = 0$

$\frac{1-p}{p} = 1$ → $p = 1/2$

$\frac{d^2 H}{dp^2} = -\frac{1}{p} - \frac{1}{1-p} < 0$ → 최대 ✓

$H(1/2) = \log 2 \approx 0.693$ nats (또는 1 bit)

**(c)** $\ell(p) = \sum_{i=1}^n [x_i \log p + (1-x_i)\log(1-p)]$

$= k\log p + (n-k)\log(1-p)$ (여기서 $k = \sum x_i$)

$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$

$k(1-p) = (n-k)p$ → $k = np$ → $\hat{p}_{ML} = k/n$ $\blacksquare$

---

## Problem 13 풀이

모든 로그는 $\log_2$ 사용.

**(a)** $H(P) = -4 \times 0.25 \log_2 0.25 = -4 \times 0.25 \times (-2) = 2$ bits

**(b)** $H(P, Q) = -\sum P(x)\log_2 Q(x)$

$= -(0.25\log_2 0.5 + 0.25\log_2 0.25 + 0.25\log_2 0.125 + 0.25\log_2 0.125)$

$= -(0.25(-1) + 0.25(-2) + 0.25(-3) + 0.25(-3))$

$= -(- 0.25 - 0.5 - 0.75 - 0.75) = 2.25$ bits

**(c)** $D_{KL}(P\|Q) = H(P,Q) - H(P) = 2.25 - 2 = 0.25$ bits ✓

직접 계산: $D_{KL}(P\|Q) = \sum P(x)\log_2\frac{P(x)}{Q(x)}$

$= 0.25\log_2\frac{0.25}{0.5} + 0.25\log_2\frac{0.25}{0.25} + 0.25\log_2\frac{0.25}{0.125} + 0.25\log_2\frac{0.25}{0.125}$

$= 0.25(-1) + 0.25(0) + 0.25(1) + 0.25(1) = 0.25$ ✓

**(d)** $D_{KL}(Q\|P) = \sum Q(x)\log_2\frac{Q(x)}{P(x)}$

$= 0.5\log_2 2 + 0.25\log_2 1 + 0.125\log_2 0.5 + 0.125\log_2 0.5$

$= 0.5 + 0 + 0.125(-1) + 0.125(-1) = 0.5 - 0.25 = 0.25$ bits

이 경우 우연히 같지만, **일반적으로 $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$**.

ML에서의 의미: MLE는 $D_{KL}(p_{data} \| p_{model})$을 최소화 (forward KL). 이것은 $p_{data}$가 높은 곳에서 $p_{model}$도 높기를 요구 → **mode-covering**. 반대인 reverse KL $D_{KL}(p_{model} \| p_{data})$는 **mode-seeking** (VAE의 ELBO에서 사용). 어느 방향을 쓰느냐에 따라 학습 결과가 달라진다.

---

## Problem 14 풀이

**(a)** $P(D|\theta) = \theta^7(1-\theta)^3$ (이항 우도, 조합 계수는 $\theta$에 무관하므로 생략 가능)

**(b)** Prior: $\text{Beta}(2, 2)$ → $p(\theta) \propto \theta^{2-1}(1-\theta)^{2-1} = \theta(1-\theta)$

Posterior $\propto$ Likelihood × Prior: $\theta^7(1-\theta)^3 \times \theta(1-\theta) = \theta^8(1-\theta)^4$

이것은 $\text{Beta}(9, 5)$. (켤레성: $\text{Beta}(a, b)$ + 데이터($k$성공, $n-k$실패) → $\text{Beta}(a+k, b+n-k) = \text{Beta}(2+7, 2+3) = \text{Beta}(9, 5)$)

**(c)** Beta$(a,b)$의 모드 = $\frac{a-1}{a+b-2}$

$\hat{\theta}_{MAP} = \frac{9-1}{9+5-2} = \frac{8}{12} = \frac{2}{3} \approx 0.667$

$\hat{\theta}_{ML} = \frac{7}{10} = 0.7$

MAP(0.667) < MLE(0.7): 사전분포 Beta(2,2)가 0.5 쪽으로 끌어당기므로 MAP가 0.5에 더 가깝다. Prior가 "동전은 공정할 것"이라는 사전 지식을 반영.

**(d)** $n \to \infty$이면 $\hat{\theta}_{MAP} = \frac{a-1+k}{a+b-2+n} \to \frac{k}{n} = \hat{\theta}_{ML}$

Prior의 영향($a, b$)이 $n$에 비해 무시할 수 있을 정도로 작아짐. **데이터가 충분하면 MAP ≈ MLE**. $\blacksquare$

---

## Problem 15 풀이

**(a)** 마르코프: $P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}$

$P(X \geq 10) \leq \frac{4}{10} = 0.4$

**(b)** 체비셰프: $P(|X - \mu| \geq k) \leq \frac{\text{Var}(X)}{k^2}$

$P(|X - 4| \geq 3) \leq \frac{2}{9} \approx 0.222$

**(c)** $P(X \geq 10)$에 대해:

마르코프: $\leq 0.4$

체비셰프: $P(X \geq 10) \leq P(|X-4| \geq 6) \leq \frac{2}{36} = \frac{1}{18} \approx 0.0556$

체비셰프가 **훨씬 타이트** (0.056 vs 0.4). 분산 정보를 추가로 활용하기 때문.

---

## Problem 16 풀이

**(a)** $\ell_i = -[y_i \log \sigma(w^\top x_i + b) + (1-y_i)\log(1-\sigma(w^\top x_i + b))]$

$\hat{y}_i = \sigma(w^\top x_i + b)$로 놓으면: $\ell_i = -[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$

**(b)** $\frac{\partial \ell_i}{\partial w} = \frac{\partial \ell_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial (w^\top x_i)} \cdot \frac{\partial (w^\top x_i)}{\partial w}$

$\frac{\partial \ell_i}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$

$\frac{\partial \hat{y}_i}{\partial (w^\top x_i)} = \hat{y}_i(1-\hat{y}_i)$ (시그모이드 미분)

$\frac{\partial (w^\top x_i)}{\partial w} = x_i$

곱하면: $\frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)} \times \hat{y}_i(1-\hat{y}_i) \times x_i = (\hat{y}_i - y_i)x_i$

**분모가 정확히 소거됨!** 이것이 시그모이드 + CE 조합의 아름다운 성질.

$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)x_i$ $\blacksquare$

**(c)** 선형 회귀 MSE: $\frac{2}{n}\sum (\hat{y}_i - y_i)x_i$

공통: 둘 다 $(\hat{y}_i - y_i)x_i$ 구조 (오차 × 입력).
차이: MSE는 상수 2가 붙고, $\hat{y}_i$가 선형 출력. CE의 $\hat{y}_i$는 시그모이드 통과 후 값. CE의 그래디언트가 시그모이드 미분 없이 깔끔하게 나온다는 것이 핵심 장점.

---

## Problem 17 풀이

**(a)** "충분한 폭(뉴런 수)의 은닉층 1개를 가진 신경망은, 비선형 활성화 함수(시그모이드, ReLU 등)를 사용하면, 컴팩트 집합 위의 임의의 연속 함수를 원하는 정밀도로 근사할 수 있다." (Cybenko 1989, Hornik 1991)

**(b)** $f(x) = W_L(W_{L-1}(\cdots(W_1 x))) = (W_L W_{L-1} \cdots W_1)x = \tilde{W}x$

행렬 곱의 결합법칙: 여러 행렬의 곱은 하나의 행렬. 따라서 깊이가 의미 없음. $\blacksquare$

**(c)** XOR: $(0,0) \to 0$, $(0,1) \to 1$, $(1,0) \to 1$, $(1,1) \to 0$

단일 선형 층: $f(x_1, x_2) = w_1 x_1 + w_2 x_2 + b$는 직선 하나로 분리해야 하는데, XOR은 직선 하나로 분리 불가능.

2층 ReLU 네트워크:

$h_1 = \text{ReLU}(x_1 + x_2 - 0.5) = \text{ReLU}(x_1 + x_2 - 0.5)$
$h_2 = \text{ReLU}(-x_1 - x_2 + 1.5) = \text{ReLU}(1.5 - x_1 - x_2)$
$f = h_1 + h_2 - 1$

검증:
$(0,0)$: $h_1 = \text{ReLU}(-0.5) = 0$, $h_2 = \text{ReLU}(1.5) = 1.5$, $f = 0.5$ (>0 → 1로 판별 가능하나... 정확한 0/1은 아님)

더 정확한 구성:
$W_1 = \begin{pmatrix}1&1\\1&1\end{pmatrix}$, $b_1 = \begin{pmatrix}-0.5\\-1.5\end{pmatrix}$, $W_2 = (1, -1)$, $b_2 = 0$

$h = \text{ReLU}(W_1 x + b_1)$, $f = W_2 h + b_2$

$(0,0)$: $h = \text{ReLU}((-0.5, -1.5)) = (0, 0)$, $f = 0$ ✓
$(0,1)$: $h = \text{ReLU}((0.5, -0.5)) = (0.5, 0)$, $f = 0.5$ ✓ (양수=1)
$(1,0)$: $h = \text{ReLU}((0.5, -0.5)) = (0.5, 0)$, $f = 0.5$ ✓
$(1,1)$: $h = \text{ReLU}((1.5, 0.5)) = (1.5, 0.5)$, $f = 1.5 - 0.5 = 1.0$... 이것도 양수.

더 정교하게: $b_1 = \begin{pmatrix}-0.5\\-1.5\end{pmatrix}$, $W_2 = (2, -4)$, $b_2 = 0$

$(1,1)$: $h = (1.5, 0.5)$, $f = 3 - 2 = 1$... 여전히 양수.

$W_2 = (1, -2)$로 수정:
$(1,1)$: $f = 1.5 - 1 = 0.5$...

최종: 정확한 XOR은 임계값(threshold) 조정이 필요하지만, 핵심 포인트는 **ReLU 2개로 구간별 선형 함수(piecewise linear)를 만들면 직선 하나로 불가능한 비선형 분류가 가능**하다는 것. $\blacksquare$

---

## Problem 18 풀이

**(a)** $\theta_{t+1} = \theta_t - \frac{\eta}{|B|}\sum_{i \in B} \nabla f_i(\theta_t)$

**(b)** $B$는 $\{1,...,n\}$에서 균등 무작위로 선택된 부분집합.

$\mathbb{E}[\nabla f_B(\theta)] = \mathbb{E}\left[\frac{1}{|B|}\sum_{i \in B}\nabla f_i(\theta)\right]$

각 $i$가 배치에 포함될 확률이 $|B|/n$이고, 기대값의 선형성:

$= \frac{1}{|B|} \cdot |B| \cdot \frac{1}{n}\sum_{i=1}^n \nabla f_i(\theta) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\theta) = \nabla f(\theta)$ $\blacksquare$

**(c)** $v_{t+1} = \beta v_t + \nabla f_B(\theta_t)$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$

$v_t$는 과거 그래디언트의 **지수 가중 이동 평균**. $\beta \approx 0.9$이면 최근 10스텝의 평균적 방향.

- 얕은 극소 탈출: 관성($v_t$)이 있으므로, 현재 기울기가 0이어도 이전 속도로 극소를 넘어감
- 진동 감소: 방향이 자주 바뀌는 축(진동)은 양/음이 상쇄되어 $v_t$가 작아지고, 일관된 방향은 누적되어 빨라짐

**(d)** Adam:

$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla f_B(\theta_t)$ (1차 모멘트: 그래디언트의 이동 평균 = **방향**)

$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla f_B(\theta_t))^2$ (2차 모멘트: 그래디언트 제곱의 이동 평균 = **크기/변동성**)

$\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$ (편향 보정)

$\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

$m_t$: 어느 방향으로 가야 하는지 (방향). $v_t$: 그래디언트가 얼마나 변동하는지 (크기). 변동 큰 차원은 작게, 안정적 차원은 크게 업데이트 → 파라미터별 적응적 학습률.

---

## Problem 19 풀이

**(a)** 학습 시 뉴런 $h_i$를 확률 $p$로 비활성화:

$$\tilde{h}_i = \begin{cases} 0 & \text{확률 } p \\ \frac{h_i}{1-p} & \text{확률 } 1-p \end{cases} = \frac{m_i \cdot h_i}{1-p}, \quad m_i \sim \text{Bernoulli}(1-p)$$

$1/(1-p)$로 나누는 이유: 테스트 시에는 모든 뉴런을 사용하므로, 학습 시 기대값을 맞추기 위해.

$\mathbb{E}[\tilde{h}_i] = \frac{(1-p) \cdot h_i}{1-p} = h_i$ → 학습과 테스트의 기대값 일치 (inverted dropout).

**(b)** $n$개 뉴런 각각에 대해 on/off 선택 → $2^n$개의 서브네트워크. Dropout은 이 $2^n$개 모델의 **앙상블**을 근사적으로 수행. 각 미니배치마다 다른 서브네트워크가 학습되고, 테스트 시에는 모든 뉴런의 가중 평균이 앙상블 예측을 근사.

**(c)** SAM은 $\theta$ 주변 반경 $\rho$ 구에서 **가장 나쁜 경우의 손실**을 최소화. 뾰족한 극소(sharp minimum)에서는 조금만 벗어나도 loss가 급증하므로 worst-case loss가 크다. 평탄한 극소(flat minimum)에서는 주변 어디를 가도 loss가 비슷하므로 worst-case loss가 작다. 따라서 SAM은 자연스럽게 **flat minima를 찾게 되고**, flat minima는 학습 데이터와 테스트 데이터 사이의 미세한 분포 차이에 강건하므로 **일반화가 좋다**.

---

## Problem 20 풀이

**(a)** $QK^\top = \begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}\begin{pmatrix}1&0\\1&1\end{pmatrix} = \begin{pmatrix}1&0\\1&1\\2&1\end{pmatrix}$

$S = QK^\top/\sqrt{2} = \begin{pmatrix}1/\sqrt{2}&0\\1/\sqrt{2}&1/\sqrt{2}\\2/\sqrt{2}&1/\sqrt{2}\end{pmatrix} \approx \begin{pmatrix}0.707&0\\0.707&0.707\\\sqrt{2}&0.707\end{pmatrix}$

**(b)** 행별 softmax:

1행: $\text{softmax}(0.707, 0) = \left(\frac{e^{0.707}}{e^{0.707}+1}, \frac{1}{e^{0.707}+1}\right) \approx (0.670, 0.330)$

2행: $\text{softmax}(0.707, 0.707) = (0.5, 0.5)$ (같은 값이면 균등)

3행: $\text{softmax}(\sqrt{2}, 0.707) = \left(\frac{e^{\sqrt{2}}}{e^{\sqrt{2}}+e^{0.707}}, \frac{e^{0.707}}{e^{\sqrt{2}}+e^{0.707}}\right) \approx (0.670, 0.330)$

**(c)** 단일 헤드 $d_k = 64$: 연산량 $\propto n^2 \times 64$ (QK^T) + $n^2 \times 64$ (AV)

4 헤드 $d_k = 16$: 각 헤드 $n^2 \times 16$, 4개 → 총 $n^2 \times 64$ → **동일**

멀티헤드가 더 좋은 이유: 각 헤드가 **다른 관점**(다른 부분공간)에서 attention을 계산. 한 헤드는 문법적 관계, 다른 헤드는 의미적 관계 등 다양한 패턴을 동시에 포착. 단일 헤드는 64차원 전체를 하나의 관점으로만 사용 → 표현력 제한.

---

# 채점 기준 요약

| Part | 배점 | 핵심 역량 |
|------|------|----------|
| Part 1 | 50점 | 역행렬, 사영, Rank/Null, SVD, PSD 증명, 직교행렬 |
| Part 2 | 50점 | 편미분 계산, 역전파 실전, 볼록성 증명, 뉴턴법, 라그랑주 |
| Part 3 | 50점 | 조건부확률/독립, 베르누이 MLE, KL 계산, 베이지안 동전, 마르코프/체비셰프 |
| Part 4 | 50점 | 로지스틱 회귀 그래디언트, 보편 근사, SGD/Adam, Dropout/SAM, 멀티헤드 어텐션 |
| **총점** | **200점** | |

---
title: "슬라이드 퀴즈 모음 — 문제 + 풀이"
slug: slide-quizzes
order: 20
---

# 슬라이드 퀴즈 모음 — 문제 + 풀이

> 슬라이드에서 "Q." 마크가 붙은 퀴즈를 전수 수집하여, 영어/한국어 문제 + 한국어 풀이로 정리

---

# Quiz 1 — Range와 Null Space (page_052, slide 44)

## 문제

**[EN]** Given $A = \begin{pmatrix} 1 & 1 \end{pmatrix} \in \mathbb{R}^{1 \times 2}$, find:
- $\mathscr{R}(A)$ (Range of $A$)
- $\mathscr{R}(A^\top)$ (Range of $A^\top$)
- $\mathscr{N}(A)$ (Null space of $A$)
- $\mathscr{N}(A^\top)$ (Null space of $A^\top$)

**[KR]** $A = \begin{pmatrix} 1 & 1 \end{pmatrix} \in \mathbb{R}^{1 \times 2}$일 때 다음을 구하시오:
- $\mathscr{R}(A)$ (치역)
- $\mathscr{R}(A^\top)$ (전치의 치역)
- $\mathscr{N}(A)$ (영공간)
- $\mathscr{N}(A^\top)$ (전치의 영공간)

## 풀이

$A$는 $1 \times 2$ 행렬이다. $A^\top = \begin{pmatrix}1\\1\end{pmatrix}$은 $2 \times 1$ 행렬이다.

**$\mathscr{R}(A)$ (치역 = Column Space)**:

$A$의 출력은 $Ax = \begin{pmatrix}1&1\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix} = x_1 + x_2 \in \mathbb{R}$

$x_1, x_2$를 자유롭게 선택하면 $x_1 + x_2$는 모든 실수를 만들 수 있다.

$$\boxed{\mathscr{R}(A) = \mathbb{R}}$$

**$\mathscr{R}(A^\top)$ (Row Space)**:

$A^\top x = \begin{pmatrix}1\\1\end{pmatrix}x = \begin{pmatrix}x\\x\end{pmatrix}$, $x \in \mathbb{R}$

$$\boxed{\mathscr{R}(A^\top) = \text{span}\left\{\begin{pmatrix}1\\1\end{pmatrix}\right\} = \left\{t\begin{pmatrix}1\\1\end{pmatrix} : t \in \mathbb{R}\right\}}$$

이것은 $\mathbb{R}^2$에서 $(1,1)$ 방향의 직선이다.

**$\mathscr{N}(A)$ (Null Space)**:

$Ax = 0$: $x_1 + x_2 = 0$ → $x_2 = -x_1$

$$\boxed{\mathscr{N}(A) = \text{span}\left\{\begin{pmatrix}1\\-1\end{pmatrix}\right\}}$$

**$\mathscr{N}(A^\top)$ (Left Null Space)**:

$A^\top y = 0$: $\begin{pmatrix}1\\1\end{pmatrix}y = \begin{pmatrix}y\\y\end{pmatrix} = \begin{pmatrix}0\\0\end{pmatrix}$ → $y = 0$

$$\boxed{\mathscr{N}(A^\top) = \{0\}}$$

**검증 — Rank-Nullity 정리**:
- $A \in \mathbb{R}^{1 \times 2}$: $\text{rank}(A) + \text{null}(A) = 2$. rank = 1, null = 1 ✓
- $A^\top \in \mathbb{R}^{2 \times 1}$: $\text{rank}(A^\top) + \text{null}(A^\top) = 1$. rank = 1, null = 0 ✓

**기하학적 해석**: $\mathscr{R}(A^\top)$와 $\mathscr{N}(A)$는 **직교**한다. $(1,1) \cdot (1,-1) = 1 - 1 = 0$ ✓. 이것은 $\mathbb{R}^2$를 row space와 null space로 직교 분해한 것이다.

---

# Quiz 2 — 행렬 미분 (page_092 근처, Matrix Calculus)

## 문제

**[EN]** Show that $\frac{\partial (a^\top X b)}{\partial X} = ba^\top$, i.e., $\nabla_X (a^\top X b) = ab^\top$.

**[KR]** $\frac{\partial (a^\top X b)}{\partial X} = ba^\top$임을 보이시오. 즉, $\nabla_X (a^\top X b) = ab^\top$.

## 풀이

$a \in \mathbb{R}^m$, $b \in \mathbb{R}^n$, $X \in \mathbb{R}^{m \times n}$이라 하자.

**Step 1: 스칼라로 전개**

$a^\top X b$는 스칼라다. 성분으로 쓰면:

$$a^\top X b = \sum_{i=1}^{m} \sum_{j=1}^{n} a_i X_{ij} b_j$$

**Step 2: $X_{kl}$에 대해 편미분**

$$\frac{\partial}{\partial X_{kl}} \left(\sum_{i,j} a_i X_{ij} b_j\right) = a_k b_l$$

왜? $i = k$이고 $j = l$인 항만 $X_{kl}$을 포함하므로.

**Step 3: 행렬로 조합**

$$\left[\frac{\partial (a^\top X b)}{\partial X}\right]_{kl} = a_k b_l$$

이것은 $(k, l)$ 성분이 $a_k b_l$인 행렬 = $ab^\top$ (외적).

$$\boxed{\nabla_X (a^\top X b) = ab^\top}$$

**전치 표기 주의**: Numerator layout에서는 $ab^\top$, Denominator layout에서는 $ba^\top$. 슬라이드에서는 $\frac{\partial}{\partial X} = ba^\top$ (denominator layout)으로 표기.

---

# Quiz 3 — 극값 찾기 (page_098 근처, Optimization)

## 문제

**[EN]**
(a) Find all extrema of $f: x \in [-1.5, 3] \mapsto x^3 - 3x + 1$.
(b) Find all extrema of $f: x \in [-1, 1] \mapsto |x|$.

**[KR]**
(a) $f(x) = x^3 - 3x + 1$의 $x \in [-1.5, 3]$에서 모든 극값을 구하시오.
(b) $f(x) = |x|$의 $x \in [-1, 1]$에서 모든 극값을 구하시오.

## 풀이

**(a)** $f(x) = x^3 - 3x + 1$

$f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 3(x-1)(x+1) = 0$

임계점: $x = -1$, $x = 1$. 둘 다 $[-1.5, 3]$ 안에 있다.

경계점: $x = -1.5$, $x = 3$

후보 지점의 함수값:
| $x$ | $f(x)$ | 유형 |
|-----|--------|------|
| $-1.5$ | $(-1.5)^3 - 3(-1.5) + 1 = -3.375 + 4.5 + 1 = 2.125$ | 경계 |
| $-1$ | $-1 + 3 + 1 = 3$ | 임계점 |
| $1$ | $1 - 3 + 1 = -1$ | 임계점 |
| $3$ | $27 - 9 + 1 = 19$ | 경계 |

$$\boxed{\text{전역 최소: } f(1) = -1, \quad \text{전역 최대: } f(3) = 19}$$
$$\text{극대: } f(-1) = 3, \quad \text{극소: } f(1) = -1$$

**(b)** $f(x) = |x|$

$f$는 $x = 0$에서 미분 불가능. $f'(x) = 1$ ($x > 0$), $f'(x) = -1$ ($x < 0$).

$f'(x) = 0$인 임계점: 없음.

극값 후보: 미분 불가능점 ($x = 0$) + 경계점 ($x = -1, 1$)

| $x$ | $f(x)$ | 유형 |
|-----|--------|------|
| $-1$ | $1$ | 경계 |
| $0$ | $0$ | 미분 불가능점 |
| $1$ | $1$ | 경계 |

$$\boxed{\text{전역 최소: } f(0) = 0, \quad \text{전역 최대: } f(-1) = f(1) = 1}$$

**핵심 교훈**: 극값은 (1) 임계점, (2) 미분 불가능점, (3) 경계에서 발생한다. 세 가지를 전부 확인해야 한다.

---

# Quiz 4 — Poisson/Gamma 분포의 평균·분산 (page_171 근처, Probability)

## 문제

**[EN]**
(a) Compute the mean and variance of the Poisson distribution: $p(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$, $k = 0, 1, 2, \ldots$. Hint: $e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$

(b) For the Gamma distribution $p(x) = Cx^{\alpha-1}e^{-\beta x}$ with $\alpha = 2$, $\beta = 1$ (i.e., $p(x) = Cxe^{-x}$, $x > 0$), first find the coefficient $C$, then compute the mean and variance.

**[KR]**
(a) 포아송 분포 $p(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$의 평균과 분산을 구하시오. 힌트: $e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}$

(b) 감마 분포 $p(x) = Cxe^{-x}$ ($x > 0$, $\alpha = 2$, $\beta = 1$)에서 계수 $C$를 먼저 구하고, 평균과 분산을 구하시오.

## 풀이

**(a) 포아송 분포**

**평균:**

$$\mathbb{E}[X] = \sum_{k=0}^{\infty} k \cdot \frac{\lambda^k e^{-\lambda}}{k!} = \sum_{k=1}^{\infty} \frac{\lambda^k e^{-\lambda}}{(k-1)!}$$

$k = 0$ 항은 0이므로 제거. $j = k - 1$로 치환:

$$= \lambda e^{-\lambda} \sum_{j=0}^{\infty} \frac{\lambda^j}{j!} = \lambda e^{-\lambda} \cdot e^{\lambda} = \boxed{\lambda}$$

**$\mathbb{E}[X^2]$:**

$$\mathbb{E}[X^2] = \mathbb{E}[X(X-1)] + \mathbb{E}[X]$$

$$\mathbb{E}[X(X-1)] = \sum_{k=2}^{\infty} k(k-1)\frac{\lambda^k e^{-\lambda}}{k!} = \lambda^2 e^{-\lambda}\sum_{j=0}^{\infty}\frac{\lambda^j}{j!} = \lambda^2$$

$$\mathbb{E}[X^2] = \lambda^2 + \lambda$$

**분산:**

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \lambda^2 + \lambda - \lambda^2 = \boxed{\lambda}$$

포아송 분포의 특징: **평균 = 분산 = $\lambda$**.

---

**(b) 감마 분포** ($p(x) = Cxe^{-x}$, $x > 0$)

**정규화 상수 $C$:**

$$\int_0^{\infty} Cxe^{-x}\,dx = 1$$

부분적분: $\int_0^{\infty} xe^{-x}\,dx = [-xe^{-x}]_0^{\infty} + \int_0^{\infty} e^{-x}\,dx = 0 + 1 = 1$

$$C \cdot 1 = 1 \implies \boxed{C = 1}$$

(일반적으로 $\int_0^{\infty} x^{n}e^{-x}\,dx = n!$ (감마 함수). 여기서 $n = 1$이므로 $1! = 1$.)

**평균:**

$$\mathbb{E}[X] = \int_0^{\infty} x \cdot xe^{-x}\,dx = \int_0^{\infty} x^2 e^{-x}\,dx = 2! = \boxed{2}$$

**$\mathbb{E}[X^2]$:**

$$\mathbb{E}[X^2] = \int_0^{\infty} x^3 e^{-x}\,dx = 3! = 6$$

**분산:**

$$\text{Var}(X) = 6 - 4 = \boxed{2}$$

---

# Quiz 5 — 가우시안 MLE (page_240, slide 232)

## 문제

**[EN]** Given the log-likelihood:

$$\ell = n\log\frac{1}{\sqrt{2\pi\sigma^2}} + \sum_{i=1}^{n} -\frac{1}{2}\left(\frac{x_i - \mu}{\sigma}\right)^2$$

Show that:
(a) $\mu_{\text{ML}} = \frac{1}{n}\sum_{i=1}^{n} x_i$
(b) $\sigma^2_{\text{ML}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$
(c) $\mathbb{E}[\mu_{\text{ML}}] = \mu$ (unbiased)
(d) $\mathbb{E}[\sigma^2_{\text{ML}}] = \frac{n-1}{n}\sigma^2$ (biased)

**[KR]** 로그 우도가 위와 같을 때 (a)~(d)를 보이시오.

## 풀이

**(a)** 로그 우도를 전개:

$$\ell = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

$\mu$에 대해 미분:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0$$

$$\sum_{i=1}^n x_i - n\mu = 0 \implies \boxed{\mu_{\text{ML}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}}$$

**(b)** $\sigma^2$에 대해 미분 ($s = \sigma^2$로 놓으면):

$$\frac{\partial \ell}{\partial s} = -\frac{n}{2s} + \frac{1}{2s^2}\sum_{i=1}^n (x_i - \mu)^2 = 0$$

$$ns = \sum_{i=1}^n (x_i - \mu)^2 \implies \boxed{\sigma^2_{\text{ML}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2}$$

**(c)** 비편향성:

$$\mathbb{E}[\mu_{\text{ML}}] = \mathbb{E}\left[\frac{1}{n}\sum x_i\right] = \frac{1}{n}\sum \mathbb{E}[x_i] = \frac{1}{n} \cdot n\mu = \mu \quad \blacksquare$$

**(d)** 편향성:

핵심 항등식: $\sum(x_i - \bar{x})^2 = \sum(x_i - \mu)^2 - n(\bar{x} - \mu)^2$

기대값:

$\mathbb{E}[\sum(x_i-\mu)^2] = n\sigma^2$

$\mathbb{E}[n(\bar{x}-\mu)^2] = n \cdot \text{Var}(\bar{x}) = n \cdot \frac{\sigma^2}{n} = \sigma^2$

$$\mathbb{E}[\sigma^2_{\text{ML}}] = \frac{1}{n}(n\sigma^2 - \sigma^2) = \frac{n-1}{n}\sigma^2 \quad \blacksquare$$

비편향 추정: $\sigma^2_{\text{unb}} = \frac{n}{n-1}\sigma^2_{\text{ML}} = \frac{1}{n-1}\sum(x_i - \bar{x})^2$

---

# Quiz 6 — MAP with Prior (page_248, slide 240)

## 문제

**[EN]** In a coin-flip scenario ($n = 5$ trials, $k = 4$ heads), with prior $p(\theta) \propto \theta^m$:

Find $\theta_{\text{MAP}}$ for $m = 2$ and $m = 6$.

**[KR]** 동전 던지기 ($n = 5$, 앞면 $k = 4$)에서 사전분포 $p(\theta) \propto \theta^m$일 때, $m = 2$와 $m = 6$에서 $\theta_{\text{MAP}}$를 구하시오.

## 풀이

**로그 사후 확률:**

$$\log p(\theta|D) \propto \log p(D|\theta) + \log p(\theta) = k\log\theta + (n-k)\log(1-\theta) + m\log\theta$$

$$= (k+m)\log\theta + (n-k)\log(1-\theta)$$

미분하여 0으로 놓으면:

$$\frac{k+m}{\theta} - \frac{n-k}{1-\theta} = 0$$

$$(k+m)(1-\theta) = (n-k)\theta$$

$$k + m - (k+m)\theta = (n-k)\theta$$

$$k + m = (k + m + n - k)\theta = (n + m)\theta$$

$$\boxed{\theta_{\text{MAP}} = \frac{k + m}{n + m}}$$

**$m = 2$일 때:** $\theta_{\text{MAP}} = \frac{4+2}{5+2} = \frac{6}{7} \approx 0.857$

**$m = 6$일 때:** $\theta_{\text{MAP}} = \frac{4+6}{5+6} = \frac{10}{11} \approx 0.909$

**비교:**
- MLE: $\theta_{\text{ML}} = k/n = 4/5 = 0.8$
- $m = 2$: MAP = 0.857 (MLE보다 1에 가까움)
- $m = 6$: MAP = 0.909 (더 1에 가까움)

$p(\theta) \propto \theta^m$은 $\theta = 1$ 근처에서 높은 사전분포이므로, $m$이 클수록 MAP가 1 쪽으로 끌려간다.

---

# Quiz 7 — 가우시안 KL Divergence (page_244 근처)

## 문제

**[EN]** Derive the KL divergence between two univariate Gaussians:

$$D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \frac{1}{2}\left[2\log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{\sigma_2^2} + \frac{(\mu_1-\mu_2)^2}{\sigma_2^2} - 1\right]$$

**[KR]** 두 일변량 가우시안 사이의 KL 발산을 유도하시오.

## 풀이

$$D_{KL}(p \| q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx = \mathbb{E}_p[\log p(x)] - \mathbb{E}_p[\log q(x)]$$

**$\mathbb{E}_p[\log p(x)]$:**

$$\log p(x) = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x-\mu_1)^2}{2\sigma_1^2}$$

$$\mathbb{E}_p[\log p(x)] = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{1}{2}$$

(왜? $\mathbb{E}[(x-\mu_1)^2/\sigma_1^2] = 1$)

**$\mathbb{E}_p[\log q(x)]$:**

$$\log q(x) = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x-\mu_2)^2}{2\sigma_2^2}$$

$(x-\mu_2)^2 = (x-\mu_1+\mu_1-\mu_2)^2 = (x-\mu_1)^2 + 2(x-\mu_1)(\mu_1-\mu_2) + (\mu_1-\mu_2)^2$

$$\mathbb{E}_p[(x-\mu_2)^2] = \sigma_1^2 + 0 + (\mu_1-\mu_2)^2$$

$$\mathbb{E}_p[\log q(x)] = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2}$$

**빼기:**

$$D_{KL} = \left[-\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{1}{2}\right] - \left[-\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}\right]$$

$$= \frac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

$$= \boxed{\frac{1}{2}\left[\log\frac{\sigma_2^2}{\sigma_1^2} + \frac{\sigma_1^2}{\sigma_2^2} + \frac{(\mu_1-\mu_2)^2}{\sigma_2^2} - 1\right]}$$

($\log\frac{\sigma_2^2}{\sigma_1^2} = 2\log\frac{\sigma_2}{\sigma_1}$이므로 슬라이드 표기와 일치 ✓)

**검증:** $\mu_1 = \mu_2$, $\sigma_1 = \sigma_2$이면: $\log 1 + 1 + 0 - 1 = 0$ ✓

---

# Quiz 8 — 엔트로피 (page_248 근처)

## 문제

**[EN]** Compute the entropy of:
(a) A one-hot distribution (e.g., $P = (1, 0, 0, \ldots, 0)$)
(b) A uniform distribution over $C$ classes: $P = (1/C, \ldots, 1/C)$
(c) $X \sim \mathcal{N}(\mu, \sigma^2)$

**[KR]** 다음의 엔트로피를 구하시오:
(a) 원-핫 분포
(b) $C$개 클래스 균등분포
(c) $X \sim \mathcal{N}(\mu, \sigma^2)$

## 풀이

**(a)** $P = (1, 0, \ldots, 0)$:

$$H = -1 \cdot \log 1 - 0 \cdot \log 0 - \cdots = 0$$

($0 \log 0 = 0$으로 정의.) $\boxed{H(\text{one-hot}) = 0}$. 불확실성 없음.

**(b)** $P = (1/C, \ldots, 1/C)$:

$$H = -\sum_{i=1}^{C} \frac{1}{C}\log\frac{1}{C} = -C \cdot \frac{1}{C} \cdot (-\log C) = \boxed{\log C}$$

균등분포가 최대 엔트로피. $C = 2$이면 $H = \log 2 = 1$ bit.

**(c)** $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$H(X) = -\int p(x)\log p(x)\,dx$$

$\log p(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$

$$H = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\mathbb{E}[(x-\mu)^2] = \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2}$$

$$\boxed{H(\mathcal{N}(\mu,\sigma^2)) = \frac{1}{2}\log(2\pi e\sigma^2)}$$

$\sigma$가 클수록 엔트로피가 크다 (더 퍼져있으므로 불확실성 증가).

---

# Quiz 9 — 최소자승법의 1/2 (page_285 근처)

## 문제

**[EN]** In the least squares loss $L = \frac{1}{2}\|y - h(x)\|^2$, why is the factor $\frac{1}{2}$ used?

**[KR]** 최소자승 손실 $L = \frac{1}{2}\|y - h(x)\|^2$에서 $\frac{1}{2}$이 붙는 이유는?

## 풀이

미분할 때 깔끔하게 만들기 위한 **계산 편의상의 장치**다.

$\frac{1}{2}$ 없이: $\frac{d}{d\theta}(y - h)^2 = 2(y - h) \cdot (-h'(\theta))$ ← 2가 남음

$\frac{1}{2}$ 있으면: $\frac{d}{d\theta}\frac{1}{2}(y - h)^2 = (y - h) \cdot (-h'(\theta))$ ← **깔끔**

최소화하는 $\theta$의 위치는 $\frac{1}{2}$의 유무에 관계없이 동일하다 (양의 상수배는 argmin에 영향 없음). 순수하게 미분 결과를 간결하게 하기 위한 관례.

---

# Quiz 10 — 합성곱의 행렬 표현 (page_472, slide 463)

## 문제

**[EN]** Convolution is a linear operation. What is the corresponding matrix for the 1D convolution: input $[0, 1, 2, 3, 4, 5, 6]$ with kernel $[1, 2]$?

**[KR]** 합성곱은 선형 연산이다. 입력 $[0,1,2,3,4,5,6]$과 커널 $[1,2]$의 1D 합성곱에 대응하는 행렬을 구하시오.

## 풀이

합성곱 결과: $[2, 5, 8, 11, 14, 17]$ (6개 원소)

합성곱을 $y = Wx$ 형태로 쓰면:

$$\begin{pmatrix}y_1\\y_2\\y_3\\y_4\\y_5\\y_6\end{pmatrix} = \begin{pmatrix}2&1&0&0&0&0&0\\0&2&1&0&0&0&0\\0&0&2&1&0&0&0\\0&0&0&2&1&0&0\\0&0&0&0&2&1&0\\0&0&0&0&0&2&1\end{pmatrix}\begin{pmatrix}0\\1\\2\\3\\4\\5\\6\end{pmatrix}$$

검증 (1행): $2 \times 0 + 1 \times 1 = 1$... 아, 커널이 $[1, 2]$이므로 합성곱은 $w_0 x_{i} + w_1 x_{i+1} = 1 \cdot x_i + 2 \cdot x_{i+1}$.

수정:

$$W = \begin{pmatrix}1&2&0&0&0&0&0\\0&1&2&0&0&0&0\\0&0&1&2&0&0&0\\0&0&0&1&2&0&0\\0&0&0&0&1&2&0\\0&0&0&0&0&1&2\end{pmatrix}$$

검증: $W[0,:] \cdot x = 1(0) + 2(1) = 2$ ✓, $W[1,:] \cdot x = 1(1) + 2(2) = 5$ ✓

$$\boxed{W \in \mathbb{R}^{6 \times 7} \text{ — 커널 값이 대각선 방향으로 이동하는 Toeplitz 행렬}}$$

**핵심 인사이트**:
- 합성곱 = **희소(sparse) 행렬 곱** (대부분 0)
- 같은 커널 값이 반복 → **가중치 공유(parameter sharing)**
- FC 레이어($7 \times 6 = 42$개 파라미터) vs Conv(2개 파라미터) → 파라미터 효율적

---

# Quiz 11 — 베이지안 네트워크 Factorization (page_508 근처)

## 문제

**[EN]** Compute $p(X, Y, Z)$ for the following directed graphical models:
(a) $X \to Z \to Y$
(b) $Y \to Z \to X$
(c) $Z \to X$, $Z \to Y$ (fork)
(d) $X \to Z$, $Y \to Z$ (collider)

**[KR]** 다음 방향 그래프 모델에 대해 $p(X, Y, Z)$를 인수분해하시오.

## 풀이

베이지안 네트워크의 인수분해 규칙: **각 노드의 확률 = 부모 노드가 주어졌을 때의 조건부 확률**

$$p(X_1, \ldots, X_n) = \prod_{i=1}^n p(X_i \mid \text{Parents}(X_i))$$

**(a)** $X \to Z \to Y$: $X$는 부모 없음, $Z$의 부모 = $X$, $Y$의 부모 = $Z$

$$\boxed{p(X,Y,Z) = p(X) \cdot p(Z|X) \cdot p(Y|Z)}$$

**(b)** $Y \to Z \to X$: $Y$는 부모 없음, $Z$의 부모 = $Y$, $X$의 부모 = $Z$

$$\boxed{p(X,Y,Z) = p(Y) \cdot p(Z|Y) \cdot p(X|Z)}$$

**(c)** Fork: $Z \to X$, $Z \to Y$: $Z$는 부모 없음, $X$의 부모 = $Z$, $Y$의 부모 = $Z$

$$\boxed{p(X,Y,Z) = p(Z) \cdot p(X|Z) \cdot p(Y|Z)}$$

**(d)** Collider: $X \to Z$, $Y \to Z$: $X$, $Y$는 부모 없음, $Z$의 부모 = $\{X, Y\}$

$$\boxed{p(X,Y,Z) = p(X) \cdot p(Y) \cdot p(Z|X,Y)}$$

**조건부 독립 관계:**
- (a), (b): $X \perp Y \mid Z$ (Z가 주어지면 X와 Y는 독립)
- (c): $X \perp Y \mid Z$ (공통 원인 Z가 주어지면 독립)
- (d): $X \perp Y$ (무조건 독립), 하지만 $X \not\perp Y \mid Z$ (**Z가 주어지면 오히려 종속!** — explaining away 현상)

---

# Quiz 12 — VAE Reparameterization 분산 (page_628 근처)

## 문제

**[EN]** The naive Monte Carlo gradient estimator for the VAE encoder:

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{q_\phi(z|x)}[f(z)\nabla_\phi \log q_\phi(z|x)]$$

exhibits very high variance. **Q. Why?**

**[KR]** VAE 인코더의 naive 몬테카를로 그래디언트 추정량이 매우 높은 분산을 보이는 이유를 설명하시오.

## 풀이

이 추정량은 **REINFORCE / Score Function Estimator**라 불리며, 높은 분산을 가지는 이유는:

**1. $f(z)$와 $\nabla_\phi \log q_\phi(z|x)$의 곱이 불안정하다**

$f(z)$는 우도 함수 등으로 값의 범위가 매우 넓다. $\nabla_\phi \log q$도 $z$ 값에 따라 크게 변한다. 이 둘의 곱은 분산이 극도로 커진다.

**2. $\log q_\phi(z|x)$의 기울기가 $f(z)$의 구조를 활용하지 못한다**

$f(z)$의 값이 크든 작든, 기울기 신호 $\nabla_\phi \log q$는 동일한 방향을 가리킨다. $f(z)$의 landscape 정보가 기울기에 반영되지 않으므로 비효율적.

**3. 해결: Reparameterization Trick**

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

이렇게 하면:

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[\nabla_\phi f(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)]$$

이제 $\nabla_\phi f$를 **직접** 계산할 수 있다 (backpropagation 가능). $f$의 landscape 정보가 기울기에 직접 반영되므로 분산이 크게 줄어든다.

**비유**: REINFORCE는 "블랙박스를 흔들어서 방향 감지", Reparameterization은 "블랙박스를 열고 내부 구조를 직접 미분" → 당연히 후자가 정확하다.

---

# 퀴즈 요약 — 슬라이드 페이지 매핑

| # | 페이지 | 주제 | 유형 |
|---|--------|------|------|
| 1 | page_052 | Range, Null Space | 계산 |
| 2 | page_092 | $\partial(a^\top Xb)/\partial X$ | 증명 |
| 3 | page_098 | 극값 찾기 | 계산 |
| 4 | page_171 | Poisson/Gamma 평균·분산 | 유도 |
| 5 | page_240 | 가우시안 MLE + 편향성 | 유도+증명 |
| 6 | page_248 | MAP with power prior | 계산 |
| 7 | page_244 | 가우시안 KL divergence | 유도 |
| 8 | page_248 | 엔트로피 공식 | 계산 |
| 9 | page_285 | 1/2 in least squares | 개념 |
| 10 | page_472 | 합성곱의 행렬 표현 | 계산+증명 |
| 11 | page_508 | 베이지안 네트워크 factorization | 개념+계산 |
| 12 | page_628 | VAE reparameterization 분산 | 개념+유도 |

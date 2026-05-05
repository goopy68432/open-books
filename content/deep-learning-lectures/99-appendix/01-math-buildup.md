---
title: "A1. 수학 빌드업 — 중1 산술 → 대학원 수식"
slug: math-buildup
order: 1
---

# A1. 수학 빌드업 — 중1 산술 → 대학원 수식

> **목적**: 모든 토픽이 의존하는 수학 도구를 한 곳에. 시험 중에 \"이 미분 어떻게 했더라?\" 할 때 빠르게 참조.

---

## 1. 산술 / 대수 (중1-3)

### 1.1 거듭제곱

$$
a^m \cdot a^n = a^{m+n}, \quad (a^m)^n = a^{mn}, \quad a^0 = 1, \quad a^{-n} = \frac{1}{a^n}
$$

### 1.2 분수

$$
\frac{a}{b} \cdot \frac{c}{d} = \frac{ac}{bd}, \quad \frac{a}{b} \div \frac{c}{d} = \frac{ad}{bc}
$$

### 1.3 등식 변형 (양변에 같은 것)

> 시험 풀이의 90% 는 \"양변에 X 곱하기/나누기\" 의 반복.

---

## 2. 함수와 그래프 (고1)

### 2.1 함수 정의

$f: X \to Y$, $x \mapsto f(x)$. 입력 → 출력 규칙.

### 2.2 합성함수

$(f \circ g)(x) = f(g(x))$. 안쪽 먼저, 바깥쪽 나중.

### 2.3 역함수

$f(f^{-1}(x)) = x$. 단조함수만 역함수 존재.

---

## 3. 로그 (고1) — 딥러닝의 핵심 도구

### 3.1 정의

$$
\log_a x = y \iff a^y = x
$$

ML 에서는 $a = e$ (자연로그) 가 표준 — 쓸 때 그냥 \"$\log$\".

### 3.2 핵심 성질 — 외워야 함

$$
\log(ab) = \log a + \log b
$$
$$
\log(a^n) = n \log a
$$
$$
\log(a/b) = \log a - \log b
$$
$$
\log 1 = 0, \quad \log e = 1, \quad \log(e^x) = x
$$

### 3.3 단조증가 → argmax 보존

$L > 0$ 에 대해 $\arg\max L = \arg\max \log L$.

> 💡 \"왜 NLL 에서 log 를 취하는가\" 의 답.

---

## 4. 미분 (고2-대1)

### 4.1 정의

$$
f'(x) = \lim_{h\to 0} \frac{f(x+h) - f(x)}{h}
$$

직관: \"$x$ 에서의 기울기\".

### 4.2 기본 미분 공식 — 외우기

| $f(x)$ | $f'(x)$ |
|--------|---------|
| $c$ (상수) | $0$ |
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\log x$ | $1/x$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |

### 4.3 곱·몫·합성

$$
(uv)' = u'v + uv'
$$
$$
\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}
$$
$$
(f(g(x)))' = f'(g(x)) \cdot g'(x) \quad \text{(chain rule)}
$$

### 4.4 시그모이드 / ReLU / Softmax 미분

$$
\sigma(x) = \frac{1}{1+e^{-x}}, \quad \sigma'(x) = \sigma(x)(1-\sigma(x))
$$

$$
\mathrm{ReLU}(x) = \max(0, x), \quad \mathrm{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

Softmax: [06 토픽](06_Softmax_Jacobian.md).

### 4.5 극값 조건

- $f'(\theta^*) = 0$ : 필요조건 (extremum 후보)
- $f''(\theta^*) > 0$ : 최솟값
- $f''(\theta^*) < 0$ : 최댓값

---

## 5. 적분 (대1)

### 5.1 기본 적분

$$
\int x^n\, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)
$$
$$
\int \frac{1}{x}\, dx = \log|x| + C
$$
$$
\int e^x\, dx = e^x + C
$$

### 5.2 Gaussian 적분 (★ 시험 출제 중간 Q2-2)

$$
\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}
$$

일반화:

$$
\int_{-\infty}^{\infty} e^{-ax^2}\, dx = \sqrt{\frac{\pi}{a}}, \quad a > 0
$$

증명: 2차원 좌표 → 극좌표 트릭. (시험에서 직접 증명 요구는 드물지만 결과는 외우기)

### 5.3 Gaussian 모멘트

$X \sim \mathcal{N}(0,1)$:

$$
E[X] = 0, \quad E[X^2] = 1, \quad E[X^3] = 0, \quad E[X^4] = 3
$$

(짝수 차수: $(2k-1)!! = 1\cdot 3\cdot 5 \cdots$)

---

## 6. 벡터 / 행렬 (대1-대학원)

### 6.1 벡터 연산

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \sum_i u_i v_i \quad \text{(inner product)}
$$
$$
\|\mathbf{v}\|^2 = \mathbf{v}^\top\mathbf{v} = \sum_i v_i^2
$$

### 6.2 행렬 곱

$(AB)_{ij} = \sum_k A_{ik} B_{kj}$. 차원: $(m\times k)(k\times n) = (m\times n)$.

### 6.3 Transpose

$(AB)^\top = B^\top A^\top$ (순서 바뀜!).

### 6.4 역행렬

$AA^{-1} = A^{-1}A = I$. 가역 ⟺ $\det A \neq 0$ ⟺ full rank.

### 6.5 Vector Gradient (대학원)

$$
\nabla_\theta(\mathbf{a}^\top\theta) = \mathbf{a}
$$
$$
\nabla_\theta(\theta^\top A \theta) = (A + A^\top)\theta = 2A\theta \quad (\text{symmetric } A)
$$
$$
\nabla_\theta\|\mathbf{y} - X\theta\|^2 = -2X^\top(\mathbf{y} - X\theta)
$$

### 6.6 Hessian

$\nabla^2 L \in \mathbb{R}^{d\times d}$, $(\nabla^2 L)_{ij} = \frac{\partial^2 L}{\partial\theta_i\partial\theta_j}$.

PSD ⟺ $\mathbf{v}^\top H \mathbf{v} \geq 0 \;\forall \mathbf{v}$.
PD  ⟺ $\mathbf{v}^\top H \mathbf{v} > 0 \;\forall \mathbf{v} \neq 0$.

---

## 7. 확률 / 통계

### 7.1 기댓값과 분산

$$
E[X] = \sum_x x\, P(x) = \int x\, p(x)\, dx
$$
$$
\mathrm{Var}(X) = E[(X - E[X])^2] = E[X^2] - E[X]^2
$$

### 7.2 IID

$X_1, \ldots, X_n$ **IID** ⟺ 같은 분포 + 독립.

독립 → joint = 곱:

$$
P(X_1, \ldots, X_n) = \prod_i P(X_i)
$$

> 💡 NLL 의 모든 곱 표현은 IID 가 정당화.

### 7.3 주요 분포

| 분포 | PMF/PDF | 평균 | 분산 |
|------|---------|------|------|
| Bernoulli($\theta$) | $\theta^x(1-\theta)^{1-x}$ | $\theta$ | $\theta(1-\theta)$ |
| Categorical($\mathbf{p}$) | $\prod_j p_j^{y_j}$ (one-hot) | — | — |
| Uniform[a,b] | $\frac{1}{b-a}$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Gaussian($\mu, \sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$ | $\mu$ | $\sigma^2$ |

### 7.4 조건부 / 베이즈

[01 베이즈 정리](01_베이즈정리_증명.md) 참조.

---

## 8. Convex 함수와 Jensen 부등식

### 8.1 Convex 정의

$f$ convex ⟺ $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \; \lambda \in [0,1]$.

→ 그래프가 \"아래로 볼록\".

### 8.2 2차 미분 판별

$f''(x) \geq 0$ for all $x$ ⟺ convex.

### 8.3 Jensen

$f$ convex, $X$ 확률변수:

$$
f(E[X]) \leq E[f(X)]
$$

[13 KL Divergence](13_KL_Divergence.md) 의 양수성 증명에 사용.

---

## 9. 자주 쓰는 공식 모음 (시험 직전 cheatsheet)

| 상황 | 공식 |
|------|------|
| Bernoulli MLE | $\theta^* = k/n$ |
| Bernoulli MAP (prior $\theta^M(1-\theta)^M$) | $\theta^* = (k+M)/(n+2M)$ |
| Gaussian likelihood NLL | $\propto \sum (y_i - h(x_i))^2$ + const |
| Cross-entropy | $H(p,q) = -\sum p_i \log q_i$ |
| KL divergence | $\mathrm{KL}(p\|q) = \sum p_i \log(p_i/q_i)$ |
| Linear regression closed form | $\theta^* = (X^\top X)^{-1} X^\top \mathbf{y}$ |
| Newton update | $\theta_{t+1} = \theta_t - H^{-1}\mathbf{g}$ |
| GD update | $\theta_{t+1} = \theta_t - \eta \mathbf{g}$ |
| Softmax derivative | $\partial s_i/\partial z_j = s_i(\delta_{ij} - s_j)$ |
| Softmax+CE derivative | $\partial L/\partial z_k = \hat p_k - y_k$ |
| Output size formula | $O = \lfloor (W-K+2P)/S\rfloor + 1$ |

---

## 10. 잘 안 외워지는 것 — Mnemonics

- **Bayes**: \"Posterior is Likelihood times Prior over Evidence.\" (\"PLPE\" 외우기)
- **NLL→MSE**: \"**exp / minus / square**\" — Gaussian의 세 마법
- **Softmax+CE 미분**: \"$\hat p$ minus $y$\" — 단순한 차이
- **Newton**: \"2차 근사의 minimum 으로 점프\"
- **Output size**: \"등차수열 항 수 = $(b-a)/d + 1$\"

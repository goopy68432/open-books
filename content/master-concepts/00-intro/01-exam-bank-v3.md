---
title: "기출문제 통합 풀이집 v3 — 작년 100% + 올해 8문제"
slug: exam-bank-v3
order: 1
---

# 기출문제 통합 풀이집 v3 — 작년 100% + 올해 8문제

> **★★★★★ 작년 시험 문제는 100% 출제됨 ★★★★★**
>
> **출처:** 작년 시험 5장 이미지 + 올해 8개 기출문제 + 강의 스크립트
>
> **사용법:** Tier 1 (작년)을 절대 마스터, Tier 2 (올해)를 7단계로 풀고, Tier 3 (변형)으로 대응력 강화.

---

## 📋 문제 분류 — 3-Tier 시스템

| Tier | 출처 | 출제 확률 | 풀이 강도 |
|------|------|---------|---------|
| **Tier 1** | 작년 시험 (이미지 5장) | **100% (확정)** | 풀이 패턴 그대로 외움 |
| **Tier 2** | 올해 기출 8문제 | 매우 높음 | 7단계 체인 마스터 |
| **Tier 3** | 강의 추가 토픽 | 변형 출제 가능 | 대응력 |

---

# 🥇 TIER 1: 작년 시험 (★★★★★ 100% 출제)

## T1-1. Convex 증명 4유형 ★★★★★

### 출제 형태 (작년)
> *1. Convex 관련*
> - $f(x) = x^2$이 convex임을 증명
> - $f(x) = -\log(x)$가 convex임을 증명
> - 주어진 함수가 convex function인지 판별/증명
> - Convex set 관련 문제

### 1-A. $f(x) = x^2$ 이 convex임을 증명

**[정의]** 함수 $f$가 **convex** ⇔
$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y), \quad \forall x, y, \forall \lambda \in [0, 1]$$

**[방법 1: 정의로 직접 증명]**

좌변:
$$f(\lambda x + (1-\lambda)y) = (\lambda x + (1-\lambda)y)^2$$
$$= \lambda^2 x^2 + 2\lambda(1-\lambda)xy + (1-\lambda)^2 y^2$$

우변:
$$\lambda f(x) + (1-\lambda)f(y) = \lambda x^2 + (1-\lambda)y^2$$

차이 (우변 − 좌변):
$$\lambda x^2 + (1-\lambda)y^2 - \lambda^2 x^2 - 2\lambda(1-\lambda)xy - (1-\lambda)^2 y^2$$
$$= \lambda(1-\lambda)x^2 - 2\lambda(1-\lambda)xy + \lambda(1-\lambda)y^2$$
$$= \lambda(1-\lambda)(x-y)^2 \geq 0 \quad \because \lambda(1-\lambda) \geq 0, (x-y)^2 \geq 0$$

따라서 $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$. **∴ $f(x) = x^2$는 convex.** ∎

**[방법 2: 2계 미분 ≥ 0]**

$f''(x) = 2 > 0 \forall x \Rightarrow f$ 는 strictly convex. ∎

---

### 1-B. $f(x) = -\log(x)$ 이 convex임을 증명 ($x > 0$)

**[방법 1: 2계 미분]**

$$f'(x) = -\frac{1}{x}, \quad f''(x) = \frac{1}{x^2} > 0 \quad (x > 0)$$

따라서 strictly convex on $(0, \infty)$. ∎

**[방법 2: 정의 (Jensen)]**

$x_1, x_2 > 0$, $\lambda \in [0,1]$에 대해 산술-기하 평균 부등식:
$$\lambda x_1 + (1-\lambda)x_2 \geq x_1^\lambda x_2^{1-\lambda}$$

양변에 $-\log$ (단조감소):
$$-\log(\lambda x_1 + (1-\lambda)x_2) \leq -\log(x_1^\lambda x_2^{1-\lambda}) = -\lambda \log x_1 - (1-\lambda)\log x_2$$
$$= \lambda f(x_1) + (1-\lambda)f(x_2) \quad \blacksquare$$

---

### 1-C. 임의 함수의 Convex 판별 — 표준 절차

| 도구 | 조건 | 비고 |
|------|------|------|
| **정의** | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ | 일반화, 항상 가능 |
| **1차 조건** | $f(y) \geq f(x) + f'(x)(y-x)$ | 미분 가능 + 1번 미분 |
| **2차 조건 (1변수)** | $f''(x) \geq 0$ | 미분 가능 + 2번 미분 |
| **2차 조건 (다변수)** | Hessian $H(f)$ 양반정치 (PSD) | $\mathbf{v}^T H \mathbf{v} \geq 0 \forall \mathbf{v}$ |

**예시: $f(x_1, x_2) = x_1^2 + x_2^2$**

Hessian:
$$H = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

$\mathbf{v}^T H \mathbf{v} = 2v_1^2 + 2v_2^2 \geq 0$ → PSD → convex. ∎

**시험 답안 패턴:**
> "$f$의 2계 미분(Hessian)을 계산: ... . 이는 PSD(또는 양수)이므로 $f$는 convex이다."

---

### 1-D. Convex Set

**[정의]** 집합 $C$가 **convex** ⇔
$$\forall x, y \in C, \forall \lambda \in [0,1]: \lambda x + (1-\lambda)y \in C$$

(두 점을 잇는 선분이 모두 C 안)

**예시:**
- ✅ Convex: $\mathbb{R}^n$, 원, 구, 직육면체, 반평면 $\{x : a^T x \leq b\}$, hyperplane
- ❌ Non-convex: 도넛, 별 모양, $\{x : \Vert x\Vert  = 1\}$ (구 표면)

**유용한 사실:**
- Convex 집합의 **교집합**은 convex (합집합은 일반적으로 X)
- Affine map $f(x) = Ax + b$로 convex 집합을 보내면 convex
- Convex 함수의 **하위레벨 집합** $\{x : f(x) \leq c\}$은 convex

---

## T1-2. KL Divergence + Jensen ★★★★★

### 출제 형태 (작년)
> *2. KL Divergence / Jensen's Inequality*
> *4. KL Divergence 계산*
> - KL divergence가 항상 ≥ 0임을 증명
> - Jensen's inequality 증명
> - 정규분포의 KL divergence 계산
> - $P \sim N(\mu_1, 1), Q \sim N(\mu_2, 1)$일 때 KL(P||Q) 구하기

### 2-A. Jensen 부등식 증명

**[정리]** $f$가 볼록(convex)이면:
$$f(E[X]) \leq E[f(X)]$$

**[증명 — 이산형 두 점]**

$X = a$ 확률 $p$, $X = b$ 확률 $1-p$.

$E[X] = pa + (1-p)b$.

볼록 정의:
$$f(pa + (1-p)b) \leq p f(a) + (1-p) f(b) = E[f(X)] \quad \blacksquare$$

**[일반화]** 측도론 + 측도 분할. (시험 답안: 두 점 증명 + "측도론적 일반화로 확장 가능" 한 줄.)

---

### 2-B. KL Divergence ≥ 0 증명

**[정의]**
$$\text{KL}(p \Vert  q) = \sum_i p_i \log\frac{p_i}{q_i} = E_p\left[\log\frac{p(X)}{q(X)}\right]$$

(연속형: $\int p \log(p/q) dx$)

**[증명 — Jensen 사용]**

$\log$는 **오목** ($\log''(x) = -1/x^2 < 0$).
오목함수의 Jensen: $E[\log Y] \leq \log E[Y]$.

$Y = q(X)/p(X)$, $X \sim p$:

$$-\text{KL}(p\Vert q) = E_p\left[\log\frac{q(X)}{p(X)}\right] \leq \log E_p\left[\frac{q(X)}{p(X)}\right]$$

내부 계산:
$$E_p[q/p] = \int p(x) \cdot \frac{q(x)}{p(x)}\,dx = \int q(x)\,dx = 1$$

따라서:
$$-\text{KL}(p\Vert q) \leq \log 1 = 0 \Rightarrow \boxed{\text{KL}(p\Vert q) \geq 0}$$

**등호 ⇔** $p = q$ a.e. ∎

---

### 2-C. ★ 정규분포 KL Divergence 계산 (작년 직접 출제)

**[문제]** $P \sim N(\mu_1, 1)$, $Q \sim N(\mu_2, 1)$일 때 $\text{KL}(P \Vert  Q)$ = ?

**[풀이]**

**1단계: pdf 명시**
$$p(x) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(x-\mu_1)^2}{2}\right), \quad q(x) = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{(x-\mu_2)^2}{2}\right)$$

**2단계: log p/q 계산**
$$\log\frac{p(x)}{q(x)} = -\frac{(x-\mu_1)^2}{2} + \frac{(x-\mu_2)^2}{2}$$

전개:
$$= \frac{1}{2}\left[(x-\mu_2)^2 - (x-\mu_1)^2\right]$$
$$= \frac{1}{2}\left[(\mu_1 - \mu_2)\cdot(2x - \mu_1 - \mu_2)\right]$$
$$= \frac{1}{2}\left[2(\mu_1-\mu_2)x - (\mu_1^2 - \mu_2^2)\right]$$
$$= (\mu_1-\mu_2)x - \frac{\mu_1^2 - \mu_2^2}{2}$$

**3단계: KL 적분 — 기댓값 형태**
$$\text{KL}(P\Vert Q) = E_p[\log(p/q)] = (\mu_1-\mu_2)E_p[X] - \frac{\mu_1^2 - \mu_2^2}{2}$$

$E_p[X] = \mu_1$이므로:
$$= (\mu_1-\mu_2)\mu_1 - \frac{\mu_1^2 - \mu_2^2}{2}$$
$$= \mu_1^2 - \mu_1\mu_2 - \frac{\mu_1^2}{2} + \frac{\mu_2^2}{2}$$
$$= \frac{\mu_1^2}{2} - \mu_1\mu_2 + \frac{\mu_2^2}{2}$$
$$= \frac{1}{2}(\mu_1 - \mu_2)^2$$

**[최종 답]**
$$\boxed{\text{KL}(P \Vert  Q) = \frac{1}{2}(\mu_1 - \mu_2)^2}$$

**[직관]** 평균 차이가 클수록 두 분포가 더 멀어진다. KL은 거리 같은 양 (단, 비대칭).

---

### 2-D. 일반 정규분포 KL (확장)

$P \sim N(\mu_1, \sigma_1^2)$, $Q \sim N(\mu_2, \sigma_2^2)$:
$$\text{KL}(P\Vert Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

(시험에서 직접 출제는 분산 동일 케이스. 이 일반식은 답안에 보너스로 인용 가능.)

---

## T1-3. Bias-Variance Decomposition ★★★★★

### 출제 형태 (작년)
> *3. Bias-Variance Decomposition*
> - Bias-variance decomposition
> - Test error가 Bias² + Variance로 분해됨을 증명하기

### 3-A. 분해 정리

**[정리]** 진짜 함수 $f(x)$, $y = f(x) + \epsilon$, $\epsilon \sim N(0, \sigma^2)$. 학습된 $\hat{f}_D$.
$$E_{D, \epsilon}\left[(y_0 - \hat{f}_D(x_0))^2\right] = \underbrace{(\bar{f} - f)^2}_{\text{Bias}^2} + \underbrace{E_D[(\bar{f} - \hat{f}_D)^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

여기서 $\bar{f}(x_0) = E_D[\hat{f}_D(x_0)]$.

### 3-B. 단계별 증명

**1단계: 분해 트릭**

$y_0 = f + \epsilon$, $\hat{f}_D = \hat{f}$로 약식. $\bar{f}$를 더하고 빼기:
$$y_0 - \hat{f}_D = (f - \bar{f}) + (\bar{f} - \hat{f}_D) + \epsilon$$

**2단계: 제곱 후 기댓값**
$$E[(y_0 - \hat{f})^2] = E[\{(f-\bar{f}) + (\bar{f}-\hat{f}) + \epsilon\}^2]$$

전개:
$$= E[(f-\bar{f})^2] + E[(\bar{f}-\hat{f})^2] + E[\epsilon^2]$$
$$+ 2E[(f-\bar{f})(\bar{f}-\hat{f})] + 2E[(f-\bar{f})\epsilon] + 2E[(\bar{f}-\hat{f})\epsilon]$$

**3단계: 교차항 3개 모두 0**

(i) $f, \bar{f}$는 D 무관 + ε와 독립:
$$E[(f-\bar{f})\epsilon] = (f-\bar{f}) \cdot E[\epsilon] = 0$$

(ii) $\hat{f}_D$는 D에 의존하지만 ε는 새로운 잡음 (독립):
$$E[(\bar{f}-\hat{f}_D)\epsilon] = E[\bar{f}-\hat{f}_D] \cdot E[\epsilon] = 0$$

(iii) $f - \bar{f}$는 D 무관 상수:
$$E[(f-\bar{f})(\bar{f} - \hat{f}_D)] = (f-\bar{f}) \cdot E[\bar{f} - \hat{f}_D]$$

$E[\bar{f}] = \bar{f}$ (정의), $E[\hat{f}_D] = \bar{f}$ → $E[\bar{f}-\hat{f}_D] = 0$.

**4단계: 남은 3항**
$$E[(y_0 - \hat{f})^2] = (f - \bar{f})^2 + E[(\bar{f}-\hat{f})^2] + \sigma^2$$
$$= \text{Bias}^2 + \text{Variance} + \text{Noise} \quad \blacksquare$$

### 3-C. 시험 답안 핵심 표현

> *"$\bar{f}$를 더하고 빼서 분해. 교차항 3개 모두 ε 또는 D-무관성으로 0. 남은 3항이 Bias² + Variance + Noise."*

---

## T1-4. ★ Gradient Descent 수렴 조건 — $\eta < 2/\lambda_{\max}(A)$ ★★★★★

### 출제 형태 (작년)
> *4. Gradient Descent 수렴 조건*
> *6. 학습률 조건*
> - $0 < \eta < 2/\lambda_{\max}(A)$ 조건에서 gradient descent의 loss가 감소함을 증명
> - 학습률이 $2/\lambda_{\max}(A)$보다 작아야 학습 가능한 이유 증명

### 4-A. 문제 설정

**Quadratic loss:**
$$L(\theta) = \frac{1}{2}\theta^T A \theta - b^T \theta + c, \quad A \in \mathbb{R}^{n \times n} \text{ symmetric PSD}$$

(또는 $L(\theta) = \frac{1}{2}\Vert y - X\theta\Vert ^2$도 quadratic.)

**그래디언트:** $\nabla L(\theta) = A\theta - b$.

**최적해:** $A\theta^* = b$.

**Gradient Descent 업데이트:**
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) = \theta_t - \eta(A\theta_t - b)$$

### 4-B. 단계별 증명

**1단계: 오차 정의**
$$e_t := \theta_t - \theta^*$$

**2단계: 오차의 재귀식**

$\theta_{t+1} = \theta_t - \eta(A\theta_t - b)$이고 $b = A\theta^*$이므로:
$$\theta_{t+1} - \theta^* = \theta_t - \theta^* - \eta(A\theta_t - A\theta^*)$$
$$e_{t+1} = e_t - \eta A e_t = (I - \eta A)e_t$$

**3단계: 손실 감소 조건**

$L$의 quadratic 형태에서:
$$L(\theta) - L(\theta^*) = \frac{1}{2}(\theta - \theta^*)^T A (\theta - \theta^*) = \frac{1}{2}e^T A e$$

따라서:
$$L(\theta_{t+1}) - L^* = \frac{1}{2}e_{t+1}^T A e_{t+1} = \frac{1}{2}e_t^T (I-\eta A)^T A (I-\eta A) e_t$$

$A$ 대칭이므로 $A$와 $(I - \eta A)$ 교환 가능:
$$= \frac{1}{2}e_t^T A (I - \eta A)^2 e_t$$

**4단계: 고유분해**

$A$ symmetric PSD → 직교 대각화: $A = Q\Lambda Q^T$, $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

$\tilde{e}_t := Q^T e_t$로 두면:
$$L(\theta_{t+1}) - L^* = \frac{1}{2}\tilde{e}_t^T \Lambda(I - \eta\Lambda)^2 \tilde{e}_t = \frac{1}{2}\sum_i \lambda_i (1 - \eta\lambda_i)^2 \tilde{e}_{t,i}^2$$

**5단계: 감소 조건 도출**

$L(\theta_{t+1}) < L(\theta_t)$가 모든 $e_t$에 대해 성립하려면, **각 i에 대해**:
$$\lambda_i (1-\eta\lambda_i)^2 < \lambda_i \cdot 1$$
$$(1-\eta\lambda_i)^2 < 1$$
$$|1 - \eta\lambda_i| < 1$$
$$-1 < 1 - \eta\lambda_i < 1$$
$$0 < \eta\lambda_i < 2$$
$$0 < \eta < \frac{2}{\lambda_i}$$

**모든 i에 대해 성립**해야 하므로:
$$\boxed{0 < \eta < \frac{2}{\lambda_{\max}(A)}}$$

### 4-C. 직관

- $\lambda_{\max}$는 가장 가파른 방향의 곡률 (Hessian 최대 고유값).
- 학습률이 $2/\lambda_{\max}$ 이상이면 가장 가파른 방향에서 **발산**.
- 작은 학습률은 모든 방향에서 안전.

### 4-D. 시험 답안 핵심 흐름

> "Quadratic loss에서 GD 업데이트를 오차 $e_t = \theta_t - \theta^*$ 형태로 쓰면 $e_{t+1} = (I - \eta A)e_t$. $A$의 직교 대각화 후 각 고유방향에서 $|1 - \eta\lambda_i| < 1$가 필요. 가장 큰 $\lambda_{\max}$에 대해 성립해야 모든 방향 수렴 → $\eta < 2/\lambda_{\max}$."

---

## T1-5. Convolution 연산 + Matrix 형태 ★★★★★

### 출제 형태 (작년)
> *5. Convolution*
> - Convolution 연산 계산
> - 1D, 2D에서 $H_{in}, k, p, s$가 주어졌을 때 $H_{out}$ 및 corresponding matrix 찾기

### 5-A. 1D Convolution 정의

입력 $\mathbf{x} = (x_1, \ldots, x_n)$, 커널 $\mathbf{w} = (w_1, \ldots, w_k)$.

**Stride = 1, padding = 0:**
$$y_i = \sum_{j=1}^k w_j x_{i+j-1}, \quad i = 1, \ldots, n-k+1$$

### 5-B. 출력 크기 공식 (★ 시험 직접 출제)

**1D:**
$$\boxed{H_{\text{out}} = \frac{H_{\text{in}} - k + 2p}{s} + 1}$$

**2D (사각형):**
$$H_{\text{out}} = \frac{H_{\text{in}} - k + 2p}{s} + 1, \quad W_{\text{out}} = \frac{W_{\text{in}} - k + 2p}{s} + 1$$

여기서:
- $H_{\text{in}}$ = 입력 크기
- $k$ = 커널 크기
- $p$ = padding (양쪽 추가)
- $s$ = stride

**예시:** $H_{\text{in}} = 7, k = 3, p = 1, s = 2$
$$H_{\text{out}} = \frac{7 - 3 + 2}{2} + 1 = \frac{6}{2} + 1 = 4$$

### 5-C. Matrix 형태 (Toeplitz)

1D conv with $\mathbf{x} = (x_1, x_2, x_3, x_4)$, $\mathbf{w} = (w_1, w_2, w_3)$, stride=1, no padding:

$$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \end{pmatrix} = \begin{pmatrix} w_1 & w_2 & w_3 & 0 \\ 0 & w_1 & w_2 & w_3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix}$$

**행렬 형태:** $\mathbf{y} = W\mathbf{x}$, where $W$는 **Toeplitz 행렬** (대각 위에 같은 값 반복).

### 5-D. 2D Convolution Matrix (Im2col)

2D 입력 $H \times W$, kernel $k \times k$ → $im2col$로 펼친 후 matrix multiplication.

**일반 패턴:**
$$\mathbf{Y} = \mathbf{W}_{\text{flat}} \cdot \text{Im2Col}(\mathbf{X})$$

(시험에서는 1D 형태로 충분히 매트릭스 표현 가능.)

### 5-E. Convolution의 Inductive Bias

> *"이미지를 다룬다 = Hypothesis Space를 제약 = Prior knowledge"*

- **Translation Equivariance**: 입력 이동 → 출력도 이동
- **Parameter Sharing**: 같은 커널을 모든 위치에 적용 → 파라미터 절감
- **Local Receptive Field**: 인접 영역만 참조

---

## T1-6. Markov Chain / DAG ★★★★

### 출제 형태 (작년)
> *6. Markov Chain / 확률 그래프 모델*
> - Markov chain의 conditional probability 구하기 (3가지 유형)
> - DAG 보고 joint distribution을 conditional probability의 곱으로 나타내기

### 6-A. Markov Property

**[정의]** 시퀀스 $X_1, X_2, \ldots, X_T$가 Markov ⇔
$$P(X_{t+1} | X_1, X_2, \ldots, X_t) = P(X_{t+1} | X_t)$$

(미래는 현재만 의존, 과거 무관)

### 6-B. Joint Probability 분해

**Chain Rule of Probability:**
$$P(X_1, X_2, \ldots, X_T) = P(X_1) \prod_{t=2}^T P(X_t | X_1, \ldots, X_{t-1})$$

**Markov 가정 적용:**
$$\boxed{P(X_1, \ldots, X_T) = P(X_1) \prod_{t=2}^T P(X_t | X_{t-1})}$$

### 6-C. DAG (Directed Acyclic Graph) — Bayesian Network

**[규칙]** DAG가 주어지면:
$$P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))$$

**예시 — DAG: $A \to B, A \to C, B \to D, C \to D$**
$$P(A, B, C, D) = P(A) \cdot P(B|A) \cdot P(C|A) \cdot P(D|B, C)$$

### 6-D. 조건부확률 3가지 유형

**유형 1: Forward — 다음 상태 예측**
$$P(X_3 | X_2) \quad (\text{직접 transition})$$

**유형 2: Backward — 과거 추론**
$$P(X_1 | X_T) \quad (\text{Bayes로 뒤집기})$$
$$P(X_1 | X_T) = \frac{P(X_T | X_1)P(X_1)}{P(X_T)}$$

**유형 3: 임의 두 시점**
$$P(X_t | X_{t+k}) \quad \text{by Bayes + transition matrix powers}$$

---

## T1-7. 모델 비교 — Markov / RNN / Transformer ★★★★

### 출제 형태 (작년)
> *7. 모델 비교*
> - Markov Chain vs RNN vs Transformer: inductive bias 및 필요 데이터 양 비교
> - RNN/Transformer 차이

### 7-A. 비교 표

| 측면 | Markov Chain | RNN | Transformer |
|------|------------|-----|-------------|
| **State** | 마지막 상태만 | 은닉 상태 (모든 과거 압축) | 명시적 토큰 시퀀스 |
| **의존성** | 1차 (이웃만) | 이론적으로 모든 과거 | 모든 위치 직접 |
| **Inductive Bias** | 매우 강함 (Markov 가정) | 강함 (시간 순서) | 약함 (위치 인코딩) |
| **파라미터** | 적음 (transition matrix) | 중간 (gate weights) | 많음 (Q,K,V) |
| **데이터 요구** | 적음 | 중간 | **많음** |
| **병렬화** | - | 어려움 (시간 순차) | **쉬움** (모든 위치 동시) |
| **장기 의존성** | 약함 (1차) | Vanishing 문제 | 강함 (직접 attention) |

### 7-B. Inductive Bias 핵심

**Markov:** "미래 = 현재만 의존" (가장 강한 제약)
**RNN:** "현재 출력 = 모든 과거를 압축한 hidden state로 결정"
**Transformer:** "각 위치는 모든 위치와 attention" (가장 약한 제약)

### 7-C. 데이터 양과의 관계

> Inductive Bias가 강할수록 → 적은 데이터로도 일반화 잘됨.
> 약할수록 → 많은 데이터 필요 (Transformer는 그래서 BIG DATA 필요).

### 7-D. RNN vs Transformer 핵심 차이

| | RNN | Transformer |
|---|-----|------------|
| 처리 방식 | 순차 (sequential) | 병렬 (parallel) |
| 장기 의존성 | Vanishing gradient | Self-attention으로 직접 |
| 학습 속도 | 느림 | 빠름 (GPU 병렬) |
| 메모리 | $O(L)$ hidden | $O(L^2)$ attention matrix |

---

## T1-8. PyTorch 코드 흐름 ★★★

### 출제 형태 (작년)
> *8. PyTorch / 코드 흐름*
> - PyTorch forward → backward까지의 함수 나열하기
> - Backpropagation 코드 흐름 정렬

### 8-A. 표준 학습 루프 5단계

```python
for input, target in dataset:
    # 1. Gradient 초기화
    optimizer.zero_grad()

    # 2. Forward Pass
    output = model(input)

    # 3. Loss 계산
    loss = loss_fn(output, target)

    # 4. Backward Pass (자동 미분)
    loss.backward()

    # 5. Parameter 업데이트
    optimizer.step()
```

### 8-B. 각 단계의 의미

| 단계 | 함수 | 역할 |
|-----|------|------|
| 1 | `optimizer.zero_grad()` | 이전 step의 gradient 누적 제거 |
| 2 | `model(input)` (= `forward`) | 입력 → 출력 (예측) |
| 3 | `loss_fn(output, target)` | 예측 ↔ 정답 비교 |
| 4 | `loss.backward()` | 자동 미분 (autograd) — 모든 파라미터의 grad 계산 |
| 5 | `optimizer.step()` | $\theta \leftarrow \theta - \eta \nabla L$ |

### 8-C. 시험 답안 형태

> "PyTorch 학습 루프는: ① zero_grad (기존 gradient 초기화) → ② forward (예측) → ③ loss 계산 → ④ backward (자동 미분으로 모든 ∂L/∂θ 계산) → ⑤ optimizer.step (파라미터 업데이트). 이 5단계가 모든 신경망 학습의 표준."

---

## T1-9. Pooling Matrix 형태 ★★★★

### 출제 형태 (작년)
> *7. Pooling*
> - Averaging Pooling 연산을 Matrix로 나타내기
> - Avg pooling 구하기

### 9-A. Average Pooling 정의

**1D, kernel = 2, stride = 2:**
$$y_i = \frac{x_{2i-1} + x_{2i}}{2}$$

### 9-B. Matrix 형태

입력 $\mathbf{x} = (x_1, x_2, x_3, x_4)$, kernel = 2, stride = 2:
$$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}\begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix}$$

행렬 P:
$$P = \frac{1}{k}\begin{pmatrix} \mathbf{1}_k & \mathbf{0} & \cdots \\ \mathbf{0} & \mathbf{1}_k & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

(여기서 $\mathbf{1}_k$는 길이 k의 1-벡터)

### 9-C. 2D Average Pooling

2×2 평균 풀링은 4개를 평균. 매트릭스 형태로 펼치면 $(1/4)$ 가중치의 sparse matrix.

### 9-D. Max Pooling은 비선형

Max pooling은 행렬 곱으로 표현 불가능. **Avg pooling만이 linear** (matrix 표현 가능).

---

## T1-10. 손실함수 관계 (MSE / NLL / CE / KL) ★★★★★

### 출제 형태 (작년)
> *3. 손실함수 관계 서술*
> - MSE와 NLL의 관계 설명 (+ 해당하는 분포 서술)
> - NLL과 KL divergence의 관계 설명
> - MSE = NLL = CE = KL 전체 관계 증명

### 10-A. 통합 흐름도

```
                  분포 가정
                     ↓
                  Likelihood
                     ↓ (i.i.d → 곱)
                     L
                     ↓ (-log)
                    NLL
              ────┬────────┬────
              ↓        ↓
         [Gauss noise]  [Categorical]
              ↓        ↓
            MSE       CE = H(p) + KL(p||q)
                              ↑
                    p 고정이면 H(p) 상수
                              ↓
                  CE 최소화 ⇔ KL 최소화
```

### 10-B. MSE = NLL under Gauss

$y = f + \epsilon$, $\epsilon \sim N(0, \sigma^2)$:
$$\text{NLL} = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum(y_i - f_i)^2$$

→ 상수 + $\frac{1}{2\sigma^2}$ × MSE.

$\arg\min$에서 상수와 양의 스칼라 무시:
$$\arg\min \text{NLL} = \arg\min \text{MSE}$$

### 10-C. CE = NLL for Categorical

$y$ one-hot, $\mathbf{p} = $ 모델 예측 확률:
$$\text{NLL} = -\sum_i y_i \log p_i = \text{CE}$$

### 10-D. CE = H(p) + KL(p||q)

**증명:**
$$\text{CE}(p, q) = -\sum_i p_i \log q_i$$
$$= -\sum_i p_i \log\frac{q_i}{p_i} - \sum_i p_i \log p_i$$
$$= \sum_i p_i \log\frac{p_i}{q_i} + H(p)$$
$$= \text{KL}(p\Vert q) + H(p) \quad \blacksquare$$

### 10-E. CE = KL (실용적)

진짜 분포 p가 고정 (학습 데이터) → $H(p)$ 상수:
$$\text{CE} 최소화 \equiv \text{KL}(p\Vert q) 최소화$$

→ 분류 학습 = q를 p에 가까이 → KL 거리 최소화.

### 10-F. 시험 답안 표준 한 단락

> "**MSE = NLL** under Gaussian noise (관측 잡음 가우시안 가정 → -log → 제곱만 남음). **CE = NLL** for Categorical (베르누이/카테고리컬 출력). 또한 **CE = H(p) + KL(p||q)**: H(p)는 진짜 분포의 엔트로피 (학습에서 상수), 따라서 CE 최소화 = KL 최소화. 결국 **MSE/CE/NLL/KL은 분포 가정 하에서 모두 동치**."

---

# 🥈 TIER 2: 올해 기출 8문제 (★★★★ 7단계 체인)

올해 기출은 `final-fire/` 폴더에서 5-파일 세트로 자세히 다룸. 여기서는 핵심만 압축.

| # | 문제 | 핵심 답 |
|---|------|--------|
| 1 | A=[[0,1],[1,0]] 고유값/벡터 + 정의 증명 | $\lambda = \pm 1$, $\mathbf{v} = (1,\pm 1)/\sqrt{2}$ |
| 2 | $X \sim N(0,1)$의 $E[X^k]$ for $k=1,2,3,4$ | 0, 1, 0, 3 |
| 3 | $X \sim$ Uniform[a,b]의 $E[X], \text{Var}[X]$ | $(a+b)/2$, $(b-a)^2/12$ |
| 4 | 베르누이 MLE | $\hat{\theta} = k/n$ |
| 5 | MAP $\propto \theta^m(1-\theta)^m$, $m \to \infty$ | $1/2$ |
| 6 | MAP $\propto \theta^m$, $m \to \infty$ | $1$ |
| 7 | Tent prior MAP (m=2, m=6) | m=2: 2/3, m=6: 1/2 |
| 8 | $\mathbf{p} = $ softmax($\mathbf{z}$)의 자코비안 | $J = \text{diag}(p) - pp^T$ |

→ 자세한 7단계 풀이는 `final-fire/01-eigen ~ 08-softmax/` 참조.

---

# 🥉 TIER 3: 강의 추가 토픽 (★★★ 변형 출제 대비)

## T3-1. 다양한 분포의 평균·분산

| 분포 | E[X] | Var[X] |
|------|-----|--------|
| Bern(θ) | θ | θ(1-θ) |
| Binomial(n, θ) | nθ | nθ(1-θ) |
| Poisson(λ) | λ | λ |
| Uniform[a,b] | (a+b)/2 | (b-a)²/12 |
| N(μ, σ²) | μ | σ² |
| Exp(λ) | 1/λ | 1/λ² |

## T3-2. 정규분포 모멘트

$X \sim N(0,1)$: $E[X^{2n}] = (2n-1)!!$, $E[X^{2n+1}] = 0$.

| n | $E[X^n]$ |
|---|---------|
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 3 |
| 5 | 0 |
| 6 | 15 |

## T3-3. 행렬 미분 — Backprop in Matrix Form

**$L = -\log\sigma(Ax+b)$의 $A, x, b$ 미분:**

$z = Ax + b$, $p = \sigma(z) = 1/(1+e^{-z})$. $L = -\log p$.

체인 룰:
$$\frac{\partial L}{\partial z} = -\frac{1}{p}\sigma'(z) = -\frac{p(1-p)}{p} = p - 1$$

(또는 $-\frac{1-p}{1} = -(1-p)$)

따라서:
- $\partial L/\partial A = (p-1) \cdot x^T$
- $\partial L/\partial x = A^T (p-1)$
- $\partial L/\partial b = (p-1)$

## T3-4. Diffusion / VAE NLL 유도 원리

### VAE ELBO
$$\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\Vert p(z))$$

**유도 핵심:** Jensen 부등식 + 잠재변수 marginalization.

### Diffusion Loss
$$L = E_{t, x_0, \epsilon}\left[\Vert \epsilon - \epsilon_\theta(x_t, t)\Vert ^2\right]$$

(noise prediction)

---

# 📊 출제 우선순위 종합 표

| Rank | 문제 | 출처 | 출제확률 |
|------|------|------|---------|
| 🥇 1 | KL Divergence ≥ 0 + 정규분포 KL | 작년 + 강의 | 100% |
| 🥇 2 | Convex 4유형 ($x^2$, $-\log x$, 판별, set) | 작년 | 100% |
| 🥇 3 | Bias-Variance 분해 | 작년 | 100% |
| 🥇 4 | GD 수렴 $\eta < 2/\lambda_{\max}$ | 작년 | 100% |
| 🥇 5 | Convolution + 출력 크기 + matrix | 작년 | 100% |
| 🥇 6 | Markov Chain conditional probability | 작년 | 100% |
| 🥇 7 | 모델 비교 (Markov vs RNN vs Transformer) | 작년 | 100% |
| 🥇 8 | PyTorch 5단계 학습 루프 | 작년 | 100% |
| 🥇 9 | Pooling Matrix | 작년 | 100% |
| 🥇 10 | MSE/NLL/CE/KL 통합 관계 | 작년 + 강의 | 100% |
| 🥈 11 | 베르누이 MLE 7단계 | 올해 | 매우 높음 |
| 🥈 12 | Softmax 자코비안 | 올해 + 작년 | 매우 높음 |
| 🥈 13 | MAP (3가지 prior) | 올해 | 매우 높음 |
| 🥈 14 | 가우스 모멘트 + 가우스 적분 | 올해 + 작년 | 매우 높음 |
| 🥈 15 | 고유값/벡터 + 정의 증명 | 올해 | 매우 높음 |
| 🥉 16 | Jensen 부등식 (단독) | 작년 강조 | 높음 |
| 🥉 17 | Quadratic 관련 | 작년 기타 | 중간 |
| 🥉 18 | VAE ELBO 유도 | 강의 | 중간 |

---

**작성:** 2026-04-26 (v3)
**기반:** 작년 시험 5장 이미지 + 올해 8문제 + 강의 스크립트
**관련 자료:**
- `STRATEGY_v3.md` — 학습 전략 + 8주 일정
- `MASTER-CONCEPTS_v3.md` — 개념 보강판
- `final-fire/` — 시험 답안 작성 훈련

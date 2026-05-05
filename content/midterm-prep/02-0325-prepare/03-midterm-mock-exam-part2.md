---
title: "Deep Learning Theory - Midterm Mock Exam (Part 2: Questions 11-20)"
slug: midterm-mock-exam-part2
order: 3
---

# Deep Learning Theory - Midterm Mock Exam (Part 2: Questions 11-20)

> **Course**: Deep Learning Theory
> **Exam Style**: Proof/derivation focused, logical process over final answers
> **Date**: 2026-03-25

---

## 문제 11 (Normal Equation Derivation)

### [EN] Problem Statement

For linear regression $y = Xw$ with loss function $L(w) = \frac{1}{2}\|Xw - y\|^2$, derive the normal equation $w^* = (X^TX)^{-1}X^Ty$.

Show each step of the matrix calculus clearly. Explain why we set the gradient to zero to find the optimal weights.

### [KR] 문제

선형 회귀 $y = Xw$에서 손실 함수 $L(w) = \frac{1}{2}\|Xw - y\|^2$가 주어졌을 때, 정규 방정식(Normal Equation) $w^* = (X^TX)^{-1}X^Ty$를 유도하시오.

행렬 미분(matrix calculus)의 각 단계를 명확히 보이고, 왜 기울기(gradient)를 0으로 놓아야 최적의 가중치를 찾을 수 있는지 설명하시오.

### 출제 의도

- **핵심 개념**: 선형 회귀의 닫힌 형태(closed-form) 해를 직접 유도하는 능력
- **왜 중요한가**: 딥러닝의 모든 최적화는 "손실 함수의 기울기 = 0"에서 출발한다. Normal equation은 이 아이디어의 가장 순수한 형태이다.
- **빅 픽처**: Gradient descent가 반복적으로 하는 일을, 선형 회귀에서는 한 번에 해석적으로 풀 수 있다. 이 유도 과정에서 배우는 행렬 미분은 이후 모든 딥러닝 수학의 기초가 된다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**먼저 기호부터 이해하자:**

- $X$: 데이터 행렬. 행이 데이터 포인트, 열이 특성(feature). 크기는 $n \times d$ (데이터 $n$개, 특성 $d$개)
- $w$: 가중치 벡터. 크기 $d \times 1$. 우리가 찾고 싶은 것!
- $y$: 정답 벡터. 크기 $n \times 1$
- $Xw$: 예측값. 행렬과 벡터를 곱하면 $n \times 1$ 벡터가 나옴
- $\|v\|^2$: 벡터 $v$의 각 원소를 제곱해서 더한 것. 예를 들어 $\|[3, 4]\|^2 = 3^2 + 4^2 = 25$

**비유로 이해하기:**

학생 3명의 키($x$)로 몸무게($y$)를 예측하는 공식 $y = wx$를 만들고 싶다고 하자. $w$는 "키 1cm당 몸무게 몇 kg?"이라는 비율이다.

예측이 정답과 차이가 나면 "오차"가 생긴다. 손실 함수 $L(w)$는 이 오차들을 모두 제곱해서 더한 것이다. 제곱하는 이유는 양수/음수 오차가 서로 상쇄되지 않게 하기 위해서이다.

**왜 기울기를 0으로 놓는가?**

산 위에서 공을 굴리면 골짜기에서 멈춘다. 골짜기 바닥에서는 어느 방향으로도 "더 낮은 곳"이 없다. 수학적으로 이것은 "기울기 = 0"이라는 뜻이다.

$L(w)$는 $w$에 대한 2차 함수(포물선)이다. 포물선의 꼭짓점에서 기울기가 0이 되고, 그 점이 바로 최솟값이다.

**유도 과정 (아주 천천히):**

**1단계: 손실 함수를 풀어 쓰기**

$\|Xw - y\|^2$은 벡터 $(Xw - y)$의 각 성분을 제곱해서 더한 것인데, 이것은 전치(transpose)를 이용하면:

$$\|Xw - y\|^2 = (Xw - y)^T(Xw - y)$$

왜냐하면 벡터 $v$에 대해 $v^Tv = v_1^2 + v_2^2 + \cdots + v_n^2 = \|v\|^2$이기 때문이다.

**2단계: 괄호 전개하기**

$(Xw - y)^T(Xw - y)$를 전개하자. 전치의 성질 $(A - B)^T = A^T - B^T$를 사용한다:

$$= (Xw)^T(Xw) - (Xw)^Ty - y^T(Xw) + y^Ty$$

전치의 성질 $(AB)^T = B^TA^T$를 사용하면 $(Xw)^T = w^TX^T$이므로:

$$= w^TX^TXw - w^TX^Ty - y^TXw + y^Ty$$

여기서 $w^TX^Ty$와 $y^TXw$는 둘 다 $1 \times 1$ 스칼라(숫자 하나)이고, 스칼라의 전치는 자기 자신이므로 $w^TX^Ty = (y^TXw)^T = y^TXw$이다. 따라서:

$$L(w) = \frac{1}{2}(w^TX^TXw - 2w^TX^Ty + y^Ty)$$

**3단계: $w$에 대해 미분하기**

행렬 미분 공식을 사용한다:
- $\frac{\partial}{\partial w}(w^TAw) = 2Aw$ (단, $A$가 대칭행렬일 때. $X^TX$는 항상 대칭!)
- $\frac{\partial}{\partial w}(b^Tw) = b$
- $\frac{\partial}{\partial w}(상수) = 0$

따라서:

$$\frac{\partial L}{\partial w} = \frac{1}{2}(2X^TXw - 2X^Ty + 0) = X^TXw - X^Ty$$

**4단계: 기울기 = 0으로 놓기**

$$X^TXw - X^Ty = 0$$
$$X^TXw = X^Ty$$

양변에 $(X^TX)^{-1}$을 곱하면:

$$w^* = (X^TX)^{-1}X^Ty$$

이것이 바로 **정규 방정식(Normal Equation)**이다!

#### 🟡 Level 2: 고등학생 눈높이

**손실 함수 전개:**

$$L(w) = \frac{1}{2}(Xw - y)^T(Xw - y) = \frac{1}{2}(w^TX^TXw - 2y^TXw + y^Ty)$$

**기울기 계산:**

행렬 미분 항등식을 적용한다:
- $\nabla_w(w^TAw) = (A + A^T)w$. $A = X^TX$는 대칭이므로 $= 2X^TXw$
- $\nabla_w(b^Tw) = b$이므로, $\nabla_w(y^TXw) = X^Ty$

$$\nabla_w L = \frac{1}{2}(2X^TXw - 2X^Ty) = X^TXw - X^Ty$$

**최적 조건:**

$\nabla_w L = 0$으로 놓으면:

$$X^TXw = X^Ty \implies w^* = (X^TX)^{-1}X^Ty$$

$X^TX$가 가역(invertible)이어야 한다. 이는 $X$의 열(column)들이 선형 독립일 때, 즉 $\text{rank}(X) = d$일 때 성립한다.

**2차 조건 확인:** $L(w)$의 헤시안(Hessian)은 $H = X^TX$인데, 이는 양의 준정부호(positive semi-definite) 행렬이다. 따라서 $L(w)$는 볼록(convex) 함수이고, 기울기 = 0인 점은 전역 최솟값(global minimum)이다.

#### 🔴 Level 3: 대학생 눈높이

$$L(w) = \frac{1}{2}\|Xw - y\|_2^2 = \frac{1}{2}w^TX^TXw - w^TX^Ty + \frac{1}{2}y^Ty$$

$$\nabla_w L = X^TXw - X^Ty = X^T(Xw - y) = 0$$

$$\Rightarrow w^* = (X^TX)^{-1}X^Ty$$

**핵심 포인트:**
- $X^TX \succ 0$ (positive definite)이면 유일해 존재. $X^TX \succeq 0$이면 pseudo-inverse $X^+ = (X^TX)^{-1}X^T$ 사용.
- $X^T(Xw^* - y) = 0$은 잔차 $(Xw^* - y)$가 $X$의 열공간(column space)에 직교함을 의미 — 기하학적으로 정사영(orthogonal projection).
- $\frac{1}{2}$ 계수는 미분 시 깔끔함을 위한 관례. 최적해에 영향 없음.
- 계산 복잡도: $O(nd^2 + d^3)$. $d$가 크면 gradient descent가 더 효율적.

---

## 문제 12 (Logistic Regression and Sigmoid)

### [EN] Problem Statement

Show that the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$ has the property that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

Derive step by step. Explain why this property makes sigmoid computationally convenient for backpropagation.

### [KR] 문제

시그모이드 함수 $\sigma(z) = \frac{1}{1 + e^{-z}}$가 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$라는 성질을 가짐을 보이시오.

단계별로 유도하고, 이 성질이 왜 역전파(backpropagation)에서 계산적으로 편리한지 설명하시오.

### 출제 의도

- **핵심 개념**: 시그모이드 도함수의 우아한 형태와 그 실용적 의미
- **왜 중요한가**: 시그모이드는 역사적으로 가장 중요한 활성화 함수이며, 로지스틱 회귀의 핵심이다. 도함수가 자기 자신으로 표현된다는 사실은 역전파의 효율성과 직결된다.
- **빅 픽처**: 순전파(forward pass)에서 이미 $\sigma(z)$를 계산했다면, 역전파에서 도함수를 구하기 위해 추가 계산이 거의 필요 없다. 이런 "계산 재사용" 패턴은 딥러닝의 효율성의 핵심이다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- $e$: 자연상수. 약 2.718... 무한히 계속되는 특별한 수.
- $e^{-z}$: $e$의 $-z$ 제곱. $z$가 크면 이 값은 0에 가까워지고, $z$가 작으면(음수면) 이 값은 매우 커진다.
- $\sigma(z)$: 어떤 숫자 $z$를 넣으면 0과 1 사이의 값을 출력하는 함수. "확률"처럼 해석할 수 있어서 편리하다.
- $\sigma'(z)$: $\sigma(z)$의 도함수. "$z$가 아주 조금 변할 때 $\sigma$가 얼마나 변하는가"의 비율.

**비유:** 시그모이드는 "스위치 조절기"와 같다. 입력이 매우 작으면 거의 0(꺼짐), 매우 크면 거의 1(켜짐), 중간이면 0.5(반반). 부드러운 on/off 스위치이다.

**유도 과정:**

**1단계: 분수 형태로 쓰기**

$$\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}$$

**2단계: 연쇄 법칙(chain rule) 적용**

$f(u) = u^{-1}$이고 $u = 1 + e^{-z}$라고 하자.

- $f'(u) = -u^{-2} = -\frac{1}{u^2}$ (거듭제곱 미분 규칙: $(x^n)' = nx^{n-1}$)
- $\frac{du}{dz} = \frac{d}{dz}(1 + e^{-z}) = 0 + e^{-z} \cdot (-1) = -e^{-z}$

연쇄 법칙: $\sigma'(z) = f'(u) \cdot \frac{du}{dz}$

$$\sigma'(z) = -\frac{1}{(1 + e^{-z})^2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

**3단계: $\sigma(z)(1 - \sigma(z))$와 같음을 보이기**

$\sigma(z) = \frac{1}{1 + e^{-z}}$이므로:

$$1 - \sigma(z) = 1 - \frac{1}{1 + e^{-z}} = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = \frac{e^{-z}}{1 + e^{-z}}$$

따라서:

$$\sigma(z) \cdot (1 - \sigma(z)) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

이것은 2단계에서 구한 $\sigma'(z)$와 정확히 같다!

$$\boxed{\sigma'(z) = \sigma(z)(1 - \sigma(z))}$$

**왜 역전파에서 편리한가?**

순전파(forward pass)에서 $\sigma(z)$를 이미 계산했다고 하자. 예를 들어 $\sigma(z) = 0.7$이라면:
- 도함수 = $0.7 \times (1 - 0.7) = 0.7 \times 0.3 = 0.21$

추가로 $e^{-z}$를 다시 계산할 필요가 전혀 없다! 곱셈 한 번이면 끝이다.

#### 🟡 Level 2: 고등학생 눈높이

$\sigma(z) = (1 + e^{-z})^{-1}$에 체인 룰 적용:

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

이를 변형한다:

$$\sigma'(z) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}}$$

$\frac{e^{-z}}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)$이므로:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z)) \quad \blacksquare$$

**역전파에서의 이점:**
- 순전파 결과 $a = \sigma(z)$를 메모리에 저장해두면, 역전파에서 $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot a(1-a)$로 즉시 계산 가능.
- 추가 지수 함수 연산 불필요 → 계산 비용 $O(1)$.

#### 🔴 Level 3: 대학생 눈높이

**유도:**

$$\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)(1-\sigma(z))$$

**핵심 포인트:**
- 로지스틱 회귀에서 NLL 기울기: $\frac{\partial}{\partial w}[-y\log\sigma(w^Tx) - (1-y)\log(1-\sigma(w^Tx))] = (\sigma(w^Tx) - y)x$. 시그모이드 도함수의 성질 덕분에 깔끔하게 정리됨.
- $\sigma'(z)$의 최댓값은 $z=0$에서 $1/4$. 이로 인해 깊은 신경망에서 기울기 소실(vanishing gradient) 문제 발생 — ReLU가 대안으로 등장한 이유.
- $\sigma(z) = \frac{1}{2}(1 + \tanh(z/2))$이므로 tanh과 affine 관계.

---

## 문제 13 (Softmax Derivation via Lagrange Multipliers)

### [EN] Problem Statement

Given the problem: maximize entropy $H(p) = -\sum_i p_i \ln p_i$ subject to constraints $\sum_i p_i = 1$ and $\sum_i p_i z_i = \mu$ (fixed mean), use Lagrange multipliers to show that the solution is the softmax (Gibbs) distribution: $p_i \propto \exp(\lambda z_i)$.

Show every step of the Lagrangian setup and differentiation.

### [KR] 문제

다음 최적화 문제를 풀어라: 엔트로피 $H(p) = -\sum_i p_i \ln p_i$를 최대화하되, 제약조건 $\sum_i p_i = 1$과 $\sum_i p_i z_i = \mu$ (고정된 평균)를 만족해야 한다.

라그랑주 승수법(Lagrange multipliers)을 사용하여 해가 소프트맥스(깁스) 분포 $p_i \propto \exp(\lambda z_i)$임을 보이시오. 라그랑지안의 설정과 미분의 모든 단계를 보이시오.

### 출제 의도

- **핵심 개념**: 소프트맥스가 단순한 "편리한 함수"가 아니라, 최대 엔트로피 원리에서 자연스럽게 유도되는 최적 분포임을 이해
- **왜 중요한가**: 딥러닝 분류에서 소프트맥스를 사용하는 이론적 근거. 제약 조건 하 최적화(constrained optimization)의 표준 기법인 라그랑주 승수법을 정확히 쓸 수 있는지 확인.
- **빅 픽처**: "가장 모르는 상태(최대 엔트로피)에서 제약조건만 만족시키면 지수족(exponential family) 분포가 나온다"는 통계역학의 핵심 원리가 딥러닝에 그대로 적용된다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- $p_i$: $i$번째 선택지의 확률. 예를 들어 주사위에서 $p_1$은 1이 나올 확률.
- $\ln$: 자연로그. "$e$를 몇 제곱해야 이 수가 되는가?" 예: $\ln e = 1$, $\ln 1 = 0$.
- $H(p)$: 엔트로피. "불확실성의 양". 동전이 공정하면($p = 0.5$) 엔트로피 최대, 한쪽이 항상 나오면 엔트로피 0.
- $\sum_i p_i = 1$: 모든 확률의 합은 1. (무언가는 반드시 일어남)
- $\sum_i p_i z_i = \mu$: 평균이 $\mu$로 고정. $z_i$는 각 선택지의 "값", $\mu$는 우리가 알고 있는 평균.

**비유:** 주사위를 만드는데, 평균이 3.5가 되도록 하면서 "가장 공정한" (가장 불확실한) 주사위를 만들고 싶다. 엔트로피를 최대화하면 그런 주사위가 된다.

**라그랑주 승수법이란?**

"산의 정상을 찾고 싶은데, 특정 등고선 위에서만 걸어야 한다"면? 등고선이라는 "벽"을 고려하면서 가장 높은 곳을 찾는 방법이 라그랑주 승수법이다.

제약 조건을 "벌금"처럼 목적 함수에 더해서, 제약 없는 문제로 변환한다.

**유도 과정:**

**1단계: 라그랑지안 설정**

목적 함수(엔트로피)에 제약 조건을 라그랑주 승수와 함께 더한다:

$$\mathcal{L}(p, \lambda_0, \lambda_1) = -\sum_i p_i \ln p_i + \lambda_0\left(\sum_i p_i - 1\right) + \lambda_1\left(\sum_i p_i z_i - \mu\right)$$

$\lambda_0$와 $\lambda_1$은 라그랑주 승수 — 각 제약조건의 "중요도"를 나타내는 미지수.

**2단계: $p_i$에 대해 편미분하고 0으로 놓기**

각 $p_i$에 대해:

$$\frac{\partial \mathcal{L}}{\partial p_i} = -\ln p_i - 1 + \lambda_0 + \lambda_1 z_i = 0$$

왜 $-\ln p_i - 1$이 나오는가?
- $\frac{d}{dp_i}(p_i \ln p_i) = \ln p_i + p_i \cdot \frac{1}{p_i} = \ln p_i + 1$ (곱의 미분법)
- 앞에 $-$ 부호가 있으므로: $-\ln p_i - 1$

**3단계: $p_i$에 대해 풀기**

$$\ln p_i = \lambda_0 - 1 + \lambda_1 z_i$$

양변에 지수 함수를 취하면:

$$p_i = e^{\lambda_0 - 1 + \lambda_1 z_i} = e^{\lambda_0 - 1} \cdot e^{\lambda_1 z_i}$$

$e^{\lambda_0 - 1}$은 모든 $i$에 대해 같은 상수이므로:

$$p_i \propto e^{\lambda_1 z_i}$$

이것이 바로 **소프트맥스(깁스) 분포**이다!

**4단계: 정규화 상수 결정**

$\sum_i p_i = 1$이어야 하므로:

$$p_i = \frac{e^{\lambda_1 z_i}}{\sum_j e^{\lambda_1 z_j}}$$

이것이 바로 우리가 아는 소프트맥스 함수의 형태이다! $\lambda_1$은 "온도의 역수"에 해당한다.

#### 🟡 Level 2: 고등학생 눈높이

**라그랑지안:**

$$\mathcal{L} = -\sum_i p_i \ln p_i + \lambda_0\left(\sum_i p_i - 1\right) + \lambda_1\left(\sum_i p_i z_i - \mu\right)$$

**KKT 필요 조건 ($\partial \mathcal{L}/\partial p_i = 0$):**

$$-\ln p_i - 1 + \lambda_0 + \lambda_1 z_i = 0 \quad \forall i$$

$$\Rightarrow p_i = \exp(\lambda_0 - 1 + \lambda_1 z_i) = \frac{\exp(\lambda_1 z_i)}{Z}$$

여기서 $Z = \sum_j \exp(\lambda_1 z_j)$는 분배 함수(partition function)이며, $\lambda_0 = 1 + \ln Z$에서 결정된다.

$\lambda_1$은 두 번째 제약 $\sum_i p_i z_i = \mu$로부터 결정된다.

**엔트로피가 최대인지 확인:** $H(p)$는 확률 심플렉스 위에서 오목(concave) 함수이고, 제약 조건은 선형이므로 정상점은 반드시 전역 최대점이다.

#### 🔴 Level 3: 대학생 눈높이

최대 엔트로피 원리(MaxEnt)에서 모멘트 제약 $\mathbb{E}_p[f_k(x)] = \mu_k$을 부과하면 지수족(exponential family) 분포가 유도된다:

$$p^*(x) = \frac{1}{Z(\boldsymbol{\lambda})} \exp\left(\sum_k \lambda_k f_k(x)\right)$$

본 문제는 $f_1(x_i) = z_i$ 하나의 충분통계량(sufficient statistic)을 가진 특수한 경우이다.

- 쌍대 문제: $\min_\lambda \ln Z(\lambda) - \lambda\mu$는 로그 분배 함수의 Legendre 변환.
- 소프트맥스의 온도 파라미터 $T = 1/\lambda_1$: $T \to 0$이면 argmax (greedy), $T \to \infty$이면 uniform.
- 딥러닝에서 마지막 레이어의 logit $z_i$에 소프트맥스를 적용하는 것은, MaxEnt 원리하에서 logit이 충분통계량 역할을 함을 의미.

---

## 문제 14 (KL Divergence Properties)

### [EN] Problem Statement

(a) Show that $\text{KL}(p \| q) \geq 0$ (Gibbs' inequality) using Jensen's inequality and the fact that $-\ln$ is convex.

(b) Explain why KL divergence is NOT symmetric, i.e., $\text{KL}(p \| q) \neq \text{KL}(q \| p)$ in general, with a concrete numerical example.

(c) Show that minimizing $\text{CE}(p_{\text{data}}, p_{\text{model}})$ is equivalent to minimizing $\text{KL}(p_{\text{data}} \| p_{\text{model}})$.

### [KR] 문제

(a) 옌센 부등식(Jensen's inequality)과 $-\ln$이 볼록 함수라는 사실을 이용하여 $\text{KL}(p \| q) \geq 0$ (깁스 부등식)을 보이시오.

(b) KL 발산이 대칭이 아닌 이유, 즉 일반적으로 $\text{KL}(p \| q) \neq \text{KL}(q \| p)$임을 구체적인 숫자 예시로 설명하시오.

(c) 교차 엔트로피 $\text{CE}(p_{\text{data}}, p_{\text{model}})$를 최소화하는 것이 $\text{KL}(p_{\text{data}} \| p_{\text{model}})$를 최소화하는 것과 동치임을 보이시오.

### 출제 의도

- **핵심 개념**: KL 발산의 세 가지 핵심 성질 — 비음수성, 비대칭성, 교차 엔트로피와의 관계
- **왜 중요한가**: 딥러닝의 손실 함수(cross-entropy loss)가 왜 "최적"인지, 그 이론적 근거가 바로 KL 발산이다.
- **빅 픽처**: 모델을 학습시킨다 = 모델 분포를 데이터 분포에 가깝게 만든다 = KL 발산을 최소화한다 = 교차 엔트로피를 최소화한다. 이 등가 관계가 딥러닝 학습의 핵심 논리이다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- $p$, $q$: 두 확률분포. 예를 들어 $p$는 실제 동전의 앞/뒤 확률, $q$는 우리가 추측한 확률.
- $\text{KL}(p \| q)$: KL 발산. "$p$의 관점에서 $q$가 얼마나 다른가"를 측정하는 수. 항상 0 이상이고, $p = q$일 때만 0.
- $\ln$: 자연로그.

**비유:** KL 발산은 "놀람의 추가량"과 같다. $p$가 진짜 확률인데 $q$라고 잘못 믿고 있으면, 실제보다 더 놀라게 된다. 그 "추가 놀람"이 KL 발산이다.

**(a) $\text{KL}(p \| q) \geq 0$ 증명:**

$$\text{KL}(p \| q) = \sum_i p_i \ln \frac{p_i}{q_i} = -\sum_i p_i \ln \frac{q_i}{p_i} = \mathbb{E}_p\left[-\ln\frac{q_i}{p_i}\right]$$

이제 **옌센 부등식**을 사용한다.

**옌센 부등식이란?** 볼록 함수(아래로 볼록한 U자 모양)에서는 "함수값의 평균 $\geq$ 평균의 함수값"이다.

수식으로: $f$가 볼록이면 $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$.

$-\ln$은 볼록 함수이다 (그래프를 그려보면 위로 볼록이 아니라 아래로... 잠깐, $-\ln x$는 2차 도함수가 $1/x^2 > 0$이므로 볼록(convex)이 맞다).

옌센 부등식 적용:

$$\mathbb{E}_p\left[-\ln\frac{q_i}{p_i}\right] \geq -\ln\left(\mathbb{E}_p\left[\frac{q_i}{p_i}\right]\right) = -\ln\left(\sum_i p_i \cdot \frac{q_i}{p_i}\right) = -\ln\left(\sum_i q_i\right) = -\ln(1) = 0$$

따라서 $\text{KL}(p \| q) \geq 0$. $\quad \blacksquare$

등호 조건: $q_i/p_i$가 모든 $i$에서 같을 때, 즉 $p = q$일 때.

**(b) 비대칭성 예시:**

$p = (0.9, 0.1)$, $q = (0.5, 0.5)$로 놓자.

$$\text{KL}(p \| q) = 0.9 \ln\frac{0.9}{0.5} + 0.1 \ln\frac{0.1}{0.5}$$
$$= 0.9 \ln 1.8 + 0.1 \ln 0.2$$
$$= 0.9 \times 0.5878 + 0.1 \times (-1.6094)$$
$$= 0.5290 - 0.1609 = 0.3681$$

$$\text{KL}(q \| p) = 0.5 \ln\frac{0.5}{0.9} + 0.5 \ln\frac{0.5}{0.1}$$
$$= 0.5 \ln 0.5556 + 0.5 \ln 5$$
$$= 0.5 \times (-0.5878) + 0.5 \times 1.6094$$
$$= -0.2939 + 0.8047 = 0.5108$$

$0.3681 \neq 0.5108$이므로 KL 발산은 대칭이 아니다!

**직관적 이유:** $\text{KL}(p \| q)$에서는 $p_i$가 가중치 역할을 한다. $p$가 집중된 분포(0.9, 0.1)이면 $p$가 높은 곳의 차이를 더 중요하게 본다. 반대로 $\text{KL}(q \| p)$에서는 $q$가 가중치이므로, 가중 방식이 달라져 결과가 다르다.

**(c) CE 최소화 = KL 최소화:**

$$\text{CE}(p_{\text{data}}, p_{\text{model}}) = -\sum_i p_{\text{data},i} \ln p_{\text{model},i}$$

KL 발산을 풀어 쓰면:

$$\text{KL}(p_{\text{data}} \| p_{\text{model}}) = \sum_i p_{\text{data},i} \ln \frac{p_{\text{data},i}}{p_{\text{model},i}}$$
$$= \sum_i p_{\text{data},i} \ln p_{\text{data},i} - \sum_i p_{\text{data},i} \ln p_{\text{model},i}$$
$$= -H(p_{\text{data}}) + \text{CE}(p_{\text{data}}, p_{\text{model}})$$

$$\therefore \text{KL}(p_{\text{data}} \| p_{\text{model}}) = \text{CE}(p_{\text{data}}, p_{\text{model}}) - H(p_{\text{data}})$$

$H(p_{\text{data}})$는 데이터의 엔트로피로, 모델 파라미터와 **무관한 상수**이다!

따라서 모델 파라미터에 대해 $\text{CE}$를 최소화하는 것은 곧 $\text{KL}$을 최소화하는 것과 동치이다.

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $f(x) = -\ln x$는 $f''(x) = 1/x^2 > 0$이므로 볼록. 옌센 부등식:

$$\text{KL}(p\|q) = \mathbb{E}_p\left[-\ln\frac{q}{p}\right] \geq -\ln\mathbb{E}_p\left[\frac{q}{p}\right] = -\ln 1 = 0 \quad \blacksquare$$

**(b)** 위 수치 예시 참조. 핵심: KL은 기대값 연산에 사용되는 분포가 다르므로 비대칭.
- $\text{KL}(p\|q)$: forward KL. 변분 추론에서 사용 — "mean-seeking" 성질.
- $\text{KL}(q\|p)$: reverse KL. "mode-seeking" 성질.

**(c)** $\text{KL}(p_d \| p_\theta) = \text{CE}(p_d, p_\theta) - H(p_d)$에서 $H(p_d)$는 $\theta$에 무관한 상수이므로:

$$\arg\min_\theta \text{CE}(p_d, p_\theta) = \arg\min_\theta \text{KL}(p_d \| p_\theta) \quad \blacksquare$$

#### 🔴 Level 3: 대학생 눈높이

**(a)** 정보 부등식(information inequality)의 직접 증명. $-\ln$의 strict convexity에 의해 등호 iff $p = q$ a.e.

**(b)** Forward KL($p\|q$)은 $p > 0$인 곳에서 $q$도 양수일 것을 강제(zero-avoiding). Reverse KL($q\|p$)은 $q$가 $p$의 mode 하나에 집중(zero-forcing). 이 비대칭성은 VAE(ELBO 최적화 = reverse KL) vs. MLE(forward KL) 선택의 근거.

**(c)** MLE 관점: $\hat{\theta}_{MLE} = \arg\max_\theta \frac{1}{N}\sum \ln p_\theta(x_i) \xrightarrow{N\to\infty} \arg\max_\theta \mathbb{E}_{p_d}[\ln p_\theta] = \arg\min_\theta \text{CE}(p_d, p_\theta) = \arg\min_\theta \text{KL}(p_d \| p_\theta)$. MLE의 점근적 최적성(asymptotic efficiency)이 KL 최소화와 등가.

---

## 문제 15 (Taylor Expansion)

### [EN] Problem Statement

Write the second-order Taylor expansion of $f(x)$ around point $x_0$. Apply it to $f(x) = e^x$ around $x_0 = 0$ and show the first 3 terms. Then explain how Newton's method uses this expansion to find function zeros.

### [KR] 문제

$f(x)$의 점 $x_0$ 주위에서의 2차 테일러 전개를 쓰시오. 이를 $f(x) = e^x$에 $x_0 = 0$ 주위에서 적용하여 처음 3개 항을 보이시오. 그런 다음 뉴턴 방법(Newton's method)이 이 전개를 어떻게 활용하여 함수의 영점을 찾는지 설명하시오.

### 출제 의도

- **핵심 개념**: 테일러 전개는 복잡한 함수를 다항식으로 근사하는 도구. 딥러닝 최적화의 수학적 기초.
- **왜 중요한가**: 2차 테일러 전개는 Newton's method, Adam optimizer의 adaptive learning rate, Fisher information matrix 등의 이론적 기반.
- **빅 픽처**: Gradient descent는 1차 근사(직선), Newton's method는 2차 근사(포물선). 더 정확한 근사 → 더 빠른 수렴. 하지만 헤시안 계산 비용이 크다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**테일러 전개란?**

복잡한 함수를 "간단한 다항식으로 흉내 내는 것"이다.

비유: 곡선 도로를 운전할 때, 짧은 구간만 보면 직선(1차)처럼 보인다. 조금 더 넓게 보면 완만한 커브(2차)처럼 보인다. 테일러 전개는 이런 "근사"를 수학적으로 하는 것이다.

**2차 테일러 전개 공식:**

$$f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2$$

- $f(x_0)$: 기준점에서의 함수값 (0차 — 상수)
- $f'(x_0)(x - x_0)$: 기울기를 이용한 보정 (1차 — 직선)
- $\frac{f''(x_0)}{2}(x - x_0)^2$: 곡률을 이용한 보정 (2차 — 포물선)
- $2! = 2 \times 1 = 2$: 팩토리얼

**$f(x) = e^x$, $x_0 = 0$에 적용:**

먼저 도함수를 구하자:
- $f(x) = e^x$이므로 $f(0) = e^0 = 1$
- $f'(x) = e^x$이므로 $f'(0) = 1$
- $f''(x) = e^x$이므로 $f''(0) = 1$

($e^x$의 특별한 성질: 미분해도 자기 자신!)

대입하면:

$$e^x \approx 1 + 1 \cdot x + \frac{1}{2}x^2 = 1 + x + \frac{x^2}{2}$$

검증: $e^{0.1} \approx 1 + 0.1 + 0.005 = 1.105$. 실제값은 $1.10517...$으로 매우 정확!

**뉴턴 방법과의 연결:**

$g(x) = 0$의 해를 찾고 싶다고 하자 (함수의 영점 찾기).

1차 테일러 전개: $g(x) \approx g(x_n) + g'(x_n)(x - x_n)$

이 근사를 0으로 놓으면:

$$0 = g(x_n) + g'(x_n)(x_{n+1} - x_n)$$

$$x_{n+1} = x_n - \frac{g(x_n)}{g'(x_n)}$$

이것이 **뉴턴 방법**의 업데이트 규칙이다!

최적화에서는 $g(x) = f'(x)$로 놓는다 (기울기가 0인 점을 찾으니까). 그러면:

$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

이는 2차 테일러 근사 $f(x) \approx f(x_n) + f'(x_n)(x - x_n) + \frac{1}{2}f''(x_n)(x-x_n)^2$의 최솟값을 직접 구하는 것과 같다!

#### 🟡 Level 2: 고등학생 눈높이

**일반 공식:**

$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k + R_n(x)$$

2차까지: $f(x) \approx f(x_0) + f'(x_0)\Delta x + \frac{1}{2}f''(x_0)(\Delta x)^2$ 여기서 $\Delta x = x - x_0$.

**$e^x$ 적용:** $f^{(k)}(0) = 1$ for all $k$이므로:

$$e^x \approx 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots = \sum_{k=0}^\infty \frac{x^k}{k!}$$

**Newton's method for optimization:**

$f(x)$의 2차 근사:

$$q(x) = f(x_n) + f'(x_n)(x - x_n) + \frac{1}{2}f''(x_n)(x - x_n)^2$$

$q'(x) = 0$으로 놓으면:

$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

다변수 일반화: $x_{n+1} = x_n - [H(x_n)]^{-1}\nabla f(x_n)$. 여기서 $H$는 헤시안 행렬.

수렴 속도: 2차 수렴(quadratic convergence) — gradient descent의 선형 수렴보다 훨씬 빠름.

#### 🔴 Level 3: 대학생 눈높이

**다변수 2차 Taylor:**

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T(\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T H(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)$$

- Newton step: $\Delta \mathbf{x} = -H^{-1}\nabla f$. 헤시안 $O(d^2)$ 저장, $O(d^3)$ 역행렬 → 대규모 DL에서는 비현실적.
- 근사법: Quasi-Newton (L-BFGS), Natural gradient ($H \approx$ Fisher information matrix), Adam (diagonal 근사).
- 수렴 분석: 충분히 가까운 초기점에서 $\|x_{n+1} - x^*\| \leq C\|x_n - x^*\|^2$ (2차 수렴). GD는 $\|x_{n+1} - x^*\| \leq (1 - \frac{m}{M})\|x_n - x^*\|$ (선형 수렴, condition number $M/m$ 의존).

---

## 문제 16 (Matrix Multiplication as Linear Transformation)

### [EN] Problem Statement

Given transformation matrix $A = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ with $\theta = 90°$, apply it to vector $v = [1, 0]^T$. What geometric operation does this matrix perform?

Explain why neural network layers (without activation) are linear transformations, and why activation functions are needed.

### [KR] 문제

변환 행렬 $A = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$에서 $\theta = 90°$일 때, 벡터 $v = [1, 0]^T$에 이 행렬을 적용하시오. 이 행렬은 어떤 기하학적 연산을 수행하는가?

신경망 층(활성화 함수 없이)이 왜 선형 변환인지, 그리고 왜 활성화 함수가 필요한지 설명하시오.

### 출제 의도

- **핵심 개념**: 행렬 곱셈의 기하학적 의미, 신경망에서 선형 변환의 한계
- **왜 중요한가**: "신경망이 왜 깊어야 하는가?"의 핵심 답변 — 선형 변환의 합성은 여전히 선형이므로, 비선형 활성화 함수가 없으면 깊은 신경망이 무의미.
- **빅 픽처**: Universal Approximation Theorem의 전제 조건인 비선형 활성화의 필요성을 기하학적으로 직관화.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**행렬 곱셈이란?**

$2 \times 2$ 행렬과 $2 \times 1$ 벡터를 곱하는 방법:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} ax + by \\ cx + dy \end{bmatrix}$$

각 행과 벡터를 짝지어 곱하고 더한다.

**$\theta = 90°$ 대입:**

$\cos 90° = 0$, $\sin 90° = 1$이므로:

$$A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

$v = [1, 0]^T$에 적용:

$$Av = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \cdot 1 + (-1) \cdot 0 \\ 1 \cdot 1 + 0 \cdot 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

**기하학적 의미:** $[1, 0]$ (오른쪽을 가리키는 화살표)이 $[0, 1]$ (위쪽을 가리키는 화살표)로 바뀌었다. 이것은 **반시계 방향으로 90° 회전**이다!

일반적으로 이 행렬은 원점을 중심으로 반시계 방향 $\theta$ 회전을 수행한다.

**왜 활성화 함수가 필요한가?**

신경망의 한 층: $h = Wx + b$ (행렬 곱 + 편향).

두 층을 쌓으면: $h_2 = W_2(W_1x + b_1) + b_2 = W_2W_1x + W_2b_1 + b_2 = W'x + b'$

여기서 $W' = W_2W_1$, $b' = W_2b_1 + b_2$. 이건 여전히 **선형 변환** 하나와 같다!

100층을 쌓아도 결국 $Wx + b$ 하나로 줄어든다. 직선으로는 곡선을 그릴 수 없듯이, 선형 변환으로는 복잡한 패턴(예: 고양이와 개를 구분하는 경계)을 만들 수 없다.

활성화 함수(예: ReLU, sigmoid)가 각 층 사이에 **비선형성**을 추가해야 한다. 그래야 여러 층이 의미 있게 합쳐져서, 복잡한 곡선 경계를 만들 수 있다.

#### 🟡 Level 2: 고등학생 눈높이

**회전 행렬 유도:**

점 $(r\cos\phi, r\sin\phi)$를 $\theta$만큼 회전하면 $(r\cos(\phi+\theta), r\sin(\phi+\theta))$이다. 삼각함수 덧셈정리 적용:

$$x' = x\cos\theta - y\sin\theta, \quad y' = x\sin\theta + y\cos\theta$$

이를 행렬로 쓰면 $A = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$.

$\theta = 90°$: $Av = [0, 1]^T$. 반시계 방향 90° 회전 확인.

**선형 변환의 한계:**

선형 변환은 다음을 만족: $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$ (superposition).

층 합성: $f_n \circ \cdots \circ f_1(x) = W_n \cdots W_1 x + \text{(bias)}$는 여전히 affine.

비선형 활성화 $\sigma$를 넣으면: $f(x) = W_2\sigma(W_1x + b_1) + b_2$는 affine이 아니다. 이렇게 해야 XOR 문제 같은 비선형 분리가 가능.

#### 🔴 Level 3: 대학생 눈높이

- 회전 행렬 $R_\theta \in SO(2)$: $\det R = 1$, $R^T R = I$ (직교행렬). 거리/각도 보존.
- 신경망에서 $W$는 일반적인 affine map — 회전+스케일+전단(shear)+이동. 활성화 없이는 $\text{rank}(W_n \cdots W_1) \leq \min_k \text{rank}(W_k)$로 표현력 제한.
- Universal Approximation Theorem: 1개 은닉층 + 비선형 활성화(non-polynomial)로 임의의 연속함수를 근사 가능. 단, 깊이가 아닌 너비(width)가 지수적으로 필요할 수 있어, 실제로는 depth가 효율적 (depth separation results, Telgarsky 2016).
- 기하학적 시각: ReLU 네트워크는 입력 공간을 선형 영역(linear regions)으로 분할. 층이 깊을수록 영역 수가 지수적 증가.

---

## 문제 17 (Expected Value and Variance)

### [EN] Problem Statement

For a discrete random variable $X$ with $P(X=1) = 0.2$, $P(X=2) = 0.5$, $P(X=3) = 0.3$:

(a) Compute $E[X]$, $E[X^2]$, and $\text{Var}(X) = E[X^2] - (E[X])^2$.

(b) Prove that $\text{Var}(aX + b) = a^2\text{Var}(X)$ for constants $a, b$.

(c) Explain why independence is needed for $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

### [KR] 문제

이산 확률 변수 $X$에 대해 $P(X=1) = 0.2$, $P(X=2) = 0.5$, $P(X=3) = 0.3$일 때:

(a) $E[X]$, $E[X^2]$, $\text{Var}(X) = E[X^2] - (E[X])^2$를 계산하시오.

(b) 상수 $a, b$에 대해 $\text{Var}(aX + b) = a^2\text{Var}(X)$임을 증명하시오.

(c) $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$가 성립하려면 왜 독립성이 필요한지 설명하시오.

### 출제 의도

- **핵심 개념**: 기댓값과 분산의 정의, 계산, 성질
- **왜 중요한가**: 기댓값은 손실 함수의 이론적 정의($L = \mathbb{E}[\ell]$), 분산은 학습의 안정성(SGD의 분산), batch normalization 등에 핵심.
- **빅 픽처**: Mini-batch SGD에서 기울기 추정의 분산이 배치 크기에 반비례하는 이유, 그리고 독립 가정이 왜 중요한지를 이해하는 기초.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- $E[X]$: 기댓값. "평균적으로 어떤 값이 나오는가?" 주사위의 기댓값은 $3.5$.
- $E[X^2]$: "$X$를 제곱한 것의 평균"
- $\text{Var}(X)$: 분산. "값들이 평균에서 얼마나 퍼져 있는가?" 분산이 크면 불확실성이 크다.

**(a) 계산:**

$$E[X] = \sum_x x \cdot P(X = x) = 1 \times 0.2 + 2 \times 0.5 + 3 \times 0.3$$
$$= 0.2 + 1.0 + 0.9 = 2.1$$

$$E[X^2] = \sum_x x^2 \cdot P(X = x) = 1^2 \times 0.2 + 2^2 \times 0.5 + 3^2 \times 0.3$$
$$= 1 \times 0.2 + 4 \times 0.5 + 9 \times 0.3 = 0.2 + 2.0 + 2.7 = 4.9$$

$$\text{Var}(X) = E[X^2] - (E[X])^2 = 4.9 - (2.1)^2 = 4.9 - 4.41 = 0.49$$

**(b) $\text{Var}(aX + b) = a^2\text{Var}(X)$ 증명:**

$Y = aX + b$라 하자.

**1단계:** $Y$의 기댓값을 구한다.

$$E[Y] = E[aX + b] = aE[X] + b$$

기댓값의 성질: 상수를 밖으로 꺼낼 수 있고, 상수의 기댓값은 자기 자신.

**2단계:** 분산 정의를 적용한다.

$$\text{Var}(Y) = E[(Y - E[Y])^2]$$
$$= E[(aX + b - aE[X] - b)^2]$$
$$= E[(a(X - E[X]))^2]$$
$$= E[a^2(X - E[X])^2]$$
$$= a^2 E[(X - E[X])^2]$$
$$= a^2 \text{Var}(X) \quad \blacksquare$$

**핵심 직관:** $b$(이동)는 분산에 영향 없다 — 모든 값을 같은 양만큼 옮기면 퍼짐 정도는 안 변한다. $a$(스케일)는 제곱으로 영향 — 모든 값을 2배로 늘리면, 편차도 2배, 편차의 제곱은 4배.

**(c) 독립성이 왜 필요한가:**

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

여기서 $\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$는 공분산이다.

$X$와 $Y$가 **독립**이면 $E[XY] = E[X]E[Y]$이므로 $\text{Cov}(X, Y) = 0$이고:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

독립이 아닌 예: $X$와 $Y$가 같은 방향으로 움직이면(양의 상관), 합의 분산은 각 분산의 합보다 **커진다**. 반대면 **작아진다**.

일상 예: 우산 매출($X$)과 장화 매출($Y$)은 양의 상관(비가 오면 둘 다 증가). 따라서 $\text{Var}(X+Y) > \text{Var}(X) + \text{Var}(Y)$.

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $E[X] = 2.1$, $E[X^2] = 4.9$, $\text{Var}(X) = 0.49$.

**(b)** $\text{Var}(aX+b) = E[(aX+b - E[aX+b])^2] = E[a^2(X-\mu_X)^2] = a^2\sigma_X^2$. $\blacksquare$

**(c)** 분산의 이중합:

$$\text{Var}\left(\sum_i X_i\right) = \sum_i \text{Var}(X_i) + 2\sum_{i<j}\text{Cov}(X_i, X_j)$$

독립이면 모든 교차항 $\text{Cov}(X_i, X_j) = 0$. SGD에서 mini-batch 기울기의 분산이 $\sigma^2/B$ ($B$: 배치크기)인 이유가 바로 이것: i.i.d. 가정 하에 $\text{Var}(\bar{g}) = \text{Var}(g)/B$.

#### 🔴 Level 3: 대학생 눈높이

- $\text{Var}(X) = 0.49$, $\sigma = 0.7$.
- $\text{Var}(aX+b) = a^2\text{Var}(X)$: batch normalization에서 $\hat{x} = (x - \mu)/\sigma$로 정규화하면 $\text{Var}(\hat{x}) = 1$.
- 비독립 시: $\text{Var}(\sum X_i) = \mathbf{1}^T\Sigma\mathbf{1}$ 여기서 $\Sigma$는 공분산 행렬. 상관관계가 있으면 분산 감소 효과가 줄어듦 → dropout이 공적응(co-adaptation)을 줄여 실질적 독립성을 높이는 메커니즘.
- SGD에서 i.i.d. 가정이 깨지면 (예: 순서대로 배치) 기울기 추정 분산이 증가 → shuffling의 중요성.

---

## 문제 18 (Gradient Descent Step-by-Step)

### [EN] Problem Statement

For loss function $L(w) = (w-3)^2 + 2(w-3)$, starting at $w_0 = 0$ with learning rate $\eta = 0.1$:

(a) Compute the gradient $\frac{dL}{dw}$.

(b) Perform 3 steps of gradient descent, showing each update.

(c) What is the optimal $w^*$? Explain why gradient descent converges to it.

(d) What happens if $\eta = 2$? (instability)

### [KR] 문제

손실 함수 $L(w) = (w-3)^2 + 2(w-3)$에서, $w_0 = 0$, 학습률 $\eta = 0.1$일 때:

(a) 기울기 $\frac{dL}{dw}$를 구하시오.

(b) 경사 하강법을 3단계 수행하고, 각 업데이트를 보이시오.

(c) 최적의 $w^*$는 무엇인가? 경사 하강법이 왜 이 값에 수렴하는지 설명하시오.

(d) $\eta = 2$이면 어떤 일이 발생하는가? (불안정성)

### 출제 의도

- **핵심 개념**: 경사 하강법의 구체적 작동 방식과 학습률의 역할
- **왜 중요한가**: 딥러닝의 모든 학습은 경사 하강법(의 변형)이다. 학습률이 너무 크거나 작으면 어떤 문제가 생기는지 직접 계산으로 체감.
- **빅 픽처**: 학습률 스케줄링, Adam 등 adaptive method의 필요성을 이해하는 출발점.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- $L(w)$: 손실 함수. $w$를 넣으면 "현재 얼마나 틀렸는가"를 숫자로 알려준다. 이 숫자를 가능한 한 작게 만들고 싶다.
- $\frac{dL}{dw}$: 기울기. "$w$를 아주 조금 늘리면 $L$이 얼마나 변하는가?"
- $\eta$: 학습률. "한 걸음의 크기". 너무 크면 목표를 지나치고, 너무 작으면 영원히 도착 못 한다.

**비유:** 눈을 가리고 산에서 내려가는 상황. 발밑의 기울기(경사)를 느끼고, 그 반대 방향으로 한 걸음씩 내딛는다. 학습률은 걸음 크기.

**(a) 기울기 계산:**

$L(w) = (w-3)^2 + 2(w-3)$

먼저 전개하자: $L(w) = w^2 - 6w + 9 + 2w - 6 = w^2 - 4w + 3$

미분: $\frac{dL}{dw} = 2w - 4$

(또는 직접: $\frac{d}{dw}(w-3)^2 = 2(w-3)$, $\frac{d}{dw}2(w-3) = 2$이므로 $\frac{dL}{dw} = 2(w-3) + 2 = 2w - 4$)

**(b) 3단계 경사 하강법:**

업데이트 규칙: $w_{n+1} = w_n - \eta \cdot \frac{dL}{dw}\bigg|_{w=w_n}$

**Step 1:** $w_0 = 0$
- 기울기: $2(0) - 4 = -4$
- 업데이트: $w_1 = 0 - 0.1 \times (-4) = 0 + 0.4 = 0.4$

**Step 2:** $w_1 = 0.4$
- 기울기: $2(0.4) - 4 = 0.8 - 4 = -3.2$
- 업데이트: $w_2 = 0.4 - 0.1 \times (-3.2) = 0.4 + 0.32 = 0.72$

**Step 3:** $w_2 = 0.72$
- 기울기: $2(0.72) - 4 = 1.44 - 4 = -2.56$
- 업데이트: $w_3 = 0.72 - 0.1 \times (-2.56) = 0.72 + 0.256 = 0.976$

관찰: $w$가 점점 어떤 값에 다가가고 있다! 기울기의 절댓값도 점점 줄어든다 ($4 \to 3.2 \to 2.56$).

**(c) 최적의 $w^*$:**

$\frac{dL}{dw} = 0$을 풀면: $2w - 4 = 0 \Rightarrow w^* = 2$

$$\boxed{w^* = 2}$$

수렴 이유: 이 함수는 아래로 볼록한 포물선($L = w^2 - 4w + 3$, $w^2$ 계수가 양수)이므로 꼭짓점이 전역 최솟값이다. 기울기가 음수이면 $w$를 증가시키고, 양수이면 감소시키므로, 항상 최솟값 방향으로 이동한다. 학습률이 충분히 작으면($\eta < 1/f''(w^*) = 1/2$) 반드시 수렴한다.

**(d) $\eta = 2$이면?**

**Step 1:** $w_0 = 0$, 기울기 $= -4$
- $w_1 = 0 - 2 \times (-4) = 8$

**Step 2:** $w_1 = 8$, 기울기 $= 2(8) - 4 = 12$
- $w_2 = 8 - 2 \times 12 = 8 - 24 = -16$

**Step 3:** $w_2 = -16$, 기울기 $= 2(-16) - 4 = -36$
- $w_3 = -16 - 2 \times (-36) = -16 + 72 = 56$

$w$가 $0 \to 8 \to -16 \to 56$으로 **발산**하고 있다! 최솟값($w^* = 2$)에서 점점 멀어진다.

학습률이 너무 크면, 골짜기를 "지나쳐서" 반대편 더 높은 곳으로 튀어가고, 다시 더 세게 튀어가고... 이를 **발산(divergence)** 이라 한다.

**안정 조건:** 2차 함수 $L = aw^2 + \cdots$에서 $\eta < 2/|L''| = 2/(2 \cdot 1) = 1$이어야 수렴. $\eta = 2 > 1$이므로 발산!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $L(w) = w^2 - 4w + 3$, $L'(w) = 2w - 4$.

**(b)** $w_{n+1} = w_n - 0.1(2w_n - 4) = 0.8w_n + 0.4$.

이는 등비급수: $w_n = w^* + (w_0 - w^*) \cdot 0.8^n = 2 + (-2)(0.8)^n = 2(1 - 0.8^n)$.

- $w_1 = 0.4$, $w_2 = 0.72$, $w_3 = 0.976$

**(c)** $w^* = 2$. 수렴 조건: $|1 - \eta L''| < 1$, 즉 $|1 - 2\eta| < 1$이므로 $0 < \eta < 1$. $\eta = 0.1$은 조건 만족, 수렴 계수 $= 0.8$.

**(d)** $\eta = 2$: $|1 - 2 \times 2| = 3 > 1$. 진동 발산. 매 스텝 오차가 3배씩 증가.

#### 🔴 Level 3: 대학생 눈높이

일반적 2차 함수 $L(w) = \frac{1}{2}Hw^2 + \cdots$에서 GD 업데이트는 $w_{n+1} - w^* = (1 - \eta H)(w_n - w^*)$.

- 수렴 조건: $|1 - \eta H| < 1 \Leftrightarrow 0 < \eta < 2/H$.
- 최적 학습률: $\eta^* = 1/H$이면 1스텝 수렴.
- 다변수: $\rho(I - \eta H) < 1$, 즉 $\eta < 2/\lambda_{\max}(H)$. Condition number $\kappa = \lambda_{\max}/\lambda_{\min}$이 크면 optimal $\eta$ 범위가 좁아짐 — ill-conditioning.
- SGD에서 노이즈가 추가되면 학습률은 $O(1/t)$로 감소시켜야 수렴 (Robbins-Monro 조건).

---

## 문제 19 (Independence in Probability)

### [EN] Problem Statement

Given joint distribution table:

|     | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.2 | 0.3 |
| X=1 | 0.2 | 0.3 |

(a) Find marginal distributions $P(X)$ and $P(Y)$.

(b) Determine if $X$ and $Y$ are independent. Show your work.

(c) Explain where the independence assumption is used in MLE derivation (why we can write the joint likelihood as a product).

### [KR] 문제

다음 결합 분포표가 주어졌을 때:

|     | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.2 | 0.3 |
| X=1 | 0.2 | 0.3 |

(a) 주변 분포 $P(X)$와 $P(Y)$를 구하시오.

(b) $X$와 $Y$가 독립인지 판별하시오. 풀이 과정을 보이시오.

(c) MLE 유도에서 독립 가정이 어디에 사용되는지 설명하시오 (왜 결합 우도를 곱으로 쓸 수 있는가).

### 출제 의도

- **핵심 개념**: 결합 분포, 주변 분포, 독립성 판정, MLE에서의 독립 가정
- **왜 중요한가**: "데이터 포인트들이 i.i.d."라는 가정은 거의 모든 딥러닝 이론의 출발점. 이 가정이 무엇을 의미하고, 왜 필요하며, 언제 깨지는지 이해해야 한다.
- **빅 픽처**: i.i.d. 가정이 깨지는 경우(시계열, 강화학습) 왜 다른 기법이 필요한지의 출발점.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- 결합 분포: $X$와 $Y$가 **동시에** 어떤 값을 가질 확률. 표의 각 칸이 하나의 결합 확률.
- 주변 분포: $X$ 혼자의 확률분포. $Y$의 값은 무시하고 행을 합친 것.
- 독립: $X$가 어떤 값이든 $Y$의 확률에 영향을 미치지 않는 것.

**(a) 주변 분포:**

$P(X)$: 각 행의 합

- $P(X=0) = 0.2 + 0.3 = 0.5$
- $P(X=1) = 0.2 + 0.3 = 0.5$

$P(Y)$: 각 열의 합

- $P(Y=0) = 0.2 + 0.2 = 0.4$
- $P(Y=1) = 0.3 + 0.3 = 0.6$

**(b) 독립성 판별:**

독립의 정의: **모든** $(i, j)$에 대해 $P(X=i, Y=j) = P(X=i) \cdot P(Y=j)$

하나씩 확인:

| | 결합 확률 | $P(X) \cdot P(Y)$ | 일치? |
|---|---|---|---|
| $P(X=0, Y=0)$ | $0.2$ | $0.5 \times 0.4 = 0.2$ | 일치! |
| $P(X=0, Y=1)$ | $0.3$ | $0.5 \times 0.6 = 0.3$ | 일치! |
| $P(X=1, Y=0)$ | $0.2$ | $0.5 \times 0.4 = 0.2$ | 일치! |
| $P(X=1, Y=1)$ | $0.3$ | $0.5 \times 0.6 = 0.3$ | 일치! |

**모든 칸에서 일치하므로 $X$와 $Y$는 독립이다!**

직관적 확인: $X=0$이든 $X=1$이든, $Y$의 조건부 분포는 항상 $(0.4, 0.6)$으로 같다. $X$를 알아도 $Y$에 대한 정보가 추가되지 않는다.

**(c) MLE에서의 독립 가정:**

데이터 $\{x_1, x_2, \ldots, x_n\}$이 독립이면:

$$P(x_1, x_2, \ldots, x_n | \theta) = P(x_1|\theta) \cdot P(x_2|\theta) \cdots P(x_n|\theta) = \prod_{i=1}^n P(x_i|\theta)$$

이것이 바로 **우도(likelihood) 함수**이다!

왜 곱으로 쓸 수 있는가? 독립의 정의 자체가 "결합 확률 = 각 확률의 곱"이기 때문이다.

곱을 로그로 바꾸면 합이 된다:

$$\ln P = \sum_{i=1}^n \ln P(x_i|\theta)$$

이것이 **로그 우도(log-likelihood)**이고, 합이므로 미분이 훨씬 쉬워진다!

만약 독립이 아니라면? 곱으로 분해할 수 없어서, $n$개 데이터의 결합 확률을 직접 다뤄야 한다. 이건 극도로 어렵다.

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $P(X=0) = P(X=1) = 0.5$, $P(Y=0) = 0.4$, $P(Y=1) = 0.6$.

**(b)** $P(X=i, Y=j) = P(X=i)P(Y=j)$를 모든 $(i,j)$에서 확인 → 독립.

동치 조건: $P(Y|X) = P(Y)$ (조건부 분포가 무조건부와 동일). 표에서 각 행을 행합으로 나누면 $(0.4, 0.6)$으로 동일 → 독립 확인.

**(c)** i.i.d. 가정: $x_1, \ldots, x_n \overset{\text{iid}}{\sim} p_\theta$.

$$\mathcal{L}(\theta) = p(x_1, \ldots, x_n | \theta) \overset{\text{ind}}{=} \prod_i p(x_i|\theta)$$

$$\ell(\theta) = \log \mathcal{L}(\theta) = \sum_i \log p(x_i|\theta)$$

독립 가정 없이는 conditional chain rule만 가능: $p(x_1, \ldots, x_n) = \prod_i p(x_i | x_1, \ldots, x_{i-1})$. 이건 autoregressive model 구조 — transformer의 GPT가 정확히 이 방식.

#### 🔴 Level 3: 대학생 눈높이

- 독립 $\Leftrightarrow$ $P(X,Y) = P(X)P(Y)$ $\Leftrightarrow$ $I(X;Y) = 0$ (mutual information).
- MLE의 i.i.d. 가정은 대수의 법칙(LLN)과 연결: $\frac{1}{n}\sum \log p_\theta(x_i) \xrightarrow{a.s.} \mathbb{E}_{p_d}[\log p_\theta]$. 독립 + 동일 분포가 필요.
- 비독립 데이터: 시계열(temporal correlation) → LSTM/Transformer, 그래프(spatial correlation) → GNN. 이 경우 likelihood factorization이 달라짐.
- Exchangeability (de Finetti): i.i.d.보다 약한 조건에서도 mixture model representation 가능 — Bayesian DL의 기초.

---

## 문제 20 (Cross-Entropy Loss Derivation)

### [EN] Problem Statement

Starting from the categorical distribution assumption $p(y=k|x, \theta) = h(x)_k$ where $h(x)$ is the softmax output:

(a) Write the likelihood for a single data point.

(b) Write the negative log-likelihood.

(c) Show this equals $-e_y^T \log h(x)$ where $e_y$ is the one-hot vector.

(d) Explain why we use log (connection to numerical stability and MLE).

### [KR] 문제

범주형 분포 가정 $p(y=k|x, \theta) = h(x)_k$ ($h(x)$는 소프트맥스 출력)으로부터:

(a) 단일 데이터 포인트에 대한 우도(likelihood)를 쓰시오.

(b) 음의 로그 우도(negative log-likelihood)를 쓰시오.

(c) 이것이 $-e_y^T \log h(x)$와 같음을 보이시오. 여기서 $e_y$는 원-핫 벡터이다.

(d) 왜 로그를 사용하는지 설명하시오 (수치적 안정성과 MLE와의 연결).

### 출제 의도

- **핵심 개념**: 분류 문제의 손실 함수(cross-entropy loss)가 어디서 오는지 — MLE에서 자연스럽게 유도됨
- **왜 중요한가**: "왜 MSE 대신 cross-entropy를 쓰는가?"라는 질문에 이론적으로 답할 수 있어야 한다. 이 유도는 확률론적 관점에서 딥러닝의 학습 목표를 정당화한다.
- **빅 픽처**: 딥러닝 = 확률 모델 + MLE. 손실 함수를 "선택"하는 것이 아니라 확률 가정에서 "유도"하는 것이다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명:**

- $x$: 입력 데이터. 예: 고양이 사진.
- $y$: 정답 레이블. 예: "고양이" = 클래스 2.
- $h(x)$: 신경망의 소프트맥스 출력. 각 클래스에 대한 예측 확률. 예: $h(x) = [0.1, 0.2, 0.7]$ (1번 클래스 10%, 2번 20%, 3번 70%).
- $h(x)_k$: $k$번째 클래스의 예측 확률.
- $e_y$: 원-핫 벡터. 정답 위치만 1, 나머지 0. 예: $y = 2$이면 $e_y = [0, 1, 0]$.

**(a) 우도 (likelihood):**

"이 모델이 정답을 맞출 확률은?"

데이터 $(x, y)$에서 정답이 $y$이고, 모델이 클래스 $y$에 $h(x)_y$의 확률을 배정했다면:

$$\text{likelihood} = p(y|x, \theta) = h(x)_y$$

예시: 정답이 클래스 2이고 $h(x) = [0.1, 0.2, 0.7]$이면, 우도 $= h(x)_2 = 0.2$.

이것을 범주형 분포의 형태로 모든 클래스를 포함해서 쓰면:

$$p(y|x, \theta) = \prod_{k=1}^{K} h(x)_k^{\mathbb{1}[y=k]}$$

여기서 $\mathbb{1}[y=k]$는 "$y=k$이면 1, 아니면 0"이다. 정답인 클래스 $k$만 1이므로, 해당 항만 살아남아 $h(x)_y$가 된다.

**(b) 음의 로그 우도 (NLL):**

$$\text{NLL} = -\log p(y|x, \theta) = -\log h(x)_y$$

예시: $h(x)_2 = 0.2$이면, NLL $= -\log 0.2 = -(-1.6094) = 1.6094$.

만약 $h(x)_2 = 0.9$이면, NLL $= -\log 0.9 = 0.1054$. 모델이 더 확신할수록 NLL이 작아진다!

곱 형태에서:

$$-\log \prod_{k=1}^K h(x)_k^{\mathbb{1}[y=k]} = -\sum_{k=1}^K \mathbb{1}[y=k] \cdot \log h(x)_k$$

**(c) 원-핫 벡터 표현:**

$e_y$는 원-핫 벡터: $y$번째 원소만 1, 나머지 0.

- $e_y = [0, \ldots, 0, \underset{y\text{번째}}{1}, 0, \ldots, 0]$

$$e_y^T \log h(x) = \sum_{k=1}^K (e_y)_k \cdot \log h(x)_k$$

$(e_y)_k = \mathbb{1}[y = k]$이므로:

$$e_y^T \log h(x) = \sum_{k=1}^K \mathbb{1}[y=k] \cdot \log h(x)_k = \log h(x)_y$$

따라서:

$$\text{NLL} = -e_y^T \log h(x) \quad \blacksquare$$

이것이 바로 **교차 엔트로피 손실(cross-entropy loss)**이다!

**(d) 왜 로그를 사용하는가?**

**이유 1: 곱 → 합 변환**

$n$개 데이터의 우도는 곱:

$$\mathcal{L}(\theta) = \prod_{i=1}^n p(y_i|x_i, \theta)$$

로그를 취하면 합:

$$\log \mathcal{L}(\theta) = \sum_{i=1}^n \log p(y_i|x_i, \theta)$$

합은 미분하기 쉽고, 컴퓨터로 계산하기도 쉽다.

**이유 2: 수치적 안정성**

확률은 0과 1 사이의 수이다. 이런 수를 수천 번 곱하면?

$$0.01 \times 0.01 \times \cdots \times 0.01 = 10^{-2000}$$

이런 극히 작은 수는 컴퓨터가 표현할 수 없다 (**언더플로**).

하지만 로그를 취하면 $-2000 \times \log 10 = -4605.2$로, 평범한 숫자가 된다!

**이유 3: 이론적 정당성**

로그는 단조 증가 함수이므로, $\prod p_i$를 최대화하는 것과 $\sum \log p_i$를 최대화하는 것은 동치이다. 따라서 MLE의 결과는 변하지 않는다.

#### 🟡 Level 2: 고등학생 눈높이

**(a)** Categorical distribution: $p(y|x,\theta) = \prod_k h(x)_k^{[y=k]} = h(x)_y$.

**(b)** $\text{NLL} = -\log h(x)_y = -\sum_k [y=k]\log h(x)_k$.

**(c)** One-hot encoding: $[y=k] = (e_y)_k$이므로 $\text{NLL} = -e_y^T \log h(x)$.

전체 데이터에 대해: $\text{Loss} = -\frac{1}{N}\sum_{i=1}^N e_{y_i}^T \log h(x_i)$. 이것이 categorical cross-entropy loss.

**(d)** 로그의 역할:
- 수치적: $\log$-$\text{sum}$-$\exp$ trick으로 소프트맥스+CE를 안정적으로 계산. $\log\text{softmax}(z)_k = z_k - \log\sum_j e^{z_j}$를 직접 계산.
- 이론적: MLE ↔ KL minimization 동치. NLL minimization = cross-entropy minimization = KL divergence minimization (Q14(c) 참조).

#### 🔴 Level 3: 대학생 눈높이

- Exponential family 관점: categorical distribution은 지수족. Natural parameter: $\eta_k = \log h_k$. Sufficient statistic: $T(y) = e_y$. Log-partition function: $A(\eta) = \log\sum_k e^{\eta_k} = 0$ (이미 정규화).
- CE loss의 gradient: $\nabla_{z_k} \text{NLL} = h(x)_k - (e_y)_k = $ softmax output $-$ target. 이 깔끔한 형태는 exponential family의 일반적 성질.
- Label smoothing: $e_y$를 $(1-\epsilon)e_y + \epsilon/K$로 대체 → 정규화 효과. 이는 model distribution과 uniform의 KL regularization에 해당.
- Focal loss ($-(1-h_y)^\gamma \log h_y$): CE의 일반화. 쉬운 예제(높은 $h_y$)의 기여를 줄여 class imbalance 해결.
- Information-theoretic view: CE = $H(p_{data}) + KL(p_{data} \| p_{model})$. CE 최소화 시 하한 $H(p_{data})$에 도달 = perfect model.

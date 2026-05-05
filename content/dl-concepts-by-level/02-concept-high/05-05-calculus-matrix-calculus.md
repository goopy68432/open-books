---
title: "05. 미적분학 & 행렬 미적분 -- 역전파의 수학적 기초"
slug: 05-calculus-matrix-calculus
order: 5
---

# 05. 미적분학 & 행렬 미적분 -- 역전파의 수학적 기초

> 이 문서는 고등학생 수준에서 미적분과 행렬 미적분을 설명합니다.

---

## 왜 이걸 배워야 할까?

딥러닝의 핵심 알고리즘인 **역전파(Backpropagation)**는 결국 **편미분의 연쇄법칙(Chain Rule)**을 행렬 단위로 수행하는 것이다. 이것을 이해하지 못하면, 신경망이 "왜" 학습되는지 이해할 수 없다.

```
[입력] --> [레이어1] --> [레이어2] --> ... --> [출력] --> [손실값]
                                                           |
역전파: <-- 기울기 <-- 기울기 <-- ... <-- 기울기 <-- 기울기 계산
```

---

## 1. 연립방정식 $Ax = b$의 해 (복습 + 심화)

### 세 가지 경우 한눈에

```
경우 1: 유일한 해          경우 2: 무한히 많은 해        경우 3: 해 없음
  *                         --------                       *  b
 /                         /////////  (해 집합)           /
o--- A의 열공간            o--- A의 열공간              o--- A의 열공간
                                                         (b가 밖에 있음)
```

### Moore-Penrose 유사역행렬의 통합적 역할

모든 경우에서 $x^* = A^+b$가 "최선의 답"을 준다:
- 유일한 해가 있으면: 그 해
- 무한히 많으면: **노름이 가장 작은** 해 (최소노름해)
- 해가 없으면: **오차가 가장 작은** 해 (최소제곱해)

### 해가 무한히 많을 때의 구조

일반해: $x = A^+b + (I - A^+A)w$ (임의의 $w$)

- $A^+b$: 최소노름해 (특수해)
- $(I - A^+A)w$: 영공간의 원소 (자유도)
- 두 성분은 **직교**하므로, 노름을 최소화하려면 $w = 0$

### 딥러닝 연결

과매개변수(overparameterized) 모델: 파라미터 수 > 데이터 수이면 해가 무한히 많다. 경사하강법(SGD)이 그 중 어떤 해를 찾는지가 현대 딥러닝 이론의 핵심 질문이다.

---

## 2. Newton's Method (뉴턴 방법)

### 핵심 아이디어

"접선으로 다음 추측을 구하는" 반복법

### 일변수 영점 찾기

$f(x) = 0$의 해를 찾고 싶을 때:

$$\boxed{x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}}$$

접선의 방정식에서 $x$축과 만나는 점을 다음 추측으로 사용한다.

### 예시: $\sqrt{7}$ 구하기

$f(x) = x^2 - 7 = 0$을 풀면 $\sqrt{7}$을 구할 수 있다.

$$x_{n+1} = x_n - \frac{x_n^2 - 7}{2x_n} = \frac{1}{2}\left(x_n + \frac{7}{x_n}\right)$$

| 단계 | 값 | 정확도 |
|------|----|--------|
| $x_0 = 3$ | 3.000000... | 소수점 0자리 |
| $x_1$ | 2.666666... | 소수점 1자리 |
| $x_2$ | 2.645833... | 소수점 3자리 |
| $x_3$ | 2.645751... | 소수점 7자리 |
| $\sqrt{7}$ | 2.6457513... | - |

4번만 반복해도 소수점 7자리까지 정확! 이것이 **2차 수렴(quadratic convergence)**의 위력이다.

### 최적화로의 확장

함수 $L(x)$의 최솟값을 찾고 싶다면?
- 최솟값에서 $\nabla L = 0$ (기울기가 0)
- Newton's method로 $\nabla L = 0$을 풀면 된다:

$$x_{n+1} = x_n - \frac{\nabla L(x_n)}{\nabla^2 L(x_n)} = x_n - [H(x_n)]^{-1}\nabla L(x_n)$$

여기서 $H = \nabla^2 L$은 **Hessian 행렬** (2차 도함수의 행렬)

### 경사하강법과의 관계

| 방법 | 업데이트 공식 | 곡률 정보 |
|------|-------------|----------|
| 경사하강법 (1차) | $x_{n+1} = x_n - \alpha \nabla L$ | 사용 안 함 |
| Newton's method (2차) | $x_{n+1} = x_n - H^{-1}\nabla L$ | Hessian 사용 |

경사하강법은 Newton's method에서 Hessian을 $\frac{1}{\alpha}I$ (항등행렬의 스칼라 배)로 근사한 것!

### 왜 딥러닝에서 순수 Newton's method를 안 쓰는가?

- Hessian 계산: $O(n^2)$ 메모리, $O(n^3)$ 시간 -- 파라미터가 수십억인 LLM에서는 불가능
- 대안: L-BFGS (Hessian의 저랭크 근사), Adam (대각 근사) 등

### 오해하기 쉬운 포인트

| 오해 | 실제 |
|------|------|
| Newton's method는 항상 수렴한다 | 초기값이 나쁘면 발산할 수 있다. $f'(x_n) = 0$이면 정의조차 안 됨 |
| 2차 방법이 항상 더 좋다 | Hessian 계산 비용이 너무 커서 실용적이지 않은 경우가 많다 |

---

## 3. 편미분과 Gradient

### 편미분: 한 변수만 살짝 변화

$f(x_1, x_2) = x_1^2 + 3x_1 x_2$일 때:

$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2 \quad (\text{$x_2$를 상수 취급하고 $x_1$에 대해 미분})$$

$$\frac{\partial f}{\partial x_2} = 3x_1 \quad (\text{$x_1$를 상수 취급하고 $x_2$에 대해 미분})$$

### Gradient (기울기 벡터)

모든 편미분을 모아 하나의 벡터로:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \end{bmatrix} = \begin{bmatrix} 2x_1 + 3x_2 \\ 3x_1 \end{bmatrix}$$

**Gradient의 의미**: 함수가 **가장 빠르게 증가하는 방향**

- 경사하강법: gradient의 **반대 방향**으로 이동하면 함수값이 줄어든다
- $\theta_{\text{새}} = \theta_{\text{이전}} - \alpha \nabla_\theta L$

### 딥러닝 학습의 핵심 도식

```
순전파:  입력 x --> 레이어들 --> 예측값 y_hat --> 손실 L(y_hat, y)
                                                        |
역전파:  ∂L/∂w <-- ... <-- ∂L/∂y_hat <-- 각 가중치별 편미분 계산
                                                        |
업데이트: w_new = w_old - α * ∂L/∂w
```

---

## 4. Jacobian 행렬

### 핵심 정의

벡터 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$의 미분 = **Jacobian 행렬**

$$\frac{\partial u}{\partial v} = \begin{bmatrix} \frac{\partial u_1}{\partial v_1} & \cdots & \frac{\partial u_1}{\partial v_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial u_m}{\partial v_1} & \cdots & \frac{\partial u_m}{\partial v_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

$(i, j)$ 원소의 의미: **"$j$번째 입력이 살짝 변할 때 $i$번째 출력이 얼마나 변하는가"**

### Gradient와 Jacobian의 관계

| 함수 유형 | 미분 결과 | 크기 |
|-----------|----------|------|
| $f: \mathbb{R}^n \to \mathbb{R}$ (스칼라 출력) | Gradient $\nabla f$ | $n$-벡터 |
| $f: \mathbb{R}^n \to \mathbb{R}^m$ (벡터 출력) | Jacobian $J$ | $m \times n$ 행렬 |

Gradient는 Jacobian의 특수한 경우 ($m = 1$)이다.

### 예시: Softmax의 Jacobian

Softmax: $p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$

Jacobian의 각 원소:

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

행렬로 쓰면: $J = \text{diag}(p) - pp^{\top}$

**의미**:
- 대각 원소: 자기 자신에 대한 민감도 ($p_i$가 클수록 덜 민감)
- 비대각 원소: 다른 클래스와의 **경쟁 관계** (하나가 올라가면 나머지는 내려감)

---

## 5. 연쇄법칙 (Chain Rule) -- 역전파의 수학적 본질

### 일변수 연쇄법칙 (고등학교 미적분)

$y = f(g(x))$이면:

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

### 다변수 연쇄법칙 (Jacobian의 곱)

$L = g(f(x))$에서 $f: \mathbb{R}^n \to \mathbb{R}^m$, $g: \mathbb{R}^m \to \mathbb{R}$이면:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial x}$$

이것은 **Jacobian의 행렬곱**이다!

### 역전파 = 연쇄법칙의 역순 계산

3개 레이어 신경망: $L = h(g(f(x)))$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial g} \cdot \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$$

역전파는 이 곱을 **출력에서 입력 방향으로** 순차적으로 계산하여, 중간 결과를 재사용한다 (동적 프로그래밍).

```
순전파:  x --f--> a --g--> b --h--> L

역전파:  ∂L/∂x <-- ∂L/∂a <-- ∂L/∂b <-- ∂L/∂L = 1
         (J_f^T)    (J_g^T)    (J_h^T)
```

### 놀라운 결과: Softmax + Cross-Entropy

Cross-entropy 손실: $L = -\sum y_i \log p_i$
Softmax 확률: $p_i = \text{softmax}(z)_i$

Chain Rule로 $\frac{\partial L}{\partial z}$를 구하면:

$$\boxed{\frac{\partial L}{\partial z} = p - y}$$

예측 확률에서 정답을 빼기만 하면 된다! 이 단순함이 Softmax + Cross-entropy 조합이 표준인 이유이다.

---

## 6. vec 연산자와 Kronecker Product (참고)

### vec 연산자

행렬을 **열 순서대로 쌓아서** 하나의 긴 벡터로 만드는 연산:

$$\text{vec}\begin{pmatrix} a & b \\ c & d \end{pmatrix} = \begin{pmatrix} a \\ c \\ b \\ d \end{pmatrix}$$

(열 순서: 1열 먼저, 2열 다음)

### Kronecker Product ($\otimes$)

$A$의 각 원소에 $B$ 전체를 곱해서 만든 블록 행렬:

$$\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \otimes B = \begin{bmatrix} a_{11}B & a_{12}B \\ a_{21}B & a_{22}B \end{bmatrix}$$

### 핵심 성질

$$(A \otimes B) \cdot \text{vec}(C) = \text{vec}(BCA^{\top})$$

이 공식은 행렬 미분을 체계적으로 계산하는 데 핵심적이다.

### 딥러닝 연결

- **K-FAC**: Fisher Information Matrix를 Kronecker 구조로 근사하여 2차 최적화를 실용적으로 구현하는 알고리즘

---

## 7. Sherman-Morrison-Woodbury 공식 (참고)

### 핵심 공식

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

### 왜 유용한가?

큰 행렬 $A$의 역행렬을 이미 알고 있을 때, **저랭크 업데이트** $UCV$가 추가되어도 작은 행렬의 역행렬만 새로 구하면 된다.

- $n \times n$ 역행렬 $O(n^3)$ 대신 $r \times r$ 역행렬만 계산 ($r \ll n$)

### 딥러닝 연결

- **LoRA**: 거대 언어모델의 가중치를 저랭크로 업데이트하는 기법의 수학적 기초

---

## 정리 / 요약

### 한 줄 요약

> 편미분은 역전파의 원자(atom)이고, Jacobian은 편미분의 행렬이며, Chain Rule은 Jacobian의 곱이다. 역전파는 Chain Rule을 뒤에서 앞으로 계산하는 것이다. 이 네 문장이 딥러닝 학습의 수학적 본질 전체이다.

### 핵심 개념 연결도

```
편미분 (∂f/∂x_i)
    |
    v
Gradient (모든 편미분의 벡터) -- 스칼라 함수의 미분
    |
    v
Jacobian (편미분의 행렬) -- 벡터 함수의 미분
    |
    v
Chain Rule (Jacobian의 곱) -- 합성함수의 미분
    |
    v
역전파 (Chain Rule의 효율적 역순 계산) -- 딥러닝 학습의 핵심
```

### 핵심 공식 정리

| 개념 | 공식 | 한 줄 설명 |
|------|------|-----------|
| Newton's method | $x_{n+1} = x_n - f(x_n)/f'(x_n)$ | 접선으로 영점 추측 |
| 경사하강법 | $\theta_{n+1} = \theta_n - \alpha \nabla L$ | gradient 반대 방향 이동 |
| Jacobian | $J_{ij} = \partial u_i / \partial v_j$ | 입출력 감도 행렬 |
| Chain Rule | $\partial L / \partial x = (\partial L / \partial f) \cdot (\partial f / \partial x)$ | Jacobian의 곱 |
| Softmax + CE | $\partial L / \partial z = p - y$ | 예측 - 정답 |
| Kronecker | $(A \otimes B)\text{vec}(C) = \text{vec}(BCA^{\top})$ | 행렬 미분의 벡터화 |
| SMW | $(A+UCV)^{-1}$을 효율적으로 계산 | 저랭크 업데이트의 역행렬 |

### 오해하기 쉬운 포인트 모음

| 오해 | 실제 |
|------|------|
| Gradient는 열벡터다 / 행벡터다 | Convention에 따라 다르다. $\frac{\partial s}{\partial v}$는 행벡터, $\nabla_v s$는 열벡터. 혼동 시 shape 에러 |
| Gradient와 Jacobian은 같다 | Gradient는 스칼라 함수의 미분(벡터), Jacobian은 벡터 함수의 미분(행렬) |
| Newton's method는 항상 빠르게 수렴 | 초기값이 해에 가까울 때만 2차 수렴 보장. 나쁜 초기값이면 발산 가능 |
| 편미분과 역전파는 별개 | 역전파 = 편미분의 Chain Rule을 효율적으로 계산하는 **알고리즘** |
| 역행렬이 없으면 끝이다 | 유사역행렬 $A^+$로 항상 최선의 해를 구할 수 있다 |
| SMW 공식은 이론적 장난감 | LoRA, Kalman filter 등에서 실용적으로 핵심 역할 |

### 이 장의 최종 메시지

복잡한 함수(신경망)를 이해하기 위해 **국소적으로 선형 근사(미분)**하고, 그 근사를 이용해 **함수를 개선(최적화)**하는 것 -- 이것이 딥러닝의 수학적 골격이다. 행렬 미적분은 이 골격의 **언어**이며, 역전파는 이 언어로 쓰인 가장 우아한 알고리즘이다.

---
title: "제5장: 미적분학과 행렬 미적분"
slug: 05-calculus-matrix-calculus
order: 5
---

# 제5장: 미적분학과 행렬 미적분

> **선수 과목**: 1-4장 전체 (특히 행렬, SVD, 유사역행렬)
> **후속 연결**: 역전파(Backpropagation), 최적화 이론

---

## 1. 동기부여 및 개요

딥러닝의 핵심 알고리즘인 **역전파(Backpropagation)**는 결국 편미분의 연쇄법칙(Chain Rule)을 행렬 단위로 수행하는 것이다. 행렬 미적분을 이해하지 못하면 신경망이 **"왜"** 학습되는지 이해할 수 없다.

본 장은 일변수 미분에서 시작하여 다변수 미적분(Gradient, Jacobian), 그리고 행렬 미적분(vec 연산자, Kronecker product)까지를 다루며, 이 모든 것이 역전파로 수렴함을 보인다.

```
일변수 미분 ──> Newton's Method ──> 다변수: Gradient, Jacobian
                                             │
vec 연산자, Kronecker Product ──────> 행렬 미분의 벡터화
                                             │
              Chain Rule (Jacobian의 곱) ──> 역전파(Backpropagation)
                                             │
                                   Softmax Jacobian ──> ∂L/∂z = p - y
```

---

## 2. Newton's Method

### 2.1 영점 찾기

함수 $f$의 영점 ($f(x) = 0$)을 찾는 반복법:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**기하학적 해석**: 현재 점에서 접선을 긋고, 접선이 $x$축과 만나는 점을 다음 추측으로 사용한다.

### 2.2 2차 수렴

초기값이 해에 충분히 가까우면, Newton's method는 **2차 수렴(quadratic convergence)**을 보인다:

$$|x_{n+1} - x^*| \leq C|x_n - x^*|^2$$

**예시 (Heron의 방법)**: $\sqrt{7}$을 구할 때 $f(x) = x^2 - 7$:

$$x_{n+1} = \frac{1}{2}\left(x_n + \frac{7}{x_n}\right)$$

| 단계 | 값 | 정확한 자릿수 |
|------|-----|-------------|
| $x_0 = 3$ | 3.0000... | 1자리 |
| $x_1$ | 2.6666... | 2자리 |
| $x_2$ | 2.6457... | 4자리 |
| $x_3$ | 2.6457513... | 10자리+ |

### 2.3 최적화로의 확장

$f(x) = \nabla L(x)$ (손실함수의 gradient)로 놓으면, Newton's method로 $\nabla L = 0$을 찾는 것은 **$L$의 극값을 찾는 것**이다:

$$x_{n+1} = x_n - [H(x_n)]^{-1}\nabla L(x_n)$$

여기서 $H = \nabla^2 L$은 **Hessian 행렬**이다. 이것이 **2차 최적화(second-order optimization)**의 기초이다.

**경사하강법과의 관계**: 경사하강법(GD)은 Hessian을 $\frac{1}{\alpha}I$로 근사한 것이다:

$$x_{n+1} = x_n - \alpha\nabla L(x_n)$$

```python
import numpy as np

# Newton's method: sqrt(7)
x = 3.0
for i in range(5):
    x = x - (x**2 - 7) / (2 * x)
    print(f"x_{i+1} = {x:.15f}")
# x_5 ≈ 2.645751311064591 (np.sqrt(7)과 소수점 15자리까지 일치)
```

### 2.4 딥러닝에서의 한계와 근사

순수 Newton's method는 Hessian 계산이 $O(n^2)$ 메모리, $O(n^3)$ 시간이 필요해서 비실용적이다:

| 근사 방법 | 핵심 아이디어 |
|-----------|-------------|
| L-BFGS | Hessian의 저랭크 근사 |
| K-FAC (Martens & Grosse, 2015) | Kronecker 구조 활용 Fisher 행렬 근사 |
| Natural Gradient | Fisher Information Matrix 기반 |
| Adam (Kingma & Ba, 2015) | 대각 pre-conditioning 근사 |

> **주의**: Newton's method는 항상 수렴하지 않는다. 초기값이 나쁘면 발산할 수 있고, $f'(x_n) = 0$이면 정의되지 않는다.

---

## 3. vec 연산자와 Kronecker Product

### 3.1 vec 연산자

**Definition 3.1.** $\text{vec}(A)$는 행렬 $A$의 열(column)을 위에서 아래로 순서대로 쌓아 하나의 긴 벡터로 만든다:

$$\text{vec}\begin{pmatrix} a & b \\ c & d \end{pmatrix} = \begin{pmatrix} a \\ c \\ b \\ d \end{pmatrix}$$

> **주의**: 열 우선(column-major) 순서이다. NumPy의 `flatten()` 기본값은 행 우선(`order='C'`)이므로, `flatten(order='F')`를 명시해야 vec에 대응한다.

### 3.2 Kronecker Product

**Definition 3.2.** $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$에 대해:

$$A \otimes B = \begin{bmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{bmatrix} \in \mathbb{R}^{mp \times nq}$$

### 3.3 핵심 성질

| 성질 | 수식 |
|------|------|
| 전치 | $(A \otimes B)^\top = A^\top \otimes B^\top$ |
| 역행렬 | $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$ |
| 혼합곱 | $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ |
| **vec 변환** | $(A \otimes B)\text{vec}(C) = \text{vec}(BCA^\top)$ |

**핵심 항등식**: $(A \otimes B)\text{vec}(C) = \text{vec}(BCA^\top)$은 행렬 미분을 벡터화하여 체계적으로 계산하는 데 필수적이다. 이것이 **행렬 미적분의 기계적 계산**을 가능하게 한다.

```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.eye(2)

# vec 변환 성질 검증
lhs = np.kron(A, B) @ C.flatten(order='F')
rhs = (B @ C @ A.T).flatten(order='F')
assert np.allclose(lhs, rhs)
```

### 3.4 딥러닝 연결: K-FAC

Kronecker product는 **K-FAC (Kronecker-Factored Approximate Curvature)** 최적화의 핵심이다. 레이어별 Fisher Information Matrix를 Kronecker 구조로 근사하여, $O(n^3)$ 대신 실용적인 비용으로 2차 최적화를 수행한다.

---

## 4. Sherman-Morrison-Woodbury 공식

### 4.1 공식

**Theorem 4.1 (SMW).** $A$가 가역이고 적절한 조건을 만족할 때:

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

### 4.2 의의

큰 행렬 $A$의 역행렬을 이미 알고 있을 때, 저랭크 업데이트 $UCV$가 추가되어도 **작은 행렬의 역행렬만** 새로 구하면 된다: $O(n^3) \to O(nr^2)$ ($r$: 업데이트의 rank).

### 4.3 딥러닝 연결

- **LoRA** (Hu et al., 2021): $W_{\text{new}} = W_0 + BA$ (저랭크 업데이트)
- **Gaussian Process**: 커널 행렬 업데이트
- **Kalman Filter**: 상태 추정 업데이트

```python
import numpy as np
n = 5
A = np.random.randn(n, n); A = A @ A.T + np.eye(n)
U = np.random.randn(n, 2); C = np.eye(2); V = np.random.randn(2, n)

direct = np.linalg.inv(A + U @ C @ V)
A_inv = np.linalg.inv(A)
smw = A_inv - A_inv @ U @ np.linalg.inv(C + V @ A_inv @ U) @ V @ A_inv
assert np.allclose(direct, smw)
```

---

## 5. Gradient와 Jacobian

### 5.1 편미분에서 Gradient로

**Definition 5.1 (Gradient).** 스칼라 함수 $f: \mathbb{R}^n \to \mathbb{R}$의 gradient:

$$\nabla_v f = \left(\frac{\partial f}{\partial v}\right)^\top = \begin{bmatrix} \frac{\partial f}{\partial v_1} \\ \vdots \\ \frac{\partial f}{\partial v_n} \end{bmatrix} \in \mathbb{R}^n$$

**직관**: $\nabla f(x)$는 $f$가 가장 빠르게 증가하는 방향을 가리킨다. 경사하강법은 이 반대 방향으로 이동한다.

### 5.2 Jacobian 행렬

**Definition 5.2 (Jacobian).** 벡터 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$의 Jacobian (Numerator Layout):

$$\frac{\partial u}{\partial v} = \begin{bmatrix} \frac{\partial u_1}{\partial v_1} & \cdots & \frac{\partial u_1}{\partial v_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial u_m}{\partial v_1} & \cdots & \frac{\partial u_m}{\partial v_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

$(i, j)$ 원소의 의미: "$j$번째 입력이 미소 변화할 때 $i$번째 출력이 얼마나 변하는가."

### 5.3 Gradient와 Jacobian의 관계

| 함수 유형 | 미분 형태 | 결과 |
|-----------|----------|------|
| $f: \mathbb{R} \to \mathbb{R}$ | $f'(x)$ | 스칼라 |
| $f: \mathbb{R}^n \to \mathbb{R}$ | $\nabla f$ | 벡터 $\in \mathbb{R}^n$ |
| $f: \mathbb{R}^n \to \mathbb{R}^m$ | Jacobian $J$ | 행렬 $\in \mathbb{R}^{m \times n}$ |

Gradient는 Jacobian의 특수한 경우 ($m = 1$)의 전치이다.

> **주의**: Numerator layout과 Denominator layout은 convention 차이이다. 본 강의는 Numerator layout을 따른다. 혼용하면 전치 오류가 발생한다.

---

## 6. Chain Rule과 역전파

### 6.1 Chain Rule의 행렬 표현

합성함수 $L = g(f(x))$의 미분:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial x}$$

이것은 **Jacobian의 행렬곱**이다.

### 6.2 역전파 = Chain Rule의 역순 계산

$n$층 신경망 $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$에서:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial x}$$

역전파는 이 곱을 **출력에서 입력 방향으로** 순차적으로 계산하여, 중간 결과를 재사용하는 **동적 프로그래밍**이다.

> **핵심 통찰**: 편미분은 역전파의 원자(atom)이다. Jacobian은 편미분의 행렬이다. Chain Rule은 Jacobian의 곱이다. 역전파는 Chain Rule을 뒤에서 앞으로 계산하는 것이다. **이 네 문장이 딥러닝 학습의 수학적 본질 전체이다.**

---

## 7. Softmax의 Jacobian

### 7.1 Softmax 함수

$$p_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}, \quad p = \text{softmax}(z)$$

### 7.2 Jacobian 유도

**Theorem 7.1.** Softmax의 Jacobian:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j) = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

행렬 형태: $J = \text{diag}(p) - pp^\top$

*증명 스케치.*
- $i = j$: $\frac{\partial}{\partial z_i}\frac{e^{z_i}}{\sum_k e^{z_k}} = \frac{e^{z_i}\sum - e^{z_i}e^{z_i}}{(\sum)^2} = p_i - p_i^2 = p_i(1-p_i)$
- $i \neq j$: $\frac{\partial}{\partial z_j}\frac{e^{z_i}}{\sum_k e^{z_k}} = \frac{-e^{z_i}e^{z_j}}{(\sum)^2} = -p_ip_j$ $\square$

### 7.3 Cross-Entropy와의 결합

**Theorem 7.2.** Cross-entropy loss $L = -\sum_i y_i \log p_i$와 Softmax를 Chain Rule로 결합하면:

$$\frac{\partial L}{\partial z} = p - y$$

*증명.* $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z}$에서 $\frac{\partial L}{\partial p_i} = -y_i/p_i$이고 $\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top$:

$$\frac{\partial L}{\partial z_j} = \sum_i \left(-\frac{y_i}{p_i}\right)(p_i\delta_{ij} - p_ip_j) = -y_j + p_j\sum_i y_i = p_j - y_j \quad \square$$

(마지막 등호에서 $\sum_i y_i = 1$ 사용)

이 우아한 결과 $\nabla_z L = p - y$가 **Softmax + Cross-entropy 조합이 분류 문제의 표준인 이유**이다.

```python
import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def softmax_jacobian(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)

z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
J = softmax_jacobian(z)

# Cross-entropy gradient 검증
y = np.array([1, 0, 0])  # one-hot
dL_dp = -y / p
dL_dz = dL_dp @ J
print(f"Chain Rule: {dL_dz}")
print(f"p - y:      {p - y}")
print(f"일치: {np.allclose(dL_dz, p - y)}")  # True
```

### 7.4 Jacobian의 구조적 의미

- 대각 원소 ($i = j$): $p_i(1 - p_i)$ → 자기 자신에 대한 민감도 (확률이 0.5에 가까울 때 최대)
- 비대각 원소 ($i \neq j$): $-p_i p_j$ → 클래스 간 **경쟁** 관계 (하나가 올라가면 나머지 내려감)

---

## 8. 표기법 정리: Numerator Layout

본 강의에서 사용하는 Numerator Layout Convention:

| 미분 | 표기 | 형태 | 크기 |
|------|------|------|------|
| 벡터를 스칼라로 | $\frac{\partial v}{\partial s}$ | 열벡터 | $n \times 1$ |
| 스칼라를 벡터로 | $\frac{\partial s}{\partial v}$ | 행벡터 | $1 \times n$ |
| Gradient | $\nabla_v s = (\frac{\partial s}{\partial v})^\top$ | 열벡터 | $n \times 1$ |
| Jacobian | $\frac{\partial u}{\partial v}$ | 행렬 | $m \times n$ |

> **주의**: PyTorch의 `.grad`는 파라미터와 같은 shape (= Gradient, 열벡터 해석). 수식의 행/열 해석과 코드의 shape을 혼동하면 shape 불일치 버그가 발생한다.

---

## 9. 연립방정식의 해 (복습 및 확장)

### 9.1 유사역행렬을 통한 통합 해법

4장의 내용을 행렬 미적분 관점에서 확장한다.

$Ax = b$의 일반해 (해가 존재할 때):

$$x = A^+b + (I - A^+A)w \quad (\forall w \in \mathbb{R}^n)$$

- $A^+b$: 최소노름해 (행공간 $\mathscr{R}(A^\top)$에 속함)
- $(I - A^+A)w$: 영공간 $\ker(A)$의 성분

직교성: $\langle A^+Ax^*, (I-A^+A)w \rangle = 0$ (Moore-Penrose 조건의 멱등성에 의해)

### 9.2 딥러닝의 과매개변수화와의 연결

과매개변수화(overparameterized) 모델은 $\text{rank}(A) < n$ (미지수가 방정식보다 많음)과 유사하다. 무한히 많은 해 중에서 SGD가 찾는 해가 왜 **최소노름해에 가까운지**가 현대 딥러닝 이론의 핵심 질문이다 (**implicit regularization**).

---

## 10. 딥러닝 적용 요약

| 수학 개념 | 딥러닝 대응 | 왜 중요한가 |
|-----------|------------|------------|
| **Gradient** $\nabla_\theta L$ | 역전파의 핵심 산출물 | **모든 학습의 기초** |
| **Jacobian** $\partial u/\partial v$ | 레이어별 미분 행렬 | **Chain Rule의 빌딩 블록** |
| **Chain Rule** | 역전파 알고리즘 | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial x}$ |
| Softmax Jacobian | 분류 출력층 역전파 | $\partial L/\partial z = p - y$ |
| Newton's Method | 2차 최적화 | L-BFGS, K-FAC |
| Kronecker Product | K-FAC, Fisher 행렬 근사 | 효율적 2차 최적화 |
| SMW 공식 | LoRA, 저랭크 업데이트 | 효율적 fine-tuning |
| 유사역행렬 | 최소제곱 학습, 선형 회귀 | Closed-form 해 |
| 최소노름해 | Implicit regularization | SGD가 찾는 해의 특성 |

---

## 11. 흔한 오해와 주의점

1. **"Gradient는 행벡터다/열벡터다"** → Convention 문제. $\frac{\partial s}{\partial v}$는 행벡터, $\nabla_v s$는 열벡터. 혼동은 shape 버그의 원인.
2. **"Jacobian = Gradient"** → Gradient는 스칼라 함수 ($m=1$)의 미분. Jacobian은 벡터 함수의 미분으로 **행렬**이다. Gradient는 Jacobian의 특수 경우.
3. **"Newton's method는 항상 빠르다"** → 초기값이 나쁘면 발산. 2차 수렴은 해 근처에서만 보장.
4. **"편미분과 역전파는 다르다"** → 역전파는 편미분의 Chain Rule을 **효율적으로 계산하는 알고리즘**이다. 수학적으로 동일.
5. **"vec은 단순 reshape"** → Column-major 순서가 핵심. Kronecker product와의 항등식은 이 순서에 의존.
6. **"SMW 공식은 이론적 장난감"** → LoRA, Kalman filter, GP에서 핵심적으로 사용.
7. **"역행렬이 없으면 끝"** → 유사역행렬 $A^+$가 모든 경우를 처리.

---

## 12. 핵심 요약

1. **편미분 → Gradient → Jacobian → Chain Rule → 역전파**: 이 다섯 단계가 딥러닝 학습의 수학적 골격이다.
2. **Newton's Method**: 접선으로 다음 추측을 구하는 반복법. 최적화로 확장하면 $x_{n+1} = x_n - H^{-1}\nabla L$. 경사하강법은 Hessian을 $\frac{1}{\alpha}I$로 근사한 것.
3. **Softmax + Cross-entropy**: Jacobian 구조 덕분에 역전파 결과가 $p - y$라는 우아한 형태가 된다.
4. **vec과 Kronecker product**: 행렬 미분을 벡터-벡터 관계로 환원하는 도구. K-FAC의 이론적 기반.
5. **SMW 공식**: 큰 역행렬에 저랭크 업데이트를 효율적으로 반영. LoRA의 수학적 배경.
6. **유사역행렬**: 해가 없으면 최소제곱, 해가 무한이면 최소노름. 딥러닝의 implicit regularization과 연결.

> **한 문장 정리**: 복잡한 함수(신경망)를 이해하기 위해 국소적으로 선형 근사(미분)하고, 그 근사를 이용해 함수를 개선(최적화)하는 것 -- 이것이 딥러닝의 수학적 골격이며, 행렬 미적분은 이 골격의 언어이고, 역전파는 이 언어로 쓰인 가장 우아한 알고리즘이다.

---

## 참고문헌

- Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press. Chapter 4, 6.
- Petersen, K. B. & Pedersen, M. S. (2012). *The Matrix Cookbook*.
- Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR*.
- Martens, J. & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. *ICML*.
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Rumelhart, D. E. et al. (1986). Learning representations by back-propagating errors. *Nature*.

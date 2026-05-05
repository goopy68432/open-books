---
title: "딥러닝 이론 모의고사 #4 — 미분/최적화 집중"
slug: 06-mock-exam-4-calculus
order: 10
---

# 딥러닝 이론 모의고사 #4 — 미분/최적화 집중

> 배점 100점 / 10문제

---

## 문제 1. [10점] 미분 표기법

**(a)** [6점] 아래 표를 완성하시오. 각 미분 유형의 결과 크기(shape)와 딥러닝에서의 역할을 적으시오.

| 미분 유형 | 입력 → 출력 | 결과 크기 | DL에서의 이름 |
|----------|-----------|---------|------------|
| 스칼라를 벡터로 미분 | $s \to \mathbb{R}^n$ | ? | ? |
| 벡터를 벡터로 미분 | $\mathbb{R}^m \to \mathbb{R}^n$ | ? | ? |
| 스칼라를 벡터로 2차 미분 | $s \to \mathbb{R}^n$ | ? | ? |

**(b)** [4점] 그래디언트 $\nabla_v s$와 $\frac{\partial s}{\partial v}$의 관계를 쓰고, 그래디언트의 기하학적 의미("가장 가파른 상승 방향")를 설명하시오.

---

## 문제 2. [10점] 핵심 미분 공식 유도

다음을 유도하시오 (결과만 쓰면 0점).

**(a)** [3점] $\frac{\partial}{\partial x}(a^\top x)$ (단, $a$는 상수 벡터)

**(b)** [4점] $\frac{\partial}{\partial x}(x^\top Sx)$ (단, $S$는 대칭행렬)

**(c)** [3점] $\nabla_W \|Wx - y\|^2$를 유도하시오.

---

## 문제 3. [10점] 체인룰

**(a)** [4점] 합성함수 $f \circ g$의 체인룰을 행렬 미분 형태로 쓰고, 야코비안의 곱으로 표현하시오.

**(b)** [6점] $f(x) = \sigma(Wx + b)$에서 $\sigma$는 element-wise 활성화 함수일 때, $\frac{\partial f}{\partial x}$를 체인룰로 유도하시오. $\sigma'$의 야코비안이 대각행렬인 이유를 설명하시오.

---

## 문제 4. [10점] Softmax 야코비안 완전 유도

$p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$일 때,

**(a)** [7점] $\frac{\partial p_i}{\partial z_j}$를 $i=j$와 $i \neq j$로 나누어 유도하시오. 몫의 미분법을 사용한 각 단계를 상세히 서술하시오.

**(b)** [3점] 결과를 행렬 형태로 쓰고, 이것이 역전파에서 어떻게 사용되는지 설명하시오.

---

## 문제 5. [10점] Newton's Method

**(a)** [4점] Newton's Method $x_{n+1} = x_n - f(x_n)/f'(x_n)$를 선형 근사(Taylor 1차)로부터 유도하시오.

**(b)** [3점] $f(x) = x^2 - 5$로 놓고 $x_0 = 2$에서 시작하여 $\sqrt{5}$를 3회 반복으로 구하시오.

**(c)** [3점] Newton's Method를 최적화에 적용하면 $\theta_{n+1} = \theta_n - H^{-1}\nabla L$이 된다. 여기서 $H$는 무엇이며, Gradient Descent와의 차이는?

---

## 문제 6. [10점] 라그랑주 승수법

**(a)** [4점] 제약 최적화 $\min_{g(x)=0} f(x)$에서 라그랑지안을 쓰고, 최적점에서 $\nabla f \parallel \nabla g$인 기하학적 이유를 설명하시오.

**(b)** [6점] $\min_{x^2+y^2=1} (x+y)$를 라그랑주 승수법으로 풀시오. 모든 KKT 조건을 확인하시오.

---

## 문제 7. [10점] Fermat 정리와 임계점

**(a)** [3점] Fermat 정리를 서술하시오. 이것이 "미분=0으로 최솟값을 찾는다"의 근거인 이유를 설명하시오.

**(b)** [4점] 임계점(critical point)이 극소인지, 극대인지, 안장점(saddle point)인지 어떻게 판별하는가? 헤시안(Hessian)을 사용한 판별법을 서술하시오.

**(c)** [3점] 딥러닝에서 안장점이 극소점보다 더 흔한 이유를 직관적으로 설명하시오.

---

## 문제 8. [10점] VJP vs JVP

**(a)** [5점] Loss $L$, 출력 $u \in \mathbb{R}^m$, 입력 $v \in \mathbb{R}^n$, 야코비안 $J \in \mathbb{R}^{m \times n}$일 때:
- JVP: $J \cdot \Delta v$의 의미와 크기
- VJP: $\Delta u^\top \cdot J$의 의미와 크기

**(b)** [5점] 파라미터가 $p$개, 출력이 스칼라 1개인 딥러닝에서 VJP가 JVP보다 효율적인 이유를 연산 횟수로 비교하시오.

---

## 문제 9. [10점] 볼록 최적화

**(a)** [4점] 볼록 함수의 정의를 쓰고, $f(x) = \|Ax-b\|^2$가 볼록임을 보이시오.

**(b)** [6점] 볼록 함수에서 "임계점 = 전역 최소점"이 성립하는 이유를 증명하시오. 이것이 정규방정식의 해가 전역 최적인 이유와 어떻게 연결되는지 설명하시오.

---

## 문제 10. [10점] 활성화 함수의 미분

**(a)** [3점] Sigmoid $\sigma(x) = 1/(1+e^{-x})$의 미분이 $\sigma'(x) = \sigma(x)(1-\sigma(x))$임을 유도하시오.

**(b)** [3점] ReLU $r(x) = \max(0,x)$의 미분을 쓰고, $x=0$에서 미분 불가능한 이유와 실전에서의 처리 방법을 설명하시오.

**(c)** [4점] "Firing"의 개념을 설명하고, 활성화 함수가 뉴런의 firing을 모델링하는 방식을 서술하시오. Sigmoid와 ReLU의 firing 특성 차이를 비교하시오.

---
---

# 모범답안

## 답 1.
| 미분 유형 | 결과 크기 | DL 이름 |
|----------|---------|--------|
| 스칼라→벡터 | 행벡터 $1 \times n$ | 그래디언트의 전치 |
| 벡터→벡터 | 행렬 $m \times n$ | 야코비안 (Jacobian) |
| 스칼라→벡터 2차 | 행렬 $n \times n$ | 헤시안 (Hessian) |

$\nabla_v s = (\partial s/\partial v)^\top$ (열벡터). 가장 가파른 상승 방향을 가리키며, 크기는 최대 변화율.

## 답 2.
### (b)
$f(x) = x^\top Sx$를 성분으로: $f = \sum_{i,j} x_i S_{ij} x_j$
$\frac{\partial f}{\partial x_k} = \sum_j S_{kj}x_j + \sum_i x_i S_{ik} = (Sx)_k + (S^\top x)_k$
$S = S^\top$이므로: $\frac{\partial f}{\partial x} = 2x^\top S$ $\square$

### (c)
$L = (Wx-y)^\top(Wx-y) = x^\top W^\top Wx - 2y^\top Wx + y^\top y$
$\frac{\partial L}{\partial W}$: 체인룰로 $u = Wx-y$라 하면
$\nabla_W L = 2(Wx-y)x^\top$ $\square$

## 답 3.
### (b)
$f = \sigma(Wx+b)$. $g(x) = Wx+b$로 놓으면 $f = \sigma \circ g$.
$$\frac{\partial f}{\partial x} = \frac{\partial \sigma}{\partial g} \cdot \frac{\partial g}{\partial x} = \text{diag}(\sigma'(Wx+b)) \cdot W$$
$\sigma'$의 야코비안이 대각행렬인 이유: $\sigma$가 element-wise이므로 $\sigma_i$는 $g_i$에만 의존하고 다른 $g_j$와 무관. → 교차 미분 = 0 → 대각.

## 답 5.
### (a)
$f(x) \approx f(x_n) + f'(x_n)(x - x_n) = 0$으로 놓으면 (1차 근사의 영점)
$x = x_n - f(x_n)/f'(x_n)$ $\square$

### (b)
$x_{n+1} = (x_n + 5/x_n)/2$
$x_0 = 2$: $x_1 = (2 + 2.5)/2 = 2.25$
$x_1 = 2.25$: $x_2 = (2.25 + 5/2.25)/2 = (2.25 + 2.222)/2 = 2.2361$
$x_2$: $x_3 = (2.2361 + 5/2.2361)/2 ≈ 2.23607$
$\sqrt{5} = 2.23607...$, 3회만에 소수점 5자리 정확.

### (c)
$H$ = 헤시안 (Loss의 2차 미분 행렬). GD는 $\theta - \alpha\nabla L$ (1차 정보만), Newton은 $\theta - H^{-1}\nabla L$ (2차 곡률 정보 활용). Newton이 수렴 빠르지만 $H^{-1}$ 계산이 $O(n^3)$으로 비쌈.

## 답 6.
### (b)
$\mathcal{L} = x+y + \lambda(x^2+y^2-1)$
$\partial/\partial x: 1 + 2\lambda x = 0$ → $x = -1/(2\lambda)$
$\partial/\partial y: 1 + 2\lambda y = 0$ → $y = -1/(2\lambda)$
$x^2+y^2=1$: $2/(4\lambda^2) = 1$ → $\lambda^2 = 1/2$ → $\lambda = \pm 1/\sqrt{2}$
$\lambda = 1/\sqrt{2}$: $x=y=-1/\sqrt{2}$, $f = -\sqrt{2}$ (최소)
$\lambda = -1/\sqrt{2}$: $x=y=1/\sqrt{2}$, $f = \sqrt{2}$ (최대)

## 답 8.
### (b)
JVP: 하나의 입력 방향 $\Delta v$에 대해 출력 변화 계산 → 파라미터 $p$개마다 1회씩 = **$p$번** forward pass
VJP: 하나의 출력 그래디언트 $\Delta u$에 대해 모든 입력 그래디언트 계산 → Loss는 스칼라 1개이므로 **1번** backward pass로 $p$개 파라미터의 그래디언트 전부 획득
효율비: VJP는 $O(1)$, JVP는 $O(p)$ → VJP가 $p$배 효율적.

## 답 10.
### (a)
$\sigma(x) = (1+e^{-x})^{-1}$
$\sigma'(x) = -(-e^{-x})(1+e^{-x})^{-2} = \frac{e^{-x}}{(1+e^{-x})^2}$
$= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \cdot \frac{1+e^{-x}-1}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$ $\square$

### (c)
Firing: 생물학적 뉴런이 충분한 입력을 받으면 "발화"(신호 전달)하는 현상.
- Sigmoid: 부드러운 firing (0~1 연속 출력). 문제: 양 극단에서 gradient 소실 (saturation)
- ReLU: 임계값 기반 firing (0 이하 = 미발화, 양수 = 선형 전달). 장점: gradient 보존, 계산 빠름

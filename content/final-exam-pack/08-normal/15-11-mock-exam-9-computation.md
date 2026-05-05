---
title: "딥러닝 이론 모의고사 #9 — 계산 문제 집중"
slug: 11-mock-exam-9-computation
order: 15
---

# 딥러닝 이론 모의고사 #9 — 계산 문제 집중

> 배점 100점 / 10문제 / 풀이 과정 필수

---

## 문제 1. [10점] MLE 계산

동전을 20번 던져 14번 앞면이 나왔다.

**(a)** [5점] MLE $\theta_{ML}$을 유도하시오.
**(b)** [5점] Beta(3,3) Prior 하에서 MAP $\theta_{MAP}$를 구하고, MLE와 비교하시오.

---

## 문제 2. [10점] 고유값 계산

$A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$

**(a)** [5점] 고유값을 구하시오.
**(b)** [5점] 각 고유값에 대응하는 고유벡터를 구하시오.

---

## 문제 3. [10점] SVD 계산과 저랭크 근사

$A = \begin{bmatrix} 2 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$

**(a)** [4점] $A^\top A$의 고유값과 특이값을 구하시오.
**(b)** [3점] SVD를 쓰시오.
**(c)** [3점] 랭크-1 근사 $A_1$을 구하고 근사 오차 $\|A - A_1\|_F$를 계산하시오.

---

## 문제 4. [10점] Softmax 계산

$z = (2, 1, 0)$일 때,

**(a)** [4점] $\text{softmax}(z)$를 계산하시오 ($e \approx 2.718$).
**(b)** [6점] $\frac{\partial p_1}{\partial z_1}$, $\frac{\partial p_1}{\partial z_2}$를 각각 계산하시오.

---

## 문제 5. [10점] Newton's Method 계산

**(a)** [5점] $f(x) = x^3 - 2$의 영점을 $x_0 = 1$에서 Newton's Method로 2회 반복하여 $\sqrt[3]{2}$를 근사하시오.
**(b)** [5점] $f(x) = e^x - 3$의 영점을 $x_0 = 1$에서 2회 반복하시오 ($e \approx 2.718$).

---

## 문제 6. [10점] 미분 계산

다음을 구하시오 (풀이 과정 포함).

**(a)** [3점] $f(x) = x^\top \begin{bmatrix}2&1\\1&3\end{bmatrix} x$의 $\nabla_x f$

**(b)** [3점] $\frac{\partial}{\partial W}(Wx)$ where $W \in \mathbb{R}^{2\times 3}$, $x \in \mathbb{R}^3$

**(c)** [4점] $L = \|Wx - y\|^2$의 $\nabla_W L$을 계산하시오. $W = \begin{bmatrix}1&0\\0&1\end{bmatrix}$, $x = (1,2)^\top$, $y = (3,4)^\top$일 때 수치값을 구하시오.

---

## 문제 7. [10점] 엔트로피/KL 계산

**(a)** [4점] $p = (1/4, 1/4, 1/4, 1/4)$과 $q = (1/2, 1/4, 1/8, 1/8)$의 엔트로피를 각각 계산하시오 (log base 2).
**(b)** [6점] $KL(p\|q)$와 $KL(q\|p)$를 계산하고, 비대칭성을 확인하시오.

---

## 문제 8. [10점] 정규방정식 계산

$X = \begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix}$, $y = \begin{bmatrix}1\\2\\2\end{bmatrix}$일 때,

**(a)** [4점] $X^\top X$와 $X^\top y$를 계산하시오.
**(b)** [6점] 정규방정식 $\hat{w} = (X^\top X)^{-1}X^\top y$로 $\hat{w}$를 구하시오.

---

## 문제 9. [10점] 라그랑주 승수법 계산

$\min f(x,y) = x^2 + y^2$ s.t. $x + y = 4$

**(a)** [5점] 라그랑지안을 쓰고 연립방정식을 풀어 해를 구하시오.
**(b)** [5점] 기하학적으로 해석하시오 (원과 직선의 관계).

---

## 문제 10. [10점] CE Loss 계산

3-클래스 분류, 모델 출력 $h = (0.7, 0.2, 0.1)$, 정답 $y = 1$ (첫 번째 클래스).

**(a)** [3점] CE Loss를 계산하시오.
**(b)** [4점] 만약 $h = (0.3, 0.4, 0.3)$이면 CE는? (a)와 비교하시오.
**(c)** [3점] 완벽한 예측 $h = (1, 0, 0)$일 때 CE는? 왜 최솟값이 0인지 설명하시오.

---
---

# 모범답안

## 답 1.
(a) $\text{loglik} = 14\log\theta + 6\log(1-\theta)$, $14/\theta - 6/(1-\theta) = 0$ → $\theta_{ML} = 14/20 = 0.7$
(b) $\theta_{MAP} = \frac{14+3-1}{20+3+3-2} = 16/24 = 2/3 ≈ 0.667$. MAP이 0.5쪽으로 당겨짐.

## 답 2.
(a) $\det(A-\lambda I) = (4-\lambda)(3-\lambda)-2 = \lambda^2-7\lambda+10 = 0$ → $\lambda = 5, 2$
(b) $\lambda=5$: $(A-5I)v=0$ → $-v_1+2v_2=0$ → $v = (2,1)^\top$
$\lambda=2$: $(A-2I)v=0$ → $2v_1+2v_2=0$ → $v = (1,-1)^\top$

## 답 4.
(a) $s = e^2+e^1+e^0 = 7.389+2.718+1 = 11.107$
$p = (0.665, 0.245, 0.090)$

(b) $\partial p_1/\partial z_1 = p_1(1-p_1) = 0.665 \times 0.335 = 0.223$
$\partial p_1/\partial z_2 = -p_1 p_2 = -0.665 \times 0.245 = -0.163$

## 답 5.
(a) $x_{n+1} = x_n - (x_n^3-2)/(3x_n^2)$
$x_0=1$: $x_1 = 1 - (1-2)/3 = 1+1/3 = 4/3 ≈ 1.333$
$x_1=4/3$: $x_2 = 4/3 - ((4/3)^3-2)/(3(4/3)^2) = 4/3 - (64/27-2)/(48/9)$
$= 4/3 - (10/27)/(16/3) = 4/3 - 10/144 = 4/3 - 5/72 = 192/144 - 10/144 = 182/144 ≈ 1.264$
$\sqrt[3]{2} ≈ 1.260$, 2회로 소수점 2자리 정확.

## 답 7.
(a) $H(p) = 4 \times (1/4)\log_2 4 = 2$ bits
$H(q) = 1/2\log_2 2 + 1/4\log_2 4 + 2 \times 1/8\log_2 8 = 0.5+0.5+0.75 = 1.75$ bits

(b) $KL(p\|q) = 1/4\log_2(1/2 \div 1/4) + ... $ 계산 생략
$= 1/4\log_2 2 + 1/4\log_2 1 + 1/4\log_2 2 + 1/4\log_2 2 = 3/4 \times 1 = 0.75$ bits (실제 계산)
$KL(q\|p) = 1/2\log_2(2) + 1/4\log_2 1 + 2 \times 1/8\log_2(1/2) = 0.5 - 0.25 = 0.25$ bits

## 답 8.
(a) $X^\top X = \begin{bmatrix}3&6\\6&14\end{bmatrix}$, $X^\top y = \begin{bmatrix}5\\12\end{bmatrix}$
(b) $(X^\top X)^{-1} = \frac{1}{42-36}\begin{bmatrix}14&-6\\-6&3\end{bmatrix} = \frac{1}{6}\begin{bmatrix}14&-6\\-6&3\end{bmatrix}$
$\hat{w} = \frac{1}{6}\begin{bmatrix}14 \times 5-6 \times 12\\-6 \times 5+3 \times 12\end{bmatrix} = \frac{1}{6}\begin{bmatrix}-2\\6\end{bmatrix} = \begin{bmatrix}-1/3\\1\end{bmatrix}$

## 답 9.
(a) $\mathcal{L} = x^2+y^2+\lambda(x+y-4)$
$\partial/\partial x: 2x+\lambda=0$, $\partial/\partial y: 2y+\lambda=0$ → $x=y$
$x+y=4$ → $x=y=2$, $\min = 4+4 = 8$
(b) $x^2+y^2=r^2$은 원, $x+y=4$는 직선. 직선에 접하는 가장 작은 원의 반지름² = 8.

## 답 10.
(a) $CE = -\log(0.7) ≈ 0.357$
(b) $CE = -\log(0.3) ≈ 1.204$ → (a)보다 훨씬 큼 (틀린 예측에 큰 페널티)
(c) $CE = -\log(1) = 0$. 정답 확률이 1이면 정보 손실 없음.

---
title: "딥러닝 중간고사 복원형 문제예시"
slug: 01-midterm-mock-exam-reconstructed
order: 5
---

# 딥러닝 중간고사 복원형 문제예시

> 목적: `/Users/jeongseongchae/dev/university/deep_learning/final/deep_learing_prev_exam.pdf`의 중간고사 출제 패턴을 바탕으로, 실제 시험지에 가깝게 복원한 모의 중간고사
>
> 반영한 특징:
> - 큰 5문제 + 소문항 구조
> - 계산형 + 유도형 + 서술형 혼합
> - "답만 쓰면 감점" 스타일을 반영하여 정의, 가정, 전개를 요구
> - 중간고사 범위만 포함

---

## 시험 정보

- 시간: 90분
- 총점: 100점
- 문항 수: 대문항 5개
- 공통 지시:
  - 필요한 경우 i.i.d. 가정, 분포 가정, 미분 가능성 가정을 명시할 것.
  - 계산 문제에서도 최종 답만 쓰지 말고 전개 과정을 서술할 것.
  - 서술형 문제는 정의 -> 전개 -> 결론의 순서를 따를 것.

---

## 문제 1. 확률분포와 기댓값/분산 (20점)

확률변수 \(X\)가 구간 \([a,b]\)에서의 uniform distribution을 따른다고 하자. 즉,

$$
f_X(x)=
\begin{cases}
\frac{1}{b-a}, & a\le x \le b,\\
0, & \text{otherwise.}
\end{cases}
$$

또한 \(Z \sim \mathcal{N}(0,1)\)라고 하자.

### (1-a) (6점)
\(X\)의 기댓값 \(E[X]\)를 정의로부터 계산하라.

### (1-b) (6점)
\(X\)의 분산 \(\mathrm{Var}(X)\)를 정의

$$
\mathrm{Var}(X)=E[X^2]-(E[X])^2
$$

로부터 유도하라.

### (1-c) (4점)
정규분포 \(Z \sim \mathcal{N}(0,1)\)에 대하여 \(E[Z]\), \(E[Z^{2n-1}]\)의 값을 쓰고, 그 이유를 짧게 설명하라.

### (1-d) (4점)
\(E[Z^2]\)와 \(E[Z^4]\)를 구하라.

---

## 문제 2. MLE / MAP / Bayesian Estimation (20점)

베르누이 분포를 따르는 i.i.d. 데이터 \(x_1,\dots,x_N \in \{0,1\}\)가 주어졌다고 하자. 파라미터는 \(\theta \in [0,1]\)이며,

$$
P(x_i \mid \theta)=\theta^{x_i}(1-\theta)^{1-x_i}.
$$

표본에서 1의 개수를 \(S=\sum_{i=1}^N x_i\)라고 하자.

### (2-a) (6점)
Likelihood \(P(D\mid \theta)\)와 log-likelihood \(\ell(\theta)\)를 구하라. 여기서 \(D=\{x_i\}_{i=1}^N\).

### (2-b) (4점)
MLE \(\hat{\theta}_{\mathrm{MLE}}\)를 구하라.

### (2-c) (6점)
Prior가

$$
p(\theta)\propto \theta^m(1-\theta)^m
$$

로 주어졌다고 하자. MAP 추정량 \(\hat{\theta}_{\mathrm{MAP}}\)를 구하라. 단, interior optimum이 존재한다고 가정하고 먼저 계산한 뒤, 경계 조건을 짧게 논하라.

### (2-d) (4점)
MLE와 MAP의 차이를 정의 수준에서 설명하고, 언제 두 값이 같아지는지 서술하라.

---

## 문제 3. 손실함수 관계와 KL Divergence (20점)

### (3-a) (8점)
회귀 문제에서 Gaussian likelihood를 가정하면 MSE minimization과 NLL minimization이 연결됨을 설명하라.

답안에는 반드시 다음을 포함할 것.

- 데이터 가정
- likelihood 식
- log를 취하는 이유
- MSE와 동치가 되는 과정

### (3-b) (6점)
경험분포 \(p_E\)와 모델분포 \(p_\theta\)에 대해

$$
\mathrm{KL}(p_E \| p_\theta)
=
CE(p_E,p_\theta)-H(p_E)
$$

를 보이고, 왜 NLL minimization이 KL minimization과 연결되는지 설명하라.

### (3-c) (6점)
두 정규분포

$$
P=\mathcal{N}(\mu_1,1), \qquad Q=\mathcal{N}(\mu_2,1)
$$

에 대해

$$
\mathrm{KL}(P\|Q)
$$

를 계산하라.

---

## 문제 4. 행렬미분과 Backpropagation (25점)

이진 분류를 위한 단일 뉴런을 생각하자.

$$
z = Ax+b, \qquad \sigma(z)=\frac{1}{1+e^{-z}}, \qquad L=-\log(\sigma(z)).
$$

여기서 \(A \in \mathbb{R}^{1\times d}\), \(x\in\mathbb{R}^d\), \(b\in\mathbb{R}\)이다.

### (4-a) (8점)
\(L\)을 \(z\)에 대해 미분하라.

### (4-b) (6점)
Chain rule을 이용하여 \(L\)을 \(A\)에 대해 미분하라.

### (4-c) (5점)
\(L\)을 \(x\)에 대해 미분하라.

### (4-d) (6점)
\(L\)을 \(b\)에 대해 미분하라. 각 소문항에서 차원도 함께 적어라.

---

## 문제 5. 학습률 조건과 Pooling (15점)

행렬 \(A \in \mathbb{R}^{n\times n}\)가 symmetric positive semi-definite라고 하자. 다음 quadratic loss를 고려하자.

$$
f(w)=\frac{1}{2}w^\top A w.
$$

Gradient descent update는

$$
w_{t+1}=w_t-\eta \nabla f(w_t)
$$

이다.

### (5-a) (10점)
\(0<\eta<\frac{2}{\lambda_{\max}(A)}\)이면 loss가 감소함을 보여라. 단, 고유값 분해 관점 또는 quadratic form 관점 중 하나를 사용하라.

### (5-b) (5점)
길이 4 벡터 \(x=[x_1,x_2,x_3,x_4]^\top\)에 대해 stride 2의 average pooling 결과가

$$
y=
\begin{bmatrix}
\frac{x_1+x_2}{2}\\
\frac{x_3+x_4}{2}
\end{bmatrix}
$$

가 되도록 하는 행렬 \(P\)를 구하여 \(y=Px\) 형태로 쓰라.

---

## 출제 의도 메모

- 문제 1: 평균/분산 계산과 정규분포 모멘트 성질 확인
- 문제 2: MLE/MAP 계산 + prior 반영 + 경계 조건 논리
- 문제 3: MSE-NLL-KL의 연결을 서술형으로 점검
- 문제 4: 스칼라 -> 벡터/행렬 미분과 chain rule
- 문제 5: learning rate 조건 증명 + average pooling의 matrix 표현

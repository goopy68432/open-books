---
title: "딥러닝 중간고사 복원형 문제예시 해설"
slug: 02-midterm-mock-exam-solutions-and-concepts
order: 6
---

# 딥러닝 중간고사 복원형 문제예시 해설

> 대응 시험지: [01_midterm_mock_exam_reconstructed.md](./01_midterm_mock_exam_reconstructed.md)
>
> 목표:
> - 실제 시험 답안처럼 `정의 -> 가정 -> 전개 -> 결론` 순서로 풀이
> - 문제를 풀기 위해 알아야 하는 핵심 개념을 함께 정리
> - 기출에서 반복된 "정의 + 수식 전개 + 응용 연결" 스타일 반영

---

## 문제 1 풀이. 확률분포와 기댓값/분산

### 문제 1에서 알아야 할 개념

- 연속확률변수의 기댓값 정의:

$$
E[X]=\int_{-\infty}^{\infty} x f_X(x)\,dx
$$

- 분산 정의:

$$
\mathrm{Var}(X)=E[(X-E[X])^2]=E[X^2]-(E[X])^2
$$

- 표준정규분포의 대칭성:
  - 홀수차 모멘트는 0
  - 짝수차 모멘트만 남음

---

### (1-a) \(E[X]\) 계산

정의에 따라

$$
E[X]=\int_a^b x \cdot \frac{1}{b-a}\,dx
$$

이다. 상수 \(\frac{1}{b-a}\)를 밖으로 빼면

$$
E[X]=\frac{1}{b-a}\int_a^b x\,dx
$$

이다. 적분하면

$$
\int_a^b x\,dx=\left[\frac{x^2}{2}\right]_a^b=\frac{b^2-a^2}{2}
$$

이므로,

$$
E[X]=\frac{1}{b-a}\cdot \frac{b^2-a^2}{2}
=\frac{(b-a)(a+b)}{2(b-a)}
=\frac{a+b}{2}.
$$

#### 결론

$$
E[X]=\frac{a+b}{2}
$$

즉, uniform distribution의 평균은 구간의 정중앙이다.

---

### (1-b) \(\mathrm{Var}(X)\) 계산

먼저 \(E[X^2]\)를 구한다.

$$
E[X^2]=\int_a^b x^2 \cdot \frac{1}{b-a}\,dx
=\frac{1}{b-a}\int_a^b x^2\,dx
$$

적분하면

$$
\int_a^b x^2\,dx=\left[\frac{x^3}{3}\right]_a^b=\frac{b^3-a^3}{3}
$$

따라서

$$
E[X^2]=\frac{b^3-a^3}{3(b-a)}
$$

이다. 여기서

$$
b^3-a^3=(b-a)(a^2+ab+b^2)
$$

이므로

$$
E[X^2]=\frac{a^2+ab+b^2}{3}.
$$

이제 분산 공식을 적용하면

$$
\mathrm{Var}(X)=E[X^2]-(E[X])^2
$$

이고,

$$
\mathrm{Var}(X)
=
\frac{a^2+ab+b^2}{3}
-
\left(\frac{a+b}{2}\right)^2
$$

이다. 공통분모 \(12\)로 정리하면

$$
\mathrm{Var}(X)
=
\frac{4(a^2+ab+b^2)-3(a^2+2ab+b^2)}{12}
$$

$$
=
\frac{a^2-2ab+b^2}{12}
=
\frac{(b-a)^2}{12}.
$$

#### 결론

$$
\mathrm{Var}(X)=\frac{(b-a)^2}{12}
$$

---

### (1-c) \(E[Z]\), \(E[Z^{2n-1}]\)

표준정규분포 \(\mathcal{N}(0,1)\)는 0을 중심으로 좌우대칭이다. 따라서

$$
E[Z]=0
$$

이다. 또한 홀수차 함수 \(z^{2n-1}\)는 원점 대칭인 홀함수이므로,
대칭인 밀도와 곱해서 전체 구간 적분하면 0이 된다.

#### 결론

$$
E[Z]=0, \qquad E[Z^{2n-1}]=0
$$

---

### (1-d) \(E[Z^2]\), \(E[Z^4]\)

표준정규분포에 대해 잘 알려진 결과는

$$
E[Z^2]=1
$$

이다. 이는 분산이 1이기 때문이기도 하다.

또한 4차 모멘트는

$$
E[Z^4]=3
$$

이다.

#### 결론

$$
E[Z^2]=1,\qquad E[Z^4]=3
$$

---

## 문제 2 풀이. MLE / MAP / Bayesian Estimation

### 문제 2에서 알아야 할 개념

- Bernoulli likelihood:

$$
P(x_i\mid\theta)=\theta^{x_i}(1-\theta)^{1-x_i}
$$

- i.i.d. 가정이 있으면 결합확률이 곱으로 분해됨
- 로그를 취하는 이유:
  - 곱을 합으로 바꾸기 위해
  - 미분을 쉽게 하기 위해
  - 수치적으로 안정적이기 때문에
- MAP은

$$
\log p(\theta\mid D)=\log p(D\mid \theta)+\log p(\theta)+C
$$

를 최대화하는 것

---

### (2-a) Likelihood와 log-likelihood

데이터가 i.i.d.라고 가정하면

$$
P(D\mid \theta)=\prod_{i=1}^N P(x_i\mid \theta)
=\prod_{i=1}^N \theta^{x_i}(1-\theta)^{1-x_i}
$$

이다. 지수들을 모으면

$$
P(D\mid \theta)=\theta^{\sum_i x_i}(1-\theta)^{\sum_i (1-x_i)}
=\theta^S(1-\theta)^{N-S}.
$$

따라서 log-likelihood는

$$
\ell(\theta)=\log P(D\mid \theta)
=S\log \theta +(N-S)\log(1-\theta)
$$

이다.

#### 결론

$$
P(D\mid \theta)=\theta^S(1-\theta)^{N-S}
$$

$$
\ell(\theta)=S\log \theta +(N-S)\log(1-\theta)
$$

---

### (2-b) MLE 계산

MLE는 \(\ell(\theta)\)를 최대화하는 \(\theta\)이다.

미분하면

$$
\frac{d\ell}{d\theta}
=
\frac{S}{\theta}
-
\frac{N-S}{1-\theta}
$$

이다. 이를 0으로 두면

$$
\frac{S}{\theta}
=
\frac{N-S}{1-\theta}
$$

$$
S(1-\theta)=\theta(N-S)
$$

$$
S-S\theta=N\theta-S\theta
$$

$$
S=N\theta
$$

따라서

$$
\hat{\theta}_{\mathrm{MLE}}=\frac{S}{N}.
$$

#### 결론

$$
\hat{\theta}_{\mathrm{MLE}}=\frac{S}{N}
$$

즉, 베르누이 MLE는 표본평균이다.

---

### (2-c) MAP 계산

Prior가

$$
p(\theta)\propto \theta^m(1-\theta)^m
$$

이므로 posterior는 비례식으로

$$
p(\theta\mid D)\propto p(D\mid \theta)p(\theta)
$$

$$
\propto \theta^S(1-\theta)^{N-S}\theta^m(1-\theta)^m
$$

$$
\propto \theta^{S+m}(1-\theta)^{N-S+m}
$$

이다.

따라서 log-posterior는 상수항을 제외하면

$$
(S+m)\log\theta + (N-S+m)\log(1-\theta)
$$

이다.

미분하면

$$
\frac{S+m}{\theta}-\frac{N-S+m}{1-\theta}=0
$$

이므로

$$
(S+m)(1-\theta)=\theta(N-S+m)
$$

전개하면

$$
S+m-\theta(S+m)=\theta(N-S+m)
$$

$$
S+m = \theta[(S+m)+(N-S+m)]
$$

$$
S+m=\theta(N+2m)
$$

따라서

$$
\hat{\theta}_{\mathrm{MAP}}=\frac{S+m}{N+2m}.
$$

#### 경계 조건

이번 prior는 \(\theta=0,1\)에서 0이 되는 대칭형 prior이므로, 일반적으로 \(m>0\)이면 interior 방향으로 끌어당기는 역할을 한다.  
다만 다른 prior나 절댓값이 포함된 prior에서는 미분 불가능한 점 또는 경계 최대값을 별도로 확인해야 한다.

#### 결론

$$
\hat{\theta}_{\mathrm{MAP}}=\frac{S+m}{N+2m}
$$

---

### (2-d) MLE와 MAP의 차이

MLE는 likelihood

$$
p(D\mid \theta)
$$

만 최대화한다. 즉, 데이터를 가장 잘 설명하는 파라미터를 찾는다.

반면 MAP은 posterior

$$
p(\theta\mid D)\propto p(D\mid \theta)p(\theta)
$$

를 최대화한다. 즉, 데이터 적합도뿐 아니라 prior도 함께 반영한다.

두 값이 같아지는 경우는 다음과 같다.

- prior가 uniform이어서 \(\log p(\theta)\)가 상수일 때
- 또는 데이터가 매우 많아 likelihood가 prior를 압도할 때

---

## 문제 3 풀이. 손실함수 관계와 KL Divergence

### 문제 3에서 알아야 할 개념

- Negative log-likelihood(NLL):

$$
\mathrm{NLL}(\theta)=-\sum_i \log p(y_i\mid x_i,\theta)
$$

- MSE:

$$
\mathrm{MSE}=\frac{1}{N}\sum_i (y_i-\hat{y}_i)^2
$$

- Cross-entropy:

$$
CE(p,q)=-\sum_x p(x)\log q(x)
$$

- Entropy:

$$
H(p)=-\sum_x p(x)\log p(x)
$$

- KL divergence:

$$
\mathrm{KL}(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
$$

---

### (3-a) Gaussian likelihood -> MSE와 NLL의 연결

회귀 문제에서

$$
y_i = f_\theta(x_i)+\epsilon_i,\qquad \epsilon_i \sim \mathcal{N}(0,\sigma^2)
$$

라고 가정하자. 그러면

$$
p(y_i\mid x_i,\theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}
\right)
$$

이다.

i.i.d. 가정하에 전체 likelihood는

$$
P(D\mid \theta)=\prod_{i=1}^N p(y_i\mid x_i,\theta)
$$

이다.

로그를 취하면

$$
\log P(D\mid \theta)
=
\sum_{i=1}^N
\left[
-\frac{1}{2}\log(2\pi\sigma^2)
-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}
\right]
$$

$$
=
-\frac{N}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-f_\theta(x_i))^2
$$

이다.

따라서 NLL은

$$
\mathrm{NLL}(\theta)
=
\frac{N}{2}\log(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-f_\theta(x_i))^2
$$

이다.

여기서 \(\sigma^2\)가 고정되어 있으면 첫 항은 상수이고, 둘째 항은 제곱오차합에 비례한다. 따라서

$$
\arg\min_\theta \mathrm{NLL}(\theta)
=
\arg\min_\theta \sum_{i=1}^N (y_i-f_\theta(x_i))^2
$$

가 된다. 즉, NLL minimization은 MSE minimization과 동치이다.

#### 핵심 의미

MSE는 그냥 경험적으로 쓰는 loss가 아니라, Gaussian noise 가정에서 나온 확률적 목적함수다.

---

### (3-b) KL = CE - H 와 NLL의 연결

정의로부터

$$
\mathrm{KL}(p_E\|p_\theta)
=
\sum_x p_E(x)\log\frac{p_E(x)}{p_\theta(x)}
$$

이다. 이를 분리하면

$$
\mathrm{KL}(p_E\|p_\theta)
=
\sum_x p_E(x)\log p_E(x)
-\sum_x p_E(x)\log p_\theta(x)
$$

이다.

첫 번째 항은 entropy의 음수이므로

$$
\sum_x p_E(x)\log p_E(x)=-H(p_E)
$$

이고, 두 번째 항은 cross-entropy의 정의상

$$
-\sum_x p_E(x)\log p_\theta(x)=CE(p_E,p_\theta)
$$

이다. 따라서

$$
\mathrm{KL}(p_E\|p_\theta)=CE(p_E,p_\theta)-H(p_E)
$$

가 성립한다.

#### 왜 NLL minimization과 연결되는가?

경험분포 \(p_E\)는 데이터로 고정되어 있으므로 \(H(p_E)\)는 상수이다. 따라서

$$
\arg\min_\theta \mathrm{KL}(p_E\|p_\theta)
=
\arg\min_\theta CE(p_E,p_\theta)
$$

가 된다. 그리고 empirical cross-entropy는 empirical NLL과 같다.  
즉, NLL minimization은 모델분포 \(p_\theta\)를 경험분포 \(p_E\)에 가깝게 만드는 과정으로 해석된다.

---

### (3-c) \(\mathrm{KL}(\mathcal{N}(\mu_1,1)\|\mathcal{N}(\mu_2,1))\)

동일한 분산 1을 가진 두 정규분포의 KL divergence 공식은

$$
\mathrm{KL}(P\|Q)=\frac{1}{2}(\mu_1-\mu_2)^2
$$

이다.

#### 간단 유도

일반적인 정규분포 KL 공식

$$
\mathrm{KL}\big(\mathcal{N}(\mu_1,\sigma_1^2)\|\mathcal{N}(\mu_2,\sigma_2^2)\big)
=
\log\frac{\sigma_2}{\sigma_1}
+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
-\frac{1}{2}
$$

에 \(\sigma_1^2=\sigma_2^2=1\)을 대입하면

$$
\mathrm{KL}(P\|Q)=0+\frac{1+(\mu_1-\mu_2)^2}{2}-\frac{1}{2}
=\frac{(\mu_1-\mu_2)^2}{2}.
$$

#### 결론

$$
\mathrm{KL}(P\|Q)=\frac{(\mu_1-\mu_2)^2}{2}
$$

---

## 문제 4 풀이. 행렬미분과 Backpropagation

### 문제 4에서 알아야 할 개념

- Sigmoid 미분:

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

- Chain rule
- 선형함수 \(z=Ax+b\)의 미분
- 스칼라 loss를 행렬/벡터에 미분할 때 차원 일치 확인

---

### 먼저 구조 정리

주어진 식은

$$
z=Ax+b,\qquad L=-\log(\sigma(z))
$$

이다.

여기서 \(z\)와 \(L\)은 스칼라이다.

---

### (4-a) \(\frac{\partial L}{\partial z}\)

먼저 \(L=-\log(\sigma(z))\)이므로 chain rule을 쓰면

$$
\frac{dL}{dz}
=
-\frac{1}{\sigma(z)}\sigma'(z)
$$

이다.

Sigmoid 미분

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

를 대입하면

$$
\frac{dL}{dz}
=
-\frac{1}{\sigma(z)}\sigma(z)(1-\sigma(z))
=-(1-\sigma(z))
$$

따라서

$$
\frac{\partial L}{\partial z}=\sigma(z)-1.
$$

#### 결론

$$
\frac{\partial L}{\partial z}=\sigma(z)-1
$$

차원: scalar

---

### (4-b) \(\frac{\partial L}{\partial A}\)

Chain rule에 의해

$$
\frac{\partial L}{\partial A}
=
\frac{\partial L}{\partial z}\frac{\partial z}{\partial A}
$$

이다.

여기서

$$
z=Ax+b
$$

이므로 \(A\)에 대한 미분은 \(x^\top\) 형태로 들어간다. 각 성분으로 보면

$$
z=\sum_{j=1}^d A_j x_j + b
$$

이므로

$$
\frac{\partial z}{\partial A_j}=x_j
$$

이다. 따라서 벡터/행렬 형태로

$$
\frac{\partial z}{\partial A}=x^\top
$$

이고,

$$
\frac{\partial L}{\partial A}=(\sigma(z)-1)x^\top.
$$

#### 결론

$$
\frac{\partial L}{\partial A}=(\sigma(z)-1)x^\top
$$

차원: \(1\times d\)

---

### (4-c) \(\frac{\partial L}{\partial x}\)

마찬가지로 chain rule:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial z}\frac{\partial z}{\partial x}
$$

이다.

여기서 \(z=Ax+b\)이므로

$$
\frac{\partial z}{\partial x}=A^\top
$$

이다. 따라서

$$
\frac{\partial L}{\partial x}
=
(\sigma(z)-1)A^\top.
$$

#### 결론

$$
\frac{\partial L}{\partial x}=(\sigma(z)-1)A^\top
$$

차원: \(d\times 1\)

---

### (4-d) \(\frac{\partial L}{\partial b}\)

\(z=Ax+b\)에서 \(b\)는 그냥 더해지는 scalar이므로

$$
\frac{\partial z}{\partial b}=1
$$

이다. 따라서

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial z}\frac{\partial z}{\partial b}
=
\sigma(z)-1.
$$

#### 결론

$$
\frac{\partial L}{\partial b}=\sigma(z)-1
$$

차원: scalar

---

## 문제 5 풀이. 학습률 조건과 Pooling

### 문제 5에서 알아야 할 개념

- Quadratic loss의 gradient:

$$
\nabla f(w)=Aw
$$

  (단, \(A\)가 symmetric일 때)

- 고유값 분해 관점:
  - 각 고유방향에서 업데이트가 독립적으로 일어남
- 평균 풀링은 선형연산이므로 행렬로 표현 가능

---

### (5-a) \(0<\eta<2/\lambda_{\max}(A)\)이면 loss 감소

먼저

$$
f(w)=\frac{1}{2}w^\top Aw
$$

이고 \(A\)가 symmetric PSD이므로

$$
\nabla f(w)=Aw
$$

이다. 따라서 update는

$$
w_{t+1}=w_t-\eta Aw_t=(I-\eta A)w_t
$$

이다.

이제 \(A\)의 고유값 분해를 쓰자.

$$
A=Q\Lambda Q^\top
$$

여기서 \(\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_n)\), \(\lambda_i\ge 0\)이다.

\(u_t=Q^\top w_t\)라고 두면

$$
u_{t+1}=(I-\eta \Lambda)u_t
$$

가 된다. 즉, 각 좌표별로

$$
u_{t+1}^{(i)}=(1-\eta \lambda_i)u_t^{(i)}
$$

이다.

Loss는 고유좌표계에서

$$
f(w_t)=\frac{1}{2}\sum_i \lambda_i (u_t^{(i)})^2
$$

로 쓸 수 있으므로,

$$
f(w_{t+1})
=
\frac{1}{2}\sum_i \lambda_i (1-\eta\lambda_i)^2 (u_t^{(i)})^2
$$

이다.

따라서 각 항에 대해

$$
(1-\eta\lambda_i)^2<1
$$

이면 loss가 감소한다. 이 조건은

$$
-1<1-\eta\lambda_i<1
$$

와 동치이며, 결국

$$
0<\eta\lambda_i<2
$$

이다.

모든 \(i\)에 대해 성립하려면 가장 큰 고유값에 대해 성립하면 충분하므로

$$
0<\eta<\frac{2}{\lambda_{\max}(A)}
$$

이면

$$
f(w_{t+1})\le f(w_t)
$$

이며, \(w_t\)가 stationary point가 아닌 한 strict decrease가 일어난다.

#### 결론

$$
0<\eta<\frac{2}{\lambda_{\max}(A)}
$$

는 gradient descent가 quadratic loss에서 안정적으로 감소하기 위한 조건이다.

---

### (5-b) Average pooling의 matrix 표현

원하는 출력이

$$
y=
\begin{bmatrix}
\frac{x_1+x_2}{2}\\
\frac{x_3+x_4}{2}
\end{bmatrix}
$$

이므로 이를 \(y=Px\) 형태로 쓰는 행렬 \(P\)는

$$
P=
\begin{bmatrix}
\frac{1}{2} & \frac{1}{2} & 0 & 0\\
0 & 0 & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
$$

이다.

확인하면

$$
Px=
\begin{bmatrix}
\frac{1}{2} & \frac{1}{2} & 0 & 0\\
0 & 0 & \frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4
\end{bmatrix}
=
\begin{bmatrix}
\frac{x_1+x_2}{2}\\
\frac{x_3+x_4}{2}
\end{bmatrix}
$$

가 되어 조건과 일치한다.

---

## 총평: 이 모의시험으로 점검해야 하는 것

### 1. 계산

- uniform mean/variance를 정의부터 적분할 수 있는가
- Bernoulli likelihood에서 MLE/MAP를 손으로 전개할 수 있는가
- softplus가 아니라 sigmoid + log 조합 미분을 chain rule로 밀 수 있는가
- pooling을 행렬로 즉시 쓸 수 있는가

### 2. 서술

- 왜 log를 취하는지 말로 설명할 수 있는가
- 왜 MSE와 NLL이 연결되는지 분포 가정까지 포함해 설명할 수 있는가
- KL과 cross-entropy와 NLL의 연결을 문장으로 묶을 수 있는가
- learning rate 조건의 의미를 "각 고유방향의 수축 조건"으로 설명할 수 있는가

### 3. 실제 시험 감점 포인트

- i.i.d. 가정 미기재
- likelihood와 log-likelihood를 혼동
- MAP에서 prior를 곱하지 않고 그냥 likelihood만 최적화
- 행렬미분에서 차원 확인 누락
- learning rate 조건에서 왜 \(\lambda_{\max}\)가 나오는지 설명 누락

---

## 빠른 복습용 한 줄 암기

- Uniform\([a,b]\): 평균은 가운데, 분산은 길이 제곱의 12분의 1
- Bernoulli MLE: 표본평균
- MAP: likelihood + prior
- Gaussian likelihood -> NLL는 제곱오차와 동치
- KL = CE - H
- \(-\log\sigma(z)\)' = \(\sigma(z)-1\)
- Quadratic GD 수렴 조건: \(0<\eta<2/\lambda_{\max}\)

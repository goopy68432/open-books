---
title: "고급 2: 기하와 정규화로 다시 보기"
slug: 02-geometry-and-regularization
order: 2
---

# 고급 2: 기하와 정규화로 다시 보기

이 문서는 선형대수와 regularization을 한 관점으로 묶습니다.

## 1. 고유값 문제의 기하

행렬:

$$
A=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

이 행렬은 좌표를 바꿉니다.

$$
A\begin{pmatrix}x\\y\end{pmatrix}
=\begin{pmatrix}y\\x\end{pmatrix}
$$

방향이 유지되는 벡터를 찾는 문제가 고유값 문제입니다.

$$
Av=\lambda v
$$

### 계산

$$
A-\lambda I=
\begin{pmatrix}
-\lambda&1\\
1&-\lambda
\end{pmatrix}
$$

$$
\det(A-\lambda I)=\lambda^2-1=0
$$

따라서:

$$
\lambda=1,-1
$$

$\lambda=1$:

$$
v_1=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix}
$$

$\lambda=-1$:

$$
v_2=\frac1{\sqrt2}\begin{pmatrix}1\\-1\end{pmatrix}
$$

검증:

$$
Av_1=v_1,\quad Av_2=-v_2
$$

## 2. 왜 $\det(A-\lambda I)=0$인가

$(A-\lambda I)v=0$에서 $v\ne0$인 해를 원합니다.

행렬 $A-\lambda I$가 가역이면 양변에 역행렬을 곱해 $v=0$밖에 나오지 않습니다.

따라서 $v\ne0$인 해가 있으려면 $A-\lambda I$가 비가역이어야 하고, 이는:

$$
\det(A-\lambda I)=0
$$

입니다.

시험에서 이 설명 한 문장이 중요합니다.

## 3. 대칭행렬과 직교 축

대칭행렬은 입력 공간을 서로 직교하는 축으로 분해하기 좋습니다.

서로 다른 고유값 $\lambda,\mu$의 고유벡터 $v,w$에 대해:

$$
Av=\lambda v,\quad Aw=\mu w
$$

대칭성 때문에:

$$
\lambda v^Tw=(Av)^Tw=v^TAw=\mu v^Tw
$$

따라서:

$$
(\lambda-\mu)v^Tw=0
$$

$\lambda\ne\mu$이면:

$$
v^Tw=0
$$

즉 고유벡터들이 직교합니다.

## 4. Regularization의 기하

딥러닝에서 L2 정규화는 가중치 벡터가 너무 길어지지 않도록 누르는 힘입니다.

$$
\lambda\|w\|^2
$$

기하적으로는 원점에서 멀어질수록 비용을 크게 만듭니다.

## 5. Gaussian prior가 L2가 되는 이유

prior:

$$
w\sim N(0,\tau^2I)
$$

pdf:

$$
p(w)=\frac{1}{(2\pi\tau^2)^{d/2}}
\exp\left(-\frac{\|w\|^2}{2\tau^2}\right)
$$

MAP은:

$$
\arg\max_w p(D|w)p(w)
$$

음의 로그를 취하면:

$$
\arg\min_w \{-\log p(D|w)-\log p(w)\}
$$

prior 부분:

$$
-\log p(w)
=\frac{d}{2}\log(2\pi\tau^2)+\frac{\|w\|^2}{2\tau^2}
$$

상수는 최적점에 영향을 주지 않으므로:

$$
-\log p(w)\equiv \frac{\|w\|^2}{2\tau^2}
$$

따라서 MAP 손실은:

$$
NLL+\lambda\|w\|^2
$$

형태가 됩니다.

## 6. Gaussian noise가 MSE가 되는 이유

회귀에서:

$$
y_i=f_w(x_i)+\epsilon_i,\quad \epsilon_i\sim N(0,\sigma^2)
$$

그러면:

$$
p(y_i|x_i,w)
=\frac{1}{\sqrt{2\pi}\sigma}
\exp\left(-\frac{(y_i-f_w(x_i))^2}{2\sigma^2}\right)
$$

i.i.d와 음의 로그:

$$
NLL
=C+\frac{1}{2\sigma^2}\sum_i (y_i-f_w(x_i))^2
$$

상수와 양의 배율을 빼면 MSE입니다.

## 7. 한 장 요약

| 확률 가정 | 음의 로그 후 남는 것 | 딥러닝 이름 |
|---|---|---|
| Gaussian likelihood | $\sum(y-f)^2$ | MSE |
| Bernoulli likelihood | $-[y\log p+(1-y)\log(1-p)]$ | Binary Cross Entropy |
| Categorical likelihood + softmax | $-\sum y_i\log p_i$ | Cross Entropy |
| Gaussian prior | $\|w\|^2$ | L2 regularization |
| Laplace prior | $\|w\|_1$ | L1 regularization |

마스터 관점에서는 "손실함수는 확률가정의 음의 로그"입니다.

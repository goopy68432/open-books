---
title: "중급 1: 분포 계산 작업대"
slug: 02-intermediate-01-distribution-workbench
order: 6
---

# 중급 1: 분포 계산 작업대

이 문서는 Gaussian, Uniform, Bernoulli를 각각 따로 외우지 않고 같은 작업대에서 계산하도록 만듭니다.

## 1. 모든 분포 계산의 공통 순서

1. 분포의 정의역을 적는다.
2. pmf 또는 pdf를 적는다.
3. 원하는 양을 기댓값 형태로 바꾼다.
4. 합 또는 적분을 계산한다.
5. 대칭성, 정규화, 분산 공식으로 검산한다.

## 2. Uniform 작업대

설정:

$$
X\sim Uniform[a,b],\quad p(x)=\frac{1}{b-a},\quad a\le x\le b
$$

평균:

$$
E[X]=\int_a^b x\frac{1}{b-a}\,dx
=\frac{1}{b-a}\left[\frac{x^2}{2}\right]_a^b
=\frac{b^2-a^2}{2(b-a)}
=\frac{a+b}{2}
$$

2차 모멘트:

$$
E[X^2]=\int_a^b x^2\frac{1}{b-a}\,dx
=\frac{b^3-a^3}{3(b-a)}
=\frac{a^2+ab+b^2}{3}
$$

분산:

$$
\operatorname{Var}(X)
=E[X^2]-E[X]^2
=\frac{a^2+ab+b^2}{3}-\frac{(a+b)^2}{4}
=\frac{(b-a)^2}{12}
$$

검산: $a=0,b=1$이면 $E[X]=1/2$, $\operatorname{Var}(X)=1/12$.

## 3. Gaussian 작업대

설정:

$$
X\sim N(0,1),\quad p(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}
$$

### 가우스 적분

$$
I=\int_{-\infty}^{\infty}e^{-x^2/2}\,dx
$$

제곱해서 2차원으로 보냅니다.

$$
I^2=\int\int_{\mathbb{R}^2}e^{-(x^2+y^2)/2}\,dx\,dy
$$

극좌표 $x=r\cos\phi$, $y=r\sin\phi$, $dxdy=rdrd\phi$:

$$
I^2=\int_0^{2\pi}\int_0^\infty e^{-r^2/2}r\,dr\,d\phi=2\pi
$$

따라서 $I=\sqrt{2\pi}$.

### 모멘트 계산

홀수 모멘트:

$$
E[X]=0,\quad E[X^3]=0
$$

이유: $x e^{-x^2/2}$와 $x^3 e^{-x^2/2}$는 기함수이고 적분 구간이 대칭입니다.

2차 모멘트:

$$
\int_{-\infty}^{\infty}x^2e^{-x^2/2}\,dx
=\int x(xe^{-x^2/2})\,dx
$$

부분적분에서 $u=x$, $dv=xe^{-x^2/2}dx$, $v=-e^{-x^2/2}$:

$$
\int x^2e^{-x^2/2}\,dx
=[-xe^{-x^2/2}]_{-\infty}^{\infty}
+\int e^{-x^2/2}\,dx
=\sqrt{2\pi}
$$

따라서 $E[X^2]=1$.

4차 모멘트:

$$
\int x^4e^{-x^2/2}\,dx
=\int x^3(xe^{-x^2/2})\,dx
=3\int x^2e^{-x^2/2}\,dx
$$

따라서 $E[X^4]=3$.

결론:

| 양 | 값 |
|---|---|
| $E[X]$ | $0$ |
| $E[X^2]$ | $1$ |
| $E[X^3]$ | $0$ |
| $E[X^4]$ | $3$ |

## 4. Bernoulli 작업대

설정:

$$
Y\sim Bern(\theta),\quad P(Y=1)=\theta,\quad P(Y=0)=1-\theta
$$

한 식으로:

$$
p(y|\theta)=\theta^y(1-\theta)^{1-y},\quad y\in\{0,1\}
$$

기댓값:

$$
E[Y]=0\cdot(1-\theta)+1\cdot\theta=\theta
$$

분산:

$$
Y^2=Y\quad \text{이므로}\quad E[Y^2]=\theta
$$

$$
\operatorname{Var}(Y)=E[Y^2]-E[Y]^2=\theta-\theta^2=\theta(1-\theta)
$$

## 5. 변형 문제 처리법

| 변형 | 처리 |
|---|---|
| $N(\mu,\sigma^2)$ | $X=\mu+\sigma Z$, $Z\sim N(0,1)$로 바꾼다 |
| $Uniform[0,1]$ | 일반식에 $a=0,b=1$ 대입 |
| 고차 Gaussian 모멘트 | $E[X^{2n}]=(2n-1)!!$, 홀수는 0 |
| Bernoulli 표본 여러 개 | 합 $k=\sum y_i$로 압축 |

## 6. 중급 훈련

1. $X\sim Uniform[-1,3]$의 평균과 분산을 구하라.
2. $Z\sim N(0,1)$일 때 $E[Z^6]$을 구하라.
3. $X=2Z+3$일 때 $E[X]$, $\operatorname{Var}(X)$를 구하라.
4. $Y\sim Bern(0.7)$일 때 $E[Y]$, $\operatorname{Var}(Y)$를 구하라.

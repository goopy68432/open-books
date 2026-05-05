---
title: "답안 키"
slug: answer-key
order: 1
---

# 답안 키

이 문서는 진단표와 훈련 문제의 핵심 답만 제공합니다. 실제 학습에서는 중간 과정을 먼저 손으로 쓰세요.

## `00-orientation/01-diagnostic.md`

### 입문 진단

1. $y_i$를 모두 더한 값이 $k$이다. Bernoulli이면 성공 횟수다.
2. $X$가 평균 0, 분산 1인 정규분포를 따른다.
3. 양변이 상수배까지 같은 모양이라는 뜻이다.
4. 입력 $z_j$를 조금 바꿀 때 출력 $p_i$가 얼마나 바뀌는지 묻는다.
5. 가능한 값 $x$에 그 값이 나올 밀도 $p(x)$를 곱해 모두 더한 연속 가중평균이다.

### 중급 진단

1. $L(\theta)=\prod_i\theta^{y_i}(1-\theta)^{1-y_i}=\theta^k(1-\theta)^{n-k}$.
2. $E[X]=(a+b)/2$, $\operatorname{Var}(X)=(b-a)^2/12$.
3. $\partial p_i/\partial z_i=p_i(1-p_i)$.
4. $\hat{\theta}_{MAP}=(k+m)/(n+2m)$.
5. $\lambda=1,-1$.

### 고급 진단

1. 로그가 단조증가함수이므로 $L(\theta_1)>L(\theta_2)$이면 $\log L(\theta_1)>\log L(\theta_2)$이다.
2. 페르마 정리: 내부의 미분 가능한 극값에서는 1차 도함수가 0이다.
3. $x^3e^{-x^2/2}$가 기함수이고 적분 구간이 대칭이다.
4. 페르마 정리는 내부 미분 가능점에만 적용되므로 경계와 비미분점은 누락된다.
5. $D_{KL}= -E_P[\log(q/p)]\ge -\log E_P[q/p]=-\log1=0$.

### 마스터 진단

핵심 답:

- i.i.d가 우도를 곱으로 만든다.
- 로그가 손실을 합으로 만든다.
- Bernoulli의 음의 로그는 CE, Gaussian의 음의 로그는 MSE다.
- MAP은 NLL에 $-\log prior$를 더한다.
- Gaussian prior의 음의 로그가 L2다.
- softmax+CE의 그래디언트는 $p-y$다.

## `02-intermediate/01-distribution-workbench.md`

1. $Uniform[-1,3]$: 평균 $1$, 분산 $(4)^2/12=4/3$.
2. $E[Z^6]=5\cdot3\cdot1=15$.
3. $X=2Z+3$이므로 $E[X]=2E[Z]+3=3$, $\operatorname{Var}(X)=4\operatorname{Var}(Z)=4$.
4. $E[Y]=0.7$, $\operatorname{Var}(Y)=0.7\cdot0.3=0.21$.

## `02-intermediate/02-estimation-engine.md`

1. prior $p(\theta)\propto\theta^a(1-\theta)^b$이면 posterior $\propto\theta^{k+a}(1-\theta)^{n-k+b}$이고:

$$
\hat{\theta}_{MAP}=\frac{k+a}{n+a+b}
$$

단, 지수 표현이 Beta의 $(\alpha-1,\beta-1)$인지 문제 표기에 따라 조심합니다.

2. $n=10,k=7,m=3$ symmetric:

$$
\frac{k+m}{n+2m}=\frac{10}{16}=\frac58
$$

3. asymmetric:

$$
\frac{k+m}{n+m}=\frac{10}{13}
$$

4. $k=0$이면 $\hat{\theta}=0$이 경계에 있으므로 $\theta\in(0,1)$ 내부 미분 조건으로 잡히지 않는다.

## `02-intermediate/03-softmax-and-loss.md`

1. $p=(0.2,0.5,0.3)^T$:

$$
J=
\begin{pmatrix}
0.16&-0.10&-0.06\\
-0.10&0.25&-0.15\\
-0.06&-0.15&0.21
\end{pmatrix}
$$

2. 각 행 합은 $0$이다.

3. class 2가 정답이면 $y=(0,1,0)$:

$$
\nabla_z L=(0.2,-0.5,0.3)^T
$$

4. $0\le p_i\le1$이므로 $p_i(1-p_i)\ge0$.

## `04-master/02-exam-simulator.md`

### 모의시험 A 핵심 답

1. $\lambda=\pm1$, $v_1=(1,1)^T/\sqrt2$, $v_2=(1,-1)^T/\sqrt2$, 직접 $Av=\lambda v$ 확인.
2. $E[X]=0$, $E[X^2]=1$, $E[X^3]=0$, $E[X^4]=3$. 가우스 적분은 제곱 후 극좌표.
3. $L=\theta^k(1-\theta)^{n-k}$, $NLL=-k\log\theta-(n-k)\log(1-\theta)$, $\hat{\theta}=k/n$.
4. symmetric: $(k+m)/(n+2m)\to1/2$. asymmetric: $(k+m)/(n+m)\to1$.
5. $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$, $J=\operatorname{diag}(p)-pp^T$, CE와 합치면 $p-y$.
6. 확률모델의 음의 로그가 손실이고, prior의 음의 로그가 정규화라는 서술이 핵심.

### 모의시험 B 핵심 답

1. $Uniform[-2,4]$: 평균 $1$, 분산 $36/12=3$.
2. $\hat{\theta}_{MAP}=(k+a)/(n+a+b)$.
3. 내부 임계점, 경계점, 미분 불가능점.
4. 위 softmax 수치 답과 동일.
5. $D_{KL}(P\|Q)=-E_P[\log(q/p)]\ge-\log E_P[q/p]=0$.

## 경고

일반 Beta prior 표기는 문맥에 따라 다릅니다.

- 문제에서 $p(\theta)\propto\theta^a(1-\theta)^b$라고 직접 주면 MAP은 $(k+a)/(n+a+b)$입니다.
- 통계학의 $Beta(\alpha,\beta)$ 표기는 $p(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}$이므로 MAP은 $(k+\alpha-1)/(n+\alpha+\beta-2)$입니다.

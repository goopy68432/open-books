---
title: "마스터 1: 하나의 이야기로 통합하기"
slug: 01-master-synthesis
order: 1
---

# 마스터 1: 하나의 이야기로 통합하기

마스터 단계의 목표는 8개 기출을 따로 떠올리는 것이 아니라, 하나의 언어로 설명하는 것입니다.

## 1. 딥러닝 이론의 중심 문장

> 모델은 확률을 만들고, 학습은 그 확률의 음의 로그를 줄이며, 정규화는 prior의 음의 로그를 더한 것이다.

이 문장 하나로 MLE, MAP, MSE, Cross Entropy, L2 정규화를 연결할 수 있습니다.

## 2. MLE에서 손실까지

데이터 $D=\{y_i\}_{i=1}^n$가 i.i.d라면:

$$
p(D|\theta)=\prod_{i=1}^n p(y_i|\theta)
$$

MLE:

$$
\hat{\theta}_{MLE}=\arg\max_\theta p(D|\theta)
$$

로그를 취해도 최대점은 같습니다.

$$
\hat{\theta}_{MLE}=\arg\max_\theta \sum_i\log p(y_i|\theta)
$$

최소화 형태:

$$
\hat{\theta}_{MLE}=\arg\min_\theta -\sum_i\log p(y_i|\theta)
$$

즉 학습 손실은 음의 로그우도입니다.

## 3. 세 가지 대표 손실의 출생

### Bernoulli → Binary Cross Entropy

$$
p(y|\theta)=\theta^y(1-\theta)^{1-y}
$$

음의 로그:

$$
-\log p(y|\theta)
=-y\log\theta-(1-y)\log(1-\theta)
$$

이것이 binary cross entropy입니다.

### Categorical + softmax → Cross Entropy

softmax:

$$
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
$$

one-hot 정답 $y$:

$$
L=-\sum_i y_i\log p_i
$$

미분하면:

$$
\frac{\partial L}{\partial z_j}=p_j-y_j
$$

### Gaussian noise → MSE

$$
y_i=f_w(x_i)+\epsilon_i,\quad \epsilon_i\sim N(0,\sigma^2)
$$

음의 로그우도:

$$
NLL=C+\frac{1}{2\sigma^2}\sum_i(y_i-f_w(x_i))^2
$$

상수와 양의 배율을 무시하면 MSE입니다.

## 4. MAP에서 정규화까지

MAP:

$$
\hat{w}_{MAP}=\arg\max_w p(D|w)p(w)
$$

음의 로그:

$$
\arg\min_w \{-\log p(D|w)-\log p(w)\}
$$

Gaussian prior:

$$
p(w)\propto \exp\left(-\frac{\|w\|^2}{2\tau^2}\right)
$$

따라서:

$$
-\log p(w)=C+\frac{\|w\|^2}{2\tau^2}
$$

즉:

$$
MAP=NLL+L2
$$

## 5. MAP prior 힘겨루기

Bernoulli 데이터:

$$
L(\theta)=\theta^k(1-\theta)^{n-k}
$$

| prior | MAP | 극한 |
|---|---|---|
| 없음 | $k/n$ | 데이터만 반영 |
| $\theta^m(1-\theta)^m$ | $(k+m)/(n+2m)$ | $m\to\infty$이면 $1/2$ |
| $\theta^m$ | $(k+m)/(n+m)$ | $m\to\infty$이면 $1$ |
| tent | 구간별 비교 | 좁아질수록 정점 우세 |

마스터 답안의 핵심은 "계산 결과" 뒤에 "데이터와 prior의 힘겨루기"를 말하는 것입니다.

## 6. 구술 방어 스크립트

### 왜 로그를 취합니까?

우도는 i.i.d 때문에 곱 형태입니다. 로그를 취하면 곱이 합이 되어 미분이 쉬워집니다. 또한 로그는 단조증가함수이므로 우도와 로그우도의 최대점이 같습니다. 실제 계산에서도 작은 확률의 곱으로 생기는 underflow를 피할 수 있습니다.

### 왜 미분을 0으로 둡니까?

모수가 내부에 있고 목적함수가 미분 가능하면, 내부 극값에서는 페르마 정리에 의해 1차 도함수가 0입니다. 단, 경계점과 미분 불가능점은 별도로 비교해야 합니다.

### 왜 MAP은 prior를 더한 것입니까?

Bayes 정리에 의해 posterior는 likelihood와 prior의 곱에 비례합니다. 로그를 취하면 곱이 합이 되므로 log posterior는 log likelihood와 log prior의 합입니다. 음의 로그를 최소화하면 NLL에 음의 log prior가 더해진 형태가 됩니다.

### 왜 softmax+CE의 미분은 $p-y$입니까?

softmax 자코비안은 $p_i(\delta_{ij}-p_j)$이고 CE의 $p_i$에 대한 미분은 $-y_i/p_i$입니다. 체인 룰로 곱하면 $p_i$가 약분되고, one-hot의 합이 1이라서 $p_j-y_j$만 남습니다.

## 7. 마스터 완료 과제

아래 문장을 증명과 예시까지 붙여 7분 안에 설명하세요.

> "딥러닝의 대표 손실함수는 임의로 만든 공식이 아니라, 확률모델과 prior를 정한 뒤 음의 로그를 취하면 나오는 결과다."

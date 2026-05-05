---
title: "중급 2: MLE와 MAP 추정 엔진"
slug: 02-estimation-engine
order: 2
---

# 중급 2: MLE와 MAP 추정 엔진

이 문서의 목표는 추정 문제를 하나의 기계처럼 푸는 것입니다.

## 1. MLE 엔진

템플릿:

```text
모델 선언 → 단일 pmf/pdf → i.i.d 곱 → 우도 → 로그 → 미분=0 → 검증
```

### Bernoulli MLE

설정:

$$
y_i\sim Bern(\theta),\quad i=1,\ldots,n,\quad k=\sum_{i=1}^n y_i
$$

단일 pmf:

$$
p(y_i|\theta)=\theta^{y_i}(1-\theta)^{1-y_i}
$$

i.i.d이므로:

$$
L(\theta)=\prod_{i=1}^n p(y_i|\theta)
=\prod_{i=1}^n\theta^{y_i}(1-\theta)^{1-y_i}
=\theta^k(1-\theta)^{n-k}
$$

로그우도:

$$
\ell(\theta)=k\log\theta+(n-k)\log(1-\theta)
$$

미분:

$$
\ell'(\theta)=\frac{k}{\theta}-\frac{n-k}{1-\theta}
$$

0으로 두면:

$$
\frac{k}{\theta}=\frac{n-k}{1-\theta}
\Rightarrow k(1-\theta)=(n-k)\theta
\Rightarrow k=n\theta
$$

결론:

$$
\hat{\theta}_{MLE}=\frac{k}{n}
$$

검증:

$$
\ell''(\theta)=-\frac{k}{\theta^2}-\frac{n-k}{(1-\theta)^2}<0
$$

따라서 내부 임계점은 최댓값입니다. 단, $k=0$ 또는 $k=n$이면 경계해를 따로 봅니다.

## 2. 로그를 취하는 이유 3개

시험 답안에서 빠지면 안 됩니다.

1. 곱이 합이 되어 미분이 쉬워진다.
2. 로그는 단조증가하므로 $\arg\max$ 위치가 보존된다.
3. 작은 확률을 많이 곱할 때 생기는 underflow를 피한다.

## 3. NLL과 손실함수

음의 로그우도:

$$
NLL(\theta)=-\ell(\theta)
=-k\log\theta-(n-k)\log(1-\theta)
$$

관계:

$$
\arg\max_\theta \ell(\theta)=\arg\min_\theta NLL(\theta)
$$

딥러닝의 "loss를 줄인다"는 말은 많은 경우 음의 로그확률을 줄인다는 말입니다.

## 4. MAP 엔진

템플릿:

```text
likelihood 작성 → prior 곱하기 → log posterior → 후보 찾기 → 경계/극한 검토
```

베이즈 정리:

$$
p(\theta|D)\propto p(D|\theta)p(\theta)
$$

MAP:

$$
\hat{\theta}_{MAP}=\arg\max_\theta p(\theta|D)
$$

### symmetric prior

prior:

$$
p(\theta)\propto \theta^m(1-\theta)^m
$$

posterior:

$$
p(\theta|D)\propto \theta^k(1-\theta)^{n-k}\theta^m(1-\theta)^m
=\theta^{k+m}(1-\theta)^{n-k+m}
$$

로그:

$$
\ell_{post}(\theta)=(k+m)\log\theta+(n-k+m)\log(1-\theta)
$$

미분=0:

$$
\frac{k+m}{\theta}-\frac{n-k+m}{1-\theta}=0
$$

결론:

$$
\hat{\theta}_{MAP}=\frac{k+m}{n+2m}
$$

극한:

$$
m\to\infty \Rightarrow \hat{\theta}_{MAP}\to \frac12
$$

해석: prior가 $0.5$를 무한히 강하게 믿으면 데이터의 영향이 사라집니다.

### asymmetric prior

prior:

$$
p(\theta)\propto \theta^m
$$

posterior:

$$
p(\theta|D)\propto \theta^{k+m}(1-\theta)^{n-k}
$$

결론:

$$
\hat{\theta}_{MAP}=\frac{k+m}{n+m}
$$

극한:

$$
m\to\infty \Rightarrow \hat{\theta}_{MAP}\to 1
$$

해석: prior가 $\theta=1$ 쪽으로 무한히 강해지면 데이터가 밀립니다.

## 5. Tent prior 엔진

tent prior는 일반 MAP보다 더 조심해야 합니다.

설정:

$$
L(\theta)=\theta^4(1-\theta)
$$

$$
p_m(\theta)=
\begin{cases}
0,&|\theta-0.5|\ge 1/m\\
m-m^2|\theta-0.5|,&|\theta-0.5|\le 1/m
\end{cases}
$$

절차:

1. prior가 0이 아닌 정의역을 구한다.
2. $0.5$ 기준으로 절댓값을 없애며 구간을 나눈다.
3. 각 구간에서 $L(\theta)p_m(\theta)$의 로그를 미분한다.
4. 나온 후보가 그 구간 안에 있는지 확인한다.
5. 경계와 $\theta=0.5$ 미분 불가능점을 비교한다.

### $m=2$

정의역: $[0,1]$

왼쪽 구간 $[0,0.5]$:

$$
p_2(\theta)=4\theta,\quad f_a(\theta)=4\theta^5(1-\theta)
$$

후보 $\theta=5/6$은 왼쪽 구간 밖이므로 왼쪽에서는 $0.5$가 최대 후보입니다.

오른쪽 구간 $[0.5,1]$:

$$
p_2(\theta)=4(1-\theta),\quad f_b(\theta)=4\theta^4(1-\theta)^2
$$

로그 미분:

$$
\frac{4}{\theta}-\frac{2}{1-\theta}=0
\Rightarrow \theta=\frac23
$$

값 비교 결과 $\theta=2/3$이 더 큽니다.

$$
\hat{\theta}_{MAP}^{m=2}=\frac23
$$

### $m=6$

정의역:

$$
\left[\frac13,\frac23\right]
$$

왼쪽에서는 증가하고 오른쪽에서는 감소합니다. 따라서 뾰족점 $\theta=0.5$가 최대입니다.

$$
\hat{\theta}_{MAP}^{m=6}=\frac12
$$

## 6. 추정 엔진의 핵심 비교

| 문제 | 데이터 힘 | prior 힘 | 결과 |
|---|---|---|---|
| MLE | 있음 | 없음 | $k/n$ |
| symmetric MAP | 있음 | $0.5$ 방향 | $(k+m)/(n+2m)$ |
| asymmetric MAP | 있음 | $1$ 방향 | $(k+m)/(n+m)$ |
| tent $m=2$ | 강함 | 넓은 tent | $2/3$ |
| tent $m=6$ | 약해짐 | 좁은 tent | $1/2$ |

## 7. 중급 훈련

1. $p(\theta)\propto\theta^a(1-\theta)^b$일 때 MAP을 구하라.
2. $n=10,k=7,m=3$인 symmetric prior의 MAP 값을 구하라.
3. $n=10,k=7,m=3$인 asymmetric prior의 MAP 값을 구하라.
4. Bernoulli MLE에서 $k=0$일 때 내부 미분 풀이가 왜 실패하는지 설명하라.

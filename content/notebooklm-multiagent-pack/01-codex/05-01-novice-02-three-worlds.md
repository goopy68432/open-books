---
title: "입문 2: 세 개의 세계로 보기"
slug: 01-novice-02-three-worlds
order: 5
---

# 입문 2: 세 개의 세계로 보기

이 과목의 식은 대부분 세 세계 중 하나에 있습니다.

1. 공간의 세계: 벡터와 행렬
2. 불확실성의 세계: 확률분포
3. 학습의 세계: 손실함수와 최적화

세계를 구분하면 문제가 덜 복잡해집니다.

## 1. 공간의 세계

주인공: 벡터, 행렬, 고유값, 직교, 노름

기본 질문:

> 어떤 방향이 행렬을 통과해도 방향을 유지하는가?

예:

$$
A=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

이 행렬은 두 좌표를 바꿉니다. $(1,1)^T$는 바꿔도 그대로이고, $(1,-1)^T$는 방향이 반대로 뒤집힙니다.

그래서 직관적으로:

| 벡터 | 행렬을 곱한 결과 | 고유값 |
|---|---|---|
| $(1,1)^T$ | $(1,1)^T$ | $1$ |
| $(1,-1)^T$ | $(-1,1)^T=-(1,-1)^T$ | $-1$ |

## 2. 불확실성의 세계

주인공: 확률변수, pmf, pdf, 기댓값, 분산, prior, posterior

기본 질문:

> 이 값이 나올 가능성은 어떤 모양인가?

분포는 모양입니다.

| 분포 | 모양 | 자주 묻는 것 |
|---|---|---|
| Bernoulli | 0 또는 1 | 성공확률 $\theta$ 추정 |
| Uniform | 구간에서 평평함 | 평균과 분산 |
| Gaussian | 종 모양 | 모멘트와 MSE |
| Beta류 prior | $\theta$에 대한 믿음 | MAP |
| Tent prior | 뾰족한 삼각형 | 경계와 비미분점 |

기댓값은 "확률로 가중한 평균"입니다.

이산형:

$$
E[X]=\sum_x xp(x)
$$

연속형:

$$
E[X]=\int xp(x)\,dx
$$

## 3. 학습의 세계

주인공: likelihood, log likelihood, NLL, cross entropy, regularization

기본 질문:

> 어떤 파라미터가 데이터를 가장 잘 설명하는가?

MLE는 데이터 설명력을 최대화합니다.

$$
\hat{\theta}_{MLE}=\arg\max_\theta p(D|\theta)
$$

MAP은 데이터 설명력에 prior까지 더합니다.

$$
\hat{\theta}_{MAP}=\arg\max_\theta p(D|\theta)p(\theta)
$$

딥러닝에서는 보통 최대화보다 최소화를 선호하므로 음의 로그를 씁니다.

$$
\arg\max \log p(D|\theta)
=
\arg\min -\log p(D|\theta)
$$

## 4. 세계 사이의 다리

| 다리 | 의미 | 대표 결과 |
|---|---|---|
| 확률 → 손실 | 음의 로그를 취한다 | Bernoulli → Cross Entropy |
| Gaussian → 손실 | 제곱항이 남는다 | Gaussian noise → MSE |
| prior → 정규화 | 음의 로그 prior를 더한다 | Gaussian prior → L2 |
| 행렬 → 학습 | 점수 벡터를 확률 벡터로 바꾼다 | softmax |
| 증명 → 답안 | 정리의 조건을 말한다 | 페르마, 베이즈, Jensen |

## 5. 입문 완료 기준

다음 질문에 답할 수 있으면 중급으로 넘어갑니다.

1. MLE와 MAP의 차이는 무엇인가?
2. pdf와 pmf의 차이는 무엇인가?
3. 왜 기댓값 계산에는 합 또는 적분이 등장하는가?
4. softmax는 어떤 세계에서 어떤 세계로 가는 함수인가?
5. 고유값 문제는 왜 $\det(A-\lambda I)=0$으로 바뀌는가?

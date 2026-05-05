---
title: "마스터 2: 모의시험 시뮬레이터"
slug: 04-master-02-exam-simulator
order: 12
---

# 마스터 2: 모의시험 시뮬레이터

이 문서는 실전 답안력을 만들기 위한 문제 세트입니다. 제한시간을 실제로 두고 풉니다.

## 사용 규칙

- 책과 원본 자료를 보지 않습니다.
- 답만 쓰지 말고 "왜?" 문장을 최소 1개씩 붙입니다.
- 미분 문제는 정의역, 내부점, 경계를 확인합니다.
- 끝나고 `90-drills/answer-key.md`로 채점합니다.

## 모의시험 A: 60분

### 1번: 고유값과 정의 검증 (10점)

$$
A=\begin{pmatrix}0&1\\1&0\end{pmatrix}
$$

고유값과 고유벡터를 구하고, 정의 $Av=\lambda v$로 검증하라.

채점 포인트:

- $\det(A-\lambda I)=0$의 이유
- $\lambda=\pm1$
- 각 고유벡터
- 직접 검증

### 2번: Gaussian 모멘트 (15점)

$X\sim N(0,1)$일 때 $E[X]$, $E[X^2]$, $E[X^3]$, $E[X^4]$를 구하라. 가우스 적분 $\int e^{-x^2/2}dx=\sqrt{2\pi}$의 증명 아이디어도 쓰라.

채점 포인트:

- 대칭성
- 부분적분
- 가우스 적분의 극좌표 변환

### 3번: Bernoulli MLE (15점)

$y_i\sim Bern(\theta)$ i.i.d, $k=\sum y_i$일 때 우도, NLL, MLE를 구하고 MLE와 NLL의 관계를 설명하라.

채점 포인트:

- i.i.d → 곱
- 로그 이유 3개
- 미분=0 이유
- 2계 미분 검증

### 4번: MAP 비교 (15점)

위 Bernoulli 설정에서 다음 prior에 대해 $\hat{\theta}_{MAP}$을 구하고 $m\to\infty$ 극한을 설명하라.

1. $p(\theta)\propto\theta^m(1-\theta)^m$
2. $p(\theta)\propto\theta^m$

채점 포인트:

- posterior $\propto likelihood\times prior$
- log posterior
- 미분과 극한
- prior 직관

### 5번: Softmax 자코비안 (15점)

$p_i=e^{z_i}/\sum_k e^{z_k}$일 때 $\partial p_i/\partial z_j$를 구하고 행렬 형태로 써라. CE와 합치면 왜 $p-y$가 되는지도 설명하라.

채점 포인트:

- $i=j$, $i\ne j$ 분리
- $p_i(\delta_{ij}-p_j)$
- $J=\operatorname{diag}(p)-pp^T$
- CE 체인 룰

### 6번: 통합 서술 (30점)

다음을 하나의 글로 연결하라.

> i.i.d, 로그우도, NLL, Cross Entropy, MSE, MAP, L2 regularization

채점 포인트:

- 각 단어를 단독 정의하지 않고 연결
- 확률모델의 음의 로그가 손실이 된다는 관점
- prior의 음의 로그가 정규화가 된다는 관점

## 모의시험 B: 변형 45분

### 1번: Uniform 변형

$X\sim Uniform[-2,4]$일 때 $E[X]$, $\operatorname{Var}(X)$를 구하라.

### 2번: 일반 Beta형 prior

Bernoulli likelihood $L(\theta)=\theta^k(1-\theta)^{n-k}$와 prior $p(\theta)\propto \theta^a(1-\theta)^b$가 있을 때 MAP을 구하라.

### 3번: tent prior 판단형

tent prior 문제에서 왜 후보가 "미분=0인 점"만이 아닌지 설명하라. 최소 3종류의 후보를 쓰라.

### 4번: softmax 수치형

$p=(0.2,0.5,0.3)^T$일 때 $J$를 구하라.

### 5번: 증명형

KL divergence가 0 이상임을 Jensen 부등식으로 증명하라.

## 부분점수 구조

답을 끝까지 못 풀어도 아래를 쓰면 점수를 확보할 수 있습니다.

| 문제 유형 | 반드시 쓸 것 |
|---|---|
| MLE | i.i.d라서 곱, 로그 이유, 미분=0 이유 |
| MAP | posterior ∝ likelihood × prior |
| Gaussian | 대칭성, 부분적분, 가우스 적분 |
| Softmax | $i=j$와 $i\ne j$ 분리 |
| 고유값 | $Av=\lambda v$, $\det(A-\lambda I)=0$ |
| Tent prior | 구간 분할, 경계, 미분 불가능점 |

## 시험 전 10분 루틴

1. MLE 7단계 말하기.
2. softmax 자코비안 한 줄 쓰기.
3. Gaussian 모멘트 표 쓰기.
4. MAP symmetric/asymmetric 표 쓰기.
5. tent prior의 후보 3종류 쓰기.

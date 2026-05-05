---
title: "단계 진단표"
slug: 00-orientation-01-diagnostic
order: 2
---

# 단계 진단표

아래 문제는 점수용이 아니라 현재 위치를 찾기 위한 것입니다. 답은 `90-drills/answer-key.md`에 있습니다.

## 입문 진단

1. $\sum_{i=1}^n y_i=k$를 말로 설명하라.
2. $X\sim N(0,1)$에서 `~`의 뜻을 설명하라.
3. $p(\theta|D)\propto p(D|\theta)p(\theta)$에서 $\propto$의 뜻을 설명하라.
4. $\partial p_i/\partial z_j$는 무엇을 묻는 기호인가?
5. $E[X]=\int xp(x)\,dx$를 "가중평균"이라는 말로 풀어써라.

통과 기준: 5개 중 4개 이상을 식 없이 말로 설명.

## 중급 진단

1. Bernoulli MLE의 우도 $L(\theta)$를 $k=\sum y_i$로 단순화하라.
2. $Uniform[a,b]$의 $E[X]$와 $\operatorname{Var}(X)$를 계산하라.
3. softmax에서 $i=j$일 때 $\partial p_i/\partial z_i$를 구하라.
4. symmetric prior $\theta^m(1-\theta)^m$의 MAP 추정량을 구하라.
5. $A=\begin{pmatrix}0&1\\1&0\end{pmatrix}$의 고유값을 구하라.

통과 기준: 계산 과정을 70% 이상 재현.

## 고급 진단

1. 왜 로그우도 최대화와 우도 최대화가 같은 위치를 주는가?
2. 왜 내부 극값에서 미분값이 0이어야 하는가?
3. 왜 $E[X^3]=0$ for $X\sim N(0,1)$인가?
4. tent prior 문제에서 왜 경계와 미분 불가능점을 따로 검사해야 하는가?
5. KL divergence가 음수가 될 수 없다는 주장을 Jensen 부등식으로 연결하라.

통과 기준: "정리 이름 + 적용 조건 + 결론" 구조로 답변.

## 마스터 진단

1. Bernoulli MLE, Gaussian MSE, Cross Entropy, L2 regularization을 하나의 이야기로 연결하라.
2. $p(\theta)\propto \theta^a(1-\theta)^b$일 때 MAP을 일반화하라.
3. softmax 자코비안의 각 행 합이 0인 이유와 의미를 설명하라.
4. $A$가 대칭행렬이면 왜 직교 고유벡터를 기대할 수 있는가?
5. 시험장에서 시간이 3분 남았고 MAP 문제를 못 풀었을 때 부분점수를 확보하는 답안을 써라.

통과 기준: 새로운 문제를 만들어도 같은 엔진으로 처리.

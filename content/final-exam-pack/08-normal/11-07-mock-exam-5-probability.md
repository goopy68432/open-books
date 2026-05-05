---
title: "딥러닝 이론 모의고사 #5 — 확률론 집중"
slug: 07-mock-exam-5-probability
order: 11
---

# 딥러닝 이론 모의고사 #5 — 확률론 집중

> 배점 100점 / 10문제

---

## 문제 1. [10점] 확률 기초 용어

**(a)** [5점] 실험(Experiment), 표본공간(Sample Space), 이벤트(Event), 확률변수(Random Variable)를 각각 정의하고, "주사위 2개 던지기"를 예시로 설명하시오.

**(b)** [5점] 라플라스 확률 정의 $P(E) = |E|/|S|$의 적용 조건(유한, 등확률)을 명시하고, 이 조건이 성립하지 않는 경우의 예를 드시오.

---

## 문제 2. [10점] 조건부 확률과 독립

**(a)** [4점] 조건부 확률 $P(E|F) = P(E \cap F)/P(F)$를 유도하고, $P(F) = 0$일 때의 문제를 언급하시오.

**(b)** [6점] 독립 $P(E \cap F) = P(E)P(F)$가 MLE 유도의 출발점인 이유를 설명하시오. 만약 데이터가 독립이 아니라면 어떤 문제가 생기는가?

---

## 문제 3. [10점] 베르누이 분포

**(a)** [4점] 베르누이 분포 $\text{Ber}(x;\theta) = \theta^x(1-\theta)^{1-x}$의 의미를 설명하고, $x=0$과 $x=1$일 때의 값을 각각 확인하시오.

**(b)** [6점] $n$번의 독립 베르누이 시행에서 $k$번 성공 시 MLE를 유도하시오. 각 단계의 "왜"를 명시하시오.

---

## 문제 4. [10점] 가우시안 분포

**(a)** [4점] 가우시안 분포 $\mathcal{N}(x;\mu,\sigma^2)$의 PDF를 쓰고, "Exponential + Minus + Square"라는 구조가 왜 평균에서 멀어질수록 확률이 감소하는지 설명하시오.

**(b)** [6점] i.i.d. 가우시안 샘플 $\{x_1,...,x_n\}$에서 $\mu$와 $\sigma^2$의 MLE를 각각 유도하시오.

---

## 문제 5. [10점] 중심극한정리 (CLT)

**(a)** [4점] CLT를 수식으로 서술하고, 필요한 조건 3가지를 명시하시오.

**(b)** [3점] CLT가 "가우시안 노이즈 가정"을 정당화하는 논리를 2줄로 설명하시오.

**(c)** [3점] CLT와 "최대 엔트로피" 논거는 모두 가우시안을 지지한다. 두 논거의 관점 차이를 비교하시오.

---

## 문제 6. [10점] 기댓값과 분산

**(a)** [4점] $\mathbb{E}[X]$와 $\text{Var}[X]$의 정의를 쓰고, $\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$를 유도하시오.

**(b)** [3점] 독립 → 비상관이지만 역은 일반적으로 거짓임을 예시로 보이시오.

**(c)** [3점] $X \sim \text{Ber}(\theta)$의 기댓값과 분산을 구하시오.

---

## 문제 7. [10점] 결합확률과 주변화

**(a)** [5점] 결합확률 $P(X,Y)$, 조건부 $P(X|Y)$, 주변확률 $P(X) = \sum_Y P(X,Y)$의 관계를 서술하시오.

**(b)** [5점] 전체 확률 법칙(Law of Total Probability) $P(A) = \sum_i P(A|B_i)P(B_i)$를 유도하고, 이것이 베이즈 정리의 분모 $P(E)$ 계산에 어떻게 사용되는지 보이시오.

---

## 문제 8. [10점] 확률분포와 AI

**(a)** [4점] Generative Model과 Discriminative Model의 차이를 수식으로 설명하시오.

**(b)** [3점] 이미지 분류에서 $P(y|x)$를 모델링하는 것은 어느 쪽인가? 왜?

**(c)** [3점] LLM(Large Language Model)에서 다음 단어 예측 $P(w_{t+1}|w_1,...,w_t)$는 어떤 확률적 의미를 가지는가?

---

## 문제 9. [10점] 확률변수의 변환

**(a)** [5점] $X \sim \mathcal{N}(0,1)$이고 $Y = aX + b$일 때, $Y$의 분포를 구하시오.

**(b)** [5점] Softmax 함수가 실수 벡터를 확률 분포(심플렉스)로 변환하는 이유를 3가지 조건(비음수, 합=1, 단조성)으로 설명하시오.

---

## 문제 10. [10점] 확률과 Loss 함수의 통합적 이해

다음 빈칸을 채우고 각각을 1줄로 설명하시오.

| 문제 유형 | 확률 가정 | 결과 Loss |
|---------|---------|---------|
| 회귀 | (a) | (b) |
| 이진 분류 | (c) | (d) |
| 다중 분류 | (e) | (f) |

**(g)** [4점] "Loss 함수의 선택은 확률 가정의 선택이다"라는 명제를 설명하시오.

---
---

# 모범답안

## 답 3.
### (b) 베르누이 MLE
$n$번 시행, $k$번 성공: $P(D|\theta) = \prod_{i=1}^n \theta^{x_i}(1-\theta)^{1-x_i} = \theta^k(1-\theta)^{n-k}$
(왜 곱: i.i.d. → 결합확률 = 개별확률의 곱)

$\log P = k\log\theta + (n-k)\log(1-\theta)$
(왜 log: 곱→합, 수치안정, 단조)

$\frac{\partial}{\partial\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$
(왜 미분=0: Fermat, 내부 극값 필요조건)

$k(1-\theta) = (n-k)\theta$ → $k = n\theta$ → $\boxed{\theta_{ML} = k/n}$ $\square$

## 답 4.
### (b) 가우시안 MLE
$\log P = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2$

$\mu$에 대해: $\partial/\partial\mu = \frac{1}{\sigma^2}\sum(x_i-\mu) = 0$ → $\boxed{\mu_{ML} = \frac{1}{n}\sum x_i}$

$\sigma^2$에 대해: $\partial/\partial\sigma^2 = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i-\mu)^2 = 0$
→ $\boxed{\sigma^2_{ML} = \frac{1}{n}\sum(x_i-\mu_{ML})^2}$

(참고: MLE의 분산 추정은 $n$으로 나누므로 편향(biased). 비편향은 $n-1$)

## 답 6.
### (b)
$X$: $P(X=1)=P(X=-1)=1/2$, $Y = X^2$. $E[XY] = E[X^3] = E[X] = 0 = E[X]E[Y]$ → 비상관.
그러나 $P(Y=1|X=1) = 1 \neq P(Y=1) = 1$ → **독립이 아님**. ($Y$가 $X$에 의해 완전 결정)

### (c)
$E[X] = 0 \cdot (1-\theta) + 1 \cdot \theta = \theta$
$E[X^2] = 0^2(1-\theta) + 1^2\theta = \theta$
$\text{Var}[X] = \theta - \theta^2 = \theta(1-\theta)$

## 답 10.
| 문제 유형 | 확률 가정 | 결과 Loss |
|---------|---------|---------|
| 회귀 | Gaussian $\mathcal{N}(h(x),\sigma^2)$ | MSE |
| 이진 분류 | Bernoulli $\text{Ber}(\sigma(w^\top x))$ | Binary CE |
| 다중 분류 | Categorical $\text{Cat}(h(x))$ | Cross-Entropy |

(g) 모든 Loss는 NLL = $-\log P(\text{data}|\theta)$의 특수한 경우. 확률 가정을 바꾸면 Loss가 바뀐다. MSE를 쓰는 것은 암묵적으로 가우시안 노이즈를 가정하는 것이고, CE를 쓰는 것은 카테고리 분포를 가정하는 것이다.

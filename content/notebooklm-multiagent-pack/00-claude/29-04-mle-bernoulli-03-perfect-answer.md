---
title: "03. 완벽 답안 — 시험장 그대로"
slug: 04-mle-bernoulli-03-perfect-answer
order: 29
---

# 03. 완벽 답안 — 시험장 그대로

---

### [문제] $y_i \sim$ Bern(θ) i.i.d (i=1..n), $k = \sum y_i$일 때 우도, NLL, MLE를 구하라.

### [풀이]

**(1) 모델**

$y_i \sim$ Bern(θ) i.i.d, $i = 1, \ldots, n$. 단일 pmf:
$$p(y_i | \theta) = \theta^{y_i}(1-\theta)^{1-y_i}, \quad y_i \in \{0, 1\}.$$

**(2) 우도함수**

i.i.d 가정에 의해 결합확률은 각 확률의 곱이다:
$$L(\theta) = \prod_{i=1}^n p(y_i | \theta) = \prod_{i=1}^n \theta^{y_i}(1-\theta)^{1-y_i}.$$

지수의 합 $\sum y_i = k$, $\sum (1-y_i) = n-k$로 단순화:
$$\boxed{L(\theta) = \theta^k (1-\theta)^{n-k}.}$$

**(3) 로그우도와 NLL**

곱은 미분이 까다롭고 수치 underflow가 발생할 수 있다. 또 $\log$는 단조증가 함수이므로 $\arg\max L = \arg\max \log L$이다. 따라서 로그를 취한다:
$$\ell(\theta) = \log L(\theta) = k \log \theta + (n-k) \log(1-\theta).$$

NLL은 ℓ에 음수를 붙인 것:
$$\boxed{\text{NLL}(\theta) = -k\log\theta - (n-k)\log(1-\theta).}$$

NLL 최소화 ⇔ ℓ 최대화 ⇔ L 최대화.

**(4) MLE**

$\theta \in (0,1)$에서 ℓ은 미분 가능하므로 페르마 정리에 의해 1차 도함수 = 0인 점이 극값 후보:
$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0.$$

정리:
$$k(1-\theta) = (n-k)\theta \Rightarrow k = n\theta \Rightarrow \theta = \frac{k}{n}.$$

**최댓값 검증:** $\dfrac{d^2\ell}{d\theta^2} = -\dfrac{k}{\theta^2} - \dfrac{n-k}{(1-\theta)^2} < 0$ → ℓ은 오목하므로 임계점이 최댓값. ∎

$$\boxed{\hat{\theta}_{\text{MLE}} = \frac{k}{n}.}$$

**(5) MLE와의 관계 서술**

MLE는 우도(또는 동등하게 로그우도)를 최대화하는 모수 추정으로, NLL의 최소화와 동치이다. 베르누이에서 $\hat{\theta}_{\text{MLE}} = k/n$은 **관측된 표본 비율**과 일치하며, 이는 MLE가 데이터에 가장 자연스러운 추정을 제공함을 보여준다.

---

## 채점 포인트

| 항목 | 배점 |
|------|------|
| **i.i.d 명시 + 곱 정당화** | 15% |
| 단일 pmf 명시 | 5% |
| 우도함수 단순화 ($\theta^k(1-\theta)^{n-k}$) | 15% |
| **로그 취하는 이유 3개 명시** | **20% ★** |
| NLL 정의 + 의미 한 줄 | 10% |
| **미분=0 + 페르마 정리 명시** | **15% ★** |
| MLE 풀이 ($k/n$) | 10% |
| 최댓값 검증 (2계 미분) | 5% |
| MLE 관계 서술 ($k/n$ = 표본비율) | 5% |

★ 표시는 가장 잘 빼먹는 부분. **반드시** 명시.

---

## 시험장 시간 배분 (10분)

| 시간 | 작업 |
|------|------|
| 0:00 ~ 1:00 | (1) 모델, pmf |
| 1:00 ~ 3:00 | (2) 우도함수, 단순화 |
| 3:00 ~ 5:00 | (3) **로그 + NLL + 3대 이유** ★ |
| 5:00 ~ 7:30 | (4) **미분=0 + 페르마 + 풀이** ★ |
| 7:30 ~ 9:00 | (4) 2계 미분 검증 |
| 9:00 ~ 10:00 | (5) 관계 서술 + 박스 |

---

## 외워야 할 핵심 문장 5개

1. **i.i.d:** "y_i들이 i.i.d 가정 하에 있으므로 결합확률은 각 확률의 곱이다."
2. **로그:** "log는 단조증가이므로 argmax가 보존되며, 곱→합으로 미분이 쉽고 underflow 방지가 가능하다."
3. **NLL:** "NLL = -log L의 최소화는 log L 최대화와 동치이다."
4. **페르마:** "ℓ은 (0,1) 내부에서 미분 가능하므로 페르마 정리에 의해 dℓ/dθ = 0 인 점이 극값 후보다."
5. **검증:** "2계 미분이 음수이므로 ℓ은 오목하고, 임계점은 최댓값이다."

---

## 다음

[`04-mastery-quiz.md`](./04-mastery-quiz.md)

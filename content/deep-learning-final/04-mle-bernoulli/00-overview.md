---
title: "기출 4번 — 베르누이 MLE (Overview)"
slug: overview
order: 0
---

# 기출 4번 — 베르누이 MLE (Overview)

> 이 챕터가 시험의 **심장**입니다. "i.i.d → 로그 → 미분=0" 체인의 정수.

## 문제 원문

$y_i \sim$ Bern(θ) (i = 1, ..., n), $y_i$들이 i.i.d.
$k = \sum_{i=1}^n y_i$ 일 때:
1. **우도함수** $L(\theta)$ 를 구하라.
2. **NLL** (음의 로그우도)을 구하라.
3. **MLE**와의 관계를 서술하라.

---

## 출제 의도

이 문제는 시험에서 **거의 매번** 출제됩니다. 평가 포인트:
1. **i.i.d 가정 명시** — "독립이라서 곱"
2. **로그 취하는 이유 글로** — "곱→합, 단조성, 수치안정"
3. **미분=0 이유 글로** — "페르마 정리"
4. **결과:** $\hat{\theta}_{\text{MLE}} = k/n$
5. **직관:** "표본평균 = 자연스럽다"

채점관이 보고 싶은 것은 **답이 아니라 7단계 체인**.

---

## 5분 핵심

| 단계 | 내용 | 산출 |
|-----|------|-----|
| 1 | 모델 명시 | $y_i \sim$ Bern(θ) i.i.d |
| 2 | 단일 pmf | $p(y\|\theta) = \theta^y(1-\theta)^{1-y}$ |
| 3 | i.i.d → 곱 | $L = \prod p(y_i\|\theta)$ |
| 4 | 단순화 | $L = \theta^k(1-\theta)^{n-k}$ |
| 5 | log → NLL | $\text{NLL} = -k\log\theta - (n-k)\log(1-\theta)$ |
| 6 | 미분=0 | $\frac{d\text{NLL}}{d\theta} = 0$ → $-k/\theta + (n-k)/(1-\theta) = 0$ |
| 7 | 풀이 | $\hat{\theta}_{\text{MLE}} = k/n$ |

---

## 학습 자료

| 파일 | 내용 |
|------|-----|
| [`01-concept.md`](./01-concept.md) | 베르누이, MLE의 철학 |
| [`02-derivation.md`](./02-derivation.md) | **8단계 풀이 + 모든 단계 "왜?"** ★ |
| [`03-perfect-answer.md`](./03-perfect-answer.md) | 시험 답안 |
| [`04-mastery-quiz.md`](./04-mastery-quiz.md) | 변형 문제 (지수, 정규 분포 MLE) |

## 관련 사전지식
- [`../00-prerequisites/03-derivative-101.md`](../00-prerequisites/03-derivative-101.md) — 미분, 페르마
- [`../00-prerequisites/08-exp-log.md`](../00-prerequisites/08-exp-log.md) — 로그 법칙
- [`../00-prerequisites/09-probability.md`](../00-prerequisites/09-probability.md) — 베르누이, i.i.d

## 관련 킬러체인
- [`../09-killer-chains/01-iid-to-product.md`](../09-killer-chains/01-iid-to-product.md)
- [`../09-killer-chains/02-why-log.md`](../09-killer-chains/02-why-log.md)
- [`../09-killer-chains/03-why-derivative-zero.md`](../09-killer-chains/03-why-derivative-zero.md)

---

## 출제 변형

- 지수분포 MLE (rate λ)
- 정규분포 MLE (μ, σ²)
- 푸아송 MLE
- 결국 모두 같은 **7단계 체인** → `10-ten-proofs/10-chain-of-mle.md`

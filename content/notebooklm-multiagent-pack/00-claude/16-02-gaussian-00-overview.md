---
title: "기출 2번 — 정규분포 모멘트 (Overview)"
slug: 02-gaussian-00-overview
order: 16
---

# 기출 2번 — 정규분포 모멘트 (Overview)

## 문제 원문

$X \sim N(0, 1)$일 때 다음을 구하라:
- $E(X)$
- $E(X^2)$
- $E(X^3)$
- $E(X^4)$

(보조정리: $\int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} \exp(-x^2/2)\,dx = \sqrt{2\pi}$를 증명하라)

---

## 출제 의도

1. **가우스 적분** ($\int e^{-x^2/2} = \sqrt{2\pi}$)을 직접 증명할 수 있는가
2. **대칭성**으로 홀수 모멘트가 0임을 즉시 인식하는가
3. **부분적분**으로 짝수 모멘트를 계산할 수 있는가
4. 모든 단계에 "왜?"를 적어 채점관을 설득하는가

---

## 5분 핵심 답

| 모멘트 | 값 | 핵심 논리 |
|-------|-----|---------|
| $E[X]$ | 0 | 기함수 적분 (대칭성) |
| $E[X^2]$ | 1 | 부분적분 + 가우스 적분 |
| $E[X^3]$ | 0 | 기함수 적분 |
| $E[X^4]$ | 3 | 부분적분 두 번 + 가우스 적분 |

**일반 공식:** $E[X^{2n}] = (2n-1)!!$, $E[X^{2n+1}] = 0$.

---

## 학습 자료

| 파일 | 내용 |
|------|-----|
| [`01-concept.md`](./01-concept.md) | 정규분포의 정체, 종 모양 그래프 |
| [`02-derivation.md`](./02-derivation.md) | 가우스 적분 증명 + 4개 모멘트 계산 |
| [`03-perfect-answer.md`](./03-perfect-answer.md) | 시험장 그대로 답안 |
| [`04-mastery-quiz.md`](./04-mastery-quiz.md) | 변형 문제 (E(X⁵), E(X⁶), N(μ,σ²)) |

---

## 관련 사전지식

- [`../00-prerequisites/05-integral-101.md`](../00-prerequisites/05-integral-101.md) — 적분, 부분적분, 기함수 대칭성
- [`../00-prerequisites/08-exp-log.md`](../00-prerequisites/08-exp-log.md) — 지수함수 미분
- [`../00-prerequisites/10-expectation.md`](../00-prerequisites/10-expectation.md) — 기댓값 정의

## 출제 변형 가능성

- $X \sim N(\mu, \sigma^2)$일 때 모멘트 (선형변환으로 환원)
- $E[X^5], E[X^6]$ 등 더 높은 차수
- $E[X^2 + 3X]$ 등 함수형
- 이변량 정규분포의 기댓값

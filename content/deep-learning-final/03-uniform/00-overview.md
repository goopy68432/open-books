---
title: "기출 3번 — 균일분포 (Overview)"
slug: overview
order: 0
---

# 기출 3번 — 균일분포 (Overview)

## 문제 원문

$X \sim$ Uniform[a, b]일 때 $E(X)$와 $\text{Var}(X)$를 구하라.

---

## 출제 의도

1. **연속 확률변수의 기댓값·분산**을 적분으로 정의하고 계산할 수 있는가
2. $\text{Var}[X] = E[X^2] - (E[X])^2$ 공식을 활용할 수 있는가
3. 정의역 $[a,b]$ 적분과 인수분해 능력

비교적 **쉬운 문제**. 점수 확보용.

---

## 5분 핵심 답

| 양 | 값 |
|----|-----|
| pdf | $p(x) = 1/(b-a)$, $x \in [a,b]$ |
| $E[X]$ | $(a+b)/2$ |
| $E[X^2]$ | $(a^2 + ab + b^2)/3$ |
| $\text{Var}[X]$ | $(b-a)^2 / 12$ |

---

## 학습 자료

| 파일 | 내용 |
|------|-----|
| [`01-concept.md`](./01-concept.md) | 균일분포의 정체, 직사각형 pdf |
| [`02-derivation.md`](./02-derivation.md) | E(X), E(X²), Var(X) 단계별 |
| [`03-perfect-answer.md`](./03-perfect-answer.md) | 시험 답안 |
| [`04-mastery-quiz.md`](./04-mastery-quiz.md) | 변형 문제 |

## 관련 사전지식
- [`../00-prerequisites/05-integral-101.md`](../00-prerequisites/05-integral-101.md)
- [`../00-prerequisites/10-expectation.md`](../00-prerequisites/10-expectation.md)

## 변형 출제 가능성
- $X \sim$ Uniform[0, 1]: $E = 1/2$, Var = 1/12 (외울 것)
- $E[X^3]$, $E[X^4]$ 등 고차 모멘트
- $E[g(X)]$ 형태 (예: $E[\sin X]$)

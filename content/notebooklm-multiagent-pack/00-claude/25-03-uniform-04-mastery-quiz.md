---
title: "04. 마스터리 퀴즈 — 균일분포"
slug: 03-uniform-04-mastery-quiz
order: 25
---

# 04. 마스터리 퀴즈 — 균일분포

---

## 문제 1: $X \sim$ Uniform[0, 1]의 $E[X^3]$

<details><summary>풀이</summary>

$$E[X^3] = \int_0^1 x^3 \cdot 1\,dx = \frac{x^4}{4}\Big|_0^1 = \frac{1}{4}.$$
</details>

---

## 문제 2: $X \sim$ Uniform[2, 6]의 평균과 분산

<details><summary>풀이</summary>

- $E[X] = (2+6)/2 = 4$
- $\text{Var}[X] = (6-2)^2/12 = 16/12 = 4/3$
</details>

---

## 문제 3: $X \sim$ Uniform[a, b]의 $E[X^n]$ 일반 공식

<details><summary>풀이</summary>

$$E[X^n] = \frac{1}{b-a}\int_a^b x^n\,dx = \frac{b^{n+1} - a^{n+1}}{(n+1)(b-a)}.$$
</details>

---

## 문제 4: $X \sim$ Uniform[-1, 1]의 $E[X^2]$

<details><summary>풀이</summary>

$E[X^2] = \frac{a^2 + ab + b^2}{3} = \frac{1 - 1 + 1}{3} = \frac{1}{3}$.

또는 $\text{Var}[X] = (b-a)^2/12 = 4/12 = 1/3$, $E[X] = 0$ → $E[X^2] = \text{Var} = 1/3$.
</details>

---

## 문제 5: $X \sim$ Uniform[0, 1]일 때 $Y = -\log X$의 분포는?

<details><summary>풀이</summary>

$P(Y \leq y) = P(-\log X \leq y) = P(X \geq e^{-y}) = 1 - e^{-y}$ for $y \geq 0$.

→ $Y \sim$ Exp(1) (지수분포). pdf $p_Y(y) = e^{-y}$.

(이 변환은 시뮬레이션에서 지수난수 생성에 사용됨)
</details>

---

## 자가 평가

| 점수 | 평가 |
|-----|-----|
| 5/5 | 마스터 |
| 4/5 | 합격 |
| ≤ 3 | `02-derivation.md` 복습 |

---

## 시험 직전 체크
- [ ] pdf $1/(b-a)$ 외웠는가?
- [ ] $E[X] = (a+b)/2$ 즉답?
- [ ] $\text{Var}[X] = (b-a)^2/12$ 즉답?
- [ ] $b^3 - a^3 = (b-a)(a^2+ab+b^2)$ 인수분해 외웠는가?

---

[`../04-mle-bernoulli/00-overview.md`](../04-mle-bernoulli/00-overview.md)

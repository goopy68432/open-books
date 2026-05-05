---
title: "04. 마스터리 퀴즈 — 정규분포 모멘트"
slug: 02-gaussian-04-mastery-quiz
order: 20
---

# 04. 마스터리 퀴즈 — 정규분포 모멘트

---

## 문제 1: $X \sim N(0,1)$의 $E[X^5]$, $E[X^6]$

<details><summary>풀이</summary>

- $E[X^5]$: 기함수 → 0
- $E[X^6] = 5 \cdot E[X^4] = 5 \cdot 3 = 15$ (재귀 $E[X^k] = (k-1)E[X^{k-2}]$)

또는 $E[X^{2n}] = (2n-1)!!$ → $E[X^6] = 5!! = 5 \cdot 3 \cdot 1 = 15$.
</details>

---

## 문제 2: $X \sim N(\mu, \sigma^2)$의 $E[X], E[X^2], \text{Var}[X]$

<details><summary>풀이</summary>

$Z = (X - \mu)/\sigma \sim N(0,1)$로 표준화. 즉 $X = \mu + \sigma Z$.

- $E[X] = \mu + \sigma E[Z] = \mu$
- $E[X^2] = E[(\mu + \sigma Z)^2] = \mu^2 + 2\mu\sigma E[Z] + \sigma^2 E[Z^2] = \mu^2 + \sigma^2$
- $\text{Var}[X] = E[X^2] - \mu^2 = \sigma^2$

**시험 답안에:** "$Z = (X-\mu)/\sigma$로 표준화하여 N(0,1)의 결과를 활용한다."
</details>

---

## 문제 3: $X \sim N(0,1)$일 때 $E[X^2 + 3X + 1]$

<details><summary>풀이</summary>

기댓값의 선형성:
$$E[X^2 + 3X + 1] = E[X^2] + 3E[X] + 1 = 1 + 0 + 1 = 2.$$
</details>

---

## 문제 4: 가우스 적분 변형 $\int_{-\infty}^\infty e^{-ax^2}\,dx$ ($a > 0$)

<details><summary>풀이</summary>

치환 $u = x\sqrt{2a}$, $du = \sqrt{2a}\,dx$:
$$\int_{-\infty}^\infty e^{-ax^2}\,dx = \frac{1}{\sqrt{2a}} \int_{-\infty}^\infty e^{-u^2/2}\,du = \frac{1}{\sqrt{2a}} \cdot \sqrt{2\pi} = \sqrt{\pi/a}.$$
</details>

---

## 문제 5: 다변량 정규분포 적분 (어려움)

$\int_{-\infty}^\infty\int_{-\infty}^\infty e^{-(x^2 + y^2)/2}\,dx\,dy$를 구하라.

<details><summary>풀이</summary>

가우스 적분 증명에서 사용한 결과 그대로:
$$= \left(\int e^{-x^2/2}\,dx\right)^2 = (\sqrt{2\pi})^2 = 2\pi.$$
</details>

---

## 자가 평가

| 점수 | 평가 |
|-----|-----|
| 5/5 | 마스터 |
| 4/5 | 합격 |
| 3/5 | `02-derivation.md` 복습 |
| 0~2 | `00-prerequisites/05-integral-101.md` (부분적분, 대칭성)부터 다시 |

---

## 시험 직전 체크

기출 2번:
- [ ] 가우스 적분 증명 5단계 외웠는가?
- [ ] 대칭성으로 홀수 모멘트 = 0 즉답?
- [ ] $E[X^2]$ 부분적분에서 $u, dv$ 선택 외웠는가?
- [ ] $E[X^4]$ 부분적분 또는 재귀로 둘 중 하나로 풀이 가능?
- [ ] 일반 공식 $(2n-1)!!$ 외웠는가?

---

[`../03-uniform/00-overview.md`](../03-uniform/00-overview.md)로 이동.

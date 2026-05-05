---
title: "05. 가우스 적분"
slug: gaussian-integral
order: 5
---

# 05. 가우스 적분

## 정리

$$\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$$

## 증명

자세히는 [`../02-gaussian/02-derivation.md`](../02-gaussian/02-derivation.md). 5단계 요약:

1. $I = \int e^{-x^2/2}\,dx$로 두기.
2. $I^2 = \iint e^{-(x^2+y^2)/2}\,dx\,dy$ (변수 분리).
3. 극좌표 $(x,y) = (r\cos\phi, r\sin\phi)$, $dx\,dy = r\,dr\,d\phi$.
4. $I^2 = \int_0^{2\pi}\int_0^\infty e^{-r^2/2} r\,dr\,d\phi = 2\pi \cdot 1 = 2\pi$.
5. $I = \sqrt{2\pi}$. ∎

## 변형

- $\int e^{-ax^2}\,dx = \sqrt{\pi/a}$
- $\int_0^\infty e^{-x^2/2}\,dx = \sqrt{\pi/2}$
- $\int e^{-(x-\mu)^2/(2\sigma^2)}\,dx = \sigma\sqrt{2\pi}$

## 시험 답안용

> "가우스 적분 $\int e^{-x^2/2} = \sqrt{2\pi}$는 $I^2$의 극좌표 변환으로 증명."

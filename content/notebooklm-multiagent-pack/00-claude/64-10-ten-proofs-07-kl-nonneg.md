---
title: "07. KL Divergence ≥ 0"
slug: 10-ten-proofs-07-kl-nonneg
order: 64
---

# 07. KL Divergence ≥ 0

## 정의

확률분포 p, q에 대해:
$$\text{KL}(p \| q) = E_p\left[\log \frac{p(X)}{q(X)}\right] = \int p(x) \log\frac{p(x)}{q(x)}\,dx$$

## 정리

$$\text{KL}(p \| q) \geq 0$$

등호 ⇔ p = q a.e.

## 증명 (Jensen 사용)

$$-\text{KL}(p\|q) = E_p\left[\log\frac{q(X)}{p(X)}\right]$$

$\log$는 오목 → Jensen ($f$ 오목이면 $f(E[X]) \geq E[f(X)]$):
$$E_p[\log(q/p)] \leq \log E_p[q/p]$$

계산:
$$E_p[q/p] = \int p \cdot \frac{q}{p}\,dx = \int q\,dx = 1$$

따라서:
$$-\text{KL}(p\|q) \leq \log 1 = 0 \Rightarrow \text{KL}(p\|q) \geq 0. \quad \blacksquare$$

## 응용

- VAE의 변분 추론
- Cross Entropy = $H(p) + \text{KL}(p\|q)$ 분해

## 시험 답안용

> "KL(p||q) ≥ 0은 Jensen으로 증명. 등호 ⇔ p = q."

---
title: "06. MAP → L2 정규화 — Gaussian prior의 마법"
slug: map-to-l2
order: 6
---

# 06. MAP → L2 정규화 — Gaussian prior의 마법

## 1. 가중치에 prior 부여

신경망 가중치 $\mathbf{w} \in \mathbb{R}^d$에 대한 prior:
$$\mathbf{w} \sim N(\mathbf{0}, \tau^2 I)$$

→ pdf:
$$p(\mathbf{w}) = \frac{1}{(2\pi\tau^2)^{d/2}} \exp\left(-\frac{\|\mathbf{w}\|^2}{2\tau^2}\right)$$

**의미:** 가중치가 0 근처에 있다는 사전 믿음.

## 2. MAP 목적 함수

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max p(\mathbf{w}|D) = \arg\max p(D|\mathbf{w}) p(\mathbf{w})$$

음의 로그:
$$\arg\min_\mathbf{w} \left[-\log p(D|\mathbf{w}) - \log p(\mathbf{w})\right]$$

$= \arg\min$ (NLL + (-log prior))

## 3. -log prior 계산

$$-\log p(\mathbf{w}) = \frac{d}{2}\log(2\pi\tau^2) + \frac{\|\mathbf{w}\|^2}{2\tau^2}$$

상수 무시 → $\|\mathbf{w}\|^2/(2\tau^2)$.

## 4. 통합 손실

가우스 잡음 회귀 + 가우스 prior:
$$\text{Loss}(\mathbf{w}) = \frac{1}{2\sigma^2}\sum_i (y_i - f_i)^2 + \frac{1}{2\tau^2}\|\mathbf{w}\|^2$$

$\lambda = \sigma^2/\tau^2$로 두면:
$$\text{Loss}(\mathbf{w}) \propto \sum_i (y_i - f_i)^2 + \lambda \|\mathbf{w}\|^2$$

**이게 L2 정규화 손실** (Ridge regression).

## 5. 베이지안 해석 표

| Prior | 정규화 |
|-------|-------|
| Gaussian $N(0, \tau^2)$ | L2: $\lambda \|\mathbf{w}\|^2$ |
| Laplace $\text{Lap}(0, b)$ | L1: $\lambda \|\mathbf{w}\|_1$ |
| Uniform | 정규화 없음 (= MLE) |
| Spike-and-slab | 희소성 (sparsity) |

## 6. 시험 답안 표준 문장

> "신경망 가중치에 가우스 prior $N(0, \tau^2 I)$를 부여하면 -log prior가 $\|\mathbf{w}\|^2/(2\tau^2)$이 된다. 따라서 MAP의 목적함수는 'NLL + L2 정규화' 형태가 되며, 이는 L2 정규화의 베이지안 해석이다."

## 7. 한 줄 요약

> "가우스 prior + MLE 손실 → L2 정규화. 정규화는 prior를 가지는 것과 같다."

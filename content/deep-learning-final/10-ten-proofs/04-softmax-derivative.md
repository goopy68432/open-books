---
title: "04. Softmax 자코비안"
slug: softmax-derivative
order: 4
---

# 04. Softmax 자코비안

## 정리

$\mathbf{p} = \text{softmax}(\mathbf{z})$, $p_i = e^{z_i}/\sum_j e^{z_j}$일 때:
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

또는 행렬 형태:
$$J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T$$

## 증명

자세히는 [`../08-softmax/02-derivation.md`](../08-softmax/02-derivation.md). 핵심 요약:

- $S = \sum_k e^{z_k}$
- 케이스 i=j: 몫 규칙 → $p_i(1-p_i)$
- 케이스 i≠j: 몫 규칙 → $-p_i p_j$
- 통합: $p_i(\delta_{ij} - p_j)$

## 응용 — Cross Entropy 그래디언트

$L = -\sum y_i \log p_i$ → $\partial L/\partial z_j = p_j - y_j$.

## 시험 답안용

> "Softmax의 자코비안은 $J = \text{diag}(p) - pp^T$. CE와 합성하면 $\partial L/\partial z_j = p_j - y_j$."

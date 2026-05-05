---
title: "09. 스펙트럴 정리 (Spectral Theorem)"
slug: spectral-theorem
order: 9
---

# 09. 스펙트럴 정리 (Spectral Theorem)

## 정리

$A \in \mathbb{R}^{n \times n}$이 **대칭** ($A^T = A$)이면:
1. 모든 고유값이 **실수**
2. **직교** 고유벡터 기저가 존재
3. $A = Q \Lambda Q^T$ 직교 대각화 가능 (Q 직교, Λ 대각)

## 직관

대칭행렬은 "일반 행렬보다 훨씬 좋은 구조" — 회전(직교) + 늘림(대각)으로 분해.

## 증명 (1) 고유값 실수

$A\mathbf{v} = \lambda \mathbf{v}$, 켤레: $A\bar{\mathbf{v}} = \bar{\lambda}\bar{\mathbf{v}}$.

$\bar{\mathbf{v}}^T A \mathbf{v} = \lambda \bar{\mathbf{v}}^T \mathbf{v}$.
대칭성: $\bar{\mathbf{v}}^T A \mathbf{v} = (A\bar{\mathbf{v}})^T \mathbf{v} = \bar{\lambda} \bar{\mathbf{v}}^T \mathbf{v}$.

→ $(\lambda - \bar{\lambda})\bar{\mathbf{v}}^T\mathbf{v} = 0$. $\bar{\mathbf{v}}^T\mathbf{v} = \|\mathbf{v}\|^2 > 0$, 따라서 $\lambda = \bar{\lambda}$ → 실수.

## 증명 (2) 다른 고유값 → 직교

$A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$, $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$, $\lambda_1 \neq \lambda_2$.

$\mathbf{v}_2^T A \mathbf{v}_1 = \lambda_1 \mathbf{v}_2^T \mathbf{v}_1$.
대칭성: $\mathbf{v}_2^T A \mathbf{v}_1 = (A\mathbf{v}_2)^T \mathbf{v}_1 = \lambda_2 \mathbf{v}_2^T \mathbf{v}_1$.

→ $(\lambda_1 - \lambda_2) \mathbf{v}_2^T \mathbf{v}_1 = 0 \Rightarrow \mathbf{v}_1 \perp \mathbf{v}_2$.

## 응용

- 기출 1번 A=[[0,1],[1,0]] 대칭 → $\mathbf{v}_1 \perp \mathbf{v}_2$ ($\frac{1}{\sqrt{2}}(1,1) \cdot \frac{1}{\sqrt{2}}(1,-1) = 0$) ✓
- PCA: 공분산 행렬은 대칭 → 직교 고유벡터로 차원 축소.

## 시험 답안용

> "대칭행렬은 실수 고유값과 직교 고유벡터 기저를 가진다. (스펙트럴 정리)"

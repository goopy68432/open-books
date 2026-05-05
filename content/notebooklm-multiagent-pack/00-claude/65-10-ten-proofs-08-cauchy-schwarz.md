---
title: "08. Cauchy-Schwarz 부등식"
slug: 10-ten-proofs-08-cauchy-schwarz
order: 65
---

# 08. Cauchy-Schwarz 부등식

## 정리

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$$

등호 ⇔ u, v 평행.

## 증명 (간단)

$t \in \mathbb{R}$에 대해 $\|t\mathbf{u} - \mathbf{v}\|^2 \geq 0$:
$$t^2 \|\mathbf{u}\|^2 - 2t \langle \mathbf{u}, \mathbf{v}\rangle + \|\mathbf{v}\|^2 \geq 0$$

t에 관한 2차식이 항상 ≥ 0 → 판별식 ≤ 0:
$$4\langle \mathbf{u}, \mathbf{v}\rangle^2 - 4\|\mathbf{u}\|^2 \|\mathbf{v}\|^2 \leq 0$$

$$|\langle \mathbf{u}, \mathbf{v}\rangle| \leq \|\mathbf{u}\| \|\mathbf{v}\|. \quad \blacksquare$$

## 응용

- 코사인 정의: $\cos\theta = \langle u, v\rangle / (\|u\|\|v\|)$, $|\cos| \leq 1$ 보장.
- 분산-공분산 부등식: $|\text{Cov}(X,Y)| \leq \sigma_X \sigma_Y$.

## 시험 답안용

> "Cauchy-Schwarz: $|\langle u,v \rangle| \leq \|u\|\|v\|$. 판별식 ≤ 0 으로 증명."

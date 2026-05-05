---
title: "01. Rank-Nullity 정리"
slug: 10-ten-proofs-01-rank-nullity
order: 58
---

# 01. Rank-Nullity 정리

## 정리

선형사상 $T: V \to W$ (또는 행렬 $A: \mathbb{R}^n \to \mathbb{R}^m$)에 대해:
$$\dim(\text{Range}(T)) + \dim(\text{Null}(T)) = \dim(V) = n$$

또는: **rank(A) + nullity(A) = n**.

## 직관

- Range(T): T의 출력이 채우는 공간 (이미지)
- Null(T): T가 0으로 보내는 입력들의 공간 (커널)
- 합 = 입력 공간 차원

"입력 차원이 n이면, 그 중 일부는 출력 채우기에 쓰이고(rank), 나머지는 0으로 보내짐(nullity)"

## 증명 스케치

1. Null(T)의 기저 $\{v_1, \ldots, v_k\}$ 선택 (k = nullity).
2. 이를 V의 기저로 확장: $\{v_1, \ldots, v_k, v_{k+1}, \ldots, v_n\}$.
3. $T(v_{k+1}), \ldots, T(v_n)$이 Range(T)의 기저임을 증명:
   - **생성:** 임의 $T(v) = T(\sum a_i v_i) = \sum_{i > k} a_i T(v_i)$ ($i \leq k$는 null).
   - **독립:** $\sum b_i T(v_i) = 0 \Rightarrow T(\sum b_i v_i) = 0 \Rightarrow \sum b_i v_i \in$ Null. 기저 선택으로 $b_i = 0$ ($i > k$).
4. rank = n - k = n - nullity. ∎

## 응용

- A가 정사각 n×n: rank = n ⇔ Null = {0} ⇔ A 가역
- $A\mathbf{x} = \mathbf{0}$의 해 공간 차원 = nullity.

## 시험 답안용 한 줄

> "선형사상 $T: V \to W$에 대해 $\dim$ Range $+ \dim$ Null $= \dim V$ (Rank-Nullity)."

---
title: "01. Softmax — 분류 문제의 출력층"
slug: 08-softmax-01-concept
order: 47
---

# 01. Softmax — 분류 문제의 출력층

## 1. 정의

벡터 $\mathbf{z} = (z_1, z_2, \ldots, z_c) \in \mathbb{R}^c$에 대해:

$$\text{softmax}(\mathbf{z})_i = p_i = \frac{e^{z_i}}{\sum_{j=1}^c e^{z_j}}$$

## 2. 성질

| 성질 | 식 | 이유 |
|------|-----|-----|
| 양수 | $p_i > 0$ | $e^z > 0$ |
| 합 = 1 | $\sum_i p_i = 1$ | 분자 합 = 분모 |
| 단조 | $z_i$ 클수록 $p_i$ 큼 | 지수 단조 |
| 큰 차이 강조 | "soft" max | 가장 큰 z_i에 가장 많은 확률 |

## 3. 직관

- 입력: 임의 실수 벡터 z (점수, logit)
- 출력: 확률분포 p (모두 양수, 합 1)

**예 (c=3):** $\mathbf{z} = (1, 2, 3)$
- $e^1 = 2.72, e^2 = 7.39, e^3 = 20.09$
- 합 ≈ 30.2
- $\mathbf{p} \approx (0.09, 0.24, 0.66)$

## 4. 신경망에서의 위치

분류 문제 출력층:
- 마지막 은닉층 → linear → z (로짓)
- z → softmax → p (예측 확률)
- p와 정답 라벨 비교 → cross-entropy 손실

## 5. 자코비안이란?

벡터 함수 $f: \mathbb{R}^n \to \mathbb{R}^m$의 미분은 **행렬**:
$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

크기: m × n.

이번 문제: $f = $ softmax, $n = m = c$ → c × c 행렬.

## 6. 풀이 전략

1. $p_i$ 식에서 분자 $e^{z_i}$, 분모 $S = \sum_j e^{z_j}$ 식별.
2. **몫의 미분** + **체인 룰** 적용.
3. 미분 변수 $z_j$가 $i = j$인지 $i \neq j$인지로 케이스 나눔.
4. 결과를 크로네커 델타로 통합.
5. 행렬 형태로 정리.

자세히는 [`02-derivation.md`](./02-derivation.md).

## 다음

[`02-derivation.md`](./02-derivation.md)

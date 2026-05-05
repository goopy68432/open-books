---
title: "기출 8번 — Softmax 미분 (∂p/∂z)"
slug: 08-softmax-00-overview
order: 46
---

# 기출 8번 — Softmax 미분 (∂p/∂z)

## 문제 원문

$\mathbf{p} = \text{softmax}(\mathbf{z})$, $\mathbf{p}, \mathbf{z} \in \mathbb{R}^c$일 때 $\partial \mathbf{p}/\partial \mathbf{z}$를 구하라 (c×c 자코비안).

## 출제 의도

1. **softmax 정의** 정확 명시
2. **몫의 미분 규칙** + **체인 룰** 정확히 적용
3. i=j (대각) vs i≠j (비대각) **두 케이스 분리**
4. 크로네커 델타로 **통합** 표기
5. 자코비안 **행렬 형태** $J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T$로 정리

## 5분 핵심

| 케이스 | $\partial p_i/\partial z_j$ |
|-------|---------------------------|
| i = j | $p_i(1 - p_i)$ |
| i ≠ j | $-p_i p_j$ |
| 통합 (델타) | $p_i(\delta_{ij} - p_j)$ |
| 행렬 형태 | $J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T$ |

## 학습 자료
- [`01-concept.md`](./01-concept.md)
- [`02-derivation.md`](./02-derivation.md) ★
- [`03-perfect-answer.md`](./03-perfect-answer.md)
- [`04-mastery-quiz.md`](./04-mastery-quiz.md)

## 관련
- [`../00-prerequisites/03-derivative-101.md`](../00-prerequisites/03-derivative-101.md) — 몫 규칙
- [`../00-prerequisites/04-chain-rule.md`](../00-prerequisites/04-chain-rule.md)
- [`../09-killer-chains/05-bernoulli-to-ce.md`](../09-killer-chains/05-bernoulli-to-ce.md) — softmax + CE 합성

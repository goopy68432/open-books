---
title: "02. 서로 다른 고유값 → 고유벡터 선형독립"
slug: 10-ten-proofs-02-eigen-independence
order: 59
---

# 02. 서로 다른 고유값 → 고유벡터 선형독립

## 정리

$\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$가 행렬 $A$의 **서로 다른** 고유값 $\lambda_1, \ldots, \lambda_k$에 대응하는 고유벡터들이면, **선형독립**이다.

## 증명 (귀납법)

### k = 1: 자명 ($\mathbf{v}_1 \neq 0$)

### k = 2: 모순법

$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 = \mathbf{0}$ 가정 ($c_1, c_2$ 둘 다 0이 아니라고).

WLOG $c_2 \neq 0$. 그러면 $\mathbf{v}_2 = -(c_1/c_2)\mathbf{v}_1 = c\mathbf{v}_1$ 어떤 $c$.

A 곱:
$$A \mathbf{v}_2 = c A \mathbf{v}_1 \Rightarrow \lambda_2 \mathbf{v}_2 = c \lambda_1 \mathbf{v}_1 = \lambda_1 \mathbf{v}_2$$

→ $(\lambda_2 - \lambda_1)\mathbf{v}_2 = \mathbf{0}$. $\mathbf{v}_2 \neq 0$이므로 $\lambda_1 = \lambda_2$ ⊥ 가정. ∎

### 일반 k (귀납):

k-1까지 성립한다고 가정. $\sum_{i=1}^k c_i \mathbf{v}_i = \mathbf{0}$.

A 곱: $\sum c_i \lambda_i \mathbf{v}_i = \mathbf{0}$.

또 원식 × $\lambda_k$: $\sum c_i \lambda_k \mathbf{v}_i = \mathbf{0}$.

차: $\sum_{i=1}^{k-1} c_i (\lambda_i - \lambda_k) \mathbf{v}_i = \mathbf{0}$.

귀납 가정: $c_i (\lambda_i - \lambda_k) = 0$. $\lambda_i \neq \lambda_k$이므로 $c_i = 0$ ($i = 1, \ldots, k-1$).

원식 → $c_k \mathbf{v}_k = 0 \Rightarrow c_k = 0$. ∎

## 응용

- A가 n×n에 n개 서로 다른 고유값 → n개 선형독립 고유벡터 → **A 대각화 가능**.
- 이 정리는 기출 1번 문제의 두 고유벡터가 자동으로 독립임을 보장.

## 시험 답안용 한 줄

> "서로 다른 고유값에 대응하는 고유벡터는 선형독립. (귀납법 + 모순법 증명, 길이 5줄)."

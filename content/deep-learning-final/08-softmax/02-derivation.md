---
title: "02. 단계별 유도 — Softmax 자코비안"
slug: derivation
order: 2
---

# 02. 단계별 유도 — Softmax 자코비안

## 단계 1: 정의 명시

$$p_i = \frac{e^{z_i}}{S}, \quad S = \sum_{j=1}^c e^{z_j}$$

**왜 명시?** 풀이의 출발. $S$를 별도 기호로 두면 식이 깔끔.

---

## 단계 2: 몫 규칙 준비

$\partial p_i/\partial z_j$를 구하려고 하는데, **분자와 분모 모두 $z_j$에 의존**:
- 분자 $e^{z_i}$: $i = j$이면 $\partial/\partial z_j = e^{z_j}$, $i \neq j$이면 0.
- 분모 $S = \sum_k e^{z_k}$: 항상 $\partial S/\partial z_j = e^{z_j}$ (k = j 항만 살아남음).

**몫 규칙:**
$$\frac{d}{dx}\frac{f(x)}{g(x)} = \frac{f' g - f g'}{g^2}$$

---

## 단계 3: 케이스 1 — i = j (대각)

$$\frac{\partial p_i}{\partial z_i} = \frac{(\partial e^{z_i}/\partial z_i) \cdot S - e^{z_i} \cdot (\partial S/\partial z_i)}{S^2}$$

대입:
- $\partial e^{z_i}/\partial z_i = e^{z_i}$
- $\partial S/\partial z_i = e^{z_i}$

$$= \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}(S - e^{z_i})}{S^2}$$

**단순화:** $e^{z_i}/S = p_i$, $(S - e^{z_i})/S = 1 - p_i$.

$$\boxed{\frac{\partial p_i}{\partial z_i} = p_i(1 - p_i)}$$

---

## 단계 4: 케이스 2 — i ≠ j (비대각)

$$\frac{\partial p_i}{\partial z_j} = \frac{(\partial e^{z_i}/\partial z_j) \cdot S - e^{z_i} \cdot (\partial S/\partial z_j)}{S^2}$$

대입:
- $\partial e^{z_i}/\partial z_j = 0$ ($i \neq j$이므로)
- $\partial S/\partial z_j = e^{z_j}$

$$= \frac{0 - e^{z_i} e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -p_i p_j$$

$$\boxed{\frac{\partial p_i}{\partial z_j} = -p_i p_j \quad (i \neq j)}$$

---

## 단계 5: 통합 (크로네커 델타)

두 케이스를 한 식으로:

$$\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)$$

**검증:**
- $i = j$: $\delta_{ii} = 1$, $p_i(1 - p_i)$ ✓
- $i \neq j$: $\delta_{ij} = 0$, $p_i(0 - p_j) = -p_i p_j$ ✓

$$\boxed{\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)}$$

이게 시험 답안의 **압축형**.

---

## 단계 6: 자코비안 행렬 형태

$J_{ij} = \partial p_i/\partial z_j = p_i \delta_{ij} - p_i p_j$

행렬 형태로 분리:
- $p_i \delta_{ij}$: 대각행렬 $\text{diag}(\mathbf{p})$의 (i,j) 원소
- $p_i p_j$: 외적 $\mathbf{pp}^T$의 (i,j) 원소

$$\boxed{J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T}$$

### 구체 예 (c = 3)

$\mathbf{p} = (p_1, p_2, p_3)^T$:

$$\text{diag}(\mathbf{p}) = \begin{pmatrix} p_1 & 0 & 0 \\ 0 & p_2 & 0 \\ 0 & 0 & p_3 \end{pmatrix}$$

$$\mathbf{pp}^T = \begin{pmatrix} p_1^2 & p_1 p_2 & p_1 p_3 \\ p_2 p_1 & p_2^2 & p_2 p_3 \\ p_3 p_1 & p_3 p_2 & p_3^2 \end{pmatrix}$$

$$J = \begin{pmatrix} p_1(1-p_1) & -p_1 p_2 & -p_1 p_3 \\ -p_2 p_1 & p_2(1-p_2) & -p_2 p_3 \\ -p_3 p_1 & -p_3 p_2 & p_3(1-p_3) \end{pmatrix}$$

---

## 단계 7: 검증 — J의 행 합

각 행의 합 (확률 보존):
$$\sum_j J_{ij} = \sum_j p_i(\delta_{ij} - p_j) = p_i \cdot 1 - p_i \sum_j p_j = p_i - p_i \cdot 1 = 0$$

**의미:** $\sum_j p_j = 1$이라는 제약이 미분 후에도 보존됨 (행 합 0).

---

## 결론

| 표현 | 식 |
|------|-----|
| 원소별 (대각) | $\partial p_i/\partial z_i = p_i(1-p_i)$ |
| 원소별 (비대각) | $\partial p_i/\partial z_j = -p_i p_j$ ($i \neq j$) |
| **통합** | $\partial p_i/\partial z_j = p_i(\delta_{ij} - p_j)$ |
| **행렬** | $J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T$ |

---

## 응용: Softmax + Cross Entropy 합성

분류 문제에서는 softmax 출력 + CE 손실을 함께 미분.

$L = -\sum_i y_i \log p_i$ (one-hot 정답 y)

$\partial L/\partial z_j = ?$

체인 룰:
$$\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j} = \sum_i \left(-\frac{y_i}{p_i}\right) p_i(\delta_{ij} - p_j)$$

$$= \sum_i (-y_i \delta_{ij} + y_i p_j) = -y_j + p_j \sum_i y_i = p_j - y_j$$

(one-hot이므로 $\sum y_i = 1$)

$$\boxed{\frac{\partial L}{\partial z_j} = p_j - y_j}$$

**놀라운 단순성!** 신경망이 "예측 - 정답"을 그래디언트로 사용하는 이유.

자세히는 [`../09-killer-chains/05-bernoulli-to-ce.md`](../09-killer-chains/05-bernoulli-to-ce.md).

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md)

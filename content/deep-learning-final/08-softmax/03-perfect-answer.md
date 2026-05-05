---
title: "03. 완벽 답안 — Softmax 미분"
slug: perfect-answer
order: 3
---

# 03. 완벽 답안 — Softmax 미분

### [문제] $\mathbf{p} = \text{softmax}(\mathbf{z})$, $\mathbf{p}, \mathbf{z} \in \mathbb{R}^c$일 때 $\partial \mathbf{p}/\partial \mathbf{z}$.

### [풀이]

**(1) 정의**

$$p_i = \frac{e^{z_i}}{S}, \quad S = \sum_{k=1}^c e^{z_k}.$$

**(2) 두 경우로 분리**

분모 $S$의 $z_j$에 대한 미분: $\partial S/\partial z_j = e^{z_j}$ (k=j 항만 살아남음).
분자 $e^{z_i}$의 $z_j$에 대한 미분:
- $i = j$: $\partial e^{z_i}/\partial z_j = e^{z_i}$
- $i \neq j$: 0

**(3) 케이스 i = j (대각)**

몫 규칙:
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S}\left(1 - \frac{e^{z_i}}{S}\right) = p_i(1 - p_i).$$

**(4) 케이스 i ≠ j (비대각)**

$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -p_i p_j.$$

**(5) 크로네커 델타로 통합**

$$\boxed{\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j).}$$

**(6) 행렬 형태**

$\delta_{ij}$ 항은 대각행렬, $p_i p_j$ 항은 외적이므로:

$$\boxed{J = \frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T.}$$

**(7) 검증**

각 행 합 = $\sum_j p_i(\delta_{ij} - p_j) = p_i - p_i \sum_j p_j = p_i - p_i = 0$. 확률 합이 1이라는 제약과 일관. ∎

---

## 채점 포인트

| 항목 | 배점 |
|------|------|
| softmax + S 정의 | 10% |
| 두 경우 분리 인식 | 10% |
| 몫 규칙 적용 (대각) | 20% |
| 비대각 결과 | 15% |
| 크로네커 델타 통합 | 15% |
| 행렬 형태 (diag - ppᵀ) | 20% |
| 검증 (행 합 = 0) | 10% |

**감점 사례:**
- 두 경우 분리 안 하고 바로 답: -25%
- 행렬 형태 안 적음: -20%
- 부호 실수 (i ≠ j에서 $+p_i p_j$): -15%

---

## 시험장 시간 배분 (8분)

| 시간 | 작업 |
|------|------|
| 0:00 ~ 0:30 | (1) 정의 |
| 0:30 ~ 1:00 | (2) 두 경우 분리 명시 |
| 1:00 ~ 3:30 | (3) 대각 몫 규칙 + 풀이 |
| 3:30 ~ 5:00 | (4) 비대각 |
| 5:00 ~ 6:00 | (5) 델타 통합 |
| 6:00 ~ 7:30 | (6) 행렬 형태 + 구체 예 (c=3) |
| 7:30 ~ 8:00 | (7) 검증 |

---

## 다음

[`04-mastery-quiz.md`](./04-mastery-quiz.md)

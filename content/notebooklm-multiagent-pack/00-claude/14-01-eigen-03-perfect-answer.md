---
title: "03. 완벽 답안 — 시험장 그대로 옮길 형식"
slug: 01-eigen-03-perfect-answer
order: 14
---

# 03. 완벽 답안 — 시험장 그대로 옮길 형식

> 이 답안을 외워서 시험장에서 그대로 재현하면 만점입니다.

---

## 답안 (5분 분량)

---

### [문제] $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$의 고유값·고유벡터를 구하고, 고유값 정의로 참임을 증명하라.

### [풀이]

**(1) 고유값 정의 명시**

고유벡터 $\mathbf{v} \neq \mathbf{0}$ 와 고유값 $\lambda$의 정의는

$$A\mathbf{v} = \lambda \mathbf{v}.$$

이를 정리하면 $(A - \lambda I)\mathbf{v} = \mathbf{0}$. 비자명한 해 $\mathbf{v} \neq \mathbf{0}$이 존재하려면 $A - \lambda I$가 정칙이 아니어야 하므로

$$\det(A - \lambda I) = 0.$$

**(2) 특성방정식 계산**

$$A - \lambda I = \begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix}, \quad \det(A - \lambda I) = \lambda^2 - 1.$$

따라서 $\lambda^2 - 1 = 0$, $\lambda = \pm 1$.

**(3) λ = 1 의 고유벡터**

$(A - I)\mathbf{v} = \mathbf{0}$:
$$\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0} \Rightarrow v_1 = v_2.$$

정규화하면
$$\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}.$$

**(4) λ = -1 의 고유벡터**

$(A + I)\mathbf{v} = \mathbf{0}$:
$$\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0} \Rightarrow v_2 = -v_1.$$

정규화하면
$$\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}.$$

**(5) 정의로 검증 (문제가 요구한 부분)**

(i) $\lambda_1 = 1$:
$$A\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = 1 \cdot \mathbf{v}_1.$$

(ii) $\lambda_2 = -1$:
$$A\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix} = -1 \cdot \mathbf{v}_2.$$

두 경우 모두 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$ 성립. ∎

**(6) 직관 (보너스)**

$A$는 두 좌표를 교환하는 행렬이다. 교환에 의해 방향이 보존되는 부분공간은 직선 $y = x$ (고유값 1)와 $y = -x$ (고유값 -1, 부호 반전)이다. 이는 위 계산과 일치하며, $A$가 대칭행렬이므로 두 고유벡터는 직교한다 ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$).

---

### [최종 답]

$$\boxed{\lambda_1 = 1, \quad \mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^T; \qquad \lambda_2 = -1, \quad \mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, -1)^T.}$$

---

## 채점 포인트 분석

| 항목 | 배점 (예상) | 답안에 포함된 위치 |
|-----|-----------|----------------|
| 정의 명시 ($A\mathbf{v} = \lambda\mathbf{v}$) | 15% | (1) |
| 특성방정식 도출 (왜 $\det = 0$?) | 15% | (1) "비자명 해..." |
| 특성방정식 계산 | 15% | (2) |
| 고유값 도출 | 10% | (2) |
| 고유벡터 계산 | 20% | (3), (4) |
| **정의로 검증** | 20% | (5) ★ 문제 명시 |
| 직관/추가 통찰 | 5% (가산점) | (6) |

**감점 사례:**
- "(1) 정의 명시"를 생략하면 -15%
- "정의로 증명"을 단순 계산으로 끝내면 -20% (이 문제 핵심)
- 정규화 안 한 답: 감점 거의 없음 (선택 사항)

---

## 시험장 5분 작성 시뮬레이션

| 시간 | 작업 |
|------|-----|
| 0:00 ~ 0:30 | (1) 정의 + 특성방정식 도출 문장 |
| 0:30 ~ 1:30 | (2) 행렬식 계산, 고유값 |
| 1:30 ~ 3:00 | (3), (4) 고유벡터 |
| 3:00 ~ 4:30 | (5) 검증 (직접 곱셈) |
| 4:30 ~ 5:00 | (6) 직관 한 줄 |

---

## 다음

[`04-mastery-quiz.md`](./04-mastery-quiz.md) — 변형 문제 5개로 자가 검증.

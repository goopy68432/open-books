---
title: '02. 7단계 유도 — 모든 단계의 "왜?"'
slug: derivation
order: 2
---

# 02. 7단계 유도 — 모든 단계의 "왜?"

> 답만 적으면 0점. 각 단계마다 "왜 이 단계를 밟는가"를 글로 적어야 합니다.

---

## 문제

$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

A의 고유값과 고유벡터를 구하고, 정의로 참임을 증명하라.

---

## 단계 1: 고유값 정의 명시

$$A\mathbf{v} = \lambda \mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}$$

**왜 이 단계?** 풀이의 출발점. 정의 없이 시작하면 채점 0점. 답안에 반드시 첫 줄에 적기.

**왜 $\mathbf{v} \neq \mathbf{0}$?** $\mathbf{v} = \mathbf{0}$이면 모든 λ에 대해 등식이 자명하게 성립 → 의미 없음.

---

## 단계 2: 이항해서 동차 선형방정식 만들기

$$A\mathbf{v} - \lambda \mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

**왜 $\lambda I$로 바꾸나?** $\lambda \mathbf{v} = \lambda I \mathbf{v}$이므로 행렬 형태로 통일하기 위해. 단위행렬 $I$가 끼어들면서 행렬-행렬 차로 정리.

---

## 단계 3: 비자명 해 존재 조건 → 행렬식 = 0

$\mathbf{v} \neq \mathbf{0}$인 해가 존재하려면, $A - \lambda I$가 **역행렬을 안 가져야** 한다.

**왜?** 만약 $(A - \lambda I)^{-1}$이 존재하면, 양변에 좌측에서 곱해서:
$$\mathbf{v} = (A - \lambda I)^{-1} \cdot \mathbf{0} = \mathbf{0}$$
강제로 $\mathbf{v} = \mathbf{0}$이 됨. → 모순.

따라서 $A - \lambda I$는 정칙(non-singular)이 아님 ⇔ $\det(A - \lambda I) = 0$.

이 식이 **특성방정식 (characteristic equation)**.

---

## 단계 4: 특성방정식 계산

$$A - \lambda I = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} - \begin{pmatrix} \lambda & 0 \\ 0 & \lambda \end{pmatrix} = \begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix}$$

행렬식 (2×2 공식 $ad - bc$):
$$\det(A - \lambda I) = (-\lambda)(-\lambda) - (1)(1) = \lambda^2 - 1$$

방정식:
$$\lambda^2 - 1 = 0$$
$$(\lambda - 1)(\lambda + 1) = 0$$
$$\lambda = 1, \quad \lambda = -1$$

**왜 두 해?** 2×2 행렬은 (중복 포함) 고유값 2개. 대수학 기본정리.

---

## 단계 5: λ = 1 의 고유벡터

$\lambda = 1$을 $(A - \lambda I)\mathbf{v} = 0$에 대입:
$$\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

방정식:
- $-v_1 + v_2 = 0 \Rightarrow v_2 = v_1$
- $v_1 - v_2 = 0 \Rightarrow v_1 = v_2$ (같은 식)

**해의 일반형:** $\mathbf{v} = t(1, 1)^T$, $t \neq 0$ (어떤 실수든 OK)

**정규화 (선택):** 답안에서는 단위벡터로:
$$\boxed{\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}}$$

**왜 정규화?** 길이 1로 만들면 직교성·정규직교성 비교가 쉽고 표준적인 표기. 정규화 안 해도 (1,1)도 정답.

---

## 단계 6: λ = -1 의 고유벡터

$\lambda = -1$을 대입:
$$\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

방정식:
- $v_1 + v_2 = 0 \Rightarrow v_2 = -v_1$

**해의 일반형:** $\mathbf{v} = t(1, -1)^T$.

**정규화:**
$$\boxed{\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}}$$

---

## 단계 7: 정의로 검증 (문제가 명시한 부분!)

이게 **점수의 절반**. 그냥 계산만 하고 끝내면 안 됨. **$A\mathbf{v} = \lambda \mathbf{v}$**를 직접 보여줘야.

### λ₁ = 1 검증

$$A \mathbf{v}_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0\cdot 1 + 1\cdot 1 \\ 1\cdot 1 + 0\cdot 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

비교:
$$\lambda_1 \mathbf{v}_1 = 1 \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

→ $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ ✓

### λ₂ = -1 검증

$$A \mathbf{v}_2 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix}$$

비교:
$$\lambda_2 \mathbf{v}_2 = -1 \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix}$$

→ $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$ ✓

---

## 보너스: 한 줄 직관 (시험 답안 마무리에 추가)

> "$A$는 좌표 swap 행렬이고, swap에 대해 변하지 않는 방향은 $y = x$ 직선(고유값 1)과 $y = -x$ 직선(고유값 -1)이다. 이는 계산과 일치한다."

이 한 줄 추가하면 **A+** 답안.

---

## 최종 답

| 고유값 | 고유벡터 (정규화) | 정의 검증 |
|-------|-----------------|---------|
| $\lambda_1 = 1$ | $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^T$ | $A\mathbf{v}_1 = (1)\mathbf{v}_1$ ✓ |
| $\lambda_2 = -1$ | $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, -1)^T$ | $A\mathbf{v}_2 = (-1)\mathbf{v}_2$ ✓ |

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md) — 시험 답안지에 그대로 적을 형식.

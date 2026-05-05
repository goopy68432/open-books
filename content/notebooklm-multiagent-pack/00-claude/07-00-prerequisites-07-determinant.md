---
title: "07. 행렬식 — det(A-λI)=0의 비밀"
slug: 00-prerequisites-07-determinant
order: 7
---

# 07. 행렬식 — det(A-λI)=0의 비밀

> 기출 1번에서 고유값을 찾는 결정적 도구.

---

## 1. 왜 배우나

기출 1번의 핵심 절차:
1. 고유값 정의: $A\mathbf{v} = \lambda \mathbf{v}$
2. 이항: $(A - \lambda I)\mathbf{v} = \mathbf{0}$
3. **$\mathbf{v} \neq \mathbf{0}$ 이려면 $\det(A - \lambda I) = 0$**
4. 이 식 풀어서 λ 찾기

**행렬식 모르면 1단계도 못 갑니다.**

---

## 2. 행렬식이란?

행렬에 **숫자 하나**를 대응시키는 규칙.

기호: $\det(A)$ 또는 $|A|$.

### 2×2 행렬식
$$\det \begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

### 3×3 행렬식 (사루스 규칙)

$$\det \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = aei + bfg + cdh - ceg - bdi - afh$$

(시험에서는 2×2가 주로 등장)

---

## 3. 행렬식의 의미

### 기하적 의미

2D: $A$가 만드는 평행사변형 **넓이**
3D: $A$가 만드는 평행육면체 **부피**

| det 값 | 의미 |
|--------|-----|
| 0 | 면적/부피가 0 (납작 — 차원 줄어듦) |
| 양수 | 방향 보존 |
| 음수 | 방향 뒤집힘 |

### 핵심 정리

$$\det(A) = 0 \iff A \text{의 열이 선형종속} \iff A\mathbf{x} = \mathbf{0} \text{이 } \mathbf{x} \neq \mathbf{0} \text{ 해를 가짐}$$

**고유값 정의에 정확히 맞아떨어짐!**

---

## 4. 고유값 방정식 유도 (시험 핵심)

### 출발: 고유값 정의

$$A\mathbf{v} = \lambda \mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}$$

### 이항

$$A\mathbf{v} - \lambda \mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

(λI를 빼는 이유: $\lambda \mathbf{v} = \lambda I \mathbf{v}$이므로)

### 핵심 논리

$\mathbf{v} \neq \mathbf{0}$인 해가 있으려면:
- $A - \lambda I$가 **역행렬을 안 가져야** 함.
- 그렇지 않으면 양변에 $(A-\lambda I)^{-1}$을 곱해서 $\mathbf{v} = \mathbf{0}$이 강제됨.

역행렬 안 가짐 ⇔ $\det(A - \lambda I) = 0$.

### 결론
$$\boxed{\det(A - \lambda I) = 0}$$

이걸 **특성방정식 (characteristic equation)** 이라 부름.

---

## 5. 기출 1번 적용

$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

### 1단계: $A - \lambda I$ 계산
$$A - \lambda I = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} - \begin{pmatrix} \lambda & 0 \\ 0 & \lambda \end{pmatrix} = \begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix}$$

### 2단계: 행렬식 = 0
$$\det \begin{pmatrix} -\lambda & 1 \\ 1 & -\lambda \end{pmatrix} = (-\lambda)(-\lambda) - (1)(1) = \lambda^2 - 1$$

### 3단계: 풀이
$$\lambda^2 - 1 = 0 \Rightarrow \lambda = \pm 1$$

→ 고유값 두 개: $\lambda_1 = 1$, $\lambda_2 = -1$.

### 4단계: 각 고유값에 해당하는 고유벡터

#### λ = 1
$$(A - I)\mathbf{v} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0}$$

→ $-v_1 + v_2 = 0$, 즉 $v_1 = v_2$. 정규화: $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^T$.

#### λ = -1
$$(A + I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \mathbf{0}$$

→ $v_1 + v_2 = 0$, 즉 $v_2 = -v_1$. 정규화: $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, -1)^T$.

### 5단계: 정의로 검증

$$A \mathbf{v}_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = 1 \cdot \mathbf{v}_1 \checkmark$$

$$A \mathbf{v}_2 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} -1 \\ 1 \end{pmatrix} = -1 \cdot \mathbf{v}_2 \checkmark$$

---

## 6. 역행렬 (inverse)

### 정의
$A^{-1}$은 $AA^{-1} = A^{-1}A = I$를 만족.

### 2×2 공식
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \Rightarrow A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

(분모 $ad - bc = \det A$. 이게 0이면 역행렬 없음!)

### 존재 조건
$$A^{-1} \text{ 존재} \iff \det(A) \neq 0$$

---

## 7. 행렬식의 핵심 성질

| 성질 | 식 |
|------|-----|
| 단위행렬 | $\det(I) = 1$ |
| 곱 | $\det(AB) = \det(A)\det(B)$ |
| 전치 | $\det(A^T) = \det(A)$ |
| 역행렬 | $\det(A^{-1}) = 1/\det(A)$ |
| 스칼라배 (n차원) | $\det(cA) = c^n \det(A)$ |

---

## 8. 시험 답안 작성법

### 특성방정식 유도 표준 문장

> "고유값 정의 $A\mathbf{v} = \lambda \mathbf{v}$에서 $(A - \lambda I)\mathbf{v} = \mathbf{0}$. $\mathbf{v} \neq \mathbf{0}$인 해가 존재하려면 $A - \lambda I$가 정칙(역행렬 가짐)이 아니어야 하므로 $\det(A - \lambda I) = 0$."

이 한 줄을 **반드시 답안에 적어야 부분점수 풀로 받습니다.**

---

## 9. 자가 점검

1. $\det \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix}$
2. $\det \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$
3. $A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$의 특성방정식 + 고유값
4. $A = \begin{pmatrix} 4 & 0 \\ 0 & 4 \end{pmatrix}$의 고유값과 고유벡터 (모든 벡터가 고유벡터인 특수 케이스!)

**답:**
1. $6 - 4 = 2$
2. $6$ (대각행렬 → 대각곱)
3. $\det(A-\lambda I) = (2-\lambda)(3-\lambda) = 0$ → $\lambda = 2, 3$
4. $\lambda = 4$ (중근), 모든 0 아닌 벡터가 고유벡터.

---

## 다음 챕터

[`08-exp-log.md`](./08-exp-log.md) — 지수·로그의 비밀.

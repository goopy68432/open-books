---
title: "01. 고유값·고유벡터의 직관"
slug: concept
order: 1
---

# 01. 고유값·고유벡터의 직관

> "행렬을 곱해도 **방향이 안 바뀌는** 벡터가 있을까?"

---

## 1. 행렬 = 변환

행렬 $A$에 벡터 $\mathbf{v}$를 곱하면 새 벡터 $A\mathbf{v}$가 나옵니다. 이건 **변환**입니다.
- 회전, 늘리기, 줄이기, 반사 등.

**예 1: 늘리기**
$$\begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$$
모든 벡터를 2배로.

**예 2: 회전 (90도)**
$$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$
방향 90도 회전.

**예 3: 좌표 바꾸기 (기출 1번!)**
$$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}$$
x, y 좌표를 swap.

---

## 2. 고유벡터 — 변환에도 방향 유지

대부분의 벡터는 행렬을 곱하면 방향이 바뀝니다. 그런데 **어떤 특별한 벡터**는 방향이 그대로!

### 정의

벡터 $\mathbf{v}$가 행렬 $A$의 **고유벡터(eigenvector)** ⇔
$$A\mathbf{v} = \lambda \mathbf{v} \quad \text{어떤 스칼라 } \lambda \text{에 대해, } \mathbf{v} \neq \mathbf{0}$$

여기서 $\lambda$가 **고유값(eigenvalue)**.

### 직관

"$A$를 곱하면 방향은 그대로지만, **길이만** $\lambda$배"

- $\lambda > 0$: 방향 유지, 길이 변화
- $\lambda < 0$: 방향 반대, 길이 \|λ\|배
- $\lambda = 0$: 0벡터로 (특이행렬일 때)

### 왜 $\mathbf{v} \neq \mathbf{0}$ 조건?

$\mathbf{v} = \mathbf{0}$이면 자명하게 $A \cdot \mathbf{0} = \mathbf{0} = \lambda \cdot \mathbf{0}$이 모든 λ에 대해 성립. 의미 없음.

---

## 3. 기출 1번 행렬의 직관

$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

이건 "**좌표를 swap**"하는 변환.

### swap 행렬에 대해 방향이 안 바뀌는 벡터?

#### 후보 1: $(1, 1)$
$$A \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$
swap해도 자기 자신! → **고유값 1**

**기하적 의미:** y=x 직선 위의 벡터는 swap에 영향 없음.

#### 후보 2: $(1, -1)$
$$A \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} -1 \\ 1 \end{pmatrix} = -1 \cdot \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$
swap하면 부호 반대! → **고유값 -1**

**기하적 의미:** y=-x 직선 위의 벡터는 swap하면 반대 방향.

### 그림으로

```
       y
       │
   v₁  •(1,1)        ← 고유값 1, 방향 유지
       ╱
      ╱
─────┼─────── x
    ╱
   ╱
  •(1,-1)  v₂        ← 고유값 -1, 방향 반대
```

---

## 4. 왜 고유값/벡터를 배우는가? (시험 외 응용)

| 응용 | 설명 |
|-----|------|
| **PCA** | 데이터 분산 가장 큰 방향 = 공분산 행렬 최대 고유벡터 |
| **신경망** | 가중치 행렬 분석, 학습률 안정성 |
| **양자역학** | 관측 가능한 양 = 헤르미트 연산자 고유값 |
| **구글 페이지랭크** | 인접행렬 고유벡터 |

이론적 가치를 알면 시험 답안에 한 줄 추가 가능.

---

## 5. 핵심 사실 (시험 답안 인용 가능)

### 사실 1: n×n 행렬은 (중복 포함) 고유값이 n개

이유: 특성다항식 $\det(A - \lambda I)$가 n차 다항식 → 대수학 기본정리로 n개 근.

### 사실 2: 서로 다른 고유값에 대응하는 고유벡터는 선형독립

증명은 [`../10-ten-proofs/02-eigen-independence.md`](../10-ten-proofs/02-eigen-independence.md).

### 사실 3: 대칭행렬은 모두 실수 고유값과 직교 고유벡터를 가진다 (스펙트럴 정리)

기출 1번 A는 대칭이므로 이 정리 적용 가능. 답안에 인용하면 수준 높아 보임.

---

## 6. 한 줄 요약

> "고유벡터는 변환에 의해 방향이 보존되는 특별한 벡터, 고유값은 그때의 길이 변화율."

---

## 다음

[`02-derivation.md`](./02-derivation.md) — 단계별 풀이의 "왜?"

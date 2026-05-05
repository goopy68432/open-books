---
title: "06. 벡터와 행렬 — 다차원의 언어"
slug: vector-matrix
order: 6
---

# 06. 벡터와 행렬 — 다차원의 언어

> 기출 1번(고유값), 기출 8번(softmax 자코비안)의 무대.

---

## 1. 왜 배우나

- 기출 1번: 행렬 A=[[0,1],[1,0]]의 고유값/벡터
- 기출 8번: softmax 자코비안 $J = \text{diag}(p) - pp^T$
- 모든 신경망: 가중치 W는 행렬, 입력 x는 벡터

---

## 2. 벡터 (Vector)

### 정의

수를 줄지어 놓은 것.

$$\mathbf{v} = \begin{pmatrix} 3 \\ 4 \end{pmatrix} \in \mathbb{R}^2$$

(2차원 실수 벡터)

### 표기 관례

- 굵은 글씨: $\mathbf{v}$, $\mathbf{x}$
- 또는 화살표: $\vec{v}$
- 좌표는 첨자: $v_1, v_2, v_3$

### 벡터의 기본 연산

#### ① 덧셈
$$\begin{pmatrix} a_1 \\ a_2 \end{pmatrix} + \begin{pmatrix} b_1 \\ b_2 \end{pmatrix} = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \end{pmatrix}$$

#### ② 스칼라 곱
$$c \begin{pmatrix} a_1 \\ a_2 \end{pmatrix} = \begin{pmatrix} c a_1 \\ c a_2 \end{pmatrix}$$

#### ③ 내적 (dot product)
$$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**기하 의미:** $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$

특히 $\mathbf{a} \cdot \mathbf{b} = 0$ ⇔ 직교 (perpendicular)

#### ④ 길이 (norm)
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\mathbf{v}^T \mathbf{v}}$$

**중1 비유:** 피타고라스 정리. (3,4)의 길이는 5.

---

## 3. 행렬 (Matrix)

### 정의

수를 직사각형으로 배열한 것.

$$A = \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix}$$

크기: 행(row) × 열(column). 위는 2×2 행렬.

### 표기 관례

- 굵은 대문자: $A$, $W$, $\Sigma$
- (i,j) 원소: $a_{ij}$ — i행 j열

---

## 4. 행렬의 기본 연산

### ① 덧셈 (같은 크기끼리)
$$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} + \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix} = \begin{pmatrix} 6 & 8 \\ 10 & 12 \end{pmatrix}$$

### ② 스칼라 곱
$$2 \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} = \begin{pmatrix} 2 & 4 \\ 6 & 8 \end{pmatrix}$$

### ③ 행렬 곱 (가장 중요!)

$A: m \times n$, $B: n \times p$일 때 $AB: m \times p$.

$$(AB)_{ij} = \sum_{k=1}^n a_{ik} b_{kj}$$

**중요:** $A$의 **열의 수** = $B$의 **행의 수** 일 때만 곱셈 가능.

#### 예시 (2×2 × 2×1)
$$\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 0\cdot 3 + 1\cdot 4 \\ 1\cdot 3 + 0\cdot 4 \end{pmatrix} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}$$

(기출 1번 행렬 A는 좌표를 swap 시킨다!)

#### 주의: $AB \neq BA$ (일반적으로)

행렬 곱은 **교환법칙 성립 안 함**.

### ④ 전치 (transpose)
$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}, \quad A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}$$

행과 열을 바꿈. $(A^T)_{ij} = A_{ji}$.

성질: $(AB)^T = B^T A^T$ (순서 뒤집힘!)

---

## 5. 특수 행렬

### 단위행렬 $I$
$$I_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad I_3 = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

곱셈에서 1의 역할: $AI = IA = A$.

### 영행렬 $O$ — 모든 원소 0

### 대각행렬 — 대각선 외 0
$$\text{diag}(d_1, d_2, d_3) = \begin{pmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{pmatrix}$$

### 대칭행렬 — $A^T = A$
$$A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{pmatrix}$$

**기출 1번 A=[[0,1],[1,0]]은 대칭행렬!** → 스펙트럴 정리 적용 가능

---

## 6. 시험 1번에 등장하는 행렬

$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**관찰 1:** 대칭. ($A^T = A$)
**관찰 2:** 기하적으로 "x좌표와 y좌표를 바꾸는" 변환.
- $A \begin{pmatrix} 3 \\ 4 \end{pmatrix} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}$
- $A \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} b \\ a \end{pmatrix}$

**관찰 3:** $A^2 = I$ (두 번 swap = 원래)
$$A^2 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = I$$

**의미:** "방향이 안 바뀌는 벡터"는 swap으로 자기 자신이 되거나(고유값 1), 부호만 바뀌어야(고유값 -1).
- (1,1)은 swap해도 (1,1) → 고유값 1
- (1,-1)은 swap하면 (-1,1) = -(1,-1) → 고유값 -1

이게 기출 1번의 답!

---

## 7. 외적 (outer product) — 자코비안에 등장

$\mathbf{p} \in \mathbb{R}^c$ 벡터에 대해:

$$\mathbf{p}\mathbf{p}^T = \begin{pmatrix} p_1 \\ p_2 \\ \vdots \\ p_c \end{pmatrix} \begin{pmatrix} p_1 & p_2 & \cdots & p_c \end{pmatrix} = \begin{pmatrix} p_1^2 & p_1 p_2 & \cdots \\ p_2 p_1 & p_2^2 & \cdots \\ \vdots & & \ddots \end{pmatrix}$$

c×c 행렬, (i,j) 원소는 $p_i p_j$.

**기출 8번:** softmax 자코비안 $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$
- 대각: $p_i - p_i^2 = p_i(1-p_i)$
- 비대각: $0 - p_i p_j = -p_i p_j$

---

## 8. 다리 놓기

| 중학교 | 대학 |
|-------|------|
| 좌표 (x, y) | 벡터 $\mathbf{v} \in \mathbb{R}^2$ |
| 점의 거리 √(x² + y²) | $\|\mathbf{v}\|$ |
| (배운 적 없음) | 행렬, 행렬 곱 |
| (배운 적 없음) | 단위행렬, 전치, 대칭행렬 |
| 직선 ax+by=c | $\mathbf{a}^T \mathbf{x} = c$ (내적) |

---

## 9. 시험 답안 작성법

### 행렬 곱 검증 시 쓰는 표현

> "(AB)의 (i,j) 원소는 A의 i번째 행 벡터와 B의 j번째 열 벡터의 **내적**이다."

### 대칭행렬 인용

> "A는 대칭행렬 ($A^T = A$)이므로 스펙트럴 정리에 의해 직교하는 고유벡터 기저가 존재한다."

---

## 10. 자가 점검

1. $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 5 \\ 6 \end{pmatrix}$
2. $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$
3. $\mathbf{v} = (1, 2, 3)$일 때 $\|\mathbf{v}\|^2$
4. $\mathbf{p} = (0.2, 0.3, 0.5)$일 때 $\mathbf{p}^T \mathbf{p}$

**답:**
1. $\begin{pmatrix} 17 \\ 39 \end{pmatrix}$
2. $\begin{pmatrix} 2 & 1 \\ 4 & 3 \end{pmatrix}$
3. $14$
4. $0.04 + 0.09 + 0.25 = 0.38$

---

## 다음 챕터

[`07-determinant.md`](./07-determinant.md) — 행렬식, 역행렬, det(A-λI)=0의 의미.

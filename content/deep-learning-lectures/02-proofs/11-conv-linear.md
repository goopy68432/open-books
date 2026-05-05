---
title: "10. Convolution = Linear Transformation — 완전 증명"
slug: conv-linear
order: 11
---

# 10. Convolution = Linear Transformation — 완전 증명

> **출제 근거**: 9주차 ★10, 퀴즈 28-30 직접 (1D Conv 계산, Conv→Matrix 1D, Conv→Matrix 2D)
> **시험 출제 방식**: \"Show that 1D convolution is a linear transformation. Construct its matrix and verify sparsity + weight sharing.\"

---

## 1. 왜 시험에 나오는가

- 9주차 핵심 결론: \"Conv = sparse + weight sharing 의 제한된 linear transformation\".
- 퀴즈 28-30 직접 출제 → 변형 거의 확정.
- CNN의 inductive bias (locality + translation invariance) 가 \"matrix 의 어떤 제한\"인지 시각적으로 보임.

---

## 2. 사전 수학

### 2.1 [고1] Linear Transformation 정의

$T: \mathbb{R}^n \to \mathbb{R}^m$ 가 **linear** ⟺ 다음 두 조건:

1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity**: $T(c\mathbf{u}) = c\, T(\mathbf{u})$, $c \in \mathbb{R}$

> 두 조건을 합쳐 \"$T(c_1\mathbf{u} + c_2\mathbf{v}) = c_1 T(\mathbf{u}) + c_2 T(\mathbf{v})$\" 한 줄로도 가능.

### 2.2 [대학원] 모든 Linear Transformation은 Matrix 와 일대일

표준기저 $\mathbf{e}_1, \ldots, \mathbf{e}_n$ 에 대한 $T$ 의 출력을 모은 matrix 가 $T$ 를 결정:

$$
A = \begin{pmatrix} | & | & & | \\ T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \\ | & | & & | \end{pmatrix}, \quad T(\mathbf{x}) = A\mathbf{x}
$$

> 💡 \"$T$ 가 어떤 matrix 인지 알고 싶으면 표준기저를 하나씩 넣어보면 됨\" — 시험 풀이의 표준 트릭.

### 2.3 1D Convolution 정의 (이산, valid)

입력 $\mathbf{x} \in \mathbb{R}^n$, 커널 $\mathbf{k} \in \mathbb{R}^K$:

$$
y_i = \sum_{j=1}^K k_j \cdot x_{i + j - 1}, \quad i = 1, \ldots, n - K + 1
$$

> ⚠️ ML 에서 \"convolution\" 이라 부르지만 수학적으로는 정확히 **cross-correlation** (커널 뒤집기 없음). 강의는 ML 컨벤션 따름.

**예: $n=4$, $K=3$, $\mathbf{k} = (k_1, k_2, k_3)$**:

$$
\begin{aligned}
y_1 &= k_1 x_1 + k_2 x_2 + k_3 x_3 \\
y_2 &= k_1 x_2 + k_2 x_3 + k_3 x_4
\end{aligned}
$$

출력 길이 $= n - K + 1 = 4 - 3 + 1 = 2$.

---

## 3. 증명 — Convolution 이 Linear

### Step 1 — Additivity

$\mathbf{x}, \mathbf{x}'$ 를 입력으로 하는 conv 출력:

$$
y_i = \sum_j k_j x_{i+j-1}, \quad y'_i = \sum_j k_j x'_{i+j-1}
$$

$\mathbf{x} + \mathbf{x}'$ 를 입력으로 하면:

$$
\sum_j k_j (x_{i+j-1} + x'_{i+j-1}) = \sum_j k_j x_{i+j-1} + \sum_j k_j x'_{i+j-1} = y_i + y'_i
$$

✅ Additivity 성립.

### Step 2 — Homogeneity

$c\mathbf{x}$ 를 입력:

$$
\sum_j k_j (c x_{i+j-1}) = c \sum_j k_j x_{i+j-1} = c\, y_i
$$

✅ Homogeneity 성립.

### Step 3 — 결론

따라서 conv 는 linear transformation, 어떤 행렬 $A$ 에 대해 $\mathbf{y} = A\mathbf{x}$ 표현 가능. ∎

---

## 4. 표준기저로 Matrix 구성 (퀴즈 29 패턴)

**예: $n=4$, $K=3$, $\mathbf{k}=(k_1,k_2,k_3)$. Conv matrix $A \in \mathbb{R}^{2\times 4}$.**

### Step 1: $A$ 의 각 열 = $T(\mathbf{e}_i)$

$\mathbf{e}_1 = (1,0,0,0)^\top$ 입력 시 conv 출력:

$$
y_1 = k_1\cdot 1 + k_2\cdot 0 + k_3\cdot 0 = k_1, \quad y_2 = k_1\cdot 0 + k_2\cdot 1 + k_3\cdot 0 = 0 \cdots
$$

잠깐, 식을 정확히: $y_2 = k_1 x_2 + k_2 x_3 + k_3 x_4 = 0$ (모두 0). 따라서 $T(\mathbf{e}_1) = (k_1, 0)$.

$\mathbf{e}_2 = (0,1,0,0)^\top$:
- $y_1 = k_1\cdot 0 + k_2\cdot 1 + k_3\cdot 0 = k_2$
- $y_2 = k_1\cdot 1 + k_2\cdot 0 + k_3\cdot 0 = k_1$

$T(\mathbf{e}_2) = (k_2, k_1)$.

$\mathbf{e}_3$: 비슷하게 $T(\mathbf{e}_3) = (k_3, k_2)$.

$\mathbf{e}_4$: $T(\mathbf{e}_4) = (0, k_3)$.

### Step 2: Matrix 조립

$$
A = \begin{pmatrix} k_1 & k_2 & k_3 & 0 \\ 0 & k_1 & k_2 & k_3 \end{pmatrix}
$$

### Step 3: 검증

$$
A \mathbf{x} = \begin{pmatrix} k_1 x_1 + k_2 x_2 + k_3 x_3 \\ k_1 x_2 + k_2 x_3 + k_3 x_4 \end{pmatrix} = \begin{pmatrix} y_1 \\ y_2 \end{pmatrix} \;✅
$$

---

## 5. Sparse + Weight Sharing 시각화 (★★★★★ 핵심)

위 matrix 를 다시 보면:

$$
A = \begin{pmatrix} \boxed{k_1} & \boxed{k_2} & \boxed{k_3} & 0 \\ 0 & \boxed{k_1} & \boxed{k_2} & \boxed{k_3} \end{pmatrix}
$$

### 5.1 Sparse (희소)

- 전체 $2 \times 4 = 8$ entries 중 비-0 = 6, 0 = 2.
- 큰 입력 $n$ 에선 $O(K)$ entries per row, 나머지 0 → **sparsity** ↑.
- → \"한 출력은 인접한 $K$ 개 입력만 봄\" = **Locality** (지역성)

### 5.2 Weight Sharing (가중치 공유)

- $k_1$ 이 위치 (1,1), (2,2) 두 곳에 등장. $k_2$ 는 (1,2),(2,3). $k_3$ 는 (1,3),(2,4).
- 각 row 가 한 칸씩 shift 된 같은 패턴 — **weight sharing**.
- → \"입력의 위치가 어디든 같은 커널을 적용\" = **Translation Equivariance**

### 5.3 Locality + Translation Invariance = CNN의 Inductive Bias

| Matrix 제한 | CNN의 Prior | 의미 |
|------------|------------|------|
| Sparse | Locality | 지역적 패턴이 의미 있음 |
| Shifted same row | Translation invariance | 같은 패턴이 어디서나 같은 의미 |

> 🎯 **9주차 결론 한 문장**: \"Conv = 일반 linear transformation 의 부분집합으로, sparse + weight sharing 이라는 두 제한이 prior (locality + translation invariance) 에 해당한다.\"

---

## 6. 2D Convolution → Matrix (퀴즈 30 패턴)

**예: $3\times 3$ 입력, $2\times 2$ 커널, valid → $2\times 2$ 출력. 9차원 벡터 → 4차원 벡터.**

입력을 row-major flatten:

$$
\mathbf{x} = (x_{11}, x_{12}, x_{13}, x_{21}, x_{22}, x_{23}, x_{31}, x_{32}, x_{33})^\top \in \mathbb{R}^9
$$

커널:

$$
K = \begin{pmatrix} k_{11} & k_{12} \\ k_{21} & k_{22} \end{pmatrix}
$$

출력 $y_{ij} = \sum_{a,b} k_{ab}\, x_{i+a-1, j+b-1}$.

각 출력:

$$
\begin{aligned}
y_{11} &= k_{11} x_{11} + k_{12} x_{12} + k_{21} x_{21} + k_{22} x_{22} \\
y_{12} &= k_{11} x_{12} + k_{12} x_{13} + k_{21} x_{22} + k_{22} x_{23} \\
y_{21} &= k_{11} x_{21} + k_{12} x_{22} + k_{21} x_{31} + k_{22} x_{32} \\
y_{22} &= k_{11} x_{22} + k_{12} x_{23} + k_{21} x_{32} + k_{22} x_{33}
\end{aligned}
$$

Matrix $A \in \mathbb{R}^{4\times 9}$:

$$
A = \begin{pmatrix}
k_{11} & k_{12} & 0 & k_{21} & k_{22} & 0 & 0 & 0 & 0 \\
0 & k_{11} & k_{12} & 0 & k_{21} & k_{22} & 0 & 0 & 0 \\
0 & 0 & 0 & k_{11} & k_{12} & 0 & k_{21} & k_{22} & 0 \\
0 & 0 & 0 & 0 & k_{11} & k_{12} & 0 & k_{21} & k_{22}
\end{pmatrix}
$$

> ✅ 같은 4개 weight ($k_{11}, k_{12}, k_{21}, k_{22}$) 가 4개 row 에서 각자 위치만 shift → weight sharing 명확.
> 행마다 4 entries 만 비-0 (out of 9) → sparsity.

---

## 7. Pooling 의 비-linearity (퀴즈 31, 보너스)

### Max Pooling

$y = \max(x_1, x_2)$.

**반례**: $\mathbf{u} = (1, 0)$, $\mathbf{v} = (0, 1)$.

$\max(\mathbf{u}) + \max(\mathbf{v}) = 1 + 1 = 2$, $\max(\mathbf{u} + \mathbf{v}) = \max(1, 1) = 1$.

$\max(\mathbf{u} + \mathbf{v}) \neq \max(\mathbf{u}) + \max(\mathbf{v})$ → **additivity 위반** → linear 아님. ∎

### Average Pooling

$y = (x_1 + x_2)/2$ — 선형 결합 → **linear** (행렬로 표현 가능, 모든 입력 weight = $1/n$).

> 💡 \"Conv 는 linear 이지만 활성화/풀링이 비-linear 라서 NN 전체는 비-linear\" — 직관 한 줄.

---

## 8. 모범 답안 템플릿 (퀴즈 28+29 합본)

```
[Setup]
1D conv with kernel k = (k_1, k_2, k_3), input x ∈ R^4, valid mode.
y_i = Σ_{j=1}^3 k_j x_{i+j-1},  i = 1, 2.

[Linearity proof]
For any x, x' ∈ R^4 and c ∈ R:
  Conv(x + x')_i = Σ_j k_j (x_{i+j-1} + x'_{i+j-1})
                 = Σ_j k_j x_{i+j-1} + Σ_j k_j x'_{i+j-1}
                 = Conv(x)_i + Conv(x')_i              (additivity)
  Conv(c x)_i    = Σ_j k_j (c x_{i+j-1}) = c Σ_j k_j x_{i+j-1}
                 = c · Conv(x)_i                       (homogeneity)
Hence Conv is linear.

[Matrix construction by standard basis]
Apply Conv to e_1, e_2, e_3, e_4:
  Conv(e_1) = (k_1, 0)         → 1st column of A
  Conv(e_2) = (k_2, k_1)       → 2nd column
  Conv(e_3) = (k_3, k_2)       → 3rd column
  Conv(e_4) = (0,   k_3)       → 4th column

A = [ k_1  k_2  k_3   0  ]
    [  0   k_1  k_2  k_3 ]

[Verification]
A x = (k_1 x_1 + k_2 x_2 + k_3 x_3,  k_1 x_2 + k_2 x_3 + k_3 x_4)ᵀ
which matches the convolution formula. ✓

[Interpretation]
- Sparsity: each row has only K=3 nonzero entries out of 4 → locality.
- Weight sharing: the same (k_1, k_2, k_3) appears shifted across rows
  → translation equivariance.
These two structural restrictions are the inductive bias of CNNs.
```

---

## 9. 자주 틀리는 함정

1. **Linearity 두 조건 중 하나만 보임** → 둘 다 필요.
2. **Padding/Stride 명시 누락**: \"valid\" mode 인지 \"same\" 인지 답안에 명시.
3. **Matrix 의 row/col 차원 헷갈림**: 입력 $\mathbb{R}^n$, 출력 $\mathbb{R}^{n-K+1}$ → matrix $A \in \mathbb{R}^{(n-K+1)\times n}$. 검증 차원으로.
4. **Sparse vs Weight sharing 둘 다 설명 안 함**: 핵심 통찰은 둘이 다른 제한이라는 점. 함께 적기.
5. **Pooling 반례에서 구체적 숫자 누락**: 추상적 \"max는 비선형\" 만 적으면 점수 낮음.

---

## 10. 연결 개념

- → [11 Output Size Formula](11_OutputSize_유도.md): 출력 크기 계산
- → [15 Inductive Bias 강도](15_Inductive_Bias_강도.md): CNN/Transformer/Linear 비교
- → [17 Linearity 검증](17_Linearity_검증.md): Max Pooling 반례 일반화

---
title: "제2장: 행렬과 선형 공간"
slug: 02-matrices-linear-spaces
order: 2
---

# 제2장: 행렬과 선형 공간

> **선수 과목**: 1장 벡터, 함수, 집합
> **후속 연결**: 3장 고유값 분해와 SVD, 4장 노름과 대각합

---

## 1. 동기부여 및 개요

딥러닝의 순전파(forward pass)는 본질적으로 **행렬 곱셈의 연쇄**이다. 가중치 행렬 $W$가 입력 벡터를 새로운 공간으로 변환하고, 이 변환의 구조(치역, 영공간, rank)가 네트워크의 표현력을 결정한다.

본 장에서는 행렬 연산의 다양한 관점, 선형 공간의 구조, 그리고 이들이 딥러닝에 어떻게 연결되는지를 다룬다.

```
행렬 표기/전치 ──> 행렬 곱셈 ──> 외적 관점, 대각 스케일링
                       │
선형 결합 ──> 선형 독립/기저/차원 ──> 부분공간
                                        │
                              치역 R(A), 영공간 N(A)
                                        │
                              Rank-Nullity ──> FTLA
```

---

## 2. 행렬 표기법과 전치

### 2.1 기본 표기

**Definition 2.1.** 행렬 $A \in \mathbb{R}^{m \times n}$은 $m$행 $n$열의 실수 배열이다. $(i,j)$ 원소를 $a_{ij}$로 표기한다.

- **열벡터**: $a_{:,j} \in \mathbb{R}^m$ -- 행렬의 $j$번째 열
- **행벡터**: $a_{i,:} \in \mathbb{R}^{1 \times n}$ -- 행렬의 $i$번째 행

**Definition 2.2 (전치).** $A^\top$의 $(i,j)$ 원소는 $a_{ji}$이다. $A \in \mathbb{R}^{m \times n}$이면 $A^\top \in \mathbb{R}^{n \times m}$.

### 2.2 전치의 핵심 성질

**Proposition 2.1.** $(AB)^\top = B^\top A^\top$ (순서가 뒤집힌다).

*증명.* $(AB)^\top$의 $(i,j)$ 원소 $= (AB)_{ji} = \sum_k a_{jk}b_{ki} = \sum_k (B^\top)_{ik}(A^\top)_{kj} = (B^\top A^\top)_{ij}$. $\square$

> **주의**: $(AB)^\top = A^\top B^\top$이 아니다. "양말 신고 신발 신으면, 벗을 때는 신발 먼저 벗는다"는 비유가 유용하다.

### 2.3 딥러닝에서의 전치

Transformer의 attention score $QK^\top$은 전치의 가장 대표적 응용이다. FlashAttention(Dao et al., 2022) 등 메모리 효율적 attention 설계에서 전치의 성질이 핵심적으로 활용된다.

---

## 3. 행렬 곱셈의 세 가지 관점

$A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$일 때 $C = AB \in \mathbb{R}^{m \times p}$.

### 3.1 내적 관점 (원소별)

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} = a_{i,:} \cdot b_{:,j}$$

시간 복잡도: $O(mnp)$. 정방행렬의 경우 $O(n^3)$.

### 3.2 열 관점 (행렬-벡터 곱의 반복)

$$AB = [Ab_1 \quad Ab_2 \quad \cdots \quad Ab_p]$$

$C$의 각 열은 $A$에 $B$의 해당 열을 곱한 것이다. **행렬-벡터 곱** $Av$는 $A$의 열벡터들의 선형 결합이다:

$$Av = \sum_{i=1}^n v_i a_{:,i}$$

이는 $Av$의 결과가 반드시 $A$의 **열공간(column space)**에 속함을 의미한다.

### 3.3 외적 관점 (Rank-1 행렬의 합)

$$AB = \sum_{k=1}^{n} a_{:,k} \cdot b_{k,:} = \sum_{k=1}^n a_k b_k^\top$$

각 $a_k b_k^\top$은 rank-1 행렬이다. 이 관점은 SVD의 저랭크 근사(3장)와 LoRA의 이론적 기반이 된다.

```python
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
B = np.array([[7, 8, 9], [10, 11, 12]]) # (2, 3)

# 세 가지 관점 모두 같은 결과
C_standard = A @ B
C_outer = sum(np.outer(A[:, k], B[k, :]) for k in range(A.shape[1]))
C_col = np.column_stack([A @ B[:, j] for j in range(B.shape[1])])
assert np.allclose(C_standard, C_outer) and np.allclose(C_standard, C_col)
```

> **딥러닝 연결**: LoRA (Hu et al., 2021)에서 $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}, r \ll d$)는 외적 관점에서 $r$개의 rank-1 행렬의 합으로 가중치를 업데이트하는 것이다.

---

## 4. 대각 행렬 스케일링

**Definition 4.1.** 대각 행렬 $D = \text{diag}(d_1, \ldots, d_n)$은 대각 원소만 0이 아닌 정방행렬이다.

### 4.1 행 스케일링과 열 스케일링

| 연산 | 효과 |
|------|------|
| $DX$ | $X$의 $i$번째 **행**에 $d_i$를 곱함 |
| $XD$ | $X$의 $j$번째 **열**에 $d_j$를 곱함 |

### 4.2 딥러닝에서의 대각 스케일링

- **Batch Normalization** (Ioffe & Szegedy, 2015): 학습 가능한 스케일 파라미터 $\gamma$는 특징별 대각 스케일링
- **Adam optimizer**: gradient의 각 차원을 $1/\sqrt{v_t + \epsilon}$로 스케일링 -- 사실상 대각 pre-conditioning
- **고유값 분해**: $A = U\Lambda U^\top$에서 $\Lambda$는 대각 행렬 (3장에서 상세히)

---

## 5. Strassen 알고리즘 (참고)

나이브 행렬 곱셈은 $O(n^3)$이지만, Strassen (1969)은 $2 \times 2$ 블록에서 8번의 곱셈을 **7번으로** 줄이는 방법을 제안했다:

$$T(n) = 7T(n/2) + O(n^2) \implies T(n) = O(n^{\log_2 7}) \approx O(n^{2.807})$$

2024년 기준 이론적 최선은 약 $O(n^{2.37})$ (Alman & Williams)이며, 하한 $\omega = 2$는 미해결 문제이다. 실제 GPU에서는 캐시/병렬성 최적화(GEMM, Tensor Core)가 이론적 알고리즘보다 효과적이다.

---

## 6. 선형 결합, 선형 독립, 기저, 차원

### 6.1 정의

**Definition 6.1 (선형 결합).** 벡터 $v_1, \ldots, v_n$과 스칼라 $a_1, \ldots, a_n$에 대해 $\sum_{i=1}^n a_i v_i$를 선형 결합이라 한다.

**Definition 6.2 (선형 독립).** $\{v_1, \ldots, v_n\}$이 선형 독립(linearly independent)이라 함은:

$$\sum_{i=1}^n a_i v_i = \mathbf{0} \implies a_1 = a_2 = \cdots = a_n = 0$$

**Definition 6.3 (Span).** $\text{span}(S)$는 $S$의 모든 유한 선형 결합의 집합이다.

**Definition 6.4 (기저).** 벡터 공간 $V$의 기저(basis)는 선형 독립이면서 $\text{span}$이 $V$ 전체인 벡터 집합이다.

**Definition 6.5 (차원).** $\dim(V) = |B|$ (기저의 원소 수). 기저는 유일하지 않지만 기저의 **크기(차원)**는 항상 같다.

### 6.2 선형 독립의 판별

$v_1, \ldots, v_n$을 열로 하는 행렬 $V = [v_1 \; \cdots \; v_n]$에 대해:

$$\{v_i\} \text{ 선형 독립} \iff \text{rank}(V) = n \iff \det(V^\top V) \neq 0 \text{ (정방일 때)}$$

```python
import numpy as np
v1, v2, v3 = np.array([1,0,1]), np.array([0,1,1]), np.array([1,1,2])
V = np.column_stack([v1, v2, v3])
print(np.linalg.matrix_rank(V))  # 2 (v3 = v1 + v2이므로 종속)
```

### 6.3 딥러닝 연결

- **Overparameterization**: 딥러닝 모델의 파라미터 수가 데이터 수를 크게 초과하지만, 가중치 공간의 **실효 차원(effective dimension)**은 훨씬 낮다 (Li et al., 2018).
- **Pruning & Quantization**: 종속적인(불필요한) 파라미터를 제거하는 것은 기저만 남기는 것과 유사하다.

---

## 7. 부분공간

### 7.1 정의와 조건

**Definition 7.1 (부분공간).** $\mathbb{R}^n$의 부분집합 $V$가 부분공간(subspace)이라 함은:

1. $\mathbf{0} \in V$
2. $u, v \in V \implies u + v \in V$ (덧셈에 닫힘)
3. $u \in V,\; c \in \mathbb{R} \implies cu \in V$ (스칼라곱에 닫힘)

> **주의**: 원점을 포함하지 않는 평면이나 직선은 부분공간이 아니다. 또한 두 부분공간의 **합집합**은 일반적으로 부분공간이 아니지만, **합** $V + W = \{v + w : v \in V, w \in W\}$는 부분공간이다.

### 7.2 딥러닝과 Manifold Hypothesis

실제 데이터는 고차원 공간의 **저차원 부분다양체(manifold)**에 놓여 있다는 가설이 Manifold Hypothesis이다. 부분공간은 가장 단순한(선형인) manifold이다. Word2Vec, BERT 등의 임베딩은 고차원 공간의 구조화된 부분공간에 의미 정보를 인코딩한다.

---

## 8. 치역과 영공간

### 8.1 정의

**Definition 8.1 (치역/열공간).** $A \in \mathbb{R}^{m \times n}$에 대해:

$$\mathscr{R}(A) := \{Av : v \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$$

$Av$는 $A$의 열벡터의 선형 결합이므로, 치역 = 열공간(column space).

**Definition 8.2 (영공간/핵).**

$$\mathscr{N}(A) := \{v \in \mathbb{R}^n : Av = \mathbf{0}\} \subseteq \mathbb{R}^n$$

### 8.2 부분공간임의 증명

**Proposition 8.1.** $\mathscr{N}(A)$는 $\mathbb{R}^n$의 부분공간이다.

*증명.*
1. $A\mathbf{0} = \mathbf{0}$ → $\mathbf{0} \in \mathscr{N}(A)$
2. $Av_1 = \mathbf{0},\; Av_2 = \mathbf{0}$ → $A(v_1 + v_2) = Av_1 + Av_2 = \mathbf{0}$
3. $Av = \mathbf{0}$ → $A(cv) = c(Av) = \mathbf{0}$ $\square$

### 8.3 직관적 해석

- **치역**: 행렬 $A$를 통과하여 도달 가능한 출력의 집합
- **영공간**: $A$를 통과하면 소멸되는 입력의 집합 -- **정보 손실 방향**

```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # rank 2

# 열공간 기저 (SVD 활용)
U, S, Vt = np.linalg.svd(A)
rank = np.sum(S > 1e-10)
col_space = U[:, :rank]  # rank개의 직교 기저

# 영공간
ns = null_space(A)
print(f"rank = {rank}, nullity = {ns.shape[1]}")  # rank=2, nullity=1
print(f"A @ null_vector = {A @ ns[:, 0]}")        # ≈ [0, 0, 0]
```

### 8.4 딥러닝 연결

- **Adversarial Robustness**: 입력의 작은 변화 $\delta$가 Jacobian의 영공간에 있으면 출력이 변하지 않지만, 영공간 바깥이면 출력이 크게 변할 수 있다.
- **Residual Connection**: $y = x + F(x)$에서 $F$의 영공간에 있는 입력은 그대로 통과 → 정보 보존.
- **Dropout**: 무작위로 열을 제거하여 치역의 차원을 줄이는 효과 → 암묵적 정규화.

---

## 9. Rank-Nullity 정리

### 9.1 정리

**Theorem 9.1 (Rank-Nullity Theorem).** $A \in \mathbb{R}^{m \times n}$에 대해:

$$n = \text{rank}(A) + \text{nullity}(A)$$

여기서 $\text{rank}(A) = \dim(\mathscr{R}(A))$, $\text{nullity}(A) = \dim(\mathscr{N}(A))$.

**직관**: 입력 공간의 차원 = 살아남는 차원 + 소멸되는 차원. 정보의 보존 법칙이다.

### 9.2 Rank의 주요 성질

| 성질 | 수식 | 의미 |
|------|------|------|
| 상한 | $\text{rank}(A) \leq \min(m, n)$ | 행과 열 수 중 작은 것 이하 |
| 전치 불변 | $\text{rank}(A) = \text{rank}(A^\top)$ | 행 rank = 열 rank |
| Gram 행렬 | $\text{rank}(A) = \text{rank}(A^\top A)$ | 최소제곱법의 기초 |
| 곱의 부등식 | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ | 합성하면 rank 감소 가능 |
| 합의 부등식 | $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$ | 더하면 rank 증가 가능 |

### 9.3 딥러닝 연결

- **LoRA**: 학습된 가중치 행렬의 rank가 full보다 훨씬 낮은 경우가 많다 → 저랭크 업데이트로 충분
- **Neural Collapse** (Papyan et al., 2020): 학습 후기에 특징 벡터의 rank가 클래스 수와 같아지는 현상
- **Effective rank** = $\exp(H(\sigma_1, \ldots, \sigma_r))$: 특이값 분포의 엔트로피 기반 연속적 rank 지표

---

## 10. 선형대수 기본정리 (FTLA)

### 10.1 네 개의 근본 부분공간

$A \in \mathbb{R}^{m \times n}$에 대해:

| 부분공간 | 정의 | 소속 공간 | 차원 |
|---------|------|----------|------|
| 열공간 $\mathscr{R}(A)$ | $\{Av\}$ | $\mathbb{R}^m$ | $r$ |
| 행공간 $\mathscr{R}(A^\top)$ | $\{A^\top u\}$ | $\mathbb{R}^n$ | $r$ |
| 영공간 $\mathscr{N}(A)$ | $\{v : Av = 0\}$ | $\mathbb{R}^n$ | $n - r$ |
| 왼쪽 영공간 $\mathscr{N}(A^\top)$ | $\{u : A^\top u = 0\}$ | $\mathbb{R}^m$ | $m - r$ |

여기서 $r = \text{rank}(A)$.

### 10.2 직교 분해

**Theorem 10.1 (FTLA).**

$$\mathbb{R}^n = \mathscr{N}(A) \oplus \mathscr{R}(A^\top), \qquad \mathbb{R}^m = \mathscr{N}(A^\top) \oplus \mathscr{R}(A)$$

여기서 $\oplus$는 **직교 직합(orthogonal direct sum)**이다.

*증명 스케치.* $\mathscr{R}(A^\top) \perp \mathscr{N}(A)$임을 보인다. $x \in \mathscr{N}(A)$이고 $y = A^\top z \in \mathscr{R}(A^\top)$이면:

$$\langle x, y \rangle = x^\top A^\top z = (Ax)^\top z = \mathbf{0}^\top z = 0 \quad \square$$

### 10.3 직관적 해석

모든 입력 벡터 $x \in \mathbb{R}^n$는 다음과 같이 유일하게 분해된다:

$$x = x_{\text{row}} + x_{\text{null}}, \quad x_{\text{row}} \in \mathscr{R}(A^\top), \; x_{\text{null}} \in \mathscr{N}(A)$$

- $x_{\text{row}}$: 행렬 $A$가 "보는" (반응하는) 성분
- $x_{\text{null}}$: 행렬 $A$가 "무시하는" (소멸시키는) 성분

### 10.4 SVD와의 관계

SVD $A = U\Sigma V^\top$는 네 개의 근본 부분공간을 모두 제공한다:
- $U$의 처음 $r$개 열 → $\mathscr{R}(A)$의 정규직교 기저
- $V$의 처음 $r$개 열 → $\mathscr{R}(A^\top)$의 정규직교 기저
- $V$의 나머지 열 → $\mathscr{N}(A)$의 정규직교 기저
- $U$의 나머지 열 → $\mathscr{N}(A^\top)$의 정규직교 기저

이 관계는 3장에서 상세히 다룬다.

```python
import numpy as np
from scipy.linalg import null_space

A = np.array([[1, 1]])  # 1x2 행렬, rank=1

U, S, Vt = np.linalg.svd(A)
rank = np.sum(S > 1e-10)

# R(A^T)의 기저와 N(A)의 기저
row_space = Vt[:rank, :].T       # [0.707, 0.707]
null_space_A = null_space(A)     # [-0.707, 0.707]

# 직교 확인
print(row_space.T @ null_space_A)  # ≈ 0

# 직합 분해
x = np.array([3, 1])
x_row = row_space @ (row_space.T @ x)
x_null = null_space_A @ (null_space_A.T @ x)
assert np.allclose(x, (x_row + x_null).flatten())
```

---

## 11. 딥러닝 적용 요약

| 수학 개념 | 딥러닝 대응 | 구체적 사례 |
|-----------|------------|------------|
| 행렬-벡터 곱 $Wx$ | Linear Layer 순전파 | `nn.Linear(in, out)` |
| 행렬 곱셈 $AB$ | 다층 변환의 합성 | Forward pass 전체 |
| 전치 $(AB)^\top = B^\top A^\top$ | Attention score | $QK^\top$ |
| 외적 관점 $\sum a_k b_k^\top$ | 저랭크 근사 | LoRA: $\Delta W = BA$ |
| 대각 스케일링 | 정규화, 적응적 학습률 | BatchNorm $\gamma$, Adam |
| 선형 독립/기저/차원 | 모델 용량 분석 | Effective rank |
| 치역/영공간 | 정보 보존/손실 분석 | Adversarial robustness |
| Rank-Nullity | 차원 보존 법칙 | Skip connection 분석 |
| FTLA | 직교 분해 | PCA, SVD 기반 분석 |

---

## 12. 흔한 오해와 주의점

1. **"$AB = BA$"** → 행렬 곱은 교환법칙이 성립하지 않는다. 레이어 순서가 바뀌면 다른 네트워크.
2. **"$(AB)^\top = A^\top B^\top$"** → 순서가 뒤집힌다: $(AB)^\top = B^\top A^\top$.
3. **"행렬 곱 = 원소별 곱"** → 원소별 곱(Hadamard, $\odot$)과 행렬 곱($@$)은 완전히 다른 연산.
4. **"열이 $n$개면 rank는 $n$"** → rank는 **독립인** 열의 수. 종속 열이 있으면 rank $< n$.
5. **"치역 = 공역($\mathbb{R}^m$)"** → rank $< m$이면 치역은 $\mathbb{R}^m$의 진부분공간.
6. **"부분공간 = 아무 부분집합"** → 원점을 포함하고, 덧셈/스칼라곱에 닫혀야 한다.
7. **"$DX = XD$"** → $DX$는 행 스케일링, $XD$는 열 스케일링으로 결과가 다르다.

---

## 13. 핵심 요약

1. **행렬 곱셈**은 내적 관점, 열 관점, 외적 관점의 세 가지로 해석된다. 외적 관점은 저랭크 근사의 기초.
2. **선형 독립/기저/차원**은 공간의 "자유도"를 측정하며, 모델의 실효 파라미터 수와 직결된다.
3. **치역**은 변환으로 도달 가능한 출력, **영공간**은 소멸되는 입력의 방향이다.
4. **Rank-Nullity 정리**: $n = \text{rank}(A) + \text{nullity}(A)$ -- 입력 차원의 보존 법칙.
5. **FTLA**: 입력/출력 공간이 각각 직교하는 두 부분공간으로 유일하게 분해된다.

> **한 문장 정리**: 행렬은 공간을 변환하는 함수이며, 그 변환의 구조(치역, 영공간, rank)가 딥러닝의 모든 선형 연산을 지배한다.

---

## 참고문헌

- Strang, G. (2019). *Linear Algebra and Its Applications*, 5th ed. Cengage.
- Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press. Chapter 2.
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Papyan, V. et al. (2020). Prevalence of Neural Collapse during the terminal phase of deep learning training.
- Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention.
- Li, C. et al. (2018). Measuring the Intrinsic Dimension of Objective Landscapes.

---
title: "02. 행렬과 선형 공간 -- 데이터 변환의 구조 이해하기"
slug: 02-matrices-linear-spaces
order: 2
---

# 02. 행렬과 선형 공간 -- 데이터 변환의 구조 이해하기

> 이 문서는 고등학생 수준에서 행렬 연산과 선형 공간의 개념을 설명합니다.

---

## 왜 이걸 배워야 할까?

딥러닝에서 데이터가 신경망을 통과할 때, 매 레이어마다 **행렬 곱셈**이 일어난다. 행렬이 어떻게 데이터를 변환하는지, 변환 후 어떤 정보가 살아남고 어떤 정보가 사라지는지를 이해하면, AI가 "왜 이렇게 작동하는지"를 꿰뚫을 수 있다.

---

## 1. 행렬 표기법과 전치 (Transpose)

### 핵심 개념

행렬은 숫자를 **행(가로줄)과 열(세로줄)**로 배열한 직사각형 표이다.

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

- 2행 3열 행렬 ($m = 2$, $n = 3$)
- $a_{12} = 2$ (1행 2열의 값)

### 전치 (Transpose)

행과 열을 뒤집는 연산: $(A^{\top})_{ij} = a_{ji}$

$$A^{\top} = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

**핵심 성질**: $(AB)^{\top} = B^{\top}A^{\top}$ (순서가 뒤집힌다!)

> 비유: 양말 신고 신발 신으면, 벗을 때는 신발부터 벗어야 한다.

### 열벡터와 행벡터

- **열벡터**: 행렬에서 세로줄 하나를 뽑은 것 ($a_{:,j} \in \mathbb{R}^m$)
- **행벡터**: 행렬에서 가로줄 하나를 뽑은 것 ($a_{i,:} \in \mathbb{R}^{1 \times n}$)

### 오해하기 쉬운 포인트

| 오해 | 실제 |
|------|------|
| $(AB)^{\top} = A^{\top}B^{\top}$ | 순서가 뒤집힌다: $(AB)^{\top} = B^{\top}A^{\top}$ |
| 전치하면 크기가 같다 | $m \times n$ 행렬의 전치는 $n \times m$이다 |

---

## 2. 행렬 곱셈 (Matrix Multiplication)

### 핵심 개념

$A \in \mathbb{R}^{m \times n}$과 $B \in \mathbb{R}^{n \times p}$의 곱 $C = AB \in \mathbb{R}^{m \times p}$:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} \cdot b_{kj} = \text{(A의 i번째 행과 B의 j번째 열의 내적)}$$

### 숫자 예시

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\ 3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

### 행렬 곱의 세 가지 관점

**1) 내적 관점**: $c_{ij}$ = A의 $i$행과 B의 $j$열의 내적

**2) 열벡터 관점**: $Av = v_1 (\text{A의 1열}) + v_2 (\text{A의 2열}) + \cdots$
- 행렬에 벡터를 곱하면 = 행렬의 열벡터들의 **가중합**

**3) 외적(Outer Product) 관점**: $AB = \sum_{k} (\text{A의 k열}) \cdot (\text{B의 k행})$
- 행렬 곱을 rank-1 행렬들의 합으로 분해

### 시간 복잡도

| 연산 | 복잡도 |
|------|--------|
| 일반 행렬 곱 ($m \times n$ $\cdot$ $n \times p$) | $O(mnp)$ |
| 정방 행렬 곱 ($n \times n$) | $O(n^3)$ |
| 행렬-벡터 곱 ($n \times n$ $\cdot$ $n \times 1$) | $O(n^2)$ |

### 중요한 규칙

- **곱셈 조건**: A의 열 수 = B의 행 수
- **교환법칙 없음**: $AB \neq BA$ (일반적으로)
- **결합법칙 성립**: $(AB)C = A(BC)$

### 딥러닝에서 왜 중요한가?

- 신경망의 순전파(forward pass) 전체가 행렬 곱의 연쇄
- GPU가 빠른 이유: 행렬 곱을 병렬로 처리하도록 설계
- Attention: $QK^{\top}V$ = 행렬 곱 세 번

### 오해하기 쉬운 포인트

| 오해 | 실제 |
|------|------|
| 행렬 곱 = 같은 위치끼리 곱하기 | 그것은 **원소별 곱(Hadamard product)**. 행렬 곱은 행과 열의 내적 |
| $AB = BA$ | 행렬 곱은 교환법칙이 성립하지 않는다 |

---

## 3. 대각 행렬 스케일링

### 핵심 개념

대각 행렬 $D = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$은 각 방향을 **따로 스케일링**한다.

$$DX = \text{각 행에 다른 배수 곱하기 (행 스케일링)}$$
$$XD = \text{각 열에 다른 배수 곱하기 (열 스케일링)}$$

### 예시

$$D = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}, \quad X = \begin{bmatrix} 1 & 4 \\ 5 & 6 \end{bmatrix}$$

$$DX = \begin{bmatrix} 2 & 8 \\ 15 & 18 \end{bmatrix}, \quad XD = \begin{bmatrix} 2 & 12 \\ 10 & 18 \end{bmatrix}$$

### 딥러닝에서 왜 중요한가?

- **Batch Normalization**의 스케일 파라미터 $\gamma$: 각 특징(feature)마다 다른 배수를 곱하는 것 = 대각 스케일링
- **고유값 분해** $A = U\Lambda U^{\top}$에서 $\Lambda$가 바로 대각 행렬 (다음 장에서 자세히)

---

## 4. 선형 결합, 선형 독립, Span, 기저, 차원

### 선형 결합 (Linear Combination)

벡터들을 스칼라 배하여 더하는 것:

$$3\begin{bmatrix}1\\0\end{bmatrix} + 2\begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}3\\2\end{bmatrix}$$

### 선형 독립 (Linearly Independent)

벡터 집합 $\{v_1, v_2, \ldots, v_n\}$이 **선형 독립**이라 함은:

$$a_1 v_1 + a_2 v_2 + \cdots + a_n v_n = \mathbf{0} \implies a_1 = a_2 = \cdots = a_n = 0$$

"어떤 벡터도 나머지의 조합으로 만들 수 없다"

- 독립 예: $\{(1,0), (0,1)\}$ -- 하나를 다른 것의 배수로 못 만든다
- 종속 예: $\{(1,0), (2,0)\}$ -- 두 번째가 첫 번째의 2배

### Span

벡터 집합으로 만들 수 있는 **모든 선형 결합의 집합**:

$$\text{span}\{v_1, v_2\} = \{a_1 v_1 + a_2 v_2 : a_1, a_2 \in \mathbb{R}\}$$

### 기저 (Basis)

**선형 독립**이면서 **span이 전체 공간**인 벡터 집합:

- $\{(1,0), (0,1)\}$은 $\mathbb{R}^2$의 기저 (표준 기저)
- $\{(1,1), (1,-1)\}$도 $\mathbb{R}^2$의 기저 (기저는 유일하지 않다!)

### 차원 (Dimension)

기저에 있는 벡터의 수: $\dim(\mathbb{R}^n) = n$

**핵심**: 벡터가 아무리 많아도, 독립인 것만 세면 그것이 차원이다.

### 딥러닝에서 왜 중요한가?

- 모델의 **파라미터 수**가 많아도 실질적으로 독립인 것은 적을 수 있다 (LoRA의 이론적 근거)
- **임베딩 차원** = 데이터를 표현하는 데 필요한 독립적 방향의 수

### 오해하기 쉬운 포인트

| 오해 | 실제 |
|------|------|
| 벡터가 많으면 차원이 높다 | 차원은 **독립인 벡터의 수**이다. 100개가 있어도 모두 같은 직선 위면 차원은 1 |
| 기저는 유일하다 | 기저는 무한히 많지만, 기저의 **크기(=차원)**는 항상 같다 |

---

## 5. 부분공간 (Subspace)

### 핵심 개념

$\mathbb{R}^n$ 안에서 다음 3가지를 모두 만족하는 부분집합:

1. 영벡터 $\mathbf{0}$을 포함
2. 덧셈에 닫혀 있음: $u, v \in V \implies u + v \in V$
3. 스칼라 곱에 닫혀 있음: $v \in V, c \in \mathbb{R} \implies cv \in V$

**직관**: 안의 벡터를 아무리 더하거나 배수해도 밖으로 나가지 않는 공간

### $\mathbb{R}^3$의 부분공간 예시

| 차원 | 모양 | 예시 |
|------|------|------|
| 0 | 원점 | $\{\mathbf{0}\}$ |
| 1 | 원점을 지나는 직선 | $\{t(1,0,1) : t \in \mathbb{R}\}$ |
| 2 | 원점을 지나는 평면 | $xy$-평면: $\{(x,y,0)\}$ |
| 3 | 전체 공간 | $\mathbb{R}^3$ |

**주의**: 원점을 지나지 않는 평면 $\{(x,y,1)\}$은 부분공간이 **아니다**!

---

## 6. 치역(Range)과 영공간(Null Space)

### 치역 (Column Space, Range)

$$\mathscr{R}(A) = \{Av : v \in \mathbb{R}^n\}$$

$A$에 모든 가능한 입력을 넣었을 때 **나올 수 있는 출력의 집합**

열벡터의 모든 선형 결합이므로 "열공간"이라고도 부른다.

### 영공간 (Null Space, Kernel)

$$\mathscr{N}(A) = \{v \in \mathbb{R}^n : Av = \mathbf{0}\}$$

$A$를 곱하면 **영벡터가 되어 사라지는** 모든 입력의 집합

### 직관적 비유

사진을 흑백으로 바꾸는 필터를 생각하자:
- **치역**: 필터를 거쳐 나올 수 있는 모든 흑백 사진
- **영공간**: 필터를 거치면 완전히 사라지는(까만 화면) 입력 -- 순수한 색 정보만 있는 경우

```
입력 공간 R^n
  |--- 행공간 R(A^T) ---> 치역 R(A) [살아남는 정보]
  |--- 영공간 N(A)   ---> 0        [사라지는 정보]
```

---

## 7. Rank-Nullity 정리

### 핵심 공식

$A \in \mathbb{R}^{m \times n}$에 대해:

$$\boxed{n = \text{rank}(A) + \text{nullity}(A)}$$

- $\text{rank}(A) = \dim(\mathscr{R}(A))$: 살아남는 차원 (독립 열의 수)
- $\text{nullity}(A) = \dim(\mathscr{N}(A))$: 사라지는 차원

**직관**: 입력 차원 = 보존되는 정보 + 손실되는 정보 (보존 법칙!)

### 예시

$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$ (2행이 1행의 2배)

- $\text{rank}(A) = 1$ (독립 행이 1개)
- $n = 3$이므로 $\text{nullity}(A) = 3 - 1 = 2$

### Rank의 주요 성질

| 성질 | 수식 |
|------|------|
| 상한 | $\text{rank}(A) \leq \min(m, n)$ |
| 전치 불변 | $\text{rank}(A) = \text{rank}(A^{\top})$ |
| 곱에 대해 | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ |
| 합에 대해 | $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$ |

### 딥러닝에서 왜 중요한가?

- **모델 압축(LoRA)**: 학습된 가중치 행렬의 rank가 전체 크기보다 훨씬 낮은 경우가 많다. 이를 이용해 저랭크 근사로 파라미터 수를 줄인다.
- **Skip Connection (ResNet)**: $y = x + F(x)$에서, $F$의 영공간에 있는 정보도 $x$를 통해 보존된다.

---

## 8. 선형대수 기본정리 (FTLA)

### 4개의 근본 부분공간

하나의 행렬 $A \in \mathbb{R}^{m \times n}$에서 4개의 부분공간이 나온다:

| 부분공간 | 소속 공간 | 차원 |
|---------|----------|------|
| 열공간 $\mathscr{R}(A)$ | $\mathbb{R}^m$ | $r$ |
| 왼쪽 영공간 $\mathscr{N}(A^{\top})$ | $\mathbb{R}^m$ | $m - r$ |
| 행공간 $\mathscr{R}(A^{\top})$ | $\mathbb{R}^n$ | $r$ |
| 영공간 $\mathscr{N}(A)$ | $\mathbb{R}^n$ | $n - r$ |

여기서 $r = \text{rank}(A)$

### 핵심 관계: 직교 분해

$$\mathbb{R}^n = \mathscr{N}(A) \oplus \mathscr{R}(A^{\top})$$

**직관**: 모든 입력 벡터를 "행렬이 보는 부분(행공간)"과 "행렬이 무시하는 부분(영공간)"으로 **겹침 없이, 빈틈 없이** 분해할 수 있다.

두 부분공간은 **직교**(수직)이다: $\mathscr{R}(A^{\top}) \perp \mathscr{N}(A)$

---

## 정리 / 요약

### 한 줄 요약

> 행렬은 공간을 변환하는 함수이며, 변환의 구조(치역/영공간/rank)가 "어떤 정보가 살아남고 사라지는지"를 결정한다.

### 개념 연결 흐름

```
행렬 표기 & 전치
      |
      v
행렬 곱셈 (내적 관점 / 열 관점 / 외적 관점)
      |
      v
선형 결합 --> 선형 독립 --> 기저 --> 차원
      |
      v
부분공간 --> 열공간(치역) + 영공간
      |
      v
Rank-Nullity 정리: n = rank + nullity
      |
      v
선형대수 기본정리: 4개의 근본 부분공간
```

### 핵심 공식 정리

| 개념 | 공식 | 한 줄 설명 |
|------|------|-----------|
| 행렬 곱 | $c_{ij} = \sum_k a_{ik} b_{kj}$ | A의 행과 B의 열의 내적 |
| 전치의 곱 | $(AB)^{\top} = B^{\top}A^{\top}$ | 순서가 뒤집힌다 |
| 외적 분해 | $AB = \sum_k a_k b_k^{\top}$ | rank-1 행렬들의 합 |
| Rank-Nullity | $n = \text{rank}(A) + \text{nullity}(A)$ | 입력 차원의 보존 법칙 |
| FTLA | $\mathbb{R}^n = \mathscr{N}(A) \oplus \mathscr{R}(A^{\top})$ | 직교 분해 |

### 오해하기 쉬운 포인트 모음

| 오해 | 실제 |
|------|------|
| $AB = BA$ | 행렬 곱은 교환법칙이 성립하지 않는다 |
| 행렬 곱 = 같은 위치끼리 곱하기 | 행렬 곱은 행과 열의 내적 |
| rank = 열의 수 | rank = **독립인** 열의 수 |
| 영공간이 비어있다 | 최소한 영벡터 $\mathbf{0}$을 포함한다 |
| 아무 평면이나 부분공간 | 부분공간은 반드시 원점을 포함해야 한다 |
| 치역 = 공역($\mathbb{R}^m$) | 치역은 공역의 부분공간일 수 있다 (rank < $m$이면) |

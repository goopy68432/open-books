---
title: "딥러닝 수학 기초 — 중간고사 모의시험 풀이 (수준별 해설)"
slug: solutions-korean
order: 3
---

# 딥러닝 수학 기초 — 중간고사 모의시험 풀이 (수준별 해설)

> 각 문제마다 **중학생 / 고등학생 / 대학생** 수준별 풀이를 제공합니다.
> 이성윤 교수님의 출제 스타일에 맞춰, **논리적 과정과 유도**를 중심으로 풀이합니다.

---

# 파트 1: 선형대수 기초 (Q1–Q10)

---

## 1번. 내적과 직교성

### 중학생 수준
**(a)** 두 벡터가 직교하면 내적 = 0이에요.
$$2 \times 1 + (-1) \times 4 + 3 \times k = 0$$
$$2 - 4 + 3k = 0$$
$$3k = 2$$
$$k = \frac{2}{3}$$

**(b)** 피타고라스 정리 검증:
- $\|\mathbf{u}\|^2 = 4 + 1 + 9 = 14$
- $\|\mathbf{v}\|^2 = 1 + 16 + 4/9 = 17 + 4/9 = 157/9$
- $\mathbf{u} + \mathbf{v} = (3, 3, 11/3)^T$
- $\|\mathbf{u}+\mathbf{v}\|^2 = 9 + 9 + 121/9 = 162/9 + 121/9 = 283/9$
- $14 + 157/9 = 126/9 + 157/9 = 283/9$ ✓ 같다!

### 고등학생 수준
**(a)** 직교 조건: $\mathbf{u} \cdot \mathbf{v} = 0$
$$\mathbf{u} \cdot \mathbf{v} = (2)(1) + (-1)(4) + (3)(k) = 2 - 4 + 3k = 3k - 2 = 0$$
$$\therefore k = \frac{2}{3}$$

**(b)** 일반적으로 $\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\mathbf{u}\cdot\mathbf{v} + \|\mathbf{v}\|^2$이고, 직교이면 $\mathbf{u}\cdot\mathbf{v} = 0$이므로:
$$\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$
이는 유클리드 공간에서의 피타고라스 정리의 일반화이다.

수치 검증: $\|\mathbf{u}\|^2 = 14$, $\|\mathbf{v}\|^2 = 1 + 16 + 4/9 = 157/9$, $\|\mathbf{u}+\mathbf{v}\|^2 = 9 + 9 + 121/9 = 283/9 = 14 + 157/9$ ✓

### 대학생 수준
**(a)** $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = 2 - 4 + 3k = 0 \implies k = 2/3$.

**(b)** 증명 (일반적):
$$\|\mathbf{u}+\mathbf{v}\|^2 = (\mathbf{u}+\mathbf{v})^T(\mathbf{u}+\mathbf{v}) = \mathbf{u}^T\mathbf{u} + \mathbf{u}^T\mathbf{v} + \mathbf{v}^T\mathbf{u} + \mathbf{v}^T\mathbf{v} = \|\mathbf{u}\|^2 + 2\langle\mathbf{u},\mathbf{v}\rangle + \|\mathbf{v}\|^2$$
$\mathbf{u} \perp \mathbf{v} \implies \langle\mathbf{u},\mathbf{v}\rangle = 0$이므로 $\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$. $\square$

이는 $n$차원으로 자연스럽게 일반화되며, 서로 직교하는 $k$개 벡터에 대해:
$$\left\|\sum_{i=1}^k \mathbf{v}_i\right\|^2 = \sum_{i=1}^k \|\mathbf{v}_i\|^2$$

---

## 2번. 열의 선형결합으로서의 행렬-벡터 곱셈

### 중학생 수준
**(a)** $A\mathbf{x}$는 $A$의 첫째 열에 2를 곱하고, 둘째 열에 5를 곱해서 더하는 거예요:
$$A\mathbf{x} = 2 \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix} + 5 \begin{pmatrix} 3 \\ -1 \\ 4 \end{pmatrix} = \begin{pmatrix} 2 \\ 4 \\ 0 \end{pmatrix} + \begin{pmatrix} 15 \\ -5 \\ 20 \end{pmatrix} = \begin{pmatrix} 17 \\ -1 \\ 20 \end{pmatrix}$$

**(b)** 이렇게 보면 "행렬 × 벡터 = 열들의 조합"이라는 걸 알 수 있어요. 즉, 결과는 항상 A의 열들을 조합해서 만들 수 있는 벡터 중 하나예요.

### 고등학생 수준
**(a)** $A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2$ (열벡터 관점):
$$= 2\begin{pmatrix}1\\2\\0\end{pmatrix} + 5\begin{pmatrix}3\\-1\\4\end{pmatrix} = \begin{pmatrix}2+15\\4-5\\0+20\end{pmatrix} = \begin{pmatrix}17\\-1\\20\end{pmatrix}$$

**(b)** 열 관점의 중요성:
- $A\mathbf{x}$의 결과는 항상 $A$의 열공간 $\text{Col}(A) = \text{span}\{\mathbf{a}_1, \mathbf{a}_2\}$에 속함
- $A\mathbf{x} = \mathbf{b}$가 해를 가지려면 $\mathbf{b} \in \text{Col}(A)$이어야 함
- 이는 "방정식이 풀리는가?"를 기하학적으로 이해하게 해줌

### 대학생 수준
**(a)** $A\mathbf{x} = \sum_{j=1}^n x_j \mathbf{a}_j$ (열의 선형결합). 이 관점은 $A$를 함수 $T_A: \mathbb{R}^n \to \mathbb{R}^m$으로 볼 때, 그 상(image)이 $\text{Col}(A)$임을 직접 보여준다.

**(b)** 행-내적 관점 ($[A\mathbf{x}]_i = \mathbf{a}_i^{(\text{row})} \cdot \mathbf{x}$)은 각 출력 원소를 독립적으로 계산하는 것이고, 열 관점은 출력을 **전체적으로** 열공간의 원소로 파악한다. 딥러닝에서 가중치 행렬 $W$의 열들은 학습된 특징(feature)을 나타내며, $W\mathbf{x}$는 입력 $\mathbf{x}$의 좌표가 이 특징들을 얼마나 활성화하는지를 보여준다. 이러한 해석은 representation learning의 핵심 직관을 제공한다.

---

## 3번. 선형 독립과 스팬

### 중학생 수준
**(a)** $\mathbf{v}_3 = \mathbf{v}_1 + \mathbf{v}_2$인지 확인해봐요... 아, 아닌데?
$\mathbf{v}_1 + \mathbf{v}_2 = (5, 7, 9)^T \neq (7, 8, 9)^T$

다른 관계를 찾아봐요: $\mathbf{v}_3 = 2\mathbf{v}_2 - \mathbf{v}_1$인지?
$2(4,5,6) - (1,2,3) = (7, 8, 9)$ ✓ 맞아요!

셋 중 하나가 나머지 둘로 만들어지므로 **선형 종속**이에요.

**(b)** 차원은 **2**. 기저는 $\{\mathbf{v}_1, \mathbf{v}_2\}$.

**(c)** $\mathbb{R}^3$를 생성하려면 차원 3이 필요한데 2밖에 안 되므로 **불가능**해요.

### 고등학생 수준
**(a)** $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + c_3\mathbf{v}_3 = \mathbf{0}$을 풀자:
$$\begin{pmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{pmatrix} \begin{pmatrix} c_1 \\ c_2 \\ c_3 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$$

행 사다리꼴로 변환:
- $R_2 \leftarrow R_2 - 2R_1$: $(0, -3, -6)$
- $R_3 \leftarrow R_3 - 3R_1$: $(0, -6, -12)$
- $R_3 \leftarrow R_3 - 2R_2$: $(0, 0, 0)$

$\begin{pmatrix} 1 & 4 & 7 \\ 0 & -3 & -6 \\ 0 & 0 & 0 \end{pmatrix}$ → 자유변수 $c_3$가 존재하므로 $\mathbf{0}$이 아닌 해가 존재 → **선형 종속**.

$c_3 = t$로 놓으면: $-3c_2 - 6t = 0 \implies c_2 = -2t$, $c_1 + 4(-2t) + 7t = 0 \implies c_1 = t$.
즉, $\mathbf{v}_1 - 2\mathbf{v}_2 + \mathbf{v}_3 = \mathbf{0}$ → $\mathbf{v}_3 = 2\mathbf{v}_2 - \mathbf{v}_1$.

**(b)** rank = 2이므로 차원 = 2. 기저: $\{\mathbf{v}_1, \mathbf{v}_2\}$ (피벗 열에 대응).

**(c)** $\dim(\text{span}) = 2 < 3 = \dim(\mathbb{R}^3)$이므로 $\mathbb{R}^3$를 생성할 수 없다.

### 대학생 수준
**(a)** 행렬 $V = [\mathbf{v}_1 | \mathbf{v}_2 | \mathbf{v}_3]$의 행렬식으로 판별:
$$\det(V) = \det\begin{pmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{pmatrix}$$
$= 1(45-48) - 4(18-24) + 7(12-15) = -3 + 24 - 21 = 0$

$\det = 0$이므로 선형 종속. 또는 RREF로 rank = 2 < 3 = 벡터 수이므로 선형 종속.

관계: $\mathbf{v}_3 = 2\mathbf{v}_2 - \mathbf{v}_1$ (등차수열 구조 — 각 성분이 1, 4, 7 / 2, 5, 8 / 3, 6, 9로 공차 3).

**(b)** $\text{rank}(V) = 2$이므로 $\dim(\text{span}) = 2$. $\{\mathbf{v}_1, \mathbf{v}_2\}$가 기저 (처음 두 열이 피벗 열). 이 스팬은 $\mathbb{R}^3$ 내의 2차원 부분공간(평면)이다.

**(c)** 기저 벡터 수 2 < $\dim(\mathbb{R}^3) = 3$이므로 전체를 생성 불가. $\text{span}$은 $\mathbb{R}^3$의 진부분공간.

---

## 4번. 기저와 차원

### 중학생 수준
**(a)** 기저란 "최소한의 재료로 모든 것을 만들 수 있는 벡터 모음"이에요.
- 조건 1: 그 벡터들로 공간의 모든 벡터를 만들 수 있음 (span)
- 조건 2: 쓸모없는 벡터가 없음 (선형 독립)

**(b)** 표현이 유일한 이유: 만약 두 가지 방법이 있다면, 빼면 0을 만드는 비자명한 조합이 생겨서 선형 독립에 모순!

### 고등학생 수준
**(a)** $V$의 기저는 다음 두 조건을 만족하는 벡터 집합 $\{v_1, \ldots, v_n\}$이다:
1. $\text{span}\{v_1, \ldots, v_n\} = V$ (생성)
2. $\{v_1, \ldots, v_n\}$이 선형 독립 (최소성)

**(b)** 유일성 증명: $v = \sum \alpha_i v_i = \sum \beta_i v_i$라 가정하면,
$$\sum (\alpha_i - \beta_i) v_i = \mathbf{0}$$
선형 독립이므로 모든 $i$에 대해 $\alpha_i - \beta_i = 0$, 즉 $\alpha_i = \beta_i$. $\square$

### 대학생 수준
**(a)** 정의: $\mathcal{B} = \{v_1, \ldots, v_n\} \subset V$가 $V$의 기저 $\iff$ $\mathcal{B}$는 $V$의 선형 독립인 생성 집합.

동치 조건: $\mathcal{B}$는 $V$의 극대 선형 독립 부분집합이다 $\iff$ $\mathcal{B}$는 $V$의 극소 생성 집합이다.

**(b)** 유일성 증명:
존재성은 생성 조건에서 따르므로, 유일성만 증명.

$v = \sum_{i=1}^n \alpha_i v_i$와 $v = \sum_{i=1}^n \beta_i v_i$를 가정하면:
$$\mathbf{0} = v - v = \sum_{i=1}^n (\alpha_i - \beta_i) v_i$$

$\{v_1, \ldots, v_n\}$이 선형 독립이므로, $\sum c_i v_i = \mathbf{0} \implies c_i = 0 \; \forall i$. 따라서 $\alpha_i - \beta_i = 0$, 즉 $\alpha_i = \beta_i$ for all $i$. $\square$

이는 좌표(coordinate) 개념의 근거가 되며, 기저를 정하면 각 벡터가 유일한 좌표 표현을 가지므로 $V \cong \mathbb{R}^n$ (동형사상).

---

## 5번. 커널, 이미지, 랭크-퇴화차수 정리

### 중학생 수준
RREF을 구해요:
$$\begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 1 & 2 & 1 & 3 \end{pmatrix} \xrightarrow{R_3 - R_1} \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 1 & 2 \end{pmatrix} \xrightarrow{R_3 - R_2} \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**(a)** 자유변수: $x_2 = s$, $x_4 = t$
- $x_3 = -2t$
- $x_1 = -2s - t$

$\ker(T) = \text{span}\left\{\begin{pmatrix}-2\\1\\0\\0\end{pmatrix}, \begin{pmatrix}-1\\0\\-2\\1\end{pmatrix}\right\}$, 차원 = **2**

**(b)** 피벗 열: 1열, 3열 → $\text{Im}(T)$의 기저 = $\left\{\begin{pmatrix}1\\0\\1\end{pmatrix}, \begin{pmatrix}0\\1\\1\end{pmatrix}\right\}$, 차원 = **2**

**(c)** $\dim(\ker) + \dim(\text{Im}) = 2 + 2 = 4$ ✓ (열의 수 = 4)

### 고등학생 수준
RREF: 위와 동일한 과정.

**(a)** $A\mathbf{x} = \mathbf{0}$을 풀면:
- $x_1 + 2x_2 + x_4 = 0$, $x_3 + 2x_4 = 0$
- 자유변수 $x_2 = s, x_4 = t$에 대해:

$$\mathbf{x} = s\begin{pmatrix}-2\\1\\0\\0\end{pmatrix} + t\begin{pmatrix}-1\\0\\-2\\1\end{pmatrix}$$

$\dim(\ker(T)) = 2$ (자유변수 수 = 열 수 - 피벗 수 = 4 - 2).

**(b)** rank = 피벗 수 = 2. 원래 행렬의 피벗에 해당하는 열이 열공간의 기저:
$$\text{Im}(T) = \text{span}\left\{\begin{pmatrix}1\\0\\1\end{pmatrix}, \begin{pmatrix}0\\1\\1\end{pmatrix}\right\}$$

**(c)** Rank-Nullity: $\text{nullity} + \text{rank} = 2 + 2 = 4 = n$ ✓

### 대학생 수준
위에 추가:
- $\ker(T)$는 $\mathbb{R}^4$의 2차원 부분공간이며, $T$에 의해 소멸되는 모든 방향을 포함
- $\text{Im}(T)$는 $\mathbb{R}^3$의 2차원 부분공간(평면)이며, $T$로 도달 가능한 모든 출력을 포함
- 선형대수 기본정리(Fundamental Theorem): $\mathbb{R}^4 = \ker(T) \oplus \text{Row}(A)$이고 $\mathbb{R}^3 = \text{Im}(T) \oplus \ker(T^*)$
- 이는 딥러닝에서 과매개변수화(overparameterized) 네트워크의 null space가 왜 존재하는지 설명함: 파라미터 공간의 차원(열 수) > 출력 제약(행 수)이면 null space가 비자명

---

## 6번. 행렬 연산의 계산 복잡도

### 중학생 수준
**(a)** $n \times n$ 행렬 두 개를 곱할 때, 결과 행렬의 각 칸을 구하려면 $n$번 곱하고 더해야 해요. 결과에 $n \times n = n^2$개 칸이 있으므로 총 $n^2 \times n = n^3$번 계산!

**(b)** 역행렬 $A^{-1}$를 구하는 건 비싸고, 여러 $\mathbf{b}$에 대해 다시 쓸 게 아니면 낭비예요. $Ax = b$를 직접 풀면 더 효율적이에요.

### 고등학생 수준
**(a)** $C = AB$에서 $C_{ij} = \sum_{k=1}^n A_{ik}B_{kj}$: 각 원소에 $n$번 곱셈+덧셈, 총 $n^2$개 원소 → $O(n^3)$.

**(b)** $A^{-1}$를 계산하면 $O(n^3)$이 들고, 곱셈 $A^{-1}b$에 $O(n^2)$이 추가됨.
LU 분해로 $Ax = b$를 직접 풀면 $O(n^3)$이지만:
- 수치적으로 더 안정적 (역행렬은 조건수가 클 때 오차가 증폭됨)
- 메모리 효율적 (n×n 역행렬을 저장하지 않아도 됨)
- 여러 $b$에 대해 반복 사용 시 LU 분해를 재사용할 수 있어 추가 비용 $O(n^2)$

### 대학생 수준
위에 추가:
- Strassen 알고리즘: $O(n^{2.807})$, 현대적 알고리즘: $O(n^{2.3728...})$ (이론적)
- 실제로는 캐시 효율, BLAS 라이브러리 최적화 등 하드웨어 고려가 중요
- 딥러닝에서: 행렬 곱셈이 forward/backward pass의 지배적 연산. GPU가 행렬 곱에 특화되어 있어 대규모 병렬 처리로 실효적 속도를 크게 향상

---

## 7번. 선형변환의 성질

### 중학생 수준
**(a)** $T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}$. 0을 곱한 것을 넣으면 0이 나와야 해요!

**(b)** 선형변환은 "덧셈과 곱셈을 보존"하므로, 여러 벡터의 조합을 입력하면 각각 변환한 것을 같은 조합으로 합칠 수 있어요.

**(c)** 기저만 알면 나머지는 자동! 모든 벡터가 기저의 조합이니까, 기저에 뭘 하는지만 알면 모든 벡터에 뭘 하는지 알 수 있어요.

### 고등학생 수준
**(a)** $T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}$ (동차성에 $c = 0$ 대입). $\square$

**(b)** 수학적 귀납법과 선형성 두 조건 사용:
$$T\left(\sum_{i=1}^k \alpha_i \mathbf{v}_i\right) = T(\alpha_1 \mathbf{v}_1 + \sum_{i=2}^k \alpha_i \mathbf{v}_i) = T(\alpha_1 \mathbf{v}_1) + T\left(\sum_{i=2}^k \alpha_i \mathbf{v}_i\right)$$
가법성을 반복 적용하고, 각 항에 동차성 적용: $T(\alpha_i \mathbf{v}_i) = \alpha_i T(\mathbf{v}_i)$.

**(c)** $V$의 기저 $\{e_1, \ldots, e_n\}$에 대해, 임의의 $\mathbf{v} = \sum \alpha_i e_i$이면:
$$T(\mathbf{v}) = \sum \alpha_i T(e_i)$$
$\alpha_i$는 $\mathbf{v}$에 의해 결정되고, $T(e_i)$는 기저에 대한 $T$의 작용이므로, $T$는 $\{T(e_1), \ldots, T(e_n)\}$에 의해 완전히 결정된다.

### 대학생 수준
**(b)** 엄밀한 증명 (수학적 귀납법):
- 기저: $k=1$. $T(\alpha_1 \mathbf{v}_1) = \alpha_1 T(\mathbf{v}_1)$ (동차성). ✓
- 귀납 가정: $k-1$개에 대해 성립.
- 귀납 단계:
$$T\left(\sum_{i=1}^k \alpha_i \mathbf{v}_i\right) = T\left(\alpha_1\mathbf{v}_1 + \sum_{i=2}^k \alpha_i\mathbf{v}_i\right) \overset{\text{add.}}{=} T(\alpha_1\mathbf{v}_1) + T\left(\sum_{i=2}^k \alpha_i\mathbf{v}_i\right) \overset{\text{ind.}}{=} \alpha_1 T(\mathbf{v}_1) + \sum_{i=2}^k \alpha_i T(\mathbf{v}_i)$$
$\square$

**(c)** 이것이 행렬 표현의 근거: $T$를 기저 $\{e_j\}$에 대해 $T(e_j) = \sum_i a_{ij} e_i$로 표현하면, 행렬 $A = (a_{ij})$가 $T$를 완전히 기술한다. 딥러닝에서 각 선형 레이어의 가중치 행렬 $W$는 바로 이 원리에 의해 선형변환을 매개변수화한다.

---

## 8번. 행렬 곱셈 순서와 결합법칙

### 중학생 수준
**(a)** $(AB)\mathbf{x}$: 먼저 $AB$를 계산 → $1000 \times 1000 \times 1000 = 10^9$번, 그 다음 결과 × $\mathbf{x}$ → $10^3$번. **총 약 $10^9$번**.

$A(B\mathbf{x})$: 먼저 $B\mathbf{x}$를 계산 → $1000 \times 1 = 1000$번, 그 다음 $A$ × 결과 → $1000 \times 1 = 1000$번. **총 약 $2000$번!**

$50만 배$ 차이!

**(b)** 레이어별로 계산하면 각 레이어에서 활성화 함수(비선형)를 적용할 수 있어요. 한 번에 $W_2 W_1$로 합치면 비선형 활성화를 넣을 수 없고, 그냥 하나의 선형변환이 돼 버려요.

### 고등학생 수준
**(a)**
- $(AB)\mathbf{x}$: $AB$ 계산에 $m \cdot n \cdot p = 10^6$ FLOPs, $(AB)\mathbf{x}$에 $m \cdot p = 10^3$. 총: $\sim 10^6$.
  - 오류 정정: $p=1$이므로 $B\mathbf{x}$는 $n \times 1$ 벡터. $AB$는 $1000 \times 1000$ 행렬에 $1000 \times 1000$ 행렬을 곱하므로 $O(n^3) = 10^9$.
- $A(B\mathbf{x})$: $B\mathbf{x}$에 $n \cdot p = 1000$, $A \cdot \text{result}$에 $m \cdot n = 10^6$. 총: $\sim 10^6$.
  - 정정: $B\mathbf{x}$는 $1000 \times 1$ → $n$ 곱셈. $A \cdot (n \times 1)$ → $mn$ 곱셈. 총 $n + mn = 1000 + 10^6 \approx 10^6$.

실제로는 $A(B\mathbf{x})$가 압도적으로 효율적: 행렬-벡터 곱($O(mn)$)을 두 번 vs 행렬-행렬 곱($O(mn^2)$).

**(b)** 비선형 활성화 때문: $\sigma(W_2(W_1\mathbf{x})) \neq \sigma((W_2W_1)\mathbf{x})$가 아니라, 실제로는 $W_2 \sigma(W_1 \mathbf{x})$. 중간에 비선형 활성화가 있으므로 $W_2 W_1$로 합칠 수 없음.

### 대학생 수준
**(a)** 행렬 체인 곱셈(Matrix Chain Multiplication) 문제의 예시. 동적 프로그래밍으로 최적 괄호화를 찾을 수 있으며, 이 경우 right-to-left 순서가 최적.

**(b)** 깊은 신경망에서 $f(\mathbf{x}) = W_L \sigma(\cdots W_2 \sigma(W_1 \mathbf{x})\cdots)$:
1. 비선형성: $\sigma$가 사이에 있으므로 행렬을 합칠 수 없음
2. 중간 활성화 저장: 역전파를 위해 각 레이어의 중간 결과($\mathbf{h}_l$)를 저장해야 함
3. 표현력: 비선형 없이 $W_L \cdots W_1$은 단일 선형변환과 동일 (Universal Approximation Theorem의 필요조건)

---

# 파트 2: 고유값 및 행렬 분해 (Q9–Q16)

---

## 9번. 고유값 계산

### 중학생 수준
**(a)** 특성방정식: $\det(A - \lambda I) = 0$
$$(4-\lambda)(3-\lambda) - 2 \times 1 = 0$$
$$12 - 7\lambda + \lambda^2 - 2 = 0$$
$$\lambda^2 - 7\lambda + 10 = 0$$
$$(\lambda - 5)(\lambda - 2) = 0$$
$$\lambda_1 = 5, \quad \lambda_2 = 2$$

**(b)** $\lambda = 5$: $(A - 5I)\mathbf{v} = 0$ → $\begin{pmatrix}-1&2\\1&-2\end{pmatrix}\mathbf{v}=0$ → $\mathbf{v}_1 = \begin{pmatrix}2\\1\end{pmatrix}$

$\lambda = 2$: $(A - 2I)\mathbf{v} = 0$ → $\begin{pmatrix}2&2\\1&1\end{pmatrix}\mathbf{v}=0$ → $\mathbf{v}_2 = \begin{pmatrix}1\\-1\end{pmatrix}$

**(c)** 검증: $A\begin{pmatrix}2\\1\end{pmatrix} = \begin{pmatrix}10\\5\end{pmatrix} = 5\begin{pmatrix}2\\1\end{pmatrix}$ ✓

### 고등학생 수준
위와 동일한 계산에 추가 관찰:
- $\text{tr}(A) = 4 + 3 = 7 = 5 + 2 = \lambda_1 + \lambda_2$ ✓
- $\det(A) = 12 - 2 = 10 = 5 \times 2 = \lambda_1 \lambda_2$ ✓
- 이 두 성질은 특성다항식 $\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$에서 비에타 공식으로 따라온다.

### 대학생 수준
위에 추가: $A$는 대칭이 아니므로 고유벡터가 직교할 필요는 없다. 실제로 $\mathbf{v}_1 \cdot \mathbf{v}_2 = 2 - 1 = 1 \neq 0$. 만약 $A$가 대칭이었다면 스펙트럼 정리에 의해 고유벡터가 직교 기저를 형성했을 것이다.

---

## 10번. 고유값의 성질과 변환

### 중학생 수준
$Av = \lambda v$에서 출발해요:

**(a)** $A^3 v = A(A(Av)) = A(A(\lambda v)) = A(\lambda^2 v) = \lambda^3 v$
고유값: $4^3=64$, $1^3=1$, $(-2)^3=-8$

**(b)** $(A-3I)v = Av - 3v = \lambda v - 3v = (\lambda-3)v$
고유값: $4-3=1$, $1-3=-2$, $-2-3=-5$

**(c)** 모든 고유값 $\neq 0$ ($4, 1, -2$ 모두 0이 아님) → $\det(A) = 4 \times 1 \times (-2) = -8 \neq 0$ → 역행렬 존재.
$A^{-1}$의 고유값: $1/4$, $1$, $-1/2$

**(d)** $\det(A) = \lambda_1 \lambda_2 \lambda_3 = 4 \times 1 \times (-2) = -8$
$\text{tr}(A) = \lambda_1 + \lambda_2 + \lambda_3 = 4 + 1 + (-2) = 3$

### 고등학생 수준
핵심 원리: $Av = \lambda v$이면 $f(A)v = f(\lambda)v$ (다항식 $f$에 대해).

| 변환 | $f(\lambda)$ | $\lambda = 4$ | $\lambda = 1$ | $\lambda = -2$ |
|---|---|---|---|---|
| $A^3$ | $\lambda^3$ | 64 | 1 | -8 |
| $A - 3I$ | $\lambda - 3$ | 1 | -2 | -5 |
| $A^{-1}$ | $1/\lambda$ | 1/4 | 1 | -1/2 |

**(d)** 특성다항식의 근과 계수의 관계:
- $\det(A) = \prod \lambda_i = -8$ (상수항의 부호 포함)
- $\text{tr}(A) = \sum \lambda_i = 3$ (최고차 다음 계수)

### 대학생 수준
위에 추가:
- $A$가 대각화 가능하면 $A = PDP^{-1}$이고 $f(A) = Pf(D)P^{-1}$. 이것이 "고유값 매핑 법칙"의 근거.
- $A^{-1}$이 존재하려면 $\det(A) \neq 0 \iff$ 모든 $\lambda_i \neq 0$ (특성다항식에서 $\lambda = 0$이 근이 아님).
- 행렬식의 기하학적 의미: $|\det(A)| = 8$은 $A$에 의한 선형변환이 부피를 8배 확대하고 방향을 뒤집음(음수).

---

## 11번. 대각화

### 중학생 수준
**(a)** $\det(A - \lambda I) = (3-\lambda)(2-\lambda) = 0$ → $\lambda_1 = 3, \lambda_2 = 2$

$\lambda=3$: $\begin{pmatrix}0&1\\0&-1\end{pmatrix}v=0$ → $v_2 = 0$ → $\mathbf{v}_1 = \begin{pmatrix}1\\0\end{pmatrix}$

$\lambda=2$: $\begin{pmatrix}1&1\\0&0\end{pmatrix}v=0$ → $v_1 = -v_2$ → $\mathbf{v}_2 = \begin{pmatrix}-1\\1\end{pmatrix}$

**(b)** $P = \begin{pmatrix}1&-1\\0&1\end{pmatrix}$, $D = \begin{pmatrix}3&0\\0&2\end{pmatrix}$

**(c)** $A^{100} = PD^{100}P^{-1}$, $D^{100} = \begin{pmatrix}3^{100}&0\\0&2^{100}\end{pmatrix}$

대각행렬의 거듭제곱은 각 대각 원소를 거듭제곱하면 끝이에요!

### 고등학생 수준
**(a)-(b)** 위와 동일.

$P^{-1} = \begin{pmatrix}1&1\\0&1\end{pmatrix}$ (2×2 역행렬 공식: $\frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}$)

**(c)** $A^{100} = PD^{100}P^{-1} = \begin{pmatrix}1&-1\\0&1\end{pmatrix}\begin{pmatrix}3^{100}&0\\0&2^{100}\end{pmatrix}\begin{pmatrix}1&1\\0&1\end{pmatrix}$

$$= \begin{pmatrix}3^{100}&-2^{100}\\0&2^{100}\end{pmatrix}\begin{pmatrix}1&1\\0&1\end{pmatrix} = \begin{pmatrix}3^{100}&3^{100}-2^{100}\\0&2^{100}\end{pmatrix}$$

대각화의 핵심 이점: $A^n$을 직접 계산하면 $n-1$번의 행렬 곱이 필요하지만, 대각화하면 대각 원소의 거듭제곱만으로 $O(1)$에 해결.

### 대학생 수준
위에 추가: 대각화 가능 조건은 기하적 중복도 = 대수적 중복도. $A$는 서로 다른 고유값 2개를 가지므로 항상 대각화 가능 (서로 다른 고유값의 고유벡터는 선형 독립).

이산 동역학계 $\mathbf{x}_{k+1} = A\mathbf{x}_k$에서 $\mathbf{x}_k = A^k \mathbf{x}_0 = PD^kP^{-1}\mathbf{x}_0$. 최대 고유값 $|\lambda_1| = 3 > 1$이므로 $\mathbf{v}_1$ 방향으로 지수적 성장, 딥러닝에서 gradient explosion/vanishing의 수학적 원리와 직결.

---

## 12번~16번은 핵심 풀이만 간략히 제시합니다.

---

## 12번. 대칭행렬과 스펙트럼 정리

### 중학생: 대칭행렬은 특별한 행렬로, 고유값이 항상 실수이고 고유벡터가 서로 직교해요.

### 고등학생:
**(a)** 스펙트럼 정리: 실수 대칭행렬 $A = A^T$는 $A = Q\Lambda Q^T$로 분해됨. $Q$는 직교행렬 ($Q^TQ = I$), $\Lambda$는 실수 대각행렬.

**(b)** $Av = \lambda v$이면 $\bar{v}^T A v = \lambda \bar{v}^T v = \lambda \|v\|^2$. 또한 $\bar{v}^T A v = \bar{v}^T A^T v = (A\bar{v})^T v = \overline{(\bar{\lambda}\bar{v})}^T v = \bar{\lambda}\bar{v}^Tv = \bar{\lambda}\|v\|^2$. 따라서 $\lambda = \bar{\lambda}$ → $\lambda \in \mathbb{R}$.

**(c)** PCA에서 공분산 행렬 $\Sigma = \frac{1}{N}X^TX$는 대칭 PSD → 고유값 ≥ 0이고 고유벡터가 직교 → 주성분이 직교 방향으로 분산을 최대화.

### 대학생: 위에 추가로, 스펙트럼 정리는 대칭행렬에 대한 SVD와 고유분해가 일치함을 보여준다: $A = Q\Lambda Q^T$에서 $U = V = Q$, $\Sigma = |\Lambda|$.

---

## 13번. SVD 개념

### 중학생: SVD는 아무 행렬이든 "회전 → 늘리기 → 회전"으로 분해하는 거예요.

### 고등학생:
**(a)** $A = U\Sigma V^T$: $U \in \mathbb{R}^{m \times m}$ (직교), $\Sigma \in \mathbb{R}^{m \times n}$ (대각), $V \in \mathbb{R}^{n \times n}$ (직교). $U$의 열 = 좌특이벡터, $V$의 열 = 우특이벡터, $\Sigma$의 대각 = 특이값 ($\sigma_i \geq 0$).

**(b)** $A^TA = V\Sigma^T U^T U\Sigma V^T = V\Sigma^T\Sigma V^T = V\text{diag}(\sigma_i^2)V^T$ → $A^TA$의 고유값 = $\sigma_i^2$.

**(c)** $A^TA$의 고유값: $25, 4$. $\text{rank}(A) = 2$ (0이 아닌 특이값의 수).

### 대학생: SVD는 선형대수의 "궁극의 분해"로, 모든 행렬에 대해 존재하며 (정사각/비정사각, 대칭/비대칭 무관), 행렬의 기하학적 작용을 완전히 기술한다.

---

## 14번. SVD를 이용한 저랭크 근사

### 중학생: 중요한 성분만 남기고 나머지를 버려서 데이터를 압축해요.

### 고등학생:
**(a)** Eckart-Young: $\|A - A_k\|_F$를 최소화하는 랭크-$k$ 행렬은 $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.

**(b)** 전체 에너지: $10^2 + 5^2 + 1^2 + 0.1^2 = 100 + 25 + 1 + 0.01 = 126.01$
랭크-2 에너지: $100 + 25 = 125$
비율: $125/126.01 \approx 99.2\%$

**(c)** PCA에서 데이터 행렬을 SVD하여 상위 $k$개 성분만 유지하면, 분산의 대부분을 보존하면서 차원을 $n \to k$로 줄일 수 있다.

### 대학생: Eckart-Young 정리는 단위 연산자 노름(spectral norm)과 Frobenius 노름 모두에서 성립. $\|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$. 딥러닝에서 가중치 행렬의 저랭크 근사는 모델 압축(LoRA 등)의 이론적 기반.

---

## 15번. PageRank

### 중학생: 웹 페이지의 중요도를 고유벡터로 구해요. "이 행렬의 고유값 1에 해당하는 고유벡터가 바로 각 페이지의 점수!"

### 고등학생:
**(a)** $\mathbf{r} = M\mathbf{r}$은 $M\mathbf{r} = 1 \cdot \mathbf{r}$과 같으므로, $\mathbf{r}$은 $M$의 고유값 $\lambda = 1$에 대한 고유벡터.

**(b)** 확률행렬(각 열의 합 = 1)에 대해 Perron-Frobenius 정리에 의해 최대 고유값이 1이고, 양수 고유벡터가 존재.

### 대학생: 엄밀히는 damping factor $\alpha$를 도입하여 $M' = \alpha M + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T$로 변형, 이를 통해 비주기성과 비가약성을 보장하여 유일한 정상 분포가 존재하게 함.

---

## 16번. 양반정치 행렬

### 중학생: PSD는 "에너지가 항상 0 이상"인 행렬이에요.

### 고등학생:
**(a)** $A$가 PSD $\iff$ 모든 $\mathbf{x}$에 대해 $\mathbf{x}^T A \mathbf{x} \geq 0$.

**(b)** $\mathbf{x}^T(B^TB)\mathbf{x} = (B\mathbf{x})^T(B\mathbf{x}) = \|B\mathbf{x}\|^2 \geq 0$. $\square$

**(c)** 공분산 행렬이 PSD이면 고유값 $\geq 0$ → 분산이 음수가 될 수 없음 → 통계적으로 의미 있는 분해 보장.

### 대학생: $B^TB$는 Gram 행렬의 특수한 경우. PSD 조건은 볼록 최적화에서 2차 형식이 아래로 볼록(convex)함을 보장하며, 이는 유일한 최솟값의 존재를 의미.

---

# 파트 3: 미분과 최적화 (Q17–Q24)

---

## 17번. 선형 근사와 뉴턴 방법

### 중학생 수준
**(a)** $f(x) \approx f(a) + f'(a)(x - a)$ — 곡선을 접선으로 대체!

**(b)** $f(x) = \sqrt{x}$, $a = 4$에서:
- $f(4) = 2$, $f'(x) = \frac{1}{2\sqrt{x}}$, $f'(4) = \frac{1}{4}$
- $\sqrt{7} \approx 2 + \frac{1}{4}(7 - 4) = 2 + 0.75 = 2.75$

(실제 값: $\sqrt{7} \approx 2.6458$... 좀 오차가 있지만 근사!)

**(c)** $g(x) = x^2 - 7$, $g'(x) = 2x$
$$x_1 = x_0 - \frac{g(x_0)}{g'(x_0)} = 3 - \frac{9-7}{6} = 3 - \frac{1}{3} = \frac{8}{3} \approx 2.667$$

**(d)** 함수 위의 점에서 접선을 그리고, 접선이 x축과 만나는 점이 다음 추정값이에요.

### 고등학생 수준
**(a)** Taylor 1차 근사: $f(x) \approx f(a) + f'(a)(x-a)$

**(b)** $a = 9$에서: $f(9) = 3$, $f'(9) = 1/6$
$\sqrt{7} \approx 3 + \frac{1}{6}(7-9) = 3 - \frac{1}{3} = \frac{8}{3} \approx 2.667$

$a = 4$에서: $f(4) = 2$, $f'(4) = 1/4$
$\sqrt{7} \approx 2 + \frac{1}{4}(7-4) = 2.75$

$a = 9$가 더 가깝게 근사 (실제 값 2.6458, $a=9$: 2.667, $a=4$: 2.75)

**(c)** Newton's method: $x_{n+1} = x_n - f(x_n)/f'(x_n)$
$x_1 = 3 - (9-7)/(2 \times 3) = 3 - 1/3 = 8/3 \approx 2.6667$

**(d)** 기하적 해석: 현재 점 $(x_0, g(x_0))$에서의 접선 $y = g'(x_0)(x-x_0) + g(x_0)$의 x절편을 다음 근사값으로 사용. 함수가 접선에 가깝다면 (곡률이 작다면) 빠르게 수렴.

### 대학생 수준
**(b)** 근사 오차: $|f(x) - L(x)| = |f''(c)| \cdot (x-a)^2/2$ (Taylor 나머지 정리). $|x-a|$가 작을수록 정확. $a=9$: $|7-9|=2$, $a=4$: $|7-4|=3$이므로 $a=9$가 더 나은 근사점.

**(c)** Newton's method의 수렴 속도: 이차 수렴(quadratic convergence). $|x_{n+1} - r| \leq C|x_n - r|^2$. 즉, 유효 자릿수가 매 반복마다 2배. 이는 1차 수렴(선형 수렴)인 이분법이나 고정점 반복보다 훨씬 빠름.

조건: $f'(r) \neq 0$ (단근), 초기값이 근에 충분히 가까울 것. $f'(r) = 0$ (중근)이면 수렴 속도가 선형으로 저하.

---

## 18번. 그래디언트

### 중학생 수준
각 변수로 미분하면 돼요:
**(a)** $\nabla f = \begin{pmatrix} 2x_1 + 3x_2 \\ 3x_1 \\ -2x_3 + 2 \end{pmatrix}$

**(b)** $(1, -1, 2)$에서: $\nabla f = \begin{pmatrix} 2-3 \\ 3 \\ -4+2 \end{pmatrix} = \begin{pmatrix} -1 \\ 3 \\ -2 \end{pmatrix}$

**(c)** 가장 빠르게 증가하는 방향: 그래디언트 방향 $(-1, 3, -2)$, 증가율: $\|\nabla f\| = \sqrt{1+9+4} = \sqrt{14}$

**(d)** 그래디언트는 가장 빠르게 올라가는 방향이므로, 반대 방향($-\nabla f$)이 가장 빠르게 내려가는 방향!

### 고등학생 수준
**(a)** $\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3}\right)^T$
- $\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2$
- $\frac{\partial f}{\partial x_2} = 3x_1$
- $\frac{\partial f}{\partial x_3} = -2x_3 + 2$

$\nabla f = (2x_1 + 3x_2, \; 3x_1, \; -2x_3 + 2)^T$

**(b)** $\nabla f(1,-1,2) = (-1, 3, -2)^T$

**(c)** 방향도함수: $D_\mathbf{u}f = \nabla f \cdot \mathbf{u}$. $\|\mathbf{u}\|=1$일 때 Cauchy-Schwarz에 의해 $D_\mathbf{u}f \leq \|\nabla f\|$이고, 등호는 $\mathbf{u} = \nabla f / \|\nabla f\|$일 때.

최대 증가 방향: $\frac{(-1,3,-2)^T}{\sqrt{14}}$, 증가율: $\sqrt{14} \approx 3.742$

**(d)** $f(\mathbf{x} + \eta\mathbf{d}) \approx f(\mathbf{x}) + \eta\nabla f^T\mathbf{d}$. 이를 최소화하려면 $\nabla f^T\mathbf{d}$를 최소화, $\|\mathbf{d}\|=1$에서 $\mathbf{d} = -\nabla f/\|\nabla f\|$일 때 최소.

### 대학생 수준
위에 추가: 그래디언트는 등고면(level set) $\{x : f(x) = c\}$에 수직인 벡터이며, 이는 라그랑주 승수법의 기하학적 기초. 또한 그래디언트의 존재는 $f$의 프레셰 미분(Fréchet derivative)의 리스 표현(Riesz representation)이다: $df(\mathbf{x})[\mathbf{h}] = \langle \nabla f(\mathbf{x}), \mathbf{h} \rangle$.

---

## 19번. 자코비안 행렬

### 중학생 수준
**(a)** 각 출력을 각 입력으로 미분해서 표로 정리해요:

$$J = \begin{pmatrix} 2x_1 & 1 \\ x_2 & x_1 \\ e^{x_1} & 0 \end{pmatrix}$$
크기: $3 \times 2$ (출력 3개, 입력 2개)

**(b)** $(0, 1)$에서: $J = \begin{pmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 0 \end{pmatrix}$

**(c)** $\mathbf{f}(0,1) = (1, 0, 1)^T$, $\Delta\mathbf{x} = (0.1, 0.05)^T$
$$\mathbf{f}(0.1, 1.05) \approx \begin{pmatrix}1\\0\\1\end{pmatrix} + \begin{pmatrix}0&1\\1&0\\1&0\end{pmatrix}\begin{pmatrix}0.1\\0.05\end{pmatrix} = \begin{pmatrix}1\\0\\1\end{pmatrix} + \begin{pmatrix}0.05\\0.1\\0.1\end{pmatrix} = \begin{pmatrix}1.05\\0.1\\1.1\end{pmatrix}$$

### 고등학생 수준
위와 동일. 추가:

**(d)** 역전파에서 합성함수 $\mathcal{L} = L \circ f_n \circ \cdots \circ f_1$의 그래디언트는 연쇄 법칙에 의해:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J_{f_1}^T J_{f_2}^T \cdots J_{f_n}^T \nabla_{\mathbf{y}} L$$
각 레이어의 자코비안을 역순으로 곱하여 그래디언트를 전파. 이것이 "역전파" 이름의 이유.

### 대학생 수준
자코비안의 크기 분석: $J \in \mathbb{R}^{m \times n}$에서 $m > n$이면 차원이 확장되고, $m < n$이면 축소됨. 이는 신경망 레이어의 설계(확장/축소)와 직결. 특이값 분해 $J = U\Sigma V^T$의 특이값이 1에서 크게 벗어나면 gradient exploding/vanishing 발생.

---

## 20번. 벡터 함수의 연쇄 법칙

### 중학생 수준
**(a)** $\nabla_\mathbf{z} h = (2z_1, 2z_2, 1)^T$

**(b)** $J_\mathbf{g} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \\ x_2 & x_1 \end{pmatrix}$

**(c)** $\nabla_\mathbf{x} f = J_\mathbf{g}^T \nabla_\mathbf{z} h = \begin{pmatrix}1&1&x_2\\1&-1&x_1\end{pmatrix}\begin{pmatrix}2z_1\\2z_2\\1\end{pmatrix}$

$z_1 = x_1+x_2$, $z_2 = x_1-x_2$, $z_3 = x_1x_2$를 대입하면:
$$= \begin{pmatrix}2(x_1+x_2) + 2(x_1-x_2) + x_2 \\ 2(x_1+x_2) - 2(x_1-x_2) + x_1\end{pmatrix} = \begin{pmatrix}4x_1 + x_2 \\ 4x_2 + x_1\end{pmatrix}$$

**(d)** 검증: $f(\mathbf{x}) = (x_1+x_2)^2 + (x_1-x_2)^2 + x_1x_2 = 2x_1^2 + 2x_2^2 + x_1x_2$
$\nabla f = (4x_1 + x_2, \; 4x_2 + x_1)^T$ ✓ 일치!

### 고등학생/대학생: 위와 동일하되, 연쇄 법칙의 행렬 형태 $\nabla_\mathbf{x} f = J_\mathbf{g}^T \nabla_\mathbf{z} h$가 역전파의 수학적 기초임을 강조. 이는 $\frac{df}{d\mathbf{x}} = \frac{df}{d\mathbf{z}} \cdot \frac{d\mathbf{z}}{d\mathbf{x}}$의 행렬 버전.

---

## 21번. 벡터-스칼라 미분

### 중학생: **(a)** $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = A$ (행렬 그대로!)
**(b)** $f = x_1^2 + \cdots + x_n^2$ → $\frac{\partial f}{\partial x_i} = 2x_i$ → $\nabla f = 2\mathbf{x}$
**(c)** $f = \sum_{ij} a_{ij}x_ix_j$, $A$가 대칭이면 $\nabla f = 2A\mathbf{x}$

### 고등학생:
**(b)** $f = \mathbf{x}^T\mathbf{x} = \sum x_i^2$. $\frac{\partial f}{\partial x_k} = 2x_k$. 벡터로: $\nabla f = 2\mathbf{x}$.

**(c)** $f = \mathbf{x}^T A\mathbf{x} = \sum_{i,j} a_{ij}x_ix_j$
$\frac{\partial f}{\partial x_k} = \sum_j a_{kj}x_j + \sum_i a_{ik}x_i = [A\mathbf{x}]_k + [A^T\mathbf{x}]_k$
$A = A^T$이면: $\nabla f = 2A\mathbf{x}$

### 대학생: 일반적으로 $\nabla_\mathbf{x}(\mathbf{x}^TA\mathbf{x}) = (A + A^T)\mathbf{x}$. $A$가 대칭이면 $2A\mathbf{x}$. 이는 2차 형식의 미분으로, 최적화에서 뉴턴 방법의 기초.

---

## 22번. 소프트맥스 함수와 그 도함수

### 중학생 수준
**(a)** 분자가 전부 양수($e^{z_i} > 0$)이고 분모가 그 합이니까, 각 값이 양수이고 합이 1이에요.

**(b)** $i = j$: $\frac{\partial \sigma_i}{\partial z_i} = \sigma_i(1 - \sigma_i)$ (시그모이드랑 비슷!)
$i \neq j$: $\frac{\partial \sigma_i}{\partial z_j} = -\sigma_i \sigma_j$ (마이너스!)

### 고등학생 수준
**(d)** 유도 ($i = j$ 경우):
$$\sigma_i = \frac{e^{z_i}}{S}, \quad S = \sum_{k} e^{z_k}$$

몫의 미분법: $\frac{d}{dx}\frac{u}{v} = \frac{u'v - uv'}{v^2}$

$$\frac{\partial \sigma_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} \cdot \frac{S - e^{z_i}}{S} = \sigma_i(1 - \sigma_i)$$

$i \neq j$ 경우: 분자에 $z_j$가 없으므로 $u' = 0$:
$$\frac{\partial \sigma_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S}\frac{e^{z_j}}{S} = -\sigma_i\sigma_j$$

**(c)** 행렬 형태: $\frac{\partial \boldsymbol{\sigma}}{\partial \mathbf{z}} = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$

### 대학생: 이 자코비안은 $K \times K$ 행렬이며, $\text{rank} = K-1$ (softmax 출력의 합이 1이라는 제약에 의해). 이는 softmax가 $K$차원을 $K-1$차원 심플렉스로 사상하기 때문.

---

## 23번. 활성화 함수

### 중학생: Sigmoid는 S자 곡선, ReLU는 음수는 0 양수는 그대로, Tanh는 -1~1 사이 S자 곡선.

### 고등학생:
**(a)** $\sigma(x) = (1+e^{-x})^{-1}$
$\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \cdot (1-\sigma(x))$

**(b)** $f'(x) = \begin{cases} 0 & x < 0 \\ 1 & x > 0 \\ \text{undefined} & x = 0 \end{cases}$
Dying ReLU: $x < 0$이면 gradient = 0 → 한 번 음수 영역에 빠지면 영원히 업데이트 안 됨.

**(c)** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. 몫의 미분법 적용:
$$\tanh'(x) = \frac{(e^x+e^{-x})^2 - (e^x-e^{-x})^2}{(e^x+e^{-x})^2} = 1 - \tanh^2(x)$$
(분자: $(a+b)^2 - (a-b)^2 = 4ab = 4$, 분모: $(a+b)^2$, $a=e^x, b=e^{-x}$)

### 대학생: Sigmoid/Tanh의 gradient는 최대 0.25/1로 제한 → 깊은 네트워크에서 gradient vanishing. ReLU는 양수 영역에서 gradient = 1로 일정하여 이 문제를 완화. Leaky ReLU ($f(x) = \max(\alpha x, x)$, $\alpha \ll 1$)는 dying ReLU 해결.

---

## 24번. 어텐션 메커니즘

### 중학생: Q는 "뭘 찾을까", K는 "나한테 뭐가 있어", V는 "실제 정보". Q와 K를 비교해서 관련된 V를 골라 읽는 거예요.

### 고등학생:
**(a)** $QK^T \in \mathbb{R}^{n \times m}$, softmax 출력: $\mathbb{R}^{n \times m}$ (행별 확률 분포), 최종: $\mathbb{R}^{n \times d_v}$.

**(b)** $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$, 각 원소가 i.i.d. 평균 0, 분산 1이면:
$\text{Var}(\mathbf{q}^T\mathbf{k}) = \text{Var}(\sum q_ik_i) = d_k$ (독립이므로 분산의 합)
$\sqrt{d_k}$로 나누면 분산이 1로 정규화됨.

**(c)** Softmax의 행별 연산: 각 query에 대해 모든 key와의 유사도를 확률 분포로 변환. 이는 value의 가중 평균에서의 가중치를 결정하며, 확률적으로 "어디에 주의를 기울일 것인가"에 대한 soft assignment.

### 대학생: 스케일링 없으면 $d_k = 512$일 때 $\text{Std}(\mathbf{q}^T\mathbf{k}) \approx 22.6$. Softmax 입력이 $|\cdot| > 20$이면 출력이 거의 one-hot → gradient $\approx 0$. $1/\sqrt{d_k}$ 스케일링으로 분산을 1로 유지하면 softmax가 부드러운 분포를 형성하여 안정적 학습 가능.

---

# 파트 4: 딥러닝 연결과 응용 (Q25–Q30)

---

## 25번. 경사하강법 유도

### 중학생: 산에서 가장 가파른 내리막길로 내려가는 것! 그래디언트의 반대 방향이 가장 빠른 하강 방향.

### 고등학생:
**(a)** $f(\mathbf{x} + \eta\mathbf{d}) \approx f(\mathbf{x}) + \eta\nabla f^T\mathbf{d}$
$\|\mathbf{d}\| = 1$일 때 $\nabla f^T\mathbf{d}$를 최소화: Cauchy-Schwarz에 의해 $\nabla f^T\mathbf{d} \geq -\|\nabla f\|$, 등호는 $\mathbf{d} = -\nabla f/\|\nabla f\|$. → 최급강하 방향은 $-\nabla f$.

**(b)** $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$. $\eta$: 학습률 (step size). 한 번에 얼마나 이동할지 결정.

**(c)** $\eta$ 과대 → 발산 (overshooting), $\eta$ 과소 → 수렴 느림. 최적: $\eta < 2/\lambda_{\max}(H)$ ($H$: 헤시안의 최대 고유값).

### 대학생: 2차 Taylor 전개에서 $f(\mathbf{x}+\Delta) \approx f + \nabla f^T\Delta + \frac{1}{2}\Delta^T H\Delta$. $\eta$가 $1/\lambda_{\max}$보다 크면 2차 항이 지배하여 함수값이 오히려 증가. "Edge of Stability" (이성윤 교수 연구 분야): GD가 $\lambda_{\max} \approx 2/\eta$ 부근에서 안정화되는 현상.

---

## 26번~30번은 핵심만 간략히:

---

## 26번. 고차원 뉴턴 방법

### 고등학생: **(a)** $H = \nabla^2 f$ = 그래디언트의 자코비안 = 2차 편미분 행렬. **(b)** 뉴턴: 2차 근사로 quadratic convergence 달성. 비용: 헤시안 계산 $O(n^2)$ + 역행렬 $O(n^3)$ → 대규모 DL에서는 비현실적 → Adam, LBFGS 등 근사 사용.

---

## 27번. 교차 엔트로피 손실의 그래디언트

### 고등학생:
**(a)** $\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}$

$\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$ (Q22에서)

연쇄: $\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \frac{\partial \mathcal{L}}{\partial \hat{y}_i}\frac{\partial \hat{y}_i}{\partial z_j} = -\sum_i \frac{y_i}{\hat{y}_i} \hat{y}_i(\delta_{ij} - \hat{y}_j) = -\sum_i y_i(\delta_{ij} - \hat{y}_j)$

$= -y_j + \hat{y}_j\sum_i y_i = -y_j + \hat{y}_j \cdot 1 = \hat{y}_j - y_j$

벡터로: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$ $\square$

**(b)** "예측 - 정답"이라는 단순한 형태로, 복잡한 softmax 미분과 log 미분이 상쇄되어 계산이 매우 효율적. 역전파에서 추가적인 행렬 연산 없이 바로 이전 레이어로 그래디언트 전파 가능.

---

## 28번. SVD 데이터 압축

### 고등학생:
**(a)** $1000 \times 800 = 800{,}000$개

**(b)** 각 $\sigma_i\mathbf{u}_i\mathbf{v}_i^T$에 $1 + 1000 + 800 = 1801$개 필요. 총: $1801k$개.

**(c)** $1801k < 800{,}000 \implies k < 444.2 \implies k \leq 444$.
$k=444$일 때 압축률: $1801 \times 444 / 800{,}000 = 799{,}644/800{,}000 \approx 99.96\%$ (거의 압축 안 됨)
실제로는 $k \ll 444$일 때만 유의미한 압축. 예: $k = 50$이면 $90{,}050/800{,}000 \approx 11.3\%$ (약 9배 압축).

**(d)** $k$가 작을수록 압축률은 좋지만, 버려진 $\sum_{i>k}\sigma_i^2$ 만큼의 정보 손실. 특이값이 빠르게 감쇠하는 이미지(매끄러운 영역 많은)에서 효과적.

---

## 29번. 학습 동역학에서의 고유값

### 고등학생: **(a)** MSE의 그래디언트: $\nabla L = X^TX\mathbf{w} - X^T\mathbf{y}$. GD 업데이트: $\mathbf{w} \leftarrow \mathbf{w} - \eta(X^TX\mathbf{w} - X^T\mathbf{y})$. $X^TX$의 고유값 방향으로 분해하면 각 방향의 수렴 속도가 $(1 - \eta\lambda_i)$의 거듭제곱. $\lambda_i$가 크면 빨리 수렴, 작으면 느리게.

**(b)** $\kappa = \lambda_{\max}/\lambda_{\min}$. $\kappa$가 크면 가장 빠른 방향과 느린 방향의 속도 차이가 커서, 하나에 맞추면 다른 것이 발산하거나 극도로 느려짐. 이를 "ill-conditioned" 문제라 함.

---

## 30번. 종합: 선형대수에서 역전파까지

### 고등학생:
$\delta = \hat{y} - y$ (스칼라 오차)

**(a)** $\frac{\partial \mathcal{L}}{\partial b_2} = \delta$, $\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = \delta \mathbf{h}$

**(b)** $\frac{\partial \mathcal{L}}{\partial \mathbf{h}} = \delta \mathbf{w}_2$

**(c)** $\mathbf{z}_1 = W_1\mathbf{x} + \mathbf{b}_1$이면, $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \odot \sigma'(\mathbf{z}_1) = \delta \mathbf{w}_2 \odot \sigma'(\mathbf{z}_1)$

$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} \mathbf{x}^T$ (외적!)
$\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}$

**(d)**
- **행렬 곱셈**: Forward pass($W_1\mathbf{x}$, $\mathbf{w}_2^T\mathbf{h}$)와 gradient 계산($\delta\mathbf{h}$, $\delta_1\mathbf{x}^T$)
- **자코비안**: $\sigma$의 원소별 미분 → 대각 자코비안 $\text{diag}(\sigma'(\mathbf{z}_1))$
- **연쇄 법칙**: 손실 → 출력 → 은닉층 → 입력층으로 gradient를 역방향 전파

역전파는 결국 **연쇄 법칙의 행렬 버전을 역순으로 적용**하는 것이며, 각 단계에서 선형대수의 기본 연산(행렬-벡터 곱, 외적, 원소별 곱)이 등장한다.

### 대학생: 위에 추가로, 역전파의 계산 복잡도는 forward pass와 동일한 $O(nd + d)$이며, 이는 자동 미분(automatic differentiation)의 reverse mode에 해당한다. Forward mode는 입력 변수 수에 비례하는 비용이 들지만, reverse mode는 출력 변수(=1, 스칼라 손실) 수에 비례하므로 딥러닝에서 압도적으로 효율적.

이성윤 교수님의 연구 주제인 implicit bias of SGD, gradient flow dynamics 등은 바로 이 역전파 과정에서의 그래디언트 특성을 분석하는 것이다.

---

**풀이 종료**

*핵심: 모든 문제에서 "왜?"를 설명할 수 있어야 합니다. 공식을 외우는 것이 아니라, 정의에서 출발하여 논리적으로 유도하는 과정이 중요합니다.*

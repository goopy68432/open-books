---
title: 'Quiz 2 — Four Fundamental Subspaces'
description: 'R(A), R(Aᵀ), N(A), N(Aᵀ) 네 부분공간'
draft: false
---

## 0. 한 줄 요약

$A=(1\ 2)\in\mathbb{R}^{1\times 2}$ ($\mathbb{R}^2\to\mathbb{R}^1$, rank 1) 의 네 부분공간:
$$
R(A)=\mathbb{R}^1,\quad R(A^\top)=\mathrm{span}\{(1,2)^\top\},\quad N(A)=\mathrm{span}\{(-2,1)^\top\},\quad N(A^\top)=\{0\}.
$$
Rank–Nullity 검증과 직교 분해 $\mathbb{R}^2 = R(A^\top)\oplus N(A)$, $R(A^\top)\perp N(A)$ 까지 자동으로 따라온다 — Strang 의 Fundamental Theorem of Linear Algebra.

---

## 1. 문제 (출제 형태 그대로)

$A=\begin{pmatrix}1 & 2\end{pmatrix}\in\mathbb R^{1\times 2}$ 일 때 다음 네 부분공간을 구하고 차원을 명시하라.

1. $R(A)$ — Image of $A$.
2. $R(A^\top)$ — Image of $A^\top$.
3. $N(A)$ — Kernel of $A$.
4. $N(A^\top)$ — Kernel of $A^\top$.

---

## 2. 출제 의도와 시험 가치 (왜 이 문제가 시험에 나오는가)

Strang 의 *Four Fundamental Subspaces* 는 선형대수 전체를 한 그림으로 압축하는 도구이다. 작은 1×2 행렬은 **(a) rank 결정 → (b) image 와 kernel 의 차원 → (c) 직교 분해** 의 전 과정을 손계산으로 끝낼 수 있는 *최소 비자명 예제*. 출제 의도:

1. Image / Kernel 의 정의를 "기계의 출력 모음 / 기계가 0 으로 보내는 입력" 으로 직관화하는지 확인.
2. Rank–Nullity 정리 ($\dim\text{dom}=\text{rank}+\text{nullity}$) 를 *그저 외운 식* 이 아니라 *실제 차원 계산* 으로 검증할 수 있는지 확인.
3. Row space ($R(A^\top)$) 와 Null space ($N(A)$) 의 직교성을 *내적 0 으로 직접 확인* 할 수 있는지 확인.
4. 도메인 $\mathbb{R}^n$ 이 어떻게 *서로 직교하는 두 부분공간의 직합* 으로 깔끔하게 분해되는지 — 이것이 SVD, 최소제곱, 차원축소의 모든 이론적 토대.

---

## 3. 사전 개념 (필요한 모든 수학 도구)

### 3.1 기호 풀이

| 기호 | 의미 |
|---|---|
| $A:\mathbb{R}^n\to\mathbb{R}^m$ | $A\in\mathbb{R}^{m\times n}$ 이 정의하는 선형사상 |
| $A^\top$ | $A$ 의 전치. $(A^\top)_{ij}=A_{ji}$. $\mathbb{R}^m\to\mathbb{R}^n$. |
| $R(A)$ | Image, range, column space. $\{Ax:x\in\mathbb{R}^n\}\subseteq\mathbb{R}^m$. |
| $R(A^\top)$ | $A$ 의 *row space* 와 동치. $\subseteq\mathbb{R}^n$. |
| $N(A)$ | Kernel, null space. $\{x:Ax=0\}\subseteq\mathbb{R}^n$. |
| $N(A^\top)$ | Left null space. $\subseteq\mathbb{R}^m$. |
| $\mathrm{span}\{v_1,\ldots,v_k\}$ | 벡터들의 모든 선형결합 $\{\sum c_i v_i\}$. |
| $\dim V$ | $V$ 의 차원 = 기저 벡터의 개수. |
| $\mathrm{rank}\,A$ | $\dim R(A)=\dim R(A^\top)$ (정리). |
| $\mathrm{nullity}\,A$ | $\dim N(A)$. |
| $U\oplus V$ | 직합. 모든 원소가 $u+v$ 로 *유일* 하게 분해됨. |
| $U\perp V$ | 직교. $\forall u\in U, v\in V:\ u\cdot v=0$. |

### 3.2 정의 정리

**Def (Image).** $R(A)=\{Ax:x\in\mathbb{R}^n\}$. 또한 $A$ 의 열들이 span 하는 공간과 동일 (column space).

**Def (Kernel).** $N(A)=\{x\in\mathbb{R}^n:Ax=0\}$.

**Def (Span).** $\mathrm{span}\{v\}=\{\alpha v:\alpha\in\mathbb{R}\}$ — 한 벡터가 그리는 직선 (영점 포함).

### 3.3 사용할 정리

**정리 (Rank).** $\mathrm{rank}\,A = \dim R(A) = \dim R(A^\top) = $ $A$ 의 일차독립 열 (또는 행) 의 최대 개수.

**정리 (Rank–Nullity).** $A:\mathbb{R}^n\to\mathbb{R}^m$ 에 대해
$$
\dim\mathrm{dom} = \mathrm{rank}\,A + \mathrm{nullity}\,A,\qquad\text{즉 }n = \mathrm{rank}\,A + \dim N(A).
$$

**정리 (Fundamental Theorem of Linear Algebra — Strang).**
$$
\mathbb{R}^n = R(A^\top)\oplus N(A),\qquad R(A^\top)\perp N(A).
$$
$$
\mathbb{R}^m = R(A)\oplus N(A^\top),\qquad R(A)\perp N(A^\top).
$$
즉 도메인은 row space 와 null space 의 직교 직합, 코도메인은 column space 와 left null space 의 직교 직합으로 분해.

---

## 4. 풀이 (모든 단계, 모든 등호 근거 명시)

### 4.0 차원 셋업

- $A:\mathbb{R}^2\to\mathbb{R}^1$.
- $A^\top:\mathbb{R}^1\to\mathbb{R}^2$.
- $A=(1\ 2)$ 는 비영행렬 ⇒ 일차독립 행 (또는 열) 적어도 1 ⇒ $\mathrm{rank}\,A=1$.
- 코도메인이 $\mathbb{R}^1$ 이라 rank 가 1 보다 클 수 없으므로 정확히 $\mathrm{rank}\,A=1$.

이 한 줄로 네 부분공간의 차원이 모두 결정된다 (Rank–Nullity 로).

### 4.1 (i) $R(A)$ — Image

**Step 1.** 정의로 풀어쓰기.
$$
R(A)=\{Ax:x=(x_1,x_2)^\top\in\mathbb{R}^2\}=\{1\cdot x_1+2\cdot x_2 : x_1,x_2\in\mathbb{R}\}.
$$
*등호 근거*: 행렬–벡터 곱 정의 $(Ax)_1 = A_{11}x_1+A_{12}x_2 = 1\cdot x_1+2\cdot x_2$.

**Step 2.** 임의 실수 $r$ 에 대해 $x_1=r,\ x_2=0$ 으로 두면 $Ax=r$. 즉 *모든* 실수가 출력으로 가능.

$$
\boxed{\;R(A)=\mathbb{R}^1,\qquad \dim R(A)=1.\;}
$$

### 4.2 (ii) $R(A^\top)$ — Row space

**Step 1.** $A^\top=\begin{pmatrix}1\\2\end{pmatrix}\in\mathbb{R}^{2\times 1}$.

**Step 2.** $A^\top y$ 는 입력 $y\in\mathbb{R}^1$ 에 대해
$$
A^\top y = \begin{pmatrix}1\\2\end{pmatrix}\cdot y = \begin{pmatrix}y\\ 2y\end{pmatrix} = y\cdot\begin{pmatrix}1\\2\end{pmatrix}.
$$

**Step 3.** $y$ 가 모든 실수를 자유롭게 움직이면 결과는 벡터 $(1,2)^\top$ 의 모든 스칼라배.

$$
\boxed{\;R(A^\top)=\mathrm{span}\!\left\{\begin{pmatrix}1\\2\end{pmatrix}\right\}\subset\mathbb{R}^2,\qquad \dim R(A^\top)=1.\;}
$$

이는 $A$ 의 *행 공간* 과 동일. (정리: $\dim R(A)=\dim R(A^\top)=\mathrm{rank}\,A$.)

### 4.3 (iii) $N(A)$ — Kernel

**Step 1.** 방정식 $Ax=0$:
$$
1\cdot x_1+2\cdot x_2=0\;\Longleftrightarrow\; x_1=-2 x_2.
$$
*등호 근거*: $A=(1\ 2)$ 의 행렬–벡터 곱 정의.

**Step 2.** 자유변수 $x_2=t$ ($t\in\mathbb{R}$) 로 두면
$$
x=\begin{pmatrix}x_1\\x_2\end{pmatrix}=\begin{pmatrix}-2t\\ t\end{pmatrix}=t\begin{pmatrix}-2\\ 1\end{pmatrix}.
$$

**Step 3.** 모든 해는 한 벡터 $(-2,1)^\top$ 의 스칼라배.

$$
\boxed{\;N(A)=\mathrm{span}\!\left\{\begin{pmatrix}-2\\ 1\end{pmatrix}\right\}\subset\mathbb{R}^2,\qquad \dim N(A)=1.\;}
$$

(부호를 뒤집어 $(2,-1)^\top$ 으로 써도 같은 부분공간 — 한 벡터의 스칼라배 전체이므로.)

### 4.4 (iv) $N(A^\top)$ — Left null space

**Step 1.** $A^\top y=0$:
$$
\begin{pmatrix}1\\2\end{pmatrix}y=\begin{pmatrix}0\\0\end{pmatrix}\;\Longleftrightarrow\; y=0\ \text{과}\ 2y=0.
$$

**Step 2.** 두 식 모두 $y=0$ 만 해.

$$
\boxed{\;N(A^\top)=\{0\}\subset\mathbb{R}^1,\qquad \dim N(A^\top)=0.\;}
$$

### 4.5 결과 표

| 공간 | 결과 | 차원 | 어디 사는가 |
|---|---|---|---|
| $R(A)$ | $\mathbb{R}^1$ | 1 | 코도메인 $\mathbb{R}^1$ |
| $R(A^\top)$ | $\mathrm{span}\{(1,2)^\top\}$ | 1 | 도메인 $\mathbb{R}^2$ |
| $N(A)$ | $\mathrm{span}\{(-2,1)^\top\}$ | 1 | 도메인 $\mathbb{R}^2$ |
| $N(A^\top)$ | $\{0\}$ | 0 | 코도메인 $\mathbb{R}^1$ |

---

## 5. 검증 (수치/대입/일관성)

### 5.1 Rank–Nullity 검증

- $A:\mathbb{R}^2\to\mathbb{R}^1$. $\dim\mathrm{dom}=2$.
  $\mathrm{rank}\,A+\dim N(A)=1+1=2$. $\checkmark$
- $A^\top:\mathbb{R}^1\to\mathbb{R}^2$. $\dim\mathrm{dom}=1$.
  $\mathrm{rank}\,A^\top+\dim N(A^\top)=1+0=1$. $\checkmark$

### 5.2 직교성 확인 — Strang 의 정리

$\mathbb{R}^2 = R(A^\top)\oplus N(A)$ 이고 두 부분공간이 직교임을 확인:
$$
\begin{pmatrix}1\\2\end{pmatrix}\cdot\begin{pmatrix}-2\\ 1\end{pmatrix}=1\cdot(-2)+2\cdot 1=-2+2=0.\quad\checkmark
$$

### 5.3 직합 분해 검증

임의의 $\mathbb{R}^2$ 벡터 $(a,b)^\top$ 가 row space 와 null space 의 합으로 *유일* 표현:

- $R(A^\top)$ 위 정사영 (단위벡터 $\hat u=(1,2)^\top/\sqrt{5}$ 사용)
$$
P_{R(A^\top)}(a,b)^\top = \frac{a+2b}{5}\binom{1}{2}.
$$
- 잔차 = null space 성분.

예: $(a,b)=(3,1)$. row 성분 $\frac{3+2}{5}(1,2)^\top=(1,2)^\top$, null 성분 $(3,1)-(1,2)=(2,-1)$. 검산: $(2,-1)\cdot(1,2)=2-2=0$ ⇒ 정말 $N(A)$. $\checkmark$ 그리고 $A\cdot(3,1)^\top=3+2=5=A\cdot(1,2)^\top$ — *오직 row space 성분만이 출력에 기여* (null 성분은 0 으로 보내짐).

### 5.4 코도메인 분해

$\mathbb{R}^1 = R(A)\oplus N(A^\top) = \mathbb{R}^1\oplus\{0\} = \mathbb{R}^1.$ $\checkmark$ (1×2 surjective 행렬에서는 left null space 가 trivial.)

---

## 6. 일반화·통찰 (관련 정리/응용)

### 6.1 4 부분공간의 큰 그림

```
     R(Aᵀ)  ────────A───────► R(A)
     (row space)            (column space)
     dim = r                 dim = r
       ⊕ ⊥                     ⊕ ⊥
     N(A)   ────────A───────► {0}   (R(A) 안에서 0 으로)
     (null space)
     dim = n − r              N(Aᵀ) dim = m − r
```

- 도메인 $\mathbb{R}^n$ 은 $R(A^\top)\oplus N(A)$ 로 *직교* 분해.
- $A$ 는 row space 위 정사영 성분만 보존하고 null space 성분은 0 으로 압축.
- 코도메인 $\mathbb{R}^m$ 은 $R(A)\oplus N(A^\top)$ 로 분해.

### 6.2 신경망 관련 통찰

- *Bottleneck* 또는 차원축소 layer ($A:\mathbb{R}^n\to\mathbb{R}^m$ with $m<n$) 는 일반적으로 $N(A)\neq\{0\}$ — 일부 입력 정보가 손실됨.
- 손실 없이 보존하려면 $\mathrm{rank}=n$ 이어야 (=$N(A)=\{0\}$, 즉 $A$ 가 단사).
- 출력이 도달 가능한 모든 점을 다 만들려면 $R(A)=\mathbb{R}^m$ — 즉 $A$ 가 전사 (=$N(A^\top)=\{0\}$).

### 6.3 SVD 와의 연결

$A=U\Sigma V^\top$ 의 *부분 분해* 가 정확히 4 부분공간을 정렬:
- $V$ 의 처음 $r$ 열 = $R(A^\top)$ 의 정규직교 기저.
- $V$ 의 나머지 열 = $N(A)$ 의 정규직교 기저.
- $U$ 의 처음 $r$ 열 = $R(A)$ 의 정규직교 기저.
- $U$ 의 나머지 열 = $N(A^\top)$ 의 정규직교 기저.

본 퀴즈는 SVD 가 만드는 4 부분공간 분해의 *축소판*.

### 6.4 최소제곱으로의 다리

선형방정식 $Ax=b$ 가 해를 가질 조건은 $b\in R(A)$. 만약 $b\notin R(A)$ 라면 *최소제곱 해* 는 $b$ 의 $R(A)$ 위 정사영을 푸는 $x$. 이 모든 게 4 부분공간 framework 위에서 자연.

---

## 7. 시험 출제 변형 5가지

### 변형 1. $A=(2\ 4)$

> 위 4 부분공간을 구하라.

*풀이.* $A$ 의 두 행렬 모두 본 퀴즈와 *비례* — 본질적으로 동일.
- $R(A)=\mathbb{R}^1$. $R(A^\top)=\mathrm{span}\{(2,4)^\top\}=\mathrm{span}\{(1,2)^\top\}$ (같은 직선).
- $N(A)$: $2x_1+4x_2=0\Rightarrow x_1=-2x_2$ ⇒ $\mathrm{span}\{(-2,1)^\top\}$. $N(A^\top)=\{0\}$.

### 변형 2. $A=\binom{1}{2}$ (2×1)

> 4 부분공간.

*풀이.* 본 퀴즈의 $A^\top$ 와 동일. 따라서 역할 교환:
- $R(A)=\mathrm{span}\{(1,2)^\top\}\subset\mathbb{R}^2$. $\dim=1$.
- $R(A^\top)=\mathbb{R}^1$. $\dim=1$.
- $N(A)=\{0\}\subset\mathbb{R}^1$. $\dim=0$.
- $N(A^\top)=\mathrm{span}\{(-2,1)^\top\}\subset\mathbb{R}^2$. $\dim=1$.

### 변형 3. 직교성

> 본 퀴즈의 $R(A^\top)$ 와 $N(A)$ 가 직교임을 확인하라.

*풀이.* $(1,2)\cdot(-2,1)=1\cdot(-2)+2\cdot 1=0$. $\checkmark$

### 변형 4. 일반 $1\times n$

> $A=(a_1,\ldots,a_n)\in\mathbb{R}^{1\times n}$, $a\neq 0$. 4 부분공간을 일반 형태로 적어라.

*풀이.*
- $R(A)=\mathbb{R}^1$ (rank 1).
- $R(A^\top)=\mathrm{span}\{a^\top\}$. $\dim=1$.
- $N(A)=\{x: a^\top x=0\}=$ 초평면. $\dim=n-1$.
- $N(A^\top)=\{0\}$. $\dim=0$.

검증: $\mathrm{rank}+\mathrm{nullity}=1+(n-1)=n$. $\checkmark$

### 변형 5. Onto / one-to-one 진단

> 본 퀴즈의 $A$ 가 onto (전사)? one-to-one (단사)?

*풀이.* Onto ⇔ $R(A)=\mathbb{R}^m=\mathbb{R}^1$. **Yes.** One-to-one ⇔ $N(A)=\{0\}$. 본 퀴즈에서 $N(A)$ 가 1차원 직선이라 $\neq \{0\}$. **No.** (즉 surjective but not injective — 차원 압축의 전형.)

---

## 8. 백지 재현 체크리스트

1. [ ] 4 부분공간의 정의 ($R, N$) 와 사는 공간을 적을 수 있다.
2. [ ] $A=(1\ 2)$ 의 rank 가 1 임을 즉시 결론.
3. [ ] $R(A)=\mathbb{R}^1$ 을 도출.
4. [ ] $A^\top=\binom{1}{2}$ 임을 적을 수 있다.
5. [ ] $R(A^\top)=\mathrm{span}\{(1,2)^\top\}$ 도출.
6. [ ] 방정식 $x_1+2x_2=0$ 을 풀어 $N(A)=\mathrm{span}\{(-2,1)^\top\}$ 도출.
7. [ ] $N(A^\top)=\{0\}$ 도출.
8. [ ] Rank–Nullity 로 $A$ (2 = 1+1), $A^\top$ (1 = 1+0) 검증.
9. [ ] Strang 의 직교성 $R(A^\top)\perp N(A)$ 를 내적 0 으로 검증.
10. [ ] 도메인 분해 $\mathbb{R}^2 = R(A^\top)\oplus N(A)$ 를 적을 수 있다.
11. [ ] 코도메인 분해 $\mathbb{R}^1 = R(A)\oplus N(A^\top)$ 를 적을 수 있다.
12. [ ] Onto/one-to-one 의 4 공간 기준 진단.

---

## 9. 핵심 공식 카드

```
[정의]
  R(A)  = {Ax : x ∈ R^n} ⊆ R^m            (image / column space)
  R(Aᵀ) = {Aᵀy: y ∈ R^m} ⊆ R^n            (row space)
  N(A)  = {x : Ax = 0} ⊆ R^n               (null space)
  N(Aᵀ) = {y : Aᵀy = 0} ⊆ R^m              (left null space)

[Rank–Nullity]
  dim(R^n) = rank(A) + dim N(A)
  dim(R^m) = rank(A) + dim N(Aᵀ)
  rank(A) = rank(Aᵀ)

[Strang Fundamental Theorem]
  R^n = R(Aᵀ) ⊕ N(A),   R(Aᵀ) ⊥ N(A)
  R^m = R(A)  ⊕ N(Aᵀ),  R(A)  ⊥ N(Aᵀ)

[A = (1 2) 결과]
  R(A)   = R^1                       (dim 1)
  R(Aᵀ)  = span{(1,2)ᵀ}              (dim 1)
  N(A)   = span{(-2,1)ᵀ}             (dim 1)
  N(Aᵀ)  = {0}                       (dim 0)
  검증: (1,2)·(-2,1) = -2+2 = 0     ✓
```

---

## 10. 다른 퀴즈와의 연결

- **Quiz 1 (선형변환):** 같은 행렬을 *작용 메커니즘* 으로 본 퀴즈. Q2 는 같은 행렬을 *공간 분해* 로 본다. 두 시각의 결합이 선형대수 1주차의 완성.
- **Quiz 12·13 (Conv Matrix):** Toeplitz / banded 행렬의 image 와 kernel 을 따져 보면 컨볼루션 layer 의 *정보 손실 / 보존* 패턴이 나온다. 본 퀴즈의 4 공간 framework 가 그대로 적용.
- **Quiz 10 (Backprop):** 출력층의 그래디언트 $p-e_y$ 가 $R(\partial L/\partial z)$ 의 어디 사는지를 보면 학습 신호의 흐름을 4 공간 위에서 분석 가능 (고급 통찰).
- **SVD / 차원축소:** 본 퀴즈의 $r=1$ 케이스가 일반 rank $r$ SVD 의 micro-version. 도메인의 *유효한* (rank 와 같은) 차원만 출력에 반영되고 나머지는 null space 로 압축된다.
- **최소제곱 / 정규방정식:** $A^\top Ax=A^\top b$ 의 해석은 $b$ 의 $R(A)$ 위 정사영 — 4 공간 framework 없이는 한 줄도 안 나온다.

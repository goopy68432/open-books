---
title: "딥러닝 이론 모의고사 #3 — 선형대수 집중"
slug: 05-mock-exam-3-linear-algebra
order: 9
---

# 딥러닝 이론 모의고사 #3 — 선형대수 집중

> 배점 100점 / 10문제 / 논리 과정 서술 필수

---

## 문제 1. [10점] 행렬 = 선형변환

**(a)** [5점] 행렬 $A \in \mathbb{R}^{m \times n}$이 정의하는 선형변환 $L_A: \mathbb{R}^n \to \mathbb{R}^m$에서, $Av$를 "열벡터들의 선형결합"으로 해석하시오. 구체적으로 $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $v = \begin{bmatrix} a \\ b \end{bmatrix}$일 때 $Av$를 열벡터의 선형결합으로 쓰시오.

**(b)** [5점] 행렬곱 $AB$의 시간복잡도가 $O(n^3)$인 이유를 결과 행렬의 각 원소 계산량으로부터 유도하시오.

---

## 문제 2. [10점] 내적과 외적

**(a)** [4점] 내적 $\langle v, w \rangle = v^\top w$와 외적 $vw^\top$의 결과 크기(shape)를 비교하고, 각각의 기하학적/대수적 의미를 설명하시오.

**(b)** [6점] Cross-Entropy Loss에서 $e_{y_i}^\top \log h(x_i)$라는 표현이 나온다. 여기서 $e_{y_i}$는 정답 클래스의 one-hot 벡터다. 이 내적이 "정답 클래스의 로그 확률만 골라낸다"는 것을 구체적으로 보이시오. (예: 3-클래스, 정답=2)

---

## 문제 3. [10점] 선형독립과 기저

**(a)** [5점] 벡터 집합 $\{v_1, v_2, ..., v_k\}$가 선형독립임의 정의를 쓰고, 직관적 의미("정보 중복 없음")를 설명하시오.

**(b)** [5점] $\mathbb{R}^3$에서 $v_1 = (1,0,0)$, $v_2 = (1,1,0)$, $v_3 = (2,1,0)$이 선형독립인지 판별하시오. 판별 과정을 명시하시오.

---

## 문제 4. [10점] Range와 Kernel

**(a)** [4점] $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$의 $\mathscr{R}(A)$와 $\mathscr{N}(A)$를 각각 구하시오.

**(b)** [6점] Rank-Nullity 정리를 이용하여 (a)의 결과를 검증하시오. 이 행렬이 가역이 아닌 이유를 5가지 이상의 동치 조건으로 설명하시오.

---

## 문제 5. [10점] 고유값/고유벡터 기초

**(a)** [4점] $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$의 고유값과 고유벡터를 구하시오.

**(b)** [6점] 대칭행렬 $S = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$의 고유값 분해 $S = U\Lambda U^\top$를 수행하시오. 고유벡터가 직교함을 확인하시오.

---

## 문제 6. [10점] 특수 행렬

**(a)** [4점] 직교행렬(Orthogonal Matrix)의 정의와 3가지 핵심 성질을 쓰시오.

**(b)** [3점] 양정치(PD) 행렬과 양반정치(PSD) 행렬의 정의와 차이를 쓰시오.

**(c)** [3점] 대칭행렬이 PSD일 조건을 고유값으로 표현하고, 이것이 "이차형식이 항상 비음수"와 동치인 이유를 설명하시오.

---

## 문제 7. [10점] SVD 계산

$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$일 때,

**(a)** [4점] $A^\top A$와 $AA^\top$를 구하고, 각각의 고유값을 구하시오.

**(b)** [6점] $A$의 SVD $A = U\Sigma V^\top$를 구하시오. $U$, $\Sigma$, $V$의 크기를 명시하시오.

---

## 문제 8. [10점] 저랭크 근사

**(a)** [5점] SVD를 이용한 랭크-$r$ 근사 $A_r = \sum_{i=1}^r \sigma_i u_i v_i^\top$가 프로베니우스 노름 의미에서 최적임을 서술하시오 (Eckart-Young).

**(b)** [5점] 100×100 이미지 행렬에서 랭크-10 근사를 사용하면 저장해야 할 값의 수가 어떻게 변하는지 계산하시오. 압축률은?

---

## 문제 9. [10점] 노름과 Trace

**(a)** [4점] L1, L2, L0 노름을 정의하고, 각각 딥러닝에서의 용도를 쓰시오.

**(b)** [3점] 프로베니우스 노름 $\|A\|_F$가 특이값의 L2 노름과 같음을 보이시오.

**(c)** [3점] $\text{Tr}(AB) = \text{Tr}(BA)$를 증명하시오. 이 성질이 $x^\top Ax = \text{Tr}(xx^\top A)$를 보이는 데 어떻게 사용되는지 설명하시오.

---

## 문제 10. [10점] 가역 행렬 동치 조건

정방행렬 $A \in \mathbb{R}^{n \times n}$에 대해 다음 중 **하나라도 성립하면 나머지 모두 성립함**을 보이시오.

> (1) $A^{-1}$ 존재, (2) rank$(A) = n$, (3) null$(A) = 0$, (4) det$(A) \neq 0$, (5) 모든 고유값 $\neq 0$

**(a)** [6점] (1)→(3)→(2)→(1) 순환 증명
**(b)** [4점] (4)↔(5)의 증명 (det = 고유값의 곱 이용)

---
---

# 모범답안

## 답 1.
### (a)
$$Av = a \begin{bmatrix}1\\3\end{bmatrix} + b\begin{bmatrix}2\\4\end{bmatrix}$$
행렬-벡터 곱은 A의 열벡터 $a_1, a_2$에 $v$의 성분 $a, b$를 가중치로 하는 선형결합이다.

### (b)
결과 $C = AB$에서 $c_{ij} = \sum_{k=1}^n a_{ik}b_{kj}$: 각 원소 계산에 $n$번 곱셈+덧셈.
결과 행렬 크기 $n \times n$ → 총 $n^2$개 원소 × 원소당 $O(n)$ = $O(n^3)$. $\square$

## 답 2.
### (b)
3-클래스, 정답=2: $e_2 = (0, 1, 0)^\top$, $\log h = (\log h_1, \log h_2, \log h_3)^\top$
$$e_2^\top \log h = 0 \cdot \log h_1 + 1 \cdot \log h_2 + 0 \cdot \log h_3 = \log h_2$$
정답 클래스(2)의 로그 확률만 선택됨. $\square$

## 답 3.
### (b)
$a(1,0,0) + b(1,1,0) + c(2,1,0) = (0,0,0)$
→ $a+b+2c=0$, $b+c=0$, $0=0$
→ $b=-c$, $a=-b-2c=c-2c=-c$ → $a=-c, b=-c$. $c=1$이면 $(-1,-1,1) \neq 0$.
**선형종속**. $v_3 = v_1 + v_2$이므로 정보 중복 있음.

## 답 4.
### (a)
$\mathscr{R}(A) = \text{span}\{(1,2)^\top\}$ (두 열이 비례: $(2,4) = 2(1,2)$)
$\mathscr{N}(A)$: $x+2y=0$ → $\mathscr{N}(A) = \text{span}\{(-2,1)^\top\}$

### (b)
$n=2$, rank=1, nullity=1 → $2 = 1+1$ ✓
비가역 이유: (1) rank≠2, (2) nullity≠0, (3) det=4-4=0, (4) 고유값 0과 5 → 0 존재, (5) 열이 선형종속

## 답 5.
### (b)
$\det(S-\lambda I) = (2-\lambda)^2-1 = 0$ → $\lambda = 3, 1$
$\lambda=3$: $(S-3I)v=0$ → $v_1 = (1,1)^\top/\sqrt{2}$
$\lambda=1$: $(S-I)v=0$ → $v_2 = (1,-1)^\top/\sqrt{2}$
$v_1^\top v_2 = (1-1)/2 = 0$ ✓ 직교
$$S = \frac{1}{2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}3&0\\0&1\end{bmatrix}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

## 답 7.
### (a)
$A^\top A = \begin{bmatrix}9&0\\0&4\end{bmatrix}$, 고유값: 9, 4
$AA^\top = \begin{bmatrix}9&0&0\\0&4&0\\0&0&0\end{bmatrix}$, 고유값: 9, 4, 0

### (b)
$\sigma_1=3, \sigma_2=2$
$V = I_2$ (이미 대각), $U = \begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}$ (표준 기저)
$\Sigma = \begin{bmatrix}3&0\\0&2\\0&0\end{bmatrix}$

## 답 8.
### (b)
원본: 100×100 = 10,000개 값
랭크-10: $U$(100×10) + $\Sigma$(10) + $V^\top$(10×100) = 1000+10+1000 = 2,010개
압축률: 2,010/10,000 = **20.1%** (약 5배 압축)

## 답 9.
### (c)
$(AB)_{ij} = \sum_k a_{ik}b_{kj}$ → $\text{Tr}(AB) = \sum_i \sum_k a_{ik}b_{ki}$
$(BA)_{kk} = \sum_i b_{ki}a_{ik}$ → $\text{Tr}(BA) = \sum_k \sum_i b_{ki}a_{ik}$
합산 순서만 다르므로 동일. $\square$

$x^\top Ax$: 스칼라. $\text{Tr}(xx^\top A) = \text{Tr}(A \cdot xx^\top)$에서 $xx^\top$는 랭크-1 행렬.
$x^\top Ax = \text{Tr}(x^\top Ax) = \text{Tr}(Axx^\top)$ (순환 성질) $\square$

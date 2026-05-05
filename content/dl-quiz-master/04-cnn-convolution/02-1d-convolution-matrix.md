---
title: 'Quiz 12 — 1D Convolution Matrix'
description: 'A ∈ R^{6×7} Toeplitz 행렬 표현'
draft: false
---

## 0. 한 줄 요약
선형사상 $f(x) = w * x: \mathbb{R}^7 \to \mathbb{R}^6$ ($w = [1, 2]$) 의 행렬 표현은 표준기저 $e_j$ 의 상 $f(e_j)$ 를 열로 모아 만든 sparse Toeplitz 행렬 $A \in \mathbb{R}^{6 \times 7}$ 이며, weight sharing 과 locality 가 행렬 구조에 그대로 박혀 있다.

## 1. 문제 (정확한 출제 형태)
커널 $w = [1, 2]$ 가 정의하는 1D convolution(cross-correlation)
$$
f: \mathbb{R}^7 \to \mathbb{R}^6, \qquad (f(x))_i = x_i + 2 x_{i+1}
$$
는 선형사상이다. 이 선형사상의 행렬 표현 $A \in \mathbb{R}^{6 \times 7}$ ($f(x) = A x$) 을 구하라.

## 2. 출제 의도와 시험 가치
- **선형대수 핵심 정리**: 선형사상 $f: \mathbb{R}^n \to \mathbb{R}^m$ 의 행렬은 $A = [f(e_1) \ f(e_2) \ \cdots \ f(e_n)]$.
- **Convolution = 행렬곱** 으로 보는 관점 → CNN 을 fully-connected (FC) 의 매우 제약된 특수 경우로 이해하는 출발점.
- **Sparsity + weight sharing** 의 시각화: 42 개 셀 중 12 개만 비영, 같은 값 $1, 2$ 가 반복.
- **Inductive bias**: sparsity, locality, weight sharing, translation equivariance 의 4 종이 모두 이 한 행렬에 시각적으로 드러남.

## 3. 사전 개념 (모든 수학 도구)

### 3.1 선형사상의 행렬 표현 정리
$f: \mathbb{R}^n \to \mathbb{R}^m$ 이 선형이면 표준기저 $\{e_1, \dots, e_n\}$ 에 대해
$$
A = \big[\, f(e_1) \mid f(e_2) \mid \cdots \mid f(e_n) \,\big] \in \mathbb{R}^{m \times n},
$$
즉 $A$ 의 $j$ 번째 **열** 은 $f(e_j)$ 이고 $f(x) = A x$ 가 모든 $x$ 에 대해 성립한다.

증명 스케치: $x = \sum_j x_j e_j$ → $f(x) = \sum_j x_j f(e_j) = \sum_j x_j A_{:,j} = A x$.

### 3.2 표준기저의 conv
$e_j \in \mathbb{R}^7$ 은 $j$ 번째만 1 이고 나머지 0. 정의 $y_i = x_i + 2 x_{i+1}$ 에 $x = e_j$ 대입:
- $x_i = 1 \iff i = j$, 이때 $y_i = 1$ → $i = j$ 위치에 1 (단 $j \le 6$).
- $x_{i+1} = 1 \iff i + 1 = j \iff i = j - 1$, 이때 $y_i = 2$ → $i = j-1$ 위치에 2 (단 $j \ge 2$).

### 3.3 Toeplitz 행렬
대각선이 일정한 행렬: $A_{ij}$ 가 $i - j$ 의 함수. 1D convolution 의 행렬 표현은 항상 Toeplitz (혹은 banded Toeplitz).

### 3.4 Convolution 의 선형성 (Q11 사전준비)
$w * (\alpha x + \beta y) = \alpha (w * x) + \beta (w * y)$ 가 성립하므로 행렬 표현이 가능하다.

## 4. 풀이 (모든 단계, 등호 근거)

### 4.1 7 개 기저의 상 계산
$w = [1, 2]$, 정의 $y_i = x_i + 2 x_{i+1}$ ($i = 1, \dots, 6$).

| $j$ | $e_j$ 의 비영 위치 | $f(e_j) \in \mathbb{R}^6$ |
|-----|-------------------|---------------------------|
| 1 | $x_1 = 1$ | $y_1 = 1$, 나머지 0 → $(1, 0, 0, 0, 0, 0)^\top$ |
| 2 | $x_2 = 1$ | $y_1 = 2$ (∵ $x_{1+1}$), $y_2 = 1$ (∵ $x_2$) → $(2, 1, 0, 0, 0, 0)^\top$ |
| 3 | $x_3 = 1$ | $y_2 = 2$, $y_3 = 1$ → $(0, 2, 1, 0, 0, 0)^\top$ |
| 4 | $x_4 = 1$ | $y_3 = 2$, $y_4 = 1$ → $(0, 0, 2, 1, 0, 0)^\top$ |
| 5 | $x_5 = 1$ | $y_4 = 2$, $y_5 = 1$ → $(0, 0, 0, 2, 1, 0)^\top$ |
| 6 | $x_6 = 1$ | $y_5 = 2$, $y_6 = 1$ → $(0, 0, 0, 0, 2, 1)^\top$ |
| 7 | $x_7 = 1$ | $y_6 = 2$ (∵ $x_{6+1}$), $y_1 \dots y_5$ 0 → $(0, 0, 0, 0, 0, 2)^\top$ |

### 4.2 열로 모아 행렬 $A$ 구성
$$
A = \begin{bmatrix}
1 & 2 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 2 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 2 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 2 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 2 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 2
\end{bmatrix} \in \mathbb{R}^{6 \times 7}.
$$

각 행은 동일한 패턴 $[1, 2]$ 가 한 칸씩 오른쪽으로 평행이동한 형태 → **Toeplitz**.

## 5. 검증

### 5.1 행 기준 재해석
$i$ 번째 행 $A_{i,:}$ 와 $x$ 의 내적 = $A_{i,i} x_i + A_{i, i+1} x_{i+1} = 1 \cdot x_i + 2 \cdot x_{i+1} = y_i$ ✓.

### 5.2 Q11-(1) 재계산으로 일치 확인
$x^{(1)} = (0, 1, 2, 3, 4, 5, 6)^\top$ 에 대해 $A x^{(1)}$ 의 각 성분:
- 1행: $1 \cdot 0 + 2 \cdot 1 = 2$
- 2행: $1 \cdot 1 + 2 \cdot 2 = 5$
- 3행: $1 \cdot 2 + 2 \cdot 3 = 8$
- 4행: $1 \cdot 3 + 2 \cdot 4 = 11$
- 5행: $1 \cdot 4 + 2 \cdot 5 = 14$
- 6행: $1 \cdot 5 + 2 \cdot 6 = 17$

$\Rightarrow A x^{(1)} = (2, 5, 8, 11, 14, 17)^\top$ — Q11-(1) 의 정답과 정확히 일치 ✓.

### 5.3 차원 sanity
$A: 6 \times 7$, $x: 7 \times 1$ → $Ax: 6 \times 1$. 출력 길이 $n - k + 1 = 6$ 과 일치 ✓.

### 5.4 비영 원소 수
각 행에 2 개씩, 총 $6 \times 2 = 12$ 개. 전체 $6 \times 7 = 42$ 셀의 28.6%. **Sparse**.

## 6. 일반화·통찰

### 6.1 CNN 의 4가지 inductive bias (이 한 행렬에서 모두 보임)
1. **Sparsity**: 비영 원소 12 / 42 ≈ 29% (FC 라면 100%).
2. **Locality**: 각 출력 $y_i$ 는 입력의 인접한 $k = 2$ 개 원소만 본다.
3. **Weight sharing**: 같은 $1, 2$ 가 모든 행에서 반복 — 학습할 파라미터는 단 2 개($w_1, w_2$), FC 라면 42 개.
4. **Translation equivariance/invariance**: 입력을 $t$ 칸 평행이동하면 출력도 $t$ 칸 이동. Toeplitz 구조의 직접적 귀결.

### 6.2 일반 형태
입력 길이 $n$, 커널 길이 $k$, valid → $A \in \mathbb{R}^{(n-k+1) \times n}$ 의 $i$ 번째 행은
$$
A_{i, j} = \begin{cases} w_{j - i + 1} & \text{if } 1 \le j - i + 1 \le k \\ 0 & \text{otherwise.} \end{cases}
$$

### 6.3 FC vs CNN 파라미터 수
- FC ($\mathbb{R}^7 \to \mathbb{R}^6$): $6 \times 7 = 42$ 개 가중치.
- CNN ($k = 2$): 2 개 가중치. **21 배 절약** + translation equivariance 보너스.

## 7. 시험 출제 변형 5가지
1. **커널 $w = [1, -1]$** 의 $A$ 작성. (1차 미분 행렬, 대각 1, 상부 대각 −1.)
2. **커널 길이 3** ($w = [a, b, c]$, $n = 7$) → $A \in \mathbb{R}^{5 \times 7}$, 각 행 $[a, b, c]$ 한 칸씩 이동.
3. **Stride 2** 일 때 행렬은? → 행렬에서 짝수 행만 골라낸 형태 ($\mathbb{R}^{3 \times 7}$).
4. **Circular convolution** (순환 경계) 의 $A$ → 정사각 $7 \times 7$ circulant 행렬, 우상 모서리에 wrap-around 항 등장.
5. **Transposed convolution**: $A^\top \in \mathbb{R}^{7 \times 6}$ 가 입력 6 → 출력 7 의 "deconv" 행렬. 디코더/업샘플의 기초.

## 8. 백지 재현 체크리스트
- [ ] 정리: $A = [f(e_1) \mid \dots \mid f(e_n)]$ 적기
- [ ] $w = [1, 2]$, $f: \mathbb{R}^7 \to \mathbb{R}^6$ 명시
- [ ] $f(e_j)$ 7개 모두 손으로 계산 (특히 $j = 1$ 과 $j = 7$ 의 경계)
- [ ] $6 \times 7$ Toeplitz 행렬 그리기
- [ ] $A x^{(1)} = (2, 5, 8, 11, 14, 17)^\top$ 검증
- [ ] sparsity / locality / weight sharing / translation equivariance 4종 언급
- [ ] FC 42 개 vs CNN 2 개 파라미터 비교

## 9. 핵심 공식 카드
- 행렬 표현: $A = [\,f(e_1) \mid f(e_2) \mid \cdots \mid f(e_n)\,]$, $A_{:,j} = f(e_j)$
- 본 문제: $A \in \mathbb{R}^{6 \times 7}$, banded Toeplitz, 각 행 $[1, 2]$ 패턴
- 비영 원소: 행당 $k = 2$ 개, 총 $(n - k + 1) k = 12$ 개
- 파라미터: CNN $k = 2$ vs FC $m n = 42$
- 검증: $A x^{(1)} = (2, 5, 8, 11, 14, 17)^\top$ (Q11-(1) 일치)
- Inductive bias 4종: sparse, local, shared, equivariant

## 10. 다른 퀴즈와의 연결
- **Q11 (1D conv 계산)**: 본 행렬 $A$ 와 $x^{(1)}$ 의 곱이 Q11-(1) 정답과 일치 — 두 퀴즈의 직접적 동치 검증.
- **Q13 (2D conv 행렬)**: 1D Toeplitz → 2D 에서 **doubly block-Toeplitz (BTTB)** 로 자연스럽게 확장. 입력을 row-major 로 평탄화하면 같은 정리 ($A_{:,j} = f(e_j)$) 가 그대로 적용됨.
- **CNN 강의 전체**: 본 퀴즈가 "CNN 은 매우 sparse 한 weight-shared FC 다" 의 가장 간결한 증명. 이후 backprop, receptive field 분석 모두 이 행렬 그림에서 출발.

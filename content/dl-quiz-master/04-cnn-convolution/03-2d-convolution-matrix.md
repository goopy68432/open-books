---
title: 'Quiz 13 — 2D Convolution Matrix'
description: 'A ∈ R^{4×9} BTTB (block Toeplitz) 행렬'
draft: false
---

## 0. 한 줄 요약
$2 \times 2$ 커널 $w = \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix}$ 가 $3 \times 3$ 입력에 적용되는 2D convolution 은, 입력을 row-major 로 평탄화하면 선형사상 $f: \mathbb{R}^9 \to \mathbb{R}^4$ 가 되며, 행렬 표현 $A \in \mathbb{R}^{4 \times 9}$ 는 doubly block-Toeplitz (BTTB) 구조의 sparse 행렬이 된다.

## 1. 문제 (정확한 출제 형태)
$2 \times 2$ 커널
$$
w = \begin{bmatrix} 0 & 1 \\ 2 & 3 \end{bmatrix}
$$
가 $3 \times 3$ 입력 $X$ 에 valid 2D convolution(cross-correlation) 으로 적용되어 $2 \times 2$ 출력 $Y$ 를 만든다. 입력을 row-major 로 평탄화한 벡터를 $x = (x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9)^\top \in \mathbb{R}^9$, 출력을 $y = (y_1, y_2, y_3, y_4)^\top \in \mathbb{R}^4$ 라 하자:
$$
X = \begin{bmatrix} x_1 & x_2 & x_3 \\ x_4 & x_5 & x_6 \\ x_7 & x_8 & x_9 \end{bmatrix}, \qquad Y = \begin{bmatrix} y_1 & y_2 \\ y_3 & y_4 \end{bmatrix}.
$$
이 선형사상의 행렬 표현 $A \in \mathbb{R}^{4 \times 9}$ ($y = A x$) 를 구하라.

## 2. 출제 의도와 시험 가치
- 1D 의 Toeplitz 표현(Q12) 을 2D 로 확장. 평탄화(im2col 의 정신적 모형) 의 정확한 인덱스 관리 능력.
- 2D convolution = 매우 sparse 한 행렬곱 으로 보는 시각.
- 출력 차원 공식 $H_{out} = H_{in} - k + 1$ 의 2D 일반화.
- BTTB(Block Toeplitz with Toeplitz Blocks) 구조에 대한 직관 — FFT 가속, transposed conv, 역행렬 분석의 기반.

## 3. 사전 개념 (모든 수학 도구)

### 3.1 2D Cross-correlation 정의
입력 $X \in \mathbb{R}^{H \times W}$, 커널 $w \in \mathbb{R}^{k_h \times k_w}$ → 출력 $Y \in \mathbb{R}^{(H - k_h + 1) \times (W - k_w + 1)}$:
$$
Y_{a, b} = \sum_{u=0}^{k_h - 1} \sum_{v=0}^{k_w - 1} w_{u, v} \cdot X_{a + u,\ b + v}.
$$
(인덱스 0-base.) PyTorch/TF "convolution" 도 실은 이 식.

### 3.2 출력 차원 (AlexNet 공식)
$$
H_{out} = \frac{H_{in} + 2 p - k}{s} + 1, \qquad W_{out} = \frac{W_{in} + 2 p - k}{s} + 1.
$$
본 문제: $H_{in} = W_{in} = 3$, $k = 2$, $p = 0$, $s = 1$ → $H_{out} = W_{out} = 2$.

### 3.3 Row-major 평탄화 인덱스
$3 \times 3$ 행렬의 $(r, c)$ (0-base) 위치는 평탄화 벡터의 $3 r + c$ (0-base) 번째 = $3 r + c + 1$ (1-base).

| $(r, c)$ | 평탄화 인덱스 (1-base) |
|----------|------------------------|
| (0, 0) | 1 |
| (0, 1) | 2 |
| (0, 2) | 3 |
| (1, 0) | 4 |
| (1, 1) | 5 |
| (1, 2) | 6 |
| (2, 0) | 7 |
| (2, 1) | 8 |
| (2, 2) | 9 |

### 3.4 출력 인덱싱
$Y$ 의 $(0,0), (0,1), (1,0), (1,1)$ 을 $y_1, y_2, y_3, y_4$ 라 한다.

### 3.5 선형사상의 행렬 표현 (Q12 와 동일)
$A_{:, j} = f(e_j)$. $j = 1, \dots, 9$ 에 대해 $e_j$ (해당 위치만 1) 를 입력으로 넣었을 때의 4 원소 출력을 열로 채운다.

## 4. 풀이 (모든 단계, 등호 근거)

### 4.1 4개 출력의 식
커널 $w$ 의 원소 $w_{0,0} = 0,\ w_{0,1} = 1,\ w_{1,0} = 2,\ w_{1,1} = 3$.

각 출력 위치 $(a, b)$ 에 대해 $Y_{a,b} = \sum_{u, v} w_{u,v} X_{a+u, b+v}$:

$y_1$ : $(a, b) = (0, 0)$
$$
y_1 = w_{00} X_{00} + w_{01} X_{01} + w_{10} X_{10} + w_{11} X_{11} = 0 \cdot x_1 + 1 \cdot x_2 + 2 \cdot x_4 + 3 \cdot x_5.
$$

$y_2$ : $(a, b) = (0, 1)$
$$
y_2 = 0 \cdot x_2 + 1 \cdot x_3 + 2 \cdot x_5 + 3 \cdot x_6.
$$

$y_3$ : $(a, b) = (1, 0)$
$$
y_3 = 0 \cdot x_4 + 1 \cdot x_5 + 2 \cdot x_7 + 3 \cdot x_8.
$$

$y_4$ : $(a, b) = (1, 1)$
$$
y_4 = 0 \cdot x_5 + 1 \cdot x_6 + 2 \cdot x_8 + 3 \cdot x_9.
$$

### 4.2 행렬 $A$ 구성 (행 기준)
각 행은 위 식의 $x_1, \dots, x_9$ 계수를 그대로 적은 것:

$$
A = \begin{bmatrix}
0 & 1 & 0 & 2 & 3 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 2 & 3 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 2 & 3 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 2 & 3
\end{bmatrix} \in \mathbb{R}^{4 \times 9}.
$$

### 4.3 행렬 $A$ 의 열 기준 (기저 검증)
$A_{:, j} = f(e_j)$ — 즉 입력 픽셀 $x_j$ 가 어느 출력에 어떤 계수로 등장하는지 모은 것.

| $j$ | 위치 $(r, c)$ | 등장하는 출력과 계수 | $A_{:, j}$ |
|-----|---------------|----------------------|------------|
| 1 | (0,0) | 어떤 출력에도 안 나옴 | $(0, 0, 0, 0)^\top$ |
| 2 | (0,1) | $y_1$ 의 계수 1 | $(1, 0, 0, 0)^\top$ |
| 3 | (0,2) | $y_2$ 의 계수 1 | $(0, 1, 0, 0)^\top$ |
| 4 | (1,0) | $y_1$ 의 계수 2 | $(2, 0, 0, 0)^\top$ |
| 5 | (1,1) | $y_1$:3, $y_2$:2, $y_3$:1, $y_4$:0 | $(3, 2, 1, 0)^\top$ |
| 6 | (1,2) | $y_2$:3, $y_4$:1 | $(0, 3, 0, 1)^\top$ |
| 7 | (2,0) | $y_3$:2 | $(0, 0, 2, 0)^\top$ |
| 8 | (2,1) | $y_3$:3, $y_4$:2 | $(0, 0, 3, 2)^\top$ |
| 9 | (2,2) | $y_4$:3 | $(0, 0, 0, 3)^\top$ |

이 9 개 열을 차례로 붙이면 4.2 의 $A$ 와 정확히 일치 ✓.

## 5. 검증

### 5.1 중심 픽셀 $x_5$ 검증 (가장 정보량 많은 열)
$x_5$ 는 $3 \times 3$ 입력의 정중앙. valid $2 \times 2$ conv 에서 4개 출력 모두에 등장하며, 각 출력에서의 위치는:
- $y_1$ (좌상): $X_{1,1}$ 자리 → 커널의 $(1,1) = w_{11} = 3$
- $y_2$ (우상): $X_{1,1}$ 자리 → 커널의 $(1,0) = w_{10} = 2$
- $y_3$ (좌하): $X_{1,1}$ 자리 → 커널의 $(0,1) = w_{01} = 1$
- $y_4$ (우하): $X_{1,1}$ 자리 → 커널의 $(0,0) = w_{00} = 0$

$\Rightarrow A_{:, 5} = (3, 2, 1, 0)^\top$ — 표와 일치 ✓.

### 5.2 모서리 픽셀 검증
- $x_1$ (좌상 모서리): valid 출력에서 한 번도 안 쓰임 → 0 열 ✓.
- $x_9$ (우하 모서리): $y_4$ 에서만 $w_{11} = 3$ 으로 사용 → $(0, 0, 0, 3)^\top$ ✓.

### 5.3 각 행의 비영 원소 수
모든 행에 정확히 4개씩 → 총 $4 \times 4 = 16$ 개 비영. 전체 $4 \times 9 = 36$ 셀의 44.4%. **Sparse**.

### 5.4 차원 sanity
$A: 4 \times 9$, $x: 9 \times 1$ → $y: 4 \times 1$, 출력 픽셀 $H_{out} W_{out} = 2 \times 2 = 4$ ✓.

### 5.5 BTTB 구조 확인
$A$ 를 $2 \times 3$ 의 블록(각 블록 $2 \times 3$) 으로 보면:
$$
A = \begin{bmatrix} W_0 & W_1 & 0 \\ 0 & W_0 & W_1 \end{bmatrix}_{\text{block}}
\quad \text{with}\quad
W_0 = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix},\
W_1 = \begin{bmatrix} 2 & 3 & 0 \\ 0 & 2 & 3 \end{bmatrix}.
$$
바깥 블록 패턴이 Toeplitz, 안쪽 각 블록도 Toeplitz → **doubly block-Toeplitz (BTTB)** ✓.

## 6. 일반화·통찰

### 6.1 일반 차원 공식 (AlexNet)
$$
H_{out} = \left\lfloor \frac{H_{in} + 2 p - k}{s} \right\rfloor + 1.
$$
여기에 $H_{in} = 3, k = 2, p = 0, s = 1$ 대입 → 2.

### 6.2 평탄화 후 행렬 크기
입력 $H \times W$, 커널 $k_h \times k_w$, valid → $A \in \mathbb{R}^{H_{out} W_{out} \,\times\, H W}$. 본 문제 $4 \times 9$.

### 6.3 파라미터 절약
- FC ($\mathbb{R}^9 \to \mathbb{R}^4$): $4 \times 9 = 36$ 개.
- CNN: $2 \times 2 = 4$ 개. **9 배 절약** + translation equivariance.

### 6.4 BTTB → FFT 가속
2D conv 는 BTTB 행렬곱 = 2D FFT 로 $O(N \log N)$ 가속 가능. 커널이 클 때 표준 트릭.

### 6.5 Transposed conv
$A^\top \in \mathbb{R}^{9 \times 4}$ 가 $\mathbb{R}^4 \to \mathbb{R}^9$ 의 업샘플 — U-Net 디코더, GAN generator 의 기초.

## 7. 시험 출제 변형 5가지
1. **커널 행렬 다른 값** ($w = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ 대각 차분 필터) → 같은 자리 패턴, 값만 1, 0, 0, −1.
2. **입력 $4 \times 4$, 커널 $2 \times 2$** → $A \in \mathbb{R}^{9 \times 16}$.
3. **입력 $3 \times 3$, 커널 $3 \times 3$** → $A \in \mathbb{R}^{1 \times 9}$ (출력 한 점), 행 하나.
4. **Stride 2, 입력 $4 \times 4$, 커널 $2 \times 2$** → $A \in \mathbb{R}^{4 \times 16}$, 출력 위치마다 2 칸씩 점프.
5. **Padding 1 (zero), 입력 $3 \times 3$, 커널 $2 \times 2$** → $A \in \mathbb{R}^{16 \times 9}$, 패딩 0 위치는 행렬의 해당 열이 사라진 형태로 표현.

## 8. 백지 재현 체크리스트
- [ ] $3 \times 3$ 입력 row-major 평탄화 인덱스 표 그리기
- [ ] AlexNet 공식으로 출력 $2 \times 2$ 확인
- [ ] $y_1, y_2, y_3, y_4$ 의 4개 식을 $x_1, \dots, x_9$ 로 풀어쓰기
- [ ] $4 \times 9$ 행렬 $A$ 작성
- [ ] 5번째 열 $A_{:, 5} = (3, 2, 1, 0)^\top$ 검증 (중심 픽셀)
- [ ] 모서리 $x_1, x_9$ 열 검증
- [ ] BTTB 블록 구조 시각화
- [ ] FC 36 vs CNN 4 파라미터 비교
- [ ] sparse / local / weight-shared / equivariant 4종 강조

## 9. 핵심 공식 카드
- 2D cross-correlation: $Y_{a,b} = \sum_{u,v} w_{u,v} X_{a+u, b+v}$
- AlexNet 차원: $H_{out} = (H_{in} + 2p - k)/s + 1$
- 행렬 표현: $A_{:, j} = f(e_j)$, 본 문제 $A \in \mathbb{R}^{4 \times 9}$
- 본 문제 $A$:
$$
A = \begin{bmatrix}
0 & 1 & 0 & 2 & 3 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 2 & 3 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 2 & 3 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 2 & 3
\end{bmatrix}
$$
- 중심 픽셀 열: $A_{:, 5} = (3, 2, 1, 0)^\top$ (커널을 180° 뒤집은 모양)
- 구조: BTTB (doubly block-Toeplitz)
- 파라미터: CNN 4 vs FC 36

## 10. 다른 퀴즈와의 연결
- **Q11 (1D conv 계산)**: 슬라이딩 곱-합의 1D 원형. 2D 는 차원만 늘었을 뿐 본질 동일.
- **Q12 (1D conv 행렬, Toeplitz)**: 1D Toeplitz 의 2D 일반화가 BTTB. "기저의 상을 열로" 는 1D 와 정확히 같은 절차이며, 평탄화 인덱싱만 추가됨.
- **CNN 강의 전체**: 본 퀴즈가 "2D conv = 매우 sparse 한 weight-shared FC" 의 가장 압축된 증명. AlexNet 의 차원 공식, im2col, transposed conv, FFT 가속, U-Net 디코더 모두 이 BTTB 그림에서 출발.

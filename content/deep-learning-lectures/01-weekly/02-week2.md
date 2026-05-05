---
title: "딥러닝이론 2주차 분석"
slug: week2
order: 2
---

# 딥러닝이론 2주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-2주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### Linear Algebra 노테이션 (가로/세로 stacking)
- **정의:** 일반적으로 vector는 column. 가로로 늘어놓으면 $[a_1, a_2, \dots, a_n]$ ($m\times n$ matrix), semicolon으로 vertical stack: $[a_1; a_2; \dots]$ ($mn \times 1$).
- **중요도:** ★★★★★ (5/10) [추론 보충: 이후 모든 매트릭스 식 표기에 사용]

### Inner Product (Matrix-Vector 곱과 일관성)
- **정의:** Column vector $v, w$의 내적은 $v^T w$. Matrix $\times$ vector도 "같은 위치 성분끼리" 곱셈으로 일관되게 이해 가능. $A v = \sum_i v_i a_i$ (column $a_i$ 들의 linear combination).
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 핵심 연산]

### Matrix Multiplication 시간복잡도
- **정의:** $A_{m\times n} B_{n\times p} = C_{m\times p}$. 각 원소 계산에 $n$번 곱, 전체 $O(mnp)$. Square matrix인 경우 $O(n^3)$.
- **맥락:** Strassen 등으로 $O(n^{2.8})$까지 줄이는 알고리즘도 있음. AI에서 매트릭스 곱이 가장 큰 비중이라 중요.
- **중요도:** ★★★★★★ (6/10) [명시적: 직관적으로 무겁다는 점 강조]

### Span / Linear Independence / Basis / Dimension
- **정의:**
  - Linear combination: $\sum a_i v_i$
  - Span($S$): $S$의 vector들의 모든 linear combination의 집합
  - Linear Independent: $\sum a_i v_i = 0 \Rightarrow$ 모든 $a_i = 0$
  - Basis: linear independent + spans the space
  - Dimension: basis의 개수
- **중요도:** ★★★★★★★ (7/10) [추론 보충: Rank-Nullity의 핵심 빌딩블록]

### Image $R(A)$ / Kernel(Null space) $N(A)$
- **정의:** $A: \mathbb{R}^n \to \mathbb{R}^m$ ($m\times n$ matrix)일 때
  - $R(A) = \{Ax : x\in \mathbb{R}^n\}$ — output side에 있음
  - $N(A) = \{x : Ax = 0\}$ — input side에 있음
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 시각적 직관 강조]

### Rank-Nullity Theorem
- **정의:** $m \text{(domain dim)} = \dim R(A) + \dim N(A) = \text{rank} + \text{nullity}$.
  - 주의: 강의 노테이션에서 "$m$ = input dim"
- **맥락:** Fundamental Theorem of Linear Algebra의 일부.
- **중요도:** ★★★★★★★★ (8/10) [추론 보충: 직접 예시로 검증, 핵심]

### Eigenvalue / Eigenvector
- **정의:** Square matrix $A$에 대해 $Av = \lambda v$ ($v\ne 0$)인 $v$, $\lambda$.
- **응용:** Image compression (큰 eigenvalue부터 살려서 근사), Google PageRank.
- **직관:** 큰 eigenvalue → 중요. 큰 것부터 몇 개만 써서 matrix를 근사 가능.
- **중요도:** ★★★★★★★★ (8/10) [명시적: "다양한 분야 응용", 시각 데모]

### SVD (Singular Value Decomposition)
- **정의:** Non-square matrix에 대한 eigen-analog. $A = U\Sigma V^T$. $\Sigma$의 대각 성분 = singular value, 0이 아닌 것의 개수 = rank.
- **응용:** PCA의 기본, 차원 축소, 압축.
- **중요도:** ★★★★★★ (6/10) [추론 보충: 적용은 풍부하나 깊이 다루지 않음]

### Orthogonal Matrix / Inverse
- **정의:** $U^TU = UU^T = I$인 square matrix. Inverse 계산은 $O(n^3)$로 비용 큼.
- **중요도:** ★★★★ (4/10)

### Vector를 vector로 미분 / Jacobian
- **정의:**
  - Vector를 scalar로 미분: 성분별 미분, 결과는 $n\times 1$
  - Scalar를 vector로 미분: gradient $\nabla$, 결과는 $1 \times n$ (transpose 안 함이 convention)
  - Vector를 vector로 미분: Jacobian matrix $J_{ij} = \partial u_i/\partial v_j$
- **중요도:** ★★★★★★★★★ (9/10) [명시적]
  - "이게 캘큘러스의 의미입니다... 매트릭스 2단계 중요합니다"

### Linear Approximation = Calculus의 핵심
- **정의:** 복잡한 함수를 1차로 근사. Newton's method가 대표 예시.
- **맥락:** "복잡한 함수는 다루기 어렵고 무조건 선형으로 근사할 수밖에 없다." 이게 매트릭스를 배운 이유 (매트릭스 = linear transformation).
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Newton's Method
- **정의:** $x_{t+1} = x_t - f(x_t)/f'(x_t)$. $f$를 선형근사하여 0을 찾는 iterative 방법.
- **응용:** $\sqrt{7}$ 계산. $f(x) = x^2 - 7$의 zero.
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 8주차에서 second-order method로 재등장]

### Softmax 함수
- **정의:** $p_i = \exp(g_i) / \sum_j \exp(g_j)$. Vector $g$를 probability vector $p$로 변환.
- **응용:** Attention, image classification 등 널리 사용.
- **핵심 성질:** $\partial p_i/\partial g_i = p_i(1-p_i)$, $\partial p_i/\partial g_j = -p_i p_j$ ($i\ne j$).
- **중요도:** ★★★★★★★★ (8/10) [명시적: 수업 직접 풀이]

### Grandmother Cell / Intelligence as Function
- **정의:** 뇌의 특정 cell이 grandmother input(시각/소리/냄새)에 fire한다는 가설. 함수 관점으로 intelligence를 이해.
- **맥락:** Function = thing을 algorithm으로 다루는 대상. 화면→행동 같은 모든 지능적 처리는 function.
- **중요도:** ★★★★★ (5/10) [추론 보충: 9주차 ZFNet 시각화로 다시 등장]

---

## 🔢 2. 수식 풀이 및 증명

### Rank-Nullity 검증 ($A=(1,1)$)
**문제/목표:** $A=(1,1)$ ($1\times 2$ matrix)에 대해 $R(A), N(A)$, rank, nullity를 구하고 $\text{rank}+\text{nullity}=2$ 확인.
**단계별 풀이:**
1. $Ax = x_1 + x_2$. $x_1=0$, $x_2$ 임의 → 모든 실수 가능. $R(A) = \mathbb{R}$, rank = 1.
2. $N(A) = \{x : x_1+x_2=0\}$ — 2차원에서 1차원 line. nullity = 1.
3. $1+1=2$ = domain dim. ✓

**결론:** Rank-nullity 만족.
**중요도:** ★★★★★★★★ (8/10) [명시적: 5분 시간 주고 직접 풀이시킴]

### $5\times 5$ 대각 행렬의 rank, nullity, eigen
**문제/목표:** $A = \text{diag}(4,3,2,1,0)$에 대해 $R(A), N(A)$, rank, nullity, eigenvalues 계산.
**단계별 풀이:**
1. $Ax = (4x_1, 3x_2, 2x_3, x_4, 0)$. 마지막 성분이 항상 0이므로 $R(A) = \{(y_1,y_2,y_3,y_4,0)\}$, rank = 4.
2. $N(A) = \{x: 4x_1=3x_2=2x_3=x_4=0\} = \{(0,0,0,0,x_5)\}$, nullity = 1. $4+1=5$. ✓
3. Eigenvalues: $\lambda_i = 4,3,2,1,0$ with eigenvectors $e_1,\dots,e_5$.

**중요도:** ★★★★★★ (6/10) [명시적: 칠판 풀이]

### Newton's Method로 $\sqrt{7}$ 계산
**문제/목표:** $f(x)=x^2-7=0$의 양수 근을 찾기.
**단계별 풀이:**
1. $f(x_0=3) = 2$, $f'(x_0)=6$. 선형근사 $y = 6(x-3)+2 = 6x-16$.
2. Zero: $x_1 = 16/6 = 8/3 \approx 2.667$.
3. 일반식 $x_{t+1} = x_t - (x_t^2-7)/(2x_t) = (x_t + 7/x_t)/2$.
4. 몇 번만 반복해도 소수점 깊은 자리까지 일치.

**결론:** Linear approximation으로 빠르게 수렴.
**중요도:** ★★★★★★★★ (8/10) [명시적: 직접 계산 + 8주차 재등장]

### Softmax 미분 ($p = \text{softmax}(g)$, $p, g \in \mathbb{R}^C$, 여기 $C=10$)
**문제/목표:** $\partial p / \partial g$ 계산.
**단계별 풀이:**
1. $p_i = e^{g_i}/S$, $S = \sum_j e^{g_j}$.
2. $i = j$: $\partial p_i/\partial g_i$ = $(\partial e^{g_i}/\partial g_i \cdot S - e^{g_i} \cdot \partial S/\partial g_i)/S^2 = (Se^{g_i} - e^{2g_i})/S^2 = (e^{g_i}/S)(1 - e^{g_i}/S) = p_i(1-p_i)$.
3. $i \ne j$: $\partial p_i/\partial g_j = (0\cdot S - e^{g_i}\cdot e^{g_j})/S^2 = -p_i p_j$.

**결론:** Jacobian $J_{ij} = p_i(\delta_{ij} - p_j)$.
**중요도:** ★★★★★★★★★ (9/10) [명시적]
  - "Softmax가 미분이 된다 — 이게 중요합니다."

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| Vector→Vector 미분 (Jacobian) | 9 | 정의 | "캘큘러스의 의미... 매트릭스 2단계 중요" |
| Linear approximation의 중심성 | 9 | 개념 | "복잡한 함수는 무조건 선형으로 근사할 수밖에 없다" |
| Softmax 미분 가능성 | 9 | 정리 | "이게 중요합니다 — 미분이 된다" |
| Rank-Nullity Theorem | 8 | 정리 | 직접 칠판 풀이 + 퀴즈 |
| Eigenvalue 큰 것 = 중요 | 8 | 개념 | 이미지 압축 데모, 다양한 응용 |
| Newton's Method | 7 | 방법 | 직접 풀이 + 8주차 재등장 |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. Rank-Nullity for $A=(1,1)$
**문제:**
> "랭크랑 nullity가 뭔지 답을 댓글로 적어주세요. $A=(1,1)$입니다."

**트리거 발언:** "5분 정도 드릴 테니까... 두 개에 대해서 풀어보세요"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $A: \mathbb{R}^2 \to \mathbb{R}$, $A(x_1,x_2) = x_1 + x_2$.
2. Image: 모든 실수. rank = 1.
3. Kernel: $\{(x_1,x_2): x_1+x_2=0\}$, 1차원 line. nullity = 1.
4. $1+1=2$ = input dim ✓.

**정답:** rank=1, nullity=1.

**해설:** Rank-nullity theorem의 가장 단순한 검증. Image는 output 공간의 부분, kernel은 input 공간의 부분이라는 점을 시각적으로 이해해야 함.

</details>

### Q2. $A=\text{diag}(4,3,2,1,0)$의 분석
**문제:**
> "이런 5차원, 5차원짜리 매트릭스가 있을 때 (1) image, (2) kernel, (3) eigenvalue를 다 계산해보시기 바랍니다."

**트리거 발언:** "직접 좀 할 수도 있겠지만 오늘은 이렇게 좀 제가 좀 하면서 봐보도록 하겠습니다."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Image: 마지막 성분이 항상 0이므로 $\{(y_1,y_2,y_3,y_4,0): y_i\in\mathbb{R}\}$. rank = 4.
2. Kernel: 처음 4개 성분이 모두 0인 vector $\{(0,0,0,0,x_5)\}$. nullity = 1.
3. Eigenvalues: 4,3,2,1,0 with corresponding $e_1,\dots,e_5$. $A_k$ ($k$개 큰 eigenvalue만 살린 근사)의 rank = $k$.

**정답:** rank=4, nullity=1, eigenvalues = $\{4,3,2,1,0\}$.

**해설:** 대각 행렬에서 rank = nonzero 대각 성분 개수 = nonzero eigenvalue 개수. SVD/eigen approximation의 직관 베이스.

</details>

### Q3. Softmax 미분
**문제:**
> "$p = \text{softmax}(g)$일 때 $\partial p/\partial g$를 계산하라. $p_1, p_2$를 $g_1, g_2$로 미분하는 두 가지 경우만이라도 해보세요."

**트리거 발언:** "10x10 차원으로 보낸 거고 100개 값을 다 채워야 될까... 시도만이라도 좀 하려고 하셨으면 좋겠습니다. 58분 정도까지 한번 해볼까요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $p_i = e^{g_i}/S$, $S = \sum_j e^{g_j}$.
2. **$i=j$일 때:** Quotient rule.
$\frac{\partial p_i}{\partial g_i} = \frac{e^{g_i}\cdot S - e^{g_i}\cdot e^{g_i}}{S^2} = \frac{e^{g_i}}{S}\left(1 - \frac{e^{g_i}}{S}\right) = p_i(1 - p_i)$.
3. **$i \ne j$일 때:** $\partial e^{g_i}/\partial g_j = 0$, $\partial S/\partial g_j = e^{g_j}$.
$\frac{\partial p_i}{\partial g_j} = \frac{0 - e^{g_i}\cdot e^{g_j}}{S^2} = -p_i p_j$.

**정답:** Jacobian $J_{ij} = p_i(\delta_{ij} - p_j)$.

**해설:** Softmax가 attention과 cross-entropy loss에서 항상 등장하는데, 이 미분 공식이 backprop의 핵심. $p$만으로 깔끔하게 표현된다는 점이 중요.

</details>

---

## 📎 5. 기타 참고사항

- 시작 멘트: 첫 시간 미니 알지브라 노테이션부터 진행, 선형대수 기본은 안다고 가정.
- 매트릭스 multiplication을 위한 정의 ($O(MNP)$ 시간) 강조.
- Outer product 언급은 스킵.
- Inverse는 $O(n^3)$, "느린 알고리즘"이라는 직관만 가져가라.
- SVD 계산법(power iteration 등)은 스킵, 의미만 전달.
- $Ax=b$ 시스템은 "완벽하게 알려져 있어서" 정리는 했지만 직접 다룰 때 다시.
- 마지막 30분 정도부터 Calculus 도입: 루트 7 계산 직접 해본 것이 핵심 동기.
- Vector를 scalar로/Scalar를 vector로 미분 표기 convention (transpose 여부) 설명.
- $\nabla$ (nabla) notation: scalar를 vector로 미분한 후 transpose해 column으로 만든 것.
- 다음 시간 예고: probability 기본, 특히 Bernoulli, Gaussian. Bayesian 관점이 다음 주의 핵심.
- 4주차에 Bayesian probability 본격 진행 예고.
- 행정: 녹화본 4번 추가 시청 권장. 7시 40분에 줌 오픈해서 실시간 진행.


---
title: "딥러닝 이론 기말 대비 종합 학습 가이드"
slug: dl-final-study-guide
order: 11
---

# 딥러닝 이론 기말 대비 종합 학습 가이드

> **시험 정보**: 주관식(퀴즈 유사), 답 도출 논리 과정 서술, 오프라인
> **채점 철학**: "답만 적으면 0점" — 모든 유도의 **각 단계에 "왜"를 글로** 설명해야 점수
> **킬러 포인트**: i.i.d → 로그 → 미분=0 유도 체인 (출제 ~35%)
> **범위**: 1~4주차 (선형대수, 미적분, 확률, 베이지안, ML/MAP, 정보이론)

---

## 목차

| Part | 주제 | 시험 비중 |
|------|------|---------|
| A | [선형대수 기초](#partA) | ★★★ |
| B | [고유값 및 행렬 분해](#partB) | ★★★ |
| C | [미분 및 최적화](#partC) | ★★★ |
| D | [확률론 기초](#partD) | ★★★ |
| E | [베이지안 확률과 추정](#partE) | ★★★★ |
| F | [ML/MAP과 모델 비교](#partF) | ★★★★ |
| G | [정보이론 기초](#partG) | ★★★ |
| H | [10대 핵심 유도 완전 재현](#partH) | ★★★★★ |

---

<a id="partA"></a>
## Part A. 선형대수 기초

> **마인드맵 출처**: 2주차 마인드맵(KR/EN), 2주차 선행영상 마인드맵(KR/EN)

### A.1 행렬 = 선형변환의 표현

**한 줄 결론**: 행렬은 "데이터를 변환하는 장치"이며, 각 열이 좌표축을 어디로 보내는지를 코딩한 것이다.

#### 입문 단계

행렬은 숫자를 직사각형으로 배열한 표다. 하지만 딥러닝에서 행렬의 진짜 의미는 **"공간의 변환"**이다.

비유하면: 행렬은 거울이 물체를 뒤집듯이, 데이터를 이동·회전·변형하는 기계다.

#### 중급 단계

$$A \in \mathbb{R}^{m \times n}: \quad L_A(v) = Av \quad (\mathbb{R}^n \to \mathbb{R}^m)$$

행렬 $A$의 각 열은 기저벡터 $e_i$의 임베딩이다:
$$A = [L(e_1) \quad L(e_2) \quad \cdots \quad L(e_n)]$$

**행렬-벡터 곱 = 열벡터들의 선형결합**:
$$Av = v_1 a_1 + v_2 a_2 + \cdots + v_n a_n$$

쉽게 말하면: "행렬에 벡터를 곱하는 건, 행렬의 열 벡터들을 가중합하는 것이다."

#### 고급 단계

- 선형 변환 $L$과 행렬 $A_L$ 사이에 **일대일 대응**: $L \mapsto A_L \mapsto L_{A_L} = L$
- 행렬곱 시간복잡도: $O(n^3)$, Strassen: $O(n^{2.807})$
- 역행렬 계산: $O(n^3)$ (LU 분해)

### A.2 표기법과 벡터 연산

| 표기 | 의미 | 예시 |
|------|------|------|
| $[a_1 \; a_2 \; \cdots]$ | 수평 스택 (horizontal) | 열벡터 나란히 |
| $[a_1; \; a_2; \; \cdots]$ | 수직 스택 (세미콜론) | 열벡터 세로로 |
| $\langle v, w \rangle = v^\top w$ | **내적** (Inner Product) | 유사도 측정 |
| $v w^\top$ | **외적** (Outer Product) | 랭크-1 행렬 |

**내적의 DL 핵심 역할**: CE Loss에서 $e_{y_i}^\top \log h(x_i)$ = 정답 클래스의 로그 확률만 골라내는 연산

### A.3 벡터 공간의 주요 개념

| 개념 | 정의 | 왜 중요한가 |
|------|------|-----------|
| **선형결합** | $\sum a_i v_i$ | 모든 NN 연산의 기본 |
| **스팬(Span)** | 벡터들의 모든 선형결합의 집합 | = 가능한 출력 공간 |
| **선형독립** | $\sum a_i v_i = 0 \Rightarrow$ 모든 $a_i = 0$ | 정보 중복 없음 |
| **기저(Basis)** | 선형독립인 생성 집합 | 최소한의 좌표계 |
| **차원(Dimension)** | 기저 벡터의 수 $\|B\|$ | 자유도 |

**남에게 설명하는 한 문장**: "기저는 공간을 빈틈없이 커버하면서도 겹치지 않는 최소 벡터 세트다."

### A.4 선형대수 기본정리 (Fundamental Theorem)

#### Range와 Null Space

| 부분공간 | 정의 | 의미 |
|---------|------|------|
| **Range** $\mathscr{R}(A)$ | $\{Av : v \in \mathbb{R}^n\}$ | $A$가 도달할 수 있는 출력 |
| **Kernel** $\mathscr{N}(A)$ | $\{v : Av = 0\}$ | $A$가 무시하는 입력 방향 |

#### Rank-Nullity 정리 (★★★ 10대 핵심 증명 #6)

$$\boxed{n = \text{rank}(A) + \text{nullity}(A)}$$

**쉽게 말하면**: "입력 공간의 차원 = 살아남는 차원(랭크) + 죽는 차원(널리티)"

**유도 과정 (시험 서술용)**:
1. $A \in \mathbb{R}^{m \times n}$에서 $A$에 의한 선형 변환 $L_A: \mathbb{R}^n \to \mathbb{R}^m$을 생각한다.
2. $\mathscr{N}(A)$의 기저 $\{v_1, ..., v_k\}$ ($k = \text{nullity}$)를 잡는다.
3. 이를 확장하여 $\mathbb{R}^n$의 기저 $\{v_1, ..., v_k, u_1, ..., u_r\}$ ($r = n-k$)를 만든다.
4. $\{Au_1, ..., Au_r\}$이 $\mathscr{R}(A)$의 기저임을 보인다 (선형독립 + 생성).
5. 따라서 $\text{rank}(A) = r = n - k = n - \text{nullity}(A)$ $\square$

#### 4개 부분공간의 직교 관계

$$\mathbb{R}^n = \mathscr{N}(A) \oplus \mathscr{R}(A^\top), \quad \mathscr{N}(A) \perp \mathscr{R}(A^\top)$$

### A.5 가역 행렬의 동치 조건

정방행렬 $A \in \mathbb{R}^{n \times n}$에 대해 다음은 **모두 동치** (TFAE):

> 1. $A^{-1}$ 존재 (가역)
> 2. $\text{rank}(A) = n$ (full rank)
> 3. $\text{null}(A) = 0$
> 4. $Av = 0 \Rightarrow v = 0$
> 5. $Ax = b$가 유일한 해
> 6. 모든 고유값 $\neq 0$
> 7. $\det(A) \neq 0$
> 8. 열(행)들이 선형독립

**핵심 킬러 요약**: "이 10개 조건 중 하나만 알면 나머지 9개를 전부 유도 가능. 연결고리는 Rank-Nullity."

---

<a id="partB"></a>
## Part B. 고유값 및 행렬 분해

> **마인드맵 출처**: 2주차(KR/EN), 선행영상(KR/EN) — "고유값/고유벡터", "SVD", "PageRank"

### B.1 고유값과 고유벡터 (★★★ 10대 핵심 증명 #7)

**정의**: 정방행렬 $A \in \mathbb{R}^{n \times n}$에 대해

$$\boxed{Av = \lambda v \quad (v \neq 0)}$$

#### 입문 단계
비유하면: 행렬 $A$라는 "변환 기계"에 벡터 $v$를 넣었는데, 방향은 안 바뀌고 크기만 $\lambda$배 되는 특별한 벡터가 고유벡터, 그 배율이 고유값이다.

#### 중급 단계

| 행렬 유형 | 고유값 특성 | 고유벡터 특성 |
|----------|-----------|------------|
| 대칭 ($A = A^\top$) | **실수** | **서로 직교** |
| PSD ($v^\top Av \geq 0$) | 비음수 ($\lambda \geq 0$) | 직교 |
| PD ($v^\top Av > 0$) | 양수 ($\lambda > 0$) | 직교, $A$ 가역 |
| 직교 ($U^\top U = I$) | $\|\lambda\| = 1$ | 노름 보존 |

**고유값의 의미**: 큰 고유값 = 그 방향이 중요하다 = 그 방향으로 데이터 분산이 크다

#### 고급 단계 — 스펙트럼 분해 (대칭 행렬)

$$A = U\Lambda U^\top = \sum_{i=1}^n \lambda_i u_i u_i^\top$$

- $U$: 직교행렬 (고유벡터들을 열로), $\Lambda$: 대각행렬 (고유값들)
- $A^{-1} = \sum_i \lambda_i^{-1} u_i u_i^\top$ (같은 고유벡터, 역수 고유값)

**이차형식의 기하학**: $f(x) = x^\top Ax = \sum_i \lambda_i y_i^2$ ($y = U^\top x$)
→ 등고선이 타원, 고유벡터가 주축 방향, $\lambda_i^{-1/2}$가 축 길이

### B.2 특이값 분해 (SVD)

$$\boxed{A = U\Sigma V^\top = \sum_{i=1}^{\min(m,n)} \sigma_i u_i v_i^\top}$$

| 구성요소 | 크기 | 의미 |
|---------|------|------|
| $U$ | $m \times m$ | 출력 공간의 직교 기저 |
| $\Sigma$ | $m \times n$ | 특이값 (중요도 순) |
| $V$ | $n \times n$ | 입력 공간의 직교 기저 |

**저랭크 근사**: $A_r = \sum_{i=1}^r \sigma_i u_i v_i^\top$ (상위 $r$개만 유지)

**응용**: 이미지 압축 (Rank 5, 15, 50... → 원본에 가까워짐), PCA, 차원 축소

### B.3 PageRank와 Power Method (응용)

**Google PageRank**: 웹을 거대한 행렬로 보고, 그 **주요 고유벡터**를 추출하여 페이지 중요도 산출

$$PR(A) = (1-d) + d \sum_i \frac{PR(T_i)}{C(T_i)}$$

**Power Method**: $u_t \leftarrow Au_{t-1}/\|Au_{t-1}\|$ 반복 → 최대 고유벡터로 수렴

**수렴 속도**: $O((\lambda_2/\lambda_1)^t)$ — spectral gap이 클수록 빠름

### B.4 노름과 Trace

| 노름 | 정의 | DL 응용 |
|------|------|---------|
| $\|v\|_2$ (L2) | $\sqrt{\sum v_i^2}$ | Weight decay (L2 정규화) |
| $\|v\|_1$ (L1) | $\sum |v_i|$ | Sparsity (LASSO) |
| $\|v\|_0$ (L0) | 0이 아닌 원소 수 | (비미분가능, 근사 사용) |
| $\|A\|_F$ (Frobenius) | $\sqrt{\sum a_{ij}^2} = \|\sigma\|_2$ | 행렬 크기 |
| $\|A\|_\sigma$ (Spectral) | $\sigma_{\max}(A)$ | Lipschitz 상수 |

**Trace 핵심 성질**:
- $\text{Tr}(AB) = \text{Tr}(BA)$ (순환)
- $\text{Tr}(A) = \sum_i \lambda_i$ (고유값의 합)
- $x^\top Ax = \text{Tr}(xx^\top A)$ (벡터 이차형식 ↔ Trace)

---

<a id="partC"></a>
## Part C. 미분 및 최적화

> **마인드맵 출처**: 2주차(KR/EN), 3주차 선행영상(KR/EN) — "벡터 미분", "체인룰", "Softmax 미분", "최적화"

### C.1 행렬 미적분 표기법 (Numerator Layout)

| 미분 유형 | 표기 | 결과 크기 | 읽는 법 |
|----------|------|---------|--------|
| 스칼라 → 벡터로 미분 | $\frac{\partial s}{\partial v}$ | **행벡터** $1 \times n$ | 그래디언트의 전치 |
| 그래디언트 | $\nabla_v s = (\frac{\partial s}{\partial v})^\top$ | **열벡터** $n \times 1$ | 가장 가파른 상승 방향 |
| 벡터 → 벡터로 미분 | $\frac{\partial u}{\partial v}$ | **야코비안** $m \times n$ | 입출력 관계의 행렬 |
| 2차 미분 | $\frac{\partial^2 s}{\partial v^2}$ | **헤시안** $n \times n$ | 곡률 (볼록성 판단) |

### C.2 핵심 미분 공식

| 함수 | $\frac{\partial}{\partial x}$ | 비고 |
|------|------|------|
| $a^\top x$ | $a^\top$ | 선형 |
| $x^\top Sx$ ($S$ 대칭) | $2x^\top S$ | 이차형식 |
| $\|x\|^2$ | $2x^\top$ | $S = I$ |
| $Ax$ | $A$ | 선형변환 |
| $\|Wx - y\|^2$ (by $W$) | $\nabla_W = 2(Wx-y)x^\top$ | 최소제곱 핵심 |

### C.3 체인룰 (Chain Rule) — 역전파의 수학적 기초

$$\frac{\partial f \circ g(x)}{\partial x} = \underbrace{\frac{\partial f}{\partial y}\bigg|_{y=g(x)}}_{d(z) \times d(y)} \cdot \underbrace{\frac{\partial g}{\partial x}}_{d(y) \times d(x)}$$

**왜 중요한가**: 딥러닝의 역전파(backpropagation)는 체인룰의 **반복 적용**이다.

- **JVP** (forward mode): $Jx$ — 입력 변화 → 출력 변화
- **VJP** (backward mode): $y^\top J = \frac{\partial(y^\top u)}{\partial v}$ — **역전파의 핵심** (하나의 스칼라 미분으로 효율적 계산)

**행렬곱의 순서가 중요한 이유**: 차원이 맞아야 하고, 곱하는 순서에 따라 효율이 달라진다.

### C.4 Softmax 야코비안 (★★★ 10대 핵심 증명 #5)

$$p_i = \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$$

#### 유도 (시험 서술용)

**Step 1**: $p_i = e^{z_i} \cdot s^{-1}$로 놓는다 ($s = \sum_k e^{z_k}$).

**Step 2**: $i = j$일 때 (자기 자신에 대한 미분)
- 몫의 미분법: $\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \cdot s - e^{z_i} \cdot e^{z_i}}{s^2} = p_i - p_i^2 = p_i(1 - p_i)$
- **왜**: 분자 $e^{z_i}$가 $z_i$에 의존하고, 분모 $s$도 $z_i$에 의존하므로 몫의 미분법 필요

**Step 3**: $i \neq j$일 때 (다른 클래스에 대한 미분)
- $\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot s - e^{z_i} \cdot e^{z_j}}{s^2} = -p_i p_j$
- **왜**: 분자 $e^{z_i}$가 $z_j$에 무관하므로 분자 미분 = 0, 분모만 미분

**결과**:
$$\boxed{\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top}$$

### C.5 최적화 기초 개념

| 개념 | 정의 | DL에서의 역할 |
|------|------|-------------|
| **임계점** (Stationary/Critical point) | $f'(x) = 0$ | 후보 최적값 |
| **Fermat 정리** | 내부 극값이면 $f'(x) = 0$ | 미분=0의 근거 |
| **Newton's Method** | $x_{n+1} = x_n - f(x_n)/f'(x_n)$ | 영점 찾기/최적화 |

**Newton의 최적화 버전**: $f(x) = \nabla L(x)$로 놓으면 → $\theta_{n+1} = \theta_n - H^{-1}\nabla L$

**Newton's Method의 Heron 예시** (page_091): $\sqrt{7}$ 계산
- $x_{t+1} = (x_t + 7/x_t)/2$, $x_0 = 3$ → 4번 반복으로 13자리 정확

### C.6 제약 최적화: 라그랑주 승수법

**문제**: $\min_{g(x)=0} f(x)$

**라그랑지안**: $\mathcal{L}(x, \lambda) = f(x) + \lambda \cdot g(x)$

**필요조건**: $\nabla_{x,\lambda}\mathcal{L} = 0$

**기하학적 의미**: 최적점에서 $\nabla f \parallel \nabla g$ (등고선과 제약 곡선이 접함)

**KKT 조건** (부등식 포함): Stationarity + Primal feasibility + Dual feasibility ($\mu \geq 0$) + Complementary slackness ($\mu \cdot h = 0$)

---

<a id="partD"></a>
## Part D. 확률론 기초

> **마인드맵 출처**: 3주차(KR), 3주차 선행영상(KR/EN) — "기초 개념 복습", "확률 분포", "CLT"

### D.1 기본 용어

| 용어 | 정의 | 예시 |
|------|------|------|
| **실험(Experiment)** | 아웃컴을 내뱉는 과정 | 동전 던지기 |
| **표본공간(Sample Space)** $S$ | 가능한 모든 아웃컴의 집합 | $\{H, T\}$ |
| **이벤트(Event)** | $S$의 부분집합 | $\{H\}$ |
| **확률변수(Random Variable)** | $X: S \to \mathbb{R}$ | 앞면=1, 뒷면=0 |

### D.2 확률의 정의

- **라플라스 정의**: $P(E) = |E|/|S|$ (유한, 등확률 가정)
- **조건부 확률**: $P(E|F) = P(E \cap F)/P(F)$ — "F가 주어졌을 때 E의 확률"
- **독립**: $P(E \cap F) = P(E) \cdot P(F)$ — **MLE 유도의 출발점 (i.i.d.)**

### D.3 주요 분포

#### 베르누이 분포 (결과가 두 가지인 가장 간단한 분포)

$$\text{Ber}(x; \theta) = \theta^x(1-\theta)^{1-x}, \quad x \in \{0, 1\}$$

매개변수 $\theta$ 하나로 분포가 결정된다.

**DL 연결**: 이진 분류에서 $p(y=1|x) = \sigma(w^\top x) = \theta$ → Bernoulli 가정 → **CE Loss 유도**

#### 가우시안(정규) 분포

$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)$$

**핵심 요소 3가지** (마인드맵에서 강조): **Exponential** + **Minus** + **Square**
→ 평균에서 멀어질수록 지수적으로 확률 감소

**왜 "정규(normal)"인가**:
1. 주어진 평균/분산에서 **최대 엔트로피** (라그랑주 승수법으로 유도)
2. **중심극한정리(CLT)**: i.i.d. 합 → 정규분포 수렴

**DL 연결**: 회귀에서 $y \sim \mathcal{N}(h(x), \sigma^2)$ → Gaussian 가정 → **MSE Loss 유도**

### D.4 기댓값과 분산

$$\mathbb{E}[X] = \sum_x x P(X=x), \quad \text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**핵심 성질**:
- 선형성: $\mathbb{E}[aX+b] = a\mathbb{E}[X]+b$
- 독립이면: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$
- **주의**: 독립 → 비상관 (역은 일반적으로 거짓)

### D.5 중심극한정리 (CLT)

$$Z_n = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

**조건**: i.i.d. 샘플, 유한 분산, 큰 샘플 사이즈
**결과**: 독립적 요소들의 합은 노말 분포를 따른다

**쉽게 말하면**: "수많은 독립적 원인이 합쳐진 결과는 종모양 곡선(정규분포)이 된다."

**AI 응용**: 가우시안 노이즈 가정의 정당화 근거 = CLT

---

<a id="partE"></a>
## Part E. 베이지안 확률과 추정 (★★★★ 시험 핵심)

> **마인드맵 출처**: 3주차(KR) — "베이지안 확률", "추정 방법론", "베이즈 정리"

### E.1 베이지안 vs 빈도주의

| | 베이지안 | 빈도주의 |
|---|---------|---------|
| 확률의 의미 | **가설에 대한 믿음**의 정도 | 사건의 상대 빈도 |
| 대상 | Hypothesis (가설) | Event (사건) |
| 학습 | Belief Update (믿음 갱신) | 반복 실험의 빈도 |

**핵심**: 베이지안 관점에서 학습이란 **데이터를 관측하여 Prior(사전 믿음)를 Posterior(사후 믿음)로 갱신**하는 것

$$\cdots \to \text{Data} \to \text{Belief} \to \text{Data} \to \text{Belief} \to \cdots$$

### E.2 베이즈 정리 (★★★ 10대 핵심 증명 #8)

$$\boxed{P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}}$$

| 항 | 이름 | 의미 | DL 대응 |
|----|------|------|---------|
| $P(H)$ | **Prior** (사전 확률) | 데이터 관측 전의 믿음 | Regularization |
| $P(E\|H)$ | **Likelihood** (우도) | 모델이 맞다면 데이터가 나올 확률 | MLE → Loss |
| $P(H\|E)$ | **Posterior** (사후 확률) | 데이터 관측 후 갱신된 믿음 | MAP |
| $P(E)$ | **Evidence** (증거) | 정규화 상수 | 보통 무시 |

**유도 (시험 서술용)**:
1. 조건부 확률 정의: $P(H|E) = P(H \cap E) / P(E)$
2. 대칭성: $P(H \cap E) = P(E|H) \cdot P(H) = P(H|E) \cdot P(E)$
3. 정리하면: $P(H|E) = P(E|H) \cdot P(H) / P(E)$ $\square$

**남에게 설명하는 한 문장**: "데이터를 보기 전의 믿음(Prior)에, 데이터가 주는 증거(Likelihood)를 곱하면, 데이터를 본 후의 믿음(Posterior)이 된다."

### E.3 MLE (Maximum Likelihood Estimation)

**핵심**: 라이클리우드를 최대화하는 매개변수 추출. **데이터에만** 의존.

#### 킬러 유도 체인 (★★★★★ 출제 35%)

**Step 1 — i.i.d. 가정 (왜?)**:
데이터 $\{x_1, ..., x_n\}$이 독립동일분포 → 결합확률이 **곱으로 분해**
$$P(E|H) = P(x_1, ..., x_n | \theta) = \prod_{i=1}^n p(x_i | \theta)$$
→ **왜 i.i.d.가 필요한가**: 곱으로 분해할 수 없으면 이후 모든 유도가 불가능

**Step 2 — 로그를 취하는 이유 (왜?)**:
$$\log P(E|H) = \sum_{i=1}^n \log p(x_i | \theta)$$
→ **왜 로그를 취하는가**: (1) 곱→합 변환으로 계산 편의, (2) 컴퓨터의 수치 언더플로 방지, (3) 로그는 단조증가이므로 최적값 불변

**Step 3 — 미분 = 0 (왜?)**:
$$\frac{\partial}{\partial \theta} \sum_i \log p(x_i | \theta) = 0 \quad \Rightarrow \quad \theta_{ML}$$
→ **왜 미분=0인가**: Fermat 정리 — 내부 극값의 필요조건

#### 구체적 유도 예시

**베르누이 MLE** (동전 던지기: $n$번 중 $k$번 앞면):
$$\text{loglik} = k\log\theta + (n-k)\log(1-\theta)$$
$$\frac{\partial}{\partial\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0 \quad \Rightarrow \quad \boxed{\theta_{ML} = k/n}$$

**가우시안 MLE**:
$$\text{loglik} = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2$$
$$\frac{\partial}{\partial\mu} = 0 \quad \Rightarrow \quad \boxed{\mu_{ML} = \frac{1}{n}\sum x_i}$$

**MLE의 단점**: 데이터가 적을 때 극단적 결과 (예: $n=k=3$ → $\theta_{ML} = 1$, 과적합!)

### E.4 MAP (Maximum A Posteriori)

**핵심**: 사후 확률을 최대화하는 매개변수 추출. **사전 지식 반영** → 과적합 방지.

$$\theta_{MAP} = \arg\max_\theta [\underbrace{\log P(E|\theta)}_{\text{loglik}} + \underbrace{\log P(\theta)}_{\text{logprior}}]$$

**핵심 관계**: $\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$

#### 사전분포에 따른 MAP 결과

| Prior | MAP 결과 | 특징 |
|-------|---------|------|
| 균등분포 $U(0,1)$ | $\theta_{MAP} = k/n$ = MLE | Prior 무시 (Uniform Prior) |
| $p(\theta) \propto \theta(1-\theta)$ | $\theta_{MAP} = (k+1)/(n+2)$ | 0.5 쪽으로 당김 |
| Strong prior ($m \to \infty$) | $\theta_{MAP} \to 0.5$ | 데이터보다 사전지식에 의존 |

**과적합 해결 예시**: $n=3, k=3$
- MLE: $\theta = 1$ (미래 모든 동전이 앞면? 비현실적!)
- MAP ($m=1$): $\theta = 4/5 = 0.8$ (합리적)

### E.5 ML vs MAP 비교표 (★★ 시험 출제 단골)

| 항목 | ML (Maximum Likelihood) | MAP (Maximum A Posteriori) |
|------|------------------------|---------------------------|
| 확률 | $P(E\|H)$ | $P(H\|E) \propto P(E\|H) \cdot P(H)$ |
| 근거 | 데이터/관측 (귀납) | 사전 지식 (연역) |
| Prior | Uniform (무시) | Non-uniform |
| 가설 공간 | 넓음 (높은 표현력) | 제한됨 (강한 inductive bias) |
| 아키텍처 | NN, MLP, Transformer | CNN, RNN, classical ML |
| 과적합 위험 | 높음 (데이터 적을 때) | 낮음 (사전 지식이 보호) |

---

<a id="partF"></a>
## Part F. ML/MAP과 모델 비교 (★★★★)

> **마인드맵 출처**: 4주차 — "시험 안내", "파라미터 추정 방법", "모델 비교", "정보 이론"

### F.1 시험 행정 안내 (마인드맵 4주차 상단)

- 시험 스타일: **퀴즈와 유사한 주관식**
- 평가 요소: **답 도출 논리 과정 서술**
- 시험 범위: 4주차까지 배운 내용

### F.2 NLL = CE = MSE (★★★ 통합 유도)

**Gaussian 가정 → MSE (10대 핵심 유도 #1)**:
1. $y_i \sim \mathcal{N}(h(x_i), \sigma^2)$ ← Gaussian noise 가정
2. i.i.d. → $P(y|X,\theta) = \prod_i \mathcal{N}(y_i; h(x_i), \sigma^2)$
3. log → $\text{loglik} = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (y_i - h(x_i))^2$
4. NLL 최소화 = $\sum_i (y_i - h(x_i))^2$ 최소화 = **MSE** $\square$

**Bernoulli 가정 → CE (10대 핵심 유도 #2)**:
1. $y_i \sim \text{Cat}(h(x_i))$ ← Categorical 분포 가정
2. i.i.d. → $P(y|X,\theta) = \prod_i [h(x_i)]_{y_i}$
3. log → $\text{loglik} = \sum_i \log [h(x_i)]_{y_i}$
4. NLL 최소화 = $-\sum_i \log [h(x_i)]_{y_i}$ = **CE Loss** $\square$

**남에게 설명하는 한 문장**: "CE와 MSE는 같은 목적함수(NLL)의 서로 다른 확률 가정 버전이다."

### F.3 MAP → L2 Regularization (10대 핵심 유도 #3)

**Prior**: $\theta \sim \mathcal{N}(0, \sigma_p^2 I)$ (Gaussian prior)

$$\log P(\theta) = -\frac{1}{2\sigma_p^2}\|\theta\|^2 + C$$

$$\theta_{MAP} = \arg\min_\theta \left[\underbrace{\text{NLL}(\theta)}_{\text{Loss}} + \underbrace{\frac{1}{2\sigma_p^2}\|\theta\|^2}_{\text{L2 Regularization}}\right]$$

**핵심 등식 체인**:
$$\text{Ridge} = \text{Weight Decay} = \ell_2\text{-Reg} = \text{Gaussian Prior} = \text{MAP}$$

마찬가지로: $\text{LASSO} = \ell_1\text{-Reg} = \text{Laplace Prior}$

### F.4 모델 및 아키텍처 비교

| 모델 | Inductive Bias | 데이터 요구량 | 특징 |
|------|---------------|-----------|------|
| **Linear Model** | 높음 | 적을 때 유리 | $h(x) = w^\top x$ |
| **Neural Network** | 낮음~중간 | 많을 때 유리 | 높은 Expressivity |
| **CNN** | 이미지 특화 (locality) | 중간 | Prior: 지역성, 평행이동 불변 |
| **Transformer** | 낮음 | **대량 데이터 필요** | 약한 Prior, 전역적 |

### F.5 Regression과 Gaussian의 연결

```
CLT (수많은 독립 원인의 합)
    → 노이즈가 Gaussian이라는 가정의 정당화
        → y ~ N(h(x), σ²)
            → NLL = MSE (+ const)
                → 최소제곱법의 확률론적 근거
```

---

<a id="partG"></a>
## Part G. 정보이론 기초

> **마인드맵 출처**: 4주차 — "정보 이론 기초"

### G.1 세 가지 핵심 측도

| 측도 | 수식 | 의미 |
|------|------|------|
| **엔트로피** | $H(p) = -\mathbb{E}_p[\log p(X)]$ | 분포 $p$의 **불확실성** |
| **교차 엔트로피** | $CE(p,q) = -\mathbb{E}_p[\log q(X)]$ | $p$ 기준으로 $q$를 사용하는 비용 |
| **KL 발산** | $KL(p\|q) = CE(p,q) - H(p) \geq 0$ | 두 분포 사이의 **거리** |

**핵심 관계**:
$$\boxed{KL(p\|q) = CE(p,q) - H(p)}$$

$H(p)$는 상수 → **CE 최소화 = KL 최소화** → NLL 최소화와 동치

### G.2 KL ≥ 0 증명 (10대 핵심 유도 #9)

$$KL(p\|q) = \mathbb{E}_p\left[\log\frac{p(X)}{q(X)}\right] = -\mathbb{E}_p\left[\log\frac{q(X)}{p(X)}\right]$$

Jensen 부등식 ($-\log$는 볼록):
$$\geq -\log\mathbb{E}_p\left[\frac{q(X)}{p(X)}\right] = -\log\sum_x p(x)\frac{q(x)}{p(x)} = -\log\sum_x q(x) = -\log 1 = 0 \quad \square$$

### G.3 가우시안 간 KL 발산

$$KL(\mathcal{N}_1\|\mathcal{N}_2) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} + \text{tr}(\Sigma_1\Sigma_2^{-1}) + (\mu_1-\mu_2)^\top\Sigma_2^{-1}(\mu_1-\mu_2) - D\right]$$

$\Sigma_2 = \beta I$이면: $KL \propto \|\mu_1 - \mu_2\|^2$ → **CE ≈ NLL ≈ MSE** (Gaussian일 때)

---

<a id="partH"></a>
## Part H. 10대 핵심 유도 — 시험 재현 체크리스트

> **채점 핵심**: 각 단계에 "왜 이 단계를 밟는지" 한 줄씩 적어야 점수

| # | 유도 | 시작 → 결과 | 핵심 "왜" |
|---|------|-----------|---------|
| 1 | **Gaussian → MSE** | $y \sim \mathcal{N}(h(x), \sigma^2)$ → MSE | 왜 log: 곱→합. 왜 제곱: Gaussian의 지수항 |
| 2 | **Bernoulli → CE** | $y \sim \text{Cat}(h(x))$ → CE | 왜 log: i.i.d.+수치안정. 왜 -: NLL |
| 3 | **MAP → L2** | $\theta \sim \mathcal{N}(0, \sigma^2)$ → weight decay | 왜 L2: Gaussian prior의 log = 제곱합 |
| 4 | **Softmax via Lagrangian** | 엔트로피+제약 → softmax | 왜 라그랑주: 등식 제약 최적화 |
| 5 | **Softmax Jacobian** | $\partial p/\partial z$ → $\text{diag}(p)-pp^\top$ | 왜 두 케이스: $i=j$ vs $i\neq j$ |
| 6 | **Rank-Nullity** | $n = \text{rank} + \text{nullity}$ | 왜: 기저 확장으로 차원 분해 |
| 7 | **스펙트럼 분해** | $A = U\Lambda U^\top$ | 왜: 대칭행렬의 고유벡터가 직교 |
| 8 | **베이즈 정리** | 조건부확률 정의 → $P(H\|E)$ | 왜: $P(A \cap B) = P(A\|B)P(B)$의 대칭성 |
| 9 | **KL ≥ 0** | Jensen 부등식 적용 | 왜 Jensen: $-\log$가 볼록 |
| 10 | **정규방정식** | $\nabla_w \|Xw-y\|^2 = 0$ → $X^\top X\hat{\beta} = X^\top y$ | 왜: quadratic이므로 미분=0이 충분조건 |

---

## 부록: Softmax via Lagrangian 완전 유도 (10대 핵심 유도 #4)

**문제**: $\max_p \sum_i p_i z_i + \tau H(p)$ s.t. $\sum_i p_i = 1$, $p_i \geq 0$

**Step 1** — 라그랑지안 구성 (**왜**: 등식 제약이 있으므로 라그랑주 승수법 사용):
$$\mathcal{L} = \sum_i p_i z_i - \tau \sum_i p_i \log p_i + \lambda(1 - \sum_i p_i)$$

**Step 2** — $p_i$로 편미분 = 0 (**왜**: Fermat 정리, 내부 극값의 필요조건):
$$\frac{\partial\mathcal{L}}{\partial p_i} = z_i - \tau(\log p_i + 1) - \lambda = 0$$

**Step 3** — $p_i$에 대해 풀기:
$$\log p_i = \frac{z_i - \lambda - \tau}{\tau} \quad \Rightarrow \quad p_i = \exp\left(\frac{z_i}{\tau}\right) \cdot \underbrace{\exp\left(\frac{-\lambda-\tau}{\tau}\right)}_{=1/Z}$$

**Step 4** — 정규화 조건 $\sum p_i = 1$ 적용 (**왜**: 확률의 공리):
$$Z = \sum_j \exp(z_j/\tau)$$

**결과**:
$$\boxed{p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)} = \text{softmax}(z/\tau)_i}$$

**$\tau$ (temperature)의 의미**: $\tau \to 0$이면 argmax(one-hot), $\tau \to \infty$이면 uniform

---

## 최종 3줄 요약: 결국 뭐가 중요한가

> 1. **유도 체인을 처음부터 끝까지 재현**하라 — i.i.d→로그→미분=0 (Gaussian→MSE, Bernoulli→CE, MAP→L2, Lagrangian→Softmax)
> 2. **모든 단계에 "왜"를 적어라** — 왜 i.i.d(곱 분해), 왜 로그(곱→합+수치안정), 왜 미분=0(Fermat), 왜 라그랑주(등식 제약)
> 3. **통합적 관점을 보여라** — CE/MSE/NLL/KL은 모두 같은 것(확률 가정만 다름), MAP=MLE+Prior=Loss+Regularization

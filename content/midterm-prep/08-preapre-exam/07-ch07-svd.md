---
title: "Ch.07 특이값 분해(SVD)와 저랭크 근사"
slug: ch07-svd
order: 7
---

# Ch.07 특이값 분해(SVD)와 저랭크 근사

---

## 1. 학습 목표

이 장을 마친 후 다음을 할 수 있어야 한다:

- 특이값(singular value)의 정의를 정확히 진술하고, $A^TA$의 고유값과의 관계를 설명할 수 있다
- 좌특이벡터(left singular vector)와 우특이벡터(right singular vector)를 구별하고 구할 수 있다
- 임의의 $m \times n$ 행렬에 대해 특이값 분해(SVD) $A = U\Sigma V^T$를 수행할 수 있다
- Frobenius 노름의 정의를 진술하고, 특이값과의 관계를 증명할 수 있다
- **Eckart–Young 정리를 진술하고, 최적 저랭크 근사를 계산할 수 있다**
- **$A^TA$의 고유값이 음이 아님을 증명할 수 있다** (PRIORITY PROOF)
- Frobenius 노름이 특이값 제곱합의 제곱근임을 증명할 수 있다
- PCA와 SVD의 연결 관계를 설명하고, 데이터 차원 축소에 활용할 수 있다
- SVD가 딥러닝(가중치 압축, 차원 축소, 잠재 의미 분석 등)에서 어떻게 사용되는지 설명할 수 있다

---

## 2. 필수 정의

### Definition 7.1 (특이값, Singular Value)

$m \times n$ 행렬 $A$에 대해, $A^TA$의 고유값을 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq 0$이라 하자. $A$의 **특이값**(singular value)은

$$\sigma_i = \sqrt{\lambda_i(A^TA)}, \quad i = 1, 2, \ldots, n$$

으로 정의된다. 관례적으로 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \geq 0$의 내림차순으로 정렬한다.

> **해석:** 특이값은 행렬 $A$가 벡터를 얼마나 "늘이는가"를 정량적으로 나타낸다. $\sigma_1$은 $A$에 의한 최대 신장률(stretching factor)이다.

**예시:**

$$A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$

$$A^TA = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$$

고유값은 $\lambda_1 = 9, \lambda_2 = 4$이므로 특이값은 $\sigma_1 = 3, \sigma_2 = 2$이다.

**비예시:**

$\lambda = -4$가 어떤 행렬 $B$의 고유값이라 하자. 그렇다면 $\sigma = \sqrt{-4}$는 실수가 아니므로 특이값이 될 수 없다. 그러나 이런 상황은 $A^TA$에 대해서는 발생하지 않는다 (Theorem 7.2에서 증명).

---

### Definition 7.2 (좌특이벡터와 우특이벡터, Left and Right Singular Vectors)

$m \times n$ 행렬 $A$의 SVD $A = U\Sigma V^T$에서:

- **우특이벡터**(right singular vector): $V$의 열벡터 $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n \in \mathbb{R}^n$. 이들은 $A^TA$의 정규직교(orthonormal) 고유벡터이다.
- **좌특이벡터**(left singular vector): $U$의 열벡터 $\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_m \in \mathbb{R}^m$. 이들은 $AA^T$의 정규직교 고유벡터이다.

$\sigma_i > 0$인 경우 다음 관계가 성립한다:

$$\mathbf{u}_i = \frac{1}{\sigma_i} A\mathbf{v}_i$$

> **해석:** 우특이벡터는 "입력 방향", 좌특이벡터는 "출력 방향"으로 이해할 수 있다. $A$는 입력 공간의 $\mathbf{v}_i$ 방향을 출력 공간의 $\mathbf{u}_i$ 방향으로 보내면서 크기를 $\sigma_i$배 한다.

**예시:**

$$A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$

$A^TA$의 고유벡터: $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $\mathbf{v}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ (우특이벡터)

$$\mathbf{u}_1 = \frac{1}{3}A\mathbf{v}_1 = \frac{1}{3}\begin{pmatrix} 3 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad \mathbf{u}_2 = \frac{1}{2}A\mathbf{v}_2 = \frac{1}{2}\begin{pmatrix} 0 \\ 2 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

(좌특이벡터)

**비예시:**

$A^TA$의 고유벡터이지만 정규화하지 않은 $\mathbf{v} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}$는 우특이벡터가 **아니다**. 특이벡터는 반드시 단위벡터($\|\mathbf{v}\| = 1$)여야 한다.

---

### Definition 7.3 (특이값 분해, Singular Value Decomposition, SVD)

임의의 $m \times n$ 행렬 $A$는 다음과 같이 분해할 수 있다:

$$A = U\Sigma V^T$$

여기서:
- $U \in \mathbb{R}^{m \times m}$: 직교행렬 (orthogonal matrix), $U^TU = UU^T = I_m$. 열벡터가 좌특이벡터.
- $\Sigma \in \mathbb{R}^{m \times n}$: 대각행렬 (diagonal matrix), 대각 성분이 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$.
- $V \in \mathbb{R}^{n \times n}$: 직교행렬, $V^TV = VV^T = I_n$. 열벡터가 우특이벡터.

> **해석:** SVD는 임의의 선형변환을 "회전($V^T$) → 스케일링($\Sigma$) → 회전($U$)"으로 분해한다. 고유값 분해와 달리 **정사각행렬이 아니어도** 적용 가능하다.

**예시:**

$$A = \begin{pmatrix} 4 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 4 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}^T$$

$3 \times 2$ 행렬도 SVD가 가능하다.

**비예시:**

$A = U\Sigma V^T$에서 $U$와 $V$가 직교행렬이 아닌 분해, 예컨대

$$\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

는 SVD가 아니다. $U = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$가 직교행렬이 아니기 때문이다.

---

### Definition 7.4 (Frobenius 노름, Frobenius Norm)

$m \times n$ 행렬 $A = (a_{ij})$에 대해, **Frobenius 노름**은

$$\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^2} = \sqrt{\text{tr}(A^TA)}$$

으로 정의된다. 여기서 $\text{tr}(\cdot)$은 행렬의 대각합(trace)이다.

> **해석:** Frobenius 노름은 행렬의 모든 성분을 하나의 긴 벡터로 나열했을 때의 유클리드 노름과 같다. 행렬 간의 "거리"를 재는 데 자연스럽게 사용된다.

**예시:**

$$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$

$$\|A\|_F = \sqrt{1^2 + 2^2 + 3^2 + 4^2} = \sqrt{1 + 4 + 9 + 16} = \sqrt{30}$$

**비예시:**

**스펙트럼 노름**(spectral norm, operator norm) $\|A\|_2 = \sigma_1$은 Frobenius 노름과 **다르다**. 위 행렬에서 $\|A\|_2 = \sigma_1 \approx 5.465$이고 $\|A\|_F = \sqrt{30} \approx 5.477$이다. 일반적으로 $\|A\|_2 \leq \|A\|_F$이다.

---

### Definition 7.5 (랭크-$k$ 근사, Rank-$k$ Approximation)

$A = U\Sigma V^T$가 $A$의 SVD이고 $\text{rank}(A) = r$일 때, $k \leq r$에 대한 **랭크-$k$ 근사**는

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

으로 정의된다. 이는 SVD에서 가장 큰 $k$개의 특이값과 대응하는 특이벡터만 사용한 행렬이다.

등가적으로, $U_k \in \mathbb{R}^{m \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$, $V_k \in \mathbb{R}^{n \times k}$를 처음 $k$개의 열/성분으로 구성하면:

$$A_k = U_k \Sigma_k V_k^T$$

> **해석:** 작은 $k$개의 "가장 중요한 성분"만으로 원래 행렬을 근사한다. 정보 압축의 핵심 원리이다.

**예시:**

$A$의 특이값이 $\sigma_1 = 10, \sigma_2 = 3, \sigma_3 = 0.1$일 때, 랭크-1 근사 $A_1 = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T$는 원래 행렬의 가장 지배적인 성분만 포착한다.

**비예시:**

무작위로 선택한 랭크-$k$ 행렬 $B$는 일반적으로 최적의 랭크-$k$ 근사가 **아니다**. SVD를 통해 얻은 $A_k$만이 Frobenius 노름 의미에서 최적이다 (Theorem 7.3).

---

## 3. 핵심 정리와 명제

### Theorem 7.1 (SVD 존재 정리, Existence of SVD)

> 임의의 $m \times n$ 실수 행렬 $A$에 대해 특이값 분해 $A = U\Sigma V^T$가 존재한다.

**왜 중요한가:** 고유값 분해와 달리 SVD는 **모든 행렬**에 대해 존재한다. 정사각행렬이 아니어도, 대각화가 불가능한 행렬이어도 SVD는 항상 가능하다. 이것이 SVD가 수치선형대수에서 "만능 도구"로 불리는 이유이다.

**어디에 쓰이는가:**
- 행렬의 랭크 결정 (수치적으로 안정적)
- 의사역행렬(pseudoinverse) 계산
- 최소제곱 문제의 풀이
- 딥러닝에서 가중치 행렬의 분석과 압축

---

### Theorem 7.2 ($A^TA$의 고유값의 비음수성)

> 임의의 $m \times n$ 실수 행렬 $A$에 대해, $A^TA$의 고유값은 모두 **음이 아니다** (non-negative).

**증명:**

$\lambda$를 $A^TA$의 고유값이라 하고, $\mathbf{v} \neq \mathbf{0}$를 대응하는 고유벡터라 하면:

$$A^TA\mathbf{v} = \lambda\mathbf{v}$$

양변에 왼쪽에서 $\mathbf{v}^T$를 곱하면:

$$\mathbf{v}^T A^T A\mathbf{v} = \lambda \mathbf{v}^T\mathbf{v}$$

좌변을 정리하면:

$$\mathbf{v}^T A^T A\mathbf{v} = (A\mathbf{v})^T(A\mathbf{v}) = \|A\mathbf{v}\|^2 \geq 0$$

우변에서 $\mathbf{v}^T\mathbf{v} = \|\mathbf{v}\|^2 > 0$ ($\mathbf{v} \neq \mathbf{0}$이므로)이므로:

$$\lambda = \frac{\|A\mathbf{v}\|^2}{\|\mathbf{v}\|^2} \geq 0 \qquad \blacksquare$$

**왜 중요한가:** 이 정리가 있어야 $\sigma_i = \sqrt{\lambda_i}$가 실수로 정의된다. 즉, 특이값이 항상 실수이고 음이 아닌 값이 되는 근본적인 이유이다.

**어디에 쓰이는가:** SVD의 존재성 증명의 핵심 단계이다. 또한 $A^TA$가 양의 준정부호(positive semi-definite)임을 보이는 것이기도 하다.

---

### Theorem 7.3 (Eckart–Young 정리, Eckart–Young Theorem)

> $A$의 SVD가 $A = U\Sigma V^T$이고 $\text{rank}(A) = r$일 때, $k \leq r$에 대해 랭크-$k$ 근사 $A_k = \sum_{i=1}^{k}\sigma_i \mathbf{u}_i\mathbf{v}_i^T$는 Frobenius 노름 의미에서 최적이다:
>
> $$A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$$
>
> 이때 근사 오차는:
>
> $$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

**왜 중요한가:** 데이터 압축과 차원 축소의 **이론적 근거**이다. "가장 큰 특이값부터 취하면 최적"이라는 사실은 직관적이지만, 엄밀히 증명되어야 한다. 이 정리가 바로 그 증명이다.

**어디에 쓰이는가:**
- 이미지 압축: 이미지를 행렬로 보고 저랭크 근사로 압축
- 추천 시스템: 사용자-아이템 행렬의 저랭크 근사
- 자연어 처리: 잠재 의미 분석(LSA)
- 딥러닝: 가중치 행렬의 저랭크 근사를 통한 모델 압축 (LoRA 등)

---

### Theorem 7.4 (Frobenius 노름과 특이값의 관계)

> 임의의 $m \times n$ 행렬 $A$의 특이값이 $\sigma_1, \sigma_2, \ldots, \sigma_p$ ($p = \min(m,n)$)일 때:
>
> $$\|A\|_F^2 = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_p^2 = \sum_{i=1}^{p}\sigma_i^2$$

**왜 중요한가:** Frobenius 노름을 성분별로 계산하지 않고도 특이값만으로 구할 수 있다. 또한 Eckart–Young 정리에서 근사 오차를 계산하는 데 직접적으로 사용된다.

**어디에 쓰이는가:**
- 저랭크 근사의 **설명력 비율** 계산: $\frac{\sigma_1^2 + \cdots + \sigma_k^2}{\sigma_1^2 + \cdots + \sigma_p^2}$
- 행렬의 에너지(energy) 분석
- 정규화(regularization)에서 핵 노름(nuclear norm) 등과의 비교

---

### Theorem 7.5 (PCA와 SVD의 연결)

> $X \in \mathbb{R}^{n \times d}$가 **중심화된**(centered, 각 열의 평균이 0) 데이터 행렬이라 하자 ($n$개의 데이터, $d$개의 특성). $X$의 SVD가 $X = U\Sigma V^T$일 때:
>
> 1. **공분산 행렬:** $C = \frac{1}{n-1}X^TX = \frac{1}{n-1}V\Sigma^2V^T$
> 2. **주성분(principal components):** $V$의 열벡터가 주성분 방향이다.
> 3. **주성분 점수:** $XV = U\Sigma$가 주성분 점수(PC scores)이다.
> 4. **분산 설명:** $i$번째 주성분의 분산은 $\frac{\sigma_i^2}{n-1}$이다.

**왜 중요한가:** PCA를 공분산 행렬의 고유값 분해로 구현할 수도 있지만, SVD를 사용하면 수치적으로 훨씬 안정적이다. 실제 PCA 구현은 거의 모두 SVD 기반이다.

**어디에 쓰이는가:**
- 데이터 전처리: 고차원 데이터의 차원 축소
- 시각화: 2D/3D로의 투영
- 딥러닝: 입력 데이터의 분포 분석, 오토인코더의 선형 한계와의 비교

---

## 4. 공식 및 수식 유도

### 4.1 $A^TA$에서 특이값이 나오는 원리

**출발점:** $A = U\Sigma V^T$ (SVD)라 하자.

$$A^TA = (U\Sigma V^T)^T(U\Sigma V^T) = V\Sigma^T U^T U\Sigma V^T$$

$U$가 직교행렬이므로 $U^TU = I$:

$$A^TA = V\Sigma^T\Sigma V^T = V \begin{pmatrix} \sigma_1^2 & & \\ & \sigma_2^2 & \\ & & \ddots \end{pmatrix} V^T$$

이것은 $A^TA$의 **고유값 분해**이다:
- $V$의 열벡터 $\mathbf{v}_i$가 $A^TA$의 고유벡터 (= 우특이벡터)
- $\sigma_i^2$가 $A^TA$의 고유값

따라서 $\sigma_i = \sqrt{\lambda_i(A^TA)}$이다.

마찬가지로:

$$AA^T = U\Sigma V^T V\Sigma^T U^T = U\Sigma\Sigma^T U^T = U\begin{pmatrix} \sigma_1^2 & & \\ & \sigma_2^2 & \\ & & \ddots \end{pmatrix}U^T$$

이것은 $AA^T$의 고유값 분해이며, $U$의 열벡터가 $AA^T$의 고유벡터 (= 좌특이벡터)이다.

> **핵심 통찰:** $A^TA$와 $AA^T$는 **같은 0이 아닌 고유값**을 공유한다. 크기가 다른 행렬임에도 불구하고!

---

### 4.2 Frobenius 노름과 특이값의 관계 유도

$$\|A\|_F^2 = \text{tr}(A^TA)$$

$A^TA = V\text{diag}(\sigma_1^2, \ldots, \sigma_n^2)V^T$이므로:

$$\text{tr}(A^TA) = \text{tr}(V\text{diag}(\sigma_1^2, \ldots, \sigma_n^2)V^T)$$

trace의 순환 성질 $\text{tr}(XYZ) = \text{tr}(ZXY)$를 사용하면:

$$= \text{tr}(V^TV \cdot \text{diag}(\sigma_1^2, \ldots, \sigma_n^2))$$

$V^TV = I$이므로:

$$= \text{tr}(\text{diag}(\sigma_1^2, \ldots, \sigma_n^2)) = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_n^2$$

따라서:

$$\boxed{\|A\|_F^2 = \sum_{i=1}^{n}\sigma_i^2}$$

---

### 4.3 랭크-$k$ 근사 $A_k$의 유도

$A$의 SVD를 외적 형태(outer product form)로 전개하면:

$$A = U\Sigma V^T = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

여기서 $r = \text{rank}(A)$이고, 각 $\sigma_i\mathbf{u}_i\mathbf{v}_i^T$는 랭크-1 행렬이다.

**확인:** $\mathbf{u}_i\mathbf{v}_i^T$는 $m \times n$ 행렬이고, 이것은 $m$차원 열벡터와 $n$차원 행벡터의 외적이므로 랭크가 정확히 1이다.

처음 $k$개 항만 취하면:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

근사 오차:

$$A - A_k = \sum_{i=k+1}^{r}\sigma_i \mathbf{u}_i\mathbf{v}_i^T$$

$$\|A - A_k\|_F^2 = \sum_{i=k+1}^{r}\sigma_i^2$$

마지막 등호는 $\{\mathbf{u}_i\mathbf{v}_i^T\}$들이 Frobenius 내적에 대해 직교하기 때문이다:

$$\langle \mathbf{u}_i\mathbf{v}_i^T, \mathbf{u}_j\mathbf{v}_j^T \rangle_F = \text{tr}((\mathbf{u}_i\mathbf{v}_i^T)^T (\mathbf{u}_j\mathbf{v}_j^T)) = \text{tr}(\mathbf{v}_i\mathbf{u}_i^T\mathbf{u}_j\mathbf{v}_j^T) = (\mathbf{u}_i^T\mathbf{u}_j)(\mathbf{v}_j^T\mathbf{v}_i) = \delta_{ij}$$

---

## 5. 증명

### 증명 5.1: $A^TA$의 고유값이 음이 아님

**증명 전략:** $A^TA$의 임의의 고유쌍 $(\lambda, \mathbf{v})$를 잡고, $\lambda \geq 0$임을 보인다. 핵심은 $\mathbf{v}^TA^TA\mathbf{v} = \|A\mathbf{v}\|^2 \geq 0$이라는 사실을 이용하는 것이다.

**증명:**

$\lambda$를 $A^TA$의 고유값, $\mathbf{v} \neq \mathbf{0}$를 대응하는 고유벡터라 하자. 정의에 의해:

$$A^TA\mathbf{v} = \lambda\mathbf{v} \quad \cdots (1)$$

$(1)$의 양변에 왼쪽에서 $\mathbf{v}^T$를 곱하면:

$$\mathbf{v}^T(A^TA)\mathbf{v} = \lambda(\mathbf{v}^T\mathbf{v}) \quad \cdots (2)$$

좌변을 변환한다:

$$\mathbf{v}^T(A^TA)\mathbf{v} = (\mathbf{v}^TA^T)(A\mathbf{v}) = (A\mathbf{v})^T(A\mathbf{v}) = \|A\mathbf{v}\|^2$$

벡터의 노름 제곱은 항상 음이 아니므로:

$$\|A\mathbf{v}\|^2 \geq 0 \quad \cdots (3)$$

우변에서 $\mathbf{v} \neq \mathbf{0}$이므로:

$$\mathbf{v}^T\mathbf{v} = \|\mathbf{v}\|^2 > 0 \quad \cdots (4)$$

$(2), (3), (4)$를 결합하면:

$$\lambda = \frac{\|A\mathbf{v}\|^2}{\|\mathbf{v}\|^2} \geq \frac{0}{\|\mathbf{v}\|^2} = 0$$

따라서 $\lambda \geq 0$이다. $\blacksquare$

> **보충:** 더 나아가 $\lambda = 0 \iff A\mathbf{v} = \mathbf{0} \iff \mathbf{v} \in \text{Null}(A)$이다.

---

### 증명 5.2: $\|A\|_F^2 = \sum \sigma_i^2$

**증명 전략:** Frobenius 노름의 trace 표현과 SVD를 결합하고, trace의 순환 성질을 활용한다.

**증명:**

Frobenius 노름의 정의에 의해:

$$\|A\|_F^2 = \text{tr}(A^TA)$$

$A = U\Sigma V^T$를 대입하면:

$$A^TA = V\Sigma^TU^TU\Sigma V^T = V\Sigma^T\Sigma V^T$$

여기서 $\Sigma^T\Sigma \in \mathbb{R}^{n \times n}$은 대각행렬이고 대각 성분이 $\sigma_1^2, \sigma_2^2, \ldots, \sigma_n^2$이다 ($n > r$인 경우 나머지는 0).

trace의 순환 성질을 적용하면:

$$\text{tr}(A^TA) = \text{tr}(V\Sigma^T\Sigma V^T) = \text{tr}(\Sigma^T\Sigma V^TV) = \text{tr}(\Sigma^T\Sigma)$$

$V^TV = I$이므로:

$$= \text{tr}\begin{pmatrix} \sigma_1^2 & & \\ & \sigma_2^2 & \\ & & \ddots \end{pmatrix} = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_n^2$$

따라서:

$$\|A\|_F^2 = \sum_{i=1}^{n}\sigma_i^2 \qquad \blacksquare$$

---

### 증명 5.3: SVD의 구성 과정 (Sketch Proof)

**증명 전략:** $A^TA$의 고유값 분해에서 출발하여 $V$, $\Sigma$, $U$를 순서대로 구성한다.

**증명 스케치:**

**Step 1.** $A^TA$는 $n \times n$ 대칭행렬이므로, 스펙트럼 정리(Spectral Theorem)에 의해 정규직교 고유벡터들로 구성된 직교행렬 $V$와 고유값 대각행렬로 분해된다:

$$A^TA = V\text{diag}(\lambda_1, \ldots, \lambda_n)V^T, \quad \lambda_1 \geq \cdots \geq \lambda_n \geq 0$$

고유값이 음이 아닌 것은 증명 5.1에 의한다. $\sigma_i = \sqrt{\lambda_i}$로 정의한다.

**Step 2.** $r = \text{rank}(A)$라 하면 $\sigma_1, \ldots, \sigma_r > 0$이고 $\sigma_{r+1} = \cdots = \sigma_n = 0$이다.

$i = 1, \ldots, r$에 대해 정의한다:

$$\mathbf{u}_i = \frac{1}{\sigma_i}A\mathbf{v}_i$$

**Step 3.** $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$이 정규직교임을 확인한다:

$$\mathbf{u}_i^T\mathbf{u}_j = \frac{1}{\sigma_i\sigma_j}(A\mathbf{v}_i)^T(A\mathbf{v}_j) = \frac{1}{\sigma_i\sigma_j}\mathbf{v}_i^TA^TA\mathbf{v}_j = \frac{1}{\sigma_i\sigma_j}\mathbf{v}_i^T(\sigma_j^2\mathbf{v}_j) = \frac{\sigma_j}{\sigma_i}\mathbf{v}_i^T\mathbf{v}_j = \frac{\sigma_j}{\sigma_i}\delta_{ij} = \delta_{ij}$$

**Step 4.** $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$을 $\mathbb{R}^m$의 정규직교기저로 확장하여 $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$을 얻는다 (Gram–Schmidt 등 사용).

**Step 5.** $U = [\mathbf{u}_1 \cdots \mathbf{u}_m]$, $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r, 0, \ldots, 0) \in \mathbb{R}^{m \times n}$으로 놓으면:

$$A\mathbf{v}_i = \sigma_i\mathbf{u}_i \quad (i = 1, \ldots, r), \quad A\mathbf{v}_i = \mathbf{0} \quad (i = r+1, \ldots, n)$$

이를 행렬 형태로 쓰면 $AV = U\Sigma$, 따라서 $A = U\Sigma V^T$. $\blacksquare$

---

## 6. 계산 예제와 단계별 풀이

### Example 7.1: 2×2 행렬의 SVD 전체 계산

**문제:** 다음 행렬의 SVD를 구하라.

$$A = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}$$

**풀이:**

**Step 1: $A^TA$를 계산한다.**

$$A^TA = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}^T \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}\begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix} = \begin{pmatrix} 13 & 12 \\ 12 & 13 \end{pmatrix}$$

(여기서 $A$가 대칭이므로 $A^T = A$, 따라서 $A^TA = A^2$.)

**Step 2: $A^TA$의 고유값을 구한다.**

$$\det(A^TA - \lambda I) = \det\begin{pmatrix} 13 - \lambda & 12 \\ 12 & 13 - \lambda \end{pmatrix} = (13-\lambda)^2 - 144 = 0$$

$$\lambda^2 - 26\lambda + 169 - 144 = 0$$

$$\lambda^2 - 26\lambda + 25 = 0$$

$$(\lambda - 25)(\lambda - 1) = 0$$

$$\lambda_1 = 25, \quad \lambda_2 = 1$$

**Step 3: 특이값을 구한다.**

$$\sigma_1 = \sqrt{25} = 5, \quad \sigma_2 = \sqrt{1} = 1$$

$$\Sigma = \begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}$$

**Step 4: 우특이벡터 $V$를 구한다 ($A^TA$의 고유벡터).**

$\lambda_1 = 25$:

$$(A^TA - 25I)\mathbf{v} = \begin{pmatrix} -12 & 12 \\ 12 & -12 \end{pmatrix}\mathbf{v} = \mathbf{0} \implies v_1 = v_2$$

정규화: $\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

$\lambda_2 = 1$:

$$(A^TA - I)\mathbf{v} = \begin{pmatrix} 12 & 12 \\ 12 & 12 \end{pmatrix}\mathbf{v} = \mathbf{0} \implies v_1 = -v_2$$

정규화: $\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$

$$V = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 5: 좌특이벡터 $U$를 구한다.**

$$\mathbf{u}_1 = \frac{1}{\sigma_1}A\mathbf{v}_1 = \frac{1}{5}\begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{5\sqrt{2}}\begin{pmatrix} 5 \\ 5 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

$$\mathbf{u}_2 = \frac{1}{\sigma_2}A\mathbf{v}_2 = \frac{1}{1}\begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

$$U = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 6: 검증.**

$$U\Sigma V^T = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}\begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix} 5 & 1 \\ 5 & -1 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 6 & 4 \\ 4 & 6 \end{pmatrix} = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix} = A \quad \checkmark$$

---

### Example 7.2: 랭크-1 근사 계산

**문제:** Example 7.1의 행렬 $A = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix}$의 랭크-1 근사 $A_1$을 구하라.

**풀이:**

$$A_1 = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T = 5 \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \end{pmatrix}$$

$$= 5 \cdot \frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 5/2 & 5/2 \\ 5/2 & 5/2 \end{pmatrix} = \begin{pmatrix} 2.5 & 2.5 \\ 2.5 & 2.5 \end{pmatrix}$$

**검증:** $\text{rank}(A_1) = 1$ (두 행이 동일). $\checkmark$

---

### Example 7.3: Frobenius 노름과 근사 오차 계산

**문제:** Example 7.1의 $A$와 $A_1$에 대해:
1. $\|A\|_F$를 두 가지 방법으로 계산하라.
2. 근사 오차 $\|A - A_1\|_F$를 구하라.
3. 랭크-1 근사의 설명력 비율을 구하라.

**풀이:**

**1. $\|A\|_F$ 계산:**

방법 1 (성분별):

$$\|A\|_F = \sqrt{3^2 + 2^2 + 2^2 + 3^2} = \sqrt{9 + 4 + 4 + 9} = \sqrt{26}$$

방법 2 (특이값):

$$\|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2} = \sqrt{25 + 1} = \sqrt{26} \quad \checkmark$$

**2. 근사 오차:**

$$A - A_1 = \begin{pmatrix} 3 & 2 \\ 2 & 3 \end{pmatrix} - \begin{pmatrix} 2.5 & 2.5 \\ 2.5 & 2.5 \end{pmatrix} = \begin{pmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{pmatrix}$$

$$\|A - A_1\|_F = \sqrt{0.25 + 0.25 + 0.25 + 0.25} = \sqrt{1} = 1$$

Eckart–Young 정리에 의한 계산:

$$\|A - A_1\|_F = \sqrt{\sigma_2^2} = \sigma_2 = 1 \quad \checkmark$$

**3. 설명력 비율:**

$$\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} = \frac{25}{26} \approx 96.2\%$$

랭크-1 근사만으로 원래 행렬 에너지의 약 96%를 설명할 수 있다.

---

### Example 7.4: PCA와의 연결 예시

**문제:** 다음 중심화된 데이터 행렬 $X$에 대해 PCA를 SVD로 수행하라.

$$X = \begin{pmatrix} 2 & 1 \\ -1 & 1 \\ -1 & -2 \end{pmatrix} \in \mathbb{R}^{3 \times 2}$$

(3개의 데이터 포인트, 2개의 특성. 각 열의 합이 0이므로 이미 중심화되어 있다.)

**풀이:**

**Step 1: $X^TX$를 계산한다.**

$$X^TX = \begin{pmatrix} 2 & -1 & -1 \\ 1 & 1 & -2 \end{pmatrix}\begin{pmatrix} 2 & 1 \\ -1 & 1 \\ -1 & -2 \end{pmatrix} = \begin{pmatrix} 6 & 3 \\ 3 & 6 \end{pmatrix}$$

**Step 2: 고유값을 구한다.**

$$\det\begin{pmatrix} 6 - \lambda & 3 \\ 3 & 6 - \lambda \end{pmatrix} = (6-\lambda)^2 - 9 = 0$$

$$\lambda^2 - 12\lambda + 27 = 0 \implies (\lambda - 9)(\lambda - 3) = 0$$

$$\lambda_1 = 9, \quad \lambda_2 = 3$$

특이값: $\sigma_1 = 3, \quad \sigma_2 = \sqrt{3}$

**Step 3: 주성분 방향 ($V$).**

$\lambda_1 = 9$: $\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ (제1주성분 방향)

$\lambda_2 = 3$: $\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ (제2주성분 방향)

**Step 4: 주성분 점수.**

$$XV = \begin{pmatrix} 2 & 1 \\ -1 & 1 \\ -1 & -2 \end{pmatrix} \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 3 & 1 \\ 0 & -2 \\ -3 & 1 \end{pmatrix}$$

이것이 $U\Sigma$에 해당한다.

**Step 5: 분산 설명.**

- 제1주성분 분산: $\frac{\sigma_1^2}{n-1} = \frac{9}{2} = 4.5$
- 제2주성분 분산: $\frac{\sigma_2^2}{n-1} = \frac{3}{2} = 1.5$
- 설명력 비율: $\frac{9}{9+3} = 75\%$ (제1주성분), $\frac{3}{9+3} = 25\%$ (제2주성분)

데이터를 1차원으로 축소하면 분산의 75%를 보존한다.

---

## 7. 자주 나오는 함정과 반례

### 함정 7.1: $A$의 고유값과 특이값을 혼동

**함정:** "$A$의 특이값은 $A$의 고유값의 절댓값이다."

**반례:** 이것은 **대칭행렬에서만** 성립한다.

$$A = \begin{pmatrix} 0 & 2 \\ 0 & 0 \end{pmatrix}$$

$A$의 고유값: $\lambda_1 = \lambda_2 = 0$

$A^TA = \begin{pmatrix} 0 & 0 \\ 2 & 0 \end{pmatrix}\begin{pmatrix} 0 & 2 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & 4 \end{pmatrix}$이므로

$A$의 특이값: $\sigma_1 = 2, \sigma_2 = 0$

고유값은 모두 0이지만 특이값은 0이 아닌 것이 있다!

> **올바른 이해:** 특이값은 $A^TA$의 고유값의 제곱근이지, $A$ 자체의 고유값과 직접 관련되지 않는다 (대칭행렬 제외).

---

### 함정 7.2: SVD의 유일성

**함정:** "SVD는 유일하다."

**사실:** 특이값은 유일하지만, $U$와 $V$는 **일반적으로 유일하지 않다**.

- 중복 특이값이 있으면 대응하는 특이벡터의 선택에 자유도가 있다
- $\sigma_i = 0$에 대응하는 $\mathbf{u}_i$는 완전히 자유롭다 (영공간의 정규직교기저 중 아무거나 선택 가능)
- $\mathbf{u}_i$와 $\mathbf{v}_i$의 부호를 동시에 바꿔도 SVD가 유지된다: $(-\mathbf{u}_i)(\sigma_i)(-\mathbf{v}_i)^T = \sigma_i\mathbf{u}_i\mathbf{v}_i^T$

---

### 함정 7.3: $\Sigma$의 크기를 틀리게 설정

**함정:** "$A$가 $m \times n$이면 $\Sigma$도 $n \times n$이다."

**올바른 이해:** $\Sigma \in \mathbb{R}^{m \times n}$이다. $A$와 **같은 크기**이다.

$$A \in \mathbb{R}^{3 \times 2} \implies U \in \mathbb{R}^{3\times 3}, \quad \Sigma \in \mathbb{R}^{3 \times 2}, \quad V \in \mathbb{R}^{2 \times 2}$$

$$\Sigma = \begin{pmatrix} \sigma_1 & 0 \\ 0 & \sigma_2 \\ 0 & 0 \end{pmatrix}$$

> **Thin SVD (축약형 SVD):** 실무에서는 $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$으로 축약하여 $A = U_r\Sigma_r V_r^T$로 쓰기도 한다. 이 경우 $U_r$과 $V_r$은 직교행렬이 아니라 열직교(orthonormal columns)만 만족한다.

---

### 함정 7.4: 저랭크 근사의 비유일성과 Eckart–Young의 조건

**함정:** "Eckart–Young 정리에 의해 최적 랭크-$k$ 근사는 유일하다."

**반례:** $\sigma_k = \sigma_{k+1}$일 때, 최적 랭크-$k$ 근사는 **유일하지 않다**.

예를 들어 $A = I_2$ (2×2 단위행렬)의 특이값은 $\sigma_1 = \sigma_2 = 1$이다. 랭크-1 근사로 $\mathbf{u}\mathbf{v}^T$ (임의의 단위벡터 $\mathbf{u} = \mathbf{v}$)를 취하면 오차가 모두 $\sqrt{1^2} = 1$로 같다. 따라서 최적 랭크-1 근사가 무한히 많다.

> Eckart–Young 정리의 유일성은 $\sigma_k > \sigma_{k+1}$일 때만 보장된다.

---

### 함정 7.5: PCA에서 중심화를 잊는 것

**함정:** "데이터 행렬 $X$에 바로 SVD를 적용하면 PCA가 된다."

**올바른 이해:** PCA를 위해서는 반드시 **각 열의 평균을 빼서 중심화**(centering)한 후에 SVD를 적용해야 한다. 중심화하지 않은 데이터에 SVD를 적용하면 첫 번째 특이벡터가 데이터의 평균 방향을 포착하게 되어, 분산이 가장 큰 방향과 다를 수 있다.

---

## 8. 시험형 서술 문제와 모범답안

### 문제 8.1

> **"임의의 $m \times n$ 실수 행렬 $A$에 대해, $A^TA$의 고유값이 모두 음이 아님을 증명하라."**

**모범답안:**

$\lambda$를 $A^TA$의 고유값, $\mathbf{v} \neq \mathbf{0}$를 대응하는 고유벡터라 하자.

$$A^TA\mathbf{v} = \lambda\mathbf{v}$$

양변의 왼쪽에 $\mathbf{v}^T$를 곱하면:

$$\mathbf{v}^TA^TA\mathbf{v} = \lambda\mathbf{v}^T\mathbf{v}$$

좌변: $(A\mathbf{v})^T(A\mathbf{v}) = \|A\mathbf{v}\|^2 \geq 0$

우변: $\mathbf{v}^T\mathbf{v} = \|\mathbf{v}\|^2 > 0$ ($\mathbf{v} \neq \mathbf{0}$이므로)

따라서:

$$\lambda = \frac{\|A\mathbf{v}\|^2}{\|\mathbf{v}\|^2} \geq 0 \qquad \blacksquare$$

---

### 문제 8.2

> **"다음 행렬의 SVD를 구하고, 랭크-1 근사를 계산하라."**
>
> $$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**모범답안:**

**Step 1:** $A^TA$를 계산한다.

$$A^TA = \begin{pmatrix} 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

**Step 2:** 고유값을 구한다.

$$\det\begin{pmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{pmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-3)(\lambda-1) = 0$$

$$\lambda_1 = 3, \quad \lambda_2 = 1$$

$$\sigma_1 = \sqrt{3}, \quad \sigma_2 = 1$$

**Step 3:** 우특이벡터를 구한다.

$\lambda_1 = 3$: $(A^TA - 3I)\mathbf{v} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$이므로 $v_1 = v_2$.

$$\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

$\lambda_2 = 1$: $(A^TA - I)\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\mathbf{v} = \mathbf{0}$이므로 $v_1 = -v_2$.

$$\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$

**Step 4:** 좌특이벡터를 구한다.

$$\mathbf{u}_1 = \frac{1}{\sqrt{3}}A\mathbf{v}_1 = \frac{1}{\sqrt{3}}\begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix} = \frac{1}{\sqrt{6}}\begin{pmatrix} 2 \\ 1 \\ 1 \end{pmatrix}$$

$$\mathbf{u}_2 = \frac{1}{1}A\mathbf{v}_2 = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 0 \end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix} 0 \\ -1 \\ 1 \end{pmatrix}$$

$\mathbf{u}_3$는 $\mathbf{u}_1, \mathbf{u}_2$에 직교하는 단위벡터로 확장한다. $\mathbf{u}_1 \times$-방향 확인 등을 통해:

$$\mathbf{u}_3 = \frac{1}{\sqrt{3}}\begin{pmatrix} -1 \\ 1 \\ 1 \end{pmatrix}$$

(검증: $\mathbf{u}_3^T\mathbf{u}_1 = \frac{1}{\sqrt{18}}(-2+1+1) = 0$ $\checkmark$, $\mathbf{u}_3^T\mathbf{u}_2 = \frac{1}{\sqrt{6}}(0-1+1) = 0$ $\checkmark$)

**Step 5:** SVD 결과.

$$A = U\Sigma V^T, \quad U = \begin{pmatrix} \frac{2}{\sqrt{6}} & 0 & \frac{-1}{\sqrt{3}} \\ \frac{1}{\sqrt{6}} & \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{3}} \\ \frac{1}{\sqrt{6}} & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{3}} \end{pmatrix}, \quad \Sigma = \begin{pmatrix} \sqrt{3} & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad V = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**Step 6:** 랭크-1 근사.

$$A_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^T = \sqrt{3} \cdot \frac{1}{\sqrt{6}}\begin{pmatrix} 2 \\ 1 \\ 1 \end{pmatrix} \cdot \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \end{pmatrix}$$

$$= \frac{\sqrt{3}}{\sqrt{12}}\begin{pmatrix} 2 & 2 \\ 1 & 1 \\ 1 & 1 \end{pmatrix} = \frac{1}{2}\begin{pmatrix} 2 & 2 \\ 1 & 1 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1/2 & 1/2 \\ 1/2 & 1/2 \end{pmatrix}$$

근사 오차: $\|A - A_1\|_F = \sqrt{\sigma_2^2} = 1$

설명력: $\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} = \frac{3}{4} = 75\%$

---

### 문제 8.3

> **"$\|A\|_F^2 = \sum_{i=1}^{p}\sigma_i^2$임을 증명하라." ($p = \min(m,n)$)**

**모범답안:**

$A = U\Sigma V^T$를 $A$의 SVD라 하자.

$$\|A\|_F^2 = \text{tr}(A^TA)$$

$A^TA$를 SVD로 표현하면:

$$A^TA = (U\Sigma V^T)^T(U\Sigma V^T) = V\Sigma^TU^TU\Sigma V^T = V(\Sigma^T\Sigma)V^T$$

여기서 $U^TU = I$ (직교행렬의 성질)을 사용했다.

$\Sigma^T\Sigma \in \mathbb{R}^{n \times n}$은 대각행렬이고 대각 성분이 $\sigma_1^2, \sigma_2^2, \ldots, \sigma_n^2$이다.

trace의 순환 성질 $\text{tr}(XY) = \text{tr}(YX)$를 적용하면:

$$\text{tr}(A^TA) = \text{tr}(V(\Sigma^T\Sigma)V^T) = \text{tr}(V^TV(\Sigma^T\Sigma)) = \text{tr}(\Sigma^T\Sigma)$$

$V^TV = I$이므로:

$$= \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_n^2 = \sum_{i=1}^{p}\sigma_i^2$$

(단, $i > p$인 특이값은 0이므로 합에 기여하지 않는다.) $\blacksquare$

---

### 문제 8.4

> **"Eckart–Young 정리를 정확히 진술하고, 이 정리가 딥러닝에서 왜 중요한지 설명하라."**

**모범답안:**

**정리 진술 (Eckart–Young):**

$A \in \mathbb{R}^{m \times n}$이고 $\text{rank}(A) = r$이라 하자. $A$의 SVD가 $A = U\Sigma V^T$일 때, $k \leq r$에 대해

$$A_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^T$$

는 다음을 만족한다:

$$\|A - A_k\|_F = \min_{\text{rank}(B) \leq k}\|A - B\|_F = \sqrt{\sum_{i=k+1}^{r}\sigma_i^2}$$

즉, 모든 랭크-$k$ 이하 행렬 중에서 $A_k$가 Frobenius 노름 의미에서 $A$에 가장 가까운 행렬이다.

**딥러닝에서의 중요성:**

1. **모델 압축:** 신경망의 가중치 행렬 $W \in \mathbb{R}^{m \times n}$을 $W_k = U_k\Sigma_kV_k^T$로 근사하면, 원래 $mn$개의 파라미터 대신 $(m+n)k$개만 저장하면 된다. $k \ll \min(m,n)$이면 메모리와 계산량이 크게 줄어든다.

2. **LoRA (Low-Rank Adaptation):** 대규모 사전 학습 모델의 미세조정에서, 가중치 변화 $\Delta W$를 저랭크 행렬 $BA$ ($B \in \mathbb{R}^{m \times k}$, $A \in \mathbb{R}^{k \times n}$)로 매개변수화한다. Eckart–Young 정리가 이 접근의 이론적 정당화를 제공한다.

3. **차원 축소:** 입력 데이터의 전처리(PCA)에서 설명력이 높은 성분만 유지할 때, 정보 손실이 최소화됨을 보장한다.

4. **정규화 효과:** 저랭크 근사는 노이즈(작은 특이값에 대응)를 제거하는 효과가 있어, 과적합 방지에 기여할 수 있다.

---

### 문제 8.5

> **"$A = \begin{pmatrix} 2 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix}$에 대해 SVD를 구하고, $A^TA$와 $AA^T$의 고유값이 0이 아닌 부분에서 일치함을 확인하라."**

**모범답안:**

**SVD 계산:**

$A$는 이미 "대각" 형태이므로 SVD를 직접 읽어낼 수 있다.

$$A^TA = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \end{pmatrix}\begin{pmatrix} 2 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 0 & 9 \end{pmatrix}$$

고유값: $\lambda_1 = 9, \lambda_2 = 4$ (내림차순 정렬)

특이값: $\sigma_1 = 3, \sigma_2 = 2$

$A^TA$의 고유벡터: $\mathbf{v}_1 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ ($\lambda = 9$), $\mathbf{v}_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ ($\lambda = 4$)

$$V = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

좌특이벡터:

$$\mathbf{u}_1 = \frac{1}{3}A\mathbf{v}_1 = \frac{1}{3}\begin{pmatrix} 0 \\ 3 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$$

$$\mathbf{u}_2 = \frac{1}{2}A\mathbf{v}_2 = \frac{1}{2}\begin{pmatrix} 2 \\ 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$$

$\mathbf{u}_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$ (직교 확장)

$$U = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}$$

**검증:** $U\Sigma V^T$:

$$\begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}\begin{pmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 0 & 2 \\ 3 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix} = A \quad \checkmark$$

**$AA^T$의 고유값 확인:**

$$AA^T = \begin{pmatrix} 2 & 0 \\ 0 & 3 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \end{pmatrix} = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 9 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$

고유값: $9, 4, 0$

$A^TA$의 고유값: $9, 4$

**0이 아닌 고유값이 $\{9, 4\}$로 일치한다.** $\checkmark$

이는 일반적인 사실이다: $A^TA$와 $AA^T$는 0이 아닌 고유값을 공유한다. 크기가 다른 행렬이지만 ($A^TA \in \mathbb{R}^{n \times n}$, $AA^T \in \mathbb{R}^{m \times m}$), 더 큰 쪽이 추가적인 0 고유값을 가질 뿐이다.

---

### 문제 8.6

> **"SVD를 이용하여 행렬의 랭크를 어떻게 결정할 수 있는지 설명하고, 수치적으로 왜 유용한지 논하라."**

**모범답안:**

**랭크 결정:** $A = U\Sigma V^T$에서 $\text{rank}(A)$는 **0이 아닌 특이값의 개수**와 같다.

$$\text{rank}(A) = |\{i : \sigma_i > 0\}|$$

이는 다음과 같이 증명된다:

$A = \sum_{i=1}^{r}\sigma_i\mathbf{u}_i\mathbf{v}_i^T$에서 각 $\sigma_i\mathbf{u}_i\mathbf{v}_i^T$는 랭크-1 행렬이고, $\{\mathbf{u}_i\}$와 $\{\mathbf{v}_i\}$가 각각 직교하므로 이 랭크-1 행렬들은 선형독립이다. 따라서 $\text{rank}(A) = r$.

**수치적 유용성:**

1. **행 사다리꼴의 문제:** 가우스 소거법으로 랭크를 구할 때, 반올림 오차로 인해 이론적으로 0이어야 할 피벗이 $10^{-16}$ 같은 아주 작은 값이 될 수 있다. 이를 0으로 볼지 말지의 판단이 모호하다.

2. **SVD의 장점:** SVD에서는 특이값이 크기순으로 정렬되므로, "큰 특이값"과 "수치적으로 무시할 만한 작은 특이값" 사이에 명확한 gap이 있는 경우가 많다. 예를 들어 $\sigma_1 = 100, \sigma_2 = 50, \sigma_3 = 0.00001$이면 **수치적 랭크**(numerical rank)는 2로 판단할 수 있다.

3. **임계값 설정:** 통상적으로 $\sigma_i < \epsilon \cdot \sigma_1$ (여기서 $\epsilon$은 기계 정밀도)인 특이값을 0으로 처리한다. 이는 `numpy.linalg.matrix_rank`가 사용하는 방법이다.

4. **안정성:** SVD 알고리즘은 수치적으로 후방 안정(backward stable)하므로, 계산된 특이값의 상대 오차가 기계 정밀도 수준으로 보장된다.

---

### 문제 8.7

> **"$A \in \mathbb{R}^{m \times n}$의 SVD가 $A = U\Sigma V^T$일 때, $A$의 네 가지 기본 부분공간(four fundamental subspaces)을 SVD의 관점에서 서술하라."**

**모범답안:**

$\text{rank}(A) = r$이고, 특이값이 $\sigma_1 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots$일 때:

1. **열공간(Column space)** $\text{Col}(A)$:

$$\text{Col}(A) = \text{span}\{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_r\}$$

차원: $r$. 좌특이벡터 중 0이 아닌 특이값에 대응하는 것들이 열공간의 정규직교기저를 형성한다.

2. **영공간(Null space)** $\text{Null}(A)$:

$$\text{Null}(A) = \text{span}\{\mathbf{v}_{r+1}, \mathbf{v}_{r+2}, \ldots, \mathbf{v}_n\}$$

차원: $n - r$. 우특이벡터 중 특이값이 0인 것들이 영공간의 정규직교기저이다. ($A\mathbf{v}_i = \sigma_i\mathbf{u}_i = \mathbf{0}$ for $i > r$.)

3. **행공간(Row space)** $\text{Row}(A) = \text{Col}(A^T)$:

$$\text{Row}(A) = \text{span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_r\}$$

차원: $r$. 0이 아닌 특이값에 대응하는 우특이벡터들이 행공간의 정규직교기저이다.

4. **좌영공간(Left null space)** $\text{Null}(A^T)$:

$$\text{Null}(A^T) = \text{span}\{\mathbf{u}_{r+1}, \mathbf{u}_{r+2}, \ldots, \mathbf{u}_m\}$$

차원: $m - r$. 특이값 0에 대응하는 좌특이벡터들이다.

**요약:**

| 부분공간 | SVD 기저 | 차원 |
|---------|---------|------|
| $\text{Col}(A)$ | $\mathbf{u}_1, \ldots, \mathbf{u}_r$ | $r$ |
| $\text{Null}(A)$ | $\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n$ | $n-r$ |
| $\text{Row}(A)$ | $\mathbf{v}_1, \ldots, \mathbf{v}_r$ | $r$ |
| $\text{Null}(A^T)$ | $\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m$ | $m-r$ |

SVD는 네 가지 기본 부분공간의 정규직교기저를 한 번의 분해로 동시에 제공한다는 점에서, 행렬을 이해하는 가장 완전한 도구이다. $\blacksquare$

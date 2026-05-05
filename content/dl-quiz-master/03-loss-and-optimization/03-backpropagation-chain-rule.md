---
title: 'Quiz 10 — Backpropagation Chain Rule'
description: 'Softmax+CE 미분: ∂L/∂z = p - e_y'
draft: false
---

## 0. 한 줄 요약

분류 신경망의 출력층 손실 그래디언트 $\partial L / \partial z$ 를 chain rule 로 합성하면 — 세 국지 미분 $\partial L/\partial p_y = -1/p_y$, $\partial p_y/\partial p = e_y^\top$, $\partial p/\partial z = \mathrm{diag}(p) - pp^\top$ 을 곱하고 $e_y^\top \mathrm{diag}(p) = p_y e_y^\top$, $e_y^\top pp^\top = p_y p^\top$ 로 행벡터-행렬 곱을 풀면 — 깔끔하게 $\partial L / \partial z = p - e_y$ ("예측 확률 - 정답 원-핫") 가 닫힌 형태로 나온다. 이는 GLM canonical link 의 일반 정리의 한 사례다.

---

## 1. 문제 (정확한 출제 형태)

신경망
$$
x \xrightarrow{W_1} z_1 \xrightarrow{\mathrm{ReLU}} x_1 \xrightarrow{W_2} z \xrightarrow{\mathrm{softmax}} p \xrightarrow{e_y^\top} p_y \xrightarrow{-\log} L
$$

에 대해 다음 세 편미분을 구하고, chain rule 로 합성하여 $\partial L / \partial z = p - e_y$ 임을 보여라.

(1) $\partial L / \partial p_y$
(2) $\partial p_y / \partial p$
(3) $\partial p / \partial z$

---

## 2. 출제 의도와 시험 가치

이 문제는 다음 능력을 측정한다.

1. **국지 (local) Jacobian 의 정확한 작성** — 각 화살표 단위 연산의 미분을 정확히 쓴다.
2. **차원 정합 (dimension matching)** — 스칼라/벡터/행렬의 곱셈 차원을 의식적으로 다룬다.
3. **Chain rule 의 합성** — 행벡터·행렬 곱셈으로 표현되는 jacobian product 의 대수 조작.
4. **닫힌 형태 (closed form) 의 의미 파악** — $p - e_y$ 가 단순한 표현이 된다는 사실의 본질적 의미 (GLM canonical link, exponential family).

추가로 이 결과는 backprop 의 **출발점**이다. 출력층 그래디언트가 이렇게 닫혀 있어 backward pass 가 효율적이며, 이후 $\partial L/\partial W_2$, $\partial L/\partial x_1$, ... 로 한 단계씩 거슬러 적용된다.

---

## 3. 사전 개념 (모든 수학 도구)

### 3.1 Softmax

$z\in\mathbb{R}^C$ 에 대해
$$
p_i = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}
$$
$\sum_i p_i = 1$ (확률 분포).

### 3.2 Softmax Jacobian (Quiz 3 결과)

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

행렬 형태:
$$
\boxed{\;\frac{\partial p}{\partial z} = \mathrm{diag}(p) - pp^\top\;}
$$

### 3.3 원-핫 벡터 $e_y$

정답 클래스 $y\in\{1,\dots,C\}$ 에 대해 $e_y\in\mathbb{R}^C$ 는 $y$ 번째 성분만 1, 나머지 0:
$$
(e_y)_c = \delta_{y,c}
$$

성질:
- $e_y^\top p = p_y$ ($y$ 번째 성분 추출).
- $e_y^\top \mathrm{diag}(p) = p_y \, e_y^\top$ (대각행렬과의 곱: $y$ 위치만 $p_y$ 로 스케일).
- $e_y^\top (pp^\top) = (e_y^\top p)\,p^\top = p_y\,p^\top$ (스칼라 추출 후 외적).

### 3.4 Cross-Entropy Loss

정답 $y$ 에 대해
$$
L = -\log p_y = -\log(e_y^\top p)
$$
한 변수 함수의 표준 미분: $\frac{d}{du}(-\log u) = -1/u$.

### 3.5 Chain rule (벡터 형태)

스칼라 $L$, 중간 변수 $u\in\mathbb{R}^a$, 입력 $v\in\mathbb{R}^b$:
$$
\underbrace{\frac{\partial L}{\partial v}}_{1\times b} = \underbrace{\frac{\partial L}{\partial u}}_{1\times a}\cdot\underbrace{\frac{\partial u}{\partial v}}_{a\times b}
$$
관례: $\partial L / \partial v$ 는 행벡터 ($1 \times b$). 마지막에 전치하여 열벡터로 표현.

### 3.6 차원 정합 표

| 양 | 형태 | 차원 |
|-----|------|-----|
| $L$ | 스칼라 | $1\times 1$ |
| $p_y$ | 스칼라 | $1\times 1$ |
| $p$ | 열벡터 | $C\times 1$ |
| $z$ | 열벡터 | $C\times 1$ |
| $\partial L / \partial p_y$ | 스칼라 | $1\times 1$ |
| $\partial p_y / \partial p$ | 행벡터 | $1\times C$ |
| $\partial p / \partial z$ | 행렬 | $C\times C$ |
| $\partial L / \partial z$ | 행벡터 (전치하면 열) | $1\times C$ |

---

## 4. 풀이 (모든 단계, 등호 근거)

### 4.1 (1) $\partial L / \partial p_y$

$L = -\log p_y$ 는 $p_y$ 의 한 변수 함수.

$$
\frac{\partial L}{\partial p_y} = -\frac{1}{p_y}
$$

근거: $\frac{d}{du}(-\log u) = -1/u$.

$$
\boxed{\;\frac{\partial L}{\partial p_y} = -\frac{1}{p_y}\;}
$$

---

### 4.2 (2) $\partial p_y / \partial p$

$p_y = e_y^\top p = \sum_{c=1}^C (e_y)_c\, p_c$ 는 $p$ 의 선형함수 (정확히 $y$ 번째 성분 추출).

각 성분 미분:
$$
\frac{\partial p_y}{\partial p_c} = (e_y)_c = \delta_{y,c}
$$

행벡터로:
$$
\boxed{\;\frac{\partial p_y}{\partial p} = e_y^\top \in \mathbb{R}^{1\times C}\;}
$$

($y$ 번째 성분이 1, 나머지 0 인 행벡터.)

---

### 4.3 (3) $\partial p / \partial z$

Quiz 3 결과 재활용:

$$
\boxed{\;\frac{\partial p}{\partial z} = \mathrm{diag}(p) - pp^\top \in \mathbb{R}^{C\times C}\;}
$$

성분 형태로 검증: $(\mathrm{diag}(p))_{ij} = p_i\delta_{ij}$, $(pp^\top)_{ij} = p_ip_j$. 따라서 $(i,j)$ 성분 = $p_i\delta_{ij} - p_ip_j = p_i(\delta_{ij} - p_j)$. ✓

---

### 4.4 Chain rule 합성

차원 정합 확인: $(1\times 1)\cdot(1\times C)\cdot(C\times C) \to (1\times C)$. OK.

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p_y}\cdot\frac{\partial p_y}{\partial p}\cdot\frac{\partial p}{\partial z} = \left(-\frac{1}{p_y}\right) e_y^\top \bigl(\mathrm{diag}(p) - pp^\top\bigr)
$$

#### Step 1. 행벡터-행렬 곱을 분배

$$
e_y^\top (\mathrm{diag}(p) - pp^\top) = e_y^\top \mathrm{diag}(p) - e_y^\top pp^\top
$$

#### Step 2. 첫째 항: $e_y^\top \mathrm{diag}(p)$

$\mathrm{diag}(p)$ 는 대각행렬. 행벡터 $e_y^\top$ 의 $y$ 번째 성분만 1 이므로 곱하면 대각의 $y$ 번째 원소 $p_y$ 만 추출되어 $y$ 위치에 들어간다.
$$
e_y^\top \mathrm{diag}(p) = p_y\, e_y^\top
$$

성분 검증: $\bigl(e_y^\top \mathrm{diag}(p)\bigr)_j = \sum_i (e_y)_i (p_i\delta_{ij}) = (e_y)_j p_j = \delta_{y,j} p_j$. $j = y$ 일 때 $p_y$, 그 외 0. → $p_y\,e_y^\top$. ✓

#### Step 3. 둘째 항: $e_y^\top pp^\top$

결합법칙으로 $(e_y^\top p)\, p^\top$. $e_y^\top p = p_y$ (스칼라).
$$
e_y^\top pp^\top = p_y\, p^\top
$$

#### Step 4. 합성

$$
e_y^\top (\mathrm{diag}(p) - pp^\top) = p_y\, e_y^\top - p_y\, p^\top = p_y\bigl(e_y^\top - p^\top\bigr)
$$

#### Step 5. $-1/p_y$ 인수 곱

$$
\frac{\partial L}{\partial z} = \left(-\frac{1}{p_y}\right)\cdot p_y\bigl(e_y^\top - p^\top\bigr) = -(e_y^\top - p^\top) = p^\top - e_y^\top
$$

#### Step 6. 전치하여 열벡터 형태로

$$
\boxed{\;\frac{\partial L}{\partial z} = p - e_y \in \mathbb{R}^C\;}
$$

---

## 5. 검증

### 5.1 차원 검증

- 합성 단계마다 차원이 정합 ($1\times 1 \to 1\times C \to 1\times C$).
- 최종 결과 $p - e_y$ 는 $C$-차원 벡터. $z$ 와 같은 차원. ✓

### 5.2 합 보존 검증

$\sum_c p_c = 1$ 이고 $\sum_c (e_y)_c = 1$. 따라서
$$
\sum_c \left(\frac{\partial L}{\partial z}\right)_c = \sum_c p_c - \sum_c (e_y)_c = 1 - 1 = 0
$$

이는 softmax 가 합 1 의 제약 분포 위에 있음의 결과 — gradient 가 simplex 의 접공간에 머문다.

### 5.3 정성적 검증

- 정답 클래스 $y$ 의 그래디언트: $(p - e_y)_y = p_y - 1 < 0$. → $z_y$ 를 키우면 $L$ 감소. ✓
- 오답 클래스 $c \ne y$: $(p - e_y)_c = p_c - 0 = p_c > 0$. → $z_c$ 를 키우면 $L$ 증가. ✓
- $p = e_y$ (완벽 예측) 이면 그래디언트 = 0. ✓

### 5.4 성분 형태로 직접 검증

$$
\frac{\partial L}{\partial z_j} = \frac{\partial}{\partial z_j}\bigl(-\log p_y\bigr) = -\frac{1}{p_y}\cdot \frac{\partial p_y}{\partial z_j}
$$
$$
\frac{\partial p_y}{\partial z_j} = p_y(\delta_{y,j} - p_j)
$$
$$
\frac{\partial L}{\partial z_j} = -\frac{1}{p_y}\cdot p_y(\delta_{y,j} - p_j) = p_j - \delta_{y,j} = (p - e_y)_j \quad\checkmark
$$

---

## 6. 일반화·통찰

### 6.1 의미: "예측 확률 - 정답 원-핫"

분류 오차의 직관적 정의 그 자체. 그래디언트가 이 형태로 닫혀 있어:
- 구현이 단순 (한 줄 빼기).
- 수치적으로 안정.
- backward pass 가 매우 효율적.

### 6.2 GLM canonical link 일반정리

(loss, output activation) 의 정합 쌍은 모두 $\partial L / \partial z = \hat y - y$ 형태:

| 분포 | activation | loss | 그래디언트 |
|------|-----------|------|----------|
| Gaussian | identity | MSE/2 | $\hat y - y$ |
| Bernoulli | sigmoid | BCE | $\hat p - y$ |
| Categorical | softmax | CE | $p - e_y$ |
| Poisson | exp | Poisson NLL | $\hat\lambda - y$ |

이는 exponential family + canonical link 의 일반 정리. 본 문제는 그 categorical 사례.

### 6.3 backprop 의 출발점

$\partial L / \partial z = p - e_y$ 가 출력층 그래디언트. 이후:
$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z}\, x_1^\top = (p - e_y)\, x_1^\top
$$
$$
\frac{\partial L}{\partial x_1} = W_2^\top\frac{\partial L}{\partial z} = W_2^\top(p - e_y)
$$
$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial x_1}\odot \mathbb{1}[z_1 > 0] \quad (\text{ReLU 미분})
$$
$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1}\, x^\top
$$
이렇게 한 단계씩 거슬러 적용 — autograd 라이브러리의 수학적 토대.

### 6.4 미니배치 일반화

배치 $B$ 에 대해 $z\in\mathbb{R}^{B\times C}$, $p\in\mathbb{R}^{B\times C}$, $E_y\in\mathbb{R}^{B\times C}$ (각 행 원-핫).

$$
\frac{\partial L_{\text{mean}}}{\partial z} = \frac{1}{B}(p - E_y)
$$

---

## 7. 시험 출제 변형 5가지

1. **Sigmoid + BCE 버전**: $z\in\mathbb{R}$, $\hat p = \sigma(z)$, $L = -y\log\hat p - (1-y)\log(1-\hat p)$. $\partial L/\partial z = \hat p - y$ 를 chain rule 로 유도.
2. **MSE + identity 버전**: $L = \frac{1}{2}(z - y)^2$. $\partial L/\partial z = z - y$. GLM 유추 확인.
3. **Softmax Jacobian 자체 유도**: 본 문제 (3) 의 답을 인용 없이 직접 유도하라.
4. **두 단계 더 backprop**: $\partial L / \partial W_2$ 와 $\partial L / \partial x_1$ 를 추가로 계산.
5. **Label smoothing**: $e_y$ 대신 smoothed target $\tilde e_y = (1-\epsilon)e_y + \frac{\epsilon}{C}\mathbf{1}$ 일 때 $\partial L/\partial z = p - \tilde e_y$ 를 보여라.

---

## 8. 백지 재현 체크리스트

- [ ] 네트워크 다이어그램 7단계 ($x \to z_1 \to x_1 \to z \to p \to p_y \to L$).
- [ ] $\partial L/\partial p_y = -1/p_y$ (한 변수 미분).
- [ ] $\partial p_y/\partial p = e_y^\top$ (선형 추출, 행벡터).
- [ ] $\partial p/\partial z = \mathrm{diag}(p) - pp^\top$ (Quiz 3 결과).
- [ ] 차원 정합 표 ($1\times 1$, $1\times C$, $C\times C$ → $1\times C$).
- [ ] $e_y^\top \mathrm{diag}(p) = p_y\, e_y^\top$ (대각행렬 곱).
- [ ] $e_y^\top pp^\top = p_y\, p^\top$ (결합법칙 + $e_y^\top p = p_y$).
- [ ] $p_y(e_y^\top - p^\top)$ 정리 후 $-1/p_y$ 인수와 곱셈.
- [ ] 부호 정리: $-(e_y^\top - p^\top) = p^\top - e_y^\top$.
- [ ] 전치하여 $\partial L/\partial z = p - e_y$.
- [ ] 검증: 합 = 0, 정답 클래스 음수, 오답 클래스 양수.
- [ ] (보너스) 성분 형태 $\partial L/\partial z_j = p_j - \delta_{y,j}$.

---

## 9. 핵심 공식 카드

```
세 국지 미분:
  ∂L / ∂p_y = -1 / p_y
  ∂p_y / ∂p = e_y^T              (1 × C)
  ∂p / ∂z = diag(p) - p p^T       (C × C)

핵심 항등식 (행벡터-행렬 곱):
  e_y^T diag(p) = p_y e_y^T
  e_y^T (p p^T) = p_y p^T

Chain rule 합성:
  ∂L/∂z = (-1/p_y) · e_y^T · (diag(p) - p p^T)
        = (-1/p_y) · [p_y e_y^T - p_y p^T]
        = (-1/p_y) · p_y (e_y^T - p^T)
        = p^T - e_y^T

전치:
  ∂L/∂z = p - e_y     ∈ R^C

성분 형태:
  ∂L/∂z_j = p_j - δ_{y,j}

GLM 일반정리:
  Gaussian + identity + MSE     → ŷ - y
  Bernoulli + sigmoid + BCE     → p̂ - y
  Categorical + softmax + CE    → p - e_y
```

---

## 10. 다른 퀴즈와의 연결

- **Quiz 3 (Softmax Jacobian)**: $\partial p/\partial z = \mathrm{diag}(p) - pp^\top$ 를 본 문제에서 그대로 재활용. Q3 → Q10 의 직접 의존.
- **Quiz 5/6/7 (Bernoulli MLE/MAP)**: NLL 미분이 $\hat p - y$ 형태. Sigmoid + BCE 버전. Q10 의 categorical 버전과 GLM 일반정리로 연결.
- **Quiz 8 (Gaussian KL → MSE)**: Gaussian + identity 버전의 $\hat y - y$ 와 같은 GLM 결과. Q8 은 분포 거리 관점, Q10 은 그래디언트 관점.
- **Quiz 11 (Convolution)**: 다음 단계로 conv 층의 backprop. 본 문제의 출력층 그래디언트가 입력으로 들어간다.
- **공통 메시지**: 출력층의 손실 그래디언트는 (loss, activation) 짝이 정합이면 항상 "예측 - 정답" 으로 닫힌다. 이는 13문제 전체가 사실상 하나의 그래디언트를 분해해 본 것에 가깝다는 통찰의 핵심 사례.

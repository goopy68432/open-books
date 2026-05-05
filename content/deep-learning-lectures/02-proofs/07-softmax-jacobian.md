---
title: "06. Softmax 미분 (Jacobian) — 완전 유도"
slug: softmax-jacobian
order: 7
---

# 06. Softmax 미분 (Jacobian) — 완전 유도

> **출제 근거**: 2주차 ★9 + 퀴즈 6번, 중간고사 Q4 직접 출제 (★★★★★)
> **시험 출제 방식**: \"Derive $\partial s_i / \partial z_j$ for the softmax function. Express the Jacobian in matrix form.\"

---

## 1. 왜 시험에 나오는가

- Softmax 미분은 **case 분리 ($i = j$ vs $i \neq j$)** 가 필요해 학생들이 자주 틀림.
- Backprop 의 가장 흔한 마지막 layer.
- Cross-entropy 와 결합하면 $\hat p - y$ 라는 깔끔한 결과 — [09 Backprop](09_Backprop_ChainRule.md) 의 출발점.

---

## 2. 사전 수학 (중1 → 대학원)

### 2.1 [중1] 분수의 미분

$\frac{1}{x}$ 을 $x$ 로 미분 = $-\frac{1}{x^2}$.

### 2.2 [고1] Quotient Rule

$$
\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}
$$

### 2.3 [고2] Chain Rule

$f(g(x))$ 의 미분 = $f'(g(x)) \cdot g'(x)$.

### 2.4 [대1] 지수함수 미분

$$
\frac{d}{dx} e^x = e^x, \qquad \frac{\partial}{\partial z_j} e^{z_i} = \begin{cases} e^{z_i} & i = j \\ 0 & i \neq j \end{cases}
$$

> ⚠️ **핵심**: $e^{z_i}$ 는 $z_j$ 에 대해 $i=j$ 일 때만 살아남음. 이게 case 분리의 원인.

### 2.5 Kronecker Delta

$$
\delta_{ij} \;=\; \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}
$$

case 분리를 한 식으로 쓰는 도구.

### 2.6 Softmax 정의

$\mathbf{z} = (z_1, \ldots, z_K) \in \mathbb{R}^K$ 에 대해:

$$
s_i \;:=\; \mathrm{softmax}(\mathbf{z})_i \;=\; \frac{e^{z_i}}{\sum_{m=1}^K e^{z_m}}
\tag{*}
$$

**기호 해체:**

| 기호 | 의미 |
|------|------|
| $z_i$ | $i$ 번째 logit (NN 의 raw 출력) |
| $e^{z_i}$ | 양수화 (negative logit도 양수로) |
| $\sum_m e^{z_m}$ | 정규화 분모 (모든 클래스 합 = 1 보장) |
| $s_i$ | $i$ 번째 클래스 확률, $\sum_i s_i = 1$ |

> 💡 \"왜 exp 를 쓰나?\": (1) 양수화 (2) 미분 좋음 (3) max 의 부드러운 근사 (큰 logit이 dominant).

---

## 3. 유도 — Quotient Rule 정공법

### Step 1 — 표기 정리

$$
s_i = \frac{a_i}{S}, \quad a_i := e^{z_i}, \quad S := \sum_{m=1}^K e^{z_m} = \sum_m a_m
$$

### Step 2 — Quotient Rule 적용

$z_j$ 로 미분 (한 변수에 대해):

$$
\frac{\partial s_i}{\partial z_j} \;=\; \frac{\partial}{\partial z_j} \frac{a_i}{S} \;=\; \frac{(\partial a_i/\partial z_j)\, S - a_i \, (\partial S/\partial z_j)}{S^2}
\tag{1}
$$

### Step 3 — 분자 각 항 계산

#### (a) $\partial a_i / \partial z_j$

$$
\frac{\partial a_i}{\partial z_j} = \frac{\partial e^{z_i}}{\partial z_j} = \begin{cases} e^{z_i} = a_i & i = j \\ 0 & i \neq j \end{cases} = \delta_{ij}\, a_i
$$

(2.4절 그대로)

#### (b) $\partial S / \partial z_j$

$S = \sum_m e^{z_m}$ 이고 $z_j$ 로 미분 시 $m = j$ 항만 살아남음:

$$
\frac{\partial S}{\partial z_j} = e^{z_j} = a_j
$$

### Step 4 — 합성

$$
\frac{\partial s_i}{\partial z_j} \;=\; \frac{\delta_{ij} a_i \cdot S - a_i \cdot a_j}{S^2} \;=\; \frac{\delta_{ij} a_i}{S} - \frac{a_i a_j}{S^2}
$$

$\frac{a_i}{S} = s_i$ 와 $\frac{a_j}{S} = s_j$ 적용:

$$
\boxed{\;
\frac{\partial s_i}{\partial z_j} \;=\; \delta_{ij}\, s_i - s_i s_j \;=\; s_i(\delta_{ij} - s_j)
\;}
\tag{2}
$$

---

## 4. Case-by-Case 검증

### Case 1: $i = j$ (대각 성분)

$\delta_{ii} = 1$:

$$
\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)
$$

> 익숙한 형태! Sigmoid 미분 $\sigma(z)(1-\sigma(z))$ 와 동일 모양 — softmax는 sigmoid의 다클래스 확장이라 자연스러움.

### Case 2: $i \neq j$ (비대각)

$\delta_{ij} = 0$:

$$
\frac{\partial s_i}{\partial z_j} = -s_i s_j
$$

> 음수! \"$j$의 logit을 키우면 $i$의 확률은 줄어든다\" — softmax의 경쟁 구조 반영.

---

## 5. Jacobian 행렬 형태

$$
J \;=\; \frac{\partial \mathbf{s}}{\partial \mathbf{z}} \in \mathbb{R}^{K \times K}, \quad J_{ij} = s_i(\delta_{ij} - s_j)
$$

이를 행렬로:

$$
J \;=\; \mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top
$$

**Line-by-Line:**

| 항 | 의미 | 어디서 |
|------|------|--------|
| $\mathrm{diag}(\mathbf{s})$ | 대각에 $s_1, s_2, \ldots, s_K$ | $\delta_{ij}\, s_i$ → $i=j$ 일 때만 $s_i$ |
| $-\mathbf{s}\mathbf{s}^\top$ | $(i,j)$ 성분 = $-s_i s_j$ | $-s_i s_j$ 항 |

**검증** (예시 $K=3$, $\mathbf{s} = (s_1, s_2, s_3)$):

$$
\mathrm{diag}(\mathbf{s}) = \begin{pmatrix} s_1 & 0 & 0 \\ 0 & s_2 & 0 \\ 0 & 0 & s_3 \end{pmatrix}, \quad \mathbf{s}\mathbf{s}^\top = \begin{pmatrix} s_1^2 & s_1 s_2 & s_1 s_3 \\ s_2 s_1 & s_2^2 & s_2 s_3 \\ s_3 s_1 & s_3 s_2 & s_3^2 \end{pmatrix}
$$

$$
J = \begin{pmatrix} s_1 - s_1^2 & -s_1 s_2 & -s_1 s_3 \\ -s_2 s_1 & s_2 - s_2^2 & -s_2 s_3 \\ -s_3 s_1 & -s_3 s_2 & s_3 - s_3^2 \end{pmatrix} = \begin{pmatrix} s_1(1-s_1) & -s_1 s_2 & -s_1 s_3 \\ \cdots & \cdots & \cdots \end{pmatrix}
$$

대각엔 $s_i(1-s_i)$, 비대각엔 $-s_i s_j$ — Case 1, 2 와 일치 ✅.

---

## 6. 다른 유도법 — Log 미분 (간결)

### Step 1: $\log s_i$ 표현

$$
\log s_i = z_i - \log\!\left(\sum_m e^{z_m}\right) = z_i - \log S
$$

### Step 2: 미분

$$
\frac{\partial \log s_i}{\partial z_j} = \frac{\partial z_i}{\partial z_j} - \frac{1}{S}\frac{\partial S}{\partial z_j} = \delta_{ij} - \frac{a_j}{S} = \delta_{ij} - s_j
$$

### Step 3: $\frac{\partial \log s_i}{\partial z_j} = \frac{1}{s_i}\frac{\partial s_i}{\partial z_j}$ (chain rule)

따라서:

$$
\frac{\partial s_i}{\partial z_j} = s_i \cdot (\delta_{ij} - s_j)
$$

> ✅ 같은 결과, 더 짧음. \"log를 한 번 통과시키는 게 미분에 유리\"라는 일반 원리.

---

## 7. Softmax + CE 결합 미분 (Backprop 핵심 결과)

손실 $L = -\sum_j y_j \log s_j$, $\mathbf{y}$ one-hot. $z_k$ 로 미분:

### Step 1: Chain rule

$$
\frac{\partial L}{\partial z_k} = \sum_i \frac{\partial L}{\partial s_i}\cdot\frac{\partial s_i}{\partial z_k}
$$

### Step 2: $\partial L/\partial s_i$

$$
\frac{\partial L}{\partial s_i} = -\frac{y_i}{s_i}
$$

### Step 3: 합치기

$$
\frac{\partial L}{\partial z_k} = \sum_i \left(-\frac{y_i}{s_i}\right) \cdot s_i(\delta_{ik} - s_k) = -\sum_i y_i (\delta_{ik} - s_k)
$$

### Step 4: 분배

$$
= -\sum_i y_i \delta_{ik} + s_k \sum_i y_i = -y_k + s_k \cdot 1 = s_k - y_k
$$

(마지막에서 $\sum_i y_i = 1$ — one-hot 합이 1)

$$
\boxed{\; \frac{\partial L}{\partial z_k} \;=\; s_k - y_k \;=\; \hat p_k - y_k \;}
$$

> 🎯 **시험 핵심**: \"Softmax + Cross-Entropy의 logit 에 대한 gradient는 예측 확률 - 정답 (one-hot) 의 차이\" — 매우 간단한 형태.

---

## 8. 모범 답안 템플릿

```
[Setup]
Softmax: s_i = e^{z_i} / S, where S = Σ_m e^{z_m}.
Goal: compute J_{ij} = ∂s_i / ∂z_j.

[Method — quotient rule]
∂s_i/∂z_j = [(∂e^{z_i}/∂z_j) · S - e^{z_i} · (∂S/∂z_j)] / S²

  ∂e^{z_i}/∂z_j = δ_{ij} e^{z_i}        (only nonzero when i=j)
  ∂S/∂z_j      = e^{z_j}

So
  ∂s_i/∂z_j = (δ_{ij} e^{z_i} S - e^{z_i} e^{z_j}) / S²
            = δ_{ij} (e^{z_i}/S) - (e^{z_i}/S)(e^{z_j}/S)
            = s_i δ_{ij} - s_i s_j
            = s_i (δ_{ij} - s_j).

[Cases]
  i = j:  ∂s_i/∂z_i = s_i (1 - s_i)            (analog of sigmoid')
  i ≠ j:  ∂s_i/∂z_j = -s_i s_j                  (cross-class competition)

[Matrix form]
  J = diag(s) - s sᵀ.

[Coupling with cross-entropy (bonus)]
For L = -Σ_j y_j log s_j with y one-hot,
  ∂L/∂z_k = s_k - y_k = p̂_k - y_k. ∎
```

---

## 9. 자주 틀리는 함정

1. **Case 분리 누락**: $\partial e^{z_i}/\partial z_j$ 가 항상 $e^{z_i}$ 가 아님. $i=j$ 일 때만 살아남음.
2. **$\delta_{ij}$ 사용 안 하면 식이 두 줄로 갈라져서 표현이 길어짐** — Kronecker delta 적극 사용.
3. **분모 $S$ 미분 시 $j$ 항만 살아남는 점 누락**.
4. **Jacobian matrix 형태 생략**: $\mathrm{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$ 한 줄 적으면 점수 ↑.
5. **$\hat p - y$ 의 도출에서 $\sum_i y_i = 1$ (one-hot) 사용 누락**.

---

## 10. 연결 개념

- ← [05 Cross-Entropy](05_CrossEntropy_Categorical_유도.md): softmax 출력을 받음
- → [09 Backpropagation](09_Backprop_ChainRule.md): softmax+CE 의 layer-wise gradient
- → [16 Jacobian](16_Jacobian.md): vector→vector 미분 일반론

---
title: "05. Cross-Entropy 유도 (Categorical Likelihood)"
slug: crossentropy
order: 6
---

# 05. Cross-Entropy 유도 (Categorical Likelihood)

> **출제 근거**: 6주차 ★9 \"Cross-Entropy as Categorical NLL\", \"Bernoulli↔Classification\" 통합
> **시험 출제 방식**: \"Derive cross-entropy loss from a categorical likelihood. Compare with MSE-from-Gaussian.\"

---

## 1. 왜 시험에 나오는가

- **분류(Classification) 손실의 유일한 정답 근거**: \"likelihood가 Categorical이라서\".
- 04 토픽 (NLL→MSE) 과 **완벽한 평행 구조** — 회귀↔분류 매핑의 핵심.
- 6주차 \"Bernoulli↔Classification, Gaussian↔Regression\" 통합 시각의 증명.

---

## 2. 사전 수학

### 2.1 [고1] One-hot 벡터

$K$ 개 클래스 중 정답이 $c$ 번째라면:

$$
\mathbf{y} = (y_1, \ldots, y_K), \quad y_j = \begin{cases} 1 & j = c \\ 0 & j \neq c \end{cases}
$$

예: $K=3$, 정답 = class 2 → $\mathbf{y} = (0,1,0)$.

### 2.2 [대1] Categorical 분포

$K$ 클래스, 각 클래스 확률 $p_1, \ldots, p_K$, $\sum_j p_j = 1$.

PMF (one-hot 트릭으로 한 식):

$$
P(\mathbf{y} \mid \mathbf{p}) \;=\; \prod_{j=1}^K p_j^{\, y_j}
\tag{*}
$$

**왜 이렇게?** $\mathbf{y}$ 가 one-hot 이라 $y_j \in \{0,1\}$, 정답 클래스에서만 $y_c=1$:
- $\prod_j p_j^{y_j} = p_c^1 \cdot \prod_{j\neq c} p_j^0 = p_c \cdot 1 = p_c$ ✅

이 공식이 \"Bernoulli 의 다클래스 확장\":
- Bernoulli: $\theta^x(1-\theta)^{1-x}$
- Categorical: $\prod_j p_j^{y_j}$

### 2.3 모델 출력 = 확률 (Softmax)

NN의 logit $\mathbf{z} \in \mathbb{R}^K$ 를 확률로 만드는 함수:

$$
\hat{p}_j \;=\; \frac{\exp(z_j)}{\sum_{m=1}^K \exp(z_m)} \;=:\; \mathrm{softmax}(\mathbf{z})_j
$$

(softmax 미분은 [06 Softmax Jacobian](06_Softmax_Jacobian.md))

---

## 3. 문제 설정

데이터: $D = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^n$, $\mathbf{y}_i$ 는 one-hot.

🟢 **Likelihood 가정 (Categorical)**:

$$
P(\mathbf{y}_i \mid \mathbf{x}_i, h) \;=\; \prod_{j=1}^K \hat{p}_{ij}^{\, y_{ij}}, \quad \hat{\mathbf{p}}_i = h(\mathbf{x}_i)
$$

여기서 $h(\mathbf{x}_i) \in \Delta^{K-1}$ (확률 단순체) — 모델이 출력하는 클래스 확률.

가정: IID.

---

## 4. 유도 체인

### Step 1 — IID Likelihood

$$
P(D \mid h) = \prod_{i=1}^n \prod_{j=1}^K \hat{p}_{ij}^{\, y_{ij}}
\tag{1}
$$

### Step 2 — NLL

$$
\text{NLL}(h) = -\log P(D \mid h)
$$

곱→합:

$$
\text{NLL}(h) \;=\; -\sum_{i=1}^n \sum_{j=1}^K y_{ij} \log \hat{p}_{ij}
\tag{2}
$$

**Line-by-Line (식 (1)→(2)):**

| 변형 | 사용 공식 | 왜 |
|------|----------|-----|
| $-\log \prod_i \prod_j \hat{p}_{ij}^{y_{ij}}$ | $\log\prod = \sum \log$ | 곱→합 (이중) |
| $-\sum_i \sum_j \log \hat{p}_{ij}^{y_{ij}}$ | 같음 | |
| $-\sum_i \sum_j y_{ij}\log \hat p_{ij}$ | $\log a^b = b\log a$ | 지수→계수 |

### Step 3 — One-hot 단순화 (per-sample 형태)

$\mathbf{y}_i$ 가 one-hot이라 정답 인덱스 $c_i$ 에서만 $y_{ic_i}=1$, 나머지 0:

$$
\sum_{j=1}^K y_{ij} \log \hat{p}_{ij} \;=\; 1 \cdot \log \hat{p}_{i c_i} + 0 + \cdots + 0 \;=\; \log \hat{p}_{i c_i}
$$

따라서:

$$
\text{NLL}(h) \;=\; -\sum_{i=1}^n \log \hat{p}_{i c_i}
\tag{3}
$$

> 💡 \"정답 클래스의 확률 $\hat p_{c_i}$ 의 log를 합한 것의 음수\".

### Step 4 — Cross-Entropy 정의로 표현

데이터 한 점에서의 \"true distribution\" $\mathbf{y}_i$ (one-hot)와 \"predicted distribution\" $\hat{\mathbf{p}}_i$ 사이의 **cross-entropy**:

$$
H(\mathbf{y}_i, \hat{\mathbf{p}}_i) \;:=\; -\sum_{j=1}^K y_{ij} \log \hat{p}_{ij}
$$

따라서:

$$
\boxed{\;
\text{NLL}(h) \;=\; \sum_{i=1}^n H(\mathbf{y}_i, \hat{\mathbf{p}}_i) \;=\; -\sum_{i=1}^n \sum_{j=1}^K y_{ij}\log \hat{p}_{ij}
\;}
\tag{4}
$$

이것이 **Categorical Cross-Entropy Loss**.

### Step 5 — ERM 표현 (1/n 평균)

$$
\arg\min_h \text{NLL}(h) \;=\; \arg\min_h \; \frac{1}{n}\sum_{i=1}^n H(\mathbf{y}_i, \hat{\mathbf{p}}_i)
$$

이것이 \"NLL = ERM (with CE loss)\" 통합 시각.

---

## 5. Binary Cross-Entropy (Bernoulli 케이스, 비교)

$K=2$, $\mathbf{y} = (1-y, y)$, $\hat{\mathbf{p}} = (1-\hat p, \hat p)$ 로 두면:

$$
H = -[(1-y)\log(1-\hat p) + y \log \hat p]
$$

이것이 **BCE**. Bernoulli NLL 그 자체 — [02 토픽](02_MLE_Bernoulli_완전유도.md) 의 식 (3) 과 동일 형태.

> ✅ 즉: **Bernoulli 분류 손실 = BCE = Binary CE = Bernoulli NLL** — 모두 같은 것의 다른 이름.

---

## 6. 평행 구조 (회귀 ↔ 분류)

| 회귀 | 분류 |
|------|------|
| Likelihood: Gaussian | Likelihood: Categorical |
| PDF: $\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)$ | PMF: $\prod_j p_j^{y_j}$ |
| 출력: 실수 $\mu = h(x)$ | 출력: 확률벡터 $\hat{\mathbf{p}} = \mathrm{softmax}(h(x))$ |
| NLL → **MSE** $\sum(y-\mu)^2$ | NLL → **CE** $-\sum_j y_j \log \hat p_j$ |
| Loss: square | Loss: $-\log$ on the true class |

> 시험에서 \"Compare regression vs classification\" 나오면 이 표를 그대로 쓸 수 있어야 함.

---

## 7. Softmax + CE 의 결합 (Backprop 접속점)

NN 출력 $\mathbf{z}$ 를 softmax 통과 → CE loss:

$$
\hat{\mathbf{p}} = \mathrm{softmax}(\mathbf{z}), \quad L = -\sum_j y_j \log \hat{p}_j
$$

이때 **gradient 가 매우 깔끔하게**:

$$
\boxed{\; \frac{\partial L}{\partial z_k} \;=\; \hat{p}_k - y_k \;}
$$

이 결과는 [09 Backpropagation](09_Backprop_ChainRule.md) 에서 증명. 이 한 줄이 \"왜 분류에서 softmax+CE를 짝으로 쓰는가\"의 답.

---

## 8. 모범 답안 템플릿

```
[Setup]
Data D = {(x_i, y_i)}, y_i ∈ {0,1}^K one-hot. IID.
Assume y_i | x_i, h ~ Categorical(p̂_i), where p̂_i = h(x_i).
Goal: derive the cross-entropy loss as the NLL of this model.

[Step 1 — IID likelihood]
P(D | h) = Π_i Π_j p̂_{ij}^{y_{ij}}              (Categorical PMF using one-hot trick)

[Step 2 — NLL]
NLL(h) = -log P(D | h)
       = -Σ_i Σ_j y_{ij} log p̂_{ij}              (log of product of powers)

[Step 3 — One-hot simplification]
Since y_i is one-hot at the true class c_i,
  Σ_j y_{ij} log p̂_{ij} = log p̂_{i, c_i}.
So
  NLL(h) = -Σ_i log p̂_{i, c_i}.

[Step 4 — Cross-entropy form]
Defining H(y_i, p̂_i) = -Σ_j y_{ij} log p̂_{ij}, we get
  NLL(h) = Σ_i H(y_i, p̂_i).
Equivalently, in ERM form:
  argmin_h NLL(h) = argmin_h (1/n) Σ_i H(y_i, p̂_i).        ∎

[Interpretation — parallel to regression]
Just as Gaussian likelihood → MSE,
Categorical likelihood → cross-entropy.
The likelihood assumption fully determines the loss function.
This is a concrete instance of "NLL = ERM with a likelihood-
specific loss."
```

---

## 9. 자주 틀리는 함정

1. **One-hot 이라는 사실을 사용 안 함** → 식이 길어지고 \"$\hat p_{ic_i}$\" 형태로 정리 안 됨.
2. **Cross-entropy 와 KL divergence 혼동**: $H(p,q) = H(p) + \mathrm{KL}(p\|q)$. NLL = $-\log P$ 에서 직접 나오는 것은 cross-entropy. 답안에서 둘을 구분하면 점수 ↑.
3. **Softmax 내장 누락**: $\hat{\mathbf{p}}$ 가 어디서 왔는지 (logit → softmax) 한 줄 명시.
4. **부호 실수**: NLL 은 \"음수의 합 → 양수 loss\" 로 만든다. minimize 형태.

---

## 10. 연결 개념

- ↔ [04 NLL→MSE](04_NLL_MSE_Gaussian_유도.md): 회귀-분류 평행 유도
- → [06 Softmax Jacobian](06_Softmax_Jacobian.md): softmax 미분
- → [09 Backprop](09_Backprop_ChainRule.md): softmax+CE → $\hat p - y$
- → [15 Inductive Bias](15_Inductive_Bias_강도.md): likelihood 선택도 inductive bias의 일종

---
title: "09. Backpropagation + Chain Rule — 완전 유도"
slug: backprop
order: 10
---

# 09. Backpropagation + Chain Rule — 완전 유도

> **출제 근거**: 8주차 ★10 (Backprop, Chain Rule), 퀴즈 26 \"3가지 미분 계산 (softmax + CE 등)\"
> **시험 출제 방식**: \"Compute gradients of a 2-layer network with softmax+CE output via chain rule. Show step by step.\"

---

## 1. 왜 시험에 나오는가

- 8주차 핵심. NN 학습의 알고리즘적 본질.
- Chain Rule + 06,05 토픽 결합 → \"$\hat p - y$\" 결과 도출이 답안의 백미.
- 퀴즈 26 변형 매우 유력.

---

## 2. 사전 수학

### 2.1 [고2] Chain Rule (1변수)

$$
\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
$$

### 2.2 [대1] 다변수 Chain Rule

$L = L(z_1, \ldots, z_K)$, $z_k = z_k(\theta)$:

$$
\frac{\partial L}{\partial \theta} \;=\; \sum_{k=1}^K \frac{\partial L}{\partial z_k} \cdot \frac{\partial z_k}{\partial \theta}
$$

**왜 합?** $\theta$ 가 변하면 모든 $z_k$ 가 변하고, $L$ 은 모든 $z_k$ 변화의 영향을 받음. → 각 경로의 기여를 합.

### 2.3 [대학원] Computational Graph

NN 을 노드(연산)와 edge(데이터 흐름)의 그래프로 표현:

```
x → [W₁, b₁ Linear] → h₁ → [σ ReLU] → a₁ → [W₂, b₂ Linear] → z → [softmax] → ŷ → [CE w/ y] → L
```

각 노드는 \"local 미분\" 만 알면, chain rule 로 전체 미분이 자동 계산됨. 이게 backprop.

---

## 3. 모델 설정 (표준 2-layer NN)

$$
\begin{aligned}
\mathbf{h} &= W_1 \mathbf{x} + \mathbf{b}_1 & &\text{(Linear 1)} \\
\mathbf{a} &= \sigma(\mathbf{h}) & &\text{(Activation, ex. ReLU)} \\
\mathbf{z} &= W_2 \mathbf{a} + \mathbf{b}_2 & &\text{(Linear 2 — logits)} \\
\hat{\mathbf{p}} &= \mathrm{softmax}(\mathbf{z}) & &\text{(Output prob.)} \\
L &= -\sum_{j} y_j \log \hat p_j & &\text{(Cross-Entropy, } \mathbf{y}\text{ one-hot)}
\end{aligned}
$$

---

## 4. Forward Pass (값 계산)

위 5줄 그대로 — 입력 $\mathbf{x}$, 정답 $\mathbf{y}$ 로 손실 $L$ 까지 계산.

> 💡 Backprop 전에 forward 값들을 **저장**해야 함 (메모리 비용).

---

## 5. Backward Pass (Chain Rule)

목표: $\partial L / \partial W_1, \partial L / \partial \mathbf{b}_1, \partial L / \partial W_2, \partial L / \partial \mathbf{b}_2$.

### Step 1 — Output gradient $\partial L / \partial \mathbf{z}$ (★ 핵심 결과)

[06 Softmax Jacobian](06_Softmax_Jacobian.md) 7절 결과 직접 사용:

$$
\boxed{\; \frac{\partial L}{\partial z_k} \;=\; \hat p_k - y_k \;}
\tag{1}
$$

벡터 형태:

$$
\boldsymbol{\delta}_z := \frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{p}} - \mathbf{y}
$$

> 🎯 이 한 줄이 \"왜 softmax + CE 가 짝인가\"의 답. 만약 sigmoid + CE 였으면 비슷한 깔끔한 식. MSE + softmax 였으면 안 깔끔.

### Step 2 — $W_2, \mathbf{b}_2$ gradient

$\mathbf{z} = W_2 \mathbf{a} + \mathbf{b}_2$ 이므로:

$$
\frac{\partial z_k}{\partial W_{2,kj}} = a_j, \quad \frac{\partial z_k}{\partial b_{2,k}} = 1
$$

Chain rule:

$$
\frac{\partial L}{\partial W_{2,kj}} = \frac{\partial L}{\partial z_k}\cdot a_j = (\hat p_k - y_k) \cdot a_j
$$

$$
\frac{\partial L}{\partial b_{2,k}} = \frac{\partial L}{\partial z_k} = \hat p_k - y_k
$$

행렬 표기:

$$
\boxed{\; \frac{\partial L}{\partial W_2} \;=\; \boldsymbol{\delta}_z\, \mathbf{a}^\top \;=\; (\hat{\mathbf{p}}-\mathbf{y})\mathbf{a}^\top \;}
$$

$$
\boxed{\; \frac{\partial L}{\partial \mathbf{b}_2} \;=\; \boldsymbol{\delta}_z \;=\; \hat{\mathbf{p}}-\mathbf{y} \;}
$$

> 💡 패턴: \"weight gradient = (다음 층 delta) × (현재 층 입력)ᵀ\". 이게 backprop 의 일반 규칙.

### Step 3 — $\mathbf{a}$ 까지 delta 전달

$$
\frac{\partial L}{\partial a_j} = \sum_k \frac{\partial L}{\partial z_k}\cdot \frac{\partial z_k}{\partial a_j} = \sum_k (\hat p_k - y_k)\cdot W_{2,kj}
$$

벡터 형태:

$$
\boldsymbol{\delta}_a := \frac{\partial L}{\partial \mathbf{a}} = W_2^\top \boldsymbol{\delta}_z = W_2^\top (\hat{\mathbf{p}} - \mathbf{y})
$$

> 💡 Forward 에서 $W_2$ 는 \"$\mathbf{a}\to\mathbf{z}$\" 였는데, backward 에서 $W_2^\top$ 는 \"delta_z → delta_a\" 로 작용. 이게 \"backward = transpose\" 의 의미.

### Step 4 — Activation 통과

$\mathbf{a} = \sigma(\mathbf{h})$, element-wise:

$$
\frac{\partial L}{\partial h_j} = \frac{\partial L}{\partial a_j}\cdot \sigma'(h_j)
$$

벡터 (Hadamard product $\odot$):

$$
\boldsymbol{\delta}_h = \boldsymbol{\delta}_a \odot \sigma'(\mathbf{h})
$$

**ReLU 의 경우**:

$$
\sigma(h) = \max(0, h), \qquad \sigma'(h) = \begin{cases} 1 & h > 0 \\ 0 & h \leq 0 \end{cases}
$$

따라서 ReLU backward 는 \"$h_j > 0$ 인 위치만 통과시키는 mask\".

### Step 5 — $W_1, \mathbf{b}_1$ gradient

$\mathbf{h} = W_1\mathbf{x} + \mathbf{b}_1$ 이므로 Step 2 와 동일 패턴:

$$
\boxed{\; \frac{\partial L}{\partial W_1} \;=\; \boldsymbol{\delta}_h\, \mathbf{x}^\top \;}, \qquad \boxed{\; \frac{\partial L}{\partial \mathbf{b}_1} \;=\; \boldsymbol{\delta}_h \;}
$$

---

## 6. 전체 정리 (한눈에)

| 단계 | 식 |
|------|-----|
| Forward $\to L$ | $\mathbf{h} \to \mathbf{a} \to \mathbf{z} \to \hat{\mathbf{p}} \to L$ |
| $\boldsymbol{\delta}_z$ | $\hat{\mathbf{p}} - \mathbf{y}$ |
| $\partial L/\partial W_2$ | $\boldsymbol{\delta}_z \mathbf{a}^\top$ |
| $\partial L/\partial \mathbf{b}_2$ | $\boldsymbol{\delta}_z$ |
| $\boldsymbol{\delta}_a$ | $W_2^\top \boldsymbol{\delta}_z$ |
| $\boldsymbol{\delta}_h$ | $\boldsymbol{\delta}_a \odot \sigma'(\mathbf{h})$ |
| $\partial L/\partial W_1$ | $\boldsymbol{\delta}_h \mathbf{x}^\top$ |
| $\partial L/\partial \mathbf{b}_1$ | $\boldsymbol{\delta}_h$ |

> 📐 **핵심 통찰**: \"Forward 에서 입력 × W → 출력. Backward 에서 출력 delta × Wᵀ → 입력 delta. Weight gradient = 출력 delta × 입력ᵀ.\"

---

## 7. 왜 Backprop 인가 (효율성)

### 7.1 Naive 미분의 비용

각 파라미터 $\theta_k$ 마다 $L$ 을 다시 forward 해서 finite difference: $O(\#\theta \cdot \text{forward cost})$. → 수백만 파라미터에서 불가능.

### 7.2 Backprop 의 비용

- Forward 1회 + Backward 1회 = 약 forward 2-3배.
- 모든 파라미터 gradient 를 \"한 번에\" 얻음.
- Reverse-mode automatic differentiation 의 알고리즘적 이름.

> 💡 8주차 핵심: \"Chain rule 을 graph 위에서 reverse 로 적용\".

---

## 8. 모범 답안 템플릿 (퀴즈 26 형식)

```
[Setup]
Two-layer NN with ReLU and softmax+CE:
  h = W₁x + b₁
  a = ReLU(h)
  z = W₂a + b₂
  p̂ = softmax(z)
  L = -Σ_j y_j log p̂_j        (y one-hot)

[Step 1 — ∂L/∂z]
By the softmax+CE identity (use one-hot Σ y_j = 1):
  ∂L/∂z_k = p̂_k - y_k.
Vector form:  δ_z = p̂ - y.

[Step 2 — Output layer parameters]
z_k = Σ_j W₂_{kj} a_j + b₂_k, so
  ∂L/∂W₂_{kj} = (p̂_k - y_k) a_j   ⇒   ∂L/∂W₂ = δ_z aᵀ
  ∂L/∂b₂      = δ_z

[Step 3 — Backprop into hidden activations]
  ∂L/∂a_j = Σ_k δ_z_k · W₂_{kj}    ⇒   δ_a = W₂ᵀ δ_z

[Step 4 — Through ReLU]
σ'(h_j) = 1{h_j > 0}.  So
  δ_h = δ_a ⊙ σ'(h).

[Step 5 — Hidden layer parameters]
h = W₁x + b₁, so
  ∂L/∂W₁ = δ_h xᵀ,    ∂L/∂b₁ = δ_h.

[Comment]
This is one full backward pass. Cost ≈ a small constant times the
forward cost, regardless of how many parameters there are. That is
why backpropagation is the only practical way to train large
networks: it computes ∇_θ L for *all* parameters at once, in O(forward).
```

---

## 9. 자주 틀리는 함정

1. **One-hot $\sum_i y_i = 1$ 사용 누락**: $\hat p - y$ 도출에서 핵심 단계.
2. **Transpose 실수**: 출력→입력 backward 는 항상 $W^\top$. 차원 맞추기로 자가검증.
3. **Activation 미분 위치 헷갈림**: $\sigma'(\mathbf{h})$ 는 $\mathbf{h}$ 에 적용 (forward 의 pre-activation 값), $\mathbf{a}$ 에 적용하면 안 됨.
4. **\"Forward 값을 저장\" 명시 누락**: backprop은 forward 의 $\mathbf{h}, \mathbf{a}$ 값을 사용. 답안에 한 줄.
5. **Hadamard $\odot$ vs 행렬곱 혼동**: ReLU 미분은 element-wise (Hadamard).

---

## 10. 연결 개념

- ← [06 Softmax Jacobian](06_Softmax_Jacobian.md): $\hat p - y$ 의 출처
- ← [05 Cross-Entropy](05_CrossEntropy_Categorical_유도.md): 손실 정의
- → [12 GD/SGD/Adam](12_GD_SGD_Adam.md): gradient 를 받아 update
- ← [08 Newton](08_Newton_2차근사.md): \"왜 backprop 의 1차 정보로만 가는가\"의 비교 대상

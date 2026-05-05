---
title: "02. Backpropagation — 체인 룰의 신경망 적용"
slug: 11-extra-topics-02-backpropagation
order: 70
---

# 02. Backpropagation — 체인 룰의 신경망 적용

> "Backprop은 마법이 아니라 체인 룰이다."

---

## 1. 신경망 구조 (L층)

층 $l = 1, \ldots, L$:
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$
$$a^{(l)} = \sigma(z^{(l)})$$

- $a^{(0)} = x$ (입력)
- $a^{(L)} = \hat{y}$ (출력)
- 손실: $L(a^{(L)}, y)$

---

## 2. 목표

$$\frac{\partial L}{\partial W^{(l)}}, \quad \frac{\partial L}{\partial b^{(l)}}$$

각 층의 가중치에 대한 그래디언트 계산.

---

## 3. 체인 룰 (한 변수 복습)

$L = f(g(h(x)))$:
$$\frac{dL}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

신경망에 그대로 적용. 단, 변수가 **벡터/행렬**.

---

## 4. Backprop 4단계 식 (외워라)

### Step 1: 출력층 오차

$$\delta^{(L)} = \nabla_{a^{(L)}} L \odot \sigma'(z^{(L)})$$

(Hadamard 곱 ⊙: 원소별 곱)

**예시 (CE + softmax):** $\delta^{(L)} = a^{(L)} - y$ (놀라운 단순성!)

### Step 2: 오차 역전파

$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

뒤층에서 앞층으로 전달.

### Step 3: 가중치 그래디언트

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T$$

(외적 — c×1 × 1×d = c×d 행렬)

### Step 4: 편향 그래디언트

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

---

## 5. 유도 — Step 2 핵심

$L$은 $z^{(l+1)}$을 통해서만 $z^{(l)}$에 의존. 체인 룰:
$$\frac{\partial L}{\partial z^{(l)}_j} = \sum_k \frac{\partial L}{\partial z^{(l+1)}_k} \cdot \frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j}$$

$z^{(l+1)}_k = \sum_i W^{(l+1)}_{ki} a^{(l)}_i + b^{(l+1)}_k = \sum_i W^{(l+1)}_{ki} \sigma(z^{(l)}_i) + b^{(l+1)}_k$.

$$\frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j} = W^{(l+1)}_{kj} \cdot \sigma'(z^{(l)}_j)$$

따라서:
$$\delta^{(l)}_j = \sigma'(z^{(l)}_j) \sum_k W^{(l+1)}_{kj} \delta^{(l+1)}_k = \sigma'(z^{(l)}_j) \cdot [(W^{(l+1)})^T \delta^{(l+1)}]_j$$

→ 식 (2). ∎

---

## 6. 계산 그래프 (Computation Graph)

```
   x → [Layer 1] → a¹ → [Layer 2] → a² → ··· → aᴸ → L
        W¹                W²                 Wᴸ
```

**Forward pass:** 좌→우, 모든 $a^{(l)}, z^{(l)}$ 저장.
**Backward pass:** 우→좌, $\delta^{(l)}$ 계산하면서 그래디언트 누적.

---

## 7. 직관 — "오차의 책임 분배"

각 가중치의 오차 기여도를 **체인 룰**로 분해해 거꾸로 전달.

큰 가중치 $W^{(l+1)}_{kj}$ → 큰 영향 → 큰 그래디언트 받음.

---

## 8. 시험 답안 — Backprop 1줄 식 유도

### [문제] L층 신경망에서 $\partial L/\partial W^{(l)}$를 체인 룰로 유도하라.

### [풀이]

$L$이 $W^{(l)}$에 의존하는 경로: $W^{(l)} \to z^{(l)} \to a^{(l)} \to z^{(l+1)} \to \cdots \to L$.

체인 룰 (행렬 형태):
$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}.$$

$\delta^{(l)} := \partial L/\partial z^{(l)}$로 두고, $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$이므로 $\partial z^{(l)}/\partial W^{(l)}$는 $a^{(l-1)}$:

$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T.$$

$\delta^{(l)}$ 자체는 다음 층에서 재귀:
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)}).$$

출력층 base case:
$$\delta^{(L)} = \nabla_{a^{(L)}} L \odot \sigma'(z^{(L)}). \quad \blacksquare$$

---

## 9. 자주 출제되는 응용

### Q1. Vanishing Gradient
"왜 깊은 망에서 그래디언트가 사라지나?"

답: $\delta^{(l)}$이 뒤층에서 곱해질수록 $\sigma'$ < 1이 누적되어 지수적으로 0.

### Q2. ReLU 사용 이유
$\text{ReLU}'(x) = 1$ for $x > 0$ → 곱셈에서 1이 누적 → vanishing 완화.

### Q3. 가중치 행렬 곱의 의미
$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'$ — 뒤 가중치가 클수록 오차도 크게 받음.

---

## 10. 한 줄 요약

> "Backpropagation = 체인 룰을 행렬 형태로 효율적으로 적용. 4개 식만 외우면 끝."

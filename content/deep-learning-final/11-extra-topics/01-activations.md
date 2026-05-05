---
title: "01. 활성화 함수 (Activation Functions)"
slug: activations
order: 1
---

# 01. 활성화 함수 (Activation Functions)

> 신경망의 비선형성을 만드는 함수들. 각각의 **미분 공식**이 시험 핵심.

---

## 1. 왜 비선형 활성화인가?

선형 활성화만 사용하면 다층 신경망 = 1층 신경망 (선형 결합의 결합 = 선형):
$$f(x) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$$

→ 표현력이 1층과 동일. **비선형**으로 표현력 폭발.

---

## 2. 핵심 활성화 함수 6개

### ① Sigmoid (Logistic)

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**그래프:** S자형, 0~1로 압축.

**미분 (외워라!):**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**유도:**
$\sigma(x) = (1 + e^{-x})^{-1}$. 체인 룰:
$$\sigma'(x) = -(1+e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$$

분자 $e^{-x} = (1+e^{-x}) - 1$:
$$= \frac{1}{1+e^{-x}} - \frac{1}{(1+e^{-x})^2} = \sigma(x) - \sigma(x)^2 = \sigma(x)(1 - \sigma(x))$$

**단점:** 큰 $|x|$에서 $\sigma' \to 0$ → **vanishing gradient**.

---

### ② Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**그래프:** S자형, -1~1.

**미분:**
$$\tanh'(x) = 1 - \tanh^2(x)$$

**유도:**
$\tanh = (e^x - e^{-x})/(e^x + e^{-x})$. 몫 규칙:
$$\tanh'(x) = \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2}$$

분자: $4 e^x \cdot e^{-x} = 4$ (차의 제곱 공식). 분자/분모 정리:
$$= \frac{4}{(e^x + e^{-x})^2} = 1 - \tanh^2(x)$$

**관계:** $\tanh(x) = 2\sigma(2x) - 1$ (sigmoid의 변형)

**단점:** Sigmoid와 같이 vanishing gradient.

---

### ③ ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**미분:**
$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined} & x = 0 \end{cases}$$

**$x = 0$ 처리:** 컴퓨터에서는 보통 0 또는 1로 정의 (subgradient).

**장점:**
- 미분 단순 (0 또는 1)
- 양수 영역 vanishing gradient 없음
- 계산 빠름

**단점:** **Dying ReLU** — 음수 영역에서 영원히 0 (그래디언트 0 → 학습 안 됨)

---

### ④ Leaky ReLU

$$\text{LReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}, \quad \alpha = 0.01$$

**미분:**
$$\text{LReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x < 0 \end{cases}$$

**해결:** Dying ReLU 완화 (음수도 작은 그래디언트).

---

### ⑤ GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

여기서 $\Phi$는 표준정규분포의 CDF.

**근사:**
$$\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715 x^3)\right)\right)$$

**미분:** 체인 룰 + 정규분포 pdf
$$\text{GELU}'(x) = \Phi(x) + x \phi(x)$$

(여기서 $\phi$는 N(0,1) pdf)

**사용처:** Transformer (BERT, GPT) 표준.

---

### ⑥ Softplus

$$\text{Softplus}(x) = \log(1 + e^x)$$

**미분:**
$$\text{Softplus}'(x) = \frac{e^x}{1 + e^x} = \sigma(x)$$

→ "소프트플러스의 미분이 sigmoid". 매끄러운 ReLU 근사.

---

## 3. 미분 공식 요약 (외워라!)

| 함수 | 식 | 미분 |
|-----|-----|-----|
| Sigmoid σ | $1/(1+e^{-x})$ | $\sigma(1-\sigma)$ |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $1 - \tanh^2$ |
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ |
| Leaky ReLU | (위) | 1 또는 α |
| GELU | $x\Phi(x)$ | $\Phi(x) + x\phi(x)$ |
| Softplus | $\log(1+e^x)$ | $\sigma(x)$ |

---

## 4. Vanishing Gradient 분석

신경망의 그래디언트는 활성화 도함수의 **곱**:
$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(L)}} \prod_{l=1}^{L-1} W^{(l+1)} \cdot \sigma'(z^{(l)})$$

각 $\sigma' < 1$이면 곱이 **지수적으로 작아짐** → 깊은 층의 그래디언트가 0.

**해결:**
- ReLU 사용 (양수 영역 미분 = 1)
- ResNet (skip connection)
- BatchNorm
- 적절한 초기화 (Xavier, He)

---

## 5. 시험 답안 — Sigmoid 미분 유도

### [문제] $\sigma(x) = 1/(1+e^{-x})$의 미분이 $\sigma(x)(1-\sigma(x))$임을 증명하라.

### [풀이]

$\sigma(x) = (1+e^{-x})^{-1}$. 체인 룰:
$$\sigma'(x) = -(1+e^{-x})^{-2}(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}.$$

분자 $e^{-x} = (1+e^{-x}) - 1$로 분해:
$$= \frac{1}{1+e^{-x}} - \frac{1}{(1+e^{-x})^2} = \sigma(x) - \sigma(x)^2 = \sigma(x)(1-\sigma(x)). \quad \blacksquare$$

---

## 6. 시험에 자주 나오는 응용

### Sigmoid + BCE 그래디언트

$p = \sigma(z)$, $L = -[y\log p + (1-y)\log(1-p)]$:
$$\frac{\partial L}{\partial z} = p - y$$

(Softmax + CE와 동일한 단순한 형태!)

### 유도

$\partial L/\partial p = -y/p + (1-y)/(1-p)$
$\partial p/\partial z = p(1-p)$
체인:
$$\frac{\partial L}{\partial z} = \left(-\frac{y}{p} + \frac{1-y}{1-p}\right) p(1-p) = -y(1-p) + (1-y)p = p - y. \quad \blacksquare$$

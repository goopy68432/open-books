---
title: "08. 초기화 (Initialization)"
slug: 11-extra-topics-08-initialization
order: 76
---

# 08. 초기화 (Initialization)

> Xavier, He 초기화의 분산 분석 유도.

---

## 1. 왜 초기화가 중요한가?

### 문제 1: 0 초기화
$W = 0$이면 모든 뉴런이 같은 출력 → 같은 그래디언트 → 학습 안 됨.
**대칭성 깨야 함.**

### 문제 2: 너무 큰 / 작은 초기화
- 너무 큼: 활성화 포화 → vanishing gradient (sigmoid)
- 너무 작음: 신호 점차 0으로 → vanishing

→ "출력 분산이 입력 분산과 같아지도록" 초기화 필요.

---

## 2. Forward 분산 분석

층 $l$에서 $z = \sum_i w_i x_i$. $w_i, x_i$ 모두 평균 0, $w$와 $x$ 독립이라고 가정.

$$\text{Var}(z) = \sum_i \text{Var}(w_i x_i) = n_{\text{in}} \text{Var}(w) \text{Var}(x)$$

(n개 항, 각 $\text{Var}(w_i x_i) = \text{Var}(w)\text{Var}(x)$ 평균 0이라 가정)

**조건:** $\text{Var}(z) = \text{Var}(x)$ 를 원함:
$$\text{Var}(w) = \frac{1}{n_{\text{in}}}$$

---

## 3. Xavier (Glorot) 초기화

Forward + Backward 둘 다 보존하려고 평균:
$$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

### 분포 선택

**정규:** $w \sim N(0, 2/(n_{\text{in}} + n_{\text{out}}))$
**균일:** $w \sim U[-r, r]$, $r = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$ (분산 = $r^2/3$로부터)

### 사용처
- Sigmoid, Tanh 활성화 (선형 근사 영역에서 유효)

---

## 4. He 초기화 (Kaiming)

ReLU는 **음수 절반 죽임** → 분산이 절반으로:
$$\text{Var}(\text{ReLU}(z)) = \text{Var}(z)/2$$

따라서 입력 분산을 보존하려면 가중치 분산을 **2배**:
$$\text{Var}(w) = \frac{2}{n_{\text{in}}}$$

### 정규 분포
$w \sim N(0, 2/n_{\text{in}})$

### 사용처
- ReLU, LeakyReLU 활성화 (현대 NN 표준)

---

## 5. 유도 — He 초기화

ReLU를 통과한 신호의 분산:
$$E[\text{ReLU}(z)^2] = E[z^2 \mathbb{1}_{z>0}] = \frac{1}{2} E[z^2] = \frac{1}{2} \text{Var}(z)$$

(z가 평균 0 대칭이라 가정)

층 $l+1$:
$$\text{Var}(z^{(l+1)}) = n_{\text{in}} \text{Var}(w) \cdot \frac{1}{2} \text{Var}(z^{(l)})$$

분산 보존 조건:
$$1 = \frac{n_{\text{in}}}{2} \text{Var}(w) \Rightarrow \text{Var}(w) = \frac{2}{n_{\text{in}}}. \quad \blacksquare$$

---

## 6. 비교표

| 초기화 | $\text{Var}(w)$ | 활성화 |
|------|---------------|------|
| Xavier (forward) | $1/n_{\text{in}}$ | sigmoid/tanh |
| Xavier (avg) | $2/(n_{\text{in}} + n_{\text{out}})$ | sigmoid/tanh |
| He | $2/n_{\text{in}}$ | ReLU |
| LSUV | 데이터 기반 자동 조정 | 임의 |

---

## 7. 시험 답안 — He 초기화 유도

### [문제] ReLU 활성화에서 He 초기화 $\text{Var}(w) = 2/n_{\text{in}}$가 분산을 보존함을 보여라.

### [풀이]

층 $l$의 사전활성화: $z^{(l+1)} = W^{(l+1)} a^{(l)}$, $a^{(l)} = \text{ReLU}(z^{(l)})$.

가정: $w$ 평균 0, 독립, $z^{(l)}$ 평균 0 대칭 분포.

ReLU 출력의 2차 모멘트:
$$E[a^{(l)2}] = E[\text{ReLU}(z^{(l)})^2] = E[z^{(l)2} \mathbb{1}_{z>0}] = \frac{1}{2} E[z^{(l)2}] = \frac{1}{2} \text{Var}(z^{(l)})$$

(대칭성으로 양수 영역의 적분 = 전체의 절반)

다음 층 분산:
$$\text{Var}(z^{(l+1)}) = n_{\text{in}} \cdot \text{Var}(w) \cdot E[a^{(l)2}] = \frac{n_{\text{in}} \text{Var}(w)}{2} \text{Var}(z^{(l)})$$

분산 보존 ($\text{Var}(z^{(l+1)}) = \text{Var}(z^{(l)})$):
$$\frac{n_{\text{in}} \text{Var}(w)}{2} = 1 \Rightarrow \text{Var}(w) = \frac{2}{n_{\text{in}}}. \quad \blacksquare$$

---

## 8. 한 줄 요약

> "초기화는 forward/backward 분산 보존이 핵심. Sigmoid/tanh는 Xavier, ReLU는 He. He는 ReLU의 절반 죽임 보정으로 2/n_in."

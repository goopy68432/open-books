---
title: "04. 체인 룰 — 합성함수 미분의 핵심"
slug: 00-prerequisites-04-chain-rule
order: 4
---

# 04. 체인 룰 — 합성함수 미분의 핵심

> 신경망의 역전파(backpropagation)는 본질적으로 체인 룰. NLL 미분에서도 매번 등장.

---

## 1. 왜 배우나

- 기출 4번: $\frac{d}{d\theta} \log(1 - \theta)$ — 체인 룰로 $-\frac{1}{1-\theta}$
- 기출 8번: softmax 미분에서 $\frac{d}{dz_i} e^{z_i}$ — 체인 룰로 $e^{z_i}$
- 신경망 모든 학습은 체인 룰 기반

---

## 2. 정리 (체인 룰)

### 단변수

$y = f(g(x))$일 때,

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

또는 $u = g(x)$로 두면:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**기억법:** "분모분자 약분되는 것처럼 보임" (실제는 그렇지 않지만 외울 때 도움)

### 직관 (자판기 비유)

자판기 g: 동전 → 토큰 (변환율 $g'$)
자판기 f: 토큰 → 음료 (변환율 $f'$)
**합성:** 동전 1개 변화 → 토큰 $g'$만큼 변화 → 음료 $f' \cdot g'$만큼 변화

---

## 3. 적용 예시

### 예 1: $\log(1 - \theta)$ 미분

내부함수 $u = 1 - \theta$, 외부함수 $\log u$.

$$\frac{d}{d\theta} \log(1-\theta) = \frac{1}{u} \cdot \frac{du}{d\theta} = \frac{1}{1-\theta} \cdot (-1) = -\frac{1}{1-\theta}$$

### 예 2: $e^{-x^2/2}$ 미분 (가우스 적분에서 등장)

$u = -x^2/2$, $\frac{du}{dx} = -x$.

$$\frac{d}{dx} e^{-x^2/2} = e^{-x^2/2} \cdot (-x) = -x \cdot e^{-x^2/2}$$

### 예 3: $\log(L(\theta))$ 미분

$u = L(\theta)$, $\log u$.

$$\frac{d}{d\theta} \log L = \frac{1}{L} \cdot \frac{dL}{d\theta} = \frac{L'(\theta)}{L(\theta)}$$

이게 바로 **score function**의 정의! MLE 이론의 핵심 수단.

---

## 4. 체인 룰의 다단계 적용

3중 합성: $y = f(g(h(x)))$

$$\frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

**예:** $y = \exp(-(x-1)^2 / 2)$
- $h(x) = x-1$, $h' = 1$
- $g(u) = -u^2/2$, $g'(u) = -u$
- $f(v) = e^v$, $f'(v) = e^v$
- 결과: $e^{-(x-1)^2/2} \cdot \left(-(x-1)\right) \cdot 1 = -(x-1)e^{-(x-1)^2/2}$

---

## 5. 다변수로의 확장 (편미분 체인 룰)

$z = f(x, y)$, $x = g(t)$, $y = h(t)$일 때:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x} \cdot \frac{dx}{dt} + \frac{\partial f}{\partial y} \cdot \frac{dy}{dt}$$

**기출 8번 softmax 미분에서:**
- $p_i = e^{z_i} / S$, $S = \sum_j e^{z_j}$
- $z_j$ 변화 → S 변화 → $p_i$ 변화 (체인)

---

## 6. 시험에서 자주 쓰는 패턴

### 패턴 1: log + 분포의 함수
$$\frac{d}{d\theta} \log p(y|\theta) = \frac{1}{p(y|\theta)} \cdot \frac{\partial p}{\partial \theta}$$

### 패턴 2: 지수함수 합성
$$\frac{d}{dx} e^{f(x)} = e^{f(x)} \cdot f'(x)$$

### 패턴 3: 거듭제곱
$$\frac{d}{d\theta} \theta^k = k \theta^{k-1}$$

(이건 체인 룰 아니라 멱법칙 자체)

### 패턴 4: $(1-\theta)^k$
$u = 1-\theta$, $u^k$:
$$\frac{d}{d\theta}(1-\theta)^k = k(1-\theta)^{k-1} \cdot (-1) = -k(1-\theta)^{k-1}$$

---

## 7. 시험 답안 작성법

### "왜 체인 룰?"

답안에 명시:
> "$\log(1-\theta)$는 $u = 1-\theta$를 내부함수로 하는 합성함수 $\log \circ u$이다. 체인 룰에 의해 $\frac{d}{d\theta}\log(1-\theta) = \frac{1}{1-\theta} \cdot \frac{du}{d\theta} = -\frac{1}{1-\theta}$."

세 번 등장하는 사이클:
1. **합성 식별** — 어떤 게 내부, 어떤 게 외부?
2. **각각 미분**
3. **곱하기 (체인)**

---

## 8. 자가 점검

다음을 미분하라:

1. $\log(2x + 3)$
2. $e^{3x^2}$
3. $(\sin x)^5$ — 시험 범위 아니지만 연습용
4. $\log(\theta^k(1-\theta)^{n-k})$ — 먼저 로그 법칙으로 펴고
5. $\frac{e^{z_i}}{e^{z_1} + e^{z_2}}$ ($z_i$로 미분, $i = 1$일 때)

**답:**
1. $\frac{2}{2x+3}$
2. $6x \cdot e^{3x^2}$
3. $5(\sin x)^4 \cos x$
4. 먼저 $k\log\theta + (n-k)\log(1-\theta)$로 펴서 미분 → $\frac{k}{\theta} - \frac{n-k}{1-\theta}$
5. 몫 규칙 + 체인 룰 → softmax 미분 결과 (`08-softmax/02-derivation.md`)

---

## 다음 챕터

[`05-integral-101.md`](./05-integral-101.md) — 적분의 정의와 정적분.

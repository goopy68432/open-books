---
title: "02. 8단계 유도 — 시험의 심장"
slug: derivation
order: 2
---

# 02. 8단계 유도 — 시험의 심장

> 각 단계마다 "왜 이 단계를 밟는가"를 한국어로 적어야 만점.

---

## 단계 1: 모델 명시

$$y_i \sim \text{Bern}(\theta), \quad i = 1, 2, \ldots, n, \quad y_i \text{ i.i.d}$$

**왜 i.i.d 명시?**
- **독립**: 한 시행이 다른 시행과 무관 → 결합확률 = 곱
- **동일분포**: 모두 같은 θ → 같은 형태 pmf

이걸 안 적으면 **다음 단계의 곱이 정당화되지 않음**.

---

## 단계 2: 단일 pmf

$$p(y_i | \theta) = \theta^{y_i}(1-\theta)^{1-y_i}, \quad y_i \in \{0, 1\}$$

**왜 이 형태?**
- 두 경우(0/1)를 한 식으로 통합
- $y_i = 1$일 때 $\theta$, $y_i = 0$일 때 $1-\theta$로 자동 분기

---

## 단계 3: 결합확률 — i.i.d → 곱

i.i.d 가정에서 결합 pmf:
$$p(y_1, y_2, \ldots, y_n | \theta) = \prod_{i=1}^n p(y_i | \theta)$$

**왜 곱?** **독립**이라서. 독립의 정의 $P(A \cap B) = P(A)P(B)$를 n개로 확장.

---

## 단계 4: 우도함수 정의

데이터 $y_1, \ldots, y_n$은 **고정** (관측됨). $\theta$만 변수.

$$L(\theta) = \prod_{i=1}^n p(y_i | \theta) = \prod_{i=1}^n \theta^{y_i}(1-\theta)^{1-y_i}$$

**왜 우도?** 확률을 "데이터 입장 → 모수 입장"으로 시점 전환.

---

## 단계 5: 우도 단순화

지수의 **합**:
$$\sum_{i=1}^n y_i = k, \quad \sum_{i=1}^n (1 - y_i) = n - k$$

곱의 형태에서 같은 밑은 지수의 합:
$$\theta^{y_1} \cdot \theta^{y_2} \cdots \theta^{y_n} = \theta^{y_1 + y_2 + \cdots + y_n} = \theta^k$$

$$(1-\theta)^{1-y_1} \cdots (1-\theta)^{1-y_n} = (1-\theta)^{n-k}$$

따라서:
$$\boxed{L(\theta) = \theta^k (1-\theta)^{n-k}}$$

**왜 이 형태가 깔끔한가?** 모든 데이터의 기여가 **k와 n** 두 숫자로 압축됨. (충분통계량)

---

## 단계 6: 로그우도 — **시험 핵심**

$$\ell(\theta) = \log L(\theta) = \log[\theta^k (1-\theta)^{n-k}]$$

로그 법칙 $\log(ab) = \log a + \log b$, $\log(a^c) = c\log a$:
$$\ell(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

### **왜 로그를 취하는가? — 3대 이유 (반드시 답안에 적기!)**

#### 이유 1: 곱 → 합
원래 L은 곱이라 미분 시 곱셈 규칙이 반복됨. 로그를 취하면 합이 되어 **미분이 단순한 합**으로 바뀜.

#### 이유 2: 단조성
$\log$는 단조증가 함수. 따라서:
$$\arg\max_\theta L(\theta) = \arg\max_\theta \log L(\theta)$$

→ **최댓값 위치 보존**. log L 최대화해도 L 최대화한 것과 같은 θ 얻음.

#### 이유 3: 수치 안정성
n이 크면 L은 0에 매우 가까움 (작은 수의 곱) → 컴퓨터에서 underflow.
log를 취하면 큰 음수 → 안전.

**시험 답안 표준 문장:**
> "L은 곱 형태라 미분이 복잡하다. log는 단조증가이므로 $\arg\max$가 보존되며, 곱이 합으로 변환되어 미분이 쉬워지고, underflow도 방지된다. 이런 이유로 log L을 다룬다."

---

## 단계 7: NLL 정의

$$\text{NLL}(\theta) = -\ell(\theta) = -k\log\theta - (n-k)\log(1-\theta)$$

### 왜 음수를 붙이는가?

- 머신러닝의 표준은 **손실 최소화** (loss minimization).
- $\log L$은 **최대화** 대상 → 음수 붙이면 **최소화** 대상.
- 결과적으로:
$$\arg\max_\theta \log L(\theta) = \arg\min_\theta \text{NLL}(\theta)$$

**시험에서:** "NLL = -log L 정의에 의해, MLE는 NLL을 최소화하는 θ다."

---

## 단계 8: 미분=0 — 페르마 정리

$\theta \in (0, 1)$ 내부에서 NLL은 미분 가능. 내부 극값에서 1차 도함수 = 0:

$$\frac{d\,\text{NLL}}{d\theta} = -\frac{k}{\theta} + \frac{n-k}{1-\theta} \cdot (-1) \cdot (-1)$$

잠깐, 부호 정리. 원래:
$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta}$$

(체인 룰: $\frac{d}{d\theta}\log(1-\theta) = -\frac{1}{1-\theta}$, $(n-k)$ 곱)

NLL = -ℓ이므로:
$$\frac{d\,\text{NLL}}{d\theta} = -\frac{k}{\theta} + \frac{n-k}{1-\theta}$$

### 미분=0 풀이

NLL 또는 ℓ 어느 쪽으로 풀어도 같은 결과. ℓ을 0으로:
$$\frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

양변 정리:
$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$

크로스 곱:
$$k(1-\theta) = (n-k)\theta$$
$$k - k\theta = n\theta - k\theta$$
$$k = n\theta$$

$$\boxed{\hat{\theta}_{\text{MLE}} = \frac{k}{n}}$$

### **왜 미분=0?**

> "ℓ(θ)는 (0,1) 내부에서 미분 가능한 함수이고, 최댓값은 내부의 어느 점에서 달성된다 (k=0이나 k=n인 경계는 별도 확인). **페르마 정리**(미분 가능한 함수의 내부 극값에서 1차 도함수 = 0)에 의해 $d\ell/d\theta = 0$인 θ가 후보다."

### 2계 미분으로 최댓값 검증

$$\frac{d^2\ell}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2} < 0$$

두 항 모두 음수 → ℓ은 **오목**(concave) → 임계점이 **최댓값** ✓

---

## 결과 요약

| 단계 | 결과 |
|-----|------|
| 우도 | $L(\theta) = \theta^k(1-\theta)^{n-k}$ |
| 로그우도 | $\ell(\theta) = k\log\theta + (n-k)\log(1-\theta)$ |
| NLL | $\text{NLL} = -k\log\theta - (n-k)\log(1-\theta)$ |
| MLE | $\hat{\theta} = k/n$ (관측 비율) |

---

## MLE와의 관계 서술 (문제 (3))

> "MLE는 우도(또는 로그우도)를 최대화하는 모수 추정이다. NLL = -log L의 최소화는 log L 최대화와 동치이므로 결과는 같다. 베르누이의 경우 $\hat{\theta}_{\text{MLE}} = k/n$이며, 이는 **표본 비율**과 일치한다 — 매우 직관적인 결과로 MLE의 자연스러움을 보여준다."

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md) — 시험 답안 형식.

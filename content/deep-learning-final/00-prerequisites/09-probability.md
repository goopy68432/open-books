---
title: "09. 확률 입문 — 확률변수와 분포"
slug: probability
order: 9
---

# 09. 확률 입문 — 확률변수와 분포

> 모든 통계 문제의 출발점. 확률변수가 뭔지 모르면 N(0,1)도 Bern(θ)도 외계어.

---

## 1. 왜 배우나

- 기출 2번: $X \sim N(0,1)$ — 확률변수와 정규분포
- 기출 3번: $X \sim$ Uniform — 균일분포
- 기출 4-7번: $y_i \sim$ Bern(θ) — 베르누이
- "i.i.d" 정확히 이해해야 곱 형태 우도 도출

---

## 2. 표본공간과 사건

### 표본공간 $\Omega$
실험에서 일어날 수 있는 **모든 결과**의 집합.
- 동전 던지기: $\Omega = \{H, T\}$
- 주사위: $\Omega = \{1, 2, 3, 4, 5, 6\}$

### 사건 (event)
표본공간의 **부분집합**.
- "짝수가 나온다" = $\{2, 4, 6\}$

### 확률 $P(\cdot)$
사건에 [0, 1] 사이 숫자 부여.
- $P(\Omega) = 1$
- $P(A \cup B) = P(A) + P(B)$ (서로소일 때)

---

## 3. 확률변수 (Random Variable)

### 정의
표본공간의 결과에 **숫자**를 붙이는 함수: $X: \Omega \to \mathbb{R}$.

### 직관
"동전 두 번 던져서 앞면 개수" — X는 0, 1, 2 중 하나.

### 표기 관례
- 대문자 $X, Y, Z$: 확률변수 자체 (값이 정해지기 전)
- 소문자 $x, y, z$: 실제 관측된 값

---

## 4. 이산형 vs 연속형

### 이산형 (Discrete)
값이 셀 수 있음 (0, 1, 2, ...)

**확률질량함수 (pmf):** $p(x) = P(X = x)$
- 기출 4번 베르누이: $p(y) = \theta^y (1-\theta)^{1-y}$, $y \in \{0, 1\}$

성질:
- $0 \leq p(x) \leq 1$
- $\sum_x p(x) = 1$

### 연속형 (Continuous)
값이 실수 전체

**확률밀도함수 (pdf):** $p(x)$
- 기출 2번 정규분포: $p(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$
- 기출 3번 균일분포: $p(x) = 1/(b-a)$ for $x \in [a,b]$

성질:
- $p(x) \geq 0$
- $\int_{-\infty}^\infty p(x)\,dx = 1$
- $P(a \leq X \leq b) = \int_a^b p(x)\,dx$

**주의:** 연속형은 $P(X = x) = 0$ (한 점의 확률 = 0). 구간 확률만 의미 있음.

---

## 5. 시험에 나오는 4대 분포

### ① 베르누이 분포 Bern(θ) — 기출 4-7번
$$y \in \{0, 1\}, \quad P(y=1) = \theta, \quad P(y=0) = 1-\theta$$
$$p(y|\theta) = \theta^y(1-\theta)^{1-y}$$

**예:** 동전 던지기 (앞면 확률 θ)

### ② 이항분포 Binomial(n, θ)
n번 베르누이 시행, 앞면 개수 k.
$$P(K=k) = \binom{n}{k} \theta^k (1-\theta)^{n-k}$$

**기출 4번:** k = ∑yᵢ는 이항분포 따름. (직접 출제는 베르누이 i.i.d로)

### ③ 균일분포 Uniform[a, b] — 기출 3번
$$p(x) = \frac{1}{b-a}, \quad x \in [a, b]$$

**예:** 막대를 무작위로 자른 위치

### ④ 정규분포 N(μ, σ²) — 기출 2번
$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**표준정규분포** $N(0, 1)$: $\mu=0, \sigma=1$.

---

## 6. **i.i.d** — 시험의 핵심 가정

### 풀어 쓰면
**i**ndependent (독립) **i**dentically **d**istributed (동일분포)

### 의미
1. **독립:** 한 시행이 다른 시행에 영향 없음
2. **동일분포:** 모두 같은 분포에서 추출

### 표기
$$y_1, y_2, \dots, y_n \overset{\text{i.i.d}}{\sim} \text{Bern}(\theta)$$

### **왜 i.i.d면 곱이 되는가?**

독립의 정의:
$$P(A \cap B) = P(A) P(B)$$

확장: 여러 사건이 독립이면 결합확률 = 각자 확률의 **곱**.

따라서:
$$p(y_1, y_2, \dots, y_n | \theta) = \prod_{i=1}^n p(y_i | \theta)$$

(각 $y_i$가 동일분포 → $p(y_i|\theta)$가 모두 같은 형태)

**시험 답안 표준 문장:**
> "$y_i$들이 i.i.d 가정 하에 있으므로, 결합확률은 각 확률의 곱이다."

---

## 7. 조건부확률과 베이즈

### 조건부확률
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

"B가 일어났을 때 A가 일어날 확률"

### 베이즈 정리
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

또는 **시험 표기**:
$$p(\theta|D) = \frac{p(D|\theta) p(\theta)}{p(D)} \propto p(D|\theta) p(\theta)$$

- $p(D|\theta)$: 우도 (likelihood)
- $p(\theta)$: 사전분포 (prior)
- $p(\theta|D)$: 사후분포 (posterior)
- $p(D)$: 정규화 상수 (θ에 무관 → MAP에서 무시)

**기출 5,6,7번:** MAP은 사후분포의 $\arg\max$.

---

## 8. 다리 놓기

| 중학교 | 대학 |
|-------|------|
| 동전 앞면 확률 1/2 | 베르누이 Bern(1/2) |
| 주사위 평균 3.5 | 기댓값 $E[X]$ |
| (배운 적 없음) | 확률변수, pdf, pmf |
| (배운 적 없음) | i.i.d → 곱 |

---

## 9. 시험 답안 작성법

### 모델 명시 표준 문장

답안 첫 줄에 항상:
> "$y_i \sim \text{Bern}(\theta)$ i.i.d, $i = 1, \dots, n$"

→ 채점관이 "이 학생, 가정을 명확히 했군" 인식.

### 곱 변환 명시

> "i.i.d 가정에 의해, 결합확률은 각 확률의 곱이다:
> $p(y_1, \dots, y_n | \theta) = \prod_{i=1}^n p(y_i | \theta) = \prod \theta^{y_i}(1-\theta)^{1-y_i}$"

---

## 10. 자가 점검

1. 동전을 3번 던졌을 때 앞면 개수 X의 pmf?
2. $X \sim$ Uniform[0, 4]일 때 $P(1 \leq X \leq 3)$?
3. i.i.d 가정 하에 $p(y_1=1, y_2=0, y_3=1 | \theta) = ?$ (Bern(θ))
4. 베이즈 정리에서 사전분포 $p(\theta)$는 데이터 보기 전 / 후?
5. 사후분포는 $\propto$ ?

**답:**
1. $P(X=k) = \binom{3}{k}(1/2)^3$, k=0,1,2,3
2. $\int_1^3 \frac{1}{4}dx = 1/2$
3. $\theta \cdot (1-\theta) \cdot \theta = \theta^2(1-\theta)$
4. **데이터 보기 전** (prior = "사전")
5. $p(D|\theta) p(\theta)$ (likelihood × prior)

---

## 다음 챕터

[`10-expectation.md`](./10-expectation.md) — 기댓값, 분산, 모멘트.

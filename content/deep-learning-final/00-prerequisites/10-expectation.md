---
title: "10. 기댓값과 분산 — 적분으로 정의되는 핵심"
slug: expectation
order: 10
---

# 10. 기댓값과 분산 — 적분으로 정의되는 핵심

> 기출 2,3번을 푸는 핵심 도구. 적분 기호와 친해져야 합니다.

---

## 1. 왜 배우나

- 기출 2번: $E[X], E[X^2], E[X^3], E[X^4]$ for $X \sim N(0,1)$
- 기출 3번: $E[X], \text{Var}[X]$ for $X \sim$ Uniform
- 데이터 분석의 가장 기본 통계량

---

## 2. 기댓값 (Expectation)

### 직관
"평균"의 정확한 정의. 무한히 많이 시행했을 때 결과의 평균.

### 이산형 정의
$$E[X] = \sum_x x \cdot p(x)$$

**예:** 주사위 ($X \in \{1, \dots, 6\}$, $p(x) = 1/6$)
$$E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1+2+\cdots+6}{6} = 3.5$$

### 연속형 정의
$$E[X] = \int_{-\infty}^\infty x \cdot p(x)\,dx$$

**예:** $X \sim$ Uniform[a, b]
$$E[X] = \int_a^b x \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \frac{x^2}{2}\Big|_a^b = \frac{b^2 - a^2}{2(b-a)} = \frac{a+b}{2}$$

(예상대로 중간값!)

---

## 3. 기댓값의 성질

### ① 선형성
$$E[aX + b] = a E[X] + b$$
$$E[X + Y] = E[X] + E[Y]$$

(독립 여부와 무관, **항상** 성립)

### ② 함수의 기댓값 (LOTUS — Law Of The Unconscious Statistician)
$$E[g(X)] = \int g(x) p(x)\,dx$$

특히:
$$E[X^k] = \int x^k p(x)\,dx \quad \text{(k차 모멘트)}$$

**기출 2번이 정확히 이 형태!**

### ③ 독립이면 곱의 기댓값 = 기댓값의 곱
$X, Y$ 독립일 때:
$$E[XY] = E[X] \cdot E[Y]$$

(독립 아닐 때는 일반적으로 성립 안 함)

---

## 4. 분산 (Variance)

### 정의
$$\text{Var}[X] = E[(X - E[X])^2]$$

"평균에서 얼마나 흩어져 있나"

### 계산 공식 (시험에서 자주 쓰는 형태)
$$\text{Var}[X] = E[X^2] - (E[X])^2$$

#### 증명 (3줄)
$$\text{Var}[X] = E[(X - \mu)^2] \quad (\mu = E[X])$$
$$= E[X^2 - 2\mu X + \mu^2]$$
$$= E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2 \checkmark$$

### 표준편차
$$\sigma = \sqrt{\text{Var}[X]}$$

---

## 5. 분산의 성질

### ① $\text{Var}[c] = 0$ (상수의 분산은 0)
### ② $\text{Var}[aX + b] = a^2 \text{Var}[X]$ (b는 평행이동, 분산 무관)
### ③ 독립이면: $\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y]$

---

## 6. 자주 쓰는 분포의 기댓값·분산

| 분포 | E[X] | Var[X] |
|------|------|--------|
| Bern(θ) | θ | θ(1-θ) |
| Binomial(n, θ) | nθ | nθ(1-θ) |
| Uniform[a, b] | (a+b)/2 | (b-a)²/12 |
| N(μ, σ²) | μ | σ² |
| Poisson(λ) | λ | λ |

---

## 7. 모멘트 (Moments) — 기출 2번 핵심

### k차 모멘트
$$\mu_k = E[X^k]$$

### 표준정규분포의 모멘트 (외워라!)

$X \sim N(0, 1)$:
$$E[X^k] = \begin{cases} 0 & k \text{ 홀수} \\ (k-1)!! & k \text{ 짝수} \end{cases}$$

여기서 $(k-1)!! = (k-1)(k-3)(k-5)\cdots$ (이중계승)

**구체값:**
- $E[X] = 0$ (대칭)
- $E[X^2] = 1!! = 1$ (분산 = 1)
- $E[X^3] = 0$ (대칭)
- $E[X^4] = 3!! = 3 \cdot 1 = 3$
- $E[X^5] = 0$
- $E[X^6] = 5!! = 5 \cdot 3 \cdot 1 = 15$

이게 기출 2번 답입니다!

---

## 8. 기출 3번 적용

$X \sim$ Uniform[a, b]일 때:

### E[X]
$$E[X] = \int_a^b x \cdot \frac{1}{b-a}\,dx = \frac{b^2 - a^2}{2(b-a)} = \frac{a+b}{2}$$

### E[X²]
$$E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\,dx = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}$$

### Var[X]
$$\text{Var}[X] = E[X^2] - (E[X])^2 = \frac{a^2 + ab + b^2}{3} - \left(\frac{a+b}{2}\right)^2$$

전개:
$$= \frac{4(a^2+ab+b^2) - 3(a+b)^2}{12} = \frac{4a^2 + 4ab + 4b^2 - 3a^2 - 6ab - 3b^2}{12}$$
$$= \frac{a^2 - 2ab + b^2}{12} = \frac{(a-b)^2}{12} = \frac{(b-a)^2}{12}$$

---

## 9. 다리 놓기

| 중학교 | 대학 |
|-------|------|
| 자료의 평균 = 합/개수 | 기댓값 $E[X]$ |
| 분산 = (편차)² 평균 | $\text{Var}[X] = E[(X-\mu)^2]$ |
| (배운 적 없음) | $E[X^k]$ 모멘트 |
| (배운 적 없음) | LOTUS: $E[g(X)] = \int g(x)p(x)dx$ |

---

## 10. 시험 답안 작성법

### 정의 인용

"$E[X]$는 $X$의 기댓값으로 $\int x p(x)\,dx$로 정의된다."

### 대칭성 활용

X ~ N(0,1)에서 $E[X^3]$:
> "피적분함수 $x^3 e^{-x^2/2}$는 기함수이고, 적분 구간이 대칭(-∞, ∞)이므로 적분값은 0."

### 모멘트 일반화

> "표준정규분포의 짝수차 모멘트 $E[X^{2n}] = (2n-1)!!$이다."

---

## 11. 자가 점검

1. 주사위 1번 던지기: $E[X], \text{Var}[X]$
2. $X \sim$ Bern(0.3): $E[X], \text{Var}[X]$
3. $X \sim$ Uniform[0, 1]: $E[X], \text{Var}[X]$
4. $X \sim N(0,1)$: $E[X^2 + 3X + 1]$
5. $X \sim N(0,1)$: $E[X^4]$

**답:**
1. $E[X] = 3.5$, $\text{Var}[X] = 35/12 \approx 2.92$
2. $0.3, 0.21$
3. $0.5, 1/12$
4. $E[X^2] + 3E[X] + 1 = 1 + 0 + 1 = 2$
5. $3$

---

## 사전지식 마무리

10개 챕터를 모두 읽고 자가 점검을 완료했다면, 이제 기출 8문제로 들어갈 준비가 됐습니다.

**다음 단계:**
- [`../01-eigen/00-overview.md`](../01-eigen/00-overview.md) — 기출 1번 (고유값/고유벡터)부터.

또는 README의 "8주 일정"을 따라 1주차 사전지식을 한 번 더 복습하세요.

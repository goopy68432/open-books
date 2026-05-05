---
title: "02. 단계별 유도 — E(X), E(X²), Var(X)"
slug: derivation
order: 2
---

# 02. 단계별 유도 — E(X), E(X²), Var(X)

---

## 1단계: pdf 명시

$X \sim$ Uniform[a, b]:
$$p(x) = \frac{1}{b-a}, \quad x \in [a, b].$$

**왜 명시?** 답안의 출발점. 채점관에게 "정의를 안다" 표시.

---

## 2단계: E(X) 계산

기댓값 정의:
$$E[X] = \int_{-\infty}^\infty x\, p(x)\,dx = \int_a^b x \cdot \frac{1}{b-a}\,dx$$

(밖에서는 $p = 0$이므로 [a,b]만 적분)

상수 $\frac{1}{b-a}$ 빼내고:
$$= \frac{1}{b-a} \int_a^b x\,dx = \frac{1}{b-a} \cdot \left[\frac{x^2}{2}\right]_a^b = \frac{1}{b-a} \cdot \frac{b^2 - a^2}{2}$$

**인수분해** $b^2 - a^2 = (b-a)(b+a)$:
$$E[X] = \frac{(b-a)(b+a)}{2(b-a)} = \frac{a+b}{2}$$

$$\boxed{E[X] = \frac{a+b}{2}}$$

---

## 3단계: E(X²) 계산

$$E[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\,dx = \frac{1}{b-a} \cdot \left[\frac{x^3}{3}\right]_a^b = \frac{b^3 - a^3}{3(b-a)}$$

**인수분해** $b^3 - a^3 = (b-a)(a^2 + ab + b^2)$:
$$E[X^2] = \frac{(b-a)(a^2 + ab + b^2)}{3(b-a)} = \frac{a^2 + ab + b^2}{3}$$

$$\boxed{E[X^2] = \frac{a^2 + ab + b^2}{3}}$$

---

## 4단계: Var(X) 계산

$$\text{Var}[X] = E[X^2] - (E[X])^2 = \frac{a^2 + ab + b^2}{3} - \left(\frac{a+b}{2}\right)^2$$

**공통분모 12로 통분:**
$$= \frac{4(a^2 + ab + b^2)}{12} - \frac{3(a+b)^2}{12}$$

분자 전개:
- $4(a^2 + ab + b^2) = 4a^2 + 4ab + 4b^2$
- $3(a+b)^2 = 3(a^2 + 2ab + b^2) = 3a^2 + 6ab + 3b^2$

차이:
$$4a^2 + 4ab + 4b^2 - 3a^2 - 6ab - 3b^2 = a^2 - 2ab + b^2 = (a - b)^2 = (b - a)^2$$

따라서:
$$\text{Var}[X] = \frac{(b-a)^2}{12}$$

$$\boxed{\text{Var}[X] = \frac{(b-a)^2}{12}}$$

---

## 5단계: 검증

### 특수 케이스: $X \sim$ Uniform[0, 1]
- $E[X] = 1/2$ ✓ (잘 알려진 사실)
- $\text{Var}[X] = 1/12$ ✓

### 특수 케이스: $X \sim$ Uniform[-1, 1]
- $E[X] = 0$ (대칭 → 중심) ✓
- $\text{Var}[X] = (1-(-1))^2/12 = 4/12 = 1/3$ ✓

---

## 핵심 도구 정리

| 도구 | 용도 |
|------|-----|
| 적분 정의 $E[X] = \int x p\,dx$ | 기댓값 |
| 분산 공식 $E[X^2] - \mu^2$ | 분산 |
| 인수분해 $b^n - a^n = (b-a)(\cdots)$ | $b-a$ 약분 |
| 통분 (분모 12) | 분산 깔끔하게 |

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md)

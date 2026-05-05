---
title: "05. 적분 입문 — \"넓이\"를 구하는 도구"
slug: 00-prerequisites-05-integral-101
order: 5
---

# 05. 적분 입문 — "넓이"를 구하는 도구

> 기댓값 E[X], 분산 Var[X], 가우스 적분 — 모두 적분으로 정의됩니다.

---

## 1. 왜 배우나

- 기출 2번: $\int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} e^{-x^2/2} dx = 1$ 증명
- 기출 2,3번: $E[X^k] = \int x^k p(x) dx$
- 기출 3번: 균일분포 분산 = 적분 계산

---

## 2. 직관 — 적분 = 넓이

$\int_a^b f(x)\,dx$ = 함수 $f(x)$의 그래프 아래, $x = a$부터 $x = b$까지의 넓이.

### 그림으로

```
   y
   │   ┌──────┐
   │   │░░░░░░│ ← 이 영역의 넓이
   │   │░░░░░░│
   │   │░░░░░░│
   └───┴──────┴──── x
       a      b
```

### 부호

f(x)가 음수면 넓이도 음수로 친다. (부호 있는 넓이)

---

## 3. 정적분 vs 부정적분

### 부정적분 (반도함수)

$f(x) = 2x$의 부정적분은 $F(x) = x^2 + C$
- 검산: $F'(x) = 2x = f(x)$ ✓
- C는 임의의 상수

기호: $\int f(x)\,dx = F(x) + C$

### 정적분

$\int_a^b f(x)\,dx = F(b) - F(a)$
- a부터 b까지의 넓이
- 임의 상수 C는 빼는 과정에서 사라짐

---

## 4. 미적분의 기본 정리

$$\frac{d}{dx} \int_a^x f(t)\,dt = f(x)$$

**한 줄 요약:** 미분과 적분은 서로 역연산.

---

## 5. 핵심 적분 공식 5개

| f(x) | $\int f\,dx$ | 비고 |
|------|------------|------|
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$ ($n \neq -1$) | 가장 기본 |
| $\frac{1}{x}$ | $\log\|x\| + C$ | $n = -1$ 예외 |
| $e^x$ | $e^x + C$ | 자기 자신 |
| $e^{kx}$ | $\frac{1}{k} e^{kx} + C$ | 체인 룰 역 |
| $\sin x$ | $-\cos x + C$ | (시험 범위 아님) |

---

## 6. 적분 규칙

### ① 상수배
$$\int c f(x)\,dx = c \int f(x)\,dx$$

### ② 합
$$\int (f + g)\,dx = \int f\,dx + \int g\,dx$$

### ③ 부분적분 (Integration by Parts)
$$\int u\,dv = uv - \int v\,du$$

또는:
$$\int u(x)v'(x)\,dx = u(x)v(x) - \int u'(x)v(x)\,dx$$

**기출 2번 E(X²), E(X⁴) 계산에서 핵심!**

### ④ 치환적분
$u = g(x)$로 두고 $du = g'(x)dx$:
$$\int f(g(x))g'(x)\,dx = \int f(u)\,du$$

**가우스 적분 증명에서 극좌표 치환 사용!**

---

## 7. 정적분의 핵심 성질

### 적분의 한계

$$\int_a^b = \int_a^c + \int_c^b$$

(중간 점 c로 쪼개기 가능)

### 부호 바꿔 적분
$$\int_b^a = -\int_a^b$$

### 우함수/기함수 적분 (대칭성!)

**기함수**(odd function): $f(-x) = -f(x)$
- 예: $f(x) = x$, $f(x) = x^3$, $f(x) = \sin x$
- 대칭구간 적분: $\int_{-a}^a f\,dx = 0$ (양/음이 상쇄)

**우함수**(even function): $f(-x) = f(x)$
- 예: $f(x) = x^2$, $f(x) = \cos x$, $f(x) = e^{-x^2/2}$
- 대칭구간 적분: $\int_{-a}^a f\,dx = 2 \int_0^a f\,dx$

### 기출 2번 핵심 트릭

$$E[X] = \int_{-\infty}^\infty x \cdot \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx$$

피적분함수: $x \cdot e^{-x^2/2}$
- $x$: 기함수
- $e^{-x^2/2}$: 우함수
- 곱: 기함수
- 대칭구간 적분: **0** (계산 안 해도 됨!)

→ $E[X] = 0$, $E[X^3] = 0$ 즉시!

---

## 8. 부분적분 자세히 보기 (기출 2번 핵심)

### 공식
$$\int u\,dv = uv - \int v\,du$$

### 사용 전략 (LIATE 규칙)

u로 선택할 우선순위:
- **L**ogarithmic ($\log x$)
- **I**nverse trig ($\arcsin x$)
- **A**lgebraic ($x^n$)
- **T**rig ($\sin x$)
- **E**xponential ($e^x$)

### 예시: $\int x e^{-x^2/2}\,dx$

치환 $u = -x^2/2$, $du = -x\,dx$:
$$\int x e^{-x^2/2}\,dx = \int -e^u\,du = -e^{-x^2/2} + C$$

### 예시: E[X²] for X ~ N(0,1)

$$E[X^2] = \int_{-\infty}^\infty x^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\,dx$$

부분적분: $u = x$, $dv = x e^{-x^2/2}\,dx$
- $du = dx$, $v = -e^{-x^2/2}$

$$\int x^2 e^{-x^2/2}\,dx = -x e^{-x^2/2} \Big|_{-\infty}^\infty + \int_{-\infty}^\infty e^{-x^2/2}\,dx = 0 + \sqrt{2\pi}$$

따라서 $E[X^2] = \frac{1}{\sqrt{2\pi}} \cdot \sqrt{2\pi} = 1$.

(자세한 풀이는 `02-gaussian/02-derivation.md`)

---

## 9. 가우스 적분 — 시험의 보스

$$\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$$

이 식은 **부정적분으로 풀 수 없습니다** (초등함수 표현 불가). 그래서 트릭 사용:

### 증명 스케치 (자세한 건 `10-ten-proofs/05-gaussian-integral.md`)

1. $I = \int_{-\infty}^\infty e^{-x^2/2}\,dx$로 둠.
2. $I^2 = \left(\int e^{-x^2/2}\,dx\right)\left(\int e^{-y^2/2}\,dy\right) = \iint e^{-(x^2+y^2)/2}\,dx\,dy$
3. **극좌표 변환**: $x = r\cos\phi$, $y = r\sin\phi$, $dx\,dy = r\,dr\,d\phi$
4. $I^2 = \int_0^{2\pi} \int_0^\infty e^{-r^2/2} \cdot r\,dr\,d\phi = 2\pi \cdot 1 = 2\pi$
5. $I = \sqrt{2\pi}$ ✓

---

## 10. 자가 점검

1. $\int_0^1 x^2\,dx$
2. $\int_{-1}^{1} x^3\,dx$ (대칭성으로!)
3. $\int_0^\infty x e^{-x}\,dx$ (부분적분)
4. $\int_a^b \frac{1}{b-a}\,dx$
5. $\int_{-\infty}^\infty x \cdot \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx$ (대칭성!)

**답:**
1. $1/3$
2. $0$ (기함수)
3. $1$ (감마함수 Γ(2))
4. $1$ (균일분포 정규화 확인)
5. $0$ (E[X] for N(0,1))

---

## 다음 챕터

[`06-vector-matrix.md`](./06-vector-matrix.md) — 벡터·행렬 기초.

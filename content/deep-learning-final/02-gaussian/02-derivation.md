---
title: "02. 단계별 유도 — 가우스 적분 + 4개 모멘트"
slug: derivation
order: 2
---

# 02. 단계별 유도 — 가우스 적분 + 4개 모멘트

> 보조정리(가우스 적분) 1개 + 모멘트 4개 = 5개 풀이.

---

## 보조정리: 가우스 적분 증명

**증명할 것:** $\displaystyle \int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$

**왜 보조정리부터?** 모든 모멘트 계산에 이 결과가 등장합니다. 시험에서 "이걸 증명하라"고 직접 묻기도 함.

### 증명 (극좌표 변환 트릭)

#### 1단계: 제곱

$I = \int_{-\infty}^\infty e^{-x^2/2}\,dx$로 두자. 그러면

$$I^2 = \left(\int_{-\infty}^\infty e^{-x^2/2}\,dx\right)\left(\int_{-\infty}^\infty e^{-y^2/2}\,dy\right)$$

**왜 변수를 y로 바꿨나?** 두 적분이 독립이고, 곱을 이중적분으로 합치기 위해.

#### 2단계: 이중적분

$$I^2 = \int_{-\infty}^\infty \int_{-\infty}^\infty e^{-x^2/2} e^{-y^2/2}\,dx\,dy = \int_{-\infty}^\infty \int_{-\infty}^\infty e^{-(x^2+y^2)/2}\,dx\,dy$$

**왜 합쳐지나?** 푸비니 정리(Fubini): 가능한 적분 곱은 이중적분으로.

#### 3단계: 극좌표 변환

$x = r\cos\phi$, $y = r\sin\phi$. 야코비안 $|J| = r$ → $dx\,dy = r\,dr\,d\phi$.

또한 $x^2 + y^2 = r^2$.

적분 영역: $r \in [0, \infty)$, $\phi \in [0, 2\pi)$.

$$I^2 = \int_0^{2\pi} \int_0^\infty e^{-r^2/2} \cdot r \,dr\,d\phi$$

**왜 극좌표?** $e^{-(x^2+y^2)/2}$가 $r$만의 함수가 됨 → 적분 분리 가능.

#### 4단계: 안쪽 적분 ($r$에 대해)

치환 $u = r^2/2$, $du = r\,dr$:
$$\int_0^\infty e^{-r^2/2} r\,dr = \int_0^\infty e^{-u}\,du = [-e^{-u}]_0^\infty = 0 - (-1) = 1$$

#### 5단계: 바깥쪽 적분 ($\phi$에 대해)

$$I^2 = \int_0^{2\pi} 1 \,d\phi = 2\pi$$

#### 6단계: 제곱근

$I = \sqrt{2\pi}$ (양수이므로 양의 부호)

$$\boxed{\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}} \quad \blacksquare$$

---

## 결론으로부터 따라오는 사실

$$\int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\,dx = 1$$

→ N(0,1)의 pdf가 정규화되어 있다. 즉 $\int p(x)\,dx = 1$.

---

## E(X) — 1차 모멘트

$$E[X] = \int_{-\infty}^\infty x \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\,dx$$

### 풀이: 대칭성

피적분함수 $f(x) = x \cdot e^{-x^2/2}$:
- $x$: 기함수 ($x \mapsto -x$이면 부호 반전)
- $e^{-x^2/2}$: 우함수 ($x^2$이 대칭)
- 곱: **기함수**

기함수의 대칭구간 적분은 0:
$$\int_{-\infty}^\infty x \cdot e^{-x^2/2}\,dx = 0$$

$$\boxed{E[X] = 0}$$

**시험 답안에:** "피적분함수가 기함수이고 적분 구간이 대칭이므로 $E[X] = 0$이다."

---

## E(X²) — 2차 모멘트

$$E[X^2] = \int_{-\infty}^\infty x^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\,dx$$

### 풀이: 부분적분

부분적분 $\int u\,dv = uv - \int v\,du$ 적용.

**선택:**
- $u = x$, $du = dx$
- $dv = x e^{-x^2/2}\,dx$, $v = -e^{-x^2/2}$ (왜? $\frac{d}{dx}(-e^{-x^2/2}) = x e^{-x^2/2}$)

$$\int_{-\infty}^\infty x^2 e^{-x^2/2}\,dx = \int_{-\infty}^\infty x \cdot (x e^{-x^2/2})\,dx$$

$$= \left[-x e^{-x^2/2}\right]_{-\infty}^\infty + \int_{-\infty}^\infty e^{-x^2/2}\,dx$$

**경계항:** $\lim_{x \to \pm\infty} x e^{-x^2/2} = 0$ (지수 감쇠가 다항식 발산을 압도). 따라서 0.

**남은 적분:** 가우스 적분 = $\sqrt{2\pi}$.

$$\int_{-\infty}^\infty x^2 e^{-x^2/2}\,dx = 0 + \sqrt{2\pi} = \sqrt{2\pi}$$

$$E[X^2] = \frac{1}{\sqrt{2\pi}} \cdot \sqrt{2\pi} = 1$$

$$\boxed{E[X^2] = 1}$$

**해석:** N(0,1)의 분산 = $E[X^2] - (E[X])^2 = 1 - 0 = 1$. 정의와 일치.

---

## E(X³) — 3차 모멘트

$$E[X^3] = \int_{-\infty}^\infty x^3 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\,dx$$

### 풀이: 대칭성

$x^3 e^{-x^2/2}$:
- $x^3$: 기함수
- $e^{-x^2/2}$: 우함수
- 곱: **기함수**

대칭구간 적분 = 0.

$$\boxed{E[X^3] = 0}$$

---

## E(X⁴) — 4차 모멘트

$$E[X^4] = \int_{-\infty}^\infty x^4 \cdot \frac{1}{\sqrt{2\pi}} e^{-x^2/2}\,dx$$

### 풀이: 부분적분 (또는 재귀)

**선택 1: 부분적분 직접**

- $u = x^3$, $du = 3x^2\,dx$
- $dv = x e^{-x^2/2}\,dx$, $v = -e^{-x^2/2}$

$$\int x^4 e^{-x^2/2}\,dx = \int x^3 \cdot (x e^{-x^2/2})\,dx$$

$$= \left[-x^3 e^{-x^2/2}\right]_{-\infty}^\infty + \int 3x^2 e^{-x^2/2}\,dx$$

경계항 = 0 (지수 감쇠).

$$= 3 \int_{-\infty}^\infty x^2 e^{-x^2/2}\,dx = 3 \sqrt{2\pi}$$

따라서:
$$E[X^4] = \frac{1}{\sqrt{2\pi}} \cdot 3\sqrt{2\pi} = 3$$

$$\boxed{E[X^4] = 3}$$

### 선택 2: 재귀 공식

부분적분 일반화: $E[X^k] = (k-1) \cdot E[X^{k-2}]$ for k ≥ 2.

**유도:**
$$\int x^k e^{-x^2/2}\,dx \overset{\text{부분적분}}{=} (k-1)\int x^{k-2} e^{-x^2/2}\,dx$$

$E[X^k] = (k-1)E[X^{k-2}]$.

- $E[X^0] = 1$ (전체 확률)
- $E[X^2] = 1 \cdot E[X^0] = 1$
- $E[X^4] = 3 \cdot E[X^2] = 3$
- $E[X^6] = 5 \cdot E[X^4] = 15$
- $E[X^{2n}] = (2n-1)!!$

---

## 요약표

| 모멘트 | 값 | 핵심 도구 |
|-------|-----|---------|
| $E[X]$ | 0 | 대칭성 (기함수) |
| $E[X^2]$ | 1 | 부분적분 + 가우스 적분 |
| $E[X^3]$ | 0 | 대칭성 |
| $E[X^4]$ | 3 | 부분적분 (또는 재귀) |

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md) — 시험장 그대로 답안.

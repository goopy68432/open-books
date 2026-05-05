---
title: "02. 단계별 유도 — m=2, m=6 케이스"
slug: 07-map-tent-02-derivation
order: 43
---

# 02. 단계별 유도 — m=2, m=6 케이스

## 공통 설정

- $L(\theta) = \theta^4(1-\theta)$ (n=5, k=4)
- $p_m(\theta) = m - m^2|\theta - 0.5|$ (정의역 내), 외부 0
- posterior $\propto L(\theta) \cdot p_m(\theta)$ (정의역 내)

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## CASE 1: m = 2

### 1.1 prior 형태
- 정의역: $|\theta - 0.5| \leq 1/2 \Rightarrow \theta \in [0, 1]$
- $p_2(\theta) = 2 - 4|\theta - 0.5|$

### 1.2 두 영역으로 분리

**(a) $\theta \in [0, 0.5]$:** $|\theta - 0.5| = 0.5 - \theta$
$$p_2(\theta) = 2 - 4(0.5 - \theta) = 4\theta$$

**(b) $\theta \in [0.5, 1]$:** $|\theta - 0.5| = \theta - 0.5$
$$p_2(\theta) = 2 - 4(\theta - 0.5) = 4 - 4\theta$$

### 1.3 각 영역의 posterior

**(a)** $\theta \in [0, 0.5]$:
$$f_a(\theta) = \theta^4(1-\theta) \cdot 4\theta = 4\theta^5(1-\theta)$$

**(b)** $\theta \in [0.5, 1]$:
$$f_b(\theta) = \theta^4(1-\theta) \cdot (4 - 4\theta) = 4\theta^4(1-\theta)^2$$

### 1.4 log + 미분

**(a)** $\log f_a = \log 4 + 5\log\theta + \log(1-\theta)$
$$\frac{d}{d\theta}\log f_a = \frac{5}{\theta} - \frac{1}{1-\theta} = 0$$
$$5(1-\theta) = \theta \Rightarrow 5 = 6\theta \Rightarrow \theta = 5/6 \approx 0.833$$

**그러나** $5/6 > 0.5$ → 영역 (a) ($\theta \leq 0.5$) **밖**. 영역 (a)에서 임계점 없음.

(a) 영역 내에서 $f_a$의 미분 부호: $5/\theta - 1/(1-\theta)$의 분자
$5(1-\theta) - \theta = 5 - 6\theta > 0$ when $\theta < 5/6$. 영역 (a)에서 항상 양수 → $f_a$ **단조증가**.

→ 영역 (a)에서 **최댓값 = 우측 끝 $\theta = 0.5$**.

**(b)** $\log f_b = \log 4 + 4\log\theta + 2\log(1-\theta)$
$$\frac{d}{d\theta}\log f_b = \frac{4}{\theta} - \frac{2}{1-\theta} = 0$$
$$4(1-\theta) = 2\theta \Rightarrow 4 = 6\theta \Rightarrow \theta = 2/3 \approx 0.667$$

영역 (b) $[0.5, 1]$ **안**. 임계점 후보.

(b) 영역 내 $f_b$의 미분 부호: $4(1-\theta) - 2\theta = 4 - 6\theta$
- $\theta < 2/3$: 양수 (증가)
- $\theta > 2/3$: 음수 (감소)
- $\theta = 2/3$: 정점 ✓

→ 영역 (b) 최댓값은 $\theta = 2/3$.

### 1.5 두 영역 비교

영역 (a) 최댓값: $\theta = 0.5$, $f_a(0.5) = 4 \cdot (0.5)^5 \cdot 0.5 = 4 \cdot 0.5^6 = 4/64 = 1/16$

영역 (b) 최댓값: $\theta = 2/3$, $f_b(2/3) = 4 \cdot (2/3)^4 \cdot (1/3)^2 = 4 \cdot 16/81 \cdot 1/9 = 64/729 \approx 0.0878$

비교: $1/16 = 0.0625 < 0.0878$ → **(b)가 큼**.

### 1.6 정점 0.5 검토 (미분 불가능 점)

좌극한 (a 영역에서): $f_a(0.5) = 1/16$
우극한 (b 영역에서): $f_b(0.5) = 4 \cdot (0.5)^4 \cdot (0.5)^2 = 4 \cdot 0.5^6 = 1/16$

(연속이지만 미분 불연속)

좌미분 양수 (단조증가), 우미분 부호: $4/0.5 - 2/0.5 = 8 - 4 = 4 > 0$ (영역 b 시작에서도 증가)

→ 0.5는 정점 아님 (계속 증가). 영역 (b)의 2/3가 진짜 최댓값.

### 1.7 m=2 결론

$$\boxed{\hat{\theta}_{\text{MAP}}^{(m=2)} = \frac{2}{3}}$$

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## CASE 2: m = 6

### 2.1 prior 정의역
- $|\theta - 0.5| \leq 1/6 \Rightarrow \theta \in [1/3, 2/3]$
- 외부: $p_6 = 0 \Rightarrow$ posterior = 0

### 2.2 두 영역 분리

**(a)** $\theta \in [1/3, 0.5]$: $p_6 = 6 - 36(0.5 - \theta) = 36\theta - 12$
**(b)** $\theta \in [0.5, 2/3]$: $p_6 = 6 - 36(\theta - 0.5) = 24 - 36\theta$

### 2.3 posterior + log + 미분

**(a)** $f_a = \theta^4(1-\theta)(36\theta - 12) = 12\theta^4(1-\theta)(3\theta - 1)$

$\log f_a = \log 12 + 4\log\theta + \log(1-\theta) + \log(3\theta-1)$ (단, $3\theta > 1$, 즉 $\theta > 1/3$)

$$\frac{d}{d\theta}\log f_a = \frac{4}{\theta} - \frac{1}{1-\theta} + \frac{3}{3\theta-1}$$

이 식 = 0 푸는 건 다항식 → 수치적/대수적 풀이.

#### 다항식으로 풀기

각 항을 공통 분모로:
$$\frac{4(1-\theta)(3\theta-1) - \theta(3\theta-1) + 3\theta(1-\theta)}{\theta(1-\theta)(3\theta-1)} = 0$$

분자 = 0:
$4(1-\theta)(3\theta-1) - \theta(3\theta-1) + 3\theta(1-\theta) = 0$

전개:
- $4(3\theta - 1 - 3\theta^2 + \theta) = -12\theta^2 + 16\theta - 4$
- $-\theta(3\theta-1) = -3\theta^2 + \theta$
- $3\theta(1-\theta) = -3\theta^2 + 3\theta$

합:
$-18\theta^2 + 20\theta - 4 = 0$
$9\theta^2 - 10\theta + 2 = 0$

근의 공식:
$\theta = \frac{10 \pm \sqrt{100 - 72}}{18} = \frac{10 \pm \sqrt{28}}{18} = \frac{10 \pm 2\sqrt{7}}{18} = \frac{5 \pm \sqrt{7}}{9}$

$\sqrt{7} \approx 2.646$:
- $\theta_+ = (5 + 2.646)/9 \approx 0.849$ (영역 (a) [1/3, 0.5] 밖)
- $\theta_- = (5 - 2.646)/9 \approx 0.262$ (영역 (a) 밖)

→ 영역 (a)에서 임계점 없음. 부호 분석:
$\theta = 0.4$ (영역 (a) 내)에 대입:
$\frac{4}{0.4} - \frac{1}{0.6} + \frac{3}{0.2} = 10 - 1.67 + 15 = 23.3 > 0$

→ 영역 (a) 전체에서 **단조증가** → 우측 끝 $\theta = 0.5$가 최댓값.

**(b)** $f_b = \theta^4(1-\theta)(24 - 36\theta) = 12\theta^4(1-\theta)(2 - 3\theta)$

$\log f_b = \log 12 + 4\log\theta + \log(1-\theta) + \log(2 - 3\theta)$ (단, $2 > 3\theta$, 즉 $\theta < 2/3$)

$$\frac{d}{d\theta}\log f_b = \frac{4}{\theta} - \frac{1}{1-\theta} - \frac{3}{2-3\theta}$$

= 0 풀이. 공통 분모:
$4(1-\theta)(2-3\theta) - \theta(2-3\theta) - 3\theta(1-\theta) = 0$

전개:
- $4(2 - 3\theta - 2\theta + 3\theta^2) = 12\theta^2 - 20\theta + 8$
- $-\theta(2-3\theta) = 3\theta^2 - 2\theta$
- $-3\theta(1-\theta) = 3\theta^2 - 3\theta$

합: $18\theta^2 - 25\theta + 8 = 0$

근:
$\theta = \frac{25 \pm \sqrt{625 - 576}}{36} = \frac{25 \pm 7}{36}$

→ $\theta = 32/36 = 8/9 \approx 0.889$ (영역 밖) 또는 $\theta = 18/36 = 0.5$ (영역 (b)의 좌측 경계)

영역 (b) 내부 (열린구간 $(0.5, 2/3)$)에서 임계점 없음.

부호 분석 ($\theta = 0.55$ 대입):
$\frac{4}{0.55} - \frac{1}{0.45} - \frac{3}{0.35} = 7.27 - 2.22 - 8.57 = -3.52 < 0$

→ 영역 (b)에서 **단조감소** → 좌측 끝 $\theta = 0.5$가 최댓값.

### 2.4 m=6 결론

영역 (a)에서 단조증가 → 0.5가 최댓값
영역 (b)에서 단조감소 → 0.5가 최댓값
**중간 미분 불가능 점 0.5가 정점!**

$$\boxed{\hat{\theta}_{\text{MAP}}^{(m=6)} = \frac{1}{2}}$$

---

## 결과 요약

| m | 정의역 | $\hat{\theta}_{\text{MAP}}$ | 메커니즘 |
|---|-------|---------------------------|---------|
| 2 | [0, 1] | **2/3** | 영역 (b)의 내부 극값 |
| 6 | [1/3, 2/3] | **1/2** | 정점 (미분 불가능 점) |

---

## 직관

- **m=2:** 정의역 충분히 넓어서 likelihood (MLE 0.8 쪽으로 잡아당김) 영향이 prior 대칭성보다 강함 → 0.5와 0.8의 절충값 2/3
- **m=6:** prior가 너무 좁고 첨예해서 데이터 영향 무시 → 정점 0.5

→ "m이 클수록 prior 우세, 데이터 무시" 패턴은 5번과 동일.

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md)

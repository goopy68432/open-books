---
title: 'Quiz 9 — Triangular Prior MAP'
description: 'Piecewise prior + 2차방정식 풀이'
draft: false
---

## 0. 한 줄 요약

Prior $p_m(\theta) = m - m^2|\theta - 0.5|$ (받침 $[\frac{1}{2}-\frac{1}{m},\frac{1}{2}+\frac{1}{m}]$) 와 Bernoulli likelihood ($n=5, k=4$) 의 곱으로 정의되는 posterior 를 절댓값에 의한 piecewise 분리 → 각 구간 $\log q(\theta)$ 의 1계 조건 → 임계점의 받침 내 유효성 검사 → 후보값 비교 의 5단 표준 절차로 최대화하면, $m=2$ 에서는 $\theta^*_{\mathrm{MAP}} = 2/3$ (우측 구간 임계점), $m=6$ 에서는 $\theta^*_{\mathrm{MAP}} = 1/2$ (양 구간이 만나는 꼭짓점) 을 얻는다.

---

## 1. 문제 (정확한 출제 형태)

Prior:
$$
p_m(\theta) =
\begin{cases}
m - m^2\,|\theta - 0.5|, & |\theta - 0.5| \le \dfrac{1}{m} \\
0, & \text{otherwise}
\end{cases}
$$

관측: $n = 5$ 회 베르누이 시행에서 $k = 4$ 회 성공.

$m = 2$ 와 $m = 6$ 각각의 MAP 추정량 $\theta^*_{\mathrm{MAP}}$ 을 구하라.

---

## 2. 출제 의도와 시험 가치

이 문제는 다음 능력을 측정한다.

1. **Posterior $\propto$ likelihood × prior** 의 정의를 정확히 적용.
2. **절댓값 함수의 piecewise 분리** — $\theta = 0.5$ 를 기점으로 좌·우 구간 일차식 두 개로 분리.
3. **각 구간 $\log q$ 의 1계 조건** — 분수 미분과 임계점 방정식의 대수 조작.
4. **임계점의 유효성 (받침 내 여부) 검증** — 무효 임계점은 버리고, 단조성 (도함수 부호) 으로 후보를 결정.
5. **후보값 비교**: 임계점 / 경계점 / 꼭짓점 / 받침 끝점의 함숫값을 비교해 전역 최댓값 결정.

이 5단 표준 절차는 prior 의 기하학적 형태(삼각형, 사다리꼴, 그 외 piecewise) 와 무관하게 동일하게 적용되므로, 본 문제는 piecewise 분포 하의 MAP 풀이의 일반 패턴을 학습하는 데 가치가 있다.

---

## 3. 사전 개념 (모든 수학 도구)

### 3.1 Bernoulli likelihood

$n$ 회 시행에서 $k$ 회 성공:
$$
L(\theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k} \;\propto\; \theta^k(1-\theta)^{n-k}
$$

본 문제는 $\propto \theta^4(1-\theta)$.

### 3.2 Posterior

Bayes:
$$
P(\theta\mid \mathcal{D}) = \frac{P(\mathcal{D}\mid\theta)\,p_m(\theta)}{P(\mathcal{D})}
$$
분모는 $\theta$ 무관 상수이므로 $\arg\max$ 에서 무시:
$$
P(\theta\mid \mathcal{D}) \;\propto\; \theta^4(1-\theta)\,p_m(\theta)
$$

### 3.3 MAP 정의

$$
\theta^*_{\mathrm{MAP}} = \arg\max_{\theta\in[0,1]}\bigl[\theta^4(1-\theta)\,p_m(\theta)\bigr]
$$

### 3.4 절댓값의 piecewise 분리

$$
|\theta - 0.5| =
\begin{cases}
0.5 - \theta, & \theta \le 0.5 \\
\theta - 0.5, & \theta \ge 0.5
\end{cases}
$$

### 3.5 로그 변환의 효용

$\log$ 는 단조증가이므로 $\arg\max q(\theta) = \arg\max \log q(\theta)$. $\log$ 변환 후 곱이 합으로 분해되어 미분이 쉬워진다.

### 3.6 분수 미분

$$
\frac{d}{d\theta}\log\theta = \frac{1}{\theta}, \qquad \frac{d}{d\theta}\log(1-\theta) = -\frac{1}{1-\theta}
$$

### 3.7 5단 풀이 절차 (수식 차원의 알고리즘)

1. 각 $m$ 에 대해 prior 를 두 구간 ($\theta\le 0.5$, $\theta\ge 0.5$) 의 일차식으로 분리.
2. 각 구간의 posterior $q(\theta)$ 를 작성.
3. $\frac{d}{d\theta}\log q = 0$ 의 임계점 방정식을 푼다.
4. 임계점이 해당 구간에 속하는지 검증. 없으면 도함수 부호로 단조성 판단 → 경계 후보.
5. 임계점 / 경계 / 꼭짓점 / 받침 끝점 함숫값 비교 → 전역 최댓값.

---

## 4. 풀이 (모든 단계, 등호 근거)

### 4.1 $m = 2$ 케이스

#### Step 1. Prior 의 piecewise 표현

$m = 2$ 대입:
$$
p_2(\theta) = 2 - 4|\theta - 0.5|, \qquad |\theta - 0.5| \le \tfrac{1}{2}
$$
조건은 $0 \le \theta \le 1$ — 즉 받침이 $[0, 1]$ 전 구간.

절댓값 분리:
- $\theta\in[0, 0.5]$: $p_2(\theta) = 2 - 4(0.5-\theta) = 4\theta$.
- $\theta\in[0.5, 1]$: $p_2(\theta) = 2 - 4(\theta-0.5) = 4(1-\theta)$.

#### Step 2. 우측 구간 $\theta\in[0.5, 1]$

Posterior:
$$
q(\theta) = \theta^4(1-\theta)\cdot 4(1-\theta) = 4\theta^4(1-\theta)^2
$$

로그:
$$
\log q(\theta) = \log 4 + 4\log\theta + 2\log(1-\theta)
$$

미분:
$$
\frac{d}{d\theta}\log q = \frac{4}{\theta} - \frac{2}{1-\theta}
$$

1계 조건:
$$
\frac{4}{\theta} = \frac{2}{1-\theta} \iff 4(1-\theta) = 2\theta \iff 4 = 6\theta \iff \theta = \tfrac{2}{3}
$$

구간 검증: $2/3 \in [0.5, 1]$. ✓

2계 조건:
$$
\frac{d^2}{d\theta^2}\log q = -\frac{4}{\theta^2} - \frac{2}{(1-\theta)^2} < 0
$$
strictly concave → $\theta = 2/3$ 는 우측 구간의 유일 최댓값.

#### Step 3. 좌측 구간 $\theta\in[0, 0.5]$

Posterior:
$$
q(\theta) = \theta^4(1-\theta)\cdot 4\theta = 4\theta^5(1-\theta)
$$

로그:
$$
\log q(\theta) = \log 4 + 5\log\theta + \log(1-\theta)
$$

미분:
$$
\frac{d}{d\theta}\log q = \frac{5}{\theta} - \frac{1}{1-\theta}
$$

1계 조건:
$$
\frac{5}{\theta} = \frac{1}{1-\theta} \iff 5(1-\theta) = \theta \iff 5 = 6\theta \iff \theta = \tfrac{5}{6}
$$

구간 검증: $5/6 \notin [0, 0.5]$ → 무효.

부호 검사 ($\theta = 0.4$):
$$
\frac{5}{0.4} - \frac{1}{0.6} = 12.5 - 1.667 > 0
$$
좌측 구간 도함수 양수 → $\log q$ 단조 증가 → 좌측 구간 최댓값 후보는 우측 경계 $\theta = 0.5$.

#### Step 4. 후보값 비교

- $\theta = 2/3$: $q(2/3) = 4\cdot(2/3)^4\cdot(1/3)^2 = 4\cdot\frac{16}{81}\cdot\frac{1}{9} = \frac{64}{729} \approx 0.0878$.
- $\theta = 1/2$: $q(0.5) = 4\cdot(0.5)^5\cdot(0.5) = 4\cdot\frac{1}{32}\cdot\frac{1}{2} = \frac{1}{16} = 0.0625$.

$0.0878 > 0.0625$ → 전역 최댓값은 $\theta = 2/3$.

#### Step 5. 결론

$$
\boxed{\;m = 2:\;\; \theta^*_{\mathrm{MAP}} = \tfrac{2}{3}\;}
$$

---

### 4.2 $m = 6$ 케이스

#### Step 1. Prior 의 piecewise 표현

$m = 6$ 대입:
$$
p_6(\theta) = 6 - 36|\theta - 0.5|, \qquad |\theta - 0.5| \le \tfrac{1}{6}
$$
조건은 $\frac{1}{3} \le \theta \le \frac{2}{3}$.

절댓값 분리:
- $\theta\in[1/3, 0.5]$: $p_6(\theta) = 6 - 36(0.5-\theta) = 36\theta - 12$.
- $\theta\in[0.5, 2/3]$: $p_6(\theta) = 6 - 36(\theta-0.5) = 24 - 36\theta$.

받침 밖에서는 $p_6 = 0 \Rightarrow q = 0$ → 후보 아님.

#### Step 2. 우측 구간 $\theta\in[0.5, 2/3]$

Posterior:
$$
q(\theta) = \theta^4(1-\theta)(24 - 36\theta)
$$

로그 (받침 내부에서 $24 - 36\theta > 0$):
$$
\log q(\theta) = 4\log\theta + \log(1-\theta) + \log(24 - 36\theta)
$$

미분:
$$
\frac{d}{d\theta}\log q = \frac{4}{\theta} - \frac{1}{1-\theta} - \frac{36}{24 - 36\theta}
$$

마지막 항 약분: $\frac{36}{24-36\theta} = \frac{36}{12(2-3\theta)} = \frac{3}{2-3\theta}$.

1계 조건:
$$
\frac{4}{\theta} - \frac{1}{1-\theta} - \frac{3}{2-3\theta} = 0
$$

공통분모 $\theta(1-\theta)(2-3\theta)$ 곱:
$$
4(1-\theta)(2-3\theta) - \theta(2-3\theta) - 3\theta(1-\theta) = 0
$$

각 항 전개:
- $(1-\theta)(2-3\theta) = 2 - 3\theta - 2\theta + 3\theta^2 = 2 - 5\theta + 3\theta^2$
  → $4(\cdot) = 8 - 20\theta + 12\theta^2$.
- $-\theta(2-3\theta) = -2\theta + 3\theta^2$.
- $-3\theta(1-\theta) = -3\theta + 3\theta^2$.

합산:
- $\theta^2$ 항: $12 + 3 + 3 = 18$.
- $\theta^1$ 항: $-20 - 2 - 3 = -25$.
- $\theta^0$ 항: $8$.

$$
\boxed{\;18\theta^2 - 25\theta + 8 = 0\;}
$$

근의 공식:
$$
\theta = \frac{25 \pm \sqrt{625 - 576}}{36} = \frac{25 \pm \sqrt{49}}{36} = \frac{25 \pm 7}{36}
$$

두 근: $\theta_1 = \frac{32}{36} = \frac{8}{9}$, $\theta_2 = \frac{18}{36} = \frac{1}{2}$.

구간 $[0.5, 2/3] \approx [0.5, 0.667]$ 검증:
- $8/9 \approx 0.889 \notin$ 구간 → 무효.
- $1/2$ 는 구간의 좌측 경계점 (꼭짓점이기도 함).

내부 부호 검사 ($\theta = 0.55$):
$$
\frac{4}{0.55} - \frac{1}{0.45} - \frac{3}{0.35} \approx 7.273 - 2.222 - 8.571 \approx -3.520 < 0
$$
우측 구간 도함수 음수 → $\log q$ 단조 감소 → 우측 구간 최댓값 후보는 좌측 경계 $\theta = 0.5$.

#### Step 3. 좌측 구간 $\theta\in[1/3, 0.5]$

Posterior:
$$
q(\theta) = \theta^4(1-\theta)(36\theta - 12)
$$

로그:
$$
\log q(\theta) = 4\log\theta + \log(1-\theta) + \log(36\theta - 12)
$$

미분:
$$
\frac{d}{d\theta}\log q = \frac{4}{\theta} - \frac{1}{1-\theta} + \frac{36}{36\theta - 12} = \frac{4}{\theta} - \frac{1}{1-\theta} + \frac{3}{3\theta - 1}
$$

1계 조건:
$$
\frac{4}{\theta} - \frac{1}{1-\theta} + \frac{3}{3\theta - 1} = 0
$$

공통분모 $\theta(1-\theta)(3\theta - 1)$ 곱:
$$
4(1-\theta)(3\theta-1) - \theta(3\theta-1) + 3\theta(1-\theta) = 0
$$

전개:
- $(1-\theta)(3\theta-1) = 3\theta - 1 - 3\theta^2 + \theta = -3\theta^2 + 4\theta - 1$
  → $4(\cdot) = -12\theta^2 + 16\theta - 4$.
- $-\theta(3\theta-1) = -3\theta^2 + \theta$.
- $3\theta(1-\theta) = 3\theta - 3\theta^2$.

합산:
- $\theta^2$ 항: $-12 - 3 - 3 = -18$.
- $\theta^1$ 항: $16 + 1 + 3 = 20$.
- $\theta^0$ 항: $-4$.

$$
-18\theta^2 + 20\theta - 4 = 0 \iff 9\theta^2 - 10\theta + 2 = 0
$$

근의 공식:
$$
\theta = \frac{10 \pm \sqrt{100 - 72}}{18} = \frac{10 \pm \sqrt{28}}{18} = \frac{5 \pm \sqrt{7}}{9}
$$

수치 ($\sqrt{7} \approx 2.646$):
- $\theta_1 = \frac{5 + 2.646}{9} \approx 0.8495$.
- $\theta_2 = \frac{5 - 2.646}{9} \approx 0.2616$.

구간 $[1/3, 0.5] \approx [0.333, 0.5]$ 검증:
- $\theta_1 \approx 0.8495 \notin$ 구간 → 무효.
- $\theta_2 \approx 0.2616 \notin$ 구간 → 무효.

좌측 구간에 임계점 없음 → 부호 검사 ($\theta = 0.4$):
$$
\frac{4}{0.4} - \frac{1}{0.6} + \frac{3}{0.2} = 10 - 1.667 + 15 = 23.333 > 0
$$
좌측 구간 도함수 양수 → $\log q$ 단조 증가 → 좌측 구간 최댓값 후보는 우측 경계 $\theta = 0.5$.

#### Step 4. 후보값 비교

좌·우 구간 모두 후보가 $\theta = 0.5$. 받침 양 끝에서:
- $p_6(1/3) = 36\cdot 1/3 - 12 = 0$ → $q = 0$.
- $p_6(2/3) = 24 - 36\cdot 2/3 = 0$ → $q = 0$.

받침 끝점은 후보 아님. 유일 후보 $\theta = 0.5$:
$$
q(0.5) = (0.5)^4 \cdot 0.5 \cdot 6 = \frac{1}{16}\cdot\frac{1}{2}\cdot 6 = \frac{3}{16} = 0.1875
$$

수치 검증 ($\theta = 0.6$):
$$
q(0.6) = (0.6)^4\cdot 0.4\cdot(24 - 21.6) = 0.1296\cdot 0.4\cdot 2.4 \approx 0.1244 < 0.1875
$$

#### Step 5. 결론

$$
\boxed{\;m = 6:\;\; \theta^*_{\mathrm{MAP}} = \tfrac{1}{2}\;}
$$

**근거 정리**: 우측 임계점 방정식 $18\theta^2 - 25\theta + 8 = 0$ 의 두 근 $\{8/9, 1/2\}$ 중 받침 내 유효해는 $1/2$ 뿐이며, 좌측 임계점 방정식 $9\theta^2 - 10\theta + 2 = 0$ 의 두 근 $\frac{5\pm\sqrt 7}{9}$ 모두 받침 밖이다. 양 구간 모두 단조 증가/감소이므로 양 구간이 만나는 꼭짓점 $\theta = 1/2$ 가 유일 전역 최댓값.

---

## 5. 검증

### 5.1 단위 검증

$\theta\in[0,1]$ 무차원, $p_m, L, q$ 모두 무차원. 일관성 OK.

### 5.2 받침 검증

- $m = 2$: $\theta = 2/3 \in [0, 1]$. ✓
- $m = 6$: $\theta = 1/2 \in [1/3, 2/3]$. ✓

### 5.3 prior 영향의 정성적 일관성

- MLE (no prior): $\hat\theta_{\mathrm{MLE}} = k/n = 4/5 = 0.8$.
- $m = 2$ 의 prior 는 $[0,1]$ 전 구간에 걸친 약한 prior (꼭짓점 $\theta = 0.5$). MAP $= 2/3 \approx 0.667$ 는 0.8 과 0.5 사이 — prior 가 약하게 끌어당긴 결과. ✓
- $m = 6$ 의 prior 는 $[1/3, 2/3]$ 의 좁은 받침에 강하게 집중. MAP $= 0.5$ 는 prior 의 꼭짓점에 끌려간 결과. ✓
- $m$ 이 클수록 prior 가 강해져 MAP 가 prior 의 mode (0.5) 로 끌려간다. ✓

### 5.4 함숫값 sanity

- $m=2$: $q(2/3) > q(0.5)$ — likelihood 가 우세.
- $m=6$: $q(0.5) > q(0.6)$ — prior 가 우세.

---

## 6. 일반화·통찰

### 6.1 5단 표준 절차

piecewise 분포 하의 MAP / MLE 풀이의 일반 패턴.
1. piecewise 분리.
2. 구간별 posterior.
3. 1계 조건.
4. 임계점의 받침 내 유효성.
5. 후보값 (임계점 / 경계 / 꼭짓점 / 받침 끝점) 비교.

### 6.2 MAP 의 prior 강도 의존

$m \uparrow$ → prior 받침 좁아지고 prior 값이 커짐 → posterior 가 prior 의 mode 에 끌려감. $m \to \infty$ 극한: prior 가 $\theta = 0.5$ 의 점질량 → MAP $\to 0.5$. $m \to 1$ 극한: prior 가 $[0,1]$ 의 약한 분포 → MAP $\to$ MLE.

### 6.3 likelihood-prior 균형의 정량화

log-posterior:
$$
\log q(\theta) = k\log\theta + (n-k)\log(1-\theta) + \log p_m(\theta)
$$
1계 조건:
$$
\frac{k}{\theta} - \frac{n-k}{1-\theta} + \frac{p_m'(\theta)}{p_m(\theta)} = 0
$$
prior 의 score $\frac{p_m'}{p_m}$ 가 likelihood score 와 어떻게 균형을 이루는가가 MAP 의 위치를 결정.

### 6.4 piecewise prior 의 일반론

본 풀이는 prior 가 삼각형 / 사다리꼴 / 다른 piecewise linear 든 동일하게 적용. 핵심은 절댓값 → piecewise 일차식 분리, 그 후 각 piece 가 매끈한 함수가 되는 것.

---

## 7. 시험 출제 변형 5가지

1. **다른 $m$ 값**: $m = 4$ 일 때 MAP 를 구하라 (받침 $[1/4, 3/4]$).
2. **다른 관측**: $n = 10, k = 7$ 로 같은 prior 하 MAP.
3. **다른 piecewise prior**: 사다리꼴 prior $p(\theta) = c$ on $[a, b]$, linear ramp on edges. MAP 를 구하라.
4. **MAP vs MLE 비교**: 같은 데이터에 대해 MLE 와 MAP 의 차이를 prior 강도 $m$ 의 함수로 분석.
5. **Posterior 정규화**: $P(\mathcal{D}) = \int_0^1 L(\theta)p_m(\theta)\,d\theta$ 를 계산하라 ($m=2$).

---

## 8. 백지 재현 체크리스트

- [ ] Posterior $\propto$ likelihood × prior 를 명시.
- [ ] $L(\theta) \propto \theta^4(1-\theta)$ 를 쓴다.
- [ ] 절댓값을 piecewise (좌·우 구간) 로 분리한 prior 의 일차식을 명시.
- [ ] $m = 2$: 받침 $[0, 1]$, 좌측 $4\theta$, 우측 $4(1-\theta)$.
- [ ] $m = 6$: 받침 $[1/3, 2/3]$, 좌측 $36\theta - 12$, 우측 $24 - 36\theta$.
- [ ] 각 구간 $\log q$ 의 미분식.
- [ ] $m = 2$ 우측: $4(1-\theta) = 2\theta$ → $\theta = 2/3$.
- [ ] $m = 2$ 좌측: 임계점 $5/6$ 받침 밖 → 단조 증가 → 후보 $\theta = 0.5$.
- [ ] $m = 2$ 함숫값 비교: $64/729 > 1/16$ → MAP $= 2/3$.
- [ ] $m = 6$ 우측: $18\theta^2 - 25\theta + 8 = 0$ → 근 $\{8/9, 1/2\}$, 8/9 받침 밖.
- [ ] $m = 6$ 좌측: $9\theta^2 - 10\theta + 2 = 0$ → 두 근 모두 받침 밖.
- [ ] $m = 6$ 양 구간 단조성 → 꼭짓점 $\theta = 1/2$.
- [ ] $m = 6$ MAP $= 1/2$ 로 결론.

---

## 9. 핵심 공식 카드

```
Posterior:
  q(θ) ∝ L(θ) · p_m(θ) = θ^k (1-θ)^(n-k) · p_m(θ)

Triangular prior:
  p_m(θ) = m - m² |θ - 0.5|,  support |θ - 0.5| ≤ 1/m

Piecewise:
  좌측 (θ ≤ 0.5): p_m(θ) = m²θ - m²·0.5 + m
  우측 (θ ≥ 0.5): p_m(θ) = m²·0.5 + m - m²θ

m = 2 답:  θ* = 2/3       (likelihood 우세)
m = 6 답:  θ* = 1/2       (prior 우세, 꼭짓점)

5단 절차:
  1. piecewise 분리
  2. 구간별 q(θ)
  3. d/dθ log q = 0
  4. 받침 내 유효성
  5. 후보 비교
```

---

## 10. 다른 퀴즈와의 연결

- **Quiz 5/6/7 (Bernoulli MLE/MAP)**: 같은 likelihood 형태. Q5 는 prior 없는 MLE, Q6/7 은 Beta prior MAP, Q9 는 triangular prior MAP. prior 형태만 바꾼 것.
- **Quiz 6주차 (Uniform Prior)**: uniform prior 하 MAP = MLE. Q9 는 prior 가 강해질수록 MAP 가 prior mode 로 끌려가는 일반 현상의 정량적 사례.
- **Quiz 8주차 (Newton Method)**: 본 문제의 임계점 방정식 $18\theta^2 - 25\theta + 8 = 0$ 같은 비선형 방정식을 일반적으로 풀 때 Newton 법이 사용된다.
- **Quiz 8 (Gaussian KL)**: 같은 "사후/likelihood 최대화" 프레임. Q8 은 Gaussian, Q9 는 Bernoulli. 둘 다 NLL 최소화 = MLE/MAP 구조.
- **공통 메시지**: 베이지안 추론의 핵심은 likelihood-prior 의 곱을 최대화하는 표준 절차. piecewise prior 는 절댓값 분리만 추가될 뿐 같은 알고리즘.

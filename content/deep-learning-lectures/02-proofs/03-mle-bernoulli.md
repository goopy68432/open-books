---
title: "02. MLE for Bernoulli — 완전 유도"
slug: mle-bernoulli
order: 3
---

# 02. MLE for Bernoulli — 완전 유도

> **출제 근거**: 3주차 ★10, 퀴즈 10번, 중간고사 Q3(a) 직접 출제
> **시험 출제 방식**: \"Derive the maximum likelihood estimator $\theta^*_{ML}$ for a Bernoulli with $k$ heads in $n$ tosses. Justify each step.\"

---

## 1. 왜 시험에 나오는가

- 가장 단순한 ML 문제이지만, **\"likelihood → log → 미분=0 → 풀이\"의 4단계 표준 패턴**의 원형.
- 이 패턴이 이후 모든 MLE/MAP 문제 (Gaussian mean, Linear Regression, Logistic Regression)의 뼈대.
- 중간고사 출제 → 변형 출제 가능성 매우 높음.

---

## 2. 필요한 사전 수학 (중1 → 대학원)

### 2.1 [중1] 거듭제곱

$\theta^k$ 는 $\theta$ 를 $k$ 번 곱한 것. $\theta^3 = \theta \cdot \theta \cdot \theta$.

### 2.2 [중2] 독립 사건의 확률

두 사건이 **독립**이면 동시에 일어날 확률 = 각자 확률의 곱.

$$
P(A \cap B) = P(A) \cdot P(B) \quad (\text{A, B independent})
$$

### 2.3 [고1] 로그의 성질

$$
\log(ab) = \log a + \log b, \qquad \log(a^k) = k \log a
$$

**왜 로그를 쓰나?** $\theta^k(1-\theta)^{n-k}$ 같은 곱셈을 미분하기 어렵다. log 취하면 덧셈이 되어 미분이 쉽다. 그리고 log는 단조증가 → maximize가 보존된다.

### 2.4 [고2] 미분=0이 극값 조건

미분 가능한 함수 $f(\theta)$ 가 극값(최대/최소)을 가지면 그 점에서 $f'(\theta)=0$.

> ⚠️ **주의**: $f'=0$은 \"필요조건\"이지 \"충분조건\"이 아님. 시험 답안에서는 \"2차 미분 < 0이므로 최댓값\"을 한 줄 적으면 완벽.

### 2.5 [대1] 곱·몫 미분

$$
(uv)' = u'v + uv', \qquad \left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}
$$

### 2.6 [대학원 수준] Bernoulli 분포

확률변수 $X$ 가 \"앞=1, 뒤=0\"으로 두 값만 가지고 $P(X=1)=\theta$:

$$
P(X = x \mid \theta) = \theta^x (1-\theta)^{1-x}, \quad x \in \{0,1\}
$$

**기호 해체:**

| 기호 | 의미 |
|------|------|
| $X$ | 확률변수 (random variable) |
| $x$ | 관측값 (0 또는 1) |
| $\theta$ | 앞면 나올 확률 (parameter, 우리가 추정하고 싶은 것) |
| $\theta^x (1-\theta)^{1-x}$ | $x=1$이면 $\theta$, $x=0$이면 $1-\theta$ — 한 식으로 표현하는 트릭 |

**왜 한 식으로?**
- $x=1$ 대입: $\theta^1(1-\theta)^0 = \theta \cdot 1 = \theta$ ✅
- $x=0$ 대입: $\theta^0(1-\theta)^1 = 1 \cdot (1-\theta) = 1-\theta$ ✅

---

## 3. 문제 설정 (Setup)

동전을 $n$ 번 던져 $k$ 번 앞면이 나왔다. 동전이 앞면 나올 확률 $\theta$ 의 **최대우도 추정값** $\theta^*_{ML}$ 을 구하라.

데이터: $D = \{x_1, x_2, \ldots, x_n\}$, 단 각 $x_i \in \{0,1\}$, 합이 $\sum_i x_i = k$.

**가정 (반드시 답안에 명시)**:
- **IID** (Independent and Identically Distributed): 각 throw가 독립이고 같은 $\theta$ 를 따른다.

---

## 4. 유도 체인 (Step-by-Step, 단계 건너뛰지 않음)

### Step 1 — Likelihood 작성

🟢 Likelihood (= 데이터의 결합확률, $\theta$ 의 함수):

$$
L(\theta) \;=\; P(D \mid \theta) \;\stackrel{\text{IID}}{=}\; \prod_{i=1}^{n} P(x_i \mid \theta) \;=\; \prod_{i=1}^{n} \theta^{x_i}(1-\theta)^{1-x_i}
\tag{1}
$$

**Line-by-Line:**

| 표기 | 의미 | 왜 |
|------|------|-----|
| $L(\theta)$ | $\theta$ 의 함수로 본 데이터 확률 | \"likelihood\" 정의 |
| $P(D\mid\theta)$ | $\theta$ 가 주어졌을 때 D가 나올 확률 | 정의 |
| $\prod_{i=1}^n P(x_i\mid\theta)$ | 각 throw 확률의 곱 | **IID** 가정 → 독립이라 곱 |
| $\theta^{x_i}(1-\theta)^{1-x_i}$ | Bernoulli PMF | 2.6절 정의 |

### Step 2 — 곱 정리

거듭제곱 법칙 $\prod_i \theta^{x_i} = \theta^{\sum_i x_i}$ 사용:

$$
L(\theta) \;=\; \theta^{\sum_i x_i} (1-\theta)^{\sum_i (1-x_i)} \;=\; \theta^{k} (1-\theta)^{n-k}
\tag{2}
$$

**왜?**
- $\sum_i x_i = k$ (앞면 횟수)
- $\sum_i (1-x_i) = n - k$ (뒷면 횟수)

### Step 3 — Log Likelihood (NLL의 뿌리)

$$
\ell(\theta) \;:=\; \log L(\theta) \;=\; k \log \theta + (n-k) \log(1-\theta)
\tag{3}
$$

**Line-by-Line:**

| 변형 | 사용한 공식 | 왜 |
|------|------------|-----|
| $\log(\theta^k(1-\theta)^{n-k})$ | $\log(ab)=\log a+\log b$ | 곱→합 |
| $= \log\theta^k + \log(1-\theta)^{n-k}$ | 위와 동일 | |
| $= k\log\theta + (n-k)\log(1-\theta)$ | $\log a^n = n\log a$ | 지수→계수 |

**왜 log 를 취하나? (3가지 이유 — 답안에 적기)**
1. **곱이 합으로**: 미분이 쉬워짐 (term-by-term).
2. **수치적 안정**: $\theta^k$ 같은 극단적 값의 underflow 방지.
3. **단조증가**: $L$ 최대화 ↔ $\log L$ 최대화 (argmax 보존).

### Step 4 — 미분=0 조건

$\theta$ 에 대해 미분:

$$
\frac{d\ell}{d\theta} \;=\; \frac{k}{\theta} + (n-k)\cdot\frac{-1}{1-\theta} \;=\; \frac{k}{\theta} - \frac{n-k}{1-\theta}
\tag{4}
$$

**Line-by-Line:**

| 항 | 미분 결과 | 왜 |
|------|----------|-----|
| $k\log\theta$ | $k\cdot\frac{1}{\theta}$ | $(\log\theta)'=1/\theta$ |
| $(n-k)\log(1-\theta)$ | $(n-k)\cdot\frac{1}{1-\theta}\cdot(-1)$ | 합성함수 (chain rule), 안쪽 $1-\theta$ 의 미분 = $-1$ |

극값 조건:

$$
\frac{d\ell}{d\theta} = 0 \;\Longrightarrow\; \frac{k}{\theta} = \frac{n-k}{1-\theta}
\tag{5}
$$

### Step 5 — 풀이

식 (5) 의 양변에 $\theta(1-\theta)$ 를 곱:

$$
k(1-\theta) \;=\; (n-k)\theta
$$

전개:

$$
k - k\theta \;=\; n\theta - k\theta \;\Longrightarrow\; k \;=\; n\theta
$$

(양변에서 $-k\theta$ 가 사라짐)

따라서:

$$
\boxed{\; \theta^*_{ML} \;=\; \frac{k}{n} \;}
\tag{6}
$$

### Step 6 — 최댓값 검증 (감점 방지용)

2차 미분:

$$
\frac{d^2\ell}{d\theta^2} \;=\; -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2} \;<\; 0
$$

(단, $0 < k < n$, $0 < \theta < 1$). **모든 점에서 음수**이므로 $\ell$ 은 **strictly concave**, 따라서 $\theta = k/n$ 은 **유일한 최댓값**.

---

## 5. 의미 해석 (시험 답안 마지막 줄에 반드시)

> $\theta^*_{ML} = k/n$ 은 \"앞면 비율\"이라는 직관적 추정값과 정확히 일치한다.
> 즉, **MLE는 \"데이터에서 본 빈도\"를 그대로 추정값으로 쓰는 가장 자연스러운 방법**이다.
> 단, **prior가 uniform** 이라는 암묵적 가정이 있다 (이것이 MAP과 통합되는 시각 → 03 토픽).

---

## 6. 모범 답안 템플릿 (영어 출제 대비)

```
[Setup]
Let X_1, ..., X_n be IID Bernoulli(θ) random variables, with
observed values summing to k = Σ x_i (number of heads).
Goal: find θ* maximizing the likelihood L(θ) = P(D | θ).

[Step 1 — Likelihood, using IID]
L(θ) = Π_{i=1}^n θ^{x_i} (1-θ)^{1-x_i}            (IID factorization)
     = θ^k (1-θ)^{n-k}                              (Σ x_i = k)

[Step 2 — Log-likelihood (monotone, easier to differentiate)]
ℓ(θ) = log L(θ) = k log θ + (n-k) log(1-θ)

[Step 3 — Necessary condition dℓ/dθ = 0]
dℓ/dθ = k/θ - (n-k)/(1-θ) = 0
  ⇒ k(1-θ) = (n-k)θ
  ⇒ k = nθ
  ⇒ θ* = k/n.

[Step 4 — Sufficiency]
d²ℓ/dθ² = -k/θ² - (n-k)/(1-θ)² < 0 for 0<θ<1, 0<k<n,
so ℓ is strictly concave and θ* = k/n is the unique maximizer.

[Conclusion]
θ*_ML = k/n. This equals the empirical frequency of heads,
which is intuitive: MLE picks the parameter under which the
observed data is most probable, and for Bernoulli IID this
coincides with the sample mean.
```

---

## 7. 자주 틀리는 함정

1. **IID 명시 안 함** → likelihood 의 곱 표현이 정당화 안 됨. **반드시 한 줄**: \"by IID assumption\".
2. **log 왜 취하는지 설명 누락** → \"답만 적은 것\" 으로 감점.
3. **2차 미분 검증 생략** → 극값이 최댓값이라는 보장 없음.
4. **boundary case** $k=0$ 또는 $k=n$: $\log 0$ 발산 — strictly 한 conditioning 필요. 답안에서 \"$0<k<n$\" 명시.
5. **NLL 부호 헷갈림**: $\arg\max L = \arg\max \log L = \arg\min (-\log L)$. 부호 자유롭게.

---

## 8. 연결 개념

- → [03 MAP 일반화](03_MAP_일반화_유도.md): prior 추가하면 어떻게 변하는가 ($k/n$ → $(k+M)/(n+2M)$)
- → [04 NLL→MSE](04_NLL_MSE_Gaussian_유도.md): 같은 \"likelihood → log → 최적화\" 패턴이 Gaussian 에서 MSE 로
- → [09 Backprop](09_Backprop_ChainRule.md): 미분=0을 못 풀 때 (NN) gradient descent 로 대체

---
title: "04. NLL → MSE 유도 (Gaussian Likelihood) — 완전 재현"
slug: nll-mse-gaussian
order: 5
---

# 04. NLL → MSE 유도 (Gaussian Likelihood) — 완전 재현

> **출제 근거**: 4주차 ★10, 6-7주차 반복, \"교수가 외우지 말고 유도하라 강조\"
> **시험 출제 방식**: \"Show that, under a Gaussian likelihood with fixed variance, NLL minimization is equivalent to MSE minimization. Explain each step.\"

---

## 1. 왜 시험에 나오는가

- \"왜 회귀에서 MSE를 쓰는가?\" 의 **유일한 정답**: \"likelihood가 Gaussian이라고 가정했기 때문\".
- 강의에서 **exp / minus / square** 라는 이름으로 강조되는 흐름.
- NLL ↔ MSE 동치는 **확률모델 ↔ 손실함수** 매핑의 원형 (Bernoulli ↔ CE 와 쌍).

---

## 2. 사전 수학 (중1 → 대학원)

### 2.1 [중1] 제곱

$x^2 = x \cdot x$. 항상 $\geq 0$.

### 2.2 [고1] 지수함수와 로그의 역관계

$$
\log(e^x) = x, \qquad \log(\exp(-x^2)) = -x^2
$$

→ \"exp 안의 식\"이 log 를 통과하면 그대로 끌려나옴. 이것이 **Gaussian 의 핵심 트릭**.

### 2.3 [대1] Gaussian (Normal) 분포 PDF

$$
\mathcal{N}(y \mid \mu, \sigma^2) \;=\; \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(y - \mu)^2}{2\sigma^2} \right)
\tag{*}
$$

**기호 해체:**

| 기호 | 의미 |
|------|------|
| $y$ | 관측 확률변수 |
| $\mu$ | 평균 (mean) — 분포의 중심 |
| $\sigma^2$ | 분산 (variance) — 퍼진 정도 |
| $\frac{1}{\sqrt{2\pi\sigma^2}}$ | 정규화 상수 (적분 = 1 보장) |
| $\exp\big(-(y-\mu)^2/(2\sigma^2)\big)$ | Bell-shape 본체 |

**왜 \"exp / minus / square\" 가 핵심인가?**
- **square**: $(y-\mu)^2$ → 평균에서 멀수록 페널티 ↑
- **minus**: $-(y-\mu)^2$ → 멀수록 \"확률 낮음\"으로 변환
- **exp**: 양수화 + 미분 좋은 함수
- log를 취하면 exp가 사라지면서 **square 만 남음** → MSE!

### 2.4 ERM (Empirical Risk Minimization)

$$
\arg\min_h \; \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)
$$

Loss 함수의 데이터 평균 (= empirical risk) 을 최소화. NLL 도 이 형태로 변형 가능 (이게 핵심!).

---

## 3. 문제 설정

데이터: $D = \{(x_i, y_i)\}_{i=1}^n$. $x_i$ 는 입력, $y_i$ 는 실수 출력.

🟢 **Likelihood 가정 (Gaussian)**:

$$
P(y_i \mid x_i, h) \;=\; \mathcal{N}(y_i \mid h(x_i), \sigma^2)
$$

\"hypothesis $h$ 가 예측한 $h(x_i)$ 가 평균이고, 노이즈는 분산 $\sigma^2$ 의 Gaussian\".

**가정**: IID, $\sigma^2$ 는 **고정**된 상수 ($h$ 와 무관).

---

## 4. 유도 체인

### Step 1 — Joint Likelihood (IID)

$$
P(D \mid h) \;\stackrel{\text{IID}}{=}\; \prod_{i=1}^n P(y_i \mid x_i, h) \;=\; \prod_{i=1}^n \mathcal{N}(y_i \mid h(x_i), \sigma^2)
\tag{1}
$$

### Step 2 — Gaussian PDF 대입

$$
P(D \mid h) \;=\; \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(y_i - h(x_i))^2}{2\sigma^2} \right)
\tag{2}
$$

### Step 3 — NLL 정의 (Negative Log Likelihood)

$$
\text{NLL}(h) \;:=\; -\log P(D \mid h)
\tag{3}
$$

**왜 \"negative\"?** maximize $P$ = minimize $-\log P$. Optimization은 minimize 형태가 표준.

### Step 4 — 곱 → 합 + 지수 → 계수

$$
-\log P(D \mid h) \;=\; -\log \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right)
$$

곱의 로그 = 로그의 합:

$$
= -\sum_{i=1}^n \log\!\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right)\right]
$$

곱의 로그 안쪽 다시 분리:

$$
= -\sum_{i=1}^n \left[ \log \frac{1}{\sqrt{2\pi\sigma^2}} + \log \exp\!\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right) \right]
$$

$\log \exp(z) = z$ 적용 (이게 \"exp\" 사라짐):

$$
= -\sum_{i=1}^n \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - h(x_i))^2}{2\sigma^2} \right]
$$

부호 분배 ($-$ 를 안쪽에):

$$
= \sum_{i=1}^n \left[ \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y_i - h(x_i))^2}{2\sigma^2} \right]
\tag{4}
$$

### Step 5 — 상수항 분리

$$
\text{NLL}(h) \;=\; \underbrace{\frac{n}{2}\log(2\pi\sigma^2)}_{\text{constant in } h} \;+\; \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - h(x_i))^2
\tag{5}
$$

**관찰**:
- 첫 항: $h$ 와 무관 ($\sigma^2$ 는 고정 가정) → minimize 시 무시 가능
- 둘째 항: $\frac{1}{2\sigma^2}$ 는 양의 상수, **square 만 남음**

### Step 6 — Argmin 동치

$$
\arg\min_h \text{NLL}(h) \;=\; \arg\min_h \; \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - h(x_i))^2 \;=\; \arg\min_h \sum_{i=1}^n (y_i - h(x_i))^2
$$

**왜 마지막 단계?** $\frac{1}{2\sigma^2}$ 는 양의 상수 → argmin 보존. 또한 $\frac{1}{n}$ 을 곱해도 argmin 보존:

$$
\boxed{\;
\arg\min_h \text{NLL}(h) \;=\; \arg\min_h \; \underbrace{\frac{1}{n}\sum_{i=1}^n (y_i - h(x_i))^2}_{\text{MSE}}
\;}
\tag{6}
$$

---

## 5. \"exp / minus / square\" 흐름 한 줄 요약

> **Gaussian 의 exp 가 log 에 의해 사라지고, minus 는 NLL 의 minus 와 만나 부호 정리되고, square 만 남는다.**

이 한 문장이 NLL ↔ MSE 의 본질. 시험에서 결론 직전에 적으면 점수.

---

## 6. ERM 형태로의 통합

$\ell(h(x_i), y_i) := (y_i - h(x_i))^2$ 라 정의하면:

$$
\arg\min_h \text{NLL}(h) \;=\; \arg\min_h \; \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)
$$

이것이 **NLL = ERM** (7주차 ★10 통합 시각). 즉:

> **확률모델 (likelihood) 가정 → NLL → ERM 의 어떤 loss 가 결정됨.**

| Likelihood | Loss (ERM 형태) |
|-----------|----------------|
| Gaussian | MSE = $\frac{1}{n}\sum (y - h(x))^2$ |
| Bernoulli (binary) | Binary Cross-Entropy |
| Categorical (multiclass) | Cross-Entropy |
| Laplace | MAE = $\frac{1}{n}\sum |y - h(x)|$ (참고) |

→ 다음 토픽 [05 Cross-Entropy](05_CrossEntropy_Categorical_유도.md) 에서 같은 흐름 재현.

---

## 7. 모범 답안 템플릿

```
[Setup]
Data D = {(x_i, y_i)}_{i=1..n}, IID.
Assume y_i | x_i, h ~ N(h(x_i), σ²) with σ² fixed.
Goal: show argmin_h NLL(h) = argmin_h MSE(h).

[Step 1 — IID likelihood]
P(D | h) = Π_i N(y_i | h(x_i), σ²)
        = Π_i (2π σ²)^{-1/2} exp(-(y_i - h(x_i))² / (2σ²))

[Step 2 — NLL = -log P(D | h)]
Take -log of the product → sum of -log of each factor.
log of exp cancels (this is the "exp" disappearing):

NLL(h) = (n/2) log(2π σ²) + (1/(2σ²)) Σ_i (y_i - h(x_i))²

[Step 3 — Drop terms constant in h]
First term independent of h; the prefactor 1/(2σ²) > 0
preserves argmin. Therefore:

argmin_h NLL(h) = argmin_h Σ_i (y_i - h(x_i))²
                = argmin_h (1/n) Σ_i (y_i - h(x_i))²
                = argmin_h MSE(h).                       ∎

[Interpretation]
Choosing MSE as loss is *equivalent to* assuming
Gaussian noise with fixed variance on the targets.
This is the lecture's "exp / minus / square" slogan:
the Gaussian's exp is killed by log, the minus aligns
with NLL's minus, and only the square remains.
This is why MSE is the canonical regression loss.
```

---

## 8. 자주 틀리는 함정

1. **$\sigma^2$ 가 $h$에 의존하지 않는다는 가정 누락**: 만약 $\sigma^2$도 추정 대상이면 첫 항이 살아남아 \"variance estimation\" 추가. 답안에 한 줄: \"σ² fixed (independent of h)\".
2. **IID 명시 누락**: 곱 표현의 정당화 필요.
3. **로그 분리 단계 건너뛰기**: \"$\log \prod = \sum \log$\" 와 \"$\log \exp(z) = z$\" 두 단계는 반드시 명시.
4. **\"argmin 보존\" 한 줄 누락**: $\frac{1}{2\sigma^2}$ 곱하기/$\frac{1}{n}$ 곱하기는 argmin을 보존 — 한 줄로 명시.
5. **부호 실수**: NLL의 $-$ 와 exp 안의 $-$ 부호가 만나서 **+** 가 되는 점이 핵심.

---

## 9. 연결 개념

- ← [01 베이즈](01_베이즈정리_증명.md), [02 MLE Bernoulli](02_MLE_Bernoulli_완전유도.md): 같은 \"likelihood → log → 최적화\" 패턴
- → [05 Cross-Entropy](05_CrossEntropy_Categorical_유도.md): Categorical 버전
- → [07 Linear Regression Closed Form](07_LinearReg_ClosedForm.md): MSE를 실제로 최소화하면 어떻게 되는가
- → [13 KL Divergence](13_KL_Divergence.md): NLL ↔ KL 의 관계

---
title: "03. Bias-Variance Tradeoff"
slug: bias-variance
order: 3
---

# 03. Bias-Variance Tradeoff

> page_435 부근(double descent) 등장. **분해 증명**이 시험 단골.

---

## 1. 설정

진짜 함수 $f(x)$, 관측 $y = f(x) + \epsilon$, $\epsilon \sim N(0, \sigma^2)$.

데이터셋 D로부터 학습된 모델 $\hat{f}_D(x)$.

테스트점 $x_0$의 **예측 오차** (제곱 손실):
$$\text{Err}(x_0) = E_{D, \epsilon}[(y_0 - \hat{f}_D(x_0))^2]$$

---

## 2. 정리 (분해 공식)

$$\text{Err}(x_0) = \underbrace{(E_D[\hat{f}_D(x_0)] - f(x_0))^2}_{\text{Bias}^2} + \underbrace{E_D[(\hat{f}_D(x_0) - E_D[\hat{f}_D(x_0)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise (irreducible)}}$$

---

## 3. 증명 (시험 단골)

$\bar{f}(x_0) := E_D[\hat{f}_D(x_0)]$로 표기.

$$E[(y_0 - \hat{f}_D(x_0))^2]$$

**1단계:** $y_0 = f(x_0) + \epsilon$, $\hat{f}_D$를 분리. $f(x_0) = f$로 약식.

$$= E\left[\left((f + \epsilon) - \hat{f}_D\right)^2\right]$$

**2단계:** $\bar{f}$를 더하고 빼기:
$$= E\left[\left((f - \bar{f}) + (\bar{f} - \hat{f}_D) + \epsilon\right)^2\right]$$

**3단계:** 제곱 전개. 교차항 3개:

$$= E\left[(f - \bar{f})^2\right] + E\left[(\bar{f} - \hat{f}_D)^2\right] + E[\epsilon^2]$$
$$+ 2E[(f - \bar{f})(\bar{f} - \hat{f}_D)] + 2E[(f - \bar{f})\epsilon] + 2E[(\bar{f} - \hat{f}_D)\epsilon]$$

**4단계:** 교차항 처리.

(i) $f, \bar{f}$는 D 무관, ε와 독립:
$$E[(f-\bar{f})\epsilon] = (f-\bar{f}) E[\epsilon] = 0.$$

(ii) $\hat{f}_D$는 D에 의존, ε는 새로운 잡음 (독립):
$$E[(\bar{f}-\hat{f}_D)\epsilon] = E[\bar{f}-\hat{f}_D] \cdot E[\epsilon] = 0.$$

(iii) $f - \bar{f}$는 D 무관 상수:
$$E[(f-\bar{f})(\bar{f} - \hat{f}_D)] = (f-\bar{f})(E[\bar{f}] - E[\hat{f}_D]) = (f-\bar{f})(\bar{f} - \bar{f}) = 0.$$

**5단계:** 남은 항.

$$= (f - \bar{f})^2 + E[(\bar{f} - \hat{f}_D)^2] + \sigma^2$$

$$= \underbrace{(\bar{f}(x_0) - f(x_0))^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\hat{f}_D(x_0))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

∎

---

## 4. 각 항의 의미

| 항 | 의미 | 줄이는 방법 |
|---|------|-----------|
| **Bias** | 평균 예측이 진짜에서 얼마나 멀리 있나 | 모델 복잡도 ↑ |
| **Variance** | 데이터 바뀔 때 예측이 얼마나 흔들리나 | 데이터 ↑, 정규화, 모델 단순화 |
| **Noise** $\sigma^2$ | 본질적 잡음 (제거 불가) | (불가) |

---

## 5. Tradeoff

```
   오차
    │   ╲                          ╱
    │     ╲ Bias                ╱ Variance
    │       ╲                ╱
    │         ╲            ╱
    │           ╲       ╱
    │             ╲   ╱     ← 최적점
    │              ╳
    │            ╱   ╲
    │         ╱        ╲___ Total
    │      ╱
    └─────────────────────── 모델 복잡도
```

- **단순 모델:** Bias 큼, Variance 작음 (과소적합)
- **복잡 모델:** Bias 작음, Variance 큼 (과적합)
- **최적:** 둘의 합이 최소

---

## 6. Double Descent (page_435)

전통적 그래프(위)와 달리, **현대 딥러닝**에서:

```
   오차
    │\         ╱╲
    │ \      ╱   ╲
    │  \   ╱     ╲___ (over-parameterized)
    │   \_/
    │  ↑
    │  (전통적 최적점)
    └─────────────── 파라미터 수
       under   inter   over
```

- Under-parameterized: 전통적 U 모양
- Interpolation point ($n \approx p$): **분산 폭발**
- Over-parameterized: **다시 감소** (현대 딥러닝)

**해석:** 매우 큰 모델은 noise를 무시하고 매끈한 해를 찾음 (implicit regularization).

---

## 7. 시험 답안 (분해 증명)

### [문제] $E[(y - \hat{f})^2] = \text{Bias}^2 + \text{Var} + \sigma^2$임을 증명하라.

### [풀이]

$\bar{f} = E_D[\hat{f}]$로 두고 $y - \hat{f} = (f - \bar{f}) + (\bar{f} - \hat{f}) + \epsilon$로 분해.

제곱 후 기댓값:
$$E[(y-\hat{f})^2] = E[(f-\bar{f})^2] + E[(\bar{f}-\hat{f})^2] + E[\epsilon^2] + (\text{cross terms})$$

교차항 3개 모두 0:
- $f, \bar{f}$는 결정값이고 ε와 독립 → 0
- $E[\bar{f} - \hat{f}] = 0$ → 다른 교차항 0

남은 3항:
$$= \underbrace{(\bar{f} - f)^2}_{\text{Bias}^2} + \underbrace{E[(\bar{f}-\hat{f})^2]}_{\text{Var}} + \underbrace{\sigma^2}_{\text{Noise}}. \quad \blacksquare$$

---

## 8. 한 줄 요약

> "예측 오차 = Bias² + Variance + 잡음. 모델 복잡도 ↑이면 Bias ↓, Var ↑. Tradeoff 균형이 일반화의 핵심."

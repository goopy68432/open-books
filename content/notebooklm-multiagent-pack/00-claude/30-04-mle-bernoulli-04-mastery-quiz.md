---
title: "04. 마스터리 퀴즈 — Bernoulli MLE"
slug: 04-mle-bernoulli-04-mastery-quiz
order: 30
---

# 04. 마스터리 퀴즈 — Bernoulli MLE

---

## 문제 1: 정규분포 MLE — μ

$X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ i.i.d, $\sigma^2$는 알고 있음. $\hat{\mu}_{\text{MLE}}$를 구하라.

<details><summary>풀이 (8단계)</summary>

**(1)** 모델: $X_i \sim N(\mu, \sigma^2)$ i.i.d.

**(2)** pdf: $p(x|\mu) = \frac{1}{\sqrt{2\pi}\sigma}\exp(-(x-\mu)^2/(2\sigma^2))$

**(3)** i.i.d → 곱:
$$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

**(4)** log:
$$\ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \sum_i \frac{(x_i-\mu)^2}{2\sigma^2}$$

**(5)** μ에 대한 미분 (μ만 변수, σ² 고정):
$$\frac{d\ell}{d\mu} = \sum_i \frac{x_i - \mu}{\sigma^2} = \frac{1}{\sigma^2}\left(\sum x_i - n\mu\right)$$

**(6)** = 0:
$$\sum x_i = n\mu \Rightarrow \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_i x_i = \bar{x}$$

**해석:** 표본 평균이 MLE. (베르누이의 k/n과 같은 패턴!)
</details>

---

## 문제 2: 지수분포 MLE — λ

$X_i \sim$ Exp(λ) i.i.d, pdf $p(x|\lambda) = \lambda e^{-\lambda x}$, $x > 0$.

<details><summary>풀이</summary>

$$L(\lambda) = \prod \lambda e^{-\lambda x_i} = \lambda^n \exp\left(-\lambda \sum x_i\right)$$

$$\ell = n\log\lambda - \lambda \sum x_i$$

$$\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum x_i = 0 \Rightarrow \hat{\lambda} = \frac{n}{\sum x_i} = \frac{1}{\bar{x}}$$
</details>

---

## 문제 3: 베르누이 MLE의 직관

n=10, 동전을 10번 던졌더니 7번 앞면이었다. $\hat{\theta}_{\text{MLE}}$는?

<details><summary>풀이</summary>

$\hat{\theta} = k/n = 7/10 = 0.7$.

**해석:** "관측된 비율이 가장 그럴듯한 θ".
</details>

---

## 문제 4: NLL 손실 함수의 정체

베르누이 NLL이 머신러닝에서 어떤 손실 함수와 같은가?

<details><summary>풀이</summary>

**Cross-Entropy 손실** (이진 분류):
$$\text{BCE} = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

베르누이 NLL과 정확히 같은 형태. 자세한 유도는 [`../09-killer-chains/05-bernoulli-to-ce.md`](../09-killer-chains/05-bernoulli-to-ce.md).
</details>

---

## 문제 5: i.i.d 가정 깨질 때

만약 $y_i$들이 독립이 아니면 어떻게 되나?

<details><summary>풀이</summary>

결합확률이 곱으로 분해되지 않음. 일반적으로:
$$p(y_1, \ldots, y_n | \theta) \neq \prod p(y_i | \theta)$$

→ MLE 풀이가 훨씬 복잡 (조건부확률 체인, HMM 등 필요).

**시험 답안에:** "독립 가정이 핵심이며, 이게 없으면 곱셈 분해가 불가능하다."
</details>

---

## 자가 평가
| 점수 | 평가 |
|-----|-----|
| 5/5 | 마스터 — 사실상 시험 만점 가능 |
| 4/5 | 합격 |
| ≤ 3 | `02-derivation.md` 8단계 다시 |

---

## 시험 직전 체크 (★★★)

- [ ] 8단계 체인을 노트 없이 처음부터 끝까지 적을 수 있는가?
- [ ] **"왜 i.i.d면 곱?"** 한 줄 설명 가능?
- [ ] **"왜 로그?"** 3가지 이유 즉답?
- [ ] **"왜 미분=0?"** 페르마 정리 인용?
- [ ] 2계 미분 음수 → 오목 → 최댓값 검증?
- [ ] $\hat{\theta} = k/n$ = "표본 비율" 직관 한 줄?

위 6가지 모두 ✓ 가능하면 시험 만점 준비 완료.

---

[`../05-map-symmetric/00-overview.md`](../05-map-symmetric/00-overview.md)

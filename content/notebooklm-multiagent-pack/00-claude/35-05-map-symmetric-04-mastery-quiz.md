---
title: "04. 마스터리 퀴즈 — MAP 대칭 prior"
slug: 05-map-symmetric-04-mastery-quiz
order: 35
---

# 04. 마스터리 퀴즈 — MAP 대칭 prior

---

## 문제 1: prior $p(\theta) \propto \theta^a(1-\theta)^b$ (Beta(a+1, b+1))

일반화: $\hat{\theta}_{\text{MAP}}$ = ?

<details><summary>풀이</summary>

posterior ∝ $\theta^{k+a}(1-\theta)^{n-k+b}$.

미분=0: $(k+a)/\theta = (n-k+b)/(1-\theta)$

$$\hat{\theta}_{\text{MAP}} = \frac{k+a}{n+a+b}$$
</details>

---

## 문제 2: m=2, n=10, k=7일 때 MAP

<details><summary>풀이</summary>

$\hat{\theta} = (7+2)/(10+4) = 9/14 ≈ 0.643$.

(MLE는 0.7, prior 영향으로 0.5 쪽 끌어당겨짐)
</details>

---

## 문제 3: prior가 균일 (m=0)

<details><summary>풀이</summary>

$\hat{\theta}_{\text{MAP}} = k/n$ = MLE.

**의미:** 균일 prior는 정보 없음 → MAP과 MLE 동일.
</details>

---

## 문제 4: prior 정확값 (Beta normalization)

$p(\theta) = \frac{1}{B(m+1, m+1)}\theta^m(1-\theta)^m$로 정규화하면 MAP은 같은가?

<details><summary>풀이</summary>

같다. $1/B(\cdots)$는 θ 무관 상수 → log에서 상수항 → 미분에서 사라짐.
</details>

---

## 문제 5: regularization 해석

MAP과 머신러닝의 regularization은 어떤 관계?

<details><summary>풀이</summary>

NLL + (-log prior) = "loss + regularizer".

- Gaussian prior on weights → L2 정규화
- Laplace prior → L1 정규화

자세히는 [`../09-killer-chains/06-map-to-l2.md`](../09-killer-chains/06-map-to-l2.md).
</details>

---

## 시험 직전 체크
- [ ] 베이즈 정리 즉답?
- [ ] posterior ∝ likelihood × prior 합쳐서 지수 더하기 가능?
- [ ] 분자·분모 m으로 나누기 트릭 외웠는가?

[`../06-map-asymmetric/00-overview.md`](../06-map-asymmetric/00-overview.md)

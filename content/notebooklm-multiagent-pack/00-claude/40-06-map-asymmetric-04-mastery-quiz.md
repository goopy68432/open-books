---
title: "04. 마스터리 퀴즈 — 비대칭 prior"
slug: 06-map-asymmetric-04-mastery-quiz
order: 40
---

# 04. 마스터리 퀴즈 — 비대칭 prior

## 문제 1: prior $\propto (1-\theta)^m$, $m \to \infty$
<details><summary>풀이</summary>

posterior ∝ $\theta^k(1-\theta)^{n-k+m}$.

$\hat{\theta} = k/(n+m)$. $m \to \infty$이면 0.

**해석:** "θ=0에 강한 사전 믿음" → 데이터 무시 0.
</details>

## 문제 2: m=5, n=10, k=3
<details><summary>풀이</summary>
$\hat{\theta} = (3+5)/(10+5) = 8/15 \approx 0.533$. (MLE 0.3보다 1쪽 끌어당김)
</details>

## 문제 3: prior $\propto \theta^a(1-\theta)^b$ MAP 일반식
<details><summary>풀이</summary>
$\hat{\theta} = (k+a)/(n+a+b)$
</details>

## 문제 4: 5번과 6번의 차이를 한 줄로
<details><summary>풀이</summary>
"5번은 0.5에 대한 대칭적 사전 믿음, 6번은 1에 대한 비대칭 사전 믿음."
</details>

## 문제 5: m → ∞에서 결과가 다른 이유
<details><summary>풀이</summary>
prior의 정점 위치가 달라서. 5번은 0.5에 정점, 6번은 1에 정점. m → ∞이면 정점에 디랙 델타.
</details>

[`../07-map-tent/00-overview.md`](../07-map-tent/00-overview.md)

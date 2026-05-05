---
title: "02. 단계별 유도"
slug: derivation
order: 2
---

# 02. 단계별 유도

## 단계 1: posterior

$$p(\theta|D) \propto p(D|\theta) \cdot p(\theta) = \theta^k(1-\theta)^{n-k} \cdot \theta^m = \theta^{k+m}(1-\theta)^{n-k}$$

## 단계 2: log

$$\log p(\theta|D) = (k+m)\log\theta + (n-k)\log(1-\theta) + C$$

(로그 이유: 곱→합, 단조성, 수치안정)

## 단계 3: 미분 = 0 (페르마)

$$\frac{d}{d\theta}\log p = \frac{k+m}{\theta} - \frac{n-k}{1-\theta} = 0$$

## 단계 4: 풀이

$$(k+m)(1-\theta) = (n-k)\theta$$
$$(k+m) = (n-k+k+m)\theta = (n+m)\theta$$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+m}{n+m}}$$

## 단계 5: m → ∞

분자·분모를 m으로 나누면:
$$\hat{\theta}_{\text{MAP}} = \frac{k/m + 1}{n/m + 1} \xrightarrow{m \to \infty} \frac{0+1}{0+1} = 1$$

$$\boxed{\lim_{m \to \infty} \hat{\theta}_{\text{MAP}} = 1}$$

## 직관

> "prior $\theta^m$은 θ가 클수록 확률이 큼. m → ∞이면 prior가 θ=1에 집중되어 데이터를 완전히 무시하고 1로 수렴."

## 검증 (다른 극한)

- m = 0: $\hat{\theta} = k/n$ = MLE ✓
- m → ∞: 1 ✓

## 비교표 — 5번 vs 6번

| | 5번 ($\theta^m(1-\theta)^m$) | 6번 ($\theta^m$) |
|---|---|---|
| posterior | $\theta^{k+m}(1-\theta)^{n-k+m}$ | $\theta^{k+m}(1-\theta)^{n-k}$ |
| MAP | $(k+m)/(n+2m)$ | $(k+m)/(n+m)$ |
| m → ∞ | 1/2 (대칭) | 1 (비대칭) |
| 직관 | "0.5 확신" | "1 확신" |

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md)

---
title: "03. 완벽 답안 — 비대칭 prior"
slug: 06-map-asymmetric-03-perfect-answer
order: 39
---

# 03. 완벽 답안 — 비대칭 prior

### [문제] prior $p(\theta) \propto \theta^m$일 때 $m \to \infty$의 MAP.

### [풀이]

**(1)** 베이즈 정리: $p(\theta|D) \propto p(D|\theta)p(\theta)$.

**(2)** 결합:
$$p(\theta|D) \propto \theta^k(1-\theta)^{n-k} \cdot \theta^m = \theta^{k+m}(1-\theta)^{n-k}.$$

**(3)** log + 미분=0 (페르마, log 단조성·수치안정 이유):
$$\log p = (k+m)\log\theta + (n-k)\log(1-\theta) + C.$$
$$\frac{d}{d\theta}\log p = \frac{k+m}{\theta} - \frac{n-k}{1-\theta} = 0.$$

**(4)** 풀이:
$$(k+m)(1-\theta) = (n-k)\theta \Rightarrow \hat{\theta}_{\text{MAP}} = \frac{k+m}{n+m}.$$

**(5)** $m \to \infty$:
$$\hat{\theta}_{\text{MAP}} = \frac{k/m + 1}{n/m + 1} \to \frac{1}{1} = 1.$$

**(6)** 직관: prior가 θ=1에 첨예해지므로 m → ∞이면 데이터를 무시하고 1.

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+m}{n+m}, \quad \lim_{m\to\infty} = 1.}$$

---

## 다음

[`../07-map-tent/00-overview.md`](../07-map-tent/00-overview.md)

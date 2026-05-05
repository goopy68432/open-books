---
title: "03. 완벽 답안 — MAP 대칭 prior"
slug: 05-map-symmetric-03-perfect-answer
order: 34
---

# 03. 완벽 답안 — MAP 대칭 prior

### [문제] 기출 4번 상황에서 prior $p(\theta) \propto \theta^m(1-\theta)^m$일 때 $m \to \infty$이면 $\hat{\theta}_{\text{MAP}}$ = ?

### [풀이]

**(1) 베이즈 정리**

$$p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)} \propto p(D|\theta) p(\theta).$$

분모 p(D)는 θ에 무관하므로 MAP에서 무시.

**(2) Likelihood × Prior**

기출 4번에서 $p(D|\theta) = \theta^k(1-\theta)^{n-k}$. 주어진 prior와 결합:
$$p(\theta|D) \propto \theta^{k+m}(1-\theta)^{n-k+m}.$$

**(3) log posterior**

곱이 미분 어렵고 수치적으로 불안정하며, log는 단조이므로 argmax가 보존된다:
$$\log p(\theta|D) = (k+m)\log\theta + (n-k+m)\log(1-\theta) + C.$$

**(4) 미분 = 0 (페르마)**

$\theta \in (0,1)$ 내부 미분 가능. 페르마 정리:
$$\frac{d}{d\theta}\log p = \frac{k+m}{\theta} - \frac{n-k+m}{1-\theta} = 0.$$

**(5) 풀이**

$$(k+m)(1-\theta) = (n-k+m)\theta \Rightarrow k+m = (n+2m)\theta.$$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+m}{n+2m}.}$$

**(6) m → ∞**

분자·분모를 m으로 나누면:
$$\hat{\theta}_{\text{MAP}} = \frac{k/m + 1}{n/m + 2} \xrightarrow{m \to \infty} \frac{0+1}{0+2} = \frac{1}{2}.$$

$$\boxed{\lim_{m\to\infty} \hat{\theta}_{\text{MAP}} = \frac{1}{2}.}$$

**(7) 직관**

prior $\theta^m(1-\theta)^m$은 0.5에 대한 사전 확신을 나타낸다. m이 클수록 prior가 0.5에 첨예해지며, m → ∞이면 디랙 델타에 수렴해 데이터를 완전히 무시하고 0.5만 남는다. ∎

---

## 채점 포인트

| 항목 | 배점 |
|------|------|
| 베이즈 정리 + ∝ 정당화 | 15% |
| Likelihood × Prior 결합 | 15% |
| log + 이유 | 10% |
| 미분=0 + 페르마 인용 | 15% |
| MAP 풀이 (k+m)/(n+2m) | 20% |
| m → ∞ 극한 정확 | 15% |
| 직관 | 10% |

---

## 다음

[`04-mastery-quiz.md`](./04-mastery-quiz.md)

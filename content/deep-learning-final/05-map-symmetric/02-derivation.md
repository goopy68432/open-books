---
title: "02. 6단계 유도"
slug: derivation
order: 2
---

# 02. 6단계 유도

---

## 단계 1: 베이즈 정리 명시

$$p(\theta | D) = \frac{p(D|\theta) p(\theta)}{p(D)} \propto p(D|\theta) p(\theta)$$

**왜 ∝?** 분모 $p(D)$는 θ에 무관 → MAP의 $\arg\max$에서 무시 가능.

---

## 단계 2: likelihood + prior 결합

기출 4번 결과 (likelihood):
$$p(D|\theta) = L(\theta) = \theta^k (1-\theta)^{n-k}$$

이번 문제 prior:
$$p(\theta) \propto \theta^m (1-\theta)^m$$

곱:
$$p(\theta | D) \propto \theta^k(1-\theta)^{n-k} \cdot \theta^m(1-\theta)^m$$

지수 합:
$$\boxed{p(\theta | D) \propto \theta^{k+m}(1-\theta)^{n-k+m}}$$

**관찰:** posterior도 Beta 형태! (Beta는 Bernoulli의 **conjugate prior**) — "성공 k+m번, 실패 n-k+m번"인 가상의 데이터처럼 작동.

---

## 단계 3: log posterior

$$\log p(\theta|D) = (k+m)\log\theta + (n-k+m)\log(1-\theta) + \text{const}$$

(상수항은 θ 무관 → 미분에서 사라짐)

**왜 log?** 4번 문제와 동일 (곱→합, 단조성, 수치안정).

---

## 단계 4: 미분=0

$$\frac{d}{d\theta}\log p(\theta|D) = \frac{k+m}{\theta} - \frac{n-k+m}{1-\theta} = 0$$

**왜 미분=0?** 페르마 정리. 사후분포가 (0,1) 내부에서 미분 가능.

---

## 단계 5: 풀이

$$\frac{k+m}{\theta} = \frac{n-k+m}{1-\theta}$$

크로스 곱:
$$(k+m)(1-\theta) = (n-k+m)\theta$$
$$(k+m) - (k+m)\theta = (n-k+m)\theta$$
$$k+m = (n-k+m+k+m)\theta = (n+2m)\theta$$

$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+m}{n+2m}}$$

---

## 단계 6: m → ∞ 극한

분자, 분모를 **m으로 나누기**:
$$\hat{\theta}_{\text{MAP}} = \frac{k/m + 1}{n/m + 2}$$

$m \to \infty$이면 $k/m \to 0$, $n/m \to 0$:
$$\hat{\theta}_{\text{MAP}} \to \frac{0 + 1}{0 + 2} = \frac{1}{2}$$

$$\boxed{\lim_{m \to \infty} \hat{\theta}_{\text{MAP}} = \frac{1}{2}}$$

---

## 직관 정리

> "m이 클수록 prior가 0.5에 첨예해진다 (확신 강함). m → ∞이면 prior가 0.5에 집중된 디랙 델타가 되어 데이터를 완전히 무시한다."

**검증 (다른 극한):**
- m = 0: $\hat{\theta} = k/n$ = MLE ✓ (prior가 균일 = 정보 없음)
- m → ∞: 0.5 ✓ (prior가 0.5 확신)

---

## 부등식 시각화

m = 0, 1, 10, 100, ∞일 때 $\hat{\theta}$ 위치 (k=7, n=10 가정):

| m | $\hat{\theta}$ | 의미 |
|---|--------------|------|
| 0 | 7/10 = 0.70 | MLE |
| 1 | 8/12 ≈ 0.67 | 약간 0.5 쪽으로 |
| 10 | 17/30 ≈ 0.57 | 더 가까이 |
| 100 | 107/210 ≈ 0.51 | 거의 0.5 |
| ∞ | 1/2 | 완전 0.5 |

**규제(regularization) 효과** — 데이터를 prior 쪽으로 잡아당김.

---

## 다음

[`03-perfect-answer.md`](./03-perfect-answer.md)

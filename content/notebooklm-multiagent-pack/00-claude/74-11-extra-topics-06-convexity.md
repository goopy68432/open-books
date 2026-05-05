---
title: "06. 볼록성 (Convexity)"
slug: 11-extra-topics-06-convexity
order: 74
---

# 06. 볼록성 (Convexity)

> page_372 등장. Jensen 부등식과 함께 묶음.

---

## 1. 볼록 집합 (Convex Set)

집합 $C \subseteq \mathbb{R}^n$이 **볼록** ⇔ 두 점 잇는 선분이 C 안:
$$\forall x, y \in C, \lambda \in [0, 1]: \lambda x + (1-\lambda) y \in C$$

**예:** 원, 구, 직육면체, 반평면. 도넛은 X.

---

## 2. 볼록 함수 (Convex Function)

$f: C \to \mathbb{R}$가 **볼록** ⇔
$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)$$

**오목** ⇔ 부등호 반대 (-f가 볼록).

### 그래프 직관

볼록 = 위로 휘어짐:
```
   f(x)
    │     ╲
    │       ╲___╱
    │    선분이 곡선 위
```

### 시험 핵심 예
| 함수 | 볼록/오목 |
|------|---------|
| $x^2$ | 볼록 |
| $e^x$ | 볼록 |
| $-\log x$ | 볼록 ($x > 0$) |
| $\log x$ | 오목 |
| $x^p$ ($p \geq 1$) | 볼록 |
| $|x|$ | 볼록 |

---

## 3. 판정법

### 1차 조건 (미분 가능)
$f$ 볼록 ⇔ $f(y) \geq f(x) + f'(x)(y - x) \forall x, y$
"접선 위에 있음"

### 2차 조건 (2번 미분 가능)
$f$ 볼록 ⇔ $f''(x) \geq 0 \forall x$ (1변수)

다변수: $f$ 볼록 ⇔ Hessian $H(f)$가 양반정치(PSD).

---

## 4. **MSE 볼록성 증명** (시험 단골)

$L(\theta) = \sum_i (y_i - x_i^T \theta)^2$ (linear regression).

$$\nabla L = -2 \sum_i x_i (y_i - x_i^T \theta) = -2 X^T (y - X\theta)$$
$$\nabla^2 L = 2 X^T X$$

$X^T X$는 항상 **양반정치** ($v^T (X^T X) v = \|Xv\|^2 \geq 0$).

→ L 볼록. 임계점 = 전역 최솟값. ∎

---

## 5. **Cross Entropy 볼록성 증명** (시험 단골)

$L(\hat{p}) = -y\log\hat{p} - (1-y)\log(1-\hat{p})$.

$$L'' = \frac{y}{\hat{p}^2} + \frac{1-y}{(1-\hat{p})^2} > 0$$

→ L 볼록 (in $\hat{p}$). ∎

(단, NN의 가중치 $\theta$에 대해서는 일반적으로 비볼록 — 활성화 비선형성 때문.)

---

## 6. Jensen 부등식 (재방문)

$f$ 볼록 ⇔ $f(E[X]) \leq E[f(X)]$.

응용:
- KL ≥ 0: $-\log$가 볼록 → Jensen → $\log E[q/p] \geq E[\log(q/p)]$
- 분산 ≥ 0: $x^2$ 볼록 → $(E[X])^2 \leq E[X^2]$
- AM-GM: 산술-기하 평균 부등식

---

## 7. **볼록 최적화의 위력**

### 정리 (전역 최솟값 보장)

$f$가 볼록이고 임계점 $\theta^*$ ($\nabla f(\theta^*) = 0$)이 존재하면:
$$\theta^* = \arg\min_\theta f(\theta) \quad \text{(전역 최솟값)}$$

### 증명

볼록 + 1차 조건:
$$f(\theta) \geq f(\theta^*) + \nabla f(\theta^*)^T (\theta - \theta^*) = f(\theta^*) + 0 = f(\theta^*)$$

→ $\theta^*$가 모든 θ보다 작거나 같음 → 전역 최솟값. ∎

### MLE/MAP과의 연결

베르누이 NLL은 **오목** (페르마 적용 시 전역 최댓값 보장):
$$\text{NLL}''(\theta) = \frac{k}{\theta^2} + \frac{n-k}{(1-\theta)^2} > 0 \Rightarrow \text{NLL 볼록}$$
$$\Rightarrow \ell = -\text{NLL} \text{ 오목}$$

$\hat{\theta} = k/n$이 **전역 최댓값**. 시험 답안에 한 줄 추가하면 가산점.

---

## 8. 시험 답안 — Jensen + KL≥0 통합

### [문제] $\log$가 오목임을 사용해 KL ≥ 0을 증명하라.

### [풀이]

$\log''(x) = -1/x^2 < 0$이므로 $\log$는 오목.

오목 Jensen: $E[\log Y] \leq \log E[Y]$.

$Y = q(X)/p(X)$로 두고 $X \sim p$:
$$E_p[\log(q/p)] \leq \log E_p[q/p] = \log \int p \cdot (q/p)\,dx = \log 1 = 0$$

따라서:
$$\text{KL}(p\|q) = E_p[\log(p/q)] = -E_p[\log(q/p)] \geq 0. \quad \blacksquare$$

---

## 9. 한 줄 요약

> "볼록 = $f''>0$ (또는 Jensen). 볼록 함수의 임계점 = 전역 최솟값. MSE/CE 볼록 → MLE 풀이가 전역 최적."

---
title: "딥러닝 이론 모의고사 #6 — 베이지안/MLE/MAP 집중"
slug: 08-mock-exam-6-bayesian
order: 12
---

# 딥러닝 이론 모의고사 #6 — 베이지안/MLE/MAP 집중

> 배점 100점 / 8문제 / ★★★★ 시험 핵심 파트

---

## 문제 1. [12점] 베이즈 정리 유도와 해석

**(a)** [4점] 조건부 확률의 정의로부터 베이즈 정리를 유도하시오.

**(b)** [4점] $P(H|E) = \frac{P(E|H)P(H)}{P(E)}$의 각 항(Prior, Likelihood, Posterior, Evidence)의 딥러닝에서의 대응을 서술하시오.

**(c)** [4점] "베이지안 학습은 Prior를 데이터로 업데이트하여 Posterior를 얻는 과정이다"를 수식과 함께 설명하시오. 데이터가 순차적으로 들어올 때 Posterior가 다음의 Prior가 되는 재귀 구조를 보이시오.

---

## 문제 2. [12점] MLE 완전 유도

**(a)** [4점] MLE의 정의를 쓰고, "왜 Likelihood를 최대화하는가"를 직관적으로 설명하시오.

**(b)** [8점] 다음 두 경우에 대해 MLE를 완전히 유도하시오 (i.i.d. → log → 미분=0, 각 단계에 "왜"):

(i) 베르누이: 동전 $n$번 던져 $k$번 앞면
(ii) 가우시안: $\{x_1,...,x_n\} \sim \mathcal{N}(\mu, \sigma^2)$에서 $\mu$ 추정

---

## 문제 3. [15점] MAP 완전 유도

**(a)** [5점] MAP와 MLE의 차이를 수식 한 줄로 보이고, "Prior가 추가된다"는 것의 의미를 설명하시오.

**(b)** [5점] $P(\theta) = \text{Beta}(3, 3)$일 때, 동전 8번 중 6번 앞면에 대한 $\theta_{MAP}$를 구하시오.

**(c)** [5점] Gaussian Prior에서 MAP = L2 Reg를 유도하시오. Laplace Prior에서는 어떤 Reg가 되는지도 서술하시오.

---

## 문제 4. [12점] MLE의 한계와 MAP의 장점

**(a)** [4점] 동전 3번 던져 3번 앞면: MLE = 1. 이것이 왜 비현실적인지 설명하시오.

**(b)** [4점] 같은 데이터에 Beta(2,2) Prior를 적용한 MAP를 구하고, 왜 더 합리적인지 설명하시오.

**(c)** [4점] 데이터가 $n \to \infty$이면 $\theta_{MAP} \to \theta_{ML}$이 됨을 정량적으로 보이시오. (Beta Prior + 베르누이 Likelihood 사용)

---

## 문제 5. [12점] Uniform Prior = MLE

**(a)** [6점] Prior가 Uniform일 때 MAP = MLE임을 증명하시오.

**(b)** [6점] 이로부터 "MLE는 사실 Prior를 Uniform으로 놓은 MAP의 특수한 경우"라는 해석을 설명하시오. 이것이 "MLE는 모든 가설을 동등하게 본다"는 의미인 이유를 설명하시오.

---

## 문제 6. [12점] Strong Prior vs Weak Prior

**(a)** [6점] Prior의 강도(strength)가 MAP 결과에 미치는 영향을 $\theta_{MAP} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$ (Beta Prior 경우)를 사용하여 설명하시오. $\alpha, \beta$가 커지면 어떻게 되는가?

**(b)** [6점] 딥러닝에서 $\lambda$(정규화 강도)가 크면 어떤 현상이 발생하는가? 이것을 MAP 관점에서 해석하시오.

---

## 문제 7. [12점] Posterior 전체 분포

**(a)** [6점] MLE와 MAP는 점추정(point estimate)이다. Bayesian 접근의 완전한 형태는 Posterior 전체를 구하는 것이다. 이 차이를 설명하고, 왜 실전에서 점추정을 쓰는지 이유를 드시오.

**(b)** [6점] 베르누이 Likelihood + Beta Prior → Beta Posterior가 되는 "켤레 사전분포(conjugate prior)" 성질을 보이시오.

---

## 문제 8. [13점] 통합: 확률→Loss→정규화 파이프라인

아래의 완전한 유도 체인을 서술하시오 (빈칸 없이):

```
가정: y ~ N(h_θ(x), σ²), θ ~ N(0, σ_p²)
     ↓
MAP = argmax [log P(y|x,θ) + log P(θ)]
     ↓
    = argmin [ (a) + (b) ]
     ↓
    = argmin [ MSE + λ‖θ‖² ]  여기서 λ = (c)
     ↓
    = Ridge Regression
```

**(d)** [3점] 이 유도 체인에서 "왜"가 필요한 모든 지점을 나열하시오.

---
---

# 모범답안

## 답 3.
### (b)
$P(D|\theta) = \theta^6(1-\theta)^2$, $P(\theta) \propto \theta^2(1-\theta)^2$
Posterior $\propto \theta^8(1-\theta)^4$
$\frac{\partial}{\partial\theta}[8\log\theta + 4\log(1-\theta)] = \frac{8}{\theta} - \frac{4}{1-\theta} = 0$
$8(1-\theta) = 4\theta$ → $8 = 12\theta$ → $\theta_{MAP} = 2/3$

공식: $\theta_{MAP} = \frac{k+\alpha-1}{n+\alpha+\beta-2} = \frac{6+3-1}{8+3+3-2} = \frac{8}{12} = 2/3$ ✓

## 답 4.
### (b)
$\theta_{MAP} = \frac{3+2-1}{3+2+2-2} = \frac{4}{5} = 0.8$ (MLE=1보다 합리적: 4번째도 앞면일 확률이 80%)

### (c)
$\theta_{MAP} = \frac{k+\alpha-1}{n+\alpha+\beta-2}$, $\theta_{ML} = k/n$

$\theta_{MAP} = \frac{k/n + (\alpha-1)/n}{1 + (\alpha+\beta-2)/n} \xrightarrow{n\to\infty} \frac{k/n}{1} = k/n = \theta_{ML}$

Prior 항 $(\alpha-1)/n, (\beta-1)/n \to 0$ → Prior의 영향 $O(1/n)$으로 소멸. $\square$

## 답 7.
### (b)
$P(\theta|D) \propto P(D|\theta)P(\theta) = \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}$
$= \theta^{k+\alpha-1}(1-\theta)^{n-k+\beta-1}$
이것은 $\text{Beta}(k+\alpha, n-k+\beta)$의 형태! → Beta가 Bernoulli의 켤레 사전분포.

의미: 관측 $k$번 성공 → Prior의 $\alpha$에 $k$를 더하고, $\beta$에 $n-k$를 더함 → "가상 관측이 실제 관측에 합산"

## 답 8.
(a) $\frac{1}{2\sigma^2}\sum(y_i - h_\theta(x_i))^2$ (= NLL from Gaussian, const 제외)
(b) $\frac{1}{2\sigma_p^2}\|\theta\|_2^2$ (= $-\log P(\theta)$, const 제외)
(c) $\lambda = \sigma^2/\sigma_p^2$ (또는 $1/(2\sigma_p^2)$ if NLL에서 $1/(2\sigma^2)$ 이미 포함)

(d) "왜" 필요 지점:
1. 왜 Gaussian 가정 → CLT
2. 왜 i.i.d. → 곱 분해를 위해
3. 왜 log → 곱→합, 수치안정
4. 왜 argmin → 부호 반전 (NLL)
5. 왜 Gaussian Prior → 가중치가 0 근처라는 사전 믿음
6. 왜 Evidence 무시 → θ에 무관

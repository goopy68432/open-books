---
title: "📝 [V3] Model Answer - Q2. 확률적 추정 (MLE, MAP, Bayesian)"
slug: 02-model-answer-q2
order: 3
---

# 📝 [V3] Model Answer - Q2. 확률적 추정 (MLE, MAP, Bayesian)

**[평가 기준]**
* MLE, MAP 수식 전개 시 `IID 가정` 및 `Log 변환의 이유`를 명시했는가?
* 미분 불가능 지점 문제에서 미분(=0) 대신 Max 연산을 사용해야 하는 이유를 서술했는가?
* Prior Update의 개념을 수식과 함께 온라인 학습 관점으로 해석했는가?

--- 

## (a) $U(0, \theta)$ 에 대한 MLE 및 미분 불가능 지점(Boundary Condition)

**[MLE 유도]**
데이터 $D = \{x_1, \dots, x_n\}$ 이 IID 가정을 만족한다고 할 때, Likelihood는 각 확률 밀도의 곱입니다.
균등분포 $U(0, \theta)$ 의 pdf는 $f(x) = \frac{1}{\theta}$ (단, $0 \le x_i \le \theta$) 입니다.
$$L(\theta) = \prod_{i=1}^n P(x_i | \theta) = \prod_{i=1}^n \frac{1}{\theta} = \theta^{-n}$$
로그 변환(Log-likelihood)을 취합니다.
$$\ell(\theta) = -n \log \theta \quad \text{(단, 모든 } x_i \le \theta \text{)}$$

**[미분 불가능 지점 고려 및 해결 로직]**
*   **오류 원인:** 단순히 $\frac{\partial \ell}{\partial \theta} = -\frac{n}{\theta} = 0$ 으로 미분하여 풀려고 하면 근을 찾을 수 없습니다. 이는 $\theta$가 $x_i$들의 최댓값 이상이어야 한다는 경계값(Boundary) 조건이 존재하기 때문입니다.
*   **올바른 로직:** 목적함수 $-n \log \theta$ 를 최대화하려면 $\theta$ 를 **최대한 작게** 만들어야 합니다. 단, 제약조건 `if x_i <= theta for all i` 를 만족해야 하므로, $\theta$ 가 가질 수 있는 가장 작은 값은 데이터 중 가장 큰 값입니다.
*   **결론:** $\hat{\theta}_{MLE} = \max(x_1, x_2, \dots, x_n)$

--- 

## (b) 베르누이 데이터와 Prior $P(\theta) \propto \theta^m (1-\theta)^m$ 에서의 MAP 증명

**[정의 및 IID 가정]**
데이터 $D$는 총 $n$번 중 성공 $k$번, 실패 $n-k$번입니다. IID 가정에 의해 우도(Likelihood)는 다음과 같습니다.
$$P(D|\theta) = \theta^k (1-\theta)^{n-k}$$

**[Log 변환 및 목적함수(Objective Function) 설정]**
MAP는 사후 확률 $P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$ 를 최대화합니다.
Underflow 방지와 연산 편의를 위해 로그를 취합니다.
$$J(\theta) = \log P(D|\theta) + \log P(\theta)$$
$$J(\theta) = (k \log \theta + (n-k) \log(1-\theta)) + (m \log \theta + m \log(1-\theta))$$

**[항 묶기 및 미분]**
동류항을 묶으면, 데이터의 성공 횟수와 실패 횟수에 각각 $m$개씩 가상의 더미 데이터를 추가한 것과 같습니다.
$$J(\theta) = (k + m) \log \theta + (n - k + m) \log(1-\theta)$$
최적화를 위해 $\theta$ 에 대해 편미분하고 0으로 둡니다.
$$\frac{\partial J}{\partial \theta} = \frac{k + m}{\theta} - \frac{n - k + m}{1-\theta} = 0$$

**[최적 파라미터 도출]**
$$(k + m)(1-\theta) = (n - k + m)\theta$$
$$k + m - (k + m)\theta = (n - k + m)\theta$$
$$k + m = (n + 2m)\theta$$
$$\therefore \hat{\theta}_{MAP} = \frac{k + m}{n + 2m}$$
*   **응용(의미):** $m \rightarrow \infty$ 로 갈수록 $\hat{\theta}_{MAP} \rightarrow 0.5$ 에 수렴합니다. 이는 데이터보다 설계자의 사전 지식(Prior Knowledge)이 시스템 결정에 절대적 영향을 미침을 의미합니다.

--- 

## (c) 가우시안 가정 하의 $\mu, \sigma^2$ 에 대한 MLE 유도

**[가정 및 NLL 도출]**
$x_i \sim \mathcal{N}(\mu, \sigma^2)$ 이며 IID를 가정합니다.
$$L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$
로그를 취하고 부호를 반전시켜 NLL(Negative Log-Likelihood)을 만듭니다 (최소화 문제로 치환).
$$NLL(\mu, \sigma^2) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**[1. $\mu$ 에 대한 편미분 (평균 유도)]**
$\frac{\partial NLL}{\partial \mu} = \frac{1}{2\sigma^2} \sum_{i=1}^n 2(x_i - \mu)(-1) = 0$
$\sum_{i=1}^n x_i - n\mu = 0$
$$\therefore \hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i \quad \text{(표본 평균)}$$

**[2. $\sigma^2$ 에 대한 편미분 (분산 유도)]**
계산 편의를 위해 $\sigma^2 = v$ 로 치환하고 편미분합니다.
$\frac{\partial NLL}{\partial v} = \frac{n}{2v} - \frac{1}{2v^2}\sum_{i=1}^n (x_i - \mu)^2 = 0$
$\frac{n}{v} = \frac{\sum (x_i - \mu)^2}{v^2}$
$$\therefore \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2 \quad \text{(표본 분산)}$$

--- 

## (d) Prior Update (온라인 학습 로직)

**[개념 및 수식]**
베이지안 학습에서는 매번 처음부터 전체 데이터를 재학습(Full Batch)하지 않습니다.
*   시점 $t=1$: 데이터 $D_1$ 이 들어오면 Posterior를 계산합니다.
    $$P(\theta | D_1) \propto P(D_1 | \theta) P(\theta)$$
*   시점 $t=2$: 새로운 데이터 $D_2$ 가 들어오면, 어제 저장해둔 $t=1$의 Posterior $P(\theta | D_1)$ 를 오늘의 **새로운 Prior**로 사용합니다.
    $$P(\theta | D_1, D_2) \propto P(D_2 | \theta) P(\theta | D_1)$$

**[응용 (시스템적 의미)]**
이러한 Prior Update 구조는 **온라인 학습(Online Learning)** 아키텍처와 같습니다. 새로운 데이터 로그가 쌓일 때마다 기존 모델의 파라미터를 베이스라인(Prior)으로 삼아 가중치를 점진적으로 업데이트하므로, 메모리 효율이 극대화되고 시스템 부하를 줄일 수 있습니다.

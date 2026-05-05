---
title: "Level 4: Master - 완벽 답안 작성 (Perfect Answers)"
slug: 01-perfect-answers
order: 1
---

# Level 4: Master - 완벽 답안 작성 (Perfect Answers)

> **학습 목표:** 교수님의 채점 철학인 **"답만 적으면 0점"** 을 명심하며, 백지 상태에서 시험장에 제출할 완벽한 A+ 답안을 재현합니다. 수식만 나열하지 말고, **글로 이유를 명시**하는 것이 핵심입니다.

---

## 1. 기출: 고유값과 고유벡터
**[완벽 답안 예시]**
고유값과 고유벡터의 정의에 의해, 행렬 $A$의 고유값 $\lambda$와 고유벡터 $v \neq 0$는 $Av = \lambda v$를 만족해야 합니다.
이를 이항하면 $(A-\lambda I)v = 0$ 이 됩니다.
$v$가 영벡터가 아닌 해를 가지기 위해서는 앞의 행렬 $(A-\lambda I)$의 역행렬이 존재하지 않아야 하므로, 행렬식 $\det(A-\lambda I) = 0$ 을 만족해야 합니다.
$A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ 이므로, 특성다항식은 $\lambda^2 - 1 = 0$ 입니다.
따라서 $\lambda = 1$ 또는 $\lambda = -1$ 입니다.
$\lambda = 1$ 일 때, 대입하여 풀면 단위벡터 $v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ 입니다.
$\lambda = -1$ 일 때, 대입하여 풀면 단위벡터 $v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$ 입니다.
(정의 검증: $Av_1 = v_1 = 1 \cdot v_1$, $Av_2 = -v_2 = -1 \cdot v_2$ 로 성립합니다.)

## 2. 기출: 베르누이 MLE (★출제 100%)
**[완벽 답안 예시]**
주어진 데이터 $y_i$는 서로 독립이며 동일한 베르누이 분포를 따른다고(i.i.d) 가정합니다.
베르누이 분포의 확률질량함수는 $p(y_i) = \theta^{y_i} (1-\theta)^{1-y_i}$ 입니다.
i.i.d 가정에 의해, 결합 확률 밀도인 우도(Likelihood) 함수는 각 확률의 곱으로 나타낼 수 있습니다:
$L(\theta) = \prod_{i=1}^n \theta^{y_i} (1-\theta)^{1-y_i} = \theta^k (1-\theta)^{n-k}$ (단, $k = \sum y_i$).
미분의 편의성과 Underflow 방지를 위해 로그를 취하고, 최솟값 탐색 문제로 바꾸기 위해 음수를 곱하여 NLL(Negative Log Likelihood)을 구성합니다.
$\text{NLL}(\theta) = -k \log \theta - (n-k) \log(1-\theta)$
페르마의 임계점 정리에 의해, 함수의 극값은 1차 도함수가 0이 되는 지점에서 발생하므로 $\theta$에 대해 미분하여 0으로 둡니다.
$\frac{\partial \text{NLL}}{\partial \theta} = - \frac{k}{\theta} + \frac{n-k}{1-\theta} = 0$
이를 정리하면 $k(1-\theta) = \theta(n-k)$ 가 되며, 전개하면 $k = n\theta$ 가 됩니다.
따라서 최대우도추정량 $\hat{\theta}_{MLE} = \frac{k}{n}$ 입니다.
(2차 미분값 $\frac{k}{\theta^2} + \frac{n-k}{(1-\theta)^2} > 0$ 이므로 아래로 볼록한 함수이며, 이는 유일한 최솟값임을 보장합니다.)

## 3. 기출: 대칭 MAP 극한
**[완벽 답안 예시]**
베이즈 정리에 의해 사후확률은 우도와 사전확률의 곱에 비례합니다 (Posterior $\propto$ Likelihood $\times$ Prior).
로그를 취한 로그 사후확률은 $\ell(\theta) = k \log \theta + (n-k) \log(1-\theta) + m \log \theta + m \log(1-\theta)$ 입니다.
이를 묶으면 $\ell(\theta) = (k+m) \log \theta + (n-k+m) \log(1-\theta)$ 가 됩니다.
극대값을 찾기 위해 미분하여 0이 되는 지점을 찾습니다.
$\frac{\partial \ell}{\partial \theta} = \frac{k+m}{\theta} - \frac{n-k+m}{1-\theta} = 0$
$(k+m)(1-\theta) = (n-k+m)\theta$ 에서 $\theta(n+2m) = k+m$ 이므로, $\hat{\theta}_{MAP} = \frac{k+m}{n+2m}$ 입니다.
$m \to \infty$ 극한을 취하기 위해 분모 분자를 $m$으로 나누면 $\frac{k/m + 1}{n/m + 2}$ 가 되며, $m$이 무한히 커질 때 $k/m, n/m \to 0$ 이 되므로 극한값은 $1/2$ 입니다.
이는 Prior의 "반반일 것이다"라는 확신이 무한히 커지면, 실제 관측 데이터 $n, k$의 영향력이 0이 되어 결론이 $0.5$로 수렴함을 의미합니다.

## 4. 기출: Softmax 미분 (자코비안)
**[완벽 답안 예시]**
Softmax 함수는 $p_i = \frac{\exp(z_i)}{\sum_k \exp(z_k)}$ 로 정의되며, 분모를 편의상 $S$라 표기하겠습니다.
이 함수의 야코비안 행렬 요소 $\frac{\partial p_i}{\partial z_j}$ 를 구하기 위해 분수 함수의 미분(몫의 미분법)을 사용하며, 두 가지 경우로 나눕니다.

Case 1 ($i = j$): 분자 분모 모두에 $z_i$가 존재합니다.
$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot S - \exp(z_i) \cdot \exp(z_i)}{S^2} = \frac{\exp(z_i)}{S} - \left(\frac{\exp(z_i)}{S}\right)^2 = p_i - p_i^2 = p_i(1 - p_i)$

Case 2 ($i \neq j$): 분자에는 $z_j$가 없어 상수 취급(미분 시 0)되고, 분모에는 $z_j$가 존재합니다.
$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - \exp(z_i) \cdot \exp(z_j)}{S^2} = - \frac{\exp(z_i)}{S} \frac{\exp(z_j)}{S} = - p_i p_j$

이를 크로네커 델타($\delta_{ij}$)를 사용하여 하나의 식으로 표현하면 $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$ 가 됩니다.
행렬 형태로 표현하면 자코비안 $J = \text{diag}(p) - p p^T$ 로 나타낼 수 있습니다.

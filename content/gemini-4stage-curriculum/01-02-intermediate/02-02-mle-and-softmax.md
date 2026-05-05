---
title: "Level 2: Intermediate - MLE와 Softmax 미분 (MLE & Softmax)"
slug: 02-mle-and-softmax
order: 2
---

# Level 2: Intermediate - MLE와 Softmax 미분 (MLE & Softmax)

> **학습 목표:** 출제 확률 100%를 자랑하는 베르누이 MLE와 Softmax 미분의 유도 과정을 한 줄 한 줄 "왜(Why)"를 달아가며 완벽하게 소화합니다.

---

## 1. 베르누이 MLE (최대우도추정) 기출문제 유도

**문제:** $y_i \sim \text{Bernoulli}(\theta)$ 가 i.i.d 로 주어지고, $k = \sum_{i=1}^n y_i$ 일 때, MLE 방식을 통해 $\hat{\theta}_{MLE}$ 를 구하시오.

### Step-by-Step 유도 체인

1. **베르누이 PMF 정의:** $p(y_i | \theta) = \theta^{y_i} \cdot (1-\theta)^{1-y_i}$
   * **왜 이렇게 쓰나요?** $y_i=1$(성공)이면 $\theta^1 \cdot (1-\theta)^0 = \theta$ 가 되고, $y_i=0$(실패)이면 $\theta^0 \cdot (1-\theta)^1 = 1-\theta$ 가 되도록, 하나의 수식으로 영리하게 합친 것입니다.
2. **Likelihood (우도) 함수 만들기:** $L(\theta) = \prod_{i=1}^n p(y_i | \theta) = \prod_{i=1}^n \theta^{y_i} (1-\theta)^{1-y_i}$
   * **왜 다 곱하나요? (i.i.d 가정)** 각각의 동전 던지기는 서로 독립(Independent)입니다. 독립인 두 사건이 동시에 일어날 확률은 곱셈입니다. (예: 주사위 1, 동전 앞면 = $1/6 \times 1/2$)
   * 지수법칙에 의해 곱셈은 지수의 덧셈이 됩니다. $\sum y_i = k$ 이므로:
   * $L(\theta) = \theta^k \cdot (1-\theta)^{n-k}$
3. **Log Likelihood (로그 우도):** $\ell(\theta) = \log L(\theta) = k \log \theta + (n-k) \log(1-\theta)$
   * **왜 로그를 취하나요?** 첫째, 미분하기 쉬운 덧셈으로 바꾸기 위해. 둘째, 확률(0~1)을 수만 번 곱하면 컴퓨터가 0으로 인식(Underflow)해버리기 때문입니다.
4. **NLL (Negative Log Likelihood):** $-\ell(\theta) = -k \log \theta - (n-k) \log(1-\theta)$
   * **왜 음수를 붙이나요?** 머신러닝에서는 보통 오차(Loss)를 '최소화'하는 식으로 문제를 풉니다. 최댓값을 찾는 문제를, 마이너스를 붙여 최솟값을 찾는 문제로 바꾸는 표준화 과정입니다.
5. **미분하여 0이 되는 지점 찾기:** $\frac{\partial (-\ell(\theta))}{\partial \theta} = - \frac{k}{\theta} + \frac{n-k}{1-\theta} = 0$
   * **왜 미분해서 0을 만드나요?** 페르마의 정리에 의해, 부드러운 곡선에서 최댓값이나 최솟값은 기울기(미분값)가 평평한(0인) 곳에 있기 때문입니다.
6. **최종 계산:**
   * $\frac{k}{\theta} = \frac{n-k}{1-\theta} \implies k(1-\theta) = \theta(n-k)$
   * $k - k\theta = n\theta - k\theta \implies n\theta = k$
   * **결론:** $\hat{\theta}_{MLE} = \frac{k}{n}$ (총 던진 횟수 분의 성공 횟수. 우리의 직관과 완벽히 일치!)

---

## 2. Softmax 미분 기출문제 유도

**문제:** $p = \text{softmax}(z)$ 일 때, 야코비안 행렬 $\frac{\partial p}{\partial z}$ 를 구하시오.

### Step-by-Step 유도 체인

*Softmax의 정의:* $p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$

이 미분은 분수 형태이므로 **몫의 미분법** $(\frac{f}{g})' = \frac{f'g - fg'}{g^2}$ 를 사용합니다.
계모수 분모를 $S = \sum_j \exp(z_j)$ 라고 짧게 부르겠습니다. 즉, $p_i = \frac{\exp(z_i)}{S}$ 입니다.

이 미분은 두 가지 케이스로 나뉩니다. "나 자신에 대한 미분"($i=j$)과 "남에 대한 미분"($i \neq j$)입니다.

1. **Case 1: $i = j$ 일 때 (나 자신을 미분 $\frac{\partial p_i}{\partial z_i}$)**
   * 분자 $f = \exp(z_i)$ 이고, 분모 $g = S$ 입니다.
   * $z_i$로 미분하면 $f' = \exp(z_i)$ 이고, 분모 $S$ 안에도 $z_i$가 하나 들어있으므로 $g' = \exp(z_i)$ 입니다.
   * $\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot S - \exp(z_i) \cdot \exp(z_i)}{S^2}$
   * 하트 모양으로 분리하면: $\frac{\exp(z_i)}{S} - \frac{\exp(z_i)^2}{S^2}$
   * $p_i = \exp(z_i)/S$ 이므로 대입하면: **$p_i - p_i^2 = p_i(1 - p_i)$**

2. **Case 2: $i \neq j$ 일 때 (남을 미분 $\frac{\partial p_i}{\partial z_j}$)**
   * 이번엔 $z_j$로 미분합니다. 분자 $f = \exp(z_i)$ 에는 $z_j$가 없으므로 상수 취급되어 $f' = 0$ 입니다.
   * 분모 $S$ 안에는 $z_j$가 들어있으므로 $g' = \exp(z_j)$ 입니다.
   * $\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot S - \exp(z_i) \cdot \exp(z_j)}{S^2} = - \frac{\exp(z_i)}{S} \cdot \frac{\exp(z_j)}{S}$
   * 대입하면: **$- p_i p_j$**

3. **크로네커 델타를 이용한 한 줄 정리:**
   * $\delta_{ij}$ (크로네커 델타)는 $i=j$면 1, 다르면 0을 반환하는 스위치입니다.
   * $\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)$

4. **야코비안 행렬 표기 (Jacobian Matrix):**
   * 대각선(나 자신)에는 $p_i(1-p_i)$ 가 들어가고, 나머지(남)에는 $-p_i p_j$ 가 들어가는 행렬입니다.
   * $J = \text{diag}(p) - p p^T$

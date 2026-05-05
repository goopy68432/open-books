---
title: "Level 3: Advanced - 고급 최적화와 기울기 문제 (Advanced Optimization)"
slug: 03-advanced-04-advanced-optimization
order: 11
---

# Level 3: Advanced - 고급 최적화와 기울기 문제 (Advanced Optimization)

> **학습 목표:** 망(Network)이 깊어질 때 발생하는 치명적인 수학적 한계인 기울기 소실/폭발 현상을 연쇄법칙으로 증명하고, 이를 극복하는 Adam과 Momentum의 수학적 원리를 이해합니다.

---

## 1. 기울기 소실 및 폭발 (Vanishing & Exploding Gradients)

신경망이 $L$개의 층으로 이루어져 있을 때, 맨 첫 번째 층의 가중치 $W_1$ 에 대한 기울기는 연쇄법칙에 의해 다음과 같이 수십 번의 곱셈으로 이루어집니다.

$$ \frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_L} \cdot \frac{\partial a_L}{\partial z_L} \cdot \frac{\partial z_L}{\partial a_{L-1}} \cdots \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1} $$

*여기서 $z_i$는 가중치가 곱해진 값($W_i a_{i-1}$), $a_i$는 활성화 함수(Sigmoid, ReLU 등)를 통과한 값입니다.*

이 긴 곱셈 체인에서 $\frac{\partial a_i}{\partial z_i}$ 는 활성화 함수의 미분값이고, $\frac{\partial z_i}{\partial a_{i-1}}$ 는 가중치 행렬 $W_i$ 입니다.
즉, 역전파 과정은 결국 **"활성화 함수의 미분값"**과 **"가중치 $W$"**를 계속해서 곱해나가는 과정입니다.

### 기울기 소실 (Vanishing)의 원인
전통적으로 쓰이던 **Sigmoid 활성화 함수**는 미분값의 최댓값이 **$0.25$**에 불과합니다.
연쇄법칙에 의해 0.25 보다 작은 숫자들을 계속해서 10번, 20번 곱하게 되면 어떻게 될까요?
$$ (0.25)^{10} \approx 0.00000095 $$
결국 맨 앞쪽 층으로 전달되는 기울기 값은 0에 한없이 수렴하게 되어, 앞쪽 층의 파라미터는 전혀 업데이트(학습)가 되지 않습니다. 이를 **기울기 소실(Vanishing Gradient)**이라고 부릅니다. (이 문제를 해결하기 위해 미분값이 항상 1인 **ReLU**가 등장했습니다.)

### 기울기 폭발 (Exploding)의 원인
반대로 가중치 행렬 $W$의 원소들이 1보다 큰 값을 가지고 있다면, 이를 계속해서 곱할 때 기하급수적으로 커집니다.
$$ (1.5)^{10} \approx 57.6 $$
기울기가 너무 커지면 한 번 업데이트할 때 파라미터가 비정상적인 값으로 튕겨 나가버립니다(오버플로우 발생). 이를 **기울기 폭발(Exploding Gradient)**이라고 하며, 흔히 Gradient Clipping(기울기 자르기)이나 배치 정규화(Batch Normalization)로 해결합니다.

---

## 2. 모멘텀 (Momentum)과 지수 가중 이동 평균 (EMA)

일반적인 경사 하강법 $W \leftarrow W - \eta \nabla L$ 은 안장점(Saddle point)이나 얕은 지역 최소점(Local Minimum)에 갇히기 쉽습니다. 이를 타개하기 위해 과거의 기울기를 '기억'하는 모멘텀 $v$ 를 도입합니다.

$$ v_{t+1} = \beta v_t + (1 - \beta) \nabla L_t $$
$$ W_{t+1} = W_t - \eta v_{t+1} $$

*   $\nabla L_t$ 는 현재 스텝에서 새로 구한 기울기입니다.
*   $\beta$ 는 $0.9$ 정도의 숫자입니다. 즉, "과거의 관성($v_t$)을 90% 반영하고, 현재 기울기는 10%만 반영하겠다"는 뜻입니다.
*   수학적으로 이를 **지수 가중 이동 평균 (Exponential Moving Average, EMA)** 이라고 부릅니다. 지그재그로 진동하는 불필요한 움직임을 상쇄하고, 계속해서 한 방향으로 가려는 힘(관성)을 누적시킵니다.

---

## 3. Adam 최적화기 (Adaptive Moment Estimation)

Adam은 현재 딥러닝에서 가장 기본값(Default)으로 쓰이는 궁극의 최적화 알고리즘입니다. Adam은 두 가지 아이디어를 합친 것입니다.

1.  **방향 (First Moment):** 모멘텀처럼 기울기의 EMA(평균 방향)를 구합니다. $\to m_t$
2.  **보폭 (Second Moment):** 기울기를 '제곱'한 값의 EMA(분산, 변동성)를 구합니다. $\to v_t$

업데이트 공식의 핵심 아이디어:
$$ W_{t+1} = W_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon} $$

*   **수학적 직관:**
    *   $\sqrt{v_t}$ 가 분모에 있습니다.
    *   만약 어떤 가중치가 그동안 엄청나게 크게 왔다 갔다 진동했다면($v_t$ 큼), 분모가 커지므로 보폭을 줄입니다. (조심해서 가라)
    *   어떤 가중치가 그동안 조금씩밖에 안 변했다면($v_t$ 작음), 분모가 작아지므로 보폭을 늘립니다. (더 과감하게 가라)
    *   즉, 가중치마다, 상황마다 **스스로 학습률(보폭)을 적응형(Adaptive)으로 조절**하는 수학적 마법입니다.

---
title: "07. 최적화 알고리즘 (Optimization)"
slug: optimization
order: 7
---

# 07. 최적화 알고리즘 (Optimization)

> SGD, 모멘텀, Adam — 신경망 학습의 엔진.

---

## 1. Gradient Descent (GD)

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla L(\theta^{(t)})$$

- η: 학습률 (learning rate)
- $\nabla L$: 그래디언트

### 직관
손실의 가장 가파른 내리막 방향으로 한 스텝.

### 수렴 조건
$L$이 **L-Lipschitz** 그래디언트 ($\|\nabla L(x) - \nabla L(y)\| \leq L\|x-y\|$):
- $\eta < 2/L$이면 수렴 보장 (볼록일 때)
- 일반적으로 $\eta < 1/L$ 권장

---

## 2. Stochastic Gradient Descent (SGD)

전체 데이터 대신 **미니배치** 사용:
$$\nabla L \approx \nabla L_{\text{batch}} = \frac{1}{|B|} \sum_{i \in B} \nabla L_i(\theta)$$

### 장점
- 빠름 (전체 데이터 안 보고 업데이트)
- 잡음이 local minimum 탈출 도움
- 메모리 효율

### 단점
- 잡음 → 진동
- 학습률 조정 어려움

---

## 3. Momentum

이전 그래디언트를 누적해 관성 효과:
$$v^{(t+1)} = \beta v^{(t)} + \nabla L(\theta^{(t)})$$
$$\theta^{(t+1)} = \theta^{(t)} - \eta v^{(t+1)}$$

- β = 0.9 표준
- "공이 굴러가는 듯한" 효과
- 평탄한 영역에서 가속, 잡음 평균화

---

## 4. Adam (Adaptive Moment Estimation)

$$m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1)\nabla L \quad \text{(1차 모멘트)}$$
$$v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2)(\nabla L)^2 \quad \text{(2차 모멘트)}$$

편향 보정:
$$\hat{m}^{(t)} = \frac{m^{(t)}}{1 - \beta_1^t}, \quad \hat{v}^{(t)} = \frac{v^{(t)}}{1 - \beta_2^t}$$

업데이트:
$$\theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{\sqrt{\hat{v}^{(t)}} + \epsilon} \hat{m}^{(t)}$$

### 표준 하이퍼파라미터
- $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$
- $\eta = 0.001$ (보통)

### 장점
- 파라미터별 적응적 학습률
- 잘 작동 (대부분의 경우)
- 학습률 튜닝 부담 적음

---

## 5. 비교표

| 방법 | 메모리 | 속도 | 안정성 |
|------|-------|------|------|
| GD | 작음 | 느림 | 안정 |
| SGD | 작음 | 빠름 | 잡음 |
| Momentum | 중 | 빠름 | 안정 |
| Adam | 큼 (모멘트 저장) | 빠름 | 매우 안정 |

---

## 6. 학습률 스케줄링

| 전략 | 식 |
|-----|-----|
| Step decay | t마다 $\eta \leftarrow \eta \cdot \gamma$ |
| Exponential | $\eta_t = \eta_0 e^{-kt}$ |
| Cosine | $\eta_t = \eta_{\min} + 0.5(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t/T))$ |
| Warmup | 처음 천천히 늘려서 안정화 |

---

## 7. **GD 수렴 정리** (시험 출제 가능)

### 정리 (L-smooth + 볼록 함수)

$f$가 L-smooth, 볼록이고 $\eta = 1/L$일 때:
$$f(\theta^{(t)}) - f(\theta^*) \leq \frac{\|\theta^{(0)} - \theta^*\|^2}{2 \eta t}$$

→ $O(1/t)$ 수렴.

### Strongly Convex
μ-strongly convex이면 **선형 수렴**:
$$f(\theta^{(t)}) - f(\theta^*) \leq \left(1 - \mu/L\right)^t [f(\theta^{(0)}) - f(\theta^*)]$$

(시험 직접 출제는 드뭄, 인용형으로 가산점 가능)

---

## 8. 시험 답안 — SGD vs GD 비교

### [질문] 왜 신경망에서는 SGD가 GD보다 자주 쓰이나?

### [답안]

> ① **계산 효율:** GD는 매 스텝마다 전체 N개 데이터의 그래디언트를 계산. SGD는 미니배치 B개만 사용 → 한 스텝 비용 N/B배 절감.
> ② **잡음 효과:** SGD의 잡음은 안장점(saddle point)과 얕은 local minimum을 탈출시키는 implicit regularizer 역할.
> ③ **메모리:** 전체 데이터가 메모리에 안 들어가도 학습 가능.
> 단, SGD는 진동이 있어 **학습률 스케줄**과 **모멘텀** 조합이 표준.

---

## 9. 한 줄 요약

> "GD는 안정, SGD는 빠르고 잡음. Momentum은 관성, Adam은 적응적 학습률. 대부분의 NN은 SGD+momentum 또는 Adam."

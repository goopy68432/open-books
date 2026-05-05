---
title: "03. 최적화 알고리즘"
slug: 03-optimization-algorithms
order: 4
---

# 03. 최적화 알고리즘

MLE/MAP의 간단한 예제에서는 미분해서 0이 되는 해를 닫힌형으로 구했습니다. 실제 딥러닝에서는 파라미터가 너무 많아 해를 한 번에 풀 수 없으므로 gradient를 따라 조금씩 이동합니다.

## 1. Gradient Descent

목표:

$$
\min_\theta L(\theta)
$$

업데이트:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
$$

$\eta$는 learning rate입니다.

직관:

- gradient는 손실이 가장 빠르게 증가하는 방향입니다.
- 손실을 줄이려면 그 반대 방향으로 갑니다.

## 2. SGD

전체 데이터 손실:

$$
L(\theta)=\frac1n\sum_{i=1}^n L_i(\theta)
$$

Batch GD는 전체 데이터를 다 보고 gradient를 계산합니다. SGD는 일부 샘플 또는 mini-batch만 사용합니다.

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L_{B_t}(\theta_t)
$$

장점:

- 계산이 빠릅니다.
- gradient noise가 local minimum이나 saddle point 탈출에 도움될 수 있습니다.

단점:

- 경로가 흔들립니다.
- learning rate 선택이 중요합니다.

## 3. Momentum

SGD는 매번 현재 gradient만 보고 움직입니다. Momentum은 과거 방향의 이동 평균을 사용합니다.

$$
v_t=\beta v_{t-1}+(1-\beta)g_t
$$

$$
\theta_{t+1}=\theta_t-\eta v_t
$$

여기서 $g_t=\nabla_\theta L_{B_t}(\theta_t)$입니다.

직관:

- 같은 방향으로 계속 gradient가 나오면 속도가 붙습니다.
- 서로 반대 방향으로 흔들리는 gradient는 평균화되어 줄어듭니다.

시험 문장:

> Momentum은 gradient의 지수이동평균을 사용해 관성을 부여한다. 따라서 일관된 방향의 이동은 가속하고, 진동하는 방향은 완화한다.

## 4. RMSProp

RMSProp은 gradient 제곱의 이동평균으로 각 파라미터별 스케일을 조정합니다.

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2
$$

$$
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
$$

직관:

- gradient가 자주 큰 좌표는 step을 줄입니다.
- gradient가 작은 좌표는 상대적으로 더 크게 움직입니다.

시험 문장:

> RMSProp은 gradient 제곱의 이동평균으로 각 좌표의 학습률을 나누어, 좌표별 adaptive learning rate를 만든다.

## 5. Adam

Adam은 Momentum과 RMSProp을 결합합니다.

1차 모멘트:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

2차 모멘트:

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

초기 bias 보정:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

업데이트:

$$
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

직관:

- $m_t$: 어느 방향으로 갈지의 평균
- $v_t$: 좌표별 gradient 크기
- $\hat m_t/\sqrt{\hat v_t}$: 방향과 스케일을 모두 보정한 step

## 6. AdamW와 Weight Decay

L2 regularization을 손실에 더하면:

$$
L_{reg}(\theta)=L(\theta)+\lambda\|\theta\|^2
$$

gradient:

$$
\nabla L_{reg}=\nabla L+2\lambda\theta
$$

SGD에서는 이것이 weight decay와 거의 같은 효과를 냅니다. Adam에서는 adaptive scaling 때문에 L2 penalty와 weight decay가 정확히 같지 않습니다. AdamW는 weight decay를 gradient update와 분리해 직접 적용합니다.

시험에서 깊게 묻지 않으면:

> AdamW는 Adam에서 weight decay를 gradient 계산과 분리해 정규화 효과를 더 명확히 적용한 방법이다.

정도로 충분합니다.

## 7. 알고리즘 비교

| 알고리즘 | 핵심 | 장점 | 약점 |
|---|---|---|---|
| GD | 전체 gradient | 안정적 | 느림 |
| SGD | mini-batch gradient | 빠름, noise | 진동 |
| Momentum | gradient 평균 | 진동 감소, 가속 | hyperparameter 필요 |
| RMSProp | gradient 제곱 평균 | 좌표별 적응 | 방향 평균 없음 |
| Adam | Momentum + RMSProp | 실전에서 강함 | 일반화/weight decay 주의 |
| AdamW | Adam + decoupled weight decay | 현대 표준 | 설정값 의존 |

## 8. 시험 답안 문장

> 경사하강법은 손실의 gradient가 가장 빠른 증가 방향이라는 사실을 이용해, 그 반대 방향으로 파라미터를 반복적으로 업데이트한다. SGD는 전체 데이터 대신 mini-batch gradient를 사용해 계산량을 줄이고 noise를 도입한다. Momentum은 gradient의 이동평균으로 관성을 부여해 진동을 줄이고 일관된 방향을 가속한다. RMSProp은 gradient 제곱의 이동평균으로 좌표별 adaptive learning rate를 만들고, Adam은 1차 모멘트와 2차 모멘트를 함께 사용한다.

## 9. 연습

1. GD와 SGD의 차이를 계산량과 noise 관점에서 설명하라.
2. Momentum이 좁은 골짜기에서 진동을 줄이는 이유를 설명하라.
3. RMSProp이 왜 좌표별 learning rate를 만든다고 볼 수 있는가?
4. Adam의 $m_t$, $v_t$가 각각 무엇을 추정하는지 설명하라.

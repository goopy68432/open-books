---
title: "12. GD / SGD / Momentum / Adam — 비교와 유도"
slug: gd-sgd-adam
order: 13
---

# 12. GD / SGD / Momentum / Adam — 비교와 유도

> **출제 근거**: 8주차 ★10 (Adam, GD Update Rule), ★9 (SGD), ★8 (Momentum, AdaGrad/RMSProp)
> **시험 출제 방식**: \"Compare GD, SGD, Momentum, and Adam. Write each update rule and explain the role of each term.\"

---

## 1. 왜 시험에 나오는가

- 8주차 핵심 알고리즘들. 정의+비교 형식의 출제 매우 유력.
- \"Adam = Momentum + RMSProp\" 결합 의미가 통합 시각의 정점.

---

## 2. Gradient Descent (GD)

### 2.1 동기

[08 Newton](08_Newton_2차근사.md) 의 \"$\theta_{t+1} = \theta_t - H^{-1}\mathbf{g}$\" 에서 $H \approx \frac{1}{\eta}I$ 라 가정 (모든 방향에서 곡률 같음):

$$
\boxed{\; \theta_{t+1} = \theta_t - \eta\, \mathbf{g}_t \;}
$$

| 기호 | 의미 |
|------|------|
| $\eta$ | learning rate (step size). 사람이 정함 |
| $\mathbf{g}_t = \nabla L(\theta_t)$ | 현재 gradient (전체 데이터로 계산) |

### 2.2 직관

- Gradient 의 **반대 방향** 이 가장 가파르게 내려가는 방향 (1차 근사)
- 그 방향으로 $\eta$ 만큼 이동

### 2.3 한계

- 모든 데이터에 대해 gradient 계산 → 느림
- 한 번 stuck 되면 빠져나올 momentum 없음
- 모든 파라미터에 같은 $\eta$

---

## 3. Stochastic Gradient Descent (SGD)

### 3.1 Update

매 step 마다 **mini-batch** $B_t \subset D$ 로 gradient 추정:

$$
\boxed{\; \theta_{t+1} = \theta_t - \eta\, \tilde{\mathbf{g}}_t, \quad \tilde{\mathbf{g}}_t = \frac{1}{|B_t|}\sum_{i \in B_t} \nabla \ell(h_\theta(x_i), y_i) \;}
$$

### 3.2 왜 SGD가 GD보다 좋은가 (8주차 ★9)

1. **속도**: 데이터 일부만 봄 → 한 step 이 빠름 → 같은 시간에 더 많은 update.
2. **Generalization**: gradient noise 가 \"sharp minimum\" 을 회피하고 \"flat minimum\" 을 선호 → test 성능 ↑.
3. **Memory**: 전체 데이터 동시 로드 불필요.

### 3.3 Learning Rate 큰 게 좋은 이유 (8주차 ★8)

- Noisy gradient 라 작은 $\eta$ 면 noise 에 휘말림
- 큰 $\eta$ 가 \"sharp minimum 회피\" 효과 더 큼

---

## 4. Momentum (Heavy-ball / NAG)

### 4.1 직관

\"굴러가는 공\" — 이전 방향으로의 관성 추가:

$$
\begin{aligned}
\mathbf{v}_{t+1} &= \beta\, \mathbf{v}_t + \mathbf{g}_t \\
\theta_{t+1} &= \theta_t - \eta\, \mathbf{v}_{t+1}
\end{aligned}
$$

| 기호 | 의미 |
|------|------|
| $\mathbf{v}$ | velocity (누적된 gradient) |
| $\beta$ | momentum coefficient (보통 0.9). 과거 비중 |

### 4.2 효과

- **Ravine 통과** 잘 함: gradient 가 noisy 한 방향은 진동 → 평균 0, 일관된 방향만 누적
- **Local minimum 탈출** 쉬움 (관성)

### 4.3 NAG (Nesterov)

\"미리 가본 위치\" 에서 gradient 계산:

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla L(\theta_t - \eta\beta \mathbf{v}_t)
$$

미세한 lookahead → 수렴 더 빠름.

---

## 5. AdaGrad / RMSProp

### 5.1 AdaGrad

각 파라미터마다 \"누적 gradient 제곱\" 으로 lr 조정:

$$
G_t = G_{t-1} + \mathbf{g}_t^2 \quad (\text{element-wise}), \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}}\odot \mathbf{g}_t
$$

\"많이 갱신된 파라미터는 lr ↓, 적게 갱신된 파라미터는 lr ↑\" → **per-parameter adaptive**.

문제: $G_t$ 가 무한히 누적 → lr 이 0 으로 감 → 학습 정지.

### 5.2 RMSProp

$G_t$ 대신 **지수가중이동평균 (EMA)** 사용:

$$
v_t = \rho\, v_{t-1} + (1-\rho)\, \mathbf{g}_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}}\odot \mathbf{g}_t
$$

\"최근 gradient 만 영향\" → 학습 정지 안 함.

---

## 6. Adam — Momentum + RMSProp

### 6.1 Update Rule

$$
\begin{aligned}
\mathbf{m}_t &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t & &\text{(1차 모멘트, momentum)} \\
\mathbf{v}_t &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2 & &\text{(2차 모멘트, RMSProp)} \\
\hat{\mathbf{m}}_t &= \mathbf{m}_t / (1-\beta_1^t) & &\text{(bias correction)} \\
\hat{\mathbf{v}}_t &= \mathbf{v}_t / (1-\beta_2^t) & &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\odot \hat{\mathbf{m}}_t
\end{aligned}
$$

### 6.2 \"왜 Bias Correction 인가?\"

$\mathbf{m}_0 = 0$ 으로 시작하면 초기에 $\mathbf{m}_t$ 가 0으로 편향됨. $1-\beta_1^t$ 로 나눠서 \"진짜 평균\"으로 보정.

### 6.3 \"왜 두 가지 결합인가?\" (★시험 답안)

| 부분 | 역할 |
|------|------|
| $\hat{\mathbf{m}}_t$ | **방향**: 과거 gradient 의 평균 (momentum) → 진동 줄임 |
| $\sqrt{\hat{\mathbf{v}}_t}$ | **크기**: 과거 gradient² 의 평균 → 파라미터별 적응형 lr |

> 🎯 Adam = \"방향은 momentum, 크기는 RMSProp\". Newton 의 dense Hessian 을 **diagonal 근사** + **EMA** 로 대체한 quasi-Newton 의 일종.

---

## 7. 비교표 (시험 답안용)

| 알고리즘 | Update | 사용 정보 | 특징 |
|---------|--------|----------|------|
| GD | $\theta_{t+1} = \theta_t - \eta \mathbf{g}_t$ | 전체 gradient | 느림, sharp min 빠지기 쉬움 |
| SGD | $\theta_{t+1} = \theta_t - \eta \tilde{\mathbf{g}}_t$ | mini-batch | 빠르고 일반화 좋음 |
| Momentum | $\theta_{t+1} = \theta_t - \eta \mathbf{v}_{t+1}$ | + 과거 방향 | 관성 |
| AdaGrad | $\theta_{t+1} = \theta_t - \eta/\sqrt{G_t}\cdot\mathbf{g}_t$ | + 누적 $\mathbf{g}^2$ | per-param lr, 학습 정지 |
| RMSProp | EMA of $\mathbf{g}^2$ | + EMA $\mathbf{g}^2$ | per-param lr, 정지 안 함 |
| Adam | Momentum + RMSProp + bias correction | 둘 다 | NN 의 default |

---

## 8. 모범 답안 템플릿

```
[GD]
θ_{t+1} = θ_t - η g_t            (g_t = ∇L(θ_t))
Each step uses the full-batch gradient and takes a fixed step.

[SGD]
θ_{t+1} = θ_t - η g̃_t            (g̃_t from mini-batch)
Cheaper per step; injects noise that helps escape sharp minima
and improves generalization.

[Momentum]
v_{t+1} = β v_t + g_t,   θ_{t+1} = θ_t - η v_{t+1}
Accumulates past gradients (β ≈ 0.9). Damps oscillations along
noisy directions and accelerates along consistent directions.

[Adam]
m_t = β₁ m_{t-1} + (1-β₁) g_t           (1st moment / momentum)
v_t = β₂ v_{t-1} + (1-β₂) g_t²          (2nd moment / RMSProp)
m̂_t = m_t / (1-β₁^t),  v̂_t = v_t/(1-β₂^t)   (bias correction)
θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)

Adam combines:
- Momentum-style direction smoothing (m̂_t),
- RMSProp-style per-parameter adaptive scaling (1/√v̂_t).
It can be viewed as a diagonal-Hessian approximation of Newton,
with EMAs replacing exact second-order information.
```

---

## 9. 자주 틀리는 함정

1. **GD vs SGD 차이를 \"gradient 안 쓰는 거\" 로 답함** — 둘 다 gradient 사용. 차이는 **batch size**.
2. **Adam 의 두 EMA 역할 혼동**: $\mathbf{m}$ 은 1차, $\mathbf{v}$ 는 2차 (gradient 제곱).
3. **Bias correction 필요성 누락**: 초기 step 에서 $\mathbf{m}, \mathbf{v}$ 가 0 으로 편향되는 점.
4. **\"Newton 과의 관계\" 누락**: Adam = diagonal quasi-Newton. 이 한 줄로 통합 시각 점수.

---

## 10. 연결 개념

- ← [08 Newton](08_Newton_2차근사.md): Adam 의 본질
- ← [09 Backprop](09_Backprop_ChainRule.md): gradient 공급원

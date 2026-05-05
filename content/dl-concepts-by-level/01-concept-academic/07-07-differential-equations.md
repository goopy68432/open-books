---
title: "07. 미분방정식 (Differential Equations)"
slug: 07-differential-equations
order: 7
---

# 07. 미분방정식 (Differential Equations)

## 1. 동기부여 및 개요

미분방정식은 **"변화의 법칙"을 수학적으로 기술하는 도구**이다. 딥러닝과의 연결은 놀라울 정도로 직접적이다:

- ResNet의 잔차 연결 $x_{l+1} = x_l + f_\theta(x_l)$은 **오일러 방법**으로 ODE를 이산화한 것
- **Neural ODE** (Chen et al., 2018)는 이산 레이어 대신 연속 동역학계를 직접 학습
- 로지스틱 ODE의 해가 정확히 **sigmoid 함수**이며, 포화(saturation) 현상의 수학적 기원을 설명

이 장에서 다루는 1차 자율 ODE $x' = f(x)$의 분류는 딥러닝의 활성화 함수, gradient 문제, 학습 동역학을 이해하는 데 필수적인 뼈대를 제공한다.

> **선수 지식**: 미적분(적분, 편미분), 지수/로그 함수, 기초 선형대수

---

## 2. 상미분방정식의 분류

### 2.1 기본 정의

**Definition 2.1 (상미분방정식, ODE).**
미지함수 $x(t)$와 그 도함수 $x' \equiv \frac{dx}{dt}$의 관계식을 **상미분방정식(Ordinary Differential Equation)**이라 한다.

이 장에서는 **1차 자율 ODE**: $x' = f(x)$ (우변이 $t$에 명시적으로 의존하지 않음)에 집중한다.

### 2.2 우변 구조에 따른 분류

**핵심 관찰**: $f(x)$가 다항식일 때, **근(root)의 개수가 해의 질적 행동을 결정**한다.

| ODE | 우변 구조 | 해의 유형 | 근의 수 |
|:---|:---|:---|:---:|
| $x' = a$ | 상수 | 선형: $x(t) = x_0 + at$ | 0 |
| $x' = ax$ | 1차 (선형) | 지수: $x(t) = x_0 e^{at}$ | 1 |
| $x' = ax + b$ | 1차 (아핀) | 이동 지수 | 1 |
| $x' = x^2$ | 2차 (단일근) | **쌍곡**: $x(t) = \frac{1}{1/x_0 - t}$ | 1 |
| $x' = ax(1-x)$ | 2차 (이중근) | **시그모이드** | 2 |

```
해의 행동 개요:

  x
  │     쌍곡(blow-up)
  │        /│
  │       / │   시그모이드(포화)
  │      /  │  ╱─────────── 1
  │     /   │╱
  │    /   ╱│
  │   /  ╱  │  지수(발산)
  │  / ╱   .│...............
  │ /╱     │
  ──────────┼──────────────→ t
            blow-up 시점
```

---

## 3. 변수분리법 (Separation of Variables)

### 3.1 방법

자율 ODE $x' = f(x)$에서 $f(x) \neq 0$인 영역에서:

$$\frac{dx}{f(x)} = dt \implies \int \frac{dx}{f(x)} = t + C$$

초기조건 $x(0) = x_0$으로 상수 $C$를 결정한다.

### 3.2 변수분리법이 작동하는 이유

$F(x) = \int \frac{dx}{f(x)}$가 $x$의 단조함수이므로 역함수 $F^{-1}$이 존재한다:

$$F(x) = t + C \implies x(t) = F^{-1}(t + C)$$

> **주의**: $f(x^*) = 0$인 **고정점**에서는 분리 불가. 이 점은 별도의 **상수 해(equilibrium solution)** $x(t) \equiv x^*$를 형성한다.

---

## 4. 선형 ODE와 지수 해

### 4.1 $x' = ax$의 풀이

$$\frac{dx}{x} = a\,dt \implies \ln|x| = at + C \implies \boxed{x(t) = x_0 e^{at}}$$

- $a > 0$: 지수적 성장 (폭발)
- $a < 0$: 지수적 감쇠 (0으로 수렴)
- $a = 0$: 상수 해

### 4.2 $x' = ax + b$의 풀이

$$\boxed{x(t) = \left(x_0 + \frac{b}{a}\right)e^{at} - \frac{b}{a}}$$

**평형점**: $x^* = -b/a$에서 $x' = 0$. 편차 $y = x - x^*$에 대해 $y' = ay$이므로:
- $a < 0$: 모든 해가 $x^*$로 수렴 (**안정**)
- $a > 0$: 모든 해가 $x^*$에서 발산 (**불안정**)

### 4.3 Gradient Flow와의 연결

이차 손실함수 $L(\theta) = \frac{a}{2}\theta^2$에서 gradient flow:

$$\dot{\theta} = -\nabla L(\theta) = -a\theta$$

해는 $\theta(t) = \theta_0 e^{-at}$로, $a > 0$이면 **지수적 수렴**이다. 이것이 gradient descent의 연속 시간 버전이며, 학습률과 Hessian 고유값의 관계가 수렴 속도를 결정한다.

### 4.4 다변수 확장

$\dot{X} = AX$의 해는 $X(t) = \exp(At)X(0)$이며, 행렬 지수함수 $\exp(At) = \sum_{k=0}^{\infty}\frac{(At)^k}{k!}$이다. 고유값 $\lambda_i$의 실수부가 모두 음수면 원점이 전역 안정이다.

---

## 5. 쌍곡 성장과 유한시간 폭발 (Finite-Time Blow-up)

### 5.1 $x' = x^2$의 풀이

$$\frac{dx}{x^2} = dt \implies -\frac{1}{x} = t + C \implies \boxed{x(t) = \frac{1}{1/x_0 - t}}$$

**Theorem 5.1 (유한시간 폭발).**
$x_0 > 0$이면, $t^* = 1/x_0$에서 $x(t) \to \infty$. 지수 성장($t \to \infty$에서 발산)과 달리, 쌍곡 성장은 **유한 시간**에 무한대에 도달한다.

### 5.2 $x' = (x - \alpha)^2$의 풀이

$$\boxed{x(t) = \frac{1}{1/(x_0 - \alpha) - t} + \alpha}$$

평형점 $x^* = \alpha$는 **반안정(semi-stable)**:
- $x_0 > \alpha$: 유한시간에 $+\infty$로 blow-up
- $x_0 < \alpha$: $t \to \infty$에서 $\alpha$에 수렴

### 5.3 딥러닝에서의 의미

- **Gradient explosion**: RNN에서 야코비안 $\|W\| > 1$일 때 기울기가 폭발하는 현상. 비선형 항이 있으면 쌍곡적 폭발도 가능
- **Neural ODE의 수치적 문제**: blow-up 근처에서 ODE solver의 스텝 크기가 극도로 줄어들어 계산 비용 급증. FFJORD (Grathwohl et al., 2019)에서 Lipschitz 조건을 부과하는 이유

---

## 6. 로지스틱 방정식과 시그모이드 유도

### 6.1 로지스틱 ODE

$$x' = ax(1 - x)$$

- $x \approx 0$일 때: $x' \approx ax$ (지수적 성장)
- $x \approx 1$일 때: $x' \approx 0$ (포화, 성장 멈춤)
- 평형점: $x^* = 0$ (불안정), $x^* = 1$ (안정)

### 6.2 풀이: 부분분수 분해

$$\frac{dx}{x(1-x)} = a\,dt$$

부분분수: $\frac{1}{x(1-x)} = \frac{1}{x} + \frac{1}{1-x}$

적분 후 정리하면:

$$\boxed{x(t) = \frac{1}{1 + \frac{1-x_0}{x_0}e^{-at}}}$$

### 6.3 시그모이드 함수와의 연결

$x_0 \ll 1$일 때, $\frac{1-x_0}{x_0} \approx \frac{1}{x_0}$이므로:

$$x(t) \approx \frac{1}{1 + e^{-(at - \ln(1/x_0))}} = \sigma\!\left(at - \ln\frac{1}{x_0}\right)$$

여기서 $\sigma(z) = \frac{1}{1 + e^{-z}}$은 **표준 시그모이드 함수**이다.

> **Theorem 6.1.** 로지스틱 ODE $x' = ax(1-x)$의 해는 시그모이드 함수이다. 시그모이드는 "포화가 있는 성장"이라는 동역학적 원리에서 자연스럽게 유도된다.

### 6.4 반감기 (Half-Time)

$x = 0.5$가 되는 시간:

$$t_{1/2} = \frac{\ln(1/x_0 - 1)}{a} \approx \frac{\ln(1/x_0)}{a}$$

$x_0$이 작을수록, $a$가 작을수록 반감기가 길어진다.

### 6.5 딥러닝에서의 의미

1. **활성화 함수의 동역학적 해석**: sigmoid 뉴런 $\sigma(Wx + b)$는 로지스틱 벡터장의 time-1 flow로 해석 가능
2. **Vanishing gradient의 동역학적 설명**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$이므로 포화 영역($\sigma \approx 0$ 또는 $\sigma \approx 1$)에서 기울기가 소멸 --- ReLU, GELU 등 대안 활성화 함수 개발의 동기
3. **Softmax의 연속 동역학**: 다변수 로지스틱 시스템의 적절한 결합은 replicator dynamics가 되며, 그 해가 softmax

---

## 7. 일반 이중근 이차 ODE

### 7.1 $x' = a(x - \alpha)(x - \beta)$, $\alpha \neq \beta$

부분분수 분해와 적분을 통해:

$$\boxed{x(t) = \frac{\alpha - \beta}{1 - C\exp((\alpha - \beta)at)} + \beta}$$

이는 **일반화된 시그모이드(generalized sigmoid)**이며, $\alpha, \beta$가 각각 하한과 상한(평형점), $a$가 전이 속도, $x_0$가 전이 시점을 결정한다.

### 7.2 수용량 모델: $x' = ax(\beta - x)$

$$x(t) = \frac{\beta}{1 + C'e^{-a\beta t}}$$

carrying capacity가 $\beta$인 로지스틱 성장이다. 반감기는 $\propto 1/\beta$로 수용량에 반비례한다.

> **흥미로운 관찰**: 딥러닝의 학습 곡선이 시그모이드를 따르는 이유 --- 학습 초기에는 쉬운 패턴을 빠르게 학습(지수적 성장)하고, 후기에는 어려운 패턴에서 포화(한계 도달)하기 때문이다.

---

## 8. 안정성 이론과 위상 평면 (Phase Portrait)

### 8.1 1차원 안정성

1차원 자율 ODE $x' = f(x)$에서 고정점 $x^*$ ($f(x^*) = 0$)의 안정성:

- $f'(x^*) < 0$: **안정** (attracting) --- 작은 섭동이 감쇠
- $f'(x^*) > 0$: **불안정** (repelling) --- 작은 섭동이 증폭
- $f'(x^*) = 0$: 고차 분석 필요

### 8.2 2차원 선형 시스템: $\dot{X} = AX$

행렬 $A$의 trace와 determinant로 위상 초상을 분류한다:

| 영역 | 고유값 | 위상 초상 |
|:---|:---|:---|
| $\det A < 0$ | 실수, 반대 부호 | **안장점 (Saddle)** |
| $\det A > 0$, $\Delta > 0$, $\text{tr} A < 0$ | 음의 실수 | **안정 노드 (Stable Node)** |
| $\det A > 0$, $\Delta > 0$, $\text{tr} A > 0$ | 양의 실수 | **불안정 노드 (Unstable Node)** |
| $\det A > 0$, $\Delta < 0$, $\text{tr} A < 0$ | $\alpha \pm i\beta$, $\alpha < 0$ | **안정 나선 (Stable Spiral)** |
| $\det A > 0$, $\Delta < 0$, $\text{tr} A > 0$ | $\alpha \pm i\beta$, $\alpha > 0$ | **불안정 나선 (Unstable Spiral)** |
| $\det A > 0$, $\text{tr} A = 0$ | 순허수 $\pm i\beta$ | **Center** (중립 안정) |

여기서 $\Delta = (\text{tr}\,A)^2 - 4\,\text{det}\,A$ (판별식).

### 8.3 비선형 시스템의 국소 분석

비선형 시스템 $\dot{X} = F(X)$에서 고정점 $X^*$ 근방의 행동은 **야코비안** $A = F'(X^*)$로 결정된다 (Hartman-Grobman 정리).

### 8.4 딥러닝에서의 안정성

1. **학습 안정성**: SGD를 $\dot{\theta} = -\nabla L(\theta) + \xi(t)$ (Langevin 동역학)으로 모델링. 극소점 = 안정 고정점, 안장점 = saddle
2. **Batch Normalization의 효과**: 학습 동역학의 고유값 스펙트럼을 조절하여 불안정한 spiral source를 stable node로 변환
3. **Edge of Chaos**: 심층 네트워크에서 야코비안 고유값이 단위원 근처에 있을 때(center 근방) 가장 풍부한 표현력

---

## 9. 흔한 오해와 주의점

| 오해 | 실제 | 교정 |
|:---|:---|:---|
| $x' = x^2$도 지수 성장과 비슷할 것 | 지수 성장은 $t \to \infty$에서 발산, 쌍곡은 **유한 시간**에 발산. 질적으로 완전히 다름 | 두 ODE를 수치적으로 풀어 그래프 비교 |
| 시그모이드는 누군가 임의로 설계 | 로지스틱 ODE $x' = ax(1-x)$의 **자연스러운 해** | ODE → 부분분수 → 적분 → sigmoid 유도 직접 수행 |
| 평형점은 항상 안정 | 안정/불안정/반안정 중 하나 | $f'(x^*)$의 부호를 계산하여 안정성 판별 |
| "선형 ODE"의 해는 직선 | "선형"은 $x$에 대한 것이지, 해가 아님. $x' = ax$의 해는 **지수함수** | $x' = a$ (직선 해)와 $x' = ax$ (지수 해) 명확 구분 |
| ODE를 풀면 항상 닫힌 형태의 해 존재 | 대부분의 ODE는 닫힌 해가 없음 | $x' = \sin(x^2)$ 같은 단순 ODE도 적분이 닫힌 형태로 불가 |
| $x' = ax(1-x)$에서 $a$가 크면 최종값도 큼 | $a$는 전이 **속도**만 결정, 상한은 항상 1 | $a$ 변화에 따른 시그모이드 기울기 비교 |

---

## 10. 수학 --- 딥러닝 연결 요약

| 수학 개념 | 딥러닝 대응 |
|:---|:---|
| $x' = ax$ (지수 해) | Gradient flow: 이차 손실에서 파라미터 수렴/발산 |
| $x' = ax(1-x)$ (시그모이드 해) | sigmoid 활성화 함수의 수학적 기원 |
| $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | Vanishing gradient 현상의 근본 원인 |
| $x' = x^2$ (유한시간 폭발) | Gradient explosion, 수치적 불안정 |
| $\dot{X} = AX$ | Neural ODE: ResNet의 연속 한계 |
| 오일러 방법: $x_{n+1} = x_n + hf(x_n)$ | ResNet: $x_{l+1} = x_l + f_\theta(x_l)$ (skip connection) |
| 고정점 안정성 | 손실 landscape의 극소점/안장점 분류 |

---

## 11. 핵심 요약

**핵심 공식 3개**:

$$\boxed{x' = ax \;\Rightarrow\; x(t) = x_0 e^{at}} \quad \text{(지수 성장/감쇠)}$$

$$\boxed{x' = x^2 \;\Rightarrow\; x(t) = \frac{1}{1/x_0 - t}} \quad \text{(쌍곡, blow-up at } t = 1/x_0\text{)}$$

$$\boxed{x' = ax(1-x) \;\Rightarrow\; x(t) = \sigma(at + \text{const})} \quad \text{(시그모이드)}$$

**기억 흐름**:
```
상수 → 선형 → 지수 → 쌍곡 → 시그모이드 → 안정성
 x'=a   x'=ax   e^{at}  1/(1/x₀-t)   σ(z)      Poincare
 직선    곡선    발산     폭발         포화       분류
```

**한 문장 요약**: 1차 자율 ODE에서 우변이 선형이면 **지수**, 이차 단일근이면 **쌍곡(blow-up)**, 이차 이중근이면 **시그모이드**이고, 이 분류가 딥러닝의 활성화 함수, gradient 문제, 학습 동역학의 수학적 뼈대를 이룬다.

---

## 12. 수치적 방법: 오일러 방법과 ResNet

### 12.1 오일러 방법 (Euler Method)

ODE $x' = f(x)$의 닫힌 해를 구할 수 없을 때, 수치적으로 근사한다:

$$x_{n+1} = x_n + h \cdot f(x_n)$$

여기서 $h$는 **스텝 크기(step size)**이다.

### 12.2 ResNet과의 동치성

ResNet의 잔차 블록:

$$x_{l+1} = x_l + f_\theta(x_l)$$

이것은 스텝 크기 $h = 1$인 **오일러 방법**과 정확히 동일한 구조이다!

| ODE 수치 해법 | 신경망 구조 |
|:---|:---|
| 오일러 방법 $x_{n+1} = x_n + hf(x_n)$ | ResNet의 skip connection |
| 적응적 스텝 크기 | 학습 가능한 스텝 크기 |
| 고차 방법 (Runge-Kutta) | 더 정교한 블록 설계 |
| ODE solver | Neural ODE (Chen et al., 2018) |

### 12.3 Neural ODE

Neural ODE는 이산 레이어 대신 연속 동역학을 직접 학습한다:

$$\frac{dx}{dt} = f_\theta(x(t), t)$$

장점:
- **메모리 효율**: adjoint method로 역전파 시 중간 활성값 저장 불필요
- **적응적 계산**: 입력 난이도에 따라 계산량 자동 조절
- **연속 정규화 흐름**: FFJORD를 통한 밀도 추정

### 12.4 Python 의사코드: ODE 해의 비교

```python
import numpy as np

def euler_solve(f, x0, t_span, h=0.01):
    """오일러 방법으로 x' = f(x) 풀기"""
    t = np.arange(t_span[0], t_span[1], h)
    x = np.zeros_like(t)
    x[0] = x0
    for i in range(len(t) - 1):
        x[i+1] = x[i] + h * f(x[i])
    return t, x

# 세 가지 ODE 비교
f_exp = lambda x: 0.5 * x          # 지수 성장
f_hyp = lambda x: x**2             # 쌍곡 (blow-up)
f_sig = lambda x: 0.5 * x * (1-x)  # 시그모이드

# x0 = 0.1로 시작했을 때
# f_exp: 완만한 지수 성장
# f_hyp: t=10 근처에서 수직 점근선 (blow-up)
# f_sig: S자 곡선으로 1에 수렴
```

---

## 참고 문헌

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS*.
- Grathwohl, W. et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*. Westview Press.
- Kaplan, J. et al. (2020). Scaling Laws for Neural Language Models.

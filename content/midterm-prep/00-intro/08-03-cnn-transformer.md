---
title: "딥러닝 중간고사 종합 학습자료 (Part 3)"
slug: 03-cnn-transformer
order: 8
---

# 딥러닝 중간고사 종합 학습자료 (Part 3)
## CNN, Transformer, 생성 모델
### 한양대학교 이성윤 교수님 | Deep Learning

> **시험 형식**: 영어, 수학적 유도 과정 및 논리적 추론 중심
> **이 자료의 범위**: CNN (Part 8) → Sequence/Transformer (Part 9) → Generative Models (Part 10) → 공식 총정리 (Part 11)

---

# Part 8: CNN (합성곱 신경망, Convolutional Neural Networks)

---

## 8.1 합성곱 연산 (Convolution Operation)

### 8.1.1 수학적 정의

**연속 합성곱 (Continuous Convolution)**:

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau
$$

**이산 합성곱 (Discrete Convolution)**:

$$
(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] \, g[n - m]
$$

**2D 합성곱 (이미지에 적용)**:

$$
(I * K)[i, j] = \sum_{m} \sum_{n} I[i+m, \, j+n] \cdot K[m, n]
$$

여기서 $I$는 입력 이미지, $K$는 커널(필터)이다.

> **시험 포인트**: 엄밀히 말하면 딥러닝에서 사용하는 연산은 convolution이 아니라 **cross-correlation**이다. 진짜 convolution은 커널을 flip한 후 적용한다. 그러나 학습 과정에서 커널 가중치가 자동으로 학습되므로 flip 여부는 무관하다.

### 8.1.2 구체적 계산 예시

입력 $I$ (3×3), 커널 $K$ (2×2):

$$
I = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}, \quad
K = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
$$

출력 (stride=1, padding=0):

$$
O[0,0] = 1 \cdot 1 + 2 \cdot 0 + 4 \cdot 0 + 5 \cdot (-1) = 1 - 5 = -4
$$
$$
O[0,1] = 2 \cdot 1 + 3 \cdot 0 + 5 \cdot 0 + 6 \cdot (-1) = 2 - 6 = -4
$$
$$
O[1,0] = 4 \cdot 1 + 5 \cdot 0 + 7 \cdot 0 + 8 \cdot (-1) = 4 - 8 = -4
$$
$$
O[1,1] = 5 \cdot 1 + 6 \cdot 0 + 8 \cdot 0 + 9 \cdot (-1) = 5 - 9 = -4
$$

$$
O = \begin{bmatrix} -4 & -4 \\ -4 & -4 \end{bmatrix}
$$

---

## 8.2 CNN의 귀납적 편향 (Inductive Bias)

CNN이 MLP보다 이미지 처리에 효과적인 이유는 세 가지 핵심 귀납적 편향 때문이다.

### 8.2.1 지역성 (Locality / Local Connectivity)

- 각 뉴런은 입력의 **작은 지역 영역(local receptive field)**에만 연결
- MLP는 모든 입력 픽셀과 연결 → $O(H \times W \times C)$ 파라미터/뉴런
- CNN은 커널 크기만큼만 연결 → $O(k^2 \times C)$ 파라미터/뉴런
- **가정**: 인접한 픽셀들이 서로 연관성이 높다 (이미지의 자연스러운 특성)

### 8.2.2 가중치 공유 (Weight Sharing / Parameter Sharing)

- 동일한 커널(필터)이 입력의 **모든 위치**에서 동일하게 적용
- 파라미터 수 대폭 감소: 위치마다 다른 가중치를 쓸 필요 없음
- 동일한 특징(엣지, 텍스처 등)을 이미지 어디에서든 검출 가능

**비교**:
| | MLP (FC) | CNN |
|---|---|---|
| 32×32×3 입력 → 1000 출력 | 3,072,000 파라미터 | 커널 크기에 따라 수천 개 |

### 8.2.3 이동 등변성 (Translation Equivariance)

**정의**: 입력을 이동(shift)하면 출력도 동일하게 이동한다.

$$
\text{Conv}(\text{Shift}(x)) = \text{Shift}(\text{Conv}(x))
$$

수학적으로, 함수 $f$가 이동 등변이면:

$$
f(T_\Delta(x)) = T_\Delta(f(x))
$$

여기서 $T_\Delta$는 이동 연산자이다.

> **주의: 등변성(Equivariance) vs 불변성(Invariance)**
> - **Equivariance**: 입력 변환 → 출력도 같은 변환 (Convolution)
> - **Invariance**: 입력 변환 → 출력 불변 (Pooling이 근사적으로 제공)

---

## 8.3 출력 크기 공식 (Output Size Formula)

### 8.3.1 기본 공식

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1
$$

여기서:
- $H_{in}, W_{in}$: 입력의 높이, 너비
- $k$: 커널 크기
- $p$: 패딩 (padding)
- $s$: 스트라이드 (stride)

### 8.3.2 유도 과정

1. 패딩 적용 후 유효 크기: $H_{in} + 2p$
2. 커널이 차지하는 크기를 빼면 이동 가능 거리: $H_{in} + 2p - k$
3. 스트라이드 $s$씩 이동하면 가능한 위치 수: $\frac{H_{in} + 2p - k}{s}$
4. 시작 위치 포함이므로 +1: $\left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$

### 8.3.3 특수 경우

**"same" 패딩** (출력 크기 = 입력 크기, stride=1):

$$
H_{out} = H_{in} \implies p = \frac{k-1}{2}
$$

따라서 $k=3 \Rightarrow p=1$, $k=5 \Rightarrow p=2$, $k=7 \Rightarrow p=3$

### 8.3.4 연습 문제

**Q**: 입력 $224 \times 224$, 커널 $7 \times 7$, stride=2, padding=3일 때 출력 크기는?

**A**:
$$
H_{out} = \left\lfloor \frac{224 + 2(3) - 7}{2} \right\rfloor + 1 = \left\lfloor \frac{223}{2} \right\rfloor + 1 = 111 + 1 = 112
$$

출력: $112 \times 112$ (이것이 바로 ResNet 첫 번째 conv layer의 출력!)

---

## 8.4 파라미터 수 계산 (Parameter Count)

### 8.4.1 Conv2d 파라미터 수 공식

$$
\text{Params} = C_{out} \times C_{in} \times k \times k + C_{out}
$$

- $C_{out} \times C_{in} \times k^2$: 가중치 (weights)
- $C_{out}$: 편향 (bias), 각 출력 채널당 하나

### 8.4.2 텐서 차원 분석

| 요소 | 차원 |
|---|---|
| 입력 텐서 | $(N, C_{in}, H_{in}, W_{in})$ |
| 커널 (필터 뱅크) | $(C_{out}, C_{in}, k, k)$ |
| 편향 | $(C_{out},)$ |
| 출력 텐서 | $(N, C_{out}, H_{out}, W_{out})$ |

### 8.4.3 연산량 (FLOPs)

하나의 출력 위치에서의 곱셈: $C_{in} \times k^2$

전체 FLOPs (곱셈만):

$$
\text{FLOPs} = C_{out} \times C_{in} \times k^2 \times H_{out} \times W_{out}
$$

### 8.4.4 구체적 예시

**VGG의 한 레이어**: $\text{Conv}(256, 512, k=3, p=1)$, 입력 $28 \times 28$

- 파라미터: $512 \times 256 \times 3 \times 3 + 512 = 1,180,160$
- 출력 크기: $28 \times 28$ (same padding)
- FLOPs: $512 \times 256 \times 9 \times 28 \times 28 \approx 924M$

---

## 8.5 풀링 (Pooling)

### 8.5.1 Max Pooling

각 풀링 윈도우에서 **최대값**을 선택:

$$
y_{ij} = \max_{(m,n) \in R_{ij}} x_{mn}
$$

- 가장 강한 활성화를 보존
- 약간의 이동 불변성(translation invariance) 제공
- **파라미터 없음** (학습할 것이 없다)

### 8.5.2 Average Pooling

각 풀링 윈도우의 **평균값**:

$$
y_{ij} = \frac{1}{|R_{ij}|} \sum_{(m,n) \in R_{ij}} x_{mn}
$$

### 8.5.3 Global Average Pooling (GAP)

전체 feature map을 하나의 값으로 축소:

$$
y_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{c,i,j}
$$

- 입력: $(C, H, W)$ → 출력: $(C, 1, 1)$ = $(C,)$
- FC layer 대체 → 파라미터 수 대폭 감소
- 과적합(overfitting) 방지에 효과적
- **NIN (Network in Network)** 에서 처음 제안, 이후 GoogLeNet, ResNet 등에서 표준으로 사용

### 8.5.4 풀링 출력 크기

풀링에도 같은 출력 크기 공식이 적용된다:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

일반적으로 $k = s = 2, p = 0$ → 크기 절반으로 축소

---

## 8.6 수용 영역 (Receptive Field)

### 8.6.1 정의

출력의 한 뉴런이 "볼 수 있는" 입력 영역의 크기.

### 8.6.2 계산 공식

$L$개의 레이어를 거친 후의 수용 영역 $r_L$:

$$
r_L = r_{L-1} + (k_L - 1) \times \prod_{i=1}^{L-1} s_i
$$

또는 초기값 $r_0 = 1$에서 시작하여 반복 계산:

$$
r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1}
$$

$$
j_l = j_{l-1} \cdot s_l
$$

여기서 $j_l$은 레이어 $l$에서의 **jump** (입력 공간에서의 한 칸 이동 크기).

### 8.6.3 구체적 계산 예시

**3개의 3×3 conv (stride=1) 스택**:

| Layer | $k$ | $s$ | $j$ | $r$ |
|---|---|---|---|---|
| 입력 | - | - | 1 | 1 |
| Conv1 (3×3) | 3 | 1 | 1 | $1 + (3-1) \times 1 = 3$ |
| Conv2 (3×3) | 3 | 1 | 1 | $3 + (3-1) \times 1 = 5$ |
| Conv3 (3×3) | 3 | 1 | 1 | $5 + (3-1) \times 1 = 7$ |

**결론**: 3×3 conv 3개 = 7×7 수용 영역 (이것이 VGG의 핵심 아이디어!)

**Pooling 포함 예시** (Conv 3×3, s=1 → Pool 2×2, s=2 → Conv 3×3, s=1):

| Layer | $k$ | $s$ | $j$ | $r$ |
|---|---|---|---|---|
| 입력 | - | - | 1 | 1 |
| Conv1 (3×3, s=1) | 3 | 1 | 1 | 3 |
| Pool (2×2, s=2) | 2 | 2 | 2 | $3 + (2-1) \times 1 = 4$ |
| Conv2 (3×3, s=1) | 3 | 1 | 2 | $4 + (3-1) \times 2 = 8$ |

---

## 8.7 1×1 Convolution

### 8.7.1 역할

1. **채널 차원의 선형 결합**: 각 공간 위치에서 채널 간 정보를 혼합
2. **차원 축소/확장 (Bottleneck)**: 채널 수를 줄여 계산량 감소
3. **비선형성 추가**: 활성화 함수와 결합하면 채널별 비선형 변환
4. **Cross-channel interaction**: 공간 정보는 유지하면서 채널 간 상호작용

### 8.7.2 파라미터 및 연산량

$$
\text{Params}_{1 \times 1} = C_{out} \times C_{in} + C_{out}
$$

### 8.7.3 Bottleneck 구조 예시 (ResNet)

직접 3×3 conv: $(256, 256, 3, 3)$ → 파라미터: $256 \times 256 \times 9 = 589,824$

Bottleneck (1×1 → 3×3 → 1×1):
- 1×1 conv: $(256 \to 64)$: $256 \times 64 = 16,384$
- 3×3 conv: $(64 \to 64)$: $64 \times 64 \times 9 = 36,864$
- 1×1 conv: $(64 \to 256)$: $64 \times 256 = 16,384$
- **합계: 69,632** (약 **8.5배 감소!**)

---

## 8.8 주요 아키텍처 (Key Architectures)

### 8.8.1 AlexNet (2012, Krizhevsky et al.)

**구조**: 5 Conv + 3 FC layers

**핵심 기여**:
- GPU 학습의 시작 (2개 GPU 병렬)
- ReLU 활성화 함수 사용 (sigmoid/tanh 대체)
- Dropout (FC layers에 적용, $p=0.5$)
- Data augmentation (flip, crop, color jittering)
- Local Response Normalization (LRN) — 이후 BN으로 대체됨

**파라미터**: ~60M (대부분 FC layers에 집중)

### 8.8.2 VGG (2014, Simonyan & Zisserman)

**핵심 철학**: **오직 3×3 커널만 사용**

**왜 3×3인가?**

| 비교 | 하나의 7×7 | 세 개의 3×3 |
|---|---|---|
| 수용 영역 | 7×7 | 7×7 (동일!) |
| 파라미터 ($C$ 채널) | $49C^2$ | $3 \times 9C^2 = 27C^2$ |
| 비선형성 | 1번 | 3번 (더 많은 표현력) |

**증명**:
- 3×3 conv 1개: 수용 영역 3
- 3×3 conv 2개: 수용 영역 5
- 3×3 conv 3개: 수용 영역 7 = 7×7 conv 1개와 동일

파라미터 비교:
$$
\frac{3 \times (3^2 C^2)}{7^2 C^2} = \frac{27}{49} \approx 0.55
$$

→ **약 45% 파라미터 절감 + 비선형성 증가**

**VGG-16 구조**: [64]×2 → [128]×2 → [256]×3 → [512]×3 → [512]×3 → FC

### 8.8.3 ResNet (2015, He et al.)

**문제**: 네트워크가 깊어지면 학습이 어려워진다 (degradation problem).

> 이것은 단순한 overfitting이 아니다. **training error도 증가**한다.

**해결: Residual Connection (Skip Connection)**

$$
H(x) = F(x) + x
$$

여기서:
- $H(x)$: 원하는 매핑 (desired mapping)
- $F(x) = H(x) - x$: 잔차 (residual), 네트워크가 학습하는 것
- $x$: 항등 매핑 (identity mapping), skip connection

**왜 효과적인가?**

1. **그래디언트 흐름 개선**:
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \frac{\partial H}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left(\frac{\partial F}{\partial x} + 1\right)
$$

$+1$ 항이 있으므로 그래디언트가 최소 1 이상 → **vanishing gradient 완화**

2. **항등 함수 학습이 쉬움**: $F(x) = 0$을 학습하면 $H(x) = x$ (항등 매핑)
   - 추가 레이어가 최소한 손해는 보지 않음

3. **앙상블 관점**: ResNet은 다양한 길이의 경로들의 앙상블로 해석 가능

**차원 불일치 시** ($x$와 $F(x)$의 채널 수가 다를 때):

$$
H(x) = F(x) + W_s \cdot x
$$

$W_s$: 1×1 convolution으로 차원 맞춤 (projection shortcut)

**ResNet Bottleneck Block**:

```
x → [1×1, 64] → [3×3, 64] → [1×1, 256] → (+x) → out
     (reduce)    (process)    (expand)
```

---

## 8.9 Batch Normalization (BN)

### 8.9.1 동기

- **Internal Covariate Shift**: 각 레이어의 입력 분포가 학습 중 계속 변함
- 높은 learning rate 사용 어려움
- 초기화에 민감

### 8.9.2 Forward Pass 공식

미니배치 $\mathcal{B} = \{x_1, \ldots, x_m\}$에 대해:

**Step 1**: 배치 평균

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

**Step 2**: 배치 분산

$$
\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

**Step 3**: 정규화

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

**Step 4**: Scale & Shift (학습 가능 파라미터 $\gamma, \beta$)

$$
y_i = \gamma \hat{x}_i + \beta
$$

### 8.9.3 학습 가능 파라미터 $\gamma$, $\beta$의 역할

- $\gamma = \sigma_{\mathcal{B}}, \, \beta = \mu_{\mathcal{B}}$로 설정하면 정규화 취소 가능
- 네트워크가 **정규화의 정도를 스스로 학습**할 수 있음
- 표현력(representational power) 유지

### 8.9.4 추론(Inference) 시

학습 중 **이동 평균(running average)**으로 전체 데이터셋의 통계를 추적:

$$
\mu_{running} \leftarrow (1 - \alpha) \cdot \mu_{running} + \alpha \cdot \mu_{\mathcal{B}}
$$

$$
\sigma^2_{running} \leftarrow (1 - \alpha) \cdot \sigma^2_{running} + \alpha \cdot \sigma^2_{\mathcal{B}}
$$

추론 시에는 running statistics를 사용 (배치에 의존하지 않음).

### 8.9.5 CNN에서의 BN

- 채널별로 정규화: 각 채널 $c$에 대해 $(N, H, W)$ 차원에서 평균/분산 계산
- 학습 파라미터: $2 \times C$ ($\gamma$와 $\beta$ 각각 $C$개)

> **시험 포인트**: BN은 활성화 함수 **전**에 적용하는 것이 원래 논문의 제안이지만, 실무에서는 **후**에 적용하기도 한다. 시험에서는 원래 제안 (Conv → BN → ReLU)을 기준으로.

---

# Part 9: 시퀀스 모델과 트랜스포머 (Sequence Models and Transformer)

---

## 9.1 RNN (Recurrent Neural Network)

### 9.1.1 기본 공식

$$
h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

또는 concat 표기법:

$$
h_t = \phi\left(W \begin{bmatrix} x_t \\ h_{t-1} \end{bmatrix} + b\right)
$$

여기서:
- $h_t \in \mathbb{R}^d$: 시점 $t$의 은닉 상태 (hidden state)
- $x_t \in \mathbb{R}^n$: 시점 $t$의 입력
- $W_{hh} \in \mathbb{R}^{d \times d}$: hidden-to-hidden 가중치
- $W_{xh} \in \mathbb{R}^{d \times n}$: input-to-hidden 가중치
- $\phi$: 활성화 함수 (보통 $\tanh$)

**출력**:

$$
y_t = W_{hy} h_t + b_y
$$

### 9.1.2 BPTT (Backpropagation Through Time)

손실 함수가 $\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$일 때:

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_{hh}}
$$

체인 룰을 적용하면:

$$
\frac{\partial \mathcal{L}_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}
$$

핵심은 $\frac{\partial h_t}{\partial h_k}$인데, 이것은 연쇄적 곱:

$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}
$$

---

## 9.2 RNN의 기울기 소실 문제 (Vanishing Gradient)

### 9.2.1 수학적 분석

$h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t + b)$ 에서:

$$
\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\phi'(z_t)) \cdot W_{hh}
$$

여기서 $z_t = W_{hh} h_{t-1} + W_{xh} x_t + b$.

따라서:

$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \text{diag}(\phi'(z_i)) \cdot W_{hh}
$$

### 9.2.2 Norm 분석

$$
\left\| \frac{\partial h_t}{\partial h_k} \right\| \leq \prod_{i=k+1}^{t} \| \text{diag}(\phi'(z_i)) \| \cdot \| W_{hh} \|
$$

$\tanh$의 도함수: $\phi'(z) = 1 - \tanh^2(z) \in (0, 1]$

따라서 $\| \text{diag}(\phi'(z_i)) \| \leq 1$이고:

$$
\left\| \frac{\partial h_t}{\partial h_k} \right\| \leq \| W_{hh} \|^{t-k}
$$

- $\| W_{hh} \| < 1$ → $\| W_{hh} \|^{t-k} \to 0$ as $t - k \to \infty$ → **기울기 소실 (Vanishing)**
- $\| W_{hh} \| > 1$ → $\| W_{hh} \|^{t-k} \to \infty$ → **기울기 폭발 (Exploding)**

### 9.2.3 결론

- 긴 시퀀스에서 **장기 의존성(long-range dependency)**을 학습하기 어려움
- 기울기 폭발: gradient clipping으로 완화 가능
- 기울기 소실: 구조적 해결 필요 → **LSTM, GRU**

---

## 9.3 LSTM (Long Short-Term Memory)

### 9.3.1 핵심 아이디어

별도의 **셀 상태(cell state)** $c_t$를 도입하여 정보를 보존하는 "고속도로(highway)"를 만든다.

### 9.3.2 게이트 공식 (모든 게이트)

**Forget Gate** (무엇을 잊을지):

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

**Input Gate** (무엇을 기억할지):

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$

**후보 셀 상태** (새로운 정보):

$$
\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$

**Cell State 업데이트**:

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**Output Gate** (무엇을 출력할지):

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$

**Hidden State**:

$$
h_t = o_t \odot \tanh(c_t)
$$

여기서 $\odot$은 원소별 곱(element-wise multiplication), $\sigma$는 sigmoid 함수.

### 9.3.3 LSTM이 Vanishing Gradient를 해결하는 이유

셀 상태의 그래디언트:

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

Forget gate $f_t \approx 1$이면:

$$
\frac{\partial c_T}{\partial c_k} = \prod_{i=k+1}^{T} f_i \approx 1
$$

- RNN: 매번 $W_{hh}$를 곱함 → 지수적 감소/증가
- LSTM: **덧셈(+)**으로 정보 전달 → 곱셈의 연쇄가 아님
- Forget gate가 1에 가까우면 그래디언트가 거의 그대로 전달됨
- "Constant Error Carousel" - 셀 상태를 통해 그래디언트가 많은 시간 단계를 통과 가능

### 9.3.4 파라미터 수

입력 차원 $n$, hidden 차원 $d$일 때:
- 4개의 게이트 각각: $(n + d) \times d + d$ 파라미터
- 총: $4 \times [d(n + d) + d] = 4d(n + d + 1)$

---

## 9.4 GRU (Gated Recurrent Unit)

LSTM의 간소화 버전. 셀 상태 없이 게이트 2개로 동작.

### 9.4.1 공식

**Update Gate** (LSTM의 forget + input 결합):

$$
z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)
$$

**Reset Gate**:

$$
r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)
$$

**후보 Hidden State**:

$$
\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)
$$

**Hidden State 업데이트**:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### 9.4.2 LSTM vs GRU 비교

| | LSTM | GRU |
|---|---|---|
| 게이트 수 | 3 (forget, input, output) | 2 (update, reset) |
| 상태 | $h_t$, $c_t$ (두 개) | $h_t$ (하나) |
| 파라미터 | $4d(n+d+1)$ | $3d(n+d+1)$ |
| 성능 | 약간 우세 (긴 시퀀스) | 비슷하거나 약간 빠름 |

---

## 9.5 Attention Mechanism

### 9.5.1 직관

모든 시점을 동일하게 취급하지 말고, **관련성이 높은 시점에 더 집중**하자.

### 9.5.2 일반 공식

Query $q$, Key-Value 쌍 $\{(k_i, v_i)\}$에 대해:

$$
\text{Attention}(q, K, V) = \sum_{i} \alpha_i v_i
$$

$$
\alpha_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}, \quad e_i = \text{score}(q, k_i)
$$

**Score 함수의 종류**:

| 이름 | 공식 |
|---|---|
| Dot-product | $e_i = q^T k_i$ |
| Scaled dot-product | $e_i = \frac{q^T k_i}{\sqrt{d_k}}$ |
| Additive (Bahdanau) | $e_i = v^T \tanh(W_1 q + W_2 k_i)$ |

---

## 9.6 Scaled Dot-Product Attention

### 9.6.1 공식

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

### 9.6.2 차원 분석 (Full Dimension Analysis)

$n$: 시퀀스 길이, $d_k$: key/query 차원, $d_v$: value 차원

| 텐서 | 차원 |
|---|---|
| $Q$ | $(n, d_k)$ |
| $K$ | $(n, d_k)$ |
| $V$ | $(n, d_v)$ |
| $QK^T$ | $(n, d_k) \times (d_k, n) = (n, n)$ |
| $\text{softmax}(\cdot)$ | $(n, n)$ — 각 행이 확률 분포 |
| $\text{softmax}(\cdot) V$ | $(n, n) \times (n, d_v) = (n, d_v)$ |
| **Output** | $(n, d_v)$ |

**해석**: $(n, n)$ 행렬은 **attention weight matrix**로, 모든 위치 쌍 간의 관계를 나타냄.

### 9.6.3 $\sqrt{d_k}$ 스케일링: 분산 분석 증명

**문제**: $d_k$가 크면 dot product의 크기가 커져서 softmax가 극단적인 값을 갖게 됨.

**증명**:

$q_i, k_i$가 독립적으로 평균 0, 분산 1인 확률변수라고 가정하자.

$$
q^T k = \sum_{i=1}^{d_k} q_i k_i
$$

각 항 $q_i k_i$의 기대값과 분산:

$$
\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \cdot \mathbb{E}[k_i] = 0 \cdot 0 = 0
$$

$$
\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \cdot \mathbb{E}[k_i^2] - 0 = 1 \cdot 1 = 1
$$

독립이므로:

$$
\text{Var}(q^T k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
$$

따라서 $q^T k$의 표준편차는 $\sqrt{d_k}$.

$\sqrt{d_k}$로 나누면:

$$
\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

→ **분산을 1로 유지하여 softmax의 그래디언트가 적절한 범위에 있게 함**

> **시험 포인트**: 스케일링 없이 $d_k$가 크면 softmax의 입력값이 매우 크거나 작아져 거의 one-hot 형태가 됨 → 그래디언트가 거의 0 (softmax의 saturation region) → 학습 불가

---

## 9.7 Multi-Head Attention (MHA)

### 9.7.1 공식

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W_i^Q, \, K W_i^K, \, V W_i^V)
$$

### 9.7.2 차원 분석

- $d_{model}$: 모델 차원 (예: 512)
- $h$: 헤드 수 (예: 8)
- $d_k = d_v = d_{model} / h$ (예: 64)

| 파라미터 | 차원 |
|---|---|
| $W_i^Q$ | $(d_{model}, d_k)$ |
| $W_i^K$ | $(d_{model}, d_k)$ |
| $W_i^V$ | $(d_{model}, d_v)$ |
| $W^O$ | $(h \cdot d_v, d_{model})$ = $(d_{model}, d_{model})$ |

### 9.7.3 파라미터 수

$$
\text{Params}_{MHA} = h \times (3 \times d_{model} \times d_k) + d_{model}^2 = 3d_{model}^2 + d_{model}^2 = 4d_{model}^2
$$

(bias 포함 시 추가)

### 9.7.4 왜 Multi-Head인가?

1. **다양한 관계 포착**: 각 헤드가 다른 유형의 관계를 학습
   - Head 1: 구문적 관계 (syntactic)
   - Head 2: 의미적 관계 (semantic)
   - Head 3: 위치적 관계 (positional)
2. **단일 헤드 대비 계산량 동일**: $d_k = d_{model}/h$로 축소하므로 총 연산량 유지
3. **표현력 증가**: 서로 다른 부분 공간(subspace)에서의 attention을 결합

---

## 9.8 Positional Encoding (위치 인코딩)

### 9.8.1 필요성

Attention은 순서 정보를 사용하지 않음 (permutation equivariant). 시퀀스의 순서 정보를 주입해야 함.

### 9.8.2 Sinusoidal Positional Encoding

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

여기서:
- $pos$: 시퀀스에서의 위치 (0, 1, 2, ...)
- $i$: 차원 인덱스 (0, 1, ..., $d_{model}/2 - 1$)

### 9.8.3 왜 이 공식인가?

1. **상대적 위치 표현 가능**: $PE_{pos+k}$는 $PE_{pos}$의 선형 변환으로 표현 가능

$$
\begin{bmatrix} \sin(\omega_i (pos+k)) \\ \cos(\omega_i (pos+k)) \end{bmatrix} =
\begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix}
\begin{bmatrix} \sin(\omega_i pos) \\ \cos(\omega_i pos) \end{bmatrix}
$$

2. **다양한 주파수**: 낮은 차원은 높은 주파수 → 미세한 위치 차이, 높은 차원은 낮은 주파수 → 전반적 위치
3. **학습 없이 어떤 길이든 일반화 가능**

### 9.8.4 적용 방법

$$
\text{Input} = \text{Token Embedding} + \text{Positional Encoding}
$$

(덧셈, 연결이 아님)

---

## 9.9 Transformer Block

### 9.9.1 Encoder Block 구조

```
x → [Multi-Head Self-Attention] → [Add & LayerNorm] → [FFN] → [Add & LayerNorm] → out
      ↑_________________________________↗              ↑____________________________↗
              Residual Connection                        Residual Connection
```

수식으로:

$$
\text{mid} = \text{LayerNorm}(x + \text{MultiHeadAttn}(x, x, x))
$$

$$
\text{out} = \text{LayerNorm}(\text{mid} + \text{FFN}(\text{mid}))
$$

### 9.9.2 Feed-Forward Network (FFN)

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2
$$

또는 최신 모델에서:

$$
\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- 일반적으로 $d_{ff} = 4 \times d_{model}$
- FFN 파라미터: $2 \times d_{model} \times d_{ff} + d_{ff} + d_{model}$

### 9.9.3 Pre-Norm vs Post-Norm

**Post-Norm** (원래 Transformer):
$$
\text{out} = \text{LN}(x + \text{SubLayer}(x))
$$

**Pre-Norm** (이후 연구에서 선호):
$$
\text{out} = x + \text{SubLayer}(\text{LN}(x))
$$

Pre-Norm이 학습 안정성이 더 좋음 (learning rate warmup 덜 필요).

---

## 9.10 Layer Normalization vs Batch Normalization

### 9.10.1 정규화 축 비교

텐서 형태: $(N, T, d)$ — (배치, 시퀀스, 특성)

| | Batch Norm | Layer Norm |
|---|---|---|
| 정규화 축 | 배치 축 $(N)$ | 특성 축 $(d)$ |
| 통계 계산 범위 | 같은 특성, 모든 샘플 | 같은 샘플, 모든 특성 |
| 학습 파라미터 | $2d$ ($\gamma, \beta$) | $2d$ ($\gamma, \beta$) |
| 배치 크기 의존 | O (작은 배치에서 불안정) | X |
| 시퀀스 길이 의존 | 가변 길이 처리 어려움 | 문제 없음 |
| 추론 시 | running statistics 필요 | 자체 통계 사용 |
| 주요 사용처 | CNN (이미지) | Transformer, RNN (시퀀스) |

### 9.10.2 Layer Normalization 공식

입력 $x \in \mathbb{R}^d$ (하나의 샘플, 하나의 시점):

$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

> **시험 포인트**: Transformer에서 BN 대신 LN을 사용하는 이유 — (1) 시퀀스 길이가 가변적, (2) 배치 크기에 독립적, (3) 자기회귀(autoregressive) 생성 시 배치 크기 1에서도 동작

---

## 9.11 Encoder-Decoder Architecture

### 9.11.1 전체 구조

```
[Input Tokens] → [Embedding + PosEnc] → [Encoder × N] → Encoder Output
                                                              ↓
[Output Tokens] → [Embedding + PosEnc] → [Decoder × N] → [Linear → Softmax] → Output
```

### 9.11.2 Decoder의 Masked Self-Attention

**목적**: 자기회귀(autoregressive) 생성 시 미래 토큰을 보지 못하게 함.

**마스크**:

$$
\text{Mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right) V
$$

$-\infty$를 더하면 softmax 후 해당 위치의 attention weight가 0이 됨.

### 9.11.3 Cross-Attention (교차 어텐션)

Decoder에서 Encoder의 출력을 참조하는 메커니즘:

$$
\text{CrossAttn}(Q_{dec}, K_{enc}, V_{enc})
$$

- **Query**: Decoder의 현재 표현
- **Key, Value**: Encoder의 출력

→ Decoder가 "어떤 입력 부분에 집중할지" 결정

### 9.11.4 Decoder Block 구조

```
x → [Masked Self-Attention] → [Add & LN] → [Cross-Attention] → [Add & LN] → [FFN] → [Add & LN] → out
```

1. Masked Self-Attention: 이전 출력 토큰들 간의 관계
2. Cross-Attention: 입력 시퀀스(Encoder 출력) 참조
3. FFN: 비선형 변환

---

# Part 10: 생성 모델 (Generative Models)

---

## 10.1 판별 모델 vs 생성 모델

### 10.1.1 기본 구분

| | 판별 모델 (Discriminative) | 생성 모델 (Generative) |
|---|---|---|
| 학습 목표 | $p(y \mid x)$ | $p(x)$ 또는 $p(x, y)$ |
| 목적 | 분류, 예측 | 새로운 데이터 생성 |
| 예시 | Logistic Regression, SVM, CNN classifier | VAE, GAN, Flow, Diffusion |
| 할 수 있는 것 | 분류/회귀 | 생성 + 밀도 추정 + (분류) |

### 10.1.2 생성 모델의 분류

```
생성 모델
├── Explicit Density (명시적 밀도)
│   ├── Tractable: Normalizing Flow, Autoregressive
│   └── Approximate: VAE (ELBO)
└── Implicit Density (암묵적 밀도)
    └── GAN

+ Score-based / Diffusion Models (별도 카테고리)
```

---

## 10.2 VAE (Variational Autoencoder)

### 10.2.1 목표

데이터의 주변 로그 우도(marginal log-likelihood)를 최대화:

$$
\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z) \, dz
$$

문제: 이 적분은 **계산 불가능(intractable)** — 모든 가능한 $z$에 대해 적분해야 하므로.

### 10.2.2 ELBO 유도 (Jensen's Inequality 방법)

근사 사후 분포 $q_\phi(z \mid x)$를 도입:

$$
\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z) \, dz
$$

$$
= \log \int \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)} q_\phi(z \mid x) \, dz
$$

$$
= \log \, \mathbb{E}_{q_\phi(z|x)} \left[\frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)}\right]
$$

**Jensen's inequality** ($\log$는 오목 함수):

$$
\geq \mathbb{E}_{q_\phi(z|x)} \left[\log \frac{p_\theta(x \mid z) p(z)}{q_\phi(z \mid x)}\right]
$$

이것이 **ELBO (Evidence Lower Bound)**:

$$
\text{ELBO} = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \| p(z))
$$

**분해 과정**:

$$
\text{ELBO} = \mathbb{E}_{q} [\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)]
$$

$$
= \mathbb{E}_{q} [\log p_\theta(x|z)] + \mathbb{E}_{q} [\log p(z) - \log q_\phi(z|x)]
$$

$$
= \underbrace{\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)]}_{\text{Reconstruction Term}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization Term}}
$$

### 10.2.3 ELBO의 의미

$$
\log p_\theta(x) = \text{ELBO} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))
$$

- $D_{KL} \geq 0$이므로 ELBO는 $\log p_\theta(x)$의 **하한**
- ELBO를 최대화하면 log-likelihood를 (간접적으로) 최대화
- KL 항이 0이 되면 ELBO = log-likelihood (완벽한 근사)

**각 항의 역할**:
1. **Reconstruction**: $\mathbb{E}_{q}[\log p_\theta(x|z)]$ — Decoder가 $z$로부터 $x$를 잘 복원하도록
2. **Regularization**: $D_{KL}(q_\phi(z|x) \| p(z))$ — Encoder의 출력이 사전 분포 $p(z) = \mathcal{N}(0, I)$에 가깝도록

### 10.2.4 Reparameterization Trick

**문제**: $z \sim q_\phi(z|x)$에서 샘플링은 **미분 불가능** → 역전파 불가

**해결**: 샘플링을 결정적 변환으로 재매개변수화

$$
q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))
$$

일 때:

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

- 확률성(randomness)을 $\epsilon$으로 분리
- $z$는 $\mu$와 $\sigma$의 결정적 함수 → **미분 가능!**
- $\phi$에 대한 그래디언트 계산 가능

### 10.2.5 KL Divergence의 닫힌 형태 (Closed-Form)

$q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$이고 $p(z) = \mathcal{N}(0, I)$일 때:

$$
D_{KL}(q \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

> **시험 포인트**: 이 유도를 할 수 있어야 함. 가우시안 간의 KL divergence 공식에서 직접 대입.

---

## 10.3 GAN (Generative Adversarial Network)

### 10.3.1 Minimax 목적 함수

$$
\min_G \max_D \, V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

- **Discriminator $D$**: $V$를 **최대화** (진짜는 1, 가짜는 0으로 판별)
- **Generator $G$**: $V$를 **최소화** (가짜를 진짜처럼 만들어 $D$를 속임)

### 10.3.2 최적 판별자 유도 (Optimal Discriminator)

$G$ 고정, $D$에 대해 최대화:

$$
V(D, G) = \int_x \left[ p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx
$$

피적분함수를 $D(x) = y$에 대해 미분:

$$
f(y) = a \log y + b \log(1 - y)
$$

$$
f'(y) = \frac{a}{y} - \frac{b}{1-y} = 0
$$

$$
a(1-y) = by
$$

$$
y^* = \frac{a}{a+b}
$$

따라서:

$$
\boxed{D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}}
$$

### 10.3.3 JS Divergence와의 관계

$D^*$를 $V$에 대입:

$$
V(D^*, G) = \mathbb{E}_{x \sim p_{data}} \left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g} \left[\log \frac{p_g(x)}{p_{data}(x) + p_g(x)}\right]
$$

$$
= \mathbb{E}_{p_{data}} \left[\log \frac{p_{data}}{(p_{data} + p_g)/2} - \log 2\right] + \mathbb{E}_{p_g} \left[\log \frac{p_g}{(p_{data} + p_g)/2} - \log 2\right]
$$

$$
= D_{KL}\left(p_{data} \| \frac{p_{data} + p_g}{2}\right) + D_{KL}\left(p_g \| \frac{p_{data} + p_g}{2}\right) - 2\log 2
$$

$$
\boxed{V(D^*, G) = 2 \, D_{JS}(p_{data} \| p_g) - 2\log 2}
$$

**Jensen-Shannon Divergence**:

$$
D_{JS}(p \| q) = \frac{1}{2} D_{KL}(p \| m) + \frac{1}{2} D_{KL}(q \| m), \quad m = \frac{p+q}{2}
$$

- $D_{JS} \geq 0$, $D_{JS} = 0 \iff p_{data} = p_g$
- 최적 상태에서 $V(D^*, G^*) = -2\log 2$, $D^*(x) = 1/2$ (구분 불가)

### 10.3.4 실전적 문제

**Generator 학습의 문제**: 초기에 $G$가 나쁘면 $D(G(z)) \approx 0$

$$
\log(1 - D(G(z))) \approx \log 1 = 0 \quad \text{(그래디언트 거의 없음)}
$$

**실전 해결**: $\log(1 - D(G(z)))$ 최소화 대신 $\log D(G(z))$ **최대화**

- 초기에 더 강한 그래디언트 신호 제공
- 동일한 fixed point를 가짐

---

## 10.4 WGAN (Wasserstein GAN)

### 10.4.1 동기: GAN의 문제점

1. **Mode collapse**: Generator가 다양한 샘플을 생성하지 못함
2. **학습 불안정**: D와 G의 균형 맞추기 어려움
3. **JS divergence의 한계**: 두 분포의 support가 겹치지 않으면 JS divergence가 상수 ($\log 2$)가 됨 → 유용한 그래디언트 없음

### 10.4.2 Wasserstein Distance (Earth Mover's Distance)

$$
W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]
$$

- 한 분포를 다른 분포로 "옮기는" 데 필요한 최소 비용
- 두 분포의 support가 겹치지 않아도 의미 있는 거리 제공
- 연속적이고 미분 가능한 metric → 안정적인 학습

### 10.4.3 Kantorovich-Rubinstein Duality

$$
W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \left[\mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]\right]
$$

- $\|f\|_L \leq 1$: $f$가 1-Lipschitz (Lipschitz 상수 $\leq 1$)
- Discriminator → **Critic** (확률이 아닌 실수 값 출력)

### 10.4.4 WGAN 목적 함수

$$
\min_G \max_{D \in \mathcal{D}} \, \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]
$$

- $\mathcal{D}$: 1-Lipschitz 함수의 집합
- Lipschitz 제약 실현: weight clipping (원래 WGAN) 또는 gradient penalty (WGAN-GP)

---

## 10.5 Normalizing Flow

### 10.5.1 핵심 아이디어

간단한 분포 $z \sim p_z(z)$에서 **가역 변환(invertible transformation)** $f$를 적용하여 복잡한 분포 생성:

$$
x = f(z), \quad z = f^{-1}(x)
$$

### 10.5.2 Change of Variables Formula (변수 변환 공식)

$$
p_x(x) = p_z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$

로그 형태:

$$
\log p_x(x) = \log p_z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|
$$

### 10.5.3 연쇄 변환 (Composition of Flows)

$K$개의 가역 변환을 연쇄:

$$
x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z)
$$

$$
\log p_x(x) = \log p_z(z_0) + \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k^{-1}}{\partial z_k} \right|
$$

### 10.5.4 장단점

| 장점 | 단점 |
|---|---|
| **정확한** 밀도 추정 가능 (exact likelihood) | 가역 변환 설계 제약 |
| 정확한 latent 추론 | Jacobian 행렬식 계산 비용 |
| 학습 안정적 | 고차원에서 표현력 제한적 |

> **시험 포인트**: VAE는 근사적 밀도(ELBO), GAN은 암묵적 밀도, Flow는 **정확한(exact) 밀도**를 제공.

---

## 10.6 Diffusion Models (DDPM)

### 10.6.1 개요

두 과정으로 구성:
1. **Forward process**: 데이터에 노이즈를 점진적으로 추가 (고정, 학습 안 함)
2. **Reverse process**: 노이즈에서 데이터를 점진적으로 복원 (학습)

### 10.6.2 Forward Process

각 시간 단계에서 가우시안 노이즈를 추가:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t I)
$$

여기서 $\beta_t \in (0, 1)$는 노이즈 스케줄 (작은 값에서 점점 큼).

### 10.6.3 Closed-Form: $x_0$에서 $x_t$로 직접

$\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$로 정의하면:

$$
\boxed{q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I)}
$$

따라서:

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**유도** (핵심 단계):

$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1$

$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2$

$= \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1) + \sqrt{1-\alpha_2} \epsilon_2$

$= \sqrt{\alpha_1 \alpha_2} x_0 + \sqrt{\alpha_2(1-\alpha_1)} \epsilon_1 + \sqrt{1-\alpha_2} \epsilon_2$

두 독립 가우시안의 합: $\mathcal{N}(0, \alpha_2(1-\alpha_1))$과 $\mathcal{N}(0, 1-\alpha_2)$

분산의 합: $\alpha_2(1-\alpha_1) + (1-\alpha_2) = \alpha_2 - \alpha_1\alpha_2 + 1 - \alpha_2 = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2$

따라서 $x_2 = \sqrt{\bar{\alpha}_2} x_0 + \sqrt{1-\bar{\alpha}_2} \bar{\epsilon}$, $\bar{\epsilon} \sim \mathcal{N}(0, I)$

귀납적으로 일반화하면 위의 closed-form 획득.

### 10.6.4 Reverse Process (학습)

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \, \sigma_t^2 I)
$$

신경망 $\mu_\theta$가 평균을 예측. 분산 $\sigma_t^2$는 고정 ($\beta_t$ 또는 $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$).

### 10.6.5 Simple Loss (ε-prediction)

$$
\boxed{\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]}
$$

여기서:
- $t \sim \text{Uniform}\{1, \ldots, T\}$
- $x_0 \sim q(x_0)$ (데이터 분포에서 샘플)
- $\epsilon \sim \mathcal{N}(0, I)$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
- $\epsilon_\theta$: 노이즈를 예측하는 신경망

**해석**: 네트워크에게 "$x_t$에서 어떤 노이즈가 추가되었는지 맞혀봐"라고 학습.

### 10.6.6 $\mu_\theta$와 $\epsilon_\theta$의 관계

$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$에서:

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}
$$

posterior mean:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

### 10.6.7 Score Function과의 관계

**Score function** 정의:

$$
s(x) = \nabla_x \log p(x)
$$

**Score matching의 관계**:

$q(x_t | x_0)$의 score:

$$
\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

따라서:

$$
\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)
$$

**노이즈 예측 = (스케일된) score 추정**

> **시험 포인트**: Diffusion model의 세 가지 등가적 관점:
> 1. **노이즈 예측** ($\epsilon$-prediction): $\epsilon_\theta(x_t, t)$
> 2. **Score 추정**: $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$
> 3. **$x_0$ 예측**: $\hat{x}_0 = f_\theta(x_t, t)$

### 10.6.8 생성 모델 비교 요약

| | VAE | GAN | Flow | Diffusion |
|---|---|---|---|---|
| 밀도 추정 | 근사 (ELBO) | 불가 | 정확 | 근사 |
| 학습 안정성 | 좋음 | 나쁨 | 좋음 | 좋음 |
| 생성 품질 | 보통 (blurry) | 좋음 (sharp) | 보통 | 매우 좋음 |
| 생성 속도 | 빠름 | 빠름 | 빠름 | 느림 (T steps) |
| 모드 커버리지 | 좋음 | 나쁨 (mode collapse) | 좋음 | 좋음 |

---

# Part 11: 핵심 공식 총정리 (Essential Formula Cheat Sheet)

---

## 11.1 레이어별 파라미터 수 테이블

| Layer | Parameters | 비고 |
|---|---|---|
| **Linear** $(d_{in}, d_{out})$ | $d_{in} \times d_{out} + d_{out}$ | bias 포함 |
| **Conv2d** $(C_{in}, C_{out}, k)$ | $C_{out} \times C_{in} \times k^2 + C_{out}$ | bias 포함 |
| **Conv2d 1×1** $(C_{in}, C_{out})$ | $C_{out} \times C_{in} + C_{out}$ | 채널 혼합 |
| **Multi-Head Attention** $(d_{model}, h)$ | $4 d_{model}^2$ | $W^Q, W^K, W^V, W^O$ (bias 무시) |
| **FFN** $(d_{model}, d_{ff})$ | $2 \times d_{model} \times d_{ff} + d_{ff} + d_{model}$ | 두 개의 Linear |
| **LayerNorm** $(d)$ | $2d$ | $\gamma, \beta$ |
| **BatchNorm** $(C)$ | $2C$ (학습) + $2C$ (running) | $\gamma, \beta$ + $\mu_{run}, \sigma^2_{run}$ |
| **Embedding** $(V, d)$ | $V \times d$ | 어휘 크기 $V$, 임베딩 차원 $d$ |
| **LSTM** $(n, d)$ | $4d(n+d) + 4d$ | 4개 게이트 |
| **GRU** $(n, d)$ | $3d(n+d) + 3d$ | 3개 게이트 |

### Transformer Block 총 파라미터 (하나의 블록)

$$
\text{Params}_{block} = \underbrace{4d_{model}^2}_{\text{MHA}} + \underbrace{2 \times d_{model} \times d_{ff}}_{\text{FFN (weights)}} + \underbrace{d_{ff} + d_{model}}_{\text{FFN (biases)}} + \underbrace{2 \times 2d_{model}}_{\text{2 × LayerNorm}}
$$

$d_{ff} = 4d_{model}$이면:

$$
\approx 4d_{model}^2 + 8d_{model}^2 = 12d_{model}^2 \quad (\text{bias 무시 시})
$$

---

## 11.2 Optimizer 비교 테이블

| Optimizer | Update Rule | 특징 |
|---|---|---|
| **GD** (Batch) | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$ | 전체 데이터셋 사용, 느림 |
| **SGD** | $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_i(\theta)$ | 미니배치/단일 샘플, 노이즈 많음 |
| **SGD + Momentum** | $v_t = \gamma v_{t-1} + \eta \nabla \mathcal{L}$ | 관성으로 진동 감소 |
| | $\theta \leftarrow \theta - v_t$ | $\gamma \approx 0.9$ |
| **Adam** | $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$ | 1차 모멘트 (평균) |
| | $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$ | 2차 모멘트 (분산) |
| | $\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \; \hat{v}_t = \frac{v_t}{1-\beta_2^t}$ | Bias correction |
| | $\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$ | 적응적 학습률 |

**Adam 기본값**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

## 11.3 아키텍처 비교 테이블

| Model | Year | Depth | Key Innovation | Top-5 Error |
|---|---|---|---|---|
| **AlexNet** | 2012 | 8 | ReLU, GPU, Dropout | 16.4% |
| **VGG-16** | 2014 | 16 | 3×3 only, deeper | 7.3% |
| **GoogLeNet** | 2014 | 22 | Inception module, 1×1 conv | 6.7% |
| **ResNet-50** | 2015 | 50 | Skip connection | 3.57% |
| **ResNet-152** | 2015 | 152 | Deeper ResNet | 3.0% |
| **Transformer** | 2017 | - | Self-attention, 병렬화 | - |
| **ViT** | 2020 | - | Transformer for vision | - |

---

## 11.4 정보 이론 공식 (Information Theory)

### Entropy

$$
H(X) = -\sum_{x} p(x) \log p(x) = -\mathbb{E}[\log p(X)]
$$

### Cross-Entropy

$$
H(p, q) = -\sum_{x} p(x) \log q(x) = -\mathbb{E}_p[\log q(X)]
$$

### KL Divergence

$$
D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

- $D_{KL} \geq 0$ (Gibbs' inequality)
- $D_{KL}(p\|q) \neq D_{KL}(q\|p)$ (비대칭)

### JS Divergence

$$
D_{JS}(p \| q) = \frac{1}{2} D_{KL}(p \| m) + \frac{1}{2} D_{KL}(q \| m), \quad m = \frac{p+q}{2}
$$

- 대칭: $D_{JS}(p\|q) = D_{JS}(q\|p)$
- $0 \leq D_{JS} \leq \log 2$

### Mutual Information

$$
I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y)) = H(X) - H(X|Y)
$$

---

## 11.5 확률 분포 성질 테이블

| 분포 | PDF/PMF | 평균 | 분산 |
|---|---|---|---|
| **Bernoulli** $(p)$ | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ |
| **Gaussian** $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| **Categorical** $(p_1, \ldots, p_k)$ | $\prod p_i^{x_i}$ | - | - |
| **Uniform** $[a, b]$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |

### 가우시안 관련 핵심 공식

**두 가우시안의 곱**:

$$
\mathcal{N}(x; \mu_1, \sigma_1^2) \cdot \mathcal{N}(x; \mu_2, \sigma_2^2) \propto \mathcal{N}(x; \mu', {\sigma'}^2)
$$

$$
{\sigma'}^2 = \frac{\sigma_1^2 \sigma_2^2}{\sigma_1^2 + \sigma_2^2}, \quad \mu' = \frac{\sigma_2^2 \mu_1 + \sigma_1^2 \mu_2}{\sigma_1^2 + \sigma_2^2}
$$

**두 독립 가우시안의 합**:

$$
X \sim \mathcal{N}(\mu_1, \sigma_1^2), Y \sim \mathcal{N}(\mu_2, \sigma_2^2) \implies X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
$$

---

## 11.6 활성화 함수 정리

| 함수 | 정의 | 도함수 | 범위 |
|---|---|---|---|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | $(0, 1)$ |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | $(-1, 1)$ |
| **ReLU** | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | $[0, \infty)$ |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $\begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$ | $(-\infty, \infty)$ |
| **GELU** | $x \cdot \Phi(x)$ | 복잡 | $(-\infty, \infty)$ |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\text{softmax}_i (\delta_{ij} - \text{softmax}_j)$ | $(0, 1)$, 합=1 |

---

## 11.7 CNN 핵심 공식 Quick Reference

| 항목 | 공식 |
|---|---|
| 출력 크기 | $\lfloor (H_{in} + 2p - k) / s \rfloor + 1$ |
| Conv2d 파라미터 | $C_{out} \times C_{in} \times k^2 + C_{out}$ |
| Same padding ($s$=1) | $p = (k-1)/2$ |
| 수용 영역 (재귀) | $r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1}$ |
| ResNet residual | $H(x) = F(x) + x$ |
| BN forward | $y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ |

## 11.8 Transformer 핵심 공식 Quick Reference

| 항목 | 공식 |
|---|---|
| Scaled Dot-Product Attention | $\text{softmax}(QK^T / \sqrt{d_k}) V$ |
| Multi-Head Attention | $\text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$ |
| Positional Encoding (sin) | $PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$ |
| Positional Encoding (cos) | $PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$ |
| FFN | $\text{ReLU}(xW_1 + b_1)W_2 + b_2$ |
| Layer Norm | $\gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ |
| MHA 파라미터 | $4d_{model}^2$ |
| Transformer block | $\approx 12d_{model}^2$ ($d_{ff} = 4d_{model}$ 시) |

## 11.9 생성 모델 핵심 공식 Quick Reference

| 항목 | 공식 |
|---|---|
| ELBO | $\mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$ |
| Reparameterization | $z = \mu + \sigma \odot \epsilon, \; \epsilon \sim \mathcal{N}(0,I)$ |
| GAN objective | $\min_G \max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$ |
| Optimal $D^*$ | $p_{data}(x) / (p_{data}(x) + p_g(x))$ |
| GAN ↔ JS | $V(D^*,G) = 2D_{JS}(p_{data} \| p_g) - 2\log 2$ |
| Flow (change of var.) | $\log p_x(x) = \log p_z(f^{-1}(x)) + \log|\det J_{f^{-1}}|$ |
| Diffusion forward | $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| Diffusion loss | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ |
| Score function | $s(x) = \nabla_x \log p(x)$ |

---

## 시험 대비 핵심 체크리스트

### 반드시 유도할 수 있어야 하는 것들:
- [ ] CNN 출력 크기 공식 유도
- [ ] VGG 3×3 vs 7×7 파라미터 비교 증명
- [ ] ResNet skip connection의 그래디언트 흐름 분석
- [ ] RNN vanishing gradient의 수학적 분석 ($\|W_{hh}\|^{t-k}$)
- [ ] LSTM이 vanishing gradient를 해결하는 이유 (cell state의 additive update)
- [ ] $\sqrt{d_k}$ scaling의 분산 분석 증명
- [ ] ELBO 유도 (Jensen's inequality)
- [ ] GAN optimal discriminator 유도
- [ ] GAN ↔ JS divergence 관계 유도
- [ ] Diffusion forward process closed-form 유도
- [ ] Score function과 noise prediction의 관계

### 개념적으로 설명할 수 있어야 하는 것들:
- [ ] CNN의 세 가지 inductive bias
- [ ] Batch Norm의 학습 파라미터 $\gamma, \beta$의 역할
- [ ] Layer Norm vs Batch Norm의 차이와 Transformer에서 LN을 쓰는 이유
- [ ] Multi-Head Attention의 필요성
- [ ] Positional Encoding이 상대적 위치를 표현할 수 있는 이유
- [ ] Masked attention의 목적과 구현
- [ ] VAE reparameterization trick의 필요성
- [ ] WGAN의 동기 (JS divergence의 한계)
- [ ] Diffusion model의 세 가지 등가적 관점

---

> **마지막 조언**: 이 시험은 수학적 유도와 논리적 추론을 중시한다. 공식을 외우는 것보다 **왜 그런 공식이 나오는지** 유도 과정을 이해하는 것이 핵심이다. 특히 GAN의 optimal discriminator, ELBO 유도, √d_k scaling 증명은 반드시 직접 손으로 풀어볼 것.

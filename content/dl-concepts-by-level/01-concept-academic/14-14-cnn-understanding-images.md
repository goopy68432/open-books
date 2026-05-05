---
title: "14. 합성곱 신경망과 이미지 이해"
slug: 14-cnn-understanding-images
order: 14
---

# 14. 합성곱 신경망과 이미지 이해

## 14.1 동기부여 및 개요

이미지는 $H \times W \times C$의 고차원 격자 구조를 가진다. 일반 MLP로 $224 \times 224 \times 3$ 이미지를 처리하면 첫 번째 은닉층에서만 수천만 개의 파라미터가 필요하다. 합성곱 신경망(CNN)은 **지역성(locality)**과 **이동 불변성(translation invariance)**이라는 두 가지 귀납적 편향을 활용하여 파라미터 수를 극적으로 줄이면서도 이미지의 공간적 패턴을 효과적으로 학습한다.

2012년 AlexNet의 ImageNet 우승 이후 CNN은 컴퓨터 비전의 표준이 되었으며, 분류, 검출, 분할, 깊이 추정 등 거의 모든 시각 과제에서 핵심 역할을 수행한다.

**연결**: 11장의 MLP에 **공간적 구조에 대한 귀납적 편향**을 추가한 것이 CNN이며, 12장의 최적화 기법과 13장의 정규화가 학습에 적용된다.

---

## 14.2 합성곱 연산

### 14.2.1 정의

**정의 14.1 (이산 합성곱).** 1D에서 필터 $w \in \mathbb{R}^k$와 입력 $x \in \mathbb{R}^n$의 합성곱:

$$[w * x](i) = \sum_{u=0}^{k-1} w_u \cdot x_{i+u}$$

2D로 확장하면 $[W * X](i,j) = \sum_u \sum_v W_{u,v} \cdot X_{i+u, j+v}$이다.

*직관적 해석*: 작은 필터(커널)가 입력 위를 슬라이딩하면서, 각 위치에서 필터와 입력 패치의 내적(유사도)을 계산한다. 필터가 찾는 패턴과 입력이 비슷할수록 출력값이 크다.

### 14.2.2 합성곱의 행렬 해석

**정리 14.1.** 합성곱 연산은 특수한 구조를 가진 행렬곱 $y = Wx$와 동치이다. 이 행렬 $W$는 두 가지 구조적 특성을 가진다:

1. **희소성(Sparsity)**: 각 출력 뉴런은 입력의 국소 영역만 참조 $\Rightarrow$ **지역성**
2. **동일 대각 밴드(Toeplitz 구조)**: 모든 위치에서 같은 필터 값 사용 $\Rightarrow$ **가중치 공유**

```
일반 MLP (Fully Connected):    합성곱 (Convolutional):
┌───────────────┐              ┌───────────────┐
│ * * * * * * * │              │ a b c 0 0 0 0 │
│ * * * * * * * │              │ 0 a b c 0 0 0 │
│ * * * * * * * │              │ 0 0 a b c 0 0 │
│ * * * * * * * │              │ 0 0 0 a b c 0 │
│ * * * * * * * │              │ 0 0 0 0 a b c │
└───────────────┘              └───────────────┘
  파라미터: n×m개                파라미터: k개 (a,b,c)
```

### 14.2.3 출력 크기 공식

**보조정리 14.1.** 입력 크기 $H_{in}$, 커널 크기 $k$, 스트라이드 $s$, 패딩 $p$일 때:

$$H_{out} = \left\lfloor\frac{H_{in} + 2p - k}{s}\right\rfloor + 1$$

| 파라미터 | 역할 | 예시 |
|---------|------|------|
| 커널 크기 $k$ | 수용 영역(receptive field)의 크기 | $3\times3$, $5\times5$ |
| 스트라이드 $s$ | 필터 이동 간격 | $s=1$: 촘촘, $s=2$: 다운샘플링 |
| 패딩 $p$ | 입력 경계에 추가하는 값 | $p=(k-1)/2$: "same" 패딩 |

---

## 14.3 특징 검출기와 계층적 표현

### 14.3.1 특징 맵

$c_{in}$ 채널 입력에 $c_{out}$개의 $k \times k$ 필터를 적용하면:
- 각 필터: $\mathbb{R}^{c_{in} \times k \times k}$ (3D 텐서)
- 전체 커널: $\mathbb{R}^{c_{out} \times c_{in} \times k \times k}$
- 파라미터 수: $c_{out} \times c_{in} \times k^2 + c_{out}$ (바이어스 포함)

### 14.3.2 계층적 특징 학습

ZFNet(Zeiler & Fergus, 2014)의 시각화에 따르면 CNN의 각 층이 학습하는 특징은 계층적이다:

| 층 | 학습하는 특징 | 추상화 수준 |
|---|------------|-----------|
| Layer 1 | 에지, 색상 블롭 (Gabor-like) | 낮음 |
| Layer 2 | 코너, 텍스처 | 낮음-중간 |
| Layer 3 | 부분 패턴 (바퀴, 상체) | 중간 |
| Layer 4-5 | 의미적 물체 (얼굴, 꽃) | 높음 |

초기 층의 범용 특징은 **전이 학습(transfer learning)**에서 도메인 간 재사용이 가능하다.

### 14.3.3 $1 \times 1$ 합성곱

$1 \times 1$ 합성곱은 공간 해상도를 유지하면서 채널 간 선형 결합만 수행한다:

$$y_{i,j} = \sum_{c=1}^{c_{in}} w_c \cdot x_{i,j,c}$$

이는 각 픽셀 위치에서의 MLP와 동치(Network-in-Network)이며, **채널 차원 축소/확장**에 사용된다.

---

## 14.4 이동 등변성과 이동 불변성

### 14.4.1 정의

**정의 14.2.** 이동 연산 $S_\tau$에 대해:

- **등변성(Equivariance)**: $f(S_\tau(x)) = S_\tau(f(x))$ -- 입력이 이동하면 출력도 같은 만큼 이동
- **불변성(Invariance)**: $f(S_\tau(x)) = f(x)$ -- 입력이 이동해도 출력 불변

**정리 14.2.** 합성곱 층은 이동에 대해 **등변적(equivariant)**이다.

*증명 스케치.* $[w * S_\tau x](i) = \sum_u w_u (S_\tau x)_{i+u} = \sum_u w_u x_{i+u-\tau} = [w * x](i-\tau) = S_\tau[w * x](i)$. $\square$

**불변성을 얻으려면** 풀링이나 Global Average Pooling 등의 추가 연산이 필요하다.

### 14.4.2 응용에서의 구분

- **분류**: 불변성 필요 (고양이 위치에 무관하게 "고양이"로 분류)
- **검출**: 등변성 필요 (고양이가 이동하면 바운딩 박스도 이동)

---

## 14.5 풀링

### 14.5.1 주요 풀링 연산

**정의 14.3 (Max Pooling).** $k \times k$ 영역에서 최댓값을 선택한다:

$$y_{i,j} = \max_{0 \leq u,v < k} x_{si+u, sj+v}$$

**정의 14.4 (Average Pooling).** $k \times k$ 영역의 평균값을 계산한다:

$$y_{i,j} = \frac{1}{k^2}\sum_{0 \leq u,v < k} x_{si+u, sj+v}$$

**유용한 성질**: $\text{ReLU} \circ \text{MaxPool} = \text{MaxPool} \circ \text{ReLU}$ (교환 가능)

풀링의 역할:
1. 공간 해상도 축소 $\Rightarrow$ 계산량 감소
2. 수용 영역(receptive field) 확대
3. 약간의 이동 불변성 부여

**Global Average Pooling (GAP)**: 전체 공간 차원을 단일 값으로 축약하여 FC 층의 파라미터를 대폭 줄인다.

---

## 14.6 ConvNet 아키텍처의 진화

### 14.6.1 AlexNet (Krizhevsky et al., 2012)

```
입력: 227×227×3
  ↓ Conv(k=11, s=4, c=96) → 55×55×96
  ↓ MaxPool(k=3, s=2) → 27×27×96
  ↓ Conv(k=5, p=2, c=256) → 27×27×256
  ↓ MaxPool(k=3, s=2) → 13×13×256
  ↓ Conv(k=3, p=1, c=384) → 13×13×384
  ↓ Conv(k=3, p=1, c=384) → 13×13×384
  ↓ Conv(k=3, p=1, c=256) → 13×13×256
  ↓ MaxPool(k=3, s=2) → 6×6×256
  ↓ Flatten → 9216
  ↓ FC(4096) → FC(4096) → Softmax(1000)
```

혁신: ReLU, GPU 학습, Dropout, Data Augmentation.

### 14.6.2 후속 아키텍처

| 아키텍처 | 핵심 기여 | 깊이 |
|---------|---------|------|
| VGG (2014) | $3\times3$ 필터 통일, 더 깊은 구조 | 19 |
| GoogLeNet (2014) | Inception 모듈, $1\times1$ 합성곱 | 22 |
| ResNet (2015) | Skip connection ($y = F(x) + x$) | 152 |
| DenseNet (2017) | 모든 층 간 직접 연결 | 변동 |
| EfficientNet (2019) | 폭/깊이/해상도 복합 스케일링 | 변동 |

**VGG의 핵심 통찰**: $3 \times 3$ 합성곱 2개는 $5 \times 5$ 한 개와 동일한 수용 영역을 가지지만, 파라미터가 적고($2 \times 3^2 = 18$ vs $5^2 = 25$) 비선형성이 추가된다.

---

## 14.7 비전 태스크의 스펙트럼

### 14.7.1 분류, 태깅, 검출

| 태스크 | 입력 $\to$ 출력 | 손실 함수 |
|--------|--------------|---------|
| 분류 | 이미지 $\to$ 1 클래스 | Cross-Entropy + Softmax |
| 태깅 | 이미지 $\to$ 다중 태그 | Binary CE + Sigmoid (각 클래스 독립) |
| 검출 | 이미지 $\to$ 박스 + 클래스 | 분류 손실 + 바운딩 박스 회귀 |

수학적으로:
- 분류: $f: \mathbb{R}^{H \times W \times C} \to \Delta^{K-1}$ (확률 심플렉스)
- 태깅: $f: \mathbb{R}^{H \times W \times C} \to [0,1]^K$ (독립 확률)
- 검출: YOLO(You Only Look Once)는 이미지를 한 번만 통과시켜 모든 객체를 동시 검출

### 14.7.2 시맨틱 / 인스턴스 / 파놉틱 분할

**정의 14.5.** 각 분할 유형은 다음과 같이 정의된다:

- **시맨틱 분할**: $f: \mathbb{R}^{H \times W \times C} \to \{1, \ldots, K\}^{H \times W}$ (픽셀별 클래스)
- **인스턴스 분할**: 검출 + 각 객체 내 전경/배경 이진 분류 (Mask R-CNN)
- **파놉틱 분할**: stuff(하늘, 도로)와 things(차, 사람) 모두 처리

---

## 14.8 인코더-디코더 구조

### 14.8.1 기본 구조

```
인코더 (다운샘플링)           디코더 (업샘플링)
입력 ──→ Conv+Pool ──→ ··· ──→ Conv+Pool ──→ UpConv ──→ ··· ──→ 출력
  H×W     H/2×W/2              h×w           2h×2w            H×W
```

### 14.8.2 전치 합성곱 (Transposed Convolution)

**정의 14.6.** 합성곱을 $y = Wx$ ($W \in \mathbb{R}^{m \times n}$, $m < n$)로 표현하면, 전치 합성곱은:

$$z = W^\top y \in \mathbb{R}^n$$

**주의**: 이것은 $W$의 **전치(transpose)**이지 **역행렬(inverse)**이 아니다. 원래 입력을 복원하지 않는다.

### 14.8.3 Dilated Convolution

커널 원소 사이에 간격(dilation rate $r$)을 두어 파라미터 증가 없이 수용 영역을 확대:

$$\text{실효 커널 크기} = k + (k-1)(r-1)$$

### 14.8.4 대표 아키텍처

- **SegNet**: 풀링 인덱스(switch)를 기억하여 업샘플링에 활용
- **U-Net**: Skip connection으로 인코더의 고해상도 특징을 디코더에 직접 전달하여 세부 정보 보존

```
U-Net 구조:
  인코더                    디코더
  ┌──────┐  skip conn.   ┌──────┐
  │ Conv │ ────────────→ │ Conv │ → 출력
  └──┬───┘               └──┬───┘
     ↓                      ↑ (UpConv)
  ┌──────┐  skip conn.   ┌──────┐
  │ Conv │ ────────────→ │ Conv │
  └──┬───┘               └──┬───┘
     ↓                      ↑ (UpConv)
  ┌──────────────────────────────┐
  │         Bottleneck           │
  └──────────────────────────────┘
```

---

## 14.9 밀집 예측: 깊이, 표면법선, 자세 추정

밀집 예측(dense prediction)은 입력 이미지의 각 픽셀에 대해 값을 예측하는 "image-to-image" 과제이다:

| 과제 | 출력 차원 $c$ | 손실 함수 |
|------|-------------|---------|
| 깊이 예측 | $c = 1$ (스칼라 거리) | $L_2$ (회귀) |
| 표면법선 예측 | $c = 3$ (방향 벡터) | Cosine loss |
| 시맨틱 분할 | $c = K$ (클래스 수) | Cross-Entropy |
| 자세 추정 | $c = J$ (키포인트 수) | 히트맵 회귀 |

일반적 프레임워크: 인코더 $E: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{h \times w \times d}$, 디코더 $D: \mathbb{R}^{h \times w \times d} \to \mathbb{R}^{H \times W \times c}$. Multi-task learning으로 인코더를 공유하고 과제별 디코더를 붙일 수 있다.

---

## 14.10 CNN 시각화와 적대적 예제

### 14.10.1 Deconvnet 시각화

특정 뉴런의 활성화를 입력 공간으로 역투영하는 과정:
1. 특정 층/뉴런만 활성화 유지, 나머지 0
2. Max Unpooling (switch 사용)
3. ReLU 역적용
4. Transposed Convolution

### 14.10.2 적대적 예제

**정의 14.7 (FGSM, Goodfellow et al. 2015).** Fast Gradient Sign Method:

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x L(f_\theta(x), y))$$

사람 눈에는 구분할 수 없는 미세한 섭동 $\epsilon$으로 분류 결과를 완전히 뒤집을 수 있다.

**적대적 학습(Adversarial Training)**: min-max 최적화 $\min_\theta \max_{\|\delta\| \leq \epsilon} L(f_\theta(x+\delta), y)$로 강건성을 확보한다 (Madry et al., 2018).

---

## 14.11 흔한 오해와 주의점

| 오해 | 올바른 이해 |
|------|-----------|
| "합성곱은 비선형 연산이다" | 합성곱 자체는 **선형**(특수 구조의 행렬곱). 비선형성은 활성화 함수가 담당 |
| "합성곱은 이동 불변적이다" | 합성곱은 이동 **등변적**. 불변성은 풀링으로 획득 |
| "전치 합성곱은 합성곱의 역연산이다" | 전치(transpose)이지 역(inverse)이 아니다 |
| "CNN은 깊을수록 항상 좋다" | ResNet의 skip connection 없이는 vanishing gradient 문제 발생 |
| "시맨틱 분할과 인스턴스 분할은 같다" | 시맨틱은 같은 클래스 물체를 구분하지 않고, 인스턴스는 개별 구분 |
| "CNN은 전역 정보를 처음부터 활용한다" | 초기 층의 수용 영역은 매우 작아 국소 정보만 사용 |

---

## 14.12 핵심 요약

```
CNN의 두 가지 귀납적 편향:
  1. 지역성    → 희소 행렬 (각 뉴런이 국소 영역만 참조)
  2. 이동 불변성 → 가중치 공유 (같은 필터가 모든 위치에서 작동)

CNN 구조: [Conv → ReLU → Pool] × N → Flatten → FC → Softmax
아키텍처 진화: AlexNet → VGG → GoogLeNet → ResNet → EfficientNet

비전 태스크:
  분류 → 태깅 → 검출(YOLO) → 시맨틱 분할 → 인스턴스 분할(Mask R-CNN)
  → 파놉틱 분할 → 밀집 예측(깊이/법선/자세)

인코더-디코더: SegNet(switch), U-Net(skip connection)
업샘플링: 전치 합성곱(W^T), dilated convolution
출력 크기: H_out = floor((H_in + 2p - k) / s) + 1
```

**참고문헌**: LeCun et al. (1998), Krizhevsky et al. (2012), Simonyan & Zisserman (2015, VGG), He et al. (2016, ResNet), Zeiler & Fergus (2014, ZFNet), Ronneberger et al. (2015, U-Net), Goodfellow et al. (2015, FGSM), Cohen & Welling (2016, Group Equivariant CNN)

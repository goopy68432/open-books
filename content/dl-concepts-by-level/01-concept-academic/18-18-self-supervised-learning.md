---
title: "18. 자기지도학습 (Self-Supervised Learning)"
slug: 18-self-supervised-learning
order: 18
---

# 18. 자기지도학습 (Self-Supervised Learning)

## 1. 동기부여 및 개요

현실 세계의 데이터 대부분은 레이블이 없다. Geoffrey Hinton (1996)의 추산에 따르면, 뇌의 $10^{14}$개 시냅스 연결을 $10^9$초의 수명 동안 채우려면 초당 $10^5$비트의 정보가 필요하고, 이 정보는 **입력 자체**에서 올 수밖에 없다.

Yann LeCun의 "케이크 비유"에서:
- **자기지도학습 (SSL):** 케이크 본체 (genoise) -- 샘플당 수백만 비트
- **지도학습:** 아이싱 -- 샘플당 10~10,000 비트
- **강화학습:** 체리 -- 샘플당 수 비트

자기지도학습은 레이블 없이 데이터 자체의 구조를 학습하여 **범용 표현(universal representation)**을 얻는 패러다임이다.

```
지도학습 ──> 전이학습 ──> 사전훈련 ──> 자기지도학습 (SSL)
                                          |
                         +----------------+----------------+
                         |                |                |
                    대입 과제          프록시 과제       대조학습
                    (MAE 등)       (회전 예측 등)     (SimCLR, CLIP 등)
                                                          |
                                              +-----------+-----------+
                                              |                       |
                                       Contrastive              Non-contrastive
                                    (SimCLR, CLIP)         (Barlow Twins, BYOL)
```

---

## 2. 자기지도학습의 정의와 분류

### 2.1 정의

**Definition 2.1 (Self-Supervised Learning).** 외부 레이블 없이, 데이터 자체에서 알고리즘이 감독 신호를 자동 생성하여 표현을 학습하는 방법론.

입력 $x = (x_h, x_v)$를 hidden part와 visible part로 분리하고, visible part만으로 hidden part를 예측하는 과제를 통해 인코더 $f(\cdot; \theta)$가 범용 표현을 학습한다.

### 2.2 세 가지 접근법

| 접근법 | 핵심 아이디어 | 대표 방법 |
|--------|-------------|----------|
| **대입 과제 (Imputation)** | 입력의 일부를 가리고 복원 | MAE, BERT (MLM) |
| **프록시 과제 (Pretext)** | 변환 관계를 예측 | 회전 예측, 상대 위치 예측 |
| **대조 과제 (Contrastive)** | 같은 이미지의 변환은 가깝게, 다른 이미지는 멀게 | SimCLR, CLIP |

---

## 3. 대입 과제와 MAE

### 3.1 일반 구조

- 입력 분리: $x = (x_h, x_v)$ (hidden + visible)
- 목표: $\hat{x}_h = f(x_v, x_h = 0)$으로 숨겨진 부분 예측
- 대표 예시: NLP의 cloze task (빈칸 채우기), 시간적 예측 (과거 $\to$ 미래)

### 3.2 Masked Autoencoder (MAE)

He et al. (2022)의 MAE는 비전 분야의 대표적 대입 과제 방법이다:

1. 이미지를 패치로 분할하고 **75%를 무작위 마스킹**
2. 가시 패치(25%)만 인코더(ViT)에 입력
3. 마스크 토큰을 추가한 후 가벼운 디코더로 원본 픽셀 복원
4. 사전훈련 후 디코더는 버리고 인코더만 downstream에 활용

**왜 75%를 마스킹하는가?** 비전과 언어의 비대칭성 때문이다.
- 언어: 토큰이 의미 단위(semantic token)이므로 15% 마스킹으로도 충분히 어려움
- 이미지: 패치 간 공간적 중복(spatial redundancy)이 높아, 높은 마스킹 비율이 필요

```python
# MAE 학습 개념 의사코드
def mae_forward(image, mask_ratio=0.75):
    patches = patchify(image)               # 이미지를 패치로 분할
    visible, masked = random_mask(patches, mask_ratio)
    latent = encoder(visible)               # 가시 패치만 인코딩
    full = insert_mask_tokens(latent, masked_positions)
    reconstruction = decoder(full)          # 원본 복원
    loss = mse(reconstruction[masked_positions], patches[masked_positions])
    return loss
```

---

## 4. 프록시 과제 (Proxy/Pretext Tasks)

### 4.1 수학적 구조

원본 이미지 $x_1$에 변환 $t$를 적용하여 $x_2 = t(x_1)$을 생성하고, 표현 함수 $f$와 관계 함수 $r$로 변환 관계를 예측한다:

$$p(y \mid x_1, x_2) = p(y \mid r[f(x_1), f(x_2)])$$

핵심: $f$가 좋은 표현을 학습해야만 $y$(변환 관계)를 정확히 예측할 수 있다.

### 4.2 대표 방법

- **회전 예측** (Gidaris et al., 2018): 이미지를 {0, 90, 180, 270}도 회전시켜 4-class 분류
- **상대 위치 예측** (Doersch et al., 2015): 3$\times$3 그리드에서 중앙 패치 기준 주변 패치의 상대 위치(8가지) 예측

> **한계: Shortcut Learning.** 모델이 의미적 특징 대신 색수차(chromatic aberration)나 JPEG 아티팩트 같은 저수준 단서를 활용할 수 있다. 이 한계가 대조학습으로의 전환을 촉진하였다.

---

## 5. 거리 메트릭 학습 (Distance Metric Learning)

### 5.1 임베딩 공간에서의 거리

인코더 $f(\cdot; \theta)$로 임베딩 $e = f(x; \theta)$를 생성하고, 정규화된 임베딩 간 거리를 학습한다:

$$d(x, x'; \theta) = \left\| \frac{e}{\|e\|} - \frac{e'}{\|e'\|} \right\|^2 = 2 - 2 \cdot \text{cos}(e, e')$$

### 5.2 손실 함수의 발전

**Contrastive Loss** (Chopra et al., 2005):

$$\ell(x, x') = \begin{cases} \|e - e'\|^2 & \text{if } y = y' \\ (\epsilon - \|e - e'\|^2)_+ & \text{otherwise} \end{cases}$$

**Triplet Loss** (Schroff et al., 2015):

$$\ell(x, x^+, x^-) = \text{ReLU}(\|e - e^+\|^2 - \|e - e^-\|^2 + \epsilon)$$

**N-pair Loss / InfoNCE** (Sohn, 2016):

$$\ell(x, x^+, \{x_n^-\}_{n=1}^N) = -\log \frac{\exp(e^\top e^+)}{\exp(e^\top e^+) + \sum_n \exp(e^\top e_n^-)}$$

발전 방향: negative 샘플 수의 증가 ($1 \to 1 \to N$)에 따른 자연스러운 확장이다.

### 5.3 이론적 분해

N-pair loss를 Taylor 전개하면:

$$\ell \approx \underbrace{-e^\top e^+}_{\text{alignment}} + \underbrace{\mathbb{E}_{e'}[e^\top e']}_{\text{uniformity}} + \frac{1}{2}\text{Var}_{e'}[e^\top e'] + \cdots$$

이는 contrastive loss가 **alignment** (positive 쌍의 유사도 최대화)과 **uniformity** (임베딩의 균일 분포)를 동시에 최적화함을 보여준다 (Wang & Isola, 2020).

---

## 6. SimCLR

### 6.1 프레임워크

Chen et al. (2020)의 SimCLR은 4단계로 구성된다:

1. **증강 (Augmentation):** 동일 이미지 $x_0$에서 두 변환 $t, t' \sim \mathcal{T}$ 적용
2. **인코딩:** $(h, h') = (r(t(x_0)),\; r(t'(x_0)))$
3. **투사 (Projection):** $(z, z') = (g(h),\; g(h'))$
4. **대조 학습:** NT-Xent 손실 최적화

### 6.2 손실 함수 (NT-Xent)

배치 내 $N$개 이미지에서 $2N$개의 augmented 뷰를 생성한다. $(z_{2k-1}, z_{2k})$가 positive 쌍이다:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbf{1}(k \neq i) \cdot \exp(\text{sim}(z_i, z_k) / \tau)}$$

$$\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} [\ell_{2k-1, 2k} + \ell_{2k, 2k-1}]$$

여기서 $\text{sim}(z_i, z_j) = z_i^\top z_j / (\|z_i\| \|z_j\|)$이고, $\tau$는 temperature 파라미터이다.

### 6.3 핵심 설계 요소

| 요소 | 설명 |
|------|------|
| **Projection head** | $g(\cdot)$를 제거하면 성능 대폭 하락. 대조 손실이 augmentation-invariant 정보만 유지하도록 $z$를 압축하므로, $h$에 더 풍부한 정보가 보존됨 |
| **배치 크기** | 큰 배치 = 더 많은 negative = 더 정확한 추정 |
| **Temperature $\tau$** | 작은 $\tau$: hard negative에 집중, 큰 $\tau$: 균등 처리 |
| **Augmentation** | Random crop + color distortion이 핵심. Color histogram shortcut 방지 |

> **주의:** Downstream task에서는 projection head 출력 $z$가 아닌 인코더 출력 $h$를 사용한다.

---

## 7. Barlow Twins

### 7.1 핵심 아이디어

Zbontar et al. (2021)의 Barlow Twins는 **교차상관행렬(cross-correlation matrix)**을 단위행렬에 가깝게 만든다. Negative 샘플이 **불필요**하다.

### 7.2 수식

두 augmented 뷰의 표현 $\bar{r}, \bar{r}'$ (배치 정규화 적용 후)에 대해:

$$C = \frac{\bar{r}^\top \bar{r}'}{N} \in \mathbb{R}^{D \times D}$$

$$\mathcal{L} = \underbrace{\sum_i (1 - C_{ii})^2}_{\text{invariance term}} + \lambda \underbrace{\sum_i \sum_{j \neq i} C_{ij}^2}_{\text{redundancy reduction}}$$

$\lambda = 1$이면 $\mathcal{L} = \|C - I\|_F^2$ (Frobenius norm).

### 7.3 Collapse 방지 메커니즘

| 항 | 역할 |
|----|------|
| Invariance ($C_{ii} \to 1$) | 같은 이미지의 augmentation은 같은 표현 |
| Redundancy reduction ($C_{ij} \to 0$, $i \neq j$) | 각 차원이 독립적 정보를 인코딩하도록 강제 |

Invariance만 있으면 모든 표현이 상수로 **붕괴(collapse)**한다. Redundancy reduction이 이를 방지한다.

> **이론적 배경.** 신경과학자 Horace Barlow (1961)의 redundancy reduction principle에서 유래. 뇌의 감각 처리가 입력의 통계적 중복성을 제거하는 방향으로 진화했다는 가설.

---

## 8. CLIP (Contrastive Language-Image Pre-training)

### 8.1 구조

Radford et al. (2021)의 CLIP은 이미지-텍스트 쌍 4억 개로 학습한다:

- 이미지 인코더 $f_I$: 이미지 $\to$ 벡터 $I_i$
- 텍스트 인코더 $f_T$: 텍스트 $\to$ 벡터 $T_j$
- 유사도 행렬: $L_{ij} = \cos(I_i, T_j)$

### 8.2 손실 함수

$$\mathcal{L} = \sum_i -\log[\text{softmax}(L_{i,:})]_i + \sum_j -\log[\text{softmax}(L_{:,j})]_j$$

이미지$\to$텍스트 방향과 텍스트$\to$이미지 방향의 **대칭 cross-entropy**이다.

### 8.3 Zero-shot 분류

1. 각 클래스 이름을 "a photo of a {class}"로 변환하여 텍스트 인코딩 $T_k$ 생성
2. 테스트 이미지를 인코딩하여 $I$ 생성
3. 예측: $p(y_k \mid x) = \text{softmax}(\cos(I, T_{1:K}))_k$

고정된 클래스 집합에 국한되지 않는 **open-vocabulary** 분류가 가능하며, DALL-E, Stable Diffusion 등 text-to-image 생성 모델의 핵심 구성 요소로 활용된다.

---

## 9. 준지도학습과 메타학습

### 9.1 준지도학습 (Semi-Supervised Learning)

소량의 레이블 데이터 $\{(x_i, y_i)\}_{i=1}^l$과 대량의 비레이블 데이터 $\{x_j\}_{j=l+1}^{l+u}$ ($u \gg l$)를 함께 사용한다.

**Self-training (Pseudo-labeling):**
1. 레이블 데이터로 초기 모델 학습
2. 비레이블 데이터에 모델 예측으로 pseudo-label 생성
3. 높은 확신의 pseudo-label을 학습 데이터에 추가, 반복

**이론적 가정:** Smoothness (가까운 점 = 같은 레이블), Cluster (같은 클러스터 = 같은 레이블), Manifold (고차원 데이터가 저차원 매니폴드 위에 존재)

### 9.2 MAML (Model-Agnostic Meta-Learning)

Finn et al. (2017)의 MAML은 "학습하는 방법을 학습"한다.

**내부 루프 (task-specific 적응):**

$$\theta_j' = \theta - \alpha \nabla_\theta L_j(\theta)$$

**외부 루프 (메타 업데이트):**

$$\theta \leftarrow \theta - \beta \nabla_\theta \frac{1}{J} \sum_{j=1}^{J} L_j(\theta_j')$$

핵심: 외부 루프에서 내부 루프의 gradient step을 통해 미분하는 **이중 미분(second-order gradient)**이 필요하다. $\theta_j'$가 $\theta$의 함수이므로 chain rule이 적용된다.

MAML은 여러 task의 최적해 $\theta_1^*, \theta_2^*, \ldots$에 빠르게 도달할 수 있는 초기화 $\theta$를 찾는다.

---

## 10. SSL 방법론 비교 및 발전사

### 10.1 발전 타임라인

| 연도 | 방법 | 핵심 아이디어 |
|------|------|-------------|
| 2019 | MoCo | Momentum encoder + 큰 딕셔너리 |
| 2020 | SimCLR | 간단한 프레임워크, 큰 배치 의존 |
| 2020 | BYOL | Negative 없이 teacher-student |
| 2021 | Barlow Twins | 교차상관 행렬 $\to$ 단위행렬 |
| 2021 | CLIP | 이미지-텍스트 대조학습 |
| 2023 | DINOv2 | 자기증류 + ViT |

### 10.2 Collapse 방지 메커니즘 분류

| 방법 | 메커니즘 |
|------|---------|
| **Contrastive** (SimCLR) | Negative repulsion으로 직접 방지 |
| **Asymmetric architecture** (BYOL, SimSiam) | Predictor + stop gradient |
| **Feature decorrelation** (Barlow Twins, VICReg) | 차원 간 독립성 강제 |
| **Clustering** (SwAV) | 프로토타입 할당의 균등화 |

---

## 11. 흔한 오해와 주의점

| # | 오해 | 올바른 이해 |
|---|------|-------------|
| 1 | SSL은 레이블이 전혀 없는 학습이다 | 레이블이 있지만 사람이 아닌 **알고리즘이 자동 생성** |
| 2 | SimCLR의 최종 표현은 projection head 출력이다 | Downstream에서는 인코더 출력 $h$를 사용. $z$는 학습 시에만 사용 |
| 3 | Barlow Twins는 대조학습이다 | Non-contrastive. Negative 쌍을 사용하지 않음 |
| 4 | CLIP은 이미지 분류 전용이다 | 범용 이미지-텍스트 공동 임베딩. 검색, 생성 등에도 활용 |
| 5 | Triplet loss와 N-pair loss는 완전히 다르다 | N-pair loss는 triplet loss의 일반화. 활성화 함수만 다름 |
| 6 | MAML은 특정 모델 구조에만 적용 가능 | Model-**Agnostic**. 미분 가능한 모든 모델에 적용 |

---

## 12. 핵심 요약

| 개념 | 핵심 |
|------|------|
| SSL | 데이터 자체의 구조에서 감독 신호를 자동 생성하여 범용 표현 학습 |
| MAE | 75% 마스킹 + ViT 인코더. 높은 마스킹 비율이 의미 있는 표현 유도 |
| 프록시 과제 | 회전/위치 예측 등. Shortcut learning이 한계 |
| 메트릭 학습 | Contrastive $\to$ Triplet $\to$ N-pair (InfoNCE). Alignment + Uniformity |
| SimCLR | NT-Xent 손실 + projection head + 큰 배치. $\ell = -\log \frac{\exp(\text{sim}/\tau)}{\sum \exp(\text{sim}/\tau)}$ |
| Barlow Twins | $\mathcal{L} = \|C - I\|_F^2$. Negative-free, redundancy reduction |
| CLIP | 4억 이미지-텍스트 쌍. Open-vocabulary zero-shot. 대칭 cross-entropy |
| MAML | 이중 미분으로 빠른 적응이 가능한 초기화를 메타학습 |

**관련 개념 연결:** Pre-training $\to$ [17장] LLM, Contrastive learning의 temperature $\to$ [19장] EBM의 에너지 함수, MAE의 ViT 기반 $\to$ [16장] Transformer

**참고 문헌:**
- Chen et al., "A Simple Framework for Contrastive Learning," ICML 2020 (SimCLR)
- He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022
- Radford et al., "Learning Transferable Visual Models," ICML 2021 (CLIP)
- Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction," ICML 2021
- Finn et al., "Model-Agnostic Meta-Learning," ICML 2017 (MAML)

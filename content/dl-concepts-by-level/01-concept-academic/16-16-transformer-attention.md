---
title: "16. Transformer와 Attention 메커니즘"
slug: 16-transformer-attention
order: 16
---

# 16. Transformer와 Attention 메커니즘

## 1. 동기부여 및 개요

순환 신경망(RNN)은 시퀀스를 순차적으로 처리하므로 두 가지 근본적 한계가 있다.
첫째, 병렬화가 불가능하여 학습 속도가 느리다. 둘째, 긴 시퀀스에서 기울기 소실로 인해
장거리 의존성(long-range dependency) 포착이 어렵다.

Vaswani et al. (2017)의 "Attention Is All You Need"는 순환 구조를 완전히 제거하고
**Attention 메커니즘만으로** 시퀀스를 처리하는 Transformer를 제안하였다.
이 아키텍처는 병렬 연산, 장거리 의존성 포착, 확장성(scalability)을 동시에 달성하며,
GPT, BERT, ViT 등 현대 딥러닝의 거의 모든 기반 모델의 토대가 되었다.

```
선형대수 (행렬곱, 내적)
       |
       v
Scaled Dot-Product Attention ---> Multi-Head Attention
       |                                  |
  Softmax 함수                     Positional Encoding
       |                                  |
       +-------> Transformer Encoder -----+
                      |
                Transformer Decoder (+ Masked/Cross Attention)
                      |
              +-------+-------+
              |               |
         BERT/GPT         ViT (Vision)
```

---

## 2. Attention: Soft Dictionary Lookup

### 2.1 정의와 직관

**Definition 2.1 (Attention).** Query $q \in \mathbb{R}^{d_k}$, Key 집합 $\{k_j\}_{j=1}^{N} \subset \mathbb{R}^{d_k}$, Value 집합 $\{v_j\}_{j=1}^{N} \subset \mathbb{R}^{d_v}$가 주어졌을 때, Attention 출력은 다음과 같다:

$$\text{output} = \sum_{j=1}^{N} \alpha_j \, v_j, \quad \alpha_j = \frac{\exp(a(q, k_j))}{\sum_{l=1}^{N} \exp(a(q, k_l))}$$

여기서 $a: \mathbb{R}^{d_k} \times \mathbb{R}^{d_k} \to \mathbb{R}$는 유사도 함수(scoring function)이다.

**직관적 해석.** 데이터베이스에서 key로 record를 조회하는 hard lookup은 정확히 일치하는
key 하나만 반환한다. Attention은 이를 **미분 가능한 soft lookup**으로 확장한다.
모든 key와의 유사도를 확률 분포로 변환하고, 그 가중치로 value들의 **볼록 결합(convex combination)**을 구한다.

| 비교 항목 | Hard Lookup | Soft Lookup (Attention) |
|-----------|-------------|-------------------------|
| 수식 | $\sum_j \mathbf{1}(q=k_j) v_j$ | $\sum_j \alpha_j v_j$ |
| 미분 가능 | X | O |
| 출력 | 단일 value | value들의 가중 합 |

> **비모수 관점.** Attention은 Nadaraya-Watson 커널 회귀 $f(x) = \sum_j \alpha_j y_j$, $\alpha_j \propto K(x, x_j)$와 동일한 구조를 갖는다. 이 의미에서 Attention은 kernel smoother의 parametric 일반화이다.

---

## 3. Scaled Dot-Product Attention

### 3.1 정의

**Definition 3.1 (Scaled Dot-Product Attention).**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

여기서 $Q \in \mathbb{R}^{N \times d_k}$, $K \in \mathbb{R}^{N \times d_k}$, $V \in \mathbb{R}^{N \times d_v}$이며, softmax는 행(row) 단위로 적용된다.

### 3.2 스케일링의 수학적 정당성

**Theorem 3.1 (Scaling Factor의 필요성).**
$q_i, k_j$의 각 원소가 i.i.d.로 평균 0, 분산 1이면, 내적 $q_i^\top k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$의 분산은 $d_k$이다.

*증명 스케치.* $\text{Var}(q_{il} k_{jl}) = \mathbb{E}[q_{il}^2]\mathbb{E}[k_{jl}^2] = 1$이고, 독립인 $d_k$개 항의 합이므로 $\text{Var}(q_i^\top k_j) = d_k$. $\square$

$d_k$가 클 때 스케일링 없이 softmax를 적용하면 입력값이 극단적으로 커져 출력이 one-hot에 가까워지고, 기울기가 거의 0이 된다. $\sqrt{d_k}$로 나누면 분산이 1로 정규화되어 학습 초기의 기울기 흐름이 안정화된다.

> **참고.** Additive attention $a(q,k) = w_v^\top \tanh(W_q q + W_k k)$도 존재하나, dot-product attention이 행렬곱으로 효율적으로 구현되어 실무에서 표준이다.

### 3.3 시간 복잡도

$QK^\top$ 계산에 $O(N^2 d_k)$, $AV$ 곱에 $O(N^2 d_v)$이 소요되어 전체 $O(N^2 d)$이다.
이 $N^2$ 복잡도가 Transformer의 주요 병목이며, Performer (Choromanski et al., 2021) 등은
$\tilde{A}_{ij} = \phi(q_i)^\top \phi(k_j)$로 저랭크 근사하여 $O(N)$으로 줄인다.

---

## 4. Multi-Head Attention (MHA)

### 4.1 정의

**Definition 4.1 (Multi-Head Attention).**

$$\text{head}_i = \text{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V), \quad i = 1, \ldots, h$$

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O$$

여기서 $W_i^Q, W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$, $W^O \in \mathbb{R}^{hd_v \times d}$이다.

원 논문 기준: $d = 512$, $h = 8$, $d_k = d_v = d/h = 64$.

### 4.2 왜 여러 Head가 필요한가

각 head는 서로 다른 가중치 행렬 $\Sigma_i = W_i^Q W_i^{K\top} \in \mathbb{R}^{d \times d}$를 학습한다.
이 $\Sigma_i$가 서로 다른 **유사도 개념(notion of similarity)**을 정의하므로,
각 head는 "주어-동사 관계", "형용사-명사 수식", "시제 일치" 등
서로 다른 관계 패턴을 독립적으로 포착한다.

**핵심 차별점.** MHA를 단순 선형 변환 $XW$로 대체하면 입력 의존적 동적 가중치 결정이 사라진다. Self-Attention의 핵심 가치는 바로 이 **동적(dynamic), 입력 의존적(input-dependent)** 가중치에 있다.

---

## 5. Positional Encoding

### 5.1 필요성

**Theorem 5.1 (Permutation Equivariance).**
Self-Attention 연산 $f(X) = \text{softmax}(XW^Q W^{K\top} X^\top / \sqrt{d_k}) \, X W^V$는
순열 등변(permutation equivariant)이다: $f(\Pi X) = \Pi f(X)$.

*증명 스케치.* $A = \text{softmax}((\Pi X) W^Q W^{K\top} (\Pi X)^\top / \sqrt{d_k}) = \Pi A_0 \Pi^\top$이므로, 출력은 $\Pi A_0 \Pi^\top \cdot \Pi X W^V = \Pi A_0 X W^V = \Pi f(X)$. $\square$

따라서 순서 정보를 주입하지 않으면, 시퀀스 문제에서 핵심적인 어순 정보가 완전히 무시된다.

### 5.2 Sinusoidal Positional Encoding

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

위치 $pos$에 대해 $d$차원 벡터를 생성하여 임베딩에 더한다: $\text{PE}(X) = X + P$.

| 장점 | 설명 |
|------|------|
| 비학습(deterministic) | 추가 파라미터 없음 |
| 외삽(extrapolation) | 학습 시보다 긴 시퀀스에 적용 가능 |
| 상대 위치 학습 | $\text{PE}(pos+k)$가 $\text{PE}(pos)$의 선형 변환으로 표현 |

> **최신 대안.** Learnable positional embedding (BERT), RoPE (Rotary Position Embedding), ALiBi 등이 있다. RoPE는 회전 행렬 $R_\Theta^m$을 곱해 상대 위치를 내적에 직접 인코딩한다.

---

## 6. Transformer Encoder

### 6.1 구조

하나의 Encoder Layer는 다음 두 단계로 구성된다:

$$x \leftarrow \text{LN}(x + \text{MHA}(x, x, x)) \quad \text{(Self-Attention + Add\&Norm)}$$
$$x \leftarrow \text{LN}(x + \text{FFN}(x)) \quad \text{(Feed-Forward + Add\&Norm)}$$

여기서 FFN은 2층 MLP이다: $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$, 차원 $d(512) \to d_{ff}(2048) \to d(512)$.

Self-Attention에서 Q, K, V는 **모두 같은 입력 $X$**에서 나온다: MHA($X, X, X$).

```python
# 개념 설명용 의사코드
class TransformerEncoderLayer:
    def __init__(self, d_model=512, nhead=8, d_ff=2048):
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.ffn = Sequential(Linear(d_model, d_ff), ReLU(), Linear(d_ff, d_model))
        self.norm1, self.norm2 = LayerNorm(d_model), LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x, x, x))   # Self-Attention
        x = self.norm2(x + self.ffn(x))                 # Feed-Forward
        return x
```

### 6.2 설계 선택: LayerNorm vs BatchNorm

NLP에서 LayerNorm이 BatchNorm보다 효과적이다. BatchNorm은 배치 내 시퀀스 길이가 다르면 통계량이 불안정하지만, LayerNorm은 각 샘플의 특성(feature) 차원에서 정규화하므로 시퀀스 길이에 무관하다.

---

## 7. Transformer Decoder와 Cross-Attention

### 7.1 Decoder 구조

Decoder Layer는 세 단계로 구성된다:

$$y \leftarrow \text{LN}(y + \text{MaskedMHA}(y, y, y))$$
$$y \leftarrow \text{LN}(y + \text{MHA}(y, M, M)) \quad \text{(Cross-Attention, } M = \text{Encoder 출력)}$$
$$y \leftarrow \text{LN}(y + \text{FFN}(y))$$

### 7.2 Masked Self-Attention

**Definition 7.1 (Causal Mask).**

$$[QK^\top]_{ij} = \begin{cases} q_i^\top k_j & j \le i \\ -\infty & j > i \end{cases}$$

Softmax 적용 후 $j > i$인 위치는 0이 되어, 출력 $i$는 과거 입력 $j \le i$에만 의존한다.
이 **하삼각 행렬(lower triangular)** 마스크가 autoregressive 생성의 핵심이다.

### 7.3 Self-Attention vs Cross-Attention 비교

| | Self-Attention | Cross-Attention |
|---|---|---|
| 호출 | MHA($X, X, X$) | MHA($Y, M, M$) |
| 의미 | 시퀀스 내부 관계 학습 | 두 시퀀스 간 정렬(alignment) |
| Attention 행렬 | $N \times N$ | $N_{dec} \times N_{enc}$ |
| 위치 | Encoder/Decoder 1층 | Decoder 2층 |

Cross-Attention에서 Q는 Decoder 상태, K와 V는 Encoder 출력(memory)이다. 이 구조는 전통적 seq2seq에서 encoder hidden state를 하나의 context vector로 요약하던 병목을 해소한다.

### 7.4 Teacher Forcing

학습 시 이전 출력 대신 **정답(ground truth)** $y_{1:t-1}$을 입력한다:

$$p(y_{1:m}) = \prod_{t=1}^{m} p(y_t \mid y_{1:t-1})$$

---

## 8. Vision Transformer (ViT)

Dosovitskiy et al. (2021)은 이미지를 $P \times P$ 패치로 분할하고, 각 패치를 선형 투영하여 Transformer Encoder에 입력하는 **ViT**를 제안하였다.

**구조:** 패치 분할 $\to$ Linear Projection $\to$ [CLS] 토큰 + Positional Embedding $\to$ Transformer Encoder $\to$ MLP Head (분류)

| 비교 | CNN | ViT |
|------|-----|-----|
| 수용 영역 | Local (convolution kernel) | Global (self-attention) |
| Inductive bias | 강함 (locality, translation equivariance) | 약함 |
| 소규모 데이터 | 우수 | 열등 |
| 대규모 데이터 (JFT-300M) | 열등 | 우수 |

> **핵심 관찰.** 약한 inductive bias + 대규모 데이터 $>$ 강한 inductive bias + 소규모 데이터. 이는 bias-variance tradeoff의 실증적 사례이다.

---

## 9. 언어 모델: BERT와 GPT

### 9.1 BERT (Devlin et al., 2019)

- **아키텍처:** Transformer **Encoder** (양방향)
- **사전학습 목표:** Masked Language Model (MLM) + Next Sentence Prediction
- $L_{\text{MLM}} = -\mathbb{E}_m \sum_{i \in m} \log p(x_i \mid x_{-m}; \theta)$
- 12층, $d=768$, $h=12$, 110M 파라미터

### 9.2 GPT 계열 (Radford et al., 2018~)

- **아키텍처:** Transformer **Decoder** (단방향, causal mask)
- **사전학습 목표:** Causal Language Model
- $p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})$
- GPT-3: 175B 파라미터, in-context learning 능력 발현

### 9.3 패러다임: Pre-train + Fine-tune

대규모 비라벨 텍스트로 사전학습하여 범용 표현을 얻고, 소규모 라벨 데이터로 미세조정한다. 이 2단계 패러다임이 현대 NLP의 표준이다.

---

## 10. 흔한 오해와 주의점

| # | 오해 | 올바른 이해 |
|---|------|-------------|
| 1 | Q, K, V는 서로 다른 입력에서 온다 | Self-Attention에서는 모두 같은 $X$에서 유래. Cross-Attention 한정으로 다른 입력 사용 |
| 2 | $\sqrt{d_k}$는 단순 정규화 | 내적 분산이 $d_k$에 비례하는 문제 해결. Softmax gradient 소실 방지가 핵심 |
| 3 | Multi-Head는 같은 연산의 반복 | 각 head는 다른 $W_i^Q, W_i^K, W_i^V$를 학습하여 다른 관계 패턴을 포착 |
| 4 | Transformer가 RNN을 완전히 대체 | $O(N^2)$ 복잡도 문제. Mamba 등 SSM 계열 ($O(N)$)이 활발히 연구 중 |
| 5 | ViT는 항상 CNN보다 우수 | 소규모 데이터에서는 CNN이 더 우수. 대규모 데이터가 있어야 ViT가 유리 |
| 6 | Masked Attention은 학습 시에만 사용 | 추론 시에도 autoregressive 생성에 필수 (KV-cache와 함께 사용) |

---

## 11. 핵심 요약

| 개념 | 핵심 정리 |
|------|-----------|
| Attention | Soft dictionary lookup. $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Scaling | $\sqrt{d_k}$로 나누어 내적 분산을 1로 정규화 $\to$ gradient 안정화 |
| Multi-Head | $h$개 독립 attention을 병렬 실행. $\Sigma_i = W_i^Q W_i^{K\top}$이 각 head의 "관점" |
| Positional Encoding | Attention은 permutation equivariant $\to$ 순서 정보 주입 필수 |
| Encoder | Self-Attn + FFN + Add&Norm $\times N$층. 입력의 contextual representation 생성 |
| Decoder | Masked Self-Attn + Cross-Attn + FFN $\times N$층. Autoregressive 생성 |
| ViT | 이미지 = 패치 시퀀스. 대규모 데이터에서 weak bias(Transformer) $>$ strong bias(CNN) |
| BERT vs GPT | BERT: fill-in-the-blank (Encoder), GPT: next-token (Decoder) |

**관련 개념 연결:** Positional Encoding $\to$ [17장] LLM의 context window 확장, Self-Attention $\to$ [18장] 자기지도학습에서의 표현 학습, Autoregressive 생성 $\to$ [19장] 생성모델, [20장] 확산모델

**참고 문헌:**
- Vaswani et al., "Attention Is All You Need," NeurIPS 2017
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," NAACL 2019
- Dosovitskiy et al., "An Image is Worth 16x16 Words," ICLR 2021

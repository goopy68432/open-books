---
title: "15. 순환 신경망과 시퀀스 모델"
slug: 15-rnn-sequence-models
order: 15
---

# 15. 순환 신경망과 시퀀스 모델

## 15.1 동기부여 및 개요

언어, 음성, 시계열, 음악 등 데이터의 상당수는 **순서(sequence)**가 핵심이다. "I ate an apple"과 "apple an ate I"는 같은 단어로 구성되지만 전혀 다른 의미를 가진다. MLP와 CNN은 고정 크기 입력만 처리할 수 있어 **가변 길이 시퀀스**의 **시간적 의존성(temporal dependency)**을 자연스럽게 모델링할 수 없다.

순환 신경망(RNN)은 **은닉 상태(hidden state)**라는 메모리를 통해 과거 정보를 축적하며 시퀀스를 처리하는 기본 구조이다. LSTM/GRU는 기울기 소실 문제를 게이트 메커니즘으로 극복한 핵심 발전이며, Transformer는 어텐션으로 장기 의존성을 근본적으로 해결하였다.

**연결**: 14장의 CNN이 **공간적** 귀납적 편향을 인코딩한다면, RNN은 **시간적** 귀납적 편향(파라미터 공유, 순차 처리)을 인코딩한다. 12장의 역전파가 시간축으로 확장된 것이 BPTT이다.

---

## 15.2 텍스트 데이터 표현과 워드 임베딩

### 15.2.1 원-핫 인코딩의 한계

어휘 크기가 $V$인 사전에서 각 단어를 $e_i \in \{0,1\}^V$ (원-핫 벡터)로 표현하면:
- 차원이 매우 높다 ($V \sim 10^4 \text{--} 10^5$)
- 임의의 두 단어 간 거리가 동일하다: $\|e_i - e_j\|_2 = \sqrt{2}$ ($i \neq j$)
- 의미적 유사도 정보가 전혀 없다

### 15.2.2 워드 임베딩

**정의 15.1 (워드 임베딩).** 임베딩 행렬 $E \in \mathbb{R}^{K \times V}$ ($K \ll V$)를 통해:

$$e_{nt} = E x_{nt} \in \mathbb{R}^K$$

여기서 $x_{nt}$는 원-핫 벡터이다. 이 연산은 $E$의 해당 열을 조회(lookup)하는 것과 동치이다.

### 15.2.3 Word2Vec (Mikolov et al., 2013)

**분포 가설(Distributional Hypothesis)**: 비슷한 문맥에 등장하는 단어는 의미가 비슷하다.

**Skip-gram 모델**: 중심 단어 $w_t$로 문맥 창(context window) $J_t = [t-m, t+m] \setminus \{t\}$ 내의 주변 단어를 예측한다:

$$\log p(w_{1:T}) = \sum_{t=1}^{T}\sum_{j \in J_t} \log \frac{\exp(v(w_j)^\top u(w_t))}{\sum_{w' \in V}\exp(v(w')^\top u(w_t))}$$

여기서 $u$는 입력 임베딩, $v$는 출력 임베딩이다.

**CBOW 모델**: 주변 단어의 평균 벡터 $\bar{v}_t = \frac{1}{2m}\sum_{j \in J_t} v(w_j)$로 중심 단어를 예측한다.

유명한 벡터 산술: $\vec{\text{King}} - \vec{\text{Man}} + \vec{\text{Woman}} \approx \vec{\text{Queen}}$

**한계**: Word2Vec은 **정적(static) 임베딩**으로, 동음이의어("bank" = 은행/강둑)에 대해 문맥 무관한 단일 벡터를 할당한다. GPT/BERT 등의 **문맥적(contextual) 임베딩**은 이 한계를 극복한다.

---

## 15.3 확률 그래프 모델과 시퀀스

### 15.3.1 PGM에서의 결합분포 분해

**정의 15.2 (확률 그래프 모델, PGM).** 확률 변수들의 조건부 독립(CI) 관계를 DAG로 인코딩한다. DAG 구조에 따라 결합분포를 인수분해할 수 있다:

$$p(Y_{1:V}) = \prod_{i=1}^{V} p(Y_i \mid Y_{\text{pa}(i)})$$

여기서 $\text{pa}(i)$는 노드 $i$의 부모 집합이다.

### 15.3.2 마르코프 체인

**정의 15.3 (1차 마르코프 성질).** 시퀀스 $y_{1:T}$에 대해:

$$p(y_t \mid y_{1:t-1}) = p(y_t \mid y_{t-1})$$

이로부터 결합분포는 $p(y_{1:T}) = p(y_1)\prod_{t=2}^{T} p(y_t \mid y_{t-1})$으로 분해된다.

**한계**: 실제 언어에서 1차 마르코프 가정은 지나치게 강하다. "I ate an ___"과 "I built an ___"에서 빈칸의 확률 분포가 달라야 하지만, 1차 마르코프 모델은 직전 단어 "an"만 참조하므로 동일한 분포를 예측한다.

고차 마르코프 모델 $p(y_t \mid y_{t-1}, \ldots, y_{t-k})$은 파라미터 수가 $|V|^{k+1}$로 지수 폭발한다. RNN은 은닉 상태를 통해 **이론적으로 무한 기억**을 가지면서도 파라미터는 고정인 구조를 제공한다.

---

## 15.4 순환 신경망 (RNN)

### 15.4.1 기본 구조

**정의 15.4 (Elman RNN).** 시간 $t$에서의 은닉 상태 업데이트:

$$h_t = \phi(W_{hx} x_t + W_{hh} h_{t-1} + b_h)$$

여기서 $\phi$는 활성화 함수(보통 $\tanh$), $W_{hx} \in \mathbb{R}^{d_h \times d_x}$, $W_{hh} \in \mathbb{R}^{d_h \times d_h}$이다.

출력: $p(y_t \mid h_t) = \text{Cat}(y_t \mid \text{softmax}(W_{oh} h_t + b_o))$

**핵심 특성**: 모든 시간 단계에서 **동일한 파라미터** $(W_{hx}, W_{hh}, W_{oh})$를 공유한다. 이것이 RNN의 귀납적 편향이며, 시퀀스 길이가 달라도 같은 모델을 적용할 수 있는 이유이다.

```
RNN 펼침(Unrolling):

  x₁      x₂      x₃      x₄
  ↓       ↓       ↓       ↓
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ RNN │→│ RNN │→│ RNN │→│ RNN │
│ (W) │ │ (W) │ │ (W) │ │ (W) │  ← 같은 W
└──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
   ↓       ↓       ↓       ↓
  y₁      y₂      y₃      y₄
```

### 15.4.2 구조적 변형

| 변형 | 구조 | 응용 |
|------|------|------|
| **Vec2Seq** | 벡터 $x$ → 시퀀스 $y_{1:T}$ | 이미지 캡셔닝, 언어 모델 |
| **Seq2Vec** | 시퀀스 $x_{1:T}$ → 벡터 $h_T$ | 감성 분석, 문서 분류 |
| **Seq2Seq (aligned)** | 입출력 길이 동일 ($T = T'$) | 품사 태깅, NER |
| **Seq2Seq (unaligned)** | 인코더-디코더 구조 | 기계 번역, 요약 |
| **Bidirectional** | 정방향 + 역방향 결합 | 인코더 측 표현 학습 |

### 15.4.3 Seq2Vec

$$h_t = \phi(W[x_t; h_{t-1}]), \quad p(y \mid x_{1:T}) = \text{Cat}(y \mid \text{softmax}(W' h_T))$$

마지막 은닉 상태 $h_T$가 전체 시퀀스의 요약(summary)으로 사용된다.

### 15.4.4 양방향 RNN (Bidirectional RNN)

$$\vec{h}_t = \phi(W^{\to}[x_t; \vec{h}_{t-1}]), \quad \overleftarrow{h}_t = \phi(W^{\leftarrow}[x_t; \overleftarrow{h}_{t+1}])$$
$$h_t = [\vec{h}_t; \overleftarrow{h}_t]$$

**주의**: 양방향 RNN은 전체 시퀀스가 주어진 경우에만 사용 가능하므로, 자기회귀 생성 모델에는 부적합하고 **인코더** 측에서 주로 사용된다.

### 15.4.5 인코더-디코더 (Seq2Seq)

**정의 15.5 (Encoder-Decoder, Sutskever et al. 2014; Cho et al. 2014).**

- 인코더: $h_t^e = \phi(W^e[x_t; h_{t-1}^e])$, 컨텍스트 벡터 $c = h_T^e$
- 디코더: $h_t^d = \phi(W^d[c; y_{t-1}; h_{t-1}^d])$
- 출력: $p(y_t \mid h_t^d) = \text{Cat}(y_t \mid \text{softmax}(W' h_t^d))$

**정보 병목(Information Bottleneck)**: 입력 시퀀스의 모든 정보를 고정 크기 벡터 $c \in \mathbb{R}^d$에 압축해야 하므로, 입력이 길어질수록 정보 손실이 커진다. 이것이 **어텐션 메커니즘**의 핵심 동기이다.

---

## 15.5 시간 역전파 (BPTT)

### 15.5.1 기본 원리

**정의 15.6 (Backpropagation Through Time).** RNN을 시간축으로 펼친(unroll) 후 표준 역전파를 적용한다. 공유 파라미터 $W$에 대한 총 기울기는 각 시간 단계의 기울기의 합이다:

$$\frac{\partial L}{\partial W} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W}$$

### 15.5.2 시간축 기울기의 재귀 구조

연쇄 법칙을 시간축으로 적용하면:

$$\frac{\partial L_t}{\partial h_k} = \frac{\partial L_t}{\partial h_t} \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

여기서 각 항은:

$$\frac{\partial h_i}{\partial h_{i-1}} = W_{hh}^\top \cdot \text{diag}(\phi'(W_{hh} h_{i-1} + W_{hx} x_i))$$

이 **행렬 곱의 연쇄**가 기울기 소실/폭발 문제의 근원이다.

**Truncated BPTT**: 전체 시퀀스 대신 고정 길이 $k$만큼만 역전파하여 계산 비용을 절감. 단, $k$보다 긴 의존성은 학습할 수 없다.

---

## 15.6 기울기 소실과 폭발

### 15.6.1 문제의 수학적 분석

**정리 15.1 (비형식적, Bengio et al. 1994).** Vanilla RNN에서 $h_t = \tanh(W_{hh}h_{t-1} + W_{hx}x_t)$일 때:

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| = \left\|\prod_{i=k+1}^{t} W_{hh}^\top \cdot \text{diag}(\tanh'(\cdot))\right\|$$

$\tanh' \in (0, 1]$이므로:

- $W_{hh}$의 최대 특이값 $\sigma_{\max} < 1$: 기울기가 **지수적으로 감소** (소실)
- $\sigma_{\max} > 1$: 기울기가 **지수적으로 증가** (폭발)

### 15.6.2 해결 전략

| 문제 | 해결책 | 원리 |
|------|--------|------|
| 폭발 | Gradient Clipping | $g \leftarrow \min(1, \theta/\|g\|) \cdot g$ |
| 소실 | LSTM/GRU | 가산적 상태 업데이트로 기울기 경로에 항등 매핑 포함 |
| 소실 | 직교 초기화 | $W_{hh}$를 직교 행렬로 초기화하여 특이값 $\approx 1$ 유지 |
| 근본적 | Transformer | 임의 거리의 토큰을 직접 참조 (self-attention) |

---

## 15.7 LSTM (Long Short-Term Memory)

### 15.7.1 게이트 메커니즘

**정의 15.7 (LSTM, Hochreiter & Schmidhuber 1997).** 세 개의 게이트와 셀 상태:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate: 무엇을 잊을지)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input gate: 무엇을 기억할지)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \quad \text{(후보 기억)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(셀 상태 업데이트)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output gate: 무엇을 출력할지)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(은닉 상태)}$$

여기서 $\odot$는 원소별 곱(Hadamard product), $\sigma$는 시그모이드 함수이다.

### 15.7.2 기울기 소실 완화의 원리

**보조정리 15.1.** 셀 상태의 기울기:

$$\frac{\partial c_t}{\partial c_{t-1}} = \text{diag}(f_t)$$

$f_t \approx 1$이면 기울기가 거의 그대로 전파되어 소실이 완화된다. 이는 ResNet의 skip connection $y = F(x) + x$에서 $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$와 **정확히 같은 원리**이다.

셀 상태 $c_t$는 **CEC(Constant Error Carousel)** 역할을 하여, 기울기가 변형 없이 시간축을 따라 흐를 수 있는 "고속도로"를 제공한다.

**실전 팁**: Forget gate의 바이어스를 양수(예: $b_f = 1$)로 초기화하면, 초기에 $f_t \approx 1$이 되어 장기 기억 유지에 유리하다.

---

## 15.8 GRU (Gated Recurrent Unit)

### 15.8.1 정의

**정의 15.8 (GRU, Cho et al. 2014).** LSTM의 경량화 버전, 두 개의 게이트:

$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(Reset gate)}$$
$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(Update gate)}$$
$$\tilde{h}_t = \tanh(W[r_t \odot h_{t-1}, x_t]) \quad \text{(후보 은닉 상태)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(은닉 상태 업데이트)}$$

### 15.8.2 LSTM vs GRU 비교

| 특성 | LSTM | GRU |
|------|------|-----|
| 게이트 수 | 3 (forget, input, output) | 2 (reset, update) |
| 상태 | $c_t$ (셀) + $h_t$ (은닉) | $h_t$ (은닉)만 |
| 파라미터 | 더 많음 | 더 적음 ($\sim$75%) |
| 출력 제어 | output gate로 명시적 | 없음 |
| 성능 | 복잡한 장기 의존성에 유리 | 데이터가 적을 때 유리 |

GRU의 업데이트 게이트 $z_t$는 LSTM의 forget gate와 input gate를 하나로 합친 것이다: $(1-z_t)$가 forget, $z_t$가 input 역할. 두 모델 모두 **가산적 상태 업데이트**로 기울기 소실을 완화한다.

---

## 15.9 RNN에서 Transformer로

### 15.9.1 RNN의 근본적 한계

| 한계 | 설명 |
|------|------|
| 순차적 처리 | $h_t$는 $h_{t-1}$에 의존 $\Rightarrow$ GPU 병렬화 불가 |
| 정보 병목 | 인코더-디코더에서 고정 크기 $c$에 압축 |
| 장기 의존성 | 기울기 소실로 먼 과거 학습 어려움 (LSTM/GRU도 완전 해결은 못 함) |

### 15.9.2 Transformer의 어텐션

**정의 15.9 (Scaled Dot-Product Attention, Vaswani et al. 2017).**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

여기서 $Q \in \mathbb{R}^{T \times d_k}$ (Query), $K \in \mathbb{R}^{T \times d_k}$ (Key), $V \in \mathbb{R}^{T \times d_v}$ (Value).

**Multi-Head Attention**: 여러 어텐션 헤드를 병렬로 수행하여 다양한 관계 패턴을 포착한다.

### 15.9.3 RNN vs Transformer 비교

| 특성 | RNN | Transformer |
|------|-----|-------------|
| 수식 | $h_t = \phi(W[x_t; h_{t-1}])$ | $h^\ell = A(h^{\ell-1}) \cdot h^{\ell-1}$ |
| 순차적 편향 | 강함 (재귀 구조) | 약함 (위치 인코딩으로 순서 부여) |
| 참조 범위 | 은닉 상태를 통한 간접 참조 | 모든 토큰의 직접 참조 ($O(T^2)$) |
| 병렬화 | 불가 (순차 의존) | 가능 (GPU 최적) |
| 장기 의존성 | 어려움 (기울기 소실) | 용이 (직접 연결) |

**발전 계보**: 마르코프 모델 $\to$ RNN/LSTM $\to$ Attention/Transformer. 최근 Mamba(S4 계열) 등 **상태공간 모델(SSM)**이 RNN의 선형 시간 복잡도와 Transformer의 표현력을 결합하려는 시도로 주목받고 있다.

---

## 15.10 흔한 오해와 주의점

| 오해 | 올바른 이해 |
|------|-----------|
| "RNN은 가변 길이 처리를 위해 필요하다" | RNN의 핵심은 **파라미터 공유와 은닉 상태를 통한 시간적 정보 축적**이다 |
| "LSTM의 셀=장기기억, 은닉=단기기억" | 셀 상태는 **기울기 고속도로** 역할, 은닉 상태는 **필터링된 출력** |
| "기울기 소실 = 기울기가 0이 된다" | 정확히 0이 아니라 **지수적으로 작아져** 학습 신호가 무의미해짐 |
| "양방향 RNN은 항상 더 좋다" | 실시간 생성에서는 미래 정보를 볼 수 없으므로 **단방향만 가능** |
| "GRU는 LSTM의 열등한 버전이다" | 파라미터가 적어 학습이 빠르고, 많은 벤치마크에서 유사한 성능 |
| "BPTT는 별도의 알고리즘이다" | RNN을 시간축으로 **펼친 후 표준 역전파**를 적용하는 것 |

---

## 15.11 핵심 요약

| 개념 | 핵심 수식 | 의미 |
|------|---------|------|
| 워드 임베딩 | $e = Ex$, $E \in \mathbb{R}^{K \times V}$ | 원-핫의 차원 저주를 연속 벡터로 해결 |
| 마르코프 체인 | $p(y_{1:T}) = \prod_t p(y_t \mid y_{t-1})$ | 시퀀스의 조건부 독립 가정 |
| RNN | $h_t = \phi(W[x_t; h_{t-1}])$ | 은닉 상태가 가변 길이 시퀀스의 메모리 |
| BPTT | $\frac{\partial L}{\partial W} = \sum_t \frac{\partial L_t}{\partial W}$ | RNN을 펼쳐서 표준 역전파 적용 |
| 기울기 소실 | $\prod_{i=k+1}^{t} W_{hh}^\top \text{diag}(\phi')$ | $\sigma_{\max}(W_{hh}) < 1$이면 지수 감쇠 |
| LSTM | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ | 가산적 업데이트 + 3개 게이트로 장기 기억 |
| GRU | $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$ | LSTM 경량화: 2개 게이트, 단일 상태 |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k}) V$ | 모든 토큰 간 직접 참조로 장기 의존성 해결 |

> **핵심 메시지**: RNN의 본질은 **파라미터 공유** + **은닉 상태를 통한 시간 재귀**이다. LSTM/GRU는 **가산적 상태 업데이트**로 기울기 소실을 완화한다. Transformer는 RNN의 **순차적 병목**과 **장기 의존성 문제**를 어텐션으로 근본적으로 해결했다.

**참고문헌**: Hochreiter & Schmidhuber (1997, LSTM), Cho et al. (2014, GRU), Sutskever et al. (2014, Seq2Seq), Mikolov et al. (2013, Word2Vec), Vaswani et al. (2017, Transformer), Bengio et al. (1994, 기울기 소실 분석)

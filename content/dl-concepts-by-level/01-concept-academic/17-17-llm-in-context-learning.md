---
title: "17. 대규모 언어 모델(LLM)과 In-Context Learning"
slug: 17-llm-in-context-learning
order: 17
---

# 17. 대규모 언어 모델(LLM)과 In-Context Learning

## 1. 동기부여 및 개요

전통적 딥러닝은 태스크마다 대규모 라벨 데이터를 수집하고 모델을 학습해야 했다.
LLM(Large Language Model)의 등장은 이 패러다임을 근본적으로 바꾸었다.
한 번 거대한 데이터로 사전학습한 모델이, **별도의 가중치 업데이트 없이** 프롬프트만으로
새로운 태스크를 수행할 수 있다. 이것이 **In-Context Learning (ICL)**이다.

```
Transformer (Self-Attention)
       |
  Language Model --- 규모 확장 ---> LLM
       |                              |
  Supervised Learning              Scaling Laws
       |                              |
  Transfer Learning               In-Context Learning
       |                              |
  Pre-train & Fine-tune           Few-shot / Zero-shot
       |
  Adapter / LoRA (PEFT)
```

---

## 2. GPT 계열과 LLM의 진화

### 2.1 발전 과정

| 모델 | 연도 | 파라미터 수 | 핵심 혁신 |
|------|------|------------|----------|
| GPT-1 | 2018 | 117M | Transformer 기반 생성적 사전학습 |
| GPT-2 | 2019 | 1.5B | WebText 학습, task-specific 학습 제거 |
| GPT-3 | 2020 | 175B | In-Context Learning 능력 발현 |
| ChatGPT | 2022 | - | RLHF(인간 피드백 강화학습)로 정렬 |

GPT는 Transformer의 **Decoder만** 사용하는 autoregressive 모델이다:

$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1}; \theta)$$

BERT와의 아키텍처 비교:
- **BERT (양방향):** $p(x_i \mid x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)$ -- Encoder
- **GPT (단방향):** $p(x_i \mid x_1, \ldots, x_{i-1})$ -- Decoder (causal mask)

### 2.2 스케일링 법칙 (Scaling Laws)

**Theorem 2.1 (경험적 스케일링 법칙, Kaplan et al., 2020).**
모델 성능(loss $L$)은 파라미터 수 $N$, 데이터 크기 $D$, 계산량 $C$에 대해 거듭제곱 법칙을 따른다:

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}, \quad L(C) \propto C^{-\alpha_C}$$

이 법칙은 모델을 키우면 예측 가능한 방식으로 성능이 향상됨을 의미한다.
Hoffmann et al. (2022)의 Chinchilla 연구는 파라미터 수와 데이터 양의 **최적 비율**이 존재함을 보여, compute-optimal 학습 전략의 기반이 되었다.

---

## 3. Transfer Learning (전이 학습)

### 3.1 핵심 개념

전이 학습은 소스 도메인 $\mathcal{X}_s$에서 학습한 표현을 타깃 도메인 $\mathcal{X}_t$에 활용하는 기법이다.

**Definition 3.1 (Transfer Learning).** 사전학습 파라미터 $\theta_{\text{pre}}$를 초기값으로 사용하여 타깃 태스크를 학습한다:

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{target}}(\theta), \quad \theta_0 = \theta_{\text{pre}}$$

### 3.2 두 가지 전략

| 전략 | 설명 | 학습 대상 |
|------|------|----------|
| Feature Extractor | 사전학습 가중치 동결(frozen) | 새 head만 |
| Fine-tuning | 전체를 작은 학습률로 업데이트 | 전체 또는 일부 |

### 3.3 이론적 기반

Ben-David et al. (2010)의 domain adaptation 이론에 따르면:

$$\epsilon_t(h) \leq \epsilon_s(h) + d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_s, \mathcal{D}_t) + \lambda$$

여기서 $\epsilon_t$는 타깃 에러, $\epsilon_s$는 소스 에러, $d_{\mathcal{H}\Delta\mathcal{H}}$는 도메인 간 발산이다. 소스와 타깃의 구조적 유사성이 높을수록 전이가 효과적이다.

---

## 4. 효율적 미세조정 (PEFT)

### 4.1 Adapter

Houlsby et al. (2019)의 Adapter는 Transformer 각 층에 bottleneck MLP를 삽입한다:

$$\text{Adapter}(x) = x + W_{\text{up}} \cdot \sigma(W_{\text{down}} \cdot x)$$

여기서 $W_{\text{down}} \in \mathbb{R}^{d \times h}$, $W_{\text{up}} \in \mathbb{R}^{h \times d}$, $d \ll h$이다.
추가 파라미터: $2 \times d \times h$ (예: $d=64$, $h=768$이면 약 100K).
원본 모델은 동결하고 Adapter, LayerNorm, 출력 head만 학습한다.

### 4.2 LoRA (Low-Rank Adaptation)

Hu et al. (2021)의 LoRA는 가중치 업데이트를 저랭크 분해로 표현한다:

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{h \times r}, \; A \in \mathbb{R}^{r \times h}$$

$r \ll h$이므로 학습 파라미터가 $O(rh)$로, full fine-tuning의 $O(h^2)$ 대비 극적으로 감소한다.

```python
# LoRA 개념 의사코드
class LoRALinear:
    def __init__(self, in_dim, out_dim, rank=4):
        self.W = frozen_pretrained_weight  # 동결
        self.A = randn(rank, in_dim) * 0.01
        self.B = zeros(out_dim, rank)      # 초기에 Delta W = 0

    def forward(self, x):
        return x @ self.W.T + x @ self.A.T @ self.B.T  # W + BA
```

| 방법 | 추가 파라미터 | 특징 |
|------|-------------|------|
| Full Fine-tuning | $|\Theta|$ 전체 | 최고 성능, 최대 비용 |
| Adapter | $\sim 2dh$ per layer | 각 층에 bottleneck 삽입 |
| LoRA | $\sim 2rh$ per layer | 저랭크 분해, 추론 시 합산 가능 |
| Prompt Tuning | $\sim kd$ | 입력 앞에 학습 가능 토큰 추가 |

---

## 5. Pre-training & Fine-tuning 패러다임

### 5.1 사전학습 목적함수

**Causal LM (GPT 계열):**

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log p(x_t \mid x_{<t}; \theta)$$

**Masked LM (BERT 계열):**

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log p(x_i \mid x_{\backslash \mathcal{M}}; \theta)$$

여기서 $\mathcal{M}$은 랜덤 마스크 집합이다.

### 5.2 왜 Pre-training이 효과적인가

세 가지 관점에서 설명할 수 있다:

1. **최적화 관점:** 사전학습이 loss landscape에서 좋은 basin을 찾아주고, fine-tuning은 그 basin 내에서 task-specific minimum으로 이동
2. **정보 이론 관점:** $p(x)$의 풍부한 표현을 학습하면, $p(y|x)$ 학습에 정보론적 이점 제공
3. **PAC-Bayes 관점:** $\theta_{\text{pre}}$가 informative prior 역할:

$$\mathcal{L}_{\text{gen}} \leq \hat{\mathcal{L}} + \sqrt{\frac{\text{KL}(q(\theta) \| p(\theta)) + \log(n/\delta)}{2n}}$$

$p(\theta) = \mathcal{N}(\theta_{\text{pre}}, \sigma^2 I)$로 설정하면, fine-tuned 파라미터와의 KL이 작아져 일반화 보장이 강해진다.

---

## 6. In-Context Learning (ICL)

### 6.1 정의

**Definition 6.1 (In-Context Learning).** 데모 집합 $\mathcal{D} = \{(x_1, y_1), \ldots, (x_k, y_k)\}$와 질의 $x_q$가 주어졌을 때:

$$\hat{y} = \arg\max_y \; p(y \mid x_1, y_1, \ldots, x_k, y_k, x_q; \theta)$$

여기서 $\theta$는 **고정(frozen)**이다. Gradient를 계산하지 않으므로 전통적 의미의 "학습"이 아니다.

### 6.2 놀라운 실험 결과

Min et al. (2022)의 실험은 ICL의 본질에 대한 통찰을 제공한다:

| 조건 | 성능 |
|------|------|
| Gold labels (정답 라벨) | 높음 |
| Random labels (무작위 라벨) | **거의 동등** |
| No demos (예시 없음) | 낮음 |

**해석:** ICL에서 라벨 정확성보다 입력의 **분포(distribution)**와 **형식(format)**이 더 중요하다. 모델은 데모에서 태스크의 형태를 파악하는 것이지, 입출력 매핑을 외우는 것이 아니다.

### 6.3 이론적 해석

ICL의 메커니즘에 대해 세 가지 가설이 제안되었다:

**(1) 암묵적 경사 하강법 가설** (Akyurek et al., 2022; von Oswald et al., 2023):
단일 선형 attention 레이어의 연산이 최소 자승법의 한 스텝과 동치임을 보일 수 있다.

**(2) 베이지안 추론 가설** (Xie et al., 2021):
ICL은 잠재 개념 $z$에 대한 베이지안 추론으로 해석된다:

$$p(y \mid x_q, \mathcal{D}) = \int p(y \mid x_q, z) \, p(z \mid \mathcal{D}) \, dz$$

**(3) Task vector 가설:**
데모가 모델 내부에서 "task vector"를 형성하여, 해당 태스크에 맞는 연산을 활성화한다.

---

## 7. Few-shot / Zero-shot Prompting

### 7.1 분류

| 유형 | 설명 | 예시 수 |
|------|------|---------|
| Zero-shot | 예시 없이 질문 | 0 |
| One-shot | 예시 1개 | 1 |
| Few-shot | 예시 2~5개 | 2~5 |

### 7.2 성능에 영향을 미치는 요인

Few-shot prompting의 성능은 다음 요인에 의존한다:
1. **데모 선택 (demonstration selection):** 어떤 예시를 고르느냐
2. **데모 순서 (demonstration ordering):** 예시의 배치 순서
3. **프롬프트 형식 (prompt template):** 어떤 포맷으로 제시하느냐

### 7.3 Emergent Abilities

Wei et al. (2022)는 모델 크기가 임계점을 넘으면 성능이 급격히 향상되는 현상을 관찰하였다:

$$\text{Acc}(N) \approx \begin{cases} \text{chance level} & N < N_c \\ \text{above chance} & N \geq N_c \end{cases}$$

그러나 Schaeffer et al. (2024)는 이러한 emergence가 비선형 메트릭의 아티팩트일 수 있다고 반론하였다. 선형 메트릭(Brier score 등)을 사용하면 연속적 성능 향상이 관찰된다.

---

## 8. Context Window와 Long-context LLM

### 8.1 모델별 Context Window

| 모델 | Context Window |
|------|---------------|
| Gemini 1.0 Pro | 32K 토큰 |
| GPT-4 Turbo | 128K 토큰 |
| Claude 2.1 | 200K 토큰 |
| Gemini 1.5 Pro | 1M 토큰 |

### 8.2 기술적 과제

Transformer의 self-attention은 $O(n^2)$ 복잡도를 가지므로, 긴 context 처리에는 큰 비용이 든다.

**"Lost in the Middle" 현상** (Liu et al., 2024): 컨텍스트가 길어질수록 중간에 위치한 정보의 검색 정확도가 저하된다. Attention의 엔트로피가 시퀀스 길이에 따라 증가하여, 특정 토큰에 대한 집중도가 분산되기 때문이다:

$$\text{Attn}(q, K) = \text{softmax}\!\left(\frac{qK^\top}{\sqrt{d}}\right) \to \text{uniform} \quad \text{as } n \to \infty$$

**해결 기법:** Sparse Attention, Ring Attention, RoPE 확장, ALiBi, YaRN 등

---

## 9. 흔한 오해와 주의점

| # | 오해 | 올바른 이해 |
|---|------|-------------|
| 1 | LLM은 단순히 파라미터가 많은 모델이다 | 데이터 양과 질, 학습 방법, 스케일링 법칙의 조합이 핵심. Chinchilla는 데이터-파라미터 비율의 최적화가 중요함을 보임 |
| 2 | ICL은 모델이 예시에서 새로 학습하는 것이다 | 가중치가 전혀 업데이트되지 않는다. 사전학습된 능력을 프롬프트로 활성화하는 것 |
| 3 | Few-shot에서 라벨이 정확해야 성능이 좋다 | Gold label $\approx$ random label (Min et al., 2022). 분포와 형식이 더 중요 |
| 4 | Fine-tuning은 전체 모델 업데이트가 필수 | PEFT(Adapter, LoRA)로 1% 미만의 파라미터만 업데이트해도 유사한 성능 달성 |
| 5 | Transfer learning에서 소스와 타깃이 같아야 한다 | 고수준 구조적 유사성만 있으면 됨. ImageNet 특징이 의료 영상에도 유용 |
| 6 | Context window가 크면 무조건 좋다 | $O(n^2)$ 비용 급증 + Lost in the Middle 현상. RAG가 더 실용적일 수 있음 |

---

## 10. 핵심 요약

| 개념 | 핵심 |
|------|------|
| LLM 진화 | GPT-1(117M) $\to$ GPT-3(175B) $\to$ ChatGPT(+RLHF). 스케일링 법칙이 발전을 주도 |
| Scaling Law | $L(N) \propto N^{-\alpha}$. 모델/데이터/계산의 상호작용 |
| Transfer Learning | 소스 도메인의 표현을 타깃에 전이. Feature extraction vs Fine-tuning |
| PEFT | Adapter, LoRA 등으로 소수 파라미터만 학습. $\Delta W = BA$ ($r \ll h$) |
| Pre-train + Fine-tune | 비라벨 대규모 데이터로 범용 표현 $\to$ 소규모 라벨로 태스크 적응 |
| ICL | 가중치 업데이트 없이 프롬프트 내 예시로 태스크 수행. $\hat{y} = \arg\max_y p(y \mid \mathcal{D}, x_q; \theta_{\text{frozen}})$ |
| Few-shot | 0-shot $<$ 1-shot $<$ few-shot. 모델이 클수록 효과 극대화 |
| Context Window | 32K $\to$ 1M으로 확장. $O(n^2)$ 비용과 Lost in the Middle이 과제 |

**관련 개념 연결:** Transformer 아키텍처 $\to$ [16장], Self-supervised pre-training $\to$ [18장], Autoregressive 모델 $\to$ [19장] 생성모델

**참고 문헌:**
- Brown et al., "Language Models are Few-Shot Learners," NeurIPS 2020 (GPT-3)
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv 2020
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022
- Min et al., "Rethinking the Role of Demonstrations," EMNLP 2022

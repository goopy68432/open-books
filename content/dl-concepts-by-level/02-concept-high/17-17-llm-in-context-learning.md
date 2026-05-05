---
title: "17. LLM & In-Context Learning"
slug: 17-llm-in-context-learning
order: 17
---

# 17. LLM & In-Context Learning

## 왜 배우는가?

전통적인 딥러닝은 새로운 문제를 풀 때마다 대량의 정답 데이터를 모으고 모델을 처음부터 학습시켜야 했다. 하지만 LLM(Large Language Model, 대규모 언어 모델)은 이 방식을 완전히 뒤집었다. 한 번 거대한 데이터로 학습한 모델이 **별도의 추가 학습 없이 프롬프트(질문)만으로** 새로운 문제를 풀 수 있게 된 것이다. 이것이 **In-Context Learning(ICL)**이며, AI 연구에서 가장 큰 패러다임 전환이다.

---

## 1. GPT 계열과 LLM의 진화

### GPT란?

GPT = **G**enerative **P**re-trained **T**ransformer

- Transformer의 Decoder(단방향)만 사용하는 언어 모델
- "다음 단어를 예측하는" 방식으로 학습: $p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})$

### 진화 과정

| 모델 | 연도 | 파라미터 수 | 핵심 변화 |
|------|------|------------|----------|
| GPT-1 | 2018 | 1.17억 | Transformer 기반 사전학습의 시작 |
| GPT-2 | 2019 | 15억 | "Fine-tuning 없이도 된다"는 주장 |
| GPT-3 | 2020 | 1750억 | **In-Context Learning** 능력 발견 |
| ChatGPT | 2022 | - | RLHF로 인간의 의도에 맞게 정렬 |

### 스케일링 법칙 (Scaling Laws)

모델 성능은 파라미터 수 $N$에 대해 거듭제곱 법칙을 따른다:

$$L(N) \propto N^{-\alpha}$$

즉, 모델을 키우면 성능이 점점 좋아지는데, 그 관계가 매우 규칙적이다. 이 법칙이 "모델을 더 키우면 더 똑똑해진다"는 연구 방향을 이끌었다.

---

## 2. Transfer Learning (전이 학습)

### 핵심 아이디어

피아노를 잘 치는 사람이 기타를 빨리 배울 수 있는 것처럼, 한 분야에서 배운 지식을 다른 분야에 활용하는 것이다.

### 두 가지 전략

```
[큰 데이터셋으로 사전학습된 모델]
          │
    ┌─────┴──────┐
    │            │
Feature Extractor  Fine-tuning
(사전학습 가중치 동결,    (사전학습 가중치를
 새 출력층만 학습)       작은 학습률로 함께 업데이트)
```

수학적으로는 사전학습 파라미터 $\theta_{\text{pre}}$를 초기값으로 사용한다:

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{target}}(\theta), \quad \theta_0 = \theta_{\text{pre}}$$

사전학습에는 ImageNet 같은 지도 학습도 있고, 라벨 없이 학습하는 자기지도 학습도 있다. 라벨 없는 데이터가 훨씬 많으므로 자기지도 학습이 더 확장성이 높다.

---

## 3. Pre-training & Fine-tuning 패러다임

### 2단계 학습

| 단계 | 방법 | 데이터 | 목표 |
|------|------|--------|------|
| **Pre-training** (사전학습) | "다음 단어 예측" 게임 | 인터넷의 수십억 문장 (라벨 불필요) | 문법, 상식, 추론 능력 습득 |
| **Fine-tuning** (미세조정) | 특정 과제에 맞춘 추가 학습 | 소량의 라벨 데이터 | 번역, 감정 분석 등 특정 능력 |

### 사전학습 목적함수

- **Causal LM (GPT)**: $\mathcal{L} = -\sum_{t=1}^{T} \log p(x_t \mid x_{<t}; \theta)$ --- 다음 단어 예측
- **Masked LM (BERT)**: $\mathcal{L} = -\sum_{i \in \mathcal{M}} \log p(x_i \mid x_{\backslash \mathcal{M}}; \theta)$ --- 가려진 단어 예측

### Fine-tuning 방법

사전학습 파라미터에서 출발하여 과제별 손실함수로 업데이트:

$$\theta_{\text{ft}} = \theta_{\text{pre}} - \eta \nabla_\theta \mathcal{L}_{\text{task}}(\theta_{\text{pre}})$$

보통 하위 레이어(일반 지식)는 그대로 두거나 아주 작은 학습률로 업데이트하고, 출력 레이어(과제 특화)만 새로 학습한다.

---

## 4. Adapter와 효율적 미세조정 (PEFT)

### 문제: LLM 전체를 Fine-tuning하기엔 너무 크다

GPT-3는 파라미터가 1750억 개이다. 전체를 업데이트하려면 엄청난 GPU 메모리와 시간이 필요하다.

### 해결: 작은 모듈만 추가해서 학습

**Adapter** 방식은 기존 모델 가중치는 동결(frozen)하고, 각 층에 작은 **병목(bottleneck) 구조**만 끼워넣어 학습한다.

```
입력 (차원 h=768)
   │
   ↓ W_down (768 → 64)  차원 축소
   │
   ↓ 활성 함수
   │
   ↓ W_up (64 → 768)    차원 복원
   │
   + 잔차 연결
   │
출력 (차원 h=768)
```

추가 파라미터: $2 \times 64 \times 768 \approx 10$만 개. 전체 모델 대비 1% 미만이다.

### LoRA (Low-Rank Adaptation)

가중치 업데이트를 저랭크 행렬의 곱으로 표현한다:

$$W' = W + \Delta W = W + BA, \quad B \in \mathbb{R}^{h \times r},\; A \in \mathbb{R}^{r \times h}$$

$r$이 $h$보다 훨씬 작으므로 ($r \ll h$), 학습 파라미터가 크게 줄어든다. 이런 방법들을 통칭 **PEFT(Parameter-Efficient Fine-Tuning)**라 한다.

---

## 5. In-Context Learning (ICL)

### 무엇인가?

모델에게 프롬프트(입력) 안에 예시를 넣어주는 것만으로 새로운 과제를 수행하는 능력이다.

```
예시 1: "Contains no wit..." → Negative
예시 2: "Very good viewing..." → Positive
질문:   "A smile on your face" → ???
```

모델은 예시의 패턴을 파악하여 "Positive"라고 답한다.

### 놀라운 점: 가중치 업데이트가 전혀 없다!

일반적인 학습과 달리, ICL에서는 gradient를 계산하지 않고 가중치도 바꾸지 않는다. 수학적으로:

$$\hat{y} = \arg\max_y \; p(y \mid x_1, y_1, \ldots, x_k, y_k, x_q;\; \theta_{\text{frozen}})$$

$\theta$가 고정(frozen)인 상태에서 추론만으로 답을 낸다.

### 더 놀라운 사실: 라벨이 틀려도 비슷하게 작동한다

실험 결과, 정답 라벨(gold label)을 주든 무작위 라벨(random label)을 주든 성능 차이가 크지 않았다! 이는 ICL이 "라벨을 외우는 것"이 아니라 **입력의 분포와 형식(format)을 파악**하는 것에 가깝다는 증거이다.

---

## 6. Few-shot / Zero-shot Prompting

### 예시 개수에 따른 분류

| 방식 | 예시 수 | 설명 |
|------|---------|------|
| Zero-shot | 0개 | 예시 없이 바로 질문 |
| One-shot | 1개 | 예시 1개 제공 후 질문 |
| Few-shot | 2~5개 | 예시 여러 개 제공 후 질문 |

일반적으로 **예시가 많을수록**, 그리고 **모델이 클수록** 성능이 좋다.

### 성능에 영향을 미치는 요인

1. **데모 선택**: 어떤 예시를 고르느냐
2. **데모 순서**: 예시를 어떤 순서로 배치하느냐
3. **프롬프트 형식**: 어떤 포맷으로 제시하느냐

특히 모델 크기가 작으면 few-shot의 효과가 미미하지만, 크기가 커지면 few-shot에서 급격한 성능 향상(emergence)이 나타난다.

---

## 7. Context Window (맥락 창)

### AI가 한 번에 읽을 수 있는 텍스트 양

| 모델 | Context Window |
|------|---------------|
| GPT-4 Turbo | 128K 토큰 |
| Claude 2.1 | 200K 토큰 |
| Gemini 1.5 Pro | **1M 토큰** |

1M 토큰이면 약 70만 단어, 1시간 분량의 비디오를 한 번에 처리할 수 있다.

### 커질수록 좋지만 한계도 있다

- **계산 비용**: Transformer의 Self-Attention은 $O(n^2)$ 복잡도이므로, 컨텍스트가 길어지면 비용이 급증한다.
- **"Lost in the Middle" 현상**: 긴 텍스트의 중간에 있는 정보는 잘 찾지 못하는 문제가 있다. Attention이 분산되어 특정 토큰에 집중하기 어려워지기 때문이다.

---

## 오해하기 쉬운 포인트

### 1. "LLM은 파라미터가 많으면 된다"

파라미터 수만으로는 부족하다. **데이터의 양과 질**, **학습 방법**, **파라미터 대비 데이터 비율의 최적화**가 모두 중요하다. Chinchilla 연구는 같은 계산량이면 파라미터를 줄이고 데이터를 늘리는 것이 더 낫다고 밝혔다.

### 2. "ICL은 모델이 예시에서 새로 학습하는 것이다"

ICL에서는 가중치가 **전혀 업데이트되지 않는다**. 사전학습 때 이미 배운 능력을 프롬프트로 "활성화"하는 것이지, 새로 배우는 것이 아니다.

### 3. "Few-shot에서 라벨이 정확해야 성능이 좋다"

실험적으로 gold label과 random label의 성능 차이가 미미했다. ICL은 라벨 정확성보다 **형식과 분포**에서 과제 정보를 추출한다.

### 4. "Fine-tuning은 전체 모델을 업데이트해야 한다"

Adapter, LoRA 등으로 전체 파라미터의 1% 미만만 업데이트해도 충분한 성능을 낼 수 있다.

### 5. "Context window가 크면 무조건 좋다"

$O(n^2)$ 비용 증가와 "Lost in the Middle" 현상 때문에, 효과적인 정보 검색(RAG) 방법이 무한 컨텍스트보다 실용적일 수 있다.

---

## 정리/요약

| 개념 | 핵심 한 줄 |
|------|-----------|
| LLM의 진화 | GPT-1(1억) → GPT-3(1750억) → ChatGPT(+RLHF). 스케일링 법칙: $L \propto N^{-\alpha}$ |
| Transfer Learning | 큰 데이터로 사전학습 → 작은 데이터로 미세조정. 기초 실력을 새 과제에 활용 |
| Pre-train + Fine-tune | 사전학습으로 범용 표현 학습, fine-tuning으로 과제 적응 |
| PEFT | Adapter/LoRA로 1% 미만의 파라미터만 학습하여 효율적 적응. $W' = W + BA$ |
| In-Context Learning | 가중치 업데이트 없이 프롬프트 안의 예시만으로 과제 수행 |
| Few-shot Prompting | 예시 개수에 따라 0/1/few-shot. 모델이 클수록 효과가 크다 |
| Context Window | AI가 한 번에 처리하는 텍스트 양. 길수록 좋지만 $O(n^2)$ 비용과 "Lost in the Middle" 주의 |

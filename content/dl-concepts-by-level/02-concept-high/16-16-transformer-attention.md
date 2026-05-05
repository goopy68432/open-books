---
title: "16. Transformer & Attention"
slug: 16-transformer-attention
order: 16
---

# 16. Transformer & Attention

## 왜 배우는가?

RNN(순환신경망)은 단어를 하나씩 차례로 처리하기 때문에 두 가지 큰 문제가 있었다.
1. **병렬 처리 불가**: 앞 단어를 처리해야 뒷 단어를 처리할 수 있어서 느리다.
2. **장거리 의존성 소실**: 문장이 길어지면 앞부분 정보가 뒷부분까지 전달되지 못한다.

2017년 "Attention Is All You Need" 논문에서 Transformer가 등장하면서 이 문제가 해결되었다. GPT, BERT, ChatGPT, 이미지 생성 AI까지 현대 AI의 거의 모든 기반이 Transformer 위에 세워져 있다.

---

## 1. Attention: 부드러운 사전 검색

### 핵심 아이디어

도서관에서 "공룡에 대한 책"을 찾는다고 하자.

| 단계 | 비유 | Attention에서의 역할 |
|------|------|---------------------|
| 질문하기 | "공룡에 대한 책 있나요?" | **Query (Q)** |
| 책 제목 확인 | 각 책의 제목을 훑어본다 | **Key (K)** |
| 내용 읽기 | 관련 있는 책의 내용을 가져온다 | **Value (V)** |

일반 검색은 "일치하면 1, 아니면 0"인 딱딱한(hard) 검색이다. 하지만 Attention은 **부드러운(soft) 검색**으로, 관련도에 따라 여러 책의 내용을 비율에 맞게 섞어서 답을 만든다.

### 수식으로 표현하면

모든 Key에 대해 Query와의 유사도를 계산하고, 그 유사도를 확률 분포로 바꿔서 Value를 가중 합산한다.

$$\text{Attention 출력} = \sum_{j=1}^{N} \alpha_j \cdot v_j$$

여기서 $\alpha_j$는 Query와 $j$번째 Key 사이의 유사도(가중치)이고, $\sum_j \alpha_j = 1$이다. 이 가중치는 softmax 함수로 만든다.

---

## 2. Scaled Dot-Product Attention

### 유사도 측정: 내적(Dot Product)

두 벡터의 내적은 "얼마나 같은 방향을 가리키는가"를 나타낸다. 수학II에서 배운 벡터의 내적과 같은 개념이다.

$$a(q, k) = q^\top k = \sum_{i=1}^{d_k} q_i \cdot k_i$$

### 왜 $\sqrt{d_k}$로 나누는가?

벡터 차원 $d_k$가 커지면 내적 값도 커진다. 확률과 통계에서 배운 분산 개념으로 설명하면:

> 각 원소가 평균 0, 분산 1일 때, 내적 $q^\top k$의 분산은 $d_k$이다.

내적 값이 지나치게 크면 softmax가 한쪽으로 몰려서 학습이 안 된다. 그래서 $\sqrt{d_k}$로 나눠 분산을 1로 맞춘다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

### 전체 과정 다이어그램

```
Q (질문)  K (키)
   \       /
    내적 계산 → QK^T (유사도 행렬)
        |
    ÷ √d_k (스케일링)
        |
    softmax (확률 분포로 변환)
        |
    × V (가중 합산)
        |
    출력 (Attention 결과)
```

---

## 3. Multi-Head Attention (MHA)

### 왜 여러 개의 Attention이 필요한가?

하나의 Attention은 한 가지 관계만 포착한다. 하지만 문장에는 다양한 관계가 동시에 존재한다:
- "주어-동사" 관계
- "형용사-명사" 관계
- "대명사-선행사" 관계

Multi-Head Attention은 **여러 명의 분석가**가 각자 다른 관점에서 문장을 분석한 뒤 결과를 합치는 것이다.

### 수식

입력 $X$에 대해 각 Head $i$는 **별도의 가중치 행렬**을 사용한다:

$$\text{head}_i = \text{Attention}(XW_i^Q,\; XW_i^K,\; XW_i^V)$$

모든 Head의 결과를 이어붙이고(concatenate) 출력 행렬을 곱한다:

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

원래 논문에서는 $d = 512$, Head 수 $h = 8$, 각 Head의 차원 $d_k = d/h = 64$로 설정했다.

---

## 4. Positional Encoding (위치 인코딩)

### 문제: Attention은 순서를 모른다

"나는 너를 좋아해"와 "너를 나는 좋아해"는 느낌이 다르다. 그런데 Attention은 모든 단어를 동시에 보기 때문에 순서를 구분하지 못한다. (수학적으로 이를 "순서 불변(permutation equivariant)"이라 한다.)

### 해결: 번호표 붙이기

각 위치에 고유한 벡터 $P$를 만들어서 단어 임베딩에 더한다:

$$\text{PE}(X) = X + P$$

원래 Transformer에서는 삼각함수로 위치 벡터를 만든다:

$$\text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

이렇게 하면 학습할 파라미터 없이도 어떤 길이의 문장이든 위치를 표현할 수 있다.

---

## 5. Transformer Encoder

Encoder는 문장을 읽고 **각 단어의 의미를 문맥에 맞게 풍부하게** 만드는 역할이다.

### 한 층의 구조

```
입력 X
  │
  ├──→ Self-Attention (MHA(X, X, X)) ──→ (+X) ──→ LayerNorm ──→ 중간 결과
  │                                        ↑ 잔차 연결
  │
  ├──→ Feed Forward Network (2층 MLP) ──→ (+중간) ──→ LayerNorm ──→ 출력
  │                                        ↑ 잔차 연결
```

- **Self-Attention**: Q, K, V가 모두 같은 입력 X에서 나온다 (자기 자신들끼리 관계 파악)
- **Feed Forward**: 각 단어를 독립적으로 변환 ($512 \to 2048 \to 512$)
- **Add & Norm**: 잔차 연결(residual connection) + LayerNorm으로 학습 안정화

수식으로:
$$x = \text{LN}(x + \text{Self-Attention}(x))$$
$$x = \text{LN}(x + \text{FFN}(x))$$

이 층을 6번 반복한다 (원 논문 기준).

---

## 6. Transformer Decoder & Cross Attention

Decoder는 **새로운 문장을 한 단어씩 생성**하는 역할이다.

### Decoder의 세 가지 핵심 요소

| 요소 | 수식 | 역할 |
|------|------|------|
| Masked Self-Attention | MHA(Y, Y, Y) + 마스크 | 미래 단어를 못 보게 가림 |
| Cross Attention | MHA(Y, M, M) | Encoder 출력(M)을 참조 |
| Feed Forward | FFN(Y) | 각 위치 독립 변환 |

### Masked Attention: 미래를 못 보게 하기

번역할 때 "Je suis"까지 생성했다면, 아직 생성하지 않은 "etudiant"는 볼 수 없어야 한다.

미래 위치의 유사도를 $-\infty$로 설정하면, softmax 후 가중치가 0이 된다:

$$A_{ij} = \begin{cases} \text{softmax 값} & j \le i \text{ (과거와 현재)} \\ 0 & j > i \text{ (미래)} \end{cases}$$

이것은 **하삼각 행렬(lower triangular matrix)** 마스크이다.

### Cross Attention: 두 시퀀스 간의 다리

- **Q** = Decoder의 현재 상태 (Y)
- **K, V** = Encoder의 출력 (M)

Decoder가 "이 프랑스어 단어를 생성하려면 영어 원문의 어떤 부분을 봐야 할까?"라고 질문하는 것이다.

### Self-Attention vs Cross-Attention 비교

| 구분 | Self-Attention | Cross-Attention |
|------|---------------|-----------------|
| 호출 | MHA(X, X, X) | MHA(Y, M, M) |
| Q, K, V 출처 | 모두 같은 시퀀스 | Q는 Decoder, K/V는 Encoder |
| 역할 | 시퀀스 내부 관계 파악 | 두 시퀀스 간 정보 교환 |

---

## 7. Vision Transformer (ViT)

### 이미지도 Transformer로!

Transformer는 원래 텍스트용이었지만, 이미지를 작은 조각(패치)으로 나누면 "단어"처럼 취급할 수 있다.

### ViT의 과정

```
원본 이미지 (224x224)
    │
    ↓ 16x16 패치로 분할 → 196개의 패치
    │
    ↓ 각 패치를 벡터로 변환 (Linear Projection)
    │
    ↓ [CLS] 토큰 추가 + Positional Embedding
    │
    ↓ Transformer Encoder 통과
    │
    ↓ [CLS] 토큰으로 분류 (MLP Head)
```

### CNN vs ViT

| 특성 | CNN | ViT |
|------|-----|-----|
| 패턴 인식 | 지역적(local) | 전역적(global) |
| Inductive bias | 강함 (locality, translation equivariance) | 약함 |
| 적은 데이터 | 더 좋음 | 과적합 위험 |
| 대규모 데이터 | ViT에 뒤처짐 | **더 좋음** |

핵심 교훈: **약한 inductive bias + 많은 데이터 > 강한 inductive bias + 적은 데이터**

---

## 8. BERT와 GPT

| 특성 | BERT | GPT |
|------|------|-----|
| 방향 | 양방향 (bidirectional) | 단방향 (left-to-right) |
| 구조 | Transformer Encoder만 사용 | Transformer Decoder만 사용 |
| 학습 방식 | Masked Language Model (빈칸 채우기) | Causal LM (다음 단어 예측) |
| 수식 | $p(x_i \mid x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)$ | $p(x_t \mid x_1, \ldots, x_{t-1})$ |
| 용도 | 문장 이해, 분류 | 텍스트 생성, 대화 |

**BERT**는 문장에서 일부 단어를 [MASK]로 가리고 맞추는 방식으로 학습한다.

**GPT**는 이전 단어들로부터 다음 단어를 예측하는 방식으로 학습한다:

$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{1:t-1})$$

GPT-3(1750억 파라미터)부터 별도 학습 없이 예시만으로 새로운 문제를 푸는 In-Context Learning 능력이 발견되었고, ChatGPT는 여기에 RLHF(인간 피드백 강화학습)를 더해 만들어졌다.

---

## 오해하기 쉬운 포인트

### 1. "Q, K, V는 서로 다른 입력에서 온다"

Self-Attention에서는 Q, K, V **모두 같은 입력 X**에서 나온다. $Q = XW^Q$, $K = XW^K$, $V = XW^V$. 서로 다른 입력에서 오는 것은 Cross-Attention 한정이다.

### 2. "$\sqrt{d_k}$는 그냥 크기를 맞추려는 것이다"

단순 정규화가 아니라, 내적의 **분산이 $d_k$에 비례**하는 문제를 해결하는 것이다. 분산이 크면 softmax가 극단값으로 가서 gradient가 거의 0이 된다.

### 3. "Multi-Head는 같은 Attention을 반복하는 것이다"

각 Head는 **서로 다른 가중치 행렬**을 학습하므로, 서로 다른 관계 패턴을 포착한다.

### 4. "Positional Encoding은 항상 학습되는 파라미터이다"

원 Transformer는 고정된 삼각함수를 사용한다(비학습). BERT는 학습 가능한 위치 임베딩을 사용한다. 둘 다 작동한다.

### 5. "ViT는 항상 CNN보다 좋다"

작은 데이터셋에서는 CNN이 더 우수하다. ViT가 CNN을 넘어서려면 JFT-300M 수준의 대규모 데이터가 필요하다.

### 6. "Masked Attention은 학습 때만 쓴다"

추론(생성)할 때도 autoregressive하게 한 토큰씩 생성하므로 masked 구조가 유지된다.

---

## 정리/요약

| 개념 | 핵심 한 줄 |
|------|-----------|
| Attention | Query로 Key를 검색하고, 유사도에 따라 Value를 가중 합산: $\text{softmax}(QK^\top / \sqrt{d_k})V$ |
| Scaling | $\sqrt{d_k}$로 나누어 내적 분산을 1로 맞추고 gradient 안정화 |
| Multi-Head | $h$개의 독립 Attention으로 다양한 관계 패턴을 동시에 포착 |
| Positional Encoding | Attention은 순서를 모르므로 위치 정보를 더해줌: $X + P$ |
| Encoder | Self-Attention + FFN을 반복하여 문맥 표현 생성 |
| Decoder | Masked Self-Attn + Cross-Attn + FFN으로 autoregressive 생성 |
| ViT | 이미지를 패치 시퀀스로 보고 Transformer 적용. 대규모 데이터에서 CNN을 능가 |
| BERT vs GPT | 양방향(빈칸 채우기) vs 단방향(다음 단어 예측). 둘 다 Transformer 기반 |

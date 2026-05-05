---
title: "제1장: 벡터, 함수, 집합 -- 딥러닝의 수학적 기초"
slug: 01-vectors-functions-sets
order: 1
---

# 제1장: 벡터, 함수, 집합 -- 딥러닝의 수학적 기초

> **선수 과목**: 고등학교 수학 (집합, 함수, 좌표기하)
> **후속 연결**: 2장 행렬과 선형 공간, 5장 미적분과 행렬 미적분

---

## 1. 동기부여 및 개요

딥러닝은 본질적으로 **벡터를 입력받아 행렬(선형 변환)과 비선형 함수를 반복 적용하여 원하는 출력 벡터를 만드는 과정**이다. 이 과정을 엄밀하게 이해하려면 집합, 함수, 스칼라, 벡터, 선형 변환, 행렬, 텐서라는 기초 개념이 필요하다.

본 장에서는 이 기초 개념들을 수학적 정의와 함께 소개하고, 각 개념이 딥러닝의 어떤 요소에 대응하는지를 밝힌다.

```
[집합] ──정의역/공역──> [함수] ──선형 조건──> [선형 변환]
                                                  │
[스칼라] ──순서 n-튜플──> [벡터] ──행렬 표현──> [행렬] ──차원 일반화──> [텐서]
                            │
                     내적/노름/거리 ──> 유사도 측정, 손실 함수
```

---

## 2. 집합과 카테시안 곱

### 2.1 정의

**Definition 2.1 (집합).** 집합(set) $S$는 원소 판별이 잘 정의된 대상들의 모임이다. 집합은 **비순서(unordered)** 이며, 같은 원소의 중복을 허용하지 않는다.

$$A = \{1, 2, 3\}, \quad \{1, 2, 3\} = \{3, 1, 2\}$$

**Definition 2.2 (카테시안 곱).** 두 집합 $A$, $B$의 카테시안 곱(Cartesian product)은 모든 순서쌍의 집합이다:

$$A \times B := \{(a, b) : a \in A,\; b \in B\}$$

원소의 개수는 $|A \times B| = |A| \cdot |B|$이다.

### 2.2 주요 성질

| 성질 | 수식 |
|------|------|
| 비교환 | $A \times B \neq B \times A$ (순서쌍의 순서가 다름) |
| 분배 | $A \times (B \cup C) = (A \times B) \cup (A \times C)$ |

### 2.3 딥러닝에서의 역할

딥러닝의 입력 공간 $\mathbb{R}^n$은 사실 카테시안 곱 $\mathbb{R} \times \mathbb{R} \times \cdots \times \mathbb{R}$ ($n$번)이다. 학습 데이터 $\{(x_i, y_i)\}$는 입력 공간과 출력 공간의 카테시안 곱의 부분집합이다.

```python
import itertools
A = {1, 2, 3}
B = {'x', 'y'}
cartesian = set(itertools.product(A, B))
# {(1,'x'), (1,'y'), (2,'x'), (2,'y'), (3,'x'), (3,'y')}
```

> **주의**: 집합에 순서가 있다고 오해하면 안 된다. 순서가 필요한 경우 순서쌍(tuple)이나 벡터를 사용해야 한다. 이 구분이 벡터($[1,2,3] \neq [3,1,2]$)를 이해하는 데 핵심적이다.

---

## 3. 함수

### 3.1 정의와 분류

**Definition 3.1 (함수).** 함수 $f: A \to B$는 $A$의 **모든** 원소에 대해 $B$의 원소를 **유일하게** 대응시키는 규칙이다. 형식적으로, $A \times B$의 부분집합 $G_f$로서 모든 $a \in A$에 대해 $(a, b) \in G_f$인 $b$가 유일하게 존재한다.

**Definition 3.2 (단사 함수).** $f: A \to B$가 단사(injective, one-to-one)라 함은:

$$a \neq a' \implies f(a) \neq f(a')$$

대우: $f(a) = f(a') \implies a = a'$

**Definition 3.3 (지시 함수).** 명제 $p$에 대한 지시 함수(indicator function):

$$\mathbf{1}(p) = \begin{cases} 1 & \text{if } p \text{ is true} \\ 0 & \text{if } p \text{ is false} \end{cases}$$

### 3.2 딥러닝에서의 함수

딥러닝의 각 레이어는 함수 $f_\ell : \mathbb{R}^{n_{\ell-1}} \to \mathbb{R}^{n_\ell}$이며, 전체 네트워크는 함수의 합성이다:

$$f = f_L \circ f_{L-1} \circ \cdots \circ f_1$$

| 함수 개념 | 딥러닝 대응 |
|-----------|------------|
| 함수 합성 | 다층 신경망의 순전파(forward pass) |
| 단사 함수 | Normalizing Flows (역변환 가능한 생성 모델) |
| 지시 함수 | 마스킹, one-hot 인코딩, 조건부 연산 |

**Theorem 3.1 (Universal Approximation Theorem, Cybenko 1989; Hornik 1991).** 충분히 넓은 단일 은닉 레이어 신경망은 컴팩트 집합 위의 임의의 연속 함수를 원하는 정밀도로 근사할 수 있다.

> **주의**: "근사 가능"과 "학습 가능"은 다르다. 존재성 정리이지 구성적(constructive) 정리가 아니다.

```python
# 지시 함수 구현
def indicator(p: bool) -> int:
    return 1 if p else 0

# 단사 함수 확인
def is_injective(f: dict) -> bool:
    return len(f.values()) == len(set(f.values()))
```

> **흔한 오해**: "함수는 수식이다." -- 틀렸다. 함수는 매핑 규칙이며, 학습된 가중치도 하나의 함수를 정의한다.

---

## 4. 스칼라

### 4.1 정의

**Definition 4.1 (스칼라).** 스칼라(scalar)는 실수 $s \in \mathbb{R}$이다. $(\mathbb{R}, +, \times)$는 **체(field)**를 이루며, 다음을 만족한다:

- 덧셈 항등원: $0$, 곱셈 항등원: $1$
- $\forall s \neq 0,\; \exists s^{-1}$ s.t. $s \cdot s^{-1} = 1$

### 4.2 딥러닝에서의 스칼라

스칼라는 벡터 공간의 **계수(coefficient)** 역할을 한다. 딥러닝에서 중요한 스칼라 값들:

| 스칼라 | 역할 | 기호 |
|--------|------|------|
| 학습률 | 경사하강법의 스텝 크기 | $\alpha$, $\eta$ |
| 손실 값 | 모델 성능 측정 | $\mathcal{L}$ |
| Temperature | Softmax 분포 조절 | $\tau$ |

```python
import numpy as np
s = 0.01  # learning rate (스칼라)
gradient = np.array([2.0, -1.0, 0.5])
update = s * gradient  # 스칼라 * 벡터 = 벡터
```

> **주의**: Mixed-precision training에서 사용하는 float16/bfloat16은 체(field)의 공리를 정확히 만족하지 않는다(overflow, underflow, 정밀도 손실). 이는 수치적 불안정성의 원인이 된다.

---

## 5. 벡터

### 5.1 정의와 연산

**Definition 5.1 (벡터).** 벡터(vector)는 실수의 순서 있는 $n$-튜플이다:

$$v = [v_1, v_2, \ldots, v_n]^\top \in \mathbb{R}^n$$

**Definition 5.2 (내적).** 두 벡터 $v, w \in \mathbb{R}^n$의 내적(inner product):

$$\langle v, w \rangle = \sum_{i=1}^n v_i w_i$$

**내적의 공리**:
1. **대칭**: $\langle v, w \rangle = \langle w, v \rangle$
2. **선형**: $\langle av + bu, w \rangle = a\langle v, w \rangle + b\langle u, w \rangle$
3. **양정치**: $\langle v, v \rangle \geq 0$, 등호 $\iff v = \mathbf{0}$

**Definition 5.3 (노름).** 벡터 $v$의 유클리드 노름:

$$\|v\| = \sqrt{\langle v, v \rangle} = \sqrt{\sum_{i=1}^n v_i^2}$$

**Definition 5.4 (거리).** 두 벡터 사이의 유클리드 거리:

$$d(v, w) = \|v - w\|$$

### 5.2 핵심 항등식

**Proposition 5.1.** $\|v - w\|^2 = \|v\|^2 + \|w\|^2 - 2\langle v, w \rangle$

*증명.* 내적의 쌍선형성으로부터:

$$\|v - w\|^2 = \langle v-w, v-w \rangle = \langle v,v \rangle - 2\langle v,w \rangle + \langle w,w \rangle \quad \square$$

이 공식은 contrastive loss의 수학적 기반이며, 내적이 클수록(유사할수록) 거리가 줄어든다는 것을 보여준다.

### 5.3 공간의 계층 구조

$$\text{metric space} \supset \text{vector space} \supset \text{normed space} \supset \text{inner product space} \supset \text{Euclidean space } (\mathbb{R}^n)$$

### 5.4 딥러닝에서의 벡터

```python
import numpy as np
v = np.array([1, 2, 3], dtype=np.float64)
w = np.array([4, 5, 6], dtype=np.float64)

inner = np.dot(v, w)                    # 내적: 32.0
norm_v = np.linalg.norm(v)              # 노름: 3.7416...
dist = np.linalg.norm(v - w)            # 거리: 5.1961...
cosine_sim = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))  # 코사인 유사도
```

| 벡터 연산 | 딥러닝 적용 |
|-----------|------------|
| 내적 $\langle q, k \rangle$ | Transformer attention score (Vaswani et al., 2017) |
| 코사인 유사도 | 임베딩 비교 (Word2Vec, Sentence-BERT) |
| $\|v - w\|^2$ | L2 loss, contrastive loss |
| 노름 $\|w\|$ | Weight decay, gradient clipping (4장에서 상세히) |

> **흔한 오해**: "벡터는 화살표다." -- $\mathbb{R}^2$, $\mathbb{R}^3$에서의 시각화일 뿐이다. 768차원 단어 임베딩도 벡터이며, 내적과 거리가 동일하게 정의된다.

---

## 6. 선형 변환

### 6.1 정의

**Definition 6.1 (선형 변환).** $L : \mathbb{R}^n \to \mathbb{R}^m$이 선형(linear)이라 함은 다음 두 조건을 만족하는 것이다:

1. **가법성(additivity)**: $L(v + u) = L(v) + L(u)$
2. **동차성(homogeneity)**: $L(av) = aL(v)$

이를 합쳐서 **중첩 원리(superposition principle)**로 표현할 수 있다:

$$L(\alpha v + \beta u) = \alpha L(v) + \beta L(u)$$

### 6.2 선형 vs 아핀(Affine)

**Proposition 6.1.** 선형 변환 $L$은 반드시 $L(\mathbf{0}) = \mathbf{0}$을 만족한다.

*증명.* $L(\mathbf{0}) = L(0 \cdot \mathbf{0}) = 0 \cdot L(\mathbf{0}) = \mathbf{0}$ $\square$

따라서 $y = Wx + b$ ($b \neq 0$)는 선형이 아니라 **아핀(affine)** 변환이다. 딥러닝에서 `nn.Linear`라고 부르지만, bias가 있으면 수학적으로는 아핀이다.

| | 선형 변환 | 아핀 변환 |
|---|---|---|
| 정의 | $L(v) = Wv$ | $T(v) = Wv + b$ |
| 원점 보존 | $L(\mathbf{0}) = \mathbf{0}$ | $T(\mathbf{0}) = b \neq \mathbf{0}$ |
| 딥러닝 예 | bias 없는 레이어 | `nn.Linear`(bias=True) |

### 6.3 딥러닝과의 연결

- 선형 변환만으로는 XOR 문제도 풀 수 없다 → 비선형 활성화 함수가 필수적
- Transformer의 Q, K, V projection은 선형 변환이며, 비선형성은 softmax와 FFN에서 발생한다

```python
import numpy as np
# 90도 회전 (선형 변환)
theta = np.pi / 2
L = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

v, u = np.array([1, 0]), np.array([0, 1])
assert np.allclose(L @ (v + u), L @ v + L @ u)   # 가법성
assert np.allclose(L @ (3 * v), 3 * (L @ v))     # 동차성
```

---

## 7. 행렬

### 7.1 선형 변환의 행렬 표현

**Theorem 7.1.** 선형 변환과 행렬은 일대일 대응한다.

선형 변환 $L: \mathbb{R}^n \to \mathbb{R}^m$에 대해, 그 행렬 표현은:

$$A_L := [L(e_1) \quad L(e_2) \quad \cdots \quad L(e_n)] \in \mathbb{R}^{m \times n}$$

여기서 $e_i$는 $\mathbb{R}^n$의 표준 기저 벡터이다.

*증명 스케치.* 임의의 $v = \sum_{i=1}^n v_i e_i$에 대해 선형성으로부터:

$$L(v) = \sum_{i=1}^n v_i L(e_i) = A_L v$$

따라서 $L$은 기저 벡터의 상 $\{L(e_1), \ldots, L(e_n)\}$에 의해 완전히 결정된다. $\square$

### 7.2 핵심 해석

행렬의 **$j$번째 열**은 $j$번째 기저 벡터 $e_j$를 변환한 결과이다:

$$Ae_j = A \text{의 } j\text{번째 열}$$

행렬의 **$i$번째 행**은 $e_i^\top A$로 추출된다.

### 7.3 딥러닝에서의 행렬

```python
import numpy as np
A = np.array([[1, 1, 1],
              [1, 2, 3]])  # 2x3 행렬: R^3 -> R^2

# 기저 벡터에 대한 변환 = 행렬의 열
e1, e2, e3 = np.eye(3)
print(A @ e1)  # [1, 1] = 첫 번째 열
print(A @ e2)  # [1, 2] = 두 번째 열
```

| 행렬 개념 | 딥러닝 대응 |
|-----------|------------|
| $W \in \mathbb{R}^{m \times n}$ | `nn.Linear(n, m)`의 가중치 |
| 행렬-벡터 곱 $Wx$ | 순전파(forward pass)의 핵심 연산 |
| 행렬 분해 (SVD 등) | LoRA (Hu et al., 2021)에서의 저랭크 근사 |
| Attention 행렬 | $\text{softmax}(QK^\top/\sqrt{d_k})$ — 3장에서 상세히 |

> **흔한 오해**: "행렬은 숫자를 사각형으로 배열한 것이다." -- 행렬의 본질은 **선형 변환의 표현**이다. 각 열은 기저 벡터가 어디로 가는지를 나타내며, 이 관점이 있어야 SVD, eigendecomposition, LoRA 등의 의미를 이해할 수 있다.

---

## 8. 텐서

### 8.1 정의와 차수

**Definition 8.1 (텐서).** $k$차 텐서는 $\mathbb{R}^{m_1 \times m_2 \times \cdots \times m_k}$의 원소이다. 행렬의 차원 일반화이다.

| 차수 | 이름 | 형태 | 딥러닝 예시 |
|------|------|------|------------|
| 0 | 스칼라 | $\mathbb{R}$ | 손실 값 |
| 1 | 벡터 | $\mathbb{R}^n$ | 단어 임베딩 |
| 2 | 행렬 | $\mathbb{R}^{m \times n}$ | 가중치 $W$ |
| 3 | 3차 텐서 | $\mathbb{R}^{B \times C \times L}$ | 시퀀스 배치 |
| 4 | 4차 텐서 | $\mathbb{R}^{B \times C \times H \times W}$ | 이미지 배치 |

### 8.2 딥러닝에서의 텐서

PyTorch/TensorFlow의 이름 자체가 "Tensor"인 이유 -- 모든 데이터와 파라미터가 텐서로 표현된다.

```python
import torch
scalar = torch.tensor(3.14)              # 0차: shape ()
vector = torch.tensor([1, 2, 3])         # 1차: shape (3,)
matrix = torch.randn(2, 3)              # 2차: shape (2, 3)
images = torch.randn(4, 3, 224, 224)    # 4차: (배치, 채널, 높이, 너비)
```

텐서 분해(Tucker decomposition, CP decomposition)는 모델 압축과 지식 증류에 활용된다. `torch.einsum`은 텐서 연산의 일반적 표기법으로, attention 구현에서 자주 사용된다.

> **주의**: 수학적 텐서는 좌표 변환에 대한 특정 변환 규칙을 따르는 대상이다. 딥러닝에서는 "다차원 배열"의 의미로 사용하는 것이 관례이며, 물리학/미분기하학의 텐서와 구별해야 한다.

---

## 9. 그리스 문자 표기 관례

딥러닝 논문에서 빈번히 사용되는 그리스 문자:

| 기호 | 이름 | 대표 용법 |
|------|------|----------|
| $\alpha$ | 알파 | learning rate, 혼합 계수 |
| $\beta$ | 베타 | momentum 계수 (Adam: $\beta_1, \beta_2$) |
| $\gamma$ | 감마 | discount factor (RL), BatchNorm scale |
| $\epsilon$ | 엡실론 | 수치 안정성 상수, $\epsilon$-greedy |
| $\theta$ | 세타 | 모델 파라미터 전체 |
| $\lambda$ | 람다 | 정규화 계수, 고유값 |
| $\mu, \sigma$ | 뮤, 시그마 | 평균, 표준편차 |
| $\nabla$ | 나블라 | 기울기(gradient) 연산자 (5장) |
| $\Sigma$ | 대문자 시그마 | 합 연산 $\sum$, 공분산 행렬 |

---

## 10. 딥러닝 적용 요약

| 수학 개념 | 딥러닝 대응 | 왜 필요한가 |
|-----------|------------|------------|
| 집합 | 데이터셋, vocabulary, label space | 입출력 공간의 정의 |
| 카테시안 곱 | 학습 데이터 $(x, y)$ 쌍 | 지도학습의 구조 |
| 함수 | $f_\theta(x)$, 활성화 함수, 손실 함수 | 모든 연산의 기본 단위 |
| 스칼라 | learning rate, loss, temperature | 하이퍼파라미터 및 스칼라 출력 |
| 벡터 | 임베딩, hidden state, gradient | 데이터와 파라미터의 기본 표현 |
| 내적 | attention score, 코사인 유사도 | 유사도 측정 |
| 선형 변환 | `nn.Linear`, Q/K/V projection | 차원 변환, 특성 추출 |
| 행렬 | 가중치 $W$, attention 행렬 | 선형 변환의 계산 가능한 표현 |
| 텐서 | `torch.Tensor`, 이미지 배치 | 다차원 데이터의 통합 표현 |

---

## 11. 흔한 오해와 주의점

1. **"집합에 순서가 있다"** → 집합은 비순서. 순서가 필요하면 벡터(튜플)를 쓴다.
2. **"함수는 수식이다"** → 함수는 매핑 규칙. 신경망의 학습된 가중치도 함수를 정의한다.
3. **"$y = 2x + 3$은 선형 함수다"** → 수학적으로 아핀(affine). 선형 변환은 $L(\mathbf{0}) = \mathbf{0}$을 만족해야 한다.
4. **"벡터는 화살표다"** → 저차원 시각화일 뿐. 벡터는 덧셈과 스칼라곱이 정의된 추상적 대상이다.
5. **"행렬은 숫자 표다"** → 행렬의 본질은 선형 변환의 표현. 각 열이 기저 벡터의 상(image)이다.
6. **"텐서 = 다차원 배열"** → 딥러닝 관례로는 맞지만, 수학적 텐서는 좌표 변환 규칙을 따르는 대상이다.

---

## 12. 핵심 요약

1. **집합**은 데이터의 정의역/공역을 규정한다.
2. **함수**는 입출력 매핑의 추상화이며, 딥러닝 레이어의 본질이다.
3. **벡터**는 데이터 표현의 기본 단위이며, 내적으로 유사도를 측정한다.
4. **선형 변환**은 가법성 + 동차성을 만족하는 변환이며, 행렬로 표현된다.
5. **행렬**은 선형 변환을 계산 가능하게 만드는 표현으로, 열은 기저 벡터의 상이다.
6. **텐서**는 행렬을 일반화한 다차원 자료 구조로, 딥러닝의 모든 데이터를 담는다.

> **한 문장 정리**: 딥러닝은 입력 벡터를 출력 벡터로 바꾸는 최적의 행렬(가중치)을 찾는 과정이며, 본 장의 모든 개념은 이 과정의 수학적 기초를 형성한다.

---

## 참고문헌

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 2.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *MCSS*, 2(4).
- Vaswani, A. et al. (2017). Attention is All You Need. *NeurIPS*.
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

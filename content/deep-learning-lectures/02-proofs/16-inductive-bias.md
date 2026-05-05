---
title: "15. Inductive Bias 강도 비교 — Linear → NN → CNN → Transformer"
slug: inductive-bias
order: 16
---

# 15. Inductive Bias 강도 비교 — Linear → NN → CNN → Transformer

> **출제 근거**: 7주차 ★10 \"Inductive Bias 강도 비교\", 퀴즈 22-23 직접
> **시험 출제 방식**: \"Compare the inductive biases of linear models, MLPs, CNNs, and Transformers. Which has the strongest? Why?\"

---

## 1. 핵심 결론 (한 줄)

> **모델이 일반적일수록 prior 가 약함. Bitter Lesson: 더 약한 prior + 더 많은 데이터/계산 = 장기적으로 승리.**

---

## 2. 모델별 Inductive Bias

### 2.1 Linear Model — **매우 강**

$$
h(\mathbf{x}) = \theta^\top \mathbf{x}
$$

- Hypothesis space: linear functions only.
- 99.9% 의 함수 (비선형) 을 prior=0 로 배제.
- 표현력 ↓, 데이터 적게 필요, 일반화 좋음 (단순한 패턴 한정).

### 2.2 Nonlinear Basis (Polynomial / RBF)

$$
h(\mathbf{x}) = \sum_{j=1}^J \theta_j \phi_j(\mathbf{x}), \quad \phi_j \text{ fixed}
$$

- Basis 함수가 사람이 정한 \"특징\" → 그 특징의 선형 결합만 허용.
- Linear 보다 표현력 ↑, 그러나 basis 선택이 강한 prior.

### 2.3 Parametrized Feature Extractor

$$
h(\mathbf{x}) = \sum_j \theta_j \phi_j(\mathbf{x}; \alpha), \quad \alpha \text{ also learned}
$$

- 특징을 데이터에서 학습 → prior 약화.

### 2.4 2-Layer NN (MLP)

$$
h(\mathbf{x}) = W_2 \sigma(W_1 \mathbf{x} + b_1) + b_2
$$

- **Universal Approximation Theorem**: 충분히 많은 hidden unit 이면 any continuous function 근사 가능.
- 거의 모든 함수가 가능 → **prior 약함**.
- 그러나 **fully-connected** → \"입력 픽셀들 간에 어떤 관계가 있는지\" 의 prior 없음.

### 2.5 Deep NN

- 같은 universal 표현력 + **hierarchical** 표현 prior.
- \"단순 → 복잡\" feature 의 합성이라는 약한 prior 추가.

### 2.6 CNN — 약하지만 specific prior

추가 prior:
1. **Locality**: 가까운 픽셀이 관련 있다.
2. **Translation equivariance/invariance**: 패턴이 어디에 있든 같은 의미.

[10 Conv = Linear](10_Conv_Linear_증명.md) 에서 본 \"sparse + weight sharing\" 이 이 두 prior 의 수학적 형태.

→ Image data 에 적합한 prior. **MLP 보다 강함, Linear 보다 약함**.

### 2.7 Transformer — **매우 약**

- Self-attention: 모든 위치 간 관계 학습 → locality prior 없음.
- Position encoding 만 약한 위치 prior.
- Permutation-near-invariant.

→ 가장 일반적, 그래서 데이터 많이 필요. \"규모의 경제\" 가 되면 가장 강력.

---

## 3. 강도 순위 (Prior strong → weak)

```
Linear ≫ Nonlinear basis (fixed) > Parametrized basis
       > MLP ≈ Deep NN > CNN > Transformer
```

> 더 약한 prior = 더 큰 hypothesis space = 더 많은 데이터/파라미터 필요.

---

## 4. Bitter Lesson (Sutton)

> **\"인간이 도메인 지식을 손으로 넣은 prior 는 단기에는 좋지만, 장기적으로 데이터+계산 으로 학습한 약한 prior 가 이긴다.\"**

- 1980-90년대: hand-engineered features (SIFT, HOG)
- 2010s: CNN — locality prior 만 남기고 feature 는 학습
- 2020s: Transformer — 그것마저도 줄임 + scale

→ 시험에서 \"Bitter Lesson 의 의미\" 를 적으면 점수.

---

## 5. \"왜 더 약한 prior + 더 많은 데이터 = 승리\" 의 통계적 근거

### 5.1 Bias-Variance Tradeoff

- Strong prior: bias ↑, variance ↓ (작은 hypothesis space)
- Weak prior: bias ↓, variance ↑

데이터가 많아지면 variance 가 자연 감소 → weak prior 의 bias 우위가 부각.

### 5.2 Universal Approximation 의 의미

NN 의 prior 는 \"표현 가능한 함수의 종류\" 가 아니라 **\"어떤 함수가 더 쉽게 학습되는지\"** 의 prior. 이게 약한 만큼 학습이 어렵지만, 데이터/계산이 많아지면 극복.

---

## 6. 모범 답안 템플릿

```
[Claim]
Inductive bias = the implicit prior P(h) over hypothesis classes
embedded in the model architecture.
Strength: how much of the function space is excluded a priori.

[Ranking, strongest → weakest]
1. Linear:        h = θᵀx, only linear functions.
2. Nonlinear basis (fixed): h = Σ θ_j φ_j(x), basis hand-chosen.
3. Parametrized basis: φ_j has its own parameters, learned.
4. MLP / deep NN: universal approximator (UAT) — almost any
   continuous function. Still has fully-connected prior.
5. CNN: weakens MLP only by *adding* two specific priors
   (locality and translation equivariance) via sparse +
   weight-shared linear layers. So compared to MLP applied
   to images, CNN is a *useful structural restriction*; but
   relative to all possible neural functions, it is weaker
   than fully-engineered features.
6. Transformer: removes even spatial locality; only positional
   encodings remain. Permutation-near-invariant ⇒ weakest prior
   among the common deep models.

[Bitter Lesson]
Stronger priors win when data/compute is small. Weaker priors,
combined with much more data and compute, generalize better in
the long run. Empirically: hand-features → CNN → Transformer.

[Trade-off]
Weaker prior ⇒ larger effective hypothesis space ⇒ higher variance,
needs more data; stronger prior ⇒ lower variance, higher bias if
prior is misaligned with the true function.
```

---

## 7. 자주 틀리는 함정

1. **\"CNN 이 MLP 보다 prior 약함\" 이라고 답함** → 정확히는 **이미지에 한정해 더 적합한 prior**, 일반적으로는 MLP 보다 **추가 제한**.
2. **\"UAT 가 prior 가 없다는 뜻\" 이라고 답함** → 표현 가능 ≠ 학습 가능. NN 도 prior 가 있음 (예: gradient descent 가 어떤 함수를 선호하는가).
3. **Bitter Lesson 누락**: 7주차 ★9. 한 줄 적으면 점수.
4. **Bias-Variance 누락**: 정량적 근거.

---

## 8. 연결 개념

- ← [14 Hypothesis Space Restriction = MAP](14_Hypothesis_MAP.md)
- ← [10 Conv = Linear](10_Conv_Linear_증명.md): CNN prior 의 수학적 형태

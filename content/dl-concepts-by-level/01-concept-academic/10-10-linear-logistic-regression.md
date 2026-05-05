---
title: "10. 선형회귀 & 로지스틱 회귀 (Linear & Logistic Regression)"
slug: 10-linear-logistic-regression
order: 10
---

# 10. 선형회귀 & 로지스틱 회귀 (Linear & Logistic Regression)

## 1. 동기부여 및 개요

선형회귀와 로지스틱 회귀는 **딥러닝의 가장 단순한 형태의 신경망**이다:
- 뉴런 1개 = 선형회귀 (연속 출력)
- 뉴런 1개 + 시그모이드 = 로지스틱 회귀 (이진 분류)
- 뉴런 C개 + 소프트맥스 = 소프트맥스 회귀 (다중 분류)

이 두 모델에서 딥러닝의 모든 핵심 원리가 시작된다: 손실 함수 설계, MLE/MAP 추정, 경사 하강법, 정규화. 이 기초를 완벽히 이해하면 심층 신경망도 "선형 변환 + 비선형 활성화의 반복"으로 자연스럽게 확장할 수 있다.

**이 장의 핵심 통찰**: Cross-entropy와 MSE는 모두 **NLL(Negative Log-Likelihood)**이라는 같은 목적함수의 특수한 경우이다.

> **선수 지식**: 선형대수(행렬 곱, 역행렬), 미적분(편미분), 확률론(08장), 베이즈 정리(09장)

---

## 2. 지도학습 프레임워크

### 2.1 문제 설정

**Definition 2.1 (지도학습).**
데이터 $S = \{(x_i, y_i)\}_{i=1}^n$가 주어졌을 때, 새로운 입력 $x$에 대해 $y$를 잘 예측하는 함수 $h: \mathcal{X} \to \mathcal{Y}$를 학습한다.

| 문제 유형 | 출력 공간 $\mathcal{Y}$ | 예시 |
|:---|:---|:---|
| **분류** (Classification) | $\{1, 2, \ldots, C\}$ (이산) | 이미지 분류, 스팸 탐지 |
| **회귀** (Regression) | $\mathbb{R}$ (연속) | 집값 예측, 주가 예측 |

### 2.2 모집단 위험과 경험적 위험

**모집단 위험 (Population Risk)**:

$$L_\mathcal{P}(h) = \mathbb{E}_{(x,y) \sim \mathcal{P}}[\ell(y, h(x))]$$

모집단 전체를 알 수 없으므로, 훈련 데이터로 근사한다:

**경험적 위험 (Empirical Risk)**:

$$L_S(h) = \frac{1}{n}\sum_{i=1}^n \ell(y_i, h(x_i))$$

### 2.3 지도학습과 비지도학습의 확률적 관점

$$\text{비지도학습}: p(x) \quad \longleftrightarrow \quad \text{지도학습}: p(y \mid x)$$

비지도학습은 $p(x)$를 모델링(생성 모델), 지도학습은 $p(y \mid x)$를 모델링(판별 모델).

---

## 3. MLE, NLL, 그리고 손실 함수의 통합

### 3.1 핵심 아이디어

데이터가 i.i.d.라 가정하면:

$$\text{NLL}(h) = -\log P(S \mid h) = \sum_{i=1}^n -\log p(y_i \mid x_i, h) + C$$

### 3.2 분포 가정에 따른 NLL

이것이 이 장의 **가장 중요한 표**이다:

| 분포 가정 | NLL | 결과 손실 함수 |
|:---|:---|:---|
| $y_i \sim \text{Bern}(\theta)$ | $-[k\log\theta + (n-k)\log(1-\theta)]$ | **CE (Cross-Entropy)** |
| $y_i \sim \mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{2\sigma^2}\sum(y_i - \mu)^2 + C$ | **MSE** |
| 회귀: $y_i \mid x_i \sim \mathcal{N}(h(x_i), \sigma^2)$ | $\frac{n}{2\sigma^2}\text{MSE}(h) + C$ | **MSE** |
| 분류: $y_i \mid x_i \sim \text{Cat}(h(x_i))$ | $\sum_i -\log[h(x_i)]_{y_i}$ | **CE** |

**핵심 등식**:

$$\boxed{\text{가우시안 가정} + \text{MLE} = \text{MSE 최소화}}$$

$$\boxed{\text{범주형 가정} + \text{MLE} = \text{CE 최소화}}$$

### 3.3 NLL과 KL Divergence의 관계

경험적 분포 $p_S(z) = \frac{1}{n}\sum_i \delta(z - z_i)$에 대해:

$$KL(p_S \| q) = \underbrace{-H(p_S)}_{\text{상수}} + \underbrace{\frac{1}{n}\text{NLL}}_{\text{최소화 대상}}$$

따라서: **NLL 최소화 = KL 최소화 = MLE**

---

## 4. 경험적 위험 최소화 (ERM)

### 4.1 정의

**Definition 4.1 (ERM).**

$$h_S \in \arg\min_h L_S(h) = \arg\min_h \frac{1}{n}\sum_{i=1}^n \ell(y_i, h(x_i))$$

### 4.2 NLL과 ERM의 관계

MLE는 ERM의 **특수한 경우**: $\ell(y_i, h(x_i)) = -\log p(y_i \mid x_i; h)$로 설정하면 MLE가 곧 ERM.

### 4.3 파라미터화

함수 공간에서 직접 최소화할 수 없으므로, $h = f_w$로 **파라미터화**하여 유한 차원 $w$에 대해 최적화한다. 이것이 신경망 학습의 출발점이다.

---

## 5. 선형회귀 모델

### 5.1 모델 정의

$$f_w(x) = x^\top w + b$$

**편향 흡수 트릭**: $x \leftarrow [x^\top\; 1]^\top$, $w \leftarrow [w^\top\; b]^\top$이면 $f_w(x) = x^\top w$.

행렬 표현: $f_w(X) = Xw$, 여기서 $X \in \mathbb{R}^{n \times p}$ (설계 행렬).

### 5.2 확률적 해석

$$y_i = x_i^\top\beta + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)$$

오차 $\varepsilon$이 가우시안을 따른다는 가정은 CLT에 의해 정당화된다: 많은 미지의 독립적 요인의 합은 가우시안에 수렴.

---

## 6. 최소제곱법과 정규방정식

### 6.1 손실 함수

$$L_S(w) = \frac{1}{2}\|Xw - y\|^2 = \frac{1}{2}(w^\top X^\top Xw - 2y^\top Xw + y^\top y)$$

### 6.2 정규방정식의 유도

$\nabla_w L_S = X^\top Xw - X^\top y = 0$에서:

$$\boxed{X^\top X\hat{\beta} = X^\top y \quad \text{(정규방정식, Normal Equation)}}$$

### 6.3 해의 존재와 유일성

**Theorem 6.1.** 정규방정식은 항상 해가 존재한다.

| 조건 | 해 | 비고 |
|:---|:---|:---|
| $\text{rank}(X) = p$ | **유일**: $\hat{\beta} = (X^\top X)^{-1}X^\top y$ | $X^\top X$ 역행렬 존재 |
| $\text{rank}(X) < p$ (예: $n < p$) | **무한히 많음** | 최소 노름 해: $\hat{\beta} = X^+y$ (유사역행렬) |

### 6.4 기하학적 해석

$\hat{y} = X\hat{\beta}$는 $y$를 $\text{col}(X)$ (열공간)에 **직교 사영(orthogonal projection)**한 것이다. 잔차 $y - \hat{y}$는 열공간에 수직:

$$X^\top(y - X\hat{\beta}) = 0 \quad \Leftrightarrow \quad X^\top X\hat{\beta} = X^\top y$$

```
         y
        /|
       / |  ← 잔차 (y - ŷ) ⊥ col(X)
      /  |
     /   |
    /    |
   ──────●──── col(X) (열공간)
         ŷ = X β̂
```

---

## 7. 릿지 회귀와 정규화

### 7.1 릿지 회귀 (Ridge Regression)

$$L_S(w; \lambda) = \frac{1}{2}\|Xw - y\|^2 + \frac{\lambda}{2}\|w\|^2, \quad \lambda > 0$$

정규화된 정규방정식:

$$\boxed{\hat{\beta}_\lambda = (X^\top X + \lambda I)^{-1}X^\top y}$$

$X^\top X + \lambda I$는 항상 **양정치(positive definite)**이므로 역행렬이 존재하고, 해가 **항상 유일**하다.

### 7.2 확률적 해석: MAP 추정

릿지 회귀 = 가우시안 사전분포 $p(w) \propto \exp(-\frac{\lambda}{2}\|w\|^2)$를 가정한 **MAP 추정**:

$$\text{logpost} = \text{loglik} - \frac{\lambda}{2}\|w\|^2 + C$$

이것이 09장에서 다룬 MAP-정규화 연결의 구체적 실현이다.

### 7.3 $\ell_1$ vs $\ell_2$ 정규화 비교

| | $\ell_2$ (Ridge) | $\ell_1$ (LASSO) |
|:---|:---|:---|
| 정규화 항 | $\lambda\|w\|^2$ | $\lambda\|w\|_1$ |
| 사전 분포 | 가우시안 | 라플라스 |
| 해의 특성 | 작은 가중치 (weight decay) | **희소 해** (sparse, 일부 $w_i = 0$) |
| 닫힌 해 | 있음 | 없음 (수치적 풀이) |

---

## 8. 로지스틱 회귀 (Logistic Regression)

### 8.1 모델 정의

$$f_w(x) = \sigma(x^\top w) = \frac{1}{1 + \exp(-x^\top w)} \in (0, 1)$$

확률적 해석:
- $p_w(y = 1 \mid x) = \sigma(x^\top w)$
- $p_w(y = 0 \mid x) = 1 - \sigma(x^\top w)$

> **주의**: 이름이 "Logistic *Regression*"이지만, 실제로는 **이진 분류** 모델이다.

### 8.2 시그모이드와 퍼셉트론의 관계

```
        σ(z)               H(z)
    1 ─ ─ ─ ─/─── ─    1 ─ ─ ─ ─┬─── ─
             /                    │
            /                     │
           /                      │
    0 ───/─ ─ ─ ─ ─    0 ────────┘
        z                  z
    시그모이드 (미분 가능)   헤비사이드 (미분 불가)
    → 로지스틱 회귀          → 퍼셉트론
```

시그모이드를 사용하면 경사 하강법으로 학습 가능하다.

### 8.3 손실 함수: BCE (Binary Cross-Entropy)

$$L_S(w) = -\sum_{i=1}^n\left[y_i\log\sigma(x_i^\top w) + (1 - y_i)\log(1 - \sigma(x_i^\top w))\right]$$

이것은 **닫힌 해가 없어** 경사 하강법 등 수치적 최적화가 필요하다 (선형회귀와의 핵심 차이).

### 8.4 07장 미분방정식과의 연결

07장에서 보았듯이, 시그모이드는 로지스틱 ODE $x' = ax(1-x)$의 해이다. 포화 영역에서 $\sigma'(z) \approx 0$이 되는 것이 **vanishing gradient**의 근본 원인이다.

---

## 9. 소프트맥스 회귀 (다중 분류)

### 9.1 시그모이드에서 소프트맥스로

시그모이드의 일반화:

$$\sigma(z) = \frac{e^z}{e^z + 1} = \frac{e^z}{e^z + e^0} \quad \xrightarrow{\text{C개 클래스}} \quad \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}$$

### 9.2 모델

$$f_W(x) = \text{softmax}(Wx), \quad W \in \mathbb{R}^{C \times d}$$

### 9.3 손실 함수: Categorical Cross-Entropy

$$L_S(W) = -\sum_{i=1}^n \log[f_W(x_i)]_{y_i}$$

### 9.4 06장 최대 엔트로피와의 연결

06장에서 유도한 것처럼, softmax는 에너지 제약 하에서 엔트로피를 최대화하는 분포이다:

$$p_i = \frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}$$

온도 $\tau$를 조절하면 분포의 날카로움을 제어할 수 있다 (Knowledge Distillation, LLM 샘플링).

---

## 10. KL Divergence와 손실 함수의 통합적 이해

### 10.1 핵심 등식 체인

$$\arg\max_h \text{Likelihood} = \arg\min_h \text{NLL} = \arg\min_h \text{KL divergence}$$

가우시안 가정 하에서: $= \arg\min_h \text{MSE}$

### 10.2 실용적 의미

| 데이터에 대한 가정 | NLL이 유도하는 손실 | 닫힌 해 |
|:---|:---|:---|
| 가우시안 노이즈 | MSE | **있음** (정규방정식) |
| 베르누이/범주형 | CE | **없음** (수치적 풀이) |

```
가우시안 가정 ──→ NLL = MSE ──→ 정규방정식 (닫힌 해)
                                    │
                              릿지 정규화 (MAP)

베르누이 가정 ──→ NLL = BCE ──→ 경사 하강법 (수치 해)
                                    │
                              소프트맥스 (다중 클래스)

    [MLE / NLL / KL-div] ←── 모든 손실의 확률적 근거
```

---

## 11. 베이즈 최적 분류기와 생성 모델

### 11.1 베이즈 분류기

**Theorem 11.1 (베이즈 최적 분류기).**

$$f(x) = \arg\max_{k \in [C]} p(y = k \mid x)$$

이것이 오분류율을 최소화하는 이론적으로 최적인 분류기이다.

베이즈 정리 적용: $p(y \mid x) \propto p(x \mid y) \cdot p(y)$

### 11.2 나이브 베이즈 (Naive Bayes)

특징들이 조건부 독립이라는 ("나이브한") 가정:

$$p(x \mid y = k) = \prod_{j=1}^d p(x_j \mid y = k)$$

가정이 위반되더라도 실제로는 놀라울 정도로 잘 작동한다. 스팸 필터, 텍스트 분류의 강력한 베이스라인.

### 11.3 QDA와 LDA

각 클래스의 데이터가 가우시안: $p_k = \mathcal{N}(m_k, \Sigma_k)$

**QDA (Quadratic Discriminant Analysis)**: $\Sigma_1 \neq \Sigma_2$ -- **이차** 결정 경계

**LDA (Linear Discriminant Analysis)**: $\Sigma_1 = \Sigma_2 = \Sigma$ -- 이차 항이 소거되어 **선형** 결정 경계:

$$f(x) = \text{sign}[(m_1 - m_2)^\top\Sigma^{-1}x - b]$$

LDA는 로지스틱 회귀와 같은 선형 결정 경계를 가지지만, 접근 방식이 다르다: LDA는 **생성 모델**(각 클래스의 분포를 모델링), 로지스틱 회귀는 **판별 모델**(결정 경계를 직접 학습).

---

## 12. 역사적 기원: 가우스와 케레스

최소제곱법은 1801년 **칼 프리드리히 가우스**가 왜소행성 케레스의 궤도를 예측하기 위해 개발했다. 피아치가 40일간 19번 관측한 데이터만으로, 24살의 가우스는 관측 오차($\approx 30''$)를 고려한 궤도를 계산했고, 예측한 위치에서 케레스가 재발견되었다.

이 방법이 오늘날 딥러닝의 **MSE 손실 함수**로까지 이어져, 200년이 지난 지금도 매일 사용되고 있다.

---

## 13. 흔한 오해와 주의점

| 오해 | 실제 | 교정 |
|:---|:---|:---|
| MSE는 그냥 관례 | 가우시안 노이즈 가정 + MLE에서 유도됨 | 오차 분포가 라플라스면 MAE가 적절 |
| 로지스틱 회귀는 회귀 모델 | 이름과 달리 **이진 분류** 모델 | 시그모이드로 확률 출력 후 분류 |
| 정규방정식의 해는 항상 유일 | $\text{rank}(X) < p$이면 무한 해 | 릿지 정규화 또는 유사역행렬 사용 |
| MLE와 ERM은 완전히 별개 | MLE는 ERM의 특수 경우 (NLL을 손실로 사용) | 같은 프레임워크의 다른 관점 |
| CE와 MSE는 근본적으로 다름 | 둘 다 NLL의 특수 경우 | 분포 가정만 다를 뿐 |
| 나이브 베이즈의 독립 가정은 실용성 없음 | 가정 위반에도 잘 작동 | 결정 경계의 정확성이 중요 |

---

## 14. 핵심 요약

### 핵심 공식 체크리스트

| # | 공식 | 의미 |
|:---:|:---|:---|
| 1 | $f_w(x) = x^\top w$ | 선형회귀 모델 |
| 2 | $L_S(w) = \frac{1}{2}\|Xw - y\|^2$ | 최소제곱 손실 |
| 3 | $\hat{\beta} = (X^\top X)^{-1}X^\top y$ | 정규방정식 (유일 해) |
| 4 | $\hat{\beta}_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$ | 릿지 회귀 해 |
| 5 | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | 시그모이드 함수 |
| 6 | $L_S = -\sum_i[y_i\log\sigma + (1-y_i)\log(1-\sigma)]$ | BCE 손실 |
| 7 | $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | 소프트맥스 함수 |
| 8 | $\text{NLL} = \frac{n}{2\sigma^2}\text{MSE} + C$ | 가우시안 NLL = MSE |
| 9 | $f(x) = \arg\max_k p(y=k \mid x)$ | 베이즈 최적 분류기 |

**한 문장 요약**: 선형회귀와 로지스틱 회귀는 각각 "가우시안 노이즈 + MLE = MSE"와 "베르누이 가정 + MLE = CE"로, 둘 다 NLL 최소화라는 동일한 원리의 두 가지 얼굴이다.

---

## 참고 문헌

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 5: Machine Learning Basics.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapters 3-4.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, Chapters 8-10.

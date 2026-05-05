---
title: "1장 Inductive Bias — 시험의 척추"
slug: 01-inductive-bias
order: 1
---

# 1장 Inductive Bias — 시험의 척추

## 1.1 한 문장 정의 (외워야 함)

> **Inductive bias is the set of prior assumptions a learner uses to generalize beyond the training data.**

베이즈로 적으면

$$\underbrace{p(h \mid e)}_{\text{posterior}} \;\propto\; \underbrace{p(e \mid h)}_{\text{likelihood}} \;\cdot\; \underbrace{p(h)}_{\text{prior}}$$

여기서 $h \in \mathcal{H}$ (가설), $e$ (관측 데이터). **Inductive bias $=$ $p(h)$를 어떻게 깎느냐.**

- 강한 bias $\;\Longleftrightarrow\;$ $p(h)$가 좁은 영역에 집중 $\;\Longleftrightarrow\;$ $\mathcal{H}$가 작다
- 약한 bias $\;\Longleftrightarrow\;$ $p(h)$가 넓게 퍼짐 (극단: $p(h) \propto 1$, uniform)

> 직관: **가설 공간을 좁히는 것 $=$ prior를 강하게 거는 것.** 둘은 같은 말의 두 얼굴.

---

## 1.2 세 가지 동등 표현 (시험 직격)

이 동치는 7주차에서 교수님이 "**자유롭게 왕복하라**"고 강조한 부분입니다.

$$
\boxed{\;
\arg\max_{h \in \mathcal{H}} \; p(h \mid e)
\;\;\Longleftrightarrow\;\;
\arg\min_{\theta \in \Theta} \; \big[-\log p(e \mid \theta) - \log p(\theta)\big]
\;\;\Longleftrightarrow\;\;
\arg\min_{\theta} \; \underbrace{\frac{1}{N}\sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)}_{\text{Empirical Risk}} + \underbrace{\Omega(\theta)}_{\text{regularizer}}
\;}
$$

| 표현 | 의미 | 정체 |
|---|---|---|
| $\arg\max_h p(h \mid e)$ | posterior 최대 | **MAP** |
| $\arg\min_\theta -\log p(e\mid\theta)$ ($p(\theta)$ 균등) | NLL 최소 | **MLE** |
| $\arg\min_\theta \frac{1}{N}\sum \ell + \Omega$ | 손실 + 정규화 | **ERM** |

Prior $p(\theta)$의 정체:
- $p(\theta) \propto 1$ (uniform) $\;\Rightarrow\;$ 정규화 없음 $\;\Rightarrow\;$ **MLE $=$ ERM**
- $p(\theta) = \mathcal{N}(0, \sigma^2 I)$ $\;\Rightarrow\;$ $-\log p(\theta) = \frac{1}{2\sigma^2}\|\theta\|_2^2 + C$ $\;\Rightarrow\;$ **L2 정규화 (Ridge)**
- $p(\theta) = \text{Laplace}(0, b)$ $\;\Rightarrow\;$ $-\log p(\theta) = \frac{1}{b}\|\theta\|_1 + C$ $\;\Rightarrow\;$ **L1 정규화 (Lasso)**

> **이 표가 시험 한 문제 통째입니다.** "Show that L2 regularization corresponds to a Gaussian prior on the weights" 같은 문제가 6~9주차 스크립트 톤과 정확히 일치합니다.

---

## 1.3 가설 공간 사다리 (prior 강 → 약)

$$\mathcal{H}_{\text{linear}} \;\subset\; \mathcal{H}_{\text{NL basis}} \;\subset\; \mathcal{H}_{\text{2L NN}} \;\subset\; \mathcal{H}_{\text{Deep NN}}$$

| 모델 | 함수 형태 | Prior의 정체 |
|---|---|---|
| Linear | $f(x) = w^\top x + b$ | "세상은 직선이다" |
| NL basis | $f(x) = w^\top \phi(x)$, $\phi$ 고정 | "사람이 정한 feature가 좋다" |
| 2-layer NN | $f(x) = w^\top \sigma(W x + b)$ | "$\phi$도 학습한다" — Universal Approximation |
| CNN | $f$가 locality + translation 불변 | "이미지의 의미는 위치 무관·국소적" |
| Transformer | attention | "거의 없음, 데이터로 다 배운다" |

**Universal Approximation Theorem (2주차·7주차 강조)**: 충분히 wide한 2-layer NN은 임의의 연속함수를 임의 정밀도로 근사 가능.

$$\forall \varepsilon > 0,\; \forall f \in C(K),\; \exists\, \{w_i, b_i, v_i\}_{i=1}^{N} \;:\; \sup_{x \in K} \left| f(x) - \sum_{i=1}^{N} v_i \,\sigma(w_i^\top x + b_i) \right| < \varepsilon$$

> **함정**: "그럼 왜 Deep을 쓰나?" → width가 지수적으로 폭증, 최적화가 어려움. Deep은 **표현력이 아니라 효율적 표현**을 위한 것.

---

## 1.4 Bias–Variance 시험형 정리

$$\mathbb{E}\big[(y - \hat{f}(x))^2\big] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{noise}}$$

- Strong prior (작은 $\mathcal{H}$): **bias↑, variance↓** — 데이터 적을 때 유리
- Weak prior (큰 $\mathcal{H}$): **bias↓, variance↑** — 데이터 많을 때 유리

교수님 동전 예시: 3번 던져 다 앞면일 때
- MLE: $\hat{p} = \arg\max p(e\mid p) = 1$ → "항상 앞면" (과적합, weak prior 폭주)
- MAP with Beta$(\alpha, \beta)$ prior: $\hat{p} = \frac{\alpha + n_H - 1}{\alpha + \beta + n - 2}$ → 0.5 근처로 끌려옴 (strong prior가 데이터 부족을 보정)

---

## 1.5 Bitter Lesson — 시험에서 인용 가능

> "The biggest lesson from 70 years of AI research is that **general methods that leverage computation** are ultimately the most effective, by a large margin." — Sutton, 2019

수업적 번역:
$$\text{데이터 풍부} + \text{계산 풍부} \;\Longrightarrow\; \text{prior 약화 + scale 증대 = 승리}$$
$$\text{데이터 희소} \;\Longrightarrow\; \text{strong prior (CNN의 locality 등) 여전히 필수}$$

이게 **Linear → CNN → Transformer**의 역사적 곡선이 그리는 이야기입니다.

---

## 자가 점검

**Q1.** Why does L2 regularization correspond to a Gaussian prior on the weights? Derive it from $\arg\max_\theta p(\theta \mid e)$.

**Q2.** "Hypothesis space를 좁히는 것"과 "prior를 강하게 거는 것"이 동치인 이유를 한 문장으로 설명하시오.

**Q3.** CNN과 Transformer 중 어느 쪽이 inductive bias가 강한가? 그 bias의 구체적 내용은 무엇인가? 데이터가 100장일 때와 1억장일 때 어느 쪽이 유리한가?

**Q4.** Universal Approximation Theorem이 있다면 왜 deep network를 쓰는가?

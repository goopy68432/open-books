---
title: "기말고사 6~9주차 출제 심층 분석 + 풀이 보고서"
slug: exam-deep-analysis
order: 9
---

# 기말고사 6~9주차 출제 심층 분석 + 풀이 보고서

> **분석 대상:** `05_6주차_분석.md` ~ `08_9주차_분석.md` (4개 주차, 총 1,028줄)
> **출제 비중:** 기말고사의 약 60% (중간고사 변형 33% 별도)
> **목적:** 출제 가능성이 높은 개념·수식·증명을 근거와 함께 식별하고, **수식의 의미와 전개 과정을 모두 설명한 풀이**를 제공한다. 단순 수식 나열이 아니라 "왜 이런 식이 나오는지"를 줄마다 해설한다.

---

## §1. 출제 예측 요약 (Executive Summary)

### 1.1 통합 ★10/★9 핵심 항목 표

| 주차 | 항목 | ★ | 출제 형식 | 통합 시각에서의 위치 |
|---|---|---|---|---|
| 6 | Hypothesis Space Restriction = MAP | 10 | 개념·서술 | 전체 수업의 통합축 |
| 6 | Scalar→Function Hypothesis 확장 | 10 | 개념 | 패러다임 전환 |
| 6 | Bernoulli↔Classification, Gaussian↔Regression | 10 | 매핑·서술 | 핵심 통합 시각 |
| 7 | NLL ↔ ERM 동치 | 10 | 증명·유도 | Optimization 진입 게이트 |
| 7 | Linear→Nonlinear basis→2-layer NN의 prior 흐름 | 10 | 비교·서술 | Inductive Bias 척도 |
| 8 | Backpropagation | 10 | 계산 | 학습의 기본 |
| 8 | Chain Rule | 10 | 계산 | 모든 미분의 핵심 |
| 8 | Adam Optimizer | 10 | 정의 | LLM 학습 default |
| 8 | GD Update Rule | 10 | 정의·계산 | 옵티마이제이션 기초 |
| 9 | Conv = Sparse + Weight-Sharing Matrix | 10 | 증명·구성 | CNN의 핵심 정리 |
| 9 | Locality + Translation Invariance | 10 | 개념·서술 | CNN의 inductive bias |
| 9 | 이미지 처리 = Linear Transformation Restriction | 10 | 통합 시각 | 9주차 결론 |

### 1.2 출제 형식 분류 (예상)

| 형식 | 예시 | 비중 |
|---|---|---|
| ① 개념 정의·서술 | "Inductive Bias란 무엇인가" | 20% |
| ② 수식 유도 | Gaussian MLE → sample mean | 25% |
| ③ 증명 | Conv = Linear, NLL = ERM | 20% |
| ④ 계산 | Conv→Matrix 구성, AlexNet shape | 20% |
| ⑤ 비교·통합 | prior 강도 비교 | 15% |

### 1.3 주차별 출제 비중 추정

```
6주차 (MAP/Linear Reg)        : 약 15%
7주차 (NN/Inductive Bias)     : 약 18%
8주차 (Optimization/Backprop) : 약 22%
9주차 (CNN)                   : 약 20%
중간고사 변형                 : 약 25% (별도)
```

근거: 9주차 강의 마무리에서 "**저번 시험에 못 푼 것들에서 조금만 바꿔 출제 가능성**"을 명시.

---

## §2. 출제 1순위 — Tier S (★10, 출제 확실시)

### 2.1 Hypothesis Space Restriction = MAP의 통합 시각 (6주차)

**정의.** Prior로 가능한 영역을 좁히는 것 = inductive bias 주는 것 = MAP에서 prior로 영역 제한 = NN/CNN에서 architecture가 hypothesis space를 좁히는 것. **이 모두가 동일한 한 가지 행위.**

**출제 근거 (★10):**
- 교수 발언: "*이걸 이해하면 NN/CNN 다 이해*" (6주차 §3)
- 7주차 NN, 9주차 CNN까지 일관 적용되는 framing
- 6주차 §1에서 노란색 영역 = prior로 좁힌 영역으로 직접 시각화

**예상 문제 EX2.1.**
> Hypothesis Space Restriction과 MAP의 관계를 설명하고, 이를 활용해 (i) Linear Regression, (ii) CNN이 어떻게 같은 framework로 통합되는지 서술하라.

**모범답안.**

**Step 1. MAP의 정의 복기.** Bayes 정리에 의해

$$\theta^*_{\text{MAP}} = \arg\max_\theta \, \underbrace{P(D \mid \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}$$

여기서 prior $P(\theta)$는 **데이터를 보기 전 우리가 갖고 있는 사전 믿음**. 어떤 영역에서 $P(\theta)$가 높으면 그 영역을 선호, 0이면 그 영역을 완전히 배제.

> **풀이 포인트:** Prior가 0인 영역은 likelihood가 아무리 좋아도 posterior가 0. 즉 prior는 hypothesis 공간에서 "후보를 추리는" 역할.

**Step 2. Hypothesis Space와의 동치.** 학습은 hypothesis 집합 $\mathcal{H}$에서 데이터를 가장 잘 설명하는 $h$를 고르는 것. Prior로 일부 영역을 0으로 만드는 것은 곧 $\mathcal{H}$를 부분집합 $\mathcal{H}' \subset \mathcal{H}$로 좁히는 행위.

$$P(\theta) = 0 \text{ on } A \iff \mathcal{H}' = \mathcal{H} \setminus A$$

이게 "**MAP의 prior = hypothesis space restriction = inductive bias**"라는 통합 등식.

**Step 3. Linear Regression의 경우.** $\mathcal{H}_{\text{all}}$ = 모든 함수 $\mathbb{R}^n \to \mathbb{R}$. Linear regression은

$$\mathcal{H}_{\text{linear}} = \{h : h(x) = w^Tx + b\} \subset \mathcal{H}_{\text{all}}$$

만 고려. 비선형 함수에 대한 prior가 0인 hard restriction. → **선형성이라는 inductive bias.**

**Step 4. CNN의 경우.** Fully-connected layer는 모든 매트릭스 $A \in \mathbb{R}^{m\times n}$ 가능. CNN은

$$\mathcal{H}_{\text{CNN}} = \{A : A \text{ is sparse + weight-sharing}\} \subset \mathcal{H}_{\text{FC}}$$

만 사용. → **Locality + Translation Invariance라는 두 가지 inductive bias.**

**Step 5. 통합 결론.** 두 사례 모두 더 큰 hypothesis 공간의 부분집합으로 제한하는 prior. **차이는 prior의 모양뿐이고, 행위 자체는 같다.** 그래서 "Linear regression도 일종의 NN이고, CNN도 일종의 MAP"으로 읽을 수 있다.

> **풀이 포인트:** 시험에서는 "MAP의 prior와 architecture의 inductive bias가 같다"라는 한 문장의 등식을 얼마나 명확히 쓰는가가 채점 기준. "prior로 hypothesis space를 좁힘"이라는 표현 필수.

---

### 2.2 Scalar → Function Hypothesis 확장 (6주차)

**정의.** Bernoulli에서 hypothesis는 스칼라 $\theta$였지만, regression/classification에서는 함수 $h : X \to Y$. 함수란 **모든 $x$에서의 값을 한꺼번에 정한 것** = 무한 차원 hypothesis space.

**예상 문제 EX2.2.**
> Bernoulli MLE에서의 hypothesis $\theta$와 Linear Regression의 hypothesis $h(x) = ax + b$가 본질적으로 같은 framework임을 설명하라.

**모범답안.**

**Step 1. Bernoulli 복기.** $X_i \overset{\text{iid}}{\sim} \text{Bernoulli}(\theta)$. Hypothesis는 단 하나의 스칼라 $\theta \in [0,1]$. 의미: "$y=1$이 나올 확률은 $\theta$다." 모든 $i$에 대해 같은 확률.

**Step 2. Regression으로의 확장.** 입력 $x_i$가 추가되면 "$x$에 따라 $y$의 분포가 달라진다"는 모델로 확장. 자연스러운 가정:

$$y_i \mid x_i \sim \mathcal{N}(h(x_i),\, 1)$$

여기서 $h(x_i)$는 입력 $x_i$에서의 Gaussian의 **평균**. 즉 각 $x$마다 다른 평균을 갖는 Gaussian이 분포한다.

**Step 3. 시각화.** 각 $x_i$ 위에 90도 돌아간 Gaussian이 세로로 서있고, 평균이 $h(x_i)$. 모든 $x$를 따라가며 평균값을 이으면 그게 함수 $h$의 그래프.

> **풀이 포인트:** "함수"라는 개념을 추상적으로 보지 말고 "**모든 $x$에서의 값을 동시에 정한 것**"으로 읽는다. 그러면 Bernoulli의 $\theta$ → Regression의 $h$는 단지 차원 확장일 뿐.

**Step 4. MLE의 일반화.**

$$\theta^*_{\text{MLE}} = \arg\max_\theta \prod_i P(y_i \mid \theta) \;\;\xrightarrow{\;\;\text{함수 hypothesis}\;\;}\;\; h^*_{\text{MLE}} = \arg\max_h \prod_i P(y_i \mid h(x_i))$$

**Step 5. Linear Regression = Restricted Function MAP.** 함수 hypothesis 공간 전체가 아니라 $\{h : h(x) = ax + b\}$만 고려. 이는 "선형 함수 영역에서만 prior > 0, 그 외 영역에서 prior = 0"인 MAP의 특수 형태.

> **풀이 포인트:** 채점에서 "**스칼라 → 함수**" 차원 확장과 "**함수 공간에서 prior로 좁힘**"이라는 두 단계를 명시하면 만점.

---

### 2.3 Bernoulli↔Classification, Gaussian↔Regression 매핑 (6주차)

**정의.** 확률 모델이 무엇이냐에 따라 loss 함수가 자동으로 결정된다.

| 확률 모델 | 태스크 | Hypothesis $h$ | Loss |
|---|---|---|---|
| Bernoulli (2 outcomes) | Binary Classification | $h(x) = \sigma(w^Tx) \in [0,1]$ | BCE |
| Categorical (C outcomes) | Multi-class Classification | $h(x) = \text{softmax}(w^Tx) \in \Delta^{C-1}$ | Cross-Entropy |
| Gaussian | Regression | $h(x) \in \mathbb{R}$ | MSE (Squared loss) |

**예상 문제 EX2.3.**
> 위 매핑 표를 작성하고, "Gaussian likelihood → MSE" 한 케이스에 대해 직접 도출하라.

**모범답안.**

**Step 1. Gaussian likelihood 작성.** $y_i \mid x_i \sim \mathcal{N}(h(x_i), \sigma^2)$:

$$P(y_i \mid x_i, h) = \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right)$$

> **의미:** $y_i$가 평균 $h(x_i)$ 근처에서 종 모양으로 분포. $h$가 정확할수록 지수 안의 거리가 작고 likelihood가 크다.

**Step 2. NLL 작성.** $-\log$를 씌우면 곱 → 합으로 바뀌고, exp 안의 식이 그대로 내려온다.

$$-\log P(y_i \mid x_i, h) = \frac{(y_i - h(x_i))^2}{2\sigma^2} + \underbrace{\log(\sqrt{2\pi}\sigma)}_{=:\,C}$$

상수항 $C$는 $h$에 무관하므로 argmin에서 무시 가능.

**Step 3. 전체 데이터에 대한 NLL (IID 가정).**

$$-\log P(D \mid h) = \sum_i \frac{(y_i - h(x_i))^2}{2\sigma^2} + nC$$

**Step 4. argmin 변환.** $\sigma^2$이 고정 상수라면 $\frac{1}{2\sigma^2}$도 상수이므로 argmin이 동일:

$$h^* = \arg\min_h \sum_i (y_i - h(x_i))^2$$

이게 바로 **MSE (Mean Squared Error)** = Gaussian의 NLL.

> **풀이 포인트:** Gaussian의 $\exp(-\text{square})$ 구조가 log를 통과하면서 정확히 squared loss로 변환된다. **"확률 모델이 loss를 결정한다"**는 핵심 통찰. Categorical도 같은 방식 — $-\log h(x)_y$ → Cross-Entropy.

---

### 2.4 NLL ↔ ERM 동치 (7주차)

**정의.** Negative Log-Likelihood와 Empirical Risk Minimization은 본질적으로 같은 것이다.

$$\text{NLL}(h) = -\log P(D \mid h) \overset{\text{IID}}{=} \sum_i [-\log P(x_i \mid h)] = n \cdot \hat{L}_S(h)$$

여기서 $\ell(x_i, h) := -\log P(x_i \mid h)$가 "loss"의 정의이고, $\hat{L}_S(h) = \mathbb{E}_{x\sim P_S}[\ell(x, h)]$가 empirical risk.

**예상 문제 EX2.4.** §7 Q4 참조.

---

### 2.5 Linear → Nonlinear basis → 2-layer NN의 Prior 흐름 (7주차)

**정의.** Hypothesis space의 점진적 확장 = prior의 점진적 약화.

| 단계 | 모델 | 형태 | Prior 강도 |
|---|---|---|---|
| 1 | Linear | $f(x) = w^T x$ | 매우 강 |
| 2 | Nonlinear basis | $f(x) = w^T \phi(x)$ ($\phi$ 사람이 정함) | 강 |
| 3 | Parametrized basis | $f(x) = w^T \phi(x; w')$ ($\phi$도 학습) | 약 |
| 4 | 2-layer NN | $f(x) = w^T \sigma(w'^T x)$ | 매우 약 (universal approx) |
| 5 | Deep NN | 여러 층 반복 | 더 약 |
| 6 | CNN | sparse + weight-sharing | 강 (이미지 한정) |
| 7 | Transformer | full attention | 매우 약 |

**예상 문제 EX2.5.** §7 Q10 참조.

---

### 2.6 Backpropagation + Chain Rule (8주차)

**정의.** Computational graph에서 chain rule로 gradient를 효율적으로 계산. Loss로부터 leaf node로 거슬러 올라가며 누적.

**핵심 식.**

$$\frac{\partial L}{\partial w} = \sum_{\text{path}} \prod_{e \in \text{path}} \frac{\partial \text{(edge tail)}}{\partial \text{(edge head)}}$$

> **의미:** 각 edge에 부분 미분을 붙여놓고, $w$로부터 $L$까지의 모든 path의 곱을 합산. NN의 경우 path가 곧 layer 시퀀스.

**예상 문제 EX2.6.** §7 Q6 참조.

---

### 2.7 Adam Optimizer + GD (8주차)

**정의.**
- **GD:** $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$
- **SGD:** mini-batch $B$에서 $\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{|B|}\sum_{i\in B}\nabla \ell$
- **Momentum:** $m_t = \beta m_{t-1} + g_t$, $\theta_{t+1} = \theta_t - \eta m_t$
- **AdaGrad/RMSProp:** $s_t = \sum_t g_t^2$ 또는 EMA, $\theta_{t+1} = \theta_t - \eta g_t / \sqrt{s_t + \epsilon}$
- **Adam:** Momentum + RMSProp 결합 + bias correction

**예상 문제 EX2.7.** §7 Q5b 참조.

---

### 2.8 Conv = Sparse + Weight-Sharing Matrix (9주차)

**정의.** Convolution은 linear transformation이고, 표준기저로 매트릭스를 구성하면 (1) sparse, (2) weight sharing 두 패턴이 나타남. **이게 두 가지 inductive bias = Locality + Translation Invariance.**

**예상 문제 EX2.8.** §7 Q7 참조.

---

### 2.9 Locality + Translation Invariance (9주차)

**Locality.** Feature detection은 local 영역만 보면 됨 → 매트릭스 sparse 패턴.
**Translation Invariance.** 같은 feature는 위치 무관 → 매트릭스 weight sharing 패턴.

**예상 문제 EX2.9.**
> Conv의 두 가지 inductive bias를 정의하고, 매트릭스 표현에서 각각이 어떻게 나타나는지 설명하라. Fully Connected layer와 비교했을 때 hypothesis space가 어떻게 줄어드는지 정량적으로 답하라.

**모범답안.**

**Step 1. Locality의 의미.** "이미지에서 멀리 떨어진 픽셀은 서로 무관하다"는 가정. 예: 강아지의 눈을 검출할 때 반대편 모서리 픽셀과 곱하지 않아도 충분. 수학적으로 매트릭스 $A$의 각 행이 매우 적은 수의 nonzero entry만 가짐 → **sparse**.

**Step 2. Translation Invariance의 의미.** "같은 feature(예: 눈)는 이미지의 어느 위치에 있어도 동일한 detector로 검출 가능." 이를 수학으로 옮기면 같은 weight pattern이 이미지의 여러 위치에서 반복 사용됨 → **weight sharing**.

> **풀이 포인트:** 두 inductive bias 모두 "이미지가 그렇게 생겼을 것이다"라는 사람의 prior. Bitter Lesson이 말하듯 데이터가 충분하면 이런 prior 없이도 학습 가능 (Vision Transformer가 그 사례).

**Step 3. FC vs Conv 정량 비교.** 1D 입력 7차원 → 출력 6차원, kernel size 2 가정.
- FC: 매트릭스 $A \in \mathbb{R}^{6 \times 7}$ — **42개** parameter.
- Conv (kernel = $(w_1, w_2)$): 단 **2개** parameter.

→ 21배 감소. 그러나 표현 가능한 함수 집합도 좁아짐. **이게 "inductive bias로 hypothesis space를 좁힌 효과"의 정량 표현.**

**Step 4. Hypothesis space 부분집합 관계.** FC가 만들 수 있는 매트릭스 집합에서 sparse + weight-sharing 조건을 만족하는 부분집합이 곧 Conv. 즉

$$\mathcal{H}_{\text{Conv}} \subsetneq \mathcal{H}_{\text{FC}}$$

> **풀이 포인트:** "**Conv는 FC의 진부분집합**"이라는 한 줄과 "parameter 수의 비교 (2 vs 42)"가 채점 핵심.

---

### 2.10 이미지 처리 = Linear Transformation Restriction (9주차)

**정의.** 이미지 처리는 모든 매트릭스가 아니라 특정 매트릭스(sparse + weight sharing)만 사용. = Linear Algebra의 "linear transformation = matrix" 정리에 restriction을 추가한 것.

> **풀이 포인트:** "Linear algebra의 가장 중요한 정리를 그대로 conv에 적용한 것"이라는 9주차 마지막 발언이 시험 직전 메시지.

---

## §3. 출제 2순위 — Tier A (★9)

### 3.1 Restricted-Prior MAP — Boundary 답 (6주차) — §6.A에서 풀이
### 3.2 Gaussian Mean MLE → Sample Mean (6주차) — §6.B에서 풀이
### 3.3 Linear Regression Closed Form (8주차) — §6.E에서 풀이
### 3.4 Newton's Method = $L$의 2차 근사 Minimum (8주차) — §6.D에서 풀이
### 3.5 SGD가 Generalization 더 좋음 — 정성 단답 ("이유는 알려져 있지 않음")
### 3.6 Cross-Entropy as Categorical NLL — §6.C에서 풀이
### 3.7 Universal Approximation — §5에서 서술
### 3.8 Conv → Matrix 표준기저 구성 — §6.G에서 풀이
### 3.9 Output Size Formula 유도 — §6.H에서 풀이

(자세한 풀이는 §6에서 단계별로 전개.)

---

## §4. 출제 3순위 — Tier B (★7~8)

| 항목 | ★ | 출제 변형 |
|---|---|---|
| Restricted Uniform Prior PDF 작성 | 8 | Inside $1/(2a)$, outside 0 case 표기 |
| Logistic Regression 정의 | 8 | $h(x) = \sigma(w^Tx)$, 이름의 모순 |
| Softmax (Multi-class) | 8 | $h(x)$ = $C$차원 vector, 합=1 보장 |
| Bitter Lesson | 9 | 정성 서술 (한 문장 인용) |
| ReLU 미분 | 8 | 양수→1, 음수→0 |
| Same Conv 조건 | 8 | $2p = k - s$ 도출 |
| Max Pooling Non-linearity | 7 | 반례 구성 |
| AlexNet Layer Shape | 7 | Output size formula 적용 |
| XOR Problem | 8 | Linear로 못 푸는 이유 |
| Markov vs Transformer prior | 7 | 어느 쪽이 strong prior인가 |
| Inner Product = Similarity | 9 | $\|x\|\|y\|\cos\theta$ |
| Average Pooling Linear 증명 | 7 | Linear이므로 매트릭스 표현 가능 |

---

## §5. 통합 출제형 — "Inductive Bias 큰 그림"

§7 Q10에서 한 문제로 묶어 풀이.

---

## §6. 수식·증명 단계별 풀이 모음 (의미 + 전개 과정)

이 섹션은 교수가 칠판에서 직접 풀이한 모든 수식을 한 자리에 모아, **각 줄이 무엇을 의미하는지**와 **왜 이렇게 변형하는지**까지 모두 해설한다.

---

### §6.A. Restricted Uniform Prior MAP (n=k=3 예시)

**문제.** Prior $P(\theta) = \frac{1}{2a}$ for $\theta \in [\frac{1}{2}-a,\,\frac{1}{2}+a]$, 0 otherwise. 코인을 3번 던져 모두 head ($n=3, k=3$). $\theta^*_{\text{MAP}}$를 구하라.

**Step 1. Likelihood 작성.** 각 던지기는 Bernoulli이고 IID이므로:

$$P(D \mid \theta) = \prod_{i=1}^{3} \theta^{x_i}(1-\theta)^{1-x_i} = \theta^3 (1-\theta)^0 = \theta^3$$

> **의미:** 3번 모두 head이므로 $\sum x_i = 3$. $(1-\theta)$ 항이 모두 사라지고 $\theta^3$만 남음.

**Step 2. Posterior 작성 (Bayes).** Prior가 $[1/2-a, 1/2+a]$ 안에서만 nonzero:

$$P(\theta \mid D) \propto P(D \mid \theta) \cdot P(\theta) = \begin{cases} \theta^3 \cdot \frac{1}{2a} & \theta \in [1/2-a, 1/2+a] \\ 0 & \text{otherwise}\end{cases}$$

> **풀이 포인트:** Prior가 0인 구간에서 posterior도 0. Likelihood가 아무리 좋아도(MLE는 $\theta=1$이지만) 그 영역이 prior 밖이면 답이 될 수 없다.

**Step 3. Log Posterior로 변환.** 곱은 미분이 어렵고 단조 변환을 적용해도 argmax는 같으므로 log를 취한다. Inside 영역에서:

$$\log P(\theta \mid D) = 3\log\theta + \log\frac{1}{2a} + C$$

상수항 $\log\frac{1}{2a}$와 $C$는 $\theta$에 무관 → argmax에 영향 없음.

**Step 4. 함수 형태 분석.** Inside에서 maximize할 함수는 $f(\theta) = 3\log\theta$.

- $\frac{df}{d\theta} = \frac{3}{\theta} > 0$ for $\theta > 0$.
- 즉 $f$는 $\theta$에 대해 **단조 증가**.

> **풀이 포인트:** 단조 증가 함수의 최댓값은 **정의역 끝점**에서 달성된다. 미분 = 0인 내부 critical point가 없음.

**Step 5. 정의역 boundary에서 최댓값.** Inside 정의역은 $[1/2-a, 1/2+a]$. 최댓값은 upper bound:

$$\theta^*_{\text{MAP}} = \frac{1}{2} + a$$

**Step 6. 직관 해석.** MLE만 보면 $\theta = 1$ (모두 head이므로). 그러나 prior가 그 영역을 차단했기 때문에 prior가 허용하는 한도 내에서 가장 likelihood가 높은 점 = upper boundary로 데이터가 끌려옴.

> **풀이 포인트:** 시험에서 학생들이 가장 많이 틀린 부분은 답으로 $\theta = 1/2$ 또는 $\theta = 1$을 쓴 것. **Hard restriction prior에서는 boundary가 답**이라는 점을 명시.

---

### §6.B. Gaussian Mean MLE → Sample Mean

**문제.** $y_1, \ldots, y_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, 1)$. $\mu^*_{\text{MLE}}$를 도출하라.

**Step 1. Likelihood 작성.** 각 $y_i$가 평균 $\mu$, 분산 1인 Gaussian:

$$P(y_i \mid \mu) = \frac{1}{\sqrt{2\pi}}\exp\!\left(-\frac{(y_i - \mu)^2}{2}\right)$$

> **의미:** $y_i$가 $\mu$에서 멀어질수록 확률밀도가 종 모양으로 감소.

IID 가정으로 결합:

$$L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}}\exp\!\left(-\frac{(y_i - \mu)^2}{2}\right)$$

**Step 2. Log Likelihood.** 곱을 합으로 바꾸기 위해 log:

$$\log L(\mu) = -\frac{n}{2}\log(2\pi) - \sum_{i=1}^n \frac{(y_i - \mu)^2}{2}$$

> **풀이 포인트:** Log를 씌우면 (1) 곱→합으로 바뀌어 미분 쉬움, (2) exp 안의 식이 그대로 내려온다.

**Step 3. NLL.** Maximize $\log L$ ↔ minimize $-\log L$. 상수 $\frac{n}{2}\log(2\pi)$는 $\mu$에 무관:

$$\text{NLL}(\mu) = \sum_{i=1}^n \frac{(y_i - \mu)^2}{2} + C$$

**Step 4. 미분 = 0.** $\mu$에 대해 미분 (chain rule, $(y_i - \mu)^2$의 $\mu$에 대한 미분 = $-2(y_i - \mu)$):

$$\frac{d\,\text{NLL}}{d\mu} = \sum_i \frac{1}{2}\cdot 2(y_i - \mu)\cdot(-1) = -\sum_i (y_i - \mu)$$

이를 0으로:

$$\sum_i (y_i - \mu^*) = 0 \;\Rightarrow\; \sum_i y_i = n\mu^* \;\Rightarrow\; \boxed{\mu^*_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n y_i}$$

**Step 5. 직관 해석.** Gaussian의 평균 추정치 = **표본평균**. 이는 직관과 일치 — 데이터의 산술평균이 가장 likely한 $\mu$.

> **풀이 포인트:** 채점 핵심: ① IID 가정 명시 → 곱 → log → NLL ② 미분의 부호 ((-1)·(2)·(1/2) = -1) ③ 결과 해석 (sample mean).

---

### §6.C. Linear Regression with bias ($a, b$ 연립)

**문제.** $y_i \overset{\text{iid}}{\sim} \mathcal{N}(ax_i + b, 1)$. $a^*$, $b^*$를 도출하라.

**Step 1. NLL 작성.** §6.B와 같은 방식으로 ($\mu \to ax_i + b$):

$$L(a, b) = \frac{1}{2}\sum_{i=1}^n (y_i - ax_i - b)^2 + C$$

> **의미:** $h(x_i) = ax_i + b$가 $i$번째 Gaussian의 평균. 데이터가 직선에서 멀어질수록 NLL 증가.

**Step 2. $a$로 편미분.** $(y_i - ax_i - b)^2$를 $a$로 미분 시 chain rule: 외부 $2u$, 내부 $-x_i$. 따라서

$$\frac{\partial L}{\partial a} = \frac{1}{2}\sum_i 2(y_i - ax_i - b)\cdot(-x_i) = -\sum_i x_i(y_i - ax_i - b)$$

**Step 3. $b$로 편미분.** 마찬가지로 내부 미분 $-1$:

$$\frac{\partial L}{\partial b} = -\sum_i (y_i - ax_i - b)$$

**Step 4. 두 식 = 0으로 놓고 정리.**

식 (i): $\sum_i x_i y_i = a\sum_i x_i^2 + b\sum_i x_i$
식 (ii): $\sum_i y_i = a\sum_i x_i + nb$

> **풀이 포인트:** 식 (ii)에서 $\bar y = a\bar x + b$ 즉 $b = \bar y - a\bar x$ — "최적 직선은 데이터 무게중심을 지난다"는 잘 알려진 사실.

**Step 5. 연립 풀기.** (ii)에서 $b = \bar y - a\bar x$를 (i)에 대입:

$$\sum_i x_i y_i = a\sum_i x_i^2 + (\bar y - a\bar x)\sum_i x_i$$

$\sum x_i = n\bar x$, $\sum y_i = n\bar y$를 사용해 정리:

$$a^* = \frac{\sum_i x_i y_i - n\bar x \bar y}{\sum_i x_i^2 - n\bar x^2} = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}$$

$$b^* = \bar y - a^* \bar x$$

> **풀이 포인트:** $a^*$는 $x$와 $y$의 **공분산** ÷ $x$의 **분산** 형태. 통계학에서 잘 알려진 회귀 계수 공식.

---

### §6.D. Categorical NLL = Cross-Entropy

**문제.** Categorical 분포 $h(x_i) \in \Delta^{C-1}$ ($C$차원 확률 vector)에서 NLL을 도출하라.

**Step 1. PMF 작성.** $y_i \in \{1, \ldots, C\}$일 때 정답 클래스에 대한 확률:

$$P(y_i \mid x_i, h) = h(x_i)_{y_i}$$

> **의미:** $h(x_i)$가 길이 $C$인 확률 vector. $y_i$번째 성분이 정답 확률.

**Step 2. 결합 확률 (IID).** $\prod_i h(x_i)_{y_i}$.

**Step 3. NLL.** 합으로 분리:

$$-\log P(D \mid h) = -\sum_{i=1}^n \log h(x_i)_{y_i}$$

> **풀이 포인트:** 이게 정확히 **Cross-Entropy** 의 형태. One-hot ground truth $e_{y_i}$와 예측 분포 $h(x_i)$ 사이의 cross-entropy: $H(e_{y_i}, h(x_i)) = -\sum_c (e_{y_i})_c \log h(x_i)_c = -\log h(x_i)_{y_i}$.

**결론.** **Categorical likelihood ⇒ NLL = Cross-Entropy.** Classification에서 표준 손실 함수가 왜 Cross-Entropy인가에 대한 답이 곧 "Categorical 분포의 NLL이기 때문"이다.

---

### §6.E. NLL ↔ ERM 등치 증명

**문제.** $-\log P(D \mid h)$로부터 출발하여 $\hat L_S(h) = \mathbb{E}_{x\sim P_S}[\ell(x, h)]$ 형태로 변환하라.

**Step 1. IID로 인한 분리.**

$$-\log P(D \mid h) = -\log \prod_{i=1}^n P(x_i \mid h) = -\sum_{i=1}^n \log P(x_i \mid h)$$

> **풀이 포인트:** 곱의 log는 log의 합. IID 가정이 이 변환의 전제.

**Step 2. Loss 함수의 정의.**

$$\ell(x, h) := -\log P(x \mid h)$$

> **의미:** "**Loss는 NLL의 정의로부터 자연스럽게 나온다.**" Gaussian이면 squared, Categorical이면 Cross-Entropy.

**Step 3. 평균.** Argmin은 상수 곱에 무관하므로 $\frac{1}{n}$을 곱해도 됨:

$$\frac{1}{n}\sum_i \ell(x_i, h) = \hat L_S(h)$$

**Step 4. Empirical Distribution 도입.** Dirac delta로 표본 분포 정의:

$$P_S(x) := \frac{1}{n}\sum_{i=1}^n \delta(x - x_i)$$

> **의미:** $n$개의 표본이 각 위치에 $\frac{1}{n}$의 질량을 가진 이산 분포. 표본 외부에서는 0.

**Step 5. 기댓값으로 표현.**

$$\mathbb{E}_{x\sim P_S}[\ell(x, h)] = \int \ell(x, h)\, P_S(x)\, dx = \frac{1}{n}\sum_i \int \ell(x, h)\,\delta(x - x_i)\, dx = \frac{1}{n}\sum_i \ell(x_i, h)$$

**Step 6. 결론.**

$$\boxed{-\log P(D \mid h) = n \cdot \hat L_S(h) = n\cdot \mathbb{E}_{x\sim P_S}[\ell(x, h)]}$$

> **풀이 포인트:** **NLL과 ERM은 상수 $n$ 차이의 같은 식.** Loss를 $-\log P$로 정의하기만 하면 NLL minimize = ERM. 이게 7주차의 핵심 통합 시각.

---

### §6.F. Newton's Method = $L$의 2차 근사 Minimum 증명

**문제.** Newton's method update $\theta_{t+1} = \theta_t - f(\theta_t)/f'(\theta_t)$이 $f = L'$일 때 $L$에 대해 어떤 행위인지 도출하라.

**Step 1. $L$을 $\theta_t$에서 2차 Taylor 전개.**

$$\hat L(\theta) := L(\theta_t) + L'(\theta_t)(\theta - \theta_t) + \frac{1}{2}L''(\theta_t)(\theta - \theta_t)^2$$

> **의미:** $\hat L$은 quadratic이므로 (a) 1차 미분 = 0인 점이 (b) global min/max 중 하나. $L'' > 0$이면 minimum.

**Step 2. $\hat L$의 minimum 찾기.** 미분 = 0:

$$\frac{d\hat L}{d\theta} = L'(\theta_t) + L''(\theta_t)(\theta - \theta_t) = 0$$

풀면

$$\theta - \theta_t = -\frac{L'(\theta_t)}{L''(\theta_t)} \;\Rightarrow\; \theta_{t+1} = \theta_t - \frac{L'(\theta_t)}{L''(\theta_t)}$$

**Step 3. $f = L'$ 치환.**

$$f = L', \quad f' = L'' \;\Rightarrow\; \theta_{t+1} = \theta_t - \frac{f(\theta_t)}{f'(\theta_t)}$$

> **풀이 포인트:** Newton's method 식과 **정확히 일치.**

**Step 4. 의미 변환.**

- **표면적 의미:** $f(\theta) = 0$의 근사 zero를 1차 함수(접선)로 찾기.
- **심층 의미:** $L$을 quadratic으로 근사하고 그 minimum으로 점프.

> **풀이 포인트:** GD는 1차 근사(접선) → 그 방향으로 작은 보폭. Newton은 2차 근사(포물선) → 한 번에 minimum으로 점프. **그래서 Newton은 "preconditioned GD"** (precondition 행렬이 $1/L''$)로도 읽을 수 있다.

**Step 5. NN에서 못 쓰는 이유.** $L'' = $ Hessian은 $d \times d$ 차원. $d \sim 10^9$이면 Hessian 자체가 $10^{18}$ entries로 메모리 불가능. Inverse는 $O(d^3)$로 더 불가능. **그래서 Adam/RMSProp는 Hessian을 $g^2$의 EMA로 diagonal 근사한 것.**

---

### §6.G. Linear Regression Closed Form (Matrix 형태)

**문제.** $L(w) = \frac{1}{2}\|Xw - y\|^2$, $w^*$를 도출하라.

**Step 1. 식 전개.** 노름의 제곱 정의:

$$\|Xw - y\|^2 = (Xw - y)^T(Xw - y) = w^TX^TXw - 2w^TX^Ty + y^Ty$$

> **풀이 포인트:** $w^TX^Ty = (X^Ty)^Tw = y^TXw$ — scalar이므로 transpose 자유. 따라서 cross term $2w^TX^Ty$로 합쳐짐.

**Step 2. Vector 미분.** Quadratic form $\frac{1}{2}w^TAw$의 미분이 $Aw$ ($A$ symmetric일 때), linear form $w^Tb$의 미분이 $b$임을 사용:

$$\nabla_w L = X^TXw - X^Ty$$

**Step 3. = 0으로 놓기.**

$$X^TXw^* = X^Ty$$

이것이 **Normal Equation.** $A x = b$ 형태로 풀린다.

**Step 4. Inverse 적용.** $X^TX$가 invertible이라면:

$$\boxed{w^* = (X^TX)^{-1}X^Ty}$$

> **풀이 포인트:** ① Quadratic이므로 convex → 미분=0이 곧 global minimum. ② NN에서는 $L$이 비볼록이므로 closed form 없음 → GD가 유일한 방법. ③ $X^TX$의 invertibility는 $X$의 열이 linearly independent ($n \ge d$)일 때.

---

### §6.H. Backpropagation 3 미분 + Chain Rule (Softmax + CE)

**Setup.** $x \to z = w_1 x \to \tilde z = \text{ReLU}(z) \to g = w_2 \tilde z \to p = \text{softmax}(g) \to L = -\log p_y$.

**Step 1. $\partial L / \partial p_y$.** $L$이 $p_y$의 함수일 때:

$$\frac{\partial L}{\partial p_y} = \frac{\partial}{\partial p_y}(-\log p_y) = -\frac{1}{p_y}$$

> **의미:** 정답 클래스 확률이 작을수록 gradient가 큼 (학습 신호 강함).

**Step 2. Softmax Jacobian $\partial p_i / \partial g_j$.** Softmax: $p_i = e^{g_i}/S$, $S = \sum_k e^{g_k}$.

**Case 1: $i = j$.**

$$\frac{\partial p_i}{\partial g_i} = \frac{e^{g_i}\cdot S - e^{g_i}\cdot e^{g_i}}{S^2} = \frac{e^{g_i}}{S}\cdot\frac{S - e^{g_i}}{S} = p_i(1 - p_i)$$

> **풀이 포인트:** Quotient rule. 분자 $e^{g_i}$의 $g_i$ 미분 = $e^{g_i}$. 분모 $S$의 $g_i$ 미분 = $e^{g_i}$.

**Case 2: $i \ne j$.**

$$\frac{\partial p_i}{\partial g_j} = \frac{0\cdot S - e^{g_i}\cdot e^{g_j}}{S^2} = -p_i p_j$$

**통합 (Kronecker delta):**

$$\frac{\partial p_i}{\partial g_j} = p_i(\delta_{ij} - p_j)$$

> **풀이 포인트:** $\delta_{ij}$ 누락이 가장 흔한 실수. $\delta_{ii} = 1$이면 $p_i(1-p_i)$, $\delta_{ij} = 0$이면 $-p_i p_j$로 자연스럽게 두 case가 통합됨.

**Step 3. $\partial g_i / \partial w_2$.** $g = w_2 \tilde z$이므로 $g_i = \sum_j (w_2)_{ij} \tilde z_j$. 따라서

$$\frac{\partial g_i}{\partial (w_2)_{ij}} = \tilde z_j$$

매트릭스 전체로는: $\partial g / \partial w_2 = \tilde z$ (외적 형태로 결합).

**Step 4. Chain rule 결합 ($L \to g$의 collapse).**

$$\frac{\partial L}{\partial g_j} = \sum_i \frac{\partial L}{\partial p_i}\cdot\frac{\partial p_i}{\partial g_j}$$

여기서 $L = -\log p_y$이므로 $\partial L/\partial p_i = -\frac{1}{p_y}$ if $i = y$, 0 otherwise. → 합이 단 한 항만 남음:

$$\frac{\partial L}{\partial g_j} = -\frac{1}{p_y}\cdot p_y(\delta_{yj} - p_j) = -(\delta_{yj} - p_j) = p_j - \delta_{yj}$$

**벡터 형태:**

$$\boxed{\frac{\partial L}{\partial g} = p - e_y}$$

> **풀이 포인트:** **Softmax + Cross-Entropy의 결합 미분이 정확히 $p - e_y$.** 이 깔끔한 형태가 NN 구현의 표준 — softmax와 CE는 항상 함께 다뤄지는 이유.

**Step 5. $w_2$로의 gradient.**

$$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial g}\cdot\frac{\partial g}{\partial w_2} = (p - e_y)\,\tilde z^T$$

> **의미:** 정답일 때 $p_y - 1 < 0$이므로 그 행의 weight가 증가하는 방향, 오답일 때 $p_j > 0$이므로 그 행의 weight가 감소하는 방향. **이게 "정답을 강화, 오답을 약화"라는 학습 직관의 수학적 표현.**

**Step 6. ReLU로의 backprop.**

$$\frac{\partial L}{\partial \tilde z} = w_2^T (p - e_y)$$

ReLU의 미분: $\mathbb 1[z > 0]$ (양수면 1, 음수면 0). 따라서

$$\frac{\partial L}{\partial z} = w_2^T(p - e_y) \odot \mathbb 1[z > 0]$$

마지막으로 $w_1$로:

$$\frac{\partial L}{\partial w_1} = \left(w_2^T(p - e_y) \odot \mathbb 1[z > 0]\right) x^T$$

> **풀이 포인트:** 모든 backprop의 패턴 — ① 출력에서 시작 ② edge별 미분 곱하기 ③ leaf까지 누적. 차원 검증: $w_1 \in \mathbb R^{h\times d}$, gradient도 $h\times d$. 양변 차원이 맞는지 항상 확인.

---

### §6.I. Conv → Matrix 표준기저 구성 (1D)

**문제.** $w = (1, 2)$, input dim 7, output dim 6. 대응 매트릭스 $A \in \mathbb R^{6\times 7}$를 표준기저 방법으로 구성하라.

**Step 1. 표준기저의 conv 결과 계산.** Convolution은 kernel이 input 위에서 sliding하며 inner product. $e_1 = (1,0,0,0,0,0,0)^T$일 때:

- 위치 0: $w \cdot (e_1)_{0:2} = 1\cdot 1 + 2\cdot 0 = 1$
- 위치 1: $w \cdot (e_1)_{1:3} = 1\cdot 0 + 2\cdot 0 = 0$
- ... 모두 0

따라서 $f(e_1) = (1, 0, 0, 0, 0, 0)^T$.

**$e_2 = (0,1,0,0,0,0,0)^T$:**

- 위치 0: $w \cdot (0, 1) = 0\cdot 1 + 1\cdot 2 = 2$
- 위치 1: $w \cdot (1, 0) = 1\cdot 1 + 0\cdot 2 = 1$
- ... 모두 0

→ $f(e_2) = (2, 1, 0, 0, 0, 0)^T$.

> **풀이 포인트:** 기저 벡터의 1이 어느 위치에 있는지에 따라 kernel과의 곱이 어디서 nonzero인지 결정된다. $e_i$의 conv 결과는 kernel이 $i$번째 위치에 정렬되었을 때의 형태.

**Step 2. 모든 $e_i$ 반복.**

| $e_i$ | $f(e_i)$ |
|---|---|
| $e_1$ | $(1, 0, 0, 0, 0, 0)^T$ |
| $e_2$ | $(2, 1, 0, 0, 0, 0)^T$ |
| $e_3$ | $(0, 2, 1, 0, 0, 0)^T$ |
| $e_4$ | $(0, 0, 2, 1, 0, 0)^T$ |
| $e_5$ | $(0, 0, 0, 2, 1, 0)^T$ |
| $e_6$ | $(0, 0, 0, 0, 2, 1)^T$ |
| $e_7$ | $(0, 0, 0, 0, 0, 2)^T$ |

**Step 3. Column으로 stack.** Linear transformation의 매트릭스 표현: $A = [f(e_1) \mid f(e_2) \mid \cdots \mid f(e_7)]$.

$$A = \begin{pmatrix} 1 & 2 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 2 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 2 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 2 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 2 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 2 \end{pmatrix}$$

**Step 4. 두 가지 inductive bias 관찰.**

- **Sparse:** 매트릭스 entry의 대부분이 0. 각 행이 단 2개의 nonzero. 이게 **Locality**.
- **Weight sharing:** $(1, 2)$가 모든 행에 한 칸씩 시프트되며 반복. 이게 **Translation Invariance**.

> **풀이 포인트:** ① **표준기저 방법** = "선형변환 $T$의 매트릭스를 찾으려면 $T(e_i)$를 column으로 stack" (1주차의 핵심 정리) ② Conv가 매트릭스 곱의 진부분집합임을 시각적으로 확인.

**Parameter 수 비교.** FC라면 $6 \times 7 = 42$개. Conv는 $w_1, w_2$ 단 2개. **21배 절약 = inductive bias의 정량 효과.**

---

### §6.J. Output Size Formula 박스 슬라이딩 유도

**문제.** $H_{\text{out}} = \lfloor (H_{\text{in}} + 2p - k)/s \rfloor + 1$를 박스 슬라이딩 직관으로 유도하라.

**Step 1. Padding 적용 후 width.** Input width $H_{\text{in}}$의 양쪽에 $p$만큼씩 0 패딩 → 실제 conv가 적용되는 width:

$$H_{\text{padded}} = H_{\text{in}} + 2p$$

**Step 2. Kernel이 갈 수 있는 마지막 위치.** Kernel의 왼쪽 끝이 위치 0에서 시작. 마지막에는 kernel의 오른쪽 끝이 $H_{\text{padded}} - 1$에 도달. 따라서 kernel 왼쪽 끝의 마지막 위치:

$$\text{last position} = H_{\text{padded}} - k = H_{\text{in}} + 2p - k$$

> **풀이 포인트:** Kernel size $k$ 만큼은 빼야 함 — kernel이 input을 벗어날 수 없기 때문.

**Step 3. Stride로 나눈 이동 횟수.** Kernel이 한 번에 $s$칸씩 이동. 0부터 last position까지 도달하는 이동 횟수:

$$\text{steps} = \frac{H_{\text{in}} + 2p - k}{s}$$

**Step 4. 첫 위치 포함 (+1).** 위치 0도 출력 한 칸을 만들므로:

$$H_{\text{out}} = \frac{H_{\text{in}} + 2p - k}{s} + 1$$

소수가 나오면 floor.

> **풀이 포인트:** "외우지 말고 유도하라"는 교수의 강조. 박스가 sweep하는 그림을 머리속으로 그려야 자연스럽게 유도됨.

**Same Convolution 조건.** $H_{\text{out}} = H_{\text{in}}$이려면

$$H_{\text{in}} = \frac{H_{\text{in}} + 2p - k}{s} + 1$$

$s = 1$일 때 정리하면:

$$H_{\text{in}} - 1 = H_{\text{in}} + 2p - k \;\Rightarrow\; 2p = k - 1$$

일반적으로 $2p = k - s$.

---

### §6.K. Max Pooling Non-linearity 반례

**문제.** $2\times 2$ Max Pooling이 linear transformation인지 판정.

**Step 1. Linearity 정의 복기.** $f$가 linear ⟺ $f(x_1 + x_2) = f(x_1) + f(x_2)$ AND $f(\alpha x) = \alpha f(x)$.

**Step 2. 반례 구성.**

$$x_1 = \begin{pmatrix} 1 & 0 \\ 0 & 0\end{pmatrix},\qquad x_2 = \begin{pmatrix} 0 & 1 \\ 0 & 0\end{pmatrix}$$

**Step 3. 각각의 max.**

- $\max(x_1) = 1$
- $\max(x_2) = 1$
- $f(x_1) + f(x_2) = 1 + 1 = 2$

**Step 4. 합의 max.**

$$x_1 + x_2 = \begin{pmatrix} 1 & 1 \\ 0 & 0\end{pmatrix},\quad \max(x_1 + x_2) = 1$$

**Step 5. 비교.** $\max(x_1 + x_2) = 1 \ne 2 = \max(x_1) + \max(x_2)$. **Additivity 위반.** → **Max Pooling은 non-linear.**

> **풀이 포인트:** ① 한 가지 반례만 있으면 non-linearity 증명 충분 ② Average Pooling은 linear ($\text{avg}(x_1 + x_2) = \text{avg}(x_1) + \text{avg}(x_2)$이고 매트릭스 표현 가능) ③ NN에서 비선형성은 활성화 함수뿐 아니라 max pooling으로도 추가됨.

---

## §7. 예상 문제 세트 (10문항) — 깊이 있는 모범답안

각 문제는 (i) 문제 (ii) 풀이 (iii) 채점 포인트 (iv) 풀이 포인트의 4단 구성.

---

### Q1. Restricted Prior MAP — n=2/k=1 변형 (★9, 6주차)

**문제.** Prior $P(\theta) = \frac{1}{2a}$ for $\theta \in [\frac{1}{2} - a, \frac{1}{2} + a]$, 0 otherwise. 코인 2번 던져 head 1번. $\theta^*_{\text{MAP}}$를 구하라.

**풀이.**

**Step 1. Likelihood.** $P(D \mid \theta) = \theta^1 (1-\theta)^1 = \theta(1-\theta)$.

**Step 2. Inside log posterior.**

$$\log P(\theta \mid D) = \log\theta + \log(1-\theta) + C$$

**Step 3. 미분 = 0.**

$$\frac{d}{d\theta}[\log\theta + \log(1-\theta)] = \frac{1}{\theta} - \frac{1}{1-\theta} = 0$$

$\Rightarrow 1 - \theta = \theta \Rightarrow \theta = \frac{1}{2}$.

**Step 4. Boundary 검증.** $\theta = \frac{1}{2} \in [\frac{1}{2} - a, \frac{1}{2} + a]$ ✓ — interior critical point. 따라서 boundary 답이 아니라 내부 최댓값.

**Step 5. 결과.** $\boxed{\theta^*_{\text{MAP}} = \frac{1}{2}}$.

**채점 포인트.** ① Likelihood 정확히 ② 미분 = 0 풀이 ③ Boundary와 interior 구분 — n=k=3에서는 boundary, n=2/k=1에서는 interior라는 점 명시.

> **풀이 포인트:** Restricted prior MAP 문제의 모든 핵심 — inside가 단조이면 boundary, 내부 critical point가 있으면 그것이 답. 데이터 비율 $k/n$이 prior 영역 안에 있으면 내부 답, 밖에 있으면 boundary 답.

---

### Q2. Gaussian Mean MLE 도출 (★9, 6주차)

**문제.** $y_1, \ldots, y_n \overset{\text{iid}}{\sim} \mathcal N(\mu, 1)$. $\mu^*_{\text{MLE}}$를 단계별로 도출하고 결과를 직관으로 해석하라.

**풀이.** §6.B 그대로. 결과: $\mu^* = \frac{1}{n}\sum y_i$ = 표본평균.

**채점 포인트.** ① IID로 곱 → log → NLL ② 미분 부호 ③ 결과의 직관적 의미 (sample mean).

> **풀이 포인트:** "왜 sample mean인가"에 대한 답: Gaussian의 평균은 분포의 중심이고, MLE는 데이터를 가장 잘 설명하는 중심 = 산술평균.

---

### Q3. Linear Regression $h(x) = ax$ Closed Form (★9, 6/8주차)

**문제.** $y_i \overset{\text{iid}}{\sim} \mathcal N(ax_i, 1)$. $a^*$를 도출하라.

**풀이.**

**Step 1. NLL.** $L(a) = \frac{1}{2}\sum (y_i - ax_i)^2 + C$.

**Step 2. 미분.** 외부 $2(y_i - ax_i)$, 내부 $-x_i$:

$$\frac{dL}{da} = -\sum_i x_i(y_i - ax_i)$$

**Step 3. = 0 풀기.**

$$\sum_i x_i y_i = a\sum_i x_i^2 \;\Rightarrow\; \boxed{a^* = \frac{\sum_i x_i y_i}{\sum_i x_i^2}}$$

**Step 4. 직관.** 분자 = $x$와 $y$의 inner product. 분모 = $x$의 norm squared. 그래서 $a^* = \frac{\langle x, y\rangle}{\|x\|^2}$ — **$y$를 $x$ 방향으로 projection했을 때의 계수.**

**채점 포인트.** ① NLL → 미분 ② 결과 표현 ③ Inner product 형태로의 해석 (보너스).

> **풀이 포인트:** $h(x) = ax + b$의 일반 case는 식 (i), (ii) 연립 (§6.C). $b = 0$ 가정 시 식 (i) 하나만 남고 closed form이 더 단순.

---

### Q4. NLL = ERM 등치 증명 (★10, 7주차)

**문제.** $-\log P(D \mid h)$로부터 $\hat L_S(h) = \mathbb E_{x\sim P_S}[\ell(x, h)]$ 형태로 변환하는 과정을 단계별로 도출하라. $P_S$의 명시적 형태도 함께 쓰라.

**풀이.** §6.E 그대로. 결과: NLL = $n\cdot \hat L_S$, $\ell = -\log P$, $P_S = \frac{1}{n}\sum \delta(x-x_i)$.

**채점 포인트.** ① IID → 곱 → log → 합 ② $\ell = -\log P$ 정의 ③ Empirical distribution $P_S$의 명시적 형태 (Dirac delta) ④ NLL = $n\cdot \hat L_S$ 결론.

> **풀이 포인트:** "Loss의 선택은 likelihood model이 결정한다"는 통찰을 한 줄로 답할 수 있어야. Gaussian → squared, Categorical → CE, Bernoulli → BCE.

---

### Q5. Optimization 알고리즘 비교 (★10, 8주차)

**문제 (a).** GD, SGD, Momentum, Adam의 update 식을 모두 쓰고 각각의 직관을 한 줄로 설명하라.

**풀이.**

| 알고리즘 | Update 식 | 직관 |
|---|---|---|
| GD | $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$ | 전체 데이터의 gradient 반대 방향 |
| SGD | $\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{|B|}\sum_{i\in B}\nabla\ell_i$ | Mini-batch로 stochastic — 빠르고 이상하게 generalization 더 좋음 |
| Momentum | $m_t = \beta m_{t-1} + g_t,\;\theta_{t+1} = \theta_t - \eta m_t$ | 이전 gradient의 EMA — 관성 |
| AdaGrad/RMSProp | $s_t = \text{EMA}(g_t^2),\;\theta_{t+1} = \theta_t - \eta g_t/\sqrt{s_t}+\epsilon$ | Hessian의 diagonal 근사 — 곡률 큰 방향 보폭 줄임 |
| Adam | $\hat m_t = m_t/(1-\beta_1^t),\;\hat s_t = s_t/(1-\beta_2^t),\;\theta_{t+1} = \theta_t - \eta \hat m_t/\sqrt{\hat s_t}+\epsilon$ | Momentum + RMSProp + bias correction |

**문제 (b).** Adam이 어떤 두 가지 아이디어의 결합인지 설명하고, NN에서 Newton's method를 직접 쓰지 않는 이유를 답하라.

**풀이.**

**Adam의 두 결합:**
- **Momentum (1차 모멘트):** $m_t$가 gradient의 EMA → noisy gradient에서도 방향성 유지.
- **RMSProp (2차 모멘트):** $s_t$가 gradient 제곱의 EMA → 곡률이 큰 방향에서 보폭을 작게 자동 조정.
- **Bias correction:** 초기 step에서 $m_t, s_t$가 0으로 편향된 것을 $1 - \beta^t$로 보정.

**Newton 사용 불가 이유:**
- Hessian은 $d \times d$ 차원. NN에서 $d \sim 10^9$이면 Hessian이 $10^{18}$ entries — 메모리 한계 초과.
- Inverse 계산은 $O(d^3)$ — 시간 한계 초과.
- 그래서 AdaGrad/RMSProp/Adam은 Hessian의 **diagonal 근사** ($g^2$의 EMA = "각 좌표의 곡률 추정"). 이게 second-order method의 1차 근사 비용으로의 환원.

**채점 포인트.** ① 4개 update 식 모두 정확 ② Adam = Momentum + RMSProp 결합 명시 ③ Newton 못 쓰는 이유 — Hessian 차원 + inverse cost.

> **풀이 포인트:** "왜 SGD가 GD보다 generalization이 좋은가" — **이유는 알려지지 않았다**가 정답 (교수 명시). "직관과 반대지만 큰 learning rate가 좋다"도 같은 맥락.

---

### Q6. Backprop 3 미분 + Chain Rule (★10, 8주차)

**문제.** $x \to z = w_1 x \to \tilde z = \text{ReLU}(z) \to g = w_2 \tilde z \to p = \text{softmax}(g) \to L = -\log p_y$.
- (a) $\partial L / \partial p_y$, $\partial p_y / \partial g_j$, $\partial g / \partial w_2$를 각각 계산하라.
- (b) Chain rule로 $\partial L / \partial g$를 구하고 단순한 형태로 정리하라.
- (c) $\partial L / \partial w_2$를 답하라.

**풀이.** §6.H의 Step 1~5 그대로.

**핵심 결과.**
- (a) $-1/p_y$, $p_y(\delta_{yj} - p_j)$, $\tilde z$.
- (b) $\partial L/\partial g = p - e_y$ — softmax + CE의 collapse.
- (c) $\partial L/\partial w_2 = (p - e_y)\tilde z^T$.

**채점 포인트.**
- Softmax Jacobian의 $\delta_{ij}$ 항 정확
- Quotient rule 적용 (분자 미분 = 분모 미분 ≠ 0인 경우)
- Collapse 후 $p - e_y$의 깔끔한 형태 도출
- 차원 검증

> **풀이 포인트:** $\partial L / \partial g = p - e_y$가 NN 구현의 표준 — softmax+CE를 따로 구현하면 numerical 불안정, 함께 묶으면 안정. 이 collapse 식을 외우자.

---

### Q7. 1D Conv → Matrix 구성 (★10, 9주차) — 변형판

**문제.** $w = (1, 2, 3)$, input dim 5, stride 1, no padding. 대응 매트릭스 $A$를 표준기저 방법으로 구성하고, sparse 패턴과 weight sharing 위치를 표시하라. 또한 FC와 비교한 parameter 수 절약을 답하라.

**풀이.**

**Step 1. Output dim.** $H_{\text{out}} = (5 - 3)/1 + 1 = 3$. → $A \in \mathbb R^{3\times 5}$.

**Step 2. 표준기저의 conv.**

| $e_i$ | $f(e_i)$ | 설명 |
|---|---|---|
| $e_1$ | $(1, 0, 0)^T$ | 첫 위치에서 $1\cdot 1$만 nonzero |
| $e_2$ | $(2, 1, 0)^T$ | 첫 위치 $1\cdot 2 = 2$, 둘째 위치 $1\cdot 1 = 1$ |
| $e_3$ | $(3, 2, 1)^T$ | 세 위치 모두 1 곱해짐 |
| $e_4$ | $(0, 3, 2)^T$ | 둘째/셋째 위치만 |
| $e_5$ | $(0, 0, 3)^T$ | 마지막 위치에서 $1\cdot 3$ |

> **풀이 포인트:** $e_3$가 가장 "꽉 찬" 결과 ($1, 2, 3$ 전체가 contributing). 양 끝의 $e_1, e_5$는 한 entry만 nonzero — kernel과의 overlap이 작기 때문.

**Step 3. Stack columns.**

$$A = \begin{pmatrix} 1 & 2 & 3 & 0 & 0 \\ 0 & 1 & 2 & 3 & 0 \\ 0 & 0 & 1 & 2 & 3 \end{pmatrix}$$

**Step 4. 두 가지 inductive bias.**

- **Sparse:** 좌상/우하 삼각 영역에 0. 각 행은 단 3개의 nonzero entry. **= Locality** (output 위치 근처 input만 영향).
- **Weight sharing:** $(1, 2, 3)$이 각 행에서 한 칸씩 시프트되며 반복. **= Translation Invariance** (같은 detector가 위치마다 사용됨).

**Step 5. Parameter 수 비교.** FC: $3 \times 5 = 15$. Conv: 3 (kernel weights). **5배 절약.**

**채점 포인트.**
- 표준기저 방법 명시
- 매트릭스 정확성 (특히 한 칸 시프트 패턴)
- Sparse / Weight sharing 시각적 표시
- Parameter 절약 정량화

> **풀이 포인트:** "**선형변환의 매트릭스 = $T(e_i)$를 column으로 stack**"이 1주차 핵심 정리. Conv는 그 정리의 응용 사례.

---

### Q8. AlexNet Layer Shape 계산 (★8, 9주차)

**문제.** AlexNet의 다음 두 layer에서 input → output shape를 계산하라.
- (a) Conv1: input 227×227×3, kernel 11×11, stride 4, padding 0, 96 filters.
- (b) MaxPool1: input 55×55×96, kernel 3×3, stride 2, padding 0.
- (c) Conv2: input 27×27×96, kernel 5×5, stride 1, padding 2, 256 filters.

**풀이.**

**(a) Conv1.** $H_{\text{out}} = (227 - 11)/4 + 1 = 216/4 + 1 = 54 + 1 = 55$. Channels = 96. → **55×55×96.**

**(b) MaxPool1.** $H_{\text{out}} = (55 - 3)/2 + 1 = 52/2 + 1 = 26 + 1 = 27$. Channels 유지. → **27×27×96.**

**(c) Conv2.** $H_{\text{out}} = (27 + 2\cdot 2 - 5)/1 + 1 = 26/1 + 1 = 27$. Channels = 256. → **27×27×256.**

> **풀이 포인트:** Conv2는 padding $p=2$가 있어 input과 output의 spatial size가 같음 = **same convolution.** 조건 $2p = k - s = 5 - 1 = 4$ → $p = 2$ ✓.

**채점 포인트.** ① Output size formula 정확 적용 ② Padding 빼먹지 않기 ③ Channels 변환 (input channels와 output filters 구분).

---

### Q9. Max Pooling Non-linearity 반례 (★7, 9주차)

**문제.** $2 \times 2$ Max Pooling이 linear transformation인지 판정하고, linear가 아니라면 반례를 구성하라. Average Pooling과 비교하라.

**풀이.** §6.K 그대로.

**Average Pooling 비교.** $\text{avg}(x_1) = \frac{1}{4}\sum x_{1,ij}$, 같은 정의의 $x_2$. $\text{avg}(x_1 + x_2) = \frac{1}{4}\sum (x_1 + x_2)_{ij} = \text{avg}(x_1) + \text{avg}(x_2)$ ✓. **Linear** → 매트릭스 표현 가능.

**채점 포인트.** ① 반례 구성 ② Additivity 위반 명시 ③ Average와의 비교 (보너스).

> **풀이 포인트:** Max는 두 영역에서 큰 값을 선택하는 비선형 연산. Average는 단순 합의 평균이라 선형. NN에서 max pooling을 쓰는 이유 = activation 외에도 비선형성을 추가하는 효과 + invariance.

---

### Q10. Inductive Bias 강도 비교 통합 서술 (★10, 7/9주차)

**문제.** Linear Regression, 2-layer NN, CNN, Transformer의 inductive bias 강도를 비교하고 hypothesis space의 부분집합 관계로 설명하라. Bitter Lesson과의 관계도 답하라.

**풀이.**

**Step 1. Linear Regression — 가장 강한 prior.**
- $\mathcal H_{\text{linear}} = \{h : h(x) = w^Tx + b\}$.
- 입력의 선형 결합만 표현 → XOR 같은 비선형 패턴 불가능 (1969 AI Winter).
- Hypothesis space의 차원 = $d + 1$ (입력 차원 + bias) — 매우 좁음.

**Step 2. 2-layer NN — 매우 약한 prior.**
- $h(x) = w^T \sigma(w'^T x)$.
- Universal Approximation Theorem: 충분히 wide하면 임의 연속 함수 근사 가능.
- $\mathcal H_{\text{2-NN}}$이 거의 전체 함수 공간을 커버 → prior 거의 없음.

**Step 3. CNN — 특정 도메인에 강한 prior.**
- 매트릭스를 sparse + weight-sharing으로 제한.
- Locality + Translation Invariance — 두 가지 이미지 특화 prior.
- $\mathcal H_{\text{CNN}} \subsetneq \mathcal H_{\text{FC}}$ — FC NN의 진부분집합. 하지만 이미지에서는 효과적.

**Step 4. Transformer — 매우 약한 prior.**
- Full attention — 모든 token이 모든 token에 영향.
- Markov 같은 strict 의존성 제약 없음.
- 시퀀스/이미지 모두에서 weak prior로 동작 (ViT 사례).

**Step 5. 강 → 약 순서 (시퀀스 데이터 기준).**

$$\mathcal H_{\text{Linear}} \subsetneq \mathcal H_{\text{Markov}} \subsetneq \mathcal H_{\text{CNN}} \subsetneq \mathcal H_{\text{2-NN}} \approx \mathcal H_{\text{Transformer}}$$

**Step 6. Bitter Lesson과의 관계.** Sutton (2019): "70년 AI 연구의 가장 큰 교훈은 general method that leverage computation이 가장 effective였다." 즉:

- **데이터+계산이 부족할 때:** strong prior (Linear, CNN)이 좋음 — 적은 데이터로도 학습 가능.
- **데이터+계산이 충분할 때:** weak prior (Transformer)가 좋음 — prior가 오히려 답을 제한.

LLM/ChatGPT의 성공이 그 증거. CNN → ViT, RNN → Transformer 흐름이 모두 "prior 약화 + 데이터 증가"의 패턴.

**채점 포인트.** ① 4개 모델 모두 hypothesis space 부분집합 관계로 비교 ② 각 prior의 구조적 근거 (linear/sparse+WS/full attention) ③ Bitter Lesson 연결 ($\mathcal H$ 크기와 데이터 크기의 trade-off).

> **풀이 포인트:** 이 문제는 7주차의 prior 흐름 + 9주차의 CNN을 한 문제로 묶음. 시험 직전 강조된 통합 시각이라 출제 가능성 매우 높음. **"Prior 강도 = $\mathcal H$의 좁힘 정도"**라는 등식과 **"Bitter Lesson = 데이터 충분하면 weak prior가 우세"**라는 두 줄을 반드시 포함.

---

## §8. 채점 포인트 / 흔한 실수 / 시험 전날 체크리스트

### 8.1 채점 포인트 5선

1. **가정 명시.** "i.i.d.이므로 $P(D|h) = \prod_i P(x_i|h)$" — 가정을 한 줄 적기. IID 없으면 곱이 성립하지 않음.
2. **로그 변환 이유.** "곱은 미분 어려우므로 log를 취함. log는 단조 증가이므로 argmax 동일." — 한 줄 명시.
3. **미분 부호.** NLL 미분에서 $-$ 부호, $(y-h)\cdot(-1)$의 chain rule 부호 정확. 부호 하나 틀리면 전체 결과 반대.
4. **Boundary 답 인식.** Restricted prior에서 inside가 단조면 boundary가 답 ($1/2 + a$, $1/2 - a$). Interior critical point가 있는지 항상 확인.
5. **단위/차원.** Conv 매트릭스 차원 ($H_{\text{out}} \times H_{\text{in}}$), Hessian 차원 ($d \times d$), Softmax Jacobian 차원 ($C \times C$) 정확.

### 8.2 흔한 실수 5선

1. **Restricted MAP에서 $\theta = 1/2$로 답 (n=k=3에서).** Interior critical point가 없음을 놓침. ✓ 정답: $1/2 + a$.
2. **$\log 0 = -\infty$ 누락.** Outside posterior가 0임을 명시하지 않으면 풀이가 불완전.
3. **Softmax Jacobian에서 $\delta_{ij}$ 누락.** Two-case로 풀어쓰지 않으면 부호 실수. ✓ 정답: $p_i(\delta_{ij} - p_j)$.
4. **Conv → Matrix 변환에서 시프트 오류.** Stride 1이면 한 칸씩 시프트. 두 칸씩 시프트로 그리는 실수 잦음.
5. **AlexNet padding 빼먹기.** Conv2는 $p = 2$가 있어 same convolution. Padding 0으로 계산하면 $H_{\text{out}}$이 잘못 나옴.

### 8.3 시험 전날 체크리스트 (★9+ 각 한 줄)

- [ ] Hypothesis Space Restriction = MAP 통합 시각 한 문장 (prior로 hypothesis space를 좁힘)
- [ ] Bernoulli↔Classification, Gaussian↔Regression 매핑 표 빈칸 없이
- [ ] Gaussian → MSE 도출 (1줄: $\mathcal N$ → $-\log$ → squared)
- [ ] Categorical → Cross-Entropy 도출 ($-\log h(x)_y$)
- [ ] NLL = ERM 등치 도출 with $P_S = \frac{1}{n}\sum \delta(x-x_i)$
- [ ] Linear → Nonlinear basis → 2-layer NN의 prior 흐름 한 단락
- [ ] Universal Approximation 의미와 한계 (width 발산, 학습 어려움)
- [ ] GD/SGD/Momentum/Adam update 식 4개 모두
- [ ] Newton's method = $L$의 2차 근사 minimum 증명
- [ ] Hessian이 NN에서 못 쓰는 이유 ($d^2$ → diagonal 근사 = AdaGrad/RMSProp)
- [ ] Backprop 3 미분 + softmax+CE collapse → $p - e_y$
- [ ] Linear Regression closed form $w^* = (X^TX)^{-1}X^Ty$ 도출
- [ ] 1D Conv → Matrix (kernel = $(1,2)$) 표준기저 구성 + 두 inductive bias 표시
- [ ] Locality + Translation Invariance 정의 + 매트릭스 패턴 대응
- [ ] Output Size Formula 박스 슬라이딩 유도
- [ ] Same Convolution 조건 $2p = k - s$
- [ ] Max Pooling Non-linearity 반례 1개
- [ ] AlexNet Conv1: 227×227×3 → 55×55×96 계산
- [ ] Inductive Bias 강도 (Linear > CNN > 2-NN ≈ Transformer)
- [ ] Bitter Lesson 한 문장 인용 (Sutton 2019 — "general method + computation")
- [ ] Restricted Prior MAP boundary 답 조건 (inside가 단조이면 boundary)

---

## 부록. 출처 및 별점 근거 요약

| 항목 | 주차 | ★ | 명시적 발언 (요약 인용) |
|---|---|---|---|
| Hypothesis Space Restriction = MAP | 6 | 10 | "이걸 이해하면 NN/CNN 다 이해" |
| Scalar→Function Hypothesis | 6 | 10 | "여기서 잘 이해하면 다 이해" |
| Bernoulli↔Class, Gaussian↔Reg 매핑 | 6 | 10 | 표 직접 비교 |
| NLL ↔ ERM | 7 | 10 | "이 흐름을 다 이해해야" |
| Linear→NN prior 흐름 | 7 | 10 | "프라이어가 약해지는/강해지는 방향" |
| Backpropagation | 8 | 10 | "학습의 기본" |
| Chain Rule | 8 | 10 | 모든 미분의 핵심 |
| Adam | 8 | 10 | "거의 default" |
| GD Update | 8 | 10 | "그래디언트 반대 방향" |
| Conv = Sparse + Weight-Sharing | 9 | 10 | "두 개의 inductive bias" |
| Locality + Translation Invariance | 9 | 10 | "두 가지 특징" |
| 이미지 = Linear Trans. Restriction | 9 | 10 | "Linear algebra의 가장 중요한 정리를 그대로 conv에" |

---

**문서 끝.**

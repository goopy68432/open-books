---
title: "딥러닝이론 6주차 분석"
slug: week6
order: 5
---

# 딥러닝이론 6주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-6주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### 중간고사 복습 — Restricted Uniform Prior
- **정의:** $P(\theta)$가 $[1/2 - a, 1/2 + a]$ 구간에서만 nonzero, 외부에서 0인 hard restriction prior.
- **PDF:** Inside: $1/(2a)$ (uniform), outside: 0.
- **맥락:** 중간고사 2번 A번 문제. **PDF 값이 안에서 어떤 상수인지 명시 + 밖에서 0인 점 명시가 채점 포인트.**
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 시험에 나옴, 많이 틀림]

### Log 0 = $-\infty$
- **정의:** Restriction 영역 밖에서 prior=0 → log prior $= -\infty$. 따라서 어떤 likelihood가 좋아도 posterior가 0.
- **함의:** Hard restriction은 가능 영역 밖을 완전 배제 = strong inductive bias.
- **중요도:** ★★★★★★★ (7/10) [명시적: D번 학생 다수 오답 원인]

### Hypothesis Space 시각화
- **정의:**
  - 초록색 영역: 전체 hypothesis space (모든 $\theta$ 가능).
  - 노란색 영역: prior로 좁힌 영역 (e.g., $[0.4, 0.6]$).
  - **Hypothesis space를 좁히는 것 = inductive bias 주는 것.**
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 향후 NN, CNN의 핵심]

### From Scalar Hypothesis ($\theta$) to Function Hypothesis ($h$)
- **정의:** Bernoulli 예제에서 $\theta$가 hypothesis였던 것이, regression/classification에서는 함수 $h:X\to Y$로 확장.
- **시각화:** $h(x)$ 각 $x$마다 스칼라 → 모든 $x$에 대한 $h$의 값을 다 정한 것이 함수.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심 패러다임 전환]

### Bernoulli ↔ Classification, Gaussian ↔ Regression
- **정의:**
  - Bernoulli (2 outcome) → Binary classification.
  - Categorical (C outcome) → Multi-class classification.
  - Gaussian → Regression.
- **로스 함수 매핑:** Bernoulli/Categorical → Cross-Entropy. Gaussian → MSE.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심 통합 시각]

### Gaussian Mean MLE (1D, $\sigma^2=1$ 고정)
- **정의:** Data $y_1,\dots,y_n \sim \mathcal{N}(\mu, 1)$. $\mu^*_{ML} = \frac{1}{n}\sum y_i$ (sample mean).
- **중요도:** ★★★★★★★★ (8/10) [명시적: 칠판 직접 풀이]

### Function Hypothesis: $y_i \sim \mathcal{N}(h(x_i), 1)$
- **정의:** 각 $x_i$마다 Gaussian의 평균이 $h(x_i)$. **Bernoulli/$\theta$가 $\mu/h(x_i)$로 확장.**
- **시각화:** 각 $x_i$ 위에 90도 돌아간 Gaussian이 세로로 서있고, 평균이 $h(x_i)$.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Linear Regression — Hypothesis Space Restriction
- **정의:** 모든 함수가 아니라 $h(x) = ax + b$만 고려. 노란색 영역 = 선형 함수만.
- **풀이:** $a$, $b$로 미분 = 0.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Logistic Regression (Binary Classification)
- **정의:** Bernoulli framework에서 $h(x_i)$ = $\sigma(w^Tx_i)$ = sigmoid function. Output을 $[0,1]$로 squash.
- **이름의 모순:** "Regression"이 아니라 classification. 역사적 명칭.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Categorical Distribution / Softmax for Multi-class
- **정의:** $y_i \in \{1,\dots,C\}$. $h(x_i)$가 $C$차원 vector로 각 클래스 probability를 표현. Softmax로 합=1, 각 성분≥0 보장.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Cross-Entropy Loss
- **정의:** $\ell(h, x_i, y_i) = -\log h(x_i)_{y_i}$. Categorical NLL의 형태.
- **중요도:** ★★★★★★★★★ (9/10) [추론 보충: 7주차 ERM 핵심]

---

## 🔢 2. 수식 풀이 및 증명

### Restricted Uniform Prior PDF
**문제/목표:** $P(\theta)$ 표현을 정확히 쓰기 (inside/outside).
**단계별 풀이:**
1. Outside ($\theta \notin [1/2-a, 1/2+a]$): $P(\theta) = 0$.
2. Inside: 영역 길이 $2a$이므로 $P(\theta) = 1/(2a)$ (PDF는 적분=1).

**중요도:** ★★★★★★★★ (8/10)

### Restricted-Prior MAP — n=k=3 예시
**문제/목표:** 3번 던져 3번 head, restricted uniform prior인 $\theta^*_{MAP}$.
**단계별 풀이:**
1. Log posterior = $3\log\theta + \log(1/(2a))$ inside, $-\infty$ outside.
2. Inside만 고려: $3\log\theta$는 $\theta$에 단조 증가.
3. 영역 내 최대 = upper boundary $\theta = 1/2 + a$.

**결론:** $\theta^*_{MAP} = 1/2 + a$.
**중요도:** ★★★★★★★★★ (9/10) [명시적: 시험 D번]

### 3번 모두 Tail의 경우
**문제/목표:** $n=3, k=0$일 때 같은 prior에서 $\theta^*_{MAP}$.
**단계별 풀이:**
1. Log posterior = $3\log(1-\theta) + C$ inside.
2. $1-\theta$에 단조 증가 = $\theta$에 단조 감소.
3. Inside 최소 $\theta$ = $1/2 - a$.

**결론:** $\theta^*_{MAP} = 1/2 - a$.
**중요도:** ★★★★★★★ (7/10)

### Gaussian Mean MLE
**문제/목표:** $y_i\sim\mathcal{N}(\mu, 1)$ IID로 부터 $\mu^*$ 도출.
**단계별 풀이:**
1. NLL = $\sum_i \frac{(y_i - \mu)^2}{2} + C$.
2. $\mu$로 미분: $\sum_i (y_i - \mu) \cdot (-1) = 0$.
3. $-\sum y_i + n\mu = 0 \Rightarrow \mu^* = \frac{1}{n}\sum y_i$.

**결론:** Sample mean이 Gaussian의 mean MLE.
**중요도:** ★★★★★★★★★ (9/10) [명시적]

### Linear Regression for $h(x) = ax$ (1D)
**문제/목표:** $y_i \sim \mathcal{N}(h(x_i), 1)$, $h(x) = ax$일 때 $a^*$.
**단계별 풀이:**
1. NLL = $\sum_i (y_i - ax_i)^2 / 2 + C$.
2. $a$로 미분: $\sum_i (y_i - ax_i)(-x_i) = 0$.
3. $a^* = \frac{\sum y_i x_i}{\sum x_i^2}$.

**결론:** Origin을 지나는 best line의 기울기.
**중요도:** ★★★★★★★★★ (9/10) [명시적]

### Linear Regression with bias $h(x) = ax + b$
**문제/목표:** $a, b$로 동시에 미분 = 0.
**단계별 풀이:**
1. NLL = $\sum_i (y_i - ax_i - b)^2 / 2 + C$.
2. $a$ 미분: $\sum_i (y_i - ax_i - b)(-x_i) = 0$.
3. $b$ 미분: $\sum_i (y_i - ax_i - b)(-1) = 0$.
4. 두 식을 연립하여 $a^*, b^*$ 도출.

**결론:** Coupled equations이므로 단순한 closed form 아닌 연립.
**중요도:** ★★★★★★★★ (8/10)

### Multi-class Negative Log-Likelihood
**문제/목표:** Categorical $h(x_i)$에서 $-\sum_i \log h(x_i)_{y_i}$ 도출.
**단계별 풀이:**
1. $P(y_i|x_i, h) = h(x_i)_{y_i}$ (categorical).
2. $-\log P(D|h) = -\sum_i \log h(x_i)_{y_i}$.
3. = Cross-Entropy loss.

**결론:** Categorical NLL = Cross-Entropy.
**중요도:** ★★★★★★★★★ (9/10) [명시적]

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| Hypothesis Space Restriction = MAP | 10 | 통합 시각 | "이걸 이해하면 NN/CNN 다 이해" |
| Scalar→Function Hypothesis 확장 | 10 | 패러다임 | "여기서 잘 이해하면 다 이해" |
| Bernoulli→Classification, Gaussian→Regression | 10 | 매핑 | 표 직접 비교 |
| Restricted Uniform Prior MAP | 9 | 정리 | 시험 출제 |
| Linear Regression as Linear-Restricted MAP | 9 | 통합 | "이게 미니어 리그레이션" |
| Logistic Regression | 8 | 정의 | classification 핵심 |
| Cross-Entropy as Categorical NLL | 9 | 통합 | 7주차 ERM 진입 |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. Restricted Uniform Prior의 PDF
**문제:**
> "이 prior가 $[1/2-a, 1/2+a]$ 안에서 uniform이면 $P(\theta)$를 어떻게 쓸까요? 4분 드릴게요."

**트리거 발언:** "한 4분 드릴 테니까 계산 한번 해보시기 바랍니다."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. PDF는 영역 안과 밖을 case로 나눠 작성:
$P(\theta) = \begin{cases} 1/(2a) & \text{if } \theta \in [1/2-a, 1/2+a] \\ 0 & \text{otherwise}\end{cases}$.
2. 적분 = $1/(2a) \cdot 2a = 1$. ✓

**정답:** Inside: $1/(2a)$, outside: 0.

**해설:** PDF를 case로 나눠 명확히 표기하는 것이 중요. Bound 외에서 0이라는 점이 향후 MAP 계산에서 결정적.

</details>

### Q2. n=k=3 일 때 Restricted MAP
**문제:**
> "이 영역 내에서 언제 $\theta$가 가장 좋은 답일까요?"

**트리거 발언:** "이 영역 내에서 언제 제일 큰지를 찾는 거예요."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Inside log posterior = $3\log\theta$ (상수 무시).
2. $\theta$에 monotonic increasing.
3. Bound 내 최대값 = upper bound = $1/2 + a$.
4. Outside는 $-\infty$이므로 배제.

**정답:** $\theta^*_{MAP} = 1/2 + a$.

**해설:** MLE라면 $\theta=1$이지만, prior의 hard restriction으로 가능 영역의 boundary가 답. "데이터를 prior 영역으로 끌어당긴다."

</details>

### Q3. Gaussian Mean MLE
**문제:**
> "$y_i\sim\mathcal{N}(\mu, 1)$ IID로부터 가장 좋은 $\mu$는?"

**트리거 발언:** 칠판 풀이 진행 중 학생들에게 "어떻게 변화할 수 있죠?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. NLL을 $\mu$로 미분.
2. $-\sum_i (y_i - \mu) = 0$.
3. $\mu^* = \frac{1}{n}\sum_i y_i$.

**정답:** Sample mean.

**해설:** Gaussian의 평균을 추정하는 것은 데이터의 산술평균이 답이라는 직관과 일치.

</details>

### Q4. Linear Regression for $h(x)=ax$
**문제:**
> "데이터 7개를 가장 잘 설명하는 origin을 지나는 line의 기울기 $a$는?"

**트리거 발언:** 풀이 진행 중

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. NLL = $\sum (y_i - ax_i)^2/2 + C$.
2. $\partial/\partial a = -\sum x_i(y_i - ax_i) = 0$.
3. $a\sum x_i^2 = \sum x_i y_i$.
4. $a^* = \sum x_i y_i / \sum x_i^2$.

**정답:** $a^* = \sum x_i y_i / \sum x_i^2$.

**해설:** Inner product 형태로 표현되는 best slope. 1차원 least squares의 closed form.

</details>

---

## 📎 5. 기타 참고사항

- 행정: 중간고사 점수 곧 공지. 진도 조절 — "생각보다 어려우셨던 것 같아 천천히 진행."
- "강의 때 안 배운 부분이 시험에 나왔다" 학생 의견에 "강의 때 한 것도 다 풀지는 못하신 것 같아서" 답변.
- 시험 1번 A~C는 잘 풀고 D번을 많이 틀림 → restricted prior MAP를 다시 강의.
- $\theta=1/2$로 답하는 학생들에게 "왜 1/2가 아닌지" 질문 → boundary($1/2 + a$)가 답이라는 점 강조.
- Function hypothesis 시각화 — 90도 돌린 Gaussian이 각 $x_i$ 위에 서 있는 그림 매우 중요.
- "함수라고 막 어렵게 생각하지 마세요. $h(x)$ = scalar values를 모든 $x$에 대해 정한 것."
- Logistic regression 풀이는 sigmoid 도입 단계에서 미분이 복잡해서 풀이는 안 함, 개념만.
- 다음 시간 예고: NN, hypothesis space restriction 측면에서 다룸.
- Optimization은 그 다음 주.
- 학생 질문 처리: "Q. $b$를 0으로 놓고 푼 후 $b$도 함께 풀려면?" → "$y$ 절편이 있는 함수도 가능, 노란색 영역이 더 넓어진다."
- "B(intercept)에 대해서도 미분해서 풀 수 있다. A 계산하는 거 정도는 충분히 하실 수 있으면 괜찮을 것 같습니다."


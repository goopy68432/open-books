---
title: "딥러닝이론 7주차 분석"
slug: week7
order: 6
---

# 딥러닝이론 7주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-7주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### NLL ↔ Empirical Risk Minimization (ERM) 동치
- **정의:** $-\log P(D|h) = -\sum_i \log P(x_i|h)$ (IID)이고, 이를 평균낸 것이 empirical risk. 일반적인 loss function $\ell$로 추상화.
- **수식:** $\text{NLL}(h) = -\sum_i \log P(x_i|h)$ vs $L_S(h) = E_{x\sim P_S}[\ell(x, h)] = \frac{1}{n}\sum_i \ell(x_i, h)$.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심 통합 시각]

### Empirical Distribution과 Expectation
- **정의:** $P_S(x) = \frac{1}{n}\sum_i \delta(x-x_i)$. 이 분포에서 expectation = 평균.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Loss = $-\log P$의 추상화
- **정의:** Gaussian이면 squared loss, Categorical이면 cross-entropy. Loss의 선택은 likelihood model이 결정.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Function Space → Parameter Space
- **정의:** Hypothesis $h\in F$ (모든 함수)에서 $\theta$로 parametrize한 함수 $h_\theta$만 고려. Optimization은 vector $\theta$ 공간에서 진행.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Prior로서의 Function Restriction
- **정의:** Linear function만 고려 = strong prior. Linear function 공간이 전체 함수 공간의 작은 부분집합.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Perceptron (1958)
- **정의:** $f(x) = \mathbb{1}[w^T x \ge 0]$. Binary classification, decision boundary가 hyperplane (linear).
- **역사:** 1958 Rosenblatt. AI/ML의 시작과 함께. Backpropagation도 그가 일찍 제안.
- **중요도:** ★★★★★★ (6/10)

### XOR Problem
- **정의:** XOR (exclusive-or) 4점 $(0,0), (0,1), (1,0), (1,1)$의 클래스. Linear decision boundary로 분리 불가. 1969 Minsky-Papert 책에서 문제 제기 → AI Winter.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Logistic Regression / Linear Regression — 모두 Linear Models
- **정의:** Output이 $w^Tx$의 함수. Linear model의 강한 inductive bias.
- **중요도:** ★★★★★★★ (7/10)

### Nonlinear Basis Function
- **정의:** $f(x) = w^T \phi(x)$. $\phi$를 manually design한 후 그 위에서 linear. e.g. $\phi(x) = (x_1, x_1^2, x_2, x_2^2, x_1x_2, 1)$.
- **단점:** $\phi$를 사람이 manually 정해야 함 = high prior.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Parametrized Feature Extractor
- **정의:** $f(x) = w^T \phi(x; w')$. $\phi$도 학습. Prior 더 약해짐.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### 2-Layer Neural Network = Universal Approximator
- **정의:** $f(x) = w^T \sigma(w'^T x)$. 충분히 wide하면 임의의 함수를 근사 가능.
- **함의:** Hypothesis space가 거의 전체 함수 공간을 커버.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Why Deep Networks? (2-layer로 충분한데)
- **이유:** (1) Width가 굉장히 커야 함. (2) $w, w'$를 잘 찾기가 어려움.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Activation Functions
- **정의:** Sigmoid (logistic), Softmax, ReLU 등. Piecewise-linear여도 전체적으로 nonlinear면 ok.
- **ReLU:** $\max(0, x)$. 음수 → 0, 양수 → identity.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### MLP (Multi-Layer Perceptron) / Deep Network
- **정의:** 여러 층의 linear + nonlinearity 반복. $x_{k+1} = \sigma(w_k^T x_k)$.
- **중요도:** ★★★★★★★★★ (9/10)

### AlexNet (2012, 8 layers) / ResNet (2015, 152 layers)
- **정의:** Deep learning revolution의 시작 (AlexNet) / human performance(~5%) 돌파 (ResNet).
- **중요도:** ★★★★★★ (6/10)

### Bitter Lesson (Sutton 2019)
- **정의:** "70년 AI 연구의 가장 큰 교훈: general method that leverage computation이 가장 effective였다." 사람이 prior knowledge 넣는 것보다 단순+scale이 좋다.
- **함의:** Linear→NN→Deep NN→Transformer 흐름 설명.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Next-Token Prediction (LLM Pretraining)
- **정의:** 단순한 objective: 앞 token들로 다음 token 예측. Low prior, 데이터 양으로 scaling.
- **중요도:** ★★★★★★★ (7/10) [명시적]

### Markov Property vs Transformer
- **정의:** Markov: 직전 token만 의존 (strong prior). Transformer: 전체에 의존 (low prior).
- **중요도:** ★★★★★★★ (7/10) [명시적]

### Set / Permutation Invariance
- **정의:** Set 데이터를 다룰 때 순서를 바꿔도 출력이 같아야 함. Inductive bias의 한 예시.
- **응용:** 3D Point Cloud, AlphaFold (분자), permutation invariance.
- **중요도:** ★★★★★★ (6/10)

---

## 🔢 2. 수식 풀이 및 증명

### NLL ↔ ERM 연결
**문제/목표:** $\text{NLL}(h) = \sum_i [-\log P(x_i|h)]$를 ERM 형태로 표현.
**단계별 풀이:**
1. NLL = $-\sum_i \log P(x_i|h)$.
2. $\frac{1}{n}$ 곱해도 argmin은 같음.
3. $\ell(x_i, h) = -\log P(x_i|h)$로 정의.
4. $L_S(h) = \frac{1}{n}\sum_i \ell(x_i, h) = E_{x\sim P_S}[\ell(x, h)]$.

**결론:** Loss를 $-\log P$로 정의하면 NLL = ERM.
**중요도:** ★★★★★★★★★ (9/10) [명시적]

### Gaussian → Squared Loss
**유도:** 4주차 동일 — $-\log\mathcal{N}(y|h(x), \sigma^2) = (y-h(x))^2/(2\sigma^2) + C$. Squared loss 도출.

### Categorical → Cross-Entropy
**유도:** $-\log P(y_i|x_i, h) = -\log h(x_i)_{y_i}$. Cross-Entropy.

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| NLL = ERM 연결 | 10 | 통합 시각 | "이 흐름을 다 이해해야" |
| Linear→Nonlinear basis→Parametrized→2-layer NN→Deep | 10 | 진화 흐름 | "프라이어가 약해지는 흐름" |
| Universal Approximation (2-layer로 충분) | 9 | 정리 | 이론적 toolkit |
| Bitter Lesson | 9 | 철학 | "씁쓸한 교훈" |
| ReLU/Sigmoid/Softmax | 8 | 도구 | 실용 |
| Function Space→Parameter Space | 9 | 패러다임 | "벡터 공간에서 찾는다" |
| Inductive Bias의 강도 비교 | 10 | 개념 | "약해지고 강해지고를 계속 따라가야" |
| XOR Problem | 8 | 사례 | "Linear로 풀 수 없음 — AI Winter" |
| Markov vs Transformer prior | 7 | 사례 | LLM context |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. Linear→Nonlinear basis로 갈 때 prior는 강해지는가 약해지는가?
**문제:**
> "Linear model에서 nonlinear basis를 도입했을 때 prior가 강해지나요 약해지나요?"

**트리거 발언:** "프라이어가 강해지는 방향으로 가는지 약해지는 방향으로 가는지 생각을 계속 해주셔야 합니다."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Linear model = 선형 함수만 고려 = 매우 좁은 hypothesis space = strong prior.
2. Nonlinear basis로 확장 = 더 다양한 함수 가능 = hypothesis space 커짐 = prior 약해짐.

**정답:** 약해진다.

**해설:** 그러나 nonlinear basis도 사람이 $\phi$를 매뉴얼하게 정해야 하므로 여전히 inductive bias가 남아있음. 다음 단계 (parametrized basis)에서 더 약해짐.

</details>

### Q2. Parametrized Feature Extractor로 갈 때 prior는?
**문제:**
> "1→2 (linear → nonlinear basis)에서 약해졌고, 2→3 (nonlinear basis → parametrized)으로 가면?"

**트리거 발언:** "강해진다, 약해진다? 계속 약해지고 있어요"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $\phi$가 학습되므로 사람이 정해야 할 inductive bias 감소.
2. Hypothesis space 더 커짐.

**정답:** 약해진다.

**해설:** 1→2→3 (Linear→Nonlinear basis→2-layer NN) 모두 prior가 약해지는 방향. 깊이 더 추가하면 더 약해짐.

</details>

### Q3. CNN, Transformer의 Prior 강도 비교
**문제:**
> "CNN과 Transformer 중 어느 쪽이 prior가 더 강한가?"

**트리거 발언:** "Inductive bias 어떨까요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. CNN: 이미지 특화. Translation invariance, locality 등 강한 prior.
2. Transformer: 전체 토큰이 영향 가능. Markov 같은 strict 제약 없음.

**정답:** CNN이 더 강함, Transformer가 더 약함.

**해설:** 데이터가 많아지면서 weak prior (Transformer)가 더 좋아진 것. Bitter lesson과 부합.

</details>

### Q4. Markov Language Model vs Transformer
**문제:**
> "'I ate an apple'과 'I built an apple' 문장에서 'an' 다음 'apple'이 나올 확률을 모델링한다고 할 때, Markov는 어디에 강한 prior가 있는가?"

**트리거 발언:** "어떤게 prior가 많이 들어가 있을까요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Markov: 직전 token('an')만 보고 결정. 'I ate'와 'I built'를 구분 못 함.
2. 'an 뒤에 apple'이라는 가정 자체가 강한 제약 = 강한 prior.

**정답:** Markov.

**해설:** Transformer는 모든 이전 token을 참조해 'ate' vs 'built'를 구분하여 다른 확률을 부여 가능. Strong vs weak prior의 대비.

</details>

---

## 📎 5. 기타 참고사항

- 시작: 옵티미제이션은 다음 시간 진도. 이미지(CNN)는 그 다음 시간. Hypothesis space restriction 흐름 정리.
- "Hypothesis space를 줄인다 = inductive bias / prior knowledge"라는 framing 일관 유지.
- Generalization 토픽은 너무 어려워서 스킵. CNN 다룰 때 다시 언급할 예정.
- ZFNet (2013), GoogLeNet (2014), ResNet (2015) 진화 언급.
- "Why deep > shallow" 답변 미상 — "여전히 모릅니다."
- Sutton의 "The Bitter Lesson" (2019) 글 첫 문장만이라도 읽으라고 강력 권장.
- ChatGPT의 성공 = scaling. GPU, NVIDIA 주가와 연결.
- Set으로서 꽃병 point cloud 예시 — 100개 3D 점, 순서가 바뀌어도 같은 출력 (permutation invariance).
- AlphaFold(분자, 단백질 fold), 컬래보레이션 가능한 신약 개발.
- Linear function의 parameter 개수: $R^n \to R^m$의 linear function은 $m\times n + m$ 개 parameter (matrix + bias).
- Backpropagation은 옵티미제이션 다음 시간 다룸.
- Optimizer 선택의 중요성: GD, Adam, AdamW (LLM 학습), Muon (최근).
- Generalization과 optimization은 "다른 측면"임을 강조.
- 마지막 정리: "Linear model 시작 → XOR로 막힘 → Nonlinear basis(매뉴얼) → Parametrized(자동) → 2-layer NN → Deep NN. 매 단계마다 prior가 약해지는 방향."


---
title: "딥러닝이론 강의 분석 — 인덱스"
slug: index
order: 0
---

# 딥러닝이론 강의 분석 — 인덱스

## 주차별 링크

- [1주차](01_1주차_분석.md) — Introduction, Deduction vs Induction, Linear Algebra 기초, Function/Vector. **최고 ★10:** Maximum Likelihood + Prior 두 축을 이해하는 것 = 수업 전체의 목표
- [2주차](02_2주차_분석.md) — Linear Algebra (Rank-Nullity, Eigenvalue, SVD), Vector Calculus (Jacobian, Newton), Softmax 미분. **최고 ★9:** Vector→Vector 미분 (Jacobian), Linear Approximation의 중심성, Softmax 미분 가능성
- [3주차](03_3주차_분석.md) — Bayesian Probability, Bayes Theorem, Prior/Posterior/Likelihood, MLE, MAP. **최고 ★10:** Bayesian probability(degree of belief), Posterior, Bayes Theorem, MLE = Uniform-prior MAP, Gaussian의 exp/minus/square
- [4주차](04_4주차_분석.md) — MLE/MAP/Strong-MAP 비교, Inductive Bias, KL Divergence, Linear Regression의 Gaussian 기원, NLL↔MSE. **최고 ★10:** MLE/MAP/Strong-MAP 통합 시각, NLL↔MSE 유도, Inductive Bias
- [6주차](05_6주차_분석.md) — Restricted Uniform Prior, Scalar→Function Hypothesis, Bernoulli↔Classification/Gaussian↔Regression, Linear Regression. **최고 ★10:** Hypothesis Space Restriction = MAP, Scalar→Function 패러다임, Distribution↔Loss 매핑
- [7주차](06_7주차_분석.md) — NLL↔ERM, Linear→Nonlinear basis→Parametrized→2-layer NN→Deep, Universal Approximation, Bitter Lesson, CNN/Transformer prior 비교. **최고 ★10:** NLL=ERM 통합, Inductive Bias 강도 비교
- [8주차](07_8주차_분석.md) — Newton's Method 재방문, Gradient Descent, SGD, Momentum, Adam, Backpropagation. **최고 ★10:** Backprop, Chain Rule, Adam, GD Update Rule
- [9주차](08_9주차_분석.md) — CNN, Convolution = Sparse + Weight-Sharing Matrix, Locality + Translation Invariance, AlexNet 분석, ZFNet 시각화, Pooling. **최고 ★10:** Conv = 특정 Linear Transformation, Locality+Translation Invariance, Image processing = LT Restriction

---

## 전체 ★8 이상 핵심 개념 TOP 30

| 주차 | 개념 | ★ | 유형 |
|---|---|---|---|
| 1 | Maximum Likelihood + Prior 두 축 (수업 메인) | 10 | 통합 시각 |
| 3 | Bayesian Probability (degree of belief) | 10 | 개념 |
| 3 | Posterior $P(H\mid E)$ | 10 | 개념 |
| 3 | Bayes Theorem | 10 | 정리 |
| 3 | MLE = Uniform-prior MAP | 10 | 통합 시각 |
| 3 | Gaussian의 exp/minus/square | 10 | 성질 |
| 3 | Belief Update = Learning | 10 | 통합 |
| 4 | MLE↔MAP↔Strong-MAP 비교 | 10 | 통합 시각 |
| 4 | NLL↔MSE 유도 (Gaussian) | 10 | 정리 |
| 4 | Inductive Bias / Hypothesis Space | 10 | 개념 |
| 6 | Hypothesis Space Restriction = MAP | 10 | 통합 시각 |
| 6 | Scalar→Function Hypothesis 확장 | 10 | 패러다임 |
| 6 | Bernoulli↔Classification, Gaussian↔Regression | 10 | 매핑 |
| 7 | NLL = ERM 연결 | 10 | 통합 시각 |
| 7 | Linear→…→Deep NN의 Prior 약화 흐름 | 10 | 진화 |
| 7 | Inductive Bias 강도 비교 | 10 | 개념 |
| 8 | Backpropagation | 10 | 알고리즘 |
| 8 | Chain Rule | 10 | 계산 도구 |
| 8 | Adam Optimizer | 10 | 알고리즘 |
| 8 | GD Update Rule | 10 | 알고리즘 |
| 9 | Conv = Sparse + Weight-Sharing Matrix | 10 | 핵심 정리 |
| 9 | Locality + Translation Invariance (CNN IB) | 10 | 핵심 IB |
| 9 | Image Processing = LT Restriction | 10 | 통합 시각 |
| 1 | Linear Transformation = Matrix | 9 | 정리 |
| 1 | Deduction vs Induction 구분 | 9 | 개념 |
| 2 | Vector→Vector 미분 (Jacobian) | 9 | 정의 |
| 2 | Linear Approximation의 중심성 | 9 | 개념 |
| 2 | Softmax 미분 | 9 | 정리 |
| 3 | Hypothesis $h$ / $\theta$ | 9 | 개념 |
| 3 | Prior, Likelihood (each color-coded) | 9 | 개념 |
| 1 | Inner Product (대학원 면접 단골) | 8 | 정의 |
| 2 | Eigenvalue 큰 것 = 중요 | 8 | 개념 |
| 2 | Rank-Nullity Theorem | 8 | 정리 |
| 3 | IID Assumption | 8 | 가정 |
| 3 | Bernoulli Distribution | 8 | 정의 |
| 4 | KL Divergence | 8 | 정의 |
| 4 | Empirical Distribution | 8 | 정의 |
| 4 | CLT (Central Limit Theorem) | 9 | 정리 |
| 4 | Cross-Entropy / Entropy | 8 | 정의 |
| 6 | Linear Regression as Linear-Restricted MAP | 9 | 통합 |
| 6 | Logistic Regression | 8 | 정의 |
| 6 | Gaussian Mean MLE | 9 | 정리 |
| 6 | Cross-Entropy as Categorical NLL | 9 | 통합 |
| 7 | Universal Approximation (2-layer) | 9 | 정리 |
| 7 | Bitter Lesson | 9 | 철학 |
| 7 | Function Space → Parameter Space | 9 | 패러다임 |
| 7 | XOR Problem | 8 | 사례 |
| 7 | ReLU/Sigmoid/Softmax (Activation) | 8 | 도구 |
| 7 | Nonlinear Basis Function | 8 | 개념 |
| 7 | Parametrized Feature Extractor | 8 | 개념 |
| 7 | Multi-Layer Perceptron | 9 | 정의 |
| 8 | Newton = 2차 근사 minimum | 9 | 통찰 |
| 8 | SGD가 GD보다 좋음 (generalization) | 9 | 사실 |
| 8 | Learning Rate 큰 게 좋음 | 8 | 사실 |
| 8 | Stochastic Gradient Descent | 9 | 알고리즘 |
| 8 | Momentum (Heavy-ball / NAG) | 8 | 알고리즘 |
| 8 | Hessian 계산 불가 (d² 차원) | 8 | 한계 |
| 8 | AdaGrad / RMSProp | 8 | 알고리즘 |
| 9 | Inner Product = Similarity | 9 | 개념 |
| 9 | Output Size Formula | 9 | 정리 |
| 9 | Conv → Matrix (표준기저) | 9 | 방법 |
| 9 | Convolution Operation | 10 | 정의 |
| 9 | Bitter Lesson 재조명 | 8 | 철학 |
| 9 | Same/Valid Conv, Stride/Padding | 8 | 정의 |
| 9 | Feature Detector | 9 | 개념 |
| 9 | Pooling (Max/Average) | 8 | 정의 |
| 9 | ZFNet Feature Visualization | 8 | 사례 |

(60+ 항목, ★8 이상 모두 포함)

---

## 전체 QUIZ 인덱스

| # | 주차 | 문제 요약 |
|---|---|---|
| 1 | 1 | Linearity 확인 — Matrix-Vector 곱이 linear인가 |
| 2 | 1 | Linear Transformation에 대응되는 Matrix 찾기 |
| 3 | 1 | $n^2-n+41$ 소수 판정 (인덕션 함정) |
| 4 | 2 | Rank-Nullity for $A=(1,1)$ |
| 5 | 2 | $A=\text{diag}(4,3,2,1,0)$의 image, kernel, eigen 분석 |
| 6 | 2 | Softmax 미분 (Jacobian) |
| 7 | 3 | Coin's $\theta$ 분포 직접 그리기 (prior 시각화) |
| 8 | 3 | 3번 던져 3번 head 본 후 분포 갱신 (posterior 시각화) |
| 9 | 3 | Bayes Theorem 증명 |
| 10 | 3 | MLE for Bernoulli — $\theta^*_{ML}$ 도출 |
| 11 | 3 | MAP with $P(\theta)\propto\theta(1-\theta)$ |
| 12 | 3 | MAP with $P(\theta)\propto\theta^M(1-\theta)^M$, $M\to\infty$ |
| 13 | 4 | General-$M$ MAP 풀기 |
| 14 | 4 | MLE/Mid-MAP/Strong-MAP 비교 직관 |
| 15 | 4 | KL divergence 직관 (Gaussian) |
| 16 | 6 | Restricted Uniform Prior PDF |
| 17 | 6 | n=k=3 일 때 Restricted MAP |
| 18 | 6 | Gaussian Mean MLE |
| 19 | 6 | Linear Regression for $h(x)=ax$ |
| 20 | 7 | Linear→Nonlinear basis로 갈 때 prior 변화 |
| 21 | 7 | Parametrized Feature Extractor의 prior |
| 22 | 7 | CNN vs Transformer prior 강도 |
| 23 | 7 | Markov vs Transformer LM의 prior |
| 24 | 8 | Empirical Distribution과 Empirical Risk 표현 |
| 25 | 8 | Newton's Method가 $L$에 대해 무엇을 하는가 |
| 26 | 8 | Backprop의 3가지 미분 계산 (softmax + 등) |
| 27 | 8 | Linear Regression Closed Form |
| 28 | 9 | 1D Convolution 직접 계산 (4 cases, linearity 검증) |
| 29 | 9 | Conv → Matrix (1D, 7→6) |
| 30 | 9 | 2D Conv → Matrix (9→4) |
| 31 | 9 | Max Pooling은 Linear인가 (반례) |
| 32 | 9 | AlexNet Layer Shape 변환 |

총 32개 QUIZ.

---

## 시험 출제 가능성 매트릭스

### 기말고사 우선순위 (6-9주차 + 1-4주차 변형)

#### 매우 높음 (★★★★★)
- **9주차:** Convolution → Matrix 변환 (sparse + weight sharing 보이기), Output size formula 유도, Conv가 linear transformation임 증명
- **8주차:** Backpropagation 그래프에서 chain rule로 미분 계산, Linear Regression closed form, Newton's method의 의미 ($L$의 2차 근사)
- **7주차:** Inductive bias 강도를 묻는 개념 문제 (Linear↔NN↔CNN↔Transformer), NLL↔ERM 동치 표현, Universal approximation의 의미
- **6주차:** Restricted Prior MAP — boundary value $\theta^* = 1/2 \pm a$, Linear regression as restricted hypothesis space, Gaussian mean MLE

#### 높음 (★★★★)
- **6-9주차 통합:** Hypothesis space restriction = MAP의 prior 강도, Bernoulli↔Classification, Gaussian↔Regression 매핑, Cross-Entropy 도출
- **3주차 변형:** General-$M$ MAP 도출 ($\theta^* = (k+M)/(n+2M)$), Bayes Theorem 증명, MLE for Bernoulli ($\theta^*=k/n$) — **중간고사 출제됐던 것의 변형**
- **4주차 변형:** NLL↔MSE 유도 (Gaussian likelihood + log + minus → square), KL divergence

#### 중간 (★★★)
- **2주차 변형:** Vector→Vector Jacobian, Softmax 미분, Newton's method
- **1주차 변형:** Linearity 검증 (additivity, homogeneity), Linear Transformation→Matrix 구성

#### 추가 출제 가능
- AlexNet 등 구체적 architecture의 차원 계산
- ReLU/Sigmoid 미분
- Convex (Linear regression) vs Non-convex (NN) 차이
- Adam = Momentum + RMSProp 결합 의미
- Permutation invariance가 inductive bias 예시인 이유

### 시험 답안 작성 원칙 (강의 명시)
1. **답만 적으면 0점.** 답을 얻는 과정 평가가 중심.
2. **수식만 쭉 적으면 의미 없음.** 논리 과정을 서술해야 함 — IID 어디서 썼는지, 왜 미분=0을 풀었는지 등.
3. 중간고사 이전 범위에서도 **저번 시험에 학생 다수가 못 푼 것의 변형**이 약 1/3 출제 예상.
4. 객관식 X, 주관식 — 영어 출제, 번역 가능.
5. "Explain how to obtain..." 형태로 출제됨 — 그냥 obtain하라는 게 아니라 과정 설명 요구.

---

## 색상 코딩 약속 (3주차에서 도입)
- **파란색** = Prior $P(H)$ (데이터 보기 전의 믿음)
- **빨간색** = Posterior $P(H|E)$ (데이터 본 후의 믿음, 우리가 진짜 보고 싶은 것)
- **초록색** = Likelihood $P(E|H)$ (Hypothesis가 참일 때 데이터 확률, $H$의 함수)

이 색상 약속이 1~9주차 전반의 framework.

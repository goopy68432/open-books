---
title: "📄 [V3] Deep Learning 중간고사 실전 모의시험 (기출 100% 반영판)"
slug: 00-mock-exam-questions
order: 1
---

# 📄 [V3] Deep Learning 중간고사 실전 모의시험 (기출 100% 반영판)

**시험 응시 가이드 (교수님 강조사항 반영)**
*   단순 답안(결과 수식)만 작성 시 0점 처리됩니다. **[정의 $\rightarrow$ 증명 $\rightarrow$ 응용]** 의 논리적 전개(Logical Chain)를 반드시 포함하세요.
*   수식 전개 시 `IID 가정`, `Log 변환의 이유`, `상수 제거의 이유` 등 시스템적/논리적 근거를 서술해야 부분 점수를 잃지 않습니다.
*   시간 배분에 주의하세요. (큰 문제 5개 + 소문항 구성)

---

## Question 1. 확률분포와 기댓값 (Probability Distributions)
**(a)** 연속확률분포인 Uniform Distribution $U(a, b)$ 의 확률밀도함수(pdf)를 정의하고, 기댓값 $E[X]$ 와 분산 $V[X]$ 를 적분을 통해 유도하시오.
**(b)** 확률변수 $Z$ 가 표준 정규 분포 $Z \sim \mathcal{N}(0, 1)$ 를 따를 때, 홀수차 모멘트 $E[Z^{2n-1}]$ 와 짝수차 모멘트 $E[Z^{2n}]$ 를 증명하시오.
**(c)** 이산확률분포인 Poisson Distribution $Pois(\lambda)$ 의 기댓값 $E[X]$ 와 분산 $V[X]$ 를 증명하시오. (단, 분산 유도 시 $E[X(X-1)]$ 트릭을 명시할 것)
**(d)** Uniform, Normal, Poisson 세 분포의 '평균'이 시스템 설계(예: 서버 부하 패턴 예측) 관점에서 각각 어떤 의미의 차이를 가지는지 비교 서술하시오.

## Question 2. 확률적 추정 (MLE, MAP, Bayesian)
**(a)** 데이터 $D = \{x_1, x_2, \dots, x_n\}$ 이 균등분포 $U(0, \theta)$ 에서 IID로 추출되었을 때, 파라미터 $\theta$ 에 대한 MLE(최대우도추정치)를 구하고, 이 문제에서 일반적인 미분 방식($\frac{\partial}{\partial \theta}=0$)을 사용할 수 없는 이유(Boundary Condition)를 설명하시오.
**(b)** 데이터가 베르누이 분포를 따르고, 총 $n$번의 IID 시행 중 $k$번의 성공을 관측했다. 사전확률(Prior)이 $P(\theta) \propto \theta^m (1-\theta)^m$ 일 때, MAP(최대사후확률추정치) $\hat{\theta}_{MAP}$ 를 도출하기 위한 수식을 목적함수 설정부터 미분까지 전개하시오.
**(c)** 데이터가 정규분포 $\mathcal{N}(\mu, \sigma^2)$ 를 따를 때, NLL(Negative Log-Likelihood)을 정의하고, 미분을 통해 평균 $\mu$ 와 분산 $\sigma^2$ 에 대한 MLE를 각각 유도하시오.
**(d)** 베이지안 관점에서 데이터가 순차적으로 들어올 때, 시점 $t$ 에서의 Posterior가 시점 $t+1$ 에서의 Prior로 어떻게 작용하는지(Prior Update) 수식과 함께 온라인 학습(Online Learning) 관점으로 설명하시오.

## Question 3. 손실함수와 KL Divergence (Loss & Information Theory)
**(a)** 젠센 부등식(Jensen's Inequality)을 활용하여 KL Divergence 가 항상 0 이상($KL(P||Q) \ge 0$)임을 증명하시오.
**(b)** 두 정규분포 $P \sim \mathcal{N}(\mu_1, \sigma^2)$ 와 $Q \sim \mathcal{N}(\mu_2, \sigma^2)$ 간의 KL Divergence $KL(P||Q)$ 를 계산하는 과정을 수식으로 증명하시오. (분산은 동일함)
**(c)** **[전체 관계 증명]** "에러 오차가 가우시안 분포를 따른다"는 가정에서 출발하여 NLL 최소화 로직이 MSE 최소화 공식으로 도출되는 과정을 증명하고, 이를 바탕으로 "KL 최소화 $\leftrightarrow$ NLL 최소화" 관계를 전체적으로 설명하시오.
**(d)** 이산형 확률 분포 $P(X)=[0.4, 0.6]$, $Q(X)=[0.5, 0.5]$ 가 주어졌을 때, 일반 KL Divergence 값을 계산하시오.

## Question 4. 행렬 미분과 역전파 (Matrix Calculus & Backprop)
**(a)** 입력 데이터 $x$ 에 대해 선형 변환 $z = Ax+b$ 와 활성화 함수 $p = \sigma(z)$, 손실 함수 $L = -\log(p)$ 가 주어졌다 (정답 $y=1$ 인 경우). Chain rule을 사용하여 로컬 그래디언트 $\frac{\partial L}{\partial z}$ 를 스칼라 수준에서 증명하고, 이를 바탕으로 $\frac{\partial L}{\partial A}$ (행렬 $A$ 에 대한 미분)를 Shape Matching 관점에서 구하시오.
**(b)** 다중 분류 모델의 Softmax 출력 $p_i = \frac{e^{z_i}}{\sum e^{z_k}}$ 에 대해, Jacobian 미분 $\frac{\partial p_i}{\partial z_j}$ 를 유도하시오. (반드시 $i=j$ 와 $i \neq j$ 케이스를 나누어 몫의 미분법을 명시할 것)

## Question 5. 최적화 구조 (Learning Rate, Pooling, Quadratic Form)
**(a)** 2차 목적함수 $f(x) = \frac{1}{2} x^T A x - b^T x$ 를 최소화하는 Gradient Descent(GD) 과정에서, 파라미터 업데이트가 발산하지 않기 위한 학습률 한계 조건 $\eta < \frac{2}{\lambda_{max}(A)}$ 를 오차 방정식 $e_{t+1} = (I - \eta A)e_t$ 전개를 통해 증명하시오.
**(b)** 아래 4x4 입력 행렬에 대해 2x2 Filter, Stride=2 를 적용한 Average Pooling 연산을 수행하여 $2 \times 2$ 결과 행렬을 적고, 이 연산을 선형 행렬 곱셈 $y = Px$ 형태로 표현할 때 마스킹 필터 $P$ 행렬의 전체 차원(Shape)과 $0$이 아닌 요소(가중치)의 배치 규칙을 설명하시오.
$$ 
\begin{bmatrix}
2 & 4 & 1 & 3 \\
6 & 8 & 5 & 7 \\
1 & 3 & 0 & 2 \\
5 & 7 & 4 & 6
\end{bmatrix} 
$$
**(c)** 이차 형식(Quadratic Form) $x^T A x$ 기반 최적화에서, 헤시안(Hessian) 행렬 $A$ 의 고유값(Eigenvalue)이 모두 양수(Positive Definite)여야 하는 이유를 최적화 지형 탐색(Convexity vs Saddle Point) 관점에서 서술하시오.

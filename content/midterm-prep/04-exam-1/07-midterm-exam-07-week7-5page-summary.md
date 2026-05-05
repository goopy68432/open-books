---
title: "Deep Learning Theory Midterm"
slug: midterm-exam-07-week7-5page-summary
order: 7
---

# Deep Learning Theory Midterm
## Week 1-7 Final 5-Page Summary

> 용도: 시험 직전 30~60분 복습용
> 원칙: 정의를 외우지 말고, "왜 그런가"를 바로 말할 수 있게 정리
> 교수님 스타일: 정답만 쓰면 점수 거의 없음. 가정, 유도, 해석을 반드시 문장으로 쓸 것.

---

# Page 1. Bayesian View + i.i.d. + Learning Philosophy

## 1. Learning의 큰 그림

- 학습은 데이터 $E$를 보고 가설/모델 $H$의 파라미터 $\theta$를 조정하는 과정이다.
- 수식으로는 보통
  $$
  \theta^*=\arg\min_\theta \mathcal L(\theta;D)
  $$
  또는
  $$
  \theta^*=\arg\max_\theta p(D\mid \theta)
  $$
  로 쓴다.
- 핵심: 딥러닝은 결국 **함수 근사 + 확률 최적화**이다.

## 2. Deduction vs Induction

- **Deduction**: 일반 원리 $\to$ 특수 결론. 전제가 참이면 결론이 반드시 참.
- **Induction**: 유한한 관측 $\to$ 일반 법칙. 결론이 보장되지 않음.
- 머신러닝은 본질적으로 **귀납(induction)** 이다.
  - 훈련 데이터는 유한하다.
  - 그런데 보지 못한 입력에도 잘 작동하는 규칙을 만들어야 한다.
- 그래서 반드시 **inductive bias**가 필요하다.

## 3. Frequentist vs Bayesian

### Frequentist

- 확률 = 많은 반복 실험에서의 상대도수의 극한
- 파라미터 $\theta$는 **고정된 상수**
- 데이터의 무작위성을 다룸
- 대표 추정: **MLE**

### Bayesian

- 확률 = 가설에 대한 믿음의 정도
- 파라미터 $\theta$도 **확률변수**
- prior를 두고 데이터를 본 뒤 posterior로 업데이트
- 대표 추정: **MAP** 또는 full Bayesian inference

## 4. Bayes' Theorem

$$
p(\theta\mid D)=\frac{p(D\mid \theta)p(\theta)}{p(D)}
$$

- $p(\theta)$: **prior**
- $p(D\mid \theta)$: **likelihood**
- $p(\theta\mid D)$: **posterior**
- $p(D)$: **evidence** = 정규화 상수

### 꼭 말해야 하는 해석

- posterior는 “데이터를 본 뒤 업데이트된 믿음”
- likelihood는 “이 파라미터가 데이터를 얼마나 잘 설명하는가”
- prior는 “데이터 보기 전의 구조적/사전 지식”
- evidence는 $\theta$에 대해 최적화할 때 상수이므로 MAP/MLE 유도에서 제거 가능

## 5. MAP와 MLE

### MLE

$$
\hat\theta_{\text{MLE}}=\arg\max_\theta p(D\mid \theta)
$$

### MAP

$$
\hat\theta_{\text{MAP}}=\arg\max_\theta p(\theta\mid D)
=\arg\max_\theta [\log p(D\mid\theta)+\log p(\theta)]
$$

### 차이 한 줄 요약

- **MLE**: 데이터만 믿는다.
- **MAP**: 데이터 + prior를 함께 믿는다.

## 6. 왜 log를 취하는가

반드시 적을 말:

1. 곱을 합으로 바꿔 미분과 계산이 쉬워진다.
2. 매우 작은 확률의 곱에서 생기는 underflow를 막는다.
3. log는 단조증가 함수라서 $\arg\max$가 보존된다.

즉,
$$
\arg\max_\theta p(D\mid\theta)=\arg\max_\theta \log p(D\mid\theta)
$$

## 7. i.i.d. 가정

데이터 $x_1,\dots,x_n$가 i.i.d.라는 말:

- **independent**: 서로 독립
- **identically distributed**: 같은 분포, 같은 파라미터 공유

### 어디서 각각 쓰이는가

- **independence 사용**:
  $$
  p(D\mid\theta)=\prod_{i=1}^n p(x_i\mid\theta)
  $$
  즉 joint $\to$ product

- **identical distribution 사용**:
  모든 샘플이 같은 $\theta$를 공유
  $$
  x_i\sim p(x\mid \theta),\quad \forall i
  $$

### 시험 답안용 문장

> Independence is used when factorizing the joint likelihood into a product. Identical distribution is used when the same parameter $\theta$ explains all samples.

---

# Page 2. Core Distributions + Empirical Distribution + Information Theory + CLT

## 1. Bernoulli and Gaussian

### Bernoulli

$$
P(X=x\mid \theta)=\theta^x(1-\theta)^{1-x},\quad x\in\{0,1\}
$$

- 이진 분류, 동전 던지기
- 평균 $E[X]=\theta$
- 분산 $\mathrm{Var}(X)=\theta(1-\theta)$

### Gaussian

$$
\mathcal N(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- 회귀, 연속값, 노이즈 모델
- 평균 $\mu$, 분산 $\sigma^2$

## 2. 왜 Gaussian을 많이 쓰는가

1. **CLT**: 많은 독립적인 작은 요인의 합은 정규분포에 가까워진다.
2. 주어진 평균/분산에서 **엔트로피 최대**
3. 수학적으로 다루기 쉽다.

### 시험 문장

> Gaussian noise is justified because many small independent perturbations add up, and by the CLT their sum is approximately Gaussian.

## 3. Empirical Distribution

관측 데이터 $x_1,\dots,x_n$에 대해
$$
\hat p_{\text{data}}(x)=\frac{1}{n}\sum_{i=1}^n \delta(x-x_i)
$$

- 실제 데이터 분포 $p_{\text{data}}$를 모르므로, 경험적 분포로 대체
- ERM, MLE, CE 최소화의 출발점

## 4. Entropy / Cross-Entropy / KL

### Entropy

$$
H(P)=-\sum_x P(x)\log P(x)
$$

- 분포 자체의 불확실성

### Cross-Entropy

$$
H(P,Q)=-\sum_x P(x)\log Q(x)
$$

- 진짜 분포 $P$를 모델 $Q$로 코딩할 때의 평균 놀라움

### KL Divergence

$$
D_{\mathrm{KL}}(P\|Q)=\sum_x P(x)\log \frac{P(x)}{Q(x)}
$$

- $Q$가 $P$와 얼마나 다른가
- 비대칭: 거리(metric)가 아님

## 5. 핵심 관계식

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q)
$$

따라서 $P$가 고정이면
$$
\arg\min_Q H(P,Q)=\arg\min_Q D_{\mathrm{KL}}(P\|Q)
$$

### 시험에서 꼭 말할 것

- $H(P)$는 $Q$와 무관한 상수
- 그래서 **CE 최소화 = KL 최소화**
- 분류에서 CE loss는 ad hoc이 아니라 **categorical NLL**

## 6. Empirical NLL = Empirical CE

라벨 $y_1,\dots,y_n$에 대해
$$
-\frac1n\sum_{i=1}^n \log q_\theta(y_i)
= -\sum_k \hat p_{\text{data}}(k)\log q_\theta(k)
=H(\hat p_{\text{data}}, q_\theta)
$$

즉,

- NLL 최소화 = CE 최소화
- CE 최소화 = KL 최소화
- 결국 모델 분포를 empirical distribution에 맞추는 과정

## 7. CLT

독립이고 동일분포이며 유한 분산을 가지는 $X_1,\dots,X_n$에 대해
$$
\frac{\sum_{i=1}^n X_i-n\mu}{\sqrt{n}\sigma}
\xrightarrow{d}\mathcal N(0,1)
$$

### 조건 3개

1. independence
2. identical distribution
3. finite variance

### 자주 틀리는 포인트

- “모든 분포가 Gaussian이 된다”가 아님
- **표본 평균(또는 합)의 분포**가 Gaussian으로 간다

---

# Page 3. Bernoulli MLE + MAP

## 1. Bernoulli MLE

$n$번 시행 중 앞면이 $k$번이면
$$
L(\theta)=\theta^k(1-\theta)^{n-k}
$$

log를 취하면
$$
\ell(\theta)=k\log\theta+(n-k)\log(1-\theta)
$$

미분:
$$
\frac{d\ell}{d\theta}=\frac{k}{\theta}-\frac{n-k}{1-\theta}
$$

0으로 두면
$$
\hat\theta_{\text{MLE}}=\frac{k}{n}
$$

### 왜 derivative를 0으로 두는가

- 극대/극소의 **필요조건**이기 때문
- 단, 이것만으로 최대는 아님
- 2차 도함수로 확인:
  $$
  \frac{d^2\ell}{d\theta^2}
  =-\frac{k}{\theta^2}-\frac{n-k}{(1-\theta)^2}<0
  $$
  이므로 maximum

## 2. MLE의 한계

- 작은 데이터에서 극단값을 줄 수 있음
- 예: 2번 던져 2번 앞면이면 $\hat\theta_{\text{MLE}}=1$
- 즉 관측 안 된 사건(tail)에 확률 0을 줌
- 과도한 확신, overfitting 위험

### 시험 문장

> MLE uses only data. In small-data settings, this can produce overconfident extreme estimates, which hurts generalization.

## 3. MAP with Uniform prior

uniform prior: $p(\theta)=\text{const}$ on $[0,1]$

그러면
$$
\log p(\theta\mid D)
=\log p(D\mid\theta)+\log p(\theta)+\text{const}
$$
에서 $\log p(\theta)$도 상수이므로
$$
\boxed{\hat\theta_{\text{MAP}}=\hat\theta_{\text{MLE}}}
$$

즉 **uniform prior일 때 MAP = MLE**

## 4. MAP with prior proportional to $\theta(1-\theta)$

$$
p(\theta)\propto \theta(1-\theta)
$$
는 Beta$(2,2)$ prior와 같다.

posterior의 log:
$$
\log p(\theta\mid D)
=k\log\theta+(n-k)\log(1-\theta)+\log\theta+\log(1-\theta)+\text{const}
$$
$$
=(k+1)\log\theta+(n-k+1)\log(1-\theta)+\text{const}
$$

미분해서 0:
$$
\frac{k+1}{\theta}-\frac{n-k+1}{1-\theta}=0
$$
$$
\boxed{\hat\theta_{\text{MAP}}=\frac{k+1}{n+2}}
$$

예: $n=5,k=4$이면
$$
\hat\theta_{\text{MAP}}=\frac{5}{7}
$$

## 5. Strong prior $p(\theta)\propto \theta^m(1-\theta)^m$

이 prior는 $\theta=\frac12$를 강하게 선호한다.

MAP는
$$
\hat\theta_{\text{MAP}}=\frac{k+m}{n+2m}
$$

따라서
$$
\hat\theta_{\text{MAP}}\xrightarrow[m\to\infty]{}\frac12
$$

### 해석

- prior가 매우 강하면 데이터보다 prior가 지배
- “공정한 동전”이라는 믿음을 거의 바꾸지 않음

## 6. MLE vs MAP 한 줄 비교

- **MLE**: data-fit only
- **MAP**: data-fit + prior
- 데이터가 많아지면 MAP $\to$ MLE
- 데이터가 적을수록 prior의 영향이 커짐

---

# Page 4. Gaussian Likelihood -> MSE + Gaussian Prior -> L2 + Taylor/Jacobian/Softmax

## 1. Gaussian NLL -> MSE

회귀 모델:
$$
y_i=f_\theta(x_i)+\varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,\sigma^2)\ \text{i.i.d.}
$$

그러면
$$
y_i\mid x_i,\theta \sim \mathcal N(f_\theta(x_i),\sigma^2)
$$

likelihood:
$$
p(D\mid\theta)=\prod_{i=1}^n
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(y_i-f_\theta(x_i))^2}{2\sigma^2}\right)
$$

log:
$$
\log p(D\mid\theta)
=-\frac n2\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-f_\theta(x_i))^2
$$

negative log:
$$
\text{NLL}
=\frac n2\log(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-f_\theta(x_i))^2
$$

$\theta$와 무관한 상수와 양의 상수를 제거하면
$$
\arg\min_\theta \text{NLL}
=\arg\min_\theta \sum_{i=1}^n (y_i-f_\theta(x_i))^2
$$

즉,
$$
\boxed{\text{Gaussian NLL minimization} \iff \text{MSE minimization}}
$$

### 핵심 해석

- 왜 MSE인가?  
  Gaussian pdf 안의 $\exp(-z^2)$에 $-\log$를 취하면 $z^2$가 남기 때문

## 2. Gaussian prior -> L2 regularization

prior:
$$
\theta\sim\mathcal N(0,\sigma_p^2 I)
$$

그러면
$$
\log p(\theta)=-\frac{d}{2}\log(2\pi\sigma_p^2)-\frac{\|\theta\|^2}{2\sigma_p^2}
$$

MAP:
$$
\hat\theta_{\text{MAP}}
=\arg\max_\theta [\log p(D\mid\theta)+\log p(\theta)]
$$

최소화 형태로 바꾸면
$$
\hat\theta_{\text{MAP}}
=\arg\min_\theta
\left[
\text{MSE}
+\lambda\|\theta\|^2
\right]
$$
where
$$
\boxed{\lambda=\frac{\sigma^2}{n\sigma_p^2}}
$$

### 해석

- 작은 $\sigma_p^2$: prior 강함 -> regularization 강함
- 큰 $n$: 데이터가 많음 -> prior 영향 작아짐
- 큰 $\sigma^2$: 데이터 noisy -> prior 더 신뢰

## 3. Taylor and local linear approximation

### scalar function

$$
f(x+\Delta x)\approx f(x)+f'(x)\Delta x
$$

### vector case

$$
g(\mathbf x+\Delta \mathbf x)\approx g(\mathbf x)+J_g(\mathbf x)\Delta\mathbf x
$$

- gradient/Jacobian은 “국소 선형화” 행렬
- 복잡한 함수도 작은 구간에서는 선형처럼 행동
- backpropagation은 이런 국소 선형 maps의 chain rule 연쇄

## 4. Softmax Jacobian

softmax:
$$
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
$$

### case 1: $i=j$
$$
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)
$$

### case 2: $i\neq j$
$$
\frac{\partial p_i}{\partial z_j}=-p_ip_j
$$

### compact form
$$
\boxed{
\frac{\partial \mathbf p}{\partial \mathbf z}
=\mathrm{diag}(\mathbf p)-\mathbf p\mathbf p^\top
}
$$

### 해석

- 하나의 logit이 커지면 그 클래스 확률은 증가
- 하지만 합이 1이어야 하므로 다른 확률들은 감소

## 5. Sigmoid + BCE gradient

sigmoid:
$$
\sigma'(z)=\sigma(z)(1-\sigma(z))
$$

BCE loss:
$$
L=-[y\log\hat y+(1-y)\log(1-\hat y)],\qquad \hat y=\sigma(z),\ z=w^\top x
$$

chain rule 결과:
$$
\frac{\partial L}{\partial w}=(\hat y-y)x
$$

### 꼭 말할 것

- BCE에서는 $\sigma'(z)$가 깔끔하게 소거된다
- 그래서 MSE보다 classification에 더 적합

---

# Page 5. SVD/PCA + Rank-Nullity + Architecture/Prior + Professor-Style Answer Tips

## 1. Rank-Nullity theorem

$A\in\mathbb R^{m\times n}$에 대해
$$
\mathrm{rank}(A)+\mathrm{nullity}(A)=n
$$

- rank: 살아남는 독립 방향 수
- nullity: $Ax=0$으로 사라지는 방향 수
- 해석: “활용하는 차원 + 버려지는 차원 = 전체 차원”

## 2. SVD

$$
A=U\Sigma V^\top
$$

- $U$: left singular vectors
- $V$: right singular vectors
- $\Sigma$: singular values

rank-$k$ approximation:
$$
A_k=\sum_{i=1}^k \sigma_i u_i v_i^\top
$$

오차:
$$
\|A-A_k\|_F^2=\sum_{i=k+1}^r \sigma_i^2
$$

### Eckart-Young

- 모든 rank-$k$ 근사 중 SVD truncation이 Frobenius norm 기준 최적

## 3. PCA

공분산 행렬 $S$에 대해
$$
\max_{\|w\|=1} w^\top S w
$$

Lagrangian:
$$
\mathcal L(w,\lambda)=w^\top Sw-\lambda(w^\top w-1)
$$

미분해서 0:
$$
Sw=\lambda w
$$

즉,
- PCA direction = covariance matrix의 eigenvector
- 1st principal component = 가장 큰 eigenvalue의 eigenvector

## 4. SVD/PCA의 해석

- 큰 singular value / eigenvalue 방향 = signal 가능성 큼
- 작은 방향 = noise 가능성 큼
- low-rank approximation은 training error를 조금 늘려도 generalization을 좋게 할 수 있음

### Gaussian noise / CLT 연결

- 많은 작은 독립 요인의 합 -> Gaussian-like noise
- PCA/SVD는 이런 noise 방향을 버리고 구조적 방향을 남기는 해석 가능

## 5. Architecture = inductive bias = implicit prior

### Linear model

- 강한 bias: “관계가 선형이다”
- 작은 데이터에 유리
- 잘못된 가정이면 underfitting

### CNN

- locality
- translation equivariance
- hierarchical features

### Transformer

- 약한 bias
- 전역 상호작용 학습 가능
- 큰 데이터에서 강력

## 6. Prior / Regularization / Generalization

- explicit prior: Gaussian prior, Laplace prior 같은 확률분포
- implicit prior: architecture, optimizer, initialization, data augmentation
- regularization은 prior의 효과를 최적화 문제에 넣은 것

### 대표 대응

- Gaussian prior $\leftrightarrow L2$
- Laplace prior $\leftrightarrow L1$

### 핵심 문장

> Prior restricts the hypothesis space, regularization implements that restriction in optimization, and this often improves generalization.

## 7. 교수님 스타일 답안 팁 7개

1. **가정을 먼저 써라.**  
   i.i.d., Gaussian noise, prior shape, centered data 등

2. **independence와 identical distribution을 따로 써라.**

3. **왜 log를 취하는지 반드시 적어라.**

4. **왜 derivative를 0으로 두는지 적어라.**  
   “stationary point의 필요조건”

5. **왜 maximum/minimum인지 확인하라.**  
   2차 도함수, concavity/convexity, Hessian

6. **최종 수식 뒤에 해석 한 문장을 붙여라.**  
   예: “prior pulls the estimate toward 1/2”

7. **딥러닝과 연결하라.**  
   CE, MSE, regularization, generalization, architecture, backpropagation

## 8. 시험 직전 마지막 체크 10개

- Bayes theorem 4항을 말할 수 있는가
- Frequentist vs Bayesian 차이를 동전 예시로 말할 수 있는가
- i.i.d.에서 independence / identical distribution의 역할을 분리해서 말할 수 있는가
- Bernoulli MLE를 처음부터 유도할 수 있는가
- Uniform prior일 때 MAP = MLE를 설명할 수 있는가
- Beta$(2,2)$ prior일 때 MAP가 왜 1/2 쪽으로 가는지 설명할 수 있는가
- Gaussian NLL -> MSE를 줄마다 설명할 수 있는가
- Gaussian prior -> L2를 유도할 수 있는가
- CE = KL + H 관계를 empirical distribution과 연결할 수 있는가
- Softmax Jacobian을 $i=j$, $i\neq j$로 나눠서 쓸 수 있는가

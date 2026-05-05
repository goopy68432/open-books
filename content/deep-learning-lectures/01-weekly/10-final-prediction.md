---
title: "기말고사 60% 범위 출제예상 심층분석 보고서"
slug: final-prediction
order: 10
---

# 기말고사 60% 범위 출제예상 심층분석 보고서

분석 대상 파일:

- `/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/05_6주차_분석.md`
- `/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/06_7주차_분석.md`
- `/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/07_8주차_분석.md`
- `/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/08_9주차_분석.md`

이 보고서는 위 4개 파일을 기준으로 기말고사에 나올 가능성이 높은 개념 정의, 수식 유도, 증명형 문제, 계산형 문제를 추린 것이다. 단순히 중요도 별표만 본 것이 아니라, 다음 기준을 함께 사용했다.

1. 강의분석 파일에서 중요도 9-10으로 표시된 항목
2. 교수자가 "핵심", "직접 풀어보라", "외우지 말고 유도하라", "시험에서 많이 틀렸다"라고 강조한 항목
3. 여러 주차를 관통하는 연결 개념
4. 정의만 묻기보다 수식으로 증명하거나 계산 문제로 바꾸기 쉬운 항목
5. 이미 퀴즈로 출제되었거나 퀴즈와 거의 같은 형식으로 변형 가능한 항목

---

## 1. 전체 결론

이번 범위의 큰 줄기는 다음 네 문장으로 요약된다.

1. **Hypothesis space를 제한하는 것은 prior 또는 inductive bias를 넣는 것이다.**
2. **확률모델의 NLL은 ERM으로 바뀌고, likelihood 가정에 따라 loss가 결정된다.**
3. **학습은 parameter space에서 loss를 줄이는 optimization 문제이며, backpropagation은 chain rule로 gradient를 계산하는 방법이다.**
4. **CNN convolution은 일반 linear transformation의 제한된 형태이며, 그 제한은 sparse matrix와 weight sharing으로 나타난다.**

따라서 출제 가능성이 가장 높은 문제는 독립적인 암기 문제가 아니라, 위 네 문장을 서로 연결하는 문제다. 예를 들어 다음과 같은 형태가 매우 유력하다.

- Restricted prior에서 MAP을 구하고, 이것을 hypothesis space restriction으로 해석하라.
- Gaussian likelihood에서 MSE가 나오는 과정을 유도하고, categorical likelihood에서 cross-entropy가 나오는 과정을 비교하라.
- NLL을 empirical risk 형태로 바꾸고, 함수공간 최적화가 parameter space 최적화로 바뀌는 이유를 설명하라.
- Newton, GD, SGD, Adam의 업데이트 식과 차이를 설명하라.
- Softmax와 cross-entropy가 결합된 backpropagation 미분을 계산하라.
- 1D 또는 2D convolution을 matrix로 나타내고, sparse와 weight sharing을 표시하라.
- Output size formula를 외우지 않고 유도하라.
- CNN의 locality, translation invariance가 왜 inductive bias인지 설명하라.

---

## 2. 최상위 출제 가능 항목 요약표

| 순위 | 항목 | 유형 | 출제 가능성 | 핵심 이유 |
|---:|---|---|---|---|
| 1 | Hypothesis space restriction = prior = inductive bias | 정의+해석 | 매우 높음 | 6, 7, 9주차를 관통하는 최상위 통합 개념 |
| 2 | NLL = ERM | 수식 유도 | 매우 높음 | 7, 8주차에서 반복, likelihood와 loss 연결의 중심 |
| 3 | Gaussian likelihood -> squared loss/MSE | 수식 유도 | 매우 높음 | 6, 7주차 핵심, KL/MSE 연결과도 이어짐 |
| 4 | Categorical likelihood -> cross-entropy | 수식 유도 | 매우 높음 | classification loss의 핵심 |
| 5 | Backpropagation과 chain rule | 계산+증명 | 매우 높음 | 8주차 중요도 10, 퀴즈형으로 이미 강조 |
| 6 | Softmax Jacobian과 CE gradient | 수식 미분 | 매우 높음 | backprop 문제로 변형하기 쉬움 |
| 7 | Convolution = linear transformation | 증명+계산 | 매우 높음 | 9주차 핵심, 퀴즈 직접 출제 |
| 8 | Convolution matrix 구성 | 계산 | 매우 높음 | 표준기저로 matrix 찾기, sparse/weight sharing 확인 |
| 9 | CNN inductive bias: locality, translation invariance | 정의+해석 | 매우 높음 | 9주차 핵심 결론 |
| 10 | Output size formula | 수식 유도 | 높음 | "외우지 말고 유도" 강조 |
| 11 | Newton's method = quadratic approximation minimization | 수식 증명 | 높음 | 8주차 핵심 통찰 |
| 12 | Linear regression closed form | 수식 유도 | 높음 | 6, 8주차 모두 등장 |
| 13 | Linear -> nonlinear basis -> parametrized extractor -> NN | 개념 흐름 | 높음 | 7주차 prior 강도 비교 핵심 |
| 14 | Adam, SGD, momentum, preconditioning | 정의+비교 | 중상 | 개념형 또는 짧은 식 문제 가능 |
| 15 | Max pooling non-linearity | 반례 증명 | 중상 | 9주차 퀴즈형, 짧고 출제 쉬움 |
| 16 | AlexNet layer shape 계산 | 계산 | 중상 | 9주차 직접 분석, output size formula 응용 |

---

## 3. 출제 가능성 판단 기준

### 3.1 반복성

같은 개념이 여러 주차에 걸쳐 반복되면 출제 가능성이 높다. 대표적으로 `hypothesis space restriction`은 다음 흐름으로 계속 반복된다.

- 6주차: restricted prior가 $\theta$의 가능 영역을 제한한다.
- 6주차: scalar hypothesis $\theta$가 function hypothesis $h$로 확장된다.
- 7주차: linear model, nonlinear basis, parametrized feature extractor는 hypothesis space의 크기를 바꾸는 방식이다.
- 9주차: convolution은 일반 linear transformation 중 sparse와 weight sharing을 만족하는 subset이다.

즉, CNN도 갑자기 나온 주제가 아니라 prior/inductive bias의 최종 예시로 나온다.

### 3.2 수식으로 유도 가능한가

시험문제로 만들기 쉬운 항목은 정의만 있는 항목보다 수식 전개가 가능한 항목이다.

예:

$$
\text{NLL}(h)
=
-\log P(D\mid h)
=
-\sum_{i=1}^n \log P(y_i\mid x_i,h)
=
\sum_{i=1}^n \ell(x_i,y_i,h).
$$

이 식은 그대로 "NLL을 ERM으로 바꾸어라"라는 문제로 출제하기 좋다.

### 3.3 퀴즈 또는 직접 풀이 여부

강의분석 파일에 퀴즈로 등장한 내용은 변형 출제 가능성이 높다.

- Restricted prior MAP
- Gaussian mean MLE
- Linear regression slope
- NLL = ERM
- Newton's method
- Backpropagation 미분
- 1D convolution 계산
- Convolution matrix 구성
- Max pooling linearity 판별
- AlexNet output shape 계산

### 3.4 교수자 발언 근거

분석 파일에는 다음 종류의 발언 근거가 반복된다.

- "이걸 이해하면 NN/CNN 다 이해"
- "외우라는 게 아니고 증명할 수 있어야"
- "직접 계산해보라"
- "중간고사에서 많이 틀림"
- "학습의 기본"
- "두 가지 inductive bias"

이런 문구가 붙은 항목은 시험에서 개념 확인 또는 계산 문제로 나올 가능성이 높다.

---

## 4. 핵심 축 1 - Prior, Hypothesis Space, Inductive Bias

### 4.1 반드시 알아야 할 정의

#### Restricted uniform prior

$$
p(\theta)
=
\begin{cases}
\dfrac{1}{2a}, & \theta\in\left[\dfrac{1}{2}-a,\dfrac{1}{2}+a\right], \\[8pt]
0, & \text{otherwise}.
\end{cases}
$$

이 prior는 $\theta$가 특정 구간 안에 있을 때만 가능하다고 보는 hard restriction prior이다.

채점 포인트는 두 가지다.

1. 구간 안에서 PDF가 $\frac{1}{2a}$임을 적어야 한다.
2. 구간 밖에서 PDF가 0임을 적어야 한다.

구간 밖에서 prior가 0이면

$$
\log p(\theta)
=
\log 0
=
-\infty.
$$

따라서 likelihood가 아무리 좋아도 posterior는 0이 된다. 이것이 hard restriction의 의미다.

#### Hypothesis space restriction

Hypothesis space를 제한한다는 것은 가능한 가설의 집합을 줄이는 것이다.

$$
\mathcal{H}_{restricted}
\subset
\mathcal{H}_{all}.
$$

6주차에서는 $\theta$의 가능 구간을 제한했고, 7주차에서는 함수 $h$의 가능 집합을 제한했고, 9주차에서는 CNN이 가능한 linear transformation의 집합을 제한했다.

#### Inductive bias

Inductive bias는 모델이 데이터를 보기 전에 미리 넣어둔 가정이다.

예:

- Linear regression: 함수가 선형일 것이라는 가정
- CNN: 이미지 feature는 local하고, 같은 feature detector를 위치마다 공유해도 된다는 가정
- Markov model: 다음 token은 직전 token에만 의존한다는 가정

### 4.2 출제 가능성이 높은 이유

이 항목은 6주차, 7주차, 9주차를 연결하는 중심 개념이다. 특히 9주차 CNN 설명은 convolution 자체보다 "convolution이 어떤 prior를 넣는가"를 묻기 좋다.

출제자는 다음처럼 물을 수 있다.

- "Restricted prior와 CNN의 공통점을 hypothesis space restriction 관점에서 설명하라."
- "Linear model에서 neural network로 갈수록 prior가 강해지는지 약해지는지 설명하라."
- "CNN은 왜 일반 matrix multiplication보다 강한 inductive bias를 갖는가?"

### 4.3 예상문제 1

**문제:** Restricted uniform prior

$$
p(\theta)
=
\begin{cases}
\dfrac{1}{2a}, & \theta\in\left[\dfrac{1}{2}-a,\dfrac{1}{2}+a\right],\\
0, & \text{otherwise}
\end{cases}
$$

가 주어졌다. 동전을 $n=4$번 던져 $k=4$번 앞면이 나왔다. MAP 추정값 $\hat{\theta}_{MAP}$를 구하고, 왜 MLE와 다른지 설명하라.

**출제 의도:** 중간고사에서 많이 틀린 restricted prior MAP의 변형이다.

**풀이 골격:**

Likelihood는

$$
P(D\mid\theta)
\propto
\theta^4.
$$

Posterior는

$$
p(\theta\mid D)
\propto
\theta^4 p(\theta).
$$

prior support 밖에서는 $p(\theta)=0$이므로 posterior도 0이다. 따라서 가능한 구간은

$$
\theta\in\left[\frac{1}{2}-a,\frac{1}{2}+a\right].
$$

이 구간 안에서는 prior가 상수이므로

$$
p(\theta\mid D)
\propto
\theta^4.
$$

$\theta^4$는 $\theta$에 대해 단조 증가하므로 최대는 upper boundary에서 발생한다.

$$
\boxed{
\hat{\theta}_{MAP}
=
\frac{1}{2}+a
}
$$

MLE는 $\theta=1$이지만, MAP은 prior가 허용하는 구간 밖으로 나갈 수 없다. 따라서 prior가 hypothesis space를 제한했고, 그 결과 답이 boundary에 걸린다.

### 4.4 예상문제 2

**문제:** 다음 네 모델을 prior가 강한 순서에서 약한 순서로 배열하고 이유를 설명하라.

1. Linear model $f(x)=w^Tx$
2. Nonlinear basis model $f(x)=w^T\phi(x)$
3. Parametrized feature extractor $f(x)=w^T\phi(x;w')$
4. Deep neural network

**예상 답안:**

강한 prior에서 약한 prior 순서는

$$
\text{Linear}
\rightarrow
\text{Nonlinear basis}
\rightarrow
\text{Parametrized feature extractor}
\rightarrow
\text{Deep neural network}.
$$

Linear model은 선형 decision boundary만 허용하므로 가장 좁은 hypothesis space를 가진다. Nonlinear basis를 쓰면 $\phi(x)$가 만든 feature 위에서 선형이므로 더 다양한 함수가 가능하다. Parametrized feature extractor는 $\phi$ 자체를 학습하므로 사람이 넣은 prior가 더 줄어든다. Deep neural network는 여러 층의 linear transformation과 nonlinearity를 반복하여 훨씬 큰 함수공간을 표현한다.

---

## 5. 핵심 축 2 - Likelihood, NLL, ERM, Loss

### 5.1 반드시 알아야 할 정의

데이터셋이

$$
D=\{(x_i,y_i)\}_{i=1}^n
$$

이고 IID라고 하자. 그러면 likelihood는

$$
P(D\mid h)
=
\prod_{i=1}^n P(y_i\mid x_i,h).
$$

Negative log-likelihood는

$$
\begin{aligned}
\text{NLL}(h)
&=
-\log P(D\mid h)\\
&=
-\log \prod_{i=1}^n P(y_i\mid x_i,h)\\
&=
-\sum_{i=1}^n \log P(y_i\mid x_i,h).
\end{aligned}
$$

여기서 loss를

$$
\ell(x_i,y_i,h)
=
-\log P(y_i\mid x_i,h)
$$

로 정의하면

$$
\text{NLL}(h)
=
\sum_{i=1}^n \ell(x_i,y_i,h).
$$

평균을 취하면 empirical risk가 된다.

$$
\hat{L}_S(h)
=
\frac{1}{n}
\sum_{i=1}^n
\ell(x_i,y_i,h).
$$

상수 $n$은 argmin을 바꾸지 않으므로

$$
\arg\min_h \text{NLL}(h)
=
\arg\min_h \hat{L}_S(h).
$$

따라서

$$
\boxed{
\text{NLL 최소화}
\equiv
\text{ERM}
}
$$

이다.

### 5.2 Empirical distribution

Empirical distribution은 dataset의 각 sample에 동일한 질량을 주는 분포다.

$$
P_S(x)
=
\frac{1}{n}
\sum_{i=1}^n
\delta(x-x_i).
$$

이 분포에서 expectation을 취하면 sample average가 된다.

$$
\mathbb{E}_{x\sim P_S}[\ell(x,h)]
=
\frac{1}{n}
\sum_{i=1}^n
\ell(x_i,h).
$$

이 식은 ERM의 정의와 같다.

### 5.3 Gaussian likelihood -> MSE

Regression에서

$$
y_i\mid x_i,h
\sim
\mathcal{N}(h(x_i),\sigma^2)
$$

라고 가정하자. 그러면

$$
P(y_i\mid x_i,h)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{(y_i-h(x_i))^2}{2\sigma^2}
\right).
$$

NLL의 한 sample 항은

$$
\begin{aligned}
-\log P(y_i\mid x_i,h)
&=
-\log
\left[
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{(y_i-h(x_i))^2}{2\sigma^2}
\right)
\right]\\
&=
\log\sqrt{2\pi\sigma^2}
+
\frac{(y_i-h(x_i))^2}{2\sigma^2}.
\end{aligned}
$$

첫 번째 항은 상수이고, $\sigma^2$가 고정이면 $\frac{1}{2\sigma^2}$도 상수이다. 따라서 Gaussian NLL을 최소화하는 것은

$$
\sum_{i=1}^n
(y_i-h(x_i))^2
$$

를 최소화하는 것과 같다. 평균을 취하면 MSE다.

$$
\boxed{
\text{Gaussian likelihood}
\Rightarrow
\text{squared loss/MSE}
}
$$

### 5.4 Categorical likelihood -> Cross-Entropy

Multi-class classification에서

$$
y_i\in\{1,\dots,C\}
$$

이고, 모델 출력이

$$
h(x_i)
=
\begin{bmatrix}
p_1\\
\vdots\\
p_C
\end{bmatrix}
$$

라고 하자. 정답 class가 $y_i$이면 categorical likelihood는

$$
P(y_i\mid x_i,h)
=
h(x_i)_{y_i}.
$$

따라서 NLL은

$$
\ell(h,x_i,y_i)
=
-\log h(x_i)_{y_i}.
$$

이것이 cross-entropy loss이다.

One-hot vector $e_{y_i}$를 쓰면

$$
\ell
=
-\sum_{c=1}^C (e_{y_i})_c \log p_c
=
-\log p_{y_i}.
$$

따라서

$$
\boxed{
\text{Categorical likelihood}
\Rightarrow
\text{Cross-Entropy}
}
$$

이다.

### 5.5 출제 가능성이 높은 이유

이 항목은 6주차의 Bernoulli/Gaussian/Categorical 모델 설명, 7주차의 NLL-ERM 연결, 8주차의 parameter optimization으로 모두 이어진다. 시험에서 하나의 긴 문제로 묶기 쉽다.

예:

1. Gaussian likelihood에서 MSE를 유도하라.
2. Categorical likelihood에서 CE를 유도하라.
3. 둘을 NLL과 ERM 관점에서 비교하라.

### 5.6 예상문제 3

**문제:** IID data $D=\{(x_i,y_i)\}_{i=1}^n$에 대해 NLL을 ERM 형태로 바꾸어라. 또한 Gaussian regression과 categorical classification에서 loss가 각각 무엇이 되는지 쓰라.

**예상 답안 핵심:**

$$
\begin{aligned}
\text{NLL}(h)
&=
-\log P(D\mid h)\\
&=
-\log \prod_i P(y_i\mid x_i,h)\\
&=
\sum_i -\log P(y_i\mid x_i,h).
\end{aligned}
$$

loss를

$$
\ell(x_i,y_i,h)
=
-\log P(y_i\mid x_i,h)
$$

로 정의하면

$$
\text{NLL}(h)
=
\sum_i \ell(x_i,y_i,h)
=
n\hat{L}_S(h).
$$

Gaussian이면

$$
\ell_i
=
\frac{(y_i-h(x_i))^2}{2\sigma^2}
+C,
$$

즉 MSE와 동치이다.

Categorical이면

$$
\ell_i
=
-\log h(x_i)_{y_i},
$$

즉 cross-entropy이다.

---

## 6. 핵심 축 3 - Gaussian Mean MLE와 Linear Regression

### 6.1 Gaussian mean MLE

데이터가

$$
y_1,\dots,y_n
\sim
\mathcal{N}(\mu,1)
$$

라고 하자. NLL은 상수를 제외하면

$$
L(\mu)
=
\frac{1}{2}
\sum_{i=1}^n
(y_i-\mu)^2.
$$

$\mu$로 미분하면

$$
\begin{aligned}
\frac{\partial L}{\partial \mu}
&=
\frac{1}{2}
\sum_{i=1}^n
2(y_i-\mu)(-1)\\
&=
-\sum_{i=1}^n(y_i-\mu).
\end{aligned}
$$

최적점에서 미분은 0이다.

$$
-\sum_{i=1}^n(y_i-\mu)=0.
$$

따라서

$$
-\sum_{i=1}^n y_i + n\mu=0,
$$

$$
n\mu=\sum_{i=1}^n y_i,
$$

$$
\boxed{
\mu^*
=
\frac{1}{n}\sum_{i=1}^n y_i
}
$$

이다.

### 6.2 Linear regression with $h(x)=ax$

모델을

$$
h(x)=ax
$$

로 제한하면 loss는

$$
L(a)
=
\frac{1}{2}
\sum_{i=1}^n
(y_i-ax_i)^2.
$$

$a$로 미분하면

$$
\begin{aligned}
\frac{\partial L}{\partial a}
&=
\frac{1}{2}
\sum_i
2(y_i-ax_i)(-x_i)\\
&=
-\sum_i x_i(y_i-ax_i).
\end{aligned}
$$

0으로 놓으면

$$
-\sum_i x_i y_i
+
a\sum_i x_i^2
=
0.
$$

따라서

$$
a\sum_i x_i^2
=
\sum_i x_i y_i,
$$

$$
\boxed{
a^*
=
\frac{\sum_i x_i y_i}{\sum_i x_i^2}
}
$$

이다.

### 6.3 Matrix form closed form

일반 linear regression을

$$
L(w)
=
\frac{1}{2}
\|Xw-y\|^2
$$

로 두자. 전개하면

$$
\begin{aligned}
L(w)
&=
\frac{1}{2}(Xw-y)^T(Xw-y)\\
&=
\frac{1}{2}
\left(
w^TX^TXw
-2w^TX^Ty
+y^Ty
\right).
\end{aligned}
$$

미분하면

$$
\nabla_w L
=
X^TXw-X^Ty.
$$

최적점에서

$$
X^TXw-X^Ty=0.
$$

따라서

$$
X^TXw=X^Ty.
$$

$X^TX$가 invertible이면

$$
\boxed{
w^*
=
(X^TX)^{-1}X^Ty
}
$$

이다.

### 6.4 출제 가능성이 높은 이유

6주차에서는 Gaussian mean MLE와 $h(x)=ax$ linear regression을 직접 칠판 풀이로 다뤘고, 8주차에서는 matrix form closed form을 다시 다뤘다. 따라서 단순 계산형 또는 수식 유도형으로 매우 적합하다.

### 6.5 예상문제 4

**문제:** $y_i\sim\mathcal{N}(ax_i,1)$라고 가정한다. $a$의 MLE를 구하라.

**정답:**

$$
\boxed{
a^*
=
\frac{\sum_i x_i y_i}{\sum_i x_i^2}
}
$$

**채점 포인트:**

1. NLL을 $\frac{1}{2}\sum_i(y_i-ax_i)^2+C$로 세운다.
2. $a$로 미분한다.
3. 미분값을 0으로 놓고 $a$를 푼다.
4. 이 결과가 origin을 지나는 best line의 기울기임을 해석한다.

---

## 7. 핵심 축 4 - Model Capacity, XOR, Universal Approximation, Bitter Lesson

### 7.1 주요 개념

#### Perceptron

$$
f(x)
=
\mathbb{1}[w^Tx\ge 0].
$$

Perceptron은 linear decision boundary만 만들 수 있다.

#### XOR problem

XOR 데이터는 다음과 같다.

| $x_1$ | $x_2$ | $y$ |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

이 데이터는 하나의 직선으로 두 class를 분리할 수 없다. 따라서 linear model의 한계를 보여준다.

#### Nonlinear basis

$$
f(x)
=
w^T\phi(x).
$$

입력을 사람이 설계한 feature $\phi(x)$로 바꾼 뒤 그 위에서 linear model을 사용한다.

#### Parametrized feature extractor

$$
f(x)
=
w^T\phi(x;w').
$$

이제 feature extractor 자체도 학습된다. 사람이 직접 feature를 정하는 정도가 줄어드므로 prior가 약해진다.

#### 2-layer neural network

$$
f(x)
=
w^T\sigma(w'^Tx).
$$

충분히 wide하면 임의의 함수를 근사할 수 있다는 universal approximation 관점이 있다.

### 7.2 출제 가능성이 높은 이유

이 부분은 계산보다는 개념형 문제로 나올 가능성이 높다. 특히 "prior가 강해지는가 약해지는가"는 7주차 퀴즈에 이미 등장했다.

### 7.3 예상문제 5

**문제:** XOR problem이 linear model로 풀리지 않는 이유를 설명하고, nonlinear basis 또는 neural network가 이를 해결할 수 있는 이유를 hypothesis space 관점에서 설명하라.

**예상 답안 핵심:**

Linear model은 decision boundary가 hyperplane이다. XOR의 양성 class $(0,1),(1,0)$과 음성 class $(0,0),(1,1)$는 하나의 직선으로 분리되지 않는다. 따라서 linear hypothesis space 안에는 XOR을 정확히 표현하는 함수가 없다.

Nonlinear basis $\phi(x)$를 도입하면 원래 입력공간에서 선형분리 불가능한 데이터가 feature space에서 선형분리 가능해질 수 있다. Neural network는 $\phi$ 자체를 학습하므로 사람이 feature를 정하지 않아도 더 넓은 hypothesis space를 탐색할 수 있다.

---

## 8. 핵심 축 5 - Optimization

### 8.1 Gradient descent

Gradient descent는 현재 위치에서 gradient 반대 방향으로 이동한다.

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta\nabla L(\theta_t)
}
$$

여기서 $\eta$는 learning rate이다. $\eta$가 작으면 안정적이지만 느리고, 크면 빠르지만 발산할 수 있다.

### 8.2 SGD

전체 dataset $S$가 아니라 mini-batch $B$에서 gradient를 계산한다.

$$
\theta_{t+1}
=
\theta_t
-
\eta
\nabla L_B(\theta_t).
$$

장점:

- 계산량이 줄어든다.
- 노이즈가 생기지만, 실전에서는 generalization이 더 좋아지는 경우가 많다.

### 8.3 Newton's method

원래 Newton's method는 $f(\theta)=0$의 해를 찾는다.

$$
\theta_{t+1}
=
\theta_t
-
\frac{f(\theta_t)}{f'(\theta_t)}.
$$

만약

$$
f=L'
$$

라면

$$
f'=L''.
$$

따라서

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\frac{L'(\theta_t)}{L''(\theta_t)}
}
$$

이다.

### 8.4 Newton's method의 2차 근사 해석

$L$을 $\theta_t$ 근처에서 2차 Taylor approximation 한다.

$$
\hat{L}(\theta)
=
L(\theta_t)
+
L'(\theta_t)(\theta-\theta_t)
+
\frac{1}{2}L''(\theta_t)(\theta-\theta_t)^2.
$$

이 근사함수의 최소점을 찾기 위해 미분한다.

$$
\hat{L}'(\theta)
=
L'(\theta_t)
+
L''(\theta_t)(\theta-\theta_t).
$$

최소점에서

$$
\hat{L}'(\theta)=0.
$$

따라서

$$
L'(\theta_t)
+
L''(\theta_t)(\theta-\theta_t)
=
0.
$$

이를 풀면

$$
L''(\theta_t)(\theta-\theta_t)
=
-L'(\theta_t),
$$

$$
\theta-\theta_t
=
-
\frac{L'(\theta_t)}{L''(\theta_t)},
$$

$$
\boxed{
\theta
=
\theta_t
-
\frac{L'(\theta_t)}{L''(\theta_t)}
}
$$

이것이 Newton update이다.

### 8.5 Momentum, preconditioning, Adam

Momentum은 과거 gradient의 moving average를 사용한다.

$$
m_t
=
\beta m_{t-1}
+
g_t.
$$

업데이트는

$$
\theta_{t+1}
=
\theta_t
-
\eta m_t.
$$

Second-order 또는 preconditioned method는 곡률 정보를 사용한다.

$$
\theta_{t+1}
=
\theta_t
-
\eta H^{-1}g_t.
$$

하지만 neural network에서는 parameter 수 $d$가 매우 크므로 Hessian은 $d\times d$이고, 역행렬 계산은 현실적으로 불가능하다.

AdaGrad/RMSProp 계열은 gradient 제곱을 이용하여 방향별 scale을 조절한다.

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{g_t}{\sqrt{s_t+\epsilon}}.
$$

Adam은 momentum과 RMSProp을 결합한 optimizer로, 강의분석 파일에서 LLM 학습의 default에 가깝다고 강조되어 있다.

### 8.6 출제 가능성이 높은 이유

Optimization은 8주차의 핵심이고, 딥러닝 학습을 실제로 가능하게 하는 부분이다. 계산 문제로는 Newton's method 유도가 가장 유력하고, 개념 문제로는 GD/SGD/Adam 비교가 유력하다.

### 8.7 예상문제 6

**문제:** $f=L'$일 때 Newton's method update가 $L$의 2차 근사 minimum을 찾는 과정임을 보여라.

**정답 방향:** 위 8.4의 Taylor approximation을 그대로 쓰면 된다.

**채점 포인트:**

1. $L$의 2차 근사식을 쓴다.
2. 근사식의 미분을 0으로 놓는다.
3. $\theta_{t+1}=\theta_t-\frac{L'(\theta_t)}{L''(\theta_t)}$를 얻는다.
4. 이것이 $f=L'$일 때 Newton's method와 같음을 연결한다.

### 8.8 예상문제 7

**문제:** GD, SGD, Newton's method, Adam의 차이를 gradient 정보 관점에서 설명하라.

**예상 답안 핵심:**

- GD: 전체 데이터 gradient를 사용한다.
- SGD: mini-batch gradient를 사용한다.
- Newton: gradient와 Hessian, 즉 1차와 2차 정보를 사용한다.
- Adam: momentum과 adaptive scaling을 결합하여 1차 gradient 통계의 moving average를 사용한다.

---

## 9. 핵심 축 6 - Backpropagation과 Chain Rule

### 9.1 기본 computational graph

강의분석 파일의 backprop 예시는 다음 구조다.

$$
x
\rightarrow
z_1=W_1x
\rightarrow
\tilde{z}=\operatorname{ReLU}(z_1)
\rightarrow
z_2=W_2\tilde{z}
\rightarrow
p=\operatorname{softmax}(z_2)
\rightarrow
p_y=e_y^Tp
\rightarrow
L=-\log p_y.
$$

핵심은 각 edge의 미분을 구한 뒤 chain rule로 곱하는 것이다.

### 9.2 반드시 외워야 할 미분

#### Cross-entropy 마지막 항

$$
L=-\log p_y.
$$

따라서

$$
\boxed{
\frac{\partial L}{\partial p_y}
=
-
\frac{1}{p_y}
}
$$

#### 정답 class 선택

$$
p_y=e_y^Tp.
$$

따라서

$$
\boxed{
\frac{\partial p_y}{\partial p}
=
e_y^T
}
$$

#### Softmax Jacobian

Softmax는

$$
p_i
=
\frac{e^{z_i}}{\sum_k e^{z_k}}.
$$

따라서

$$
\boxed{
\frac{\partial p_i}{\partial z_j}
=
p_i(\delta_{ij}-p_j)
}
$$

matrix로는

$$
\boxed{
\frac{\partial p}{\partial z}
=
\operatorname{diag}(p)-pp^T
}
$$

이다.

#### Softmax + CE 결합 결과

위 세 식을 연결하면

$$
\boxed{
\frac{\partial L}{\partial z}
=
p-e_y
}
$$

가 된다.

### 9.3 ReLU 미분

$$
\operatorname{ReLU}(u)
=
\max(0,u).
$$

미분은

$$
\operatorname{ReLU}'(u)
=
\begin{cases}
1, & u>0,\\
0, & u<0.
\end{cases}
$$

$u=0$에서는 미분이 정의되지 않지만, 구현에서는 보통 0 또는 임의의 subgradient를 사용한다.

### 9.4 출제 가능성이 높은 이유

Backpropagation은 8주차에서 중요도 10으로 표시되어 있고, 실제 퀴즈로도 미분 3개를 직접 계산하게 했다. 따라서 기말고사에서도 형태만 조금 바꾸어 출제될 가능성이 높다.

### 9.5 예상문제 8

**문제:** 다음 network가 있다.

$$
z=W_2\operatorname{ReLU}(W_1x),
\qquad
p=\operatorname{softmax}(z),
\qquad
L=-\log p_y.
$$

다음을 구하라.

1. $\frac{\partial L}{\partial p_y}$
2. $\frac{\partial p}{\partial z}$
3. $\frac{\partial L}{\partial z}$
4. $\frac{\partial L}{\partial W_2}$

**예상 답안:**

1번:

$$
\frac{\partial L}{\partial p_y}
=
-\frac{1}{p_y}.
$$

2번:

$$
\frac{\partial p_i}{\partial z_j}
=
p_i(\delta_{ij}-p_j),
$$

또는

$$
\frac{\partial p}{\partial z}
=
\operatorname{diag}(p)-pp^T.
$$

3번:

$$
\frac{\partial L}{\partial z}
=
p-e_y.
$$

4번에서

$$
a=\operatorname{ReLU}(W_1x)
$$

라고 두면

$$
z=W_2a.
$$

따라서

$$
\boxed{
\frac{\partial L}{\partial W_2}
=
(p-e_y)a^T
}
$$

이다.

---

## 10. 핵심 축 7 - Convolution과 CNN

### 10.1 이미지 처리 = linear transformation restriction

이미지를 vector로 펼치면 linear layer는 일반적으로

$$
y=Ax
$$

로 쓸 수 있다. 하지만 CNN은 모든 matrix $A$를 허용하지 않는다. Convolution에 해당하는 특수한 matrix만 허용한다.

그 matrix는 두 특징을 가진다.

1. **Sparse:** local patch만 보므로 대부분의 entry가 0이다.
2. **Weight sharing:** 같은 kernel weight가 여러 위치에서 반복된다.

이것이 CNN의 inductive bias다.

### 10.2 Inner product = similarity

Convolution은 patch와 kernel의 inner product를 계산한다.

$$
\langle x,w\rangle
=
\|x\|\|w\|\cos\theta.
$$

두 vector가 같은 방향이면 내적이 크다. 따라서 kernel은 feature detector처럼 작동한다.

또한

$$
\|x-w\|^2
=
\|x\|^2+\|w\|^2-2\langle x,w\rangle.
$$

내적이 클수록 거리가 작아진다. 따라서 inner product는 similarity measure로 해석할 수 있다.

### 10.3 Convolution linearity

Convolution 연산을 $f$라고 하자. Kernel을 고정하면 convolution은 입력 $x$에 대해 linear이다.

즉,

$$
f(x_1+x_2)
=
f(x_1)+f(x_2),
$$

$$
f(cx)
=
cf(x).
$$

왜냐하면 convolution의 각 출력은 입력 성분들의 선형결합이기 때문이다.

1D 예를 들면 kernel $w=[1,2]$일 때

$$
f(x)_i
=
x_i+2x_{i+1}.
$$

그러면

$$
\begin{aligned}
f(x+u)_i
&=
(x_i+u_i)+2(x_{i+1}+u_{i+1})\\
&=
(x_i+2x_{i+1})+(u_i+2u_{i+1})\\
&=
f(x)_i+f(u)_i.
\end{aligned}
$$

또한

$$
\begin{aligned}
f(cx)_i
&=
cx_i+2cx_{i+1}\\
&=
c(x_i+2x_{i+1})\\
&=
cf(x)_i.
\end{aligned}
$$

따라서 convolution은 linear transformation이다.

### 10.4 1D convolution matrix

입력이

$$
x\in\mathbb{R}^7
$$

이고 kernel이

$$
w=[1,2]
$$

일 때 valid convolution 출력은

$$
y\in\mathbb{R}^6
$$

이다. 각 출력은

$$
y_i=x_i+2x_{i+1}.
$$

따라서

$$
\begin{aligned}
y_1 &= x_1+2x_2,\\
y_2 &= x_2+2x_3,\\
y_3 &= x_3+2x_4,\\
y_4 &= x_4+2x_5,\\
y_5 &= x_5+2x_6,\\
y_6 &= x_6+2x_7.
\end{aligned}
$$

Matrix로 쓰면

$$
\boxed{
A
=
\begin{bmatrix}
1 & 2 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 2 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 2 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 2 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 2 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 2
\end{bmatrix}
}
$$

이다.

여기서 sparse와 weight sharing이 모두 보인다. 각 행에는 두 개의 nonzero entry만 있고, $1,2$라는 같은 weight가 행마다 반복된다.

### 10.5 2D convolution matrix

입력을 row-major로 펼친다.

$$
X
=
\begin{bmatrix}
x_1 & x_2 & x_3\\
x_4 & x_5 & x_6\\
x_7 & x_8 & x_9
\end{bmatrix}.
$$

Kernel이

$$
W
=
\begin{bmatrix}
0 & 1\\
2 & 3
\end{bmatrix}
$$

이고 valid convolution을 하면 $2\times2$ output이 나온다.

각 출력은

$$
y_1=x_2+2x_4+3x_5,
$$

$$
y_2=x_3+2x_5+3x_6,
$$

$$
y_3=x_5+2x_7+3x_8,
$$

$$
y_4=x_6+2x_8+3x_9.
$$

따라서

$$
\boxed{
A
=
\begin{bmatrix}
0 & 1 & 0 & 2 & 3 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 2 & 3 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & 2 & 3 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 2 & 3
\end{bmatrix}
}
$$

이다.

### 10.6 Output size formula

입력 높이를 $H_{in}$, padding을 $p$, kernel size를 $k$, stride를 $s$라고 하자.

Padding을 적용한 입력 크기는

$$
H_{in}+2p
$$

이다.

Kernel의 시작점이 움직일 수 있는 총 거리는

$$
H_{in}+2p-k
$$

이다. stride가 $s$이면 가능한 이동 횟수는

$$
\left\lfloor
\frac{H_{in}+2p-k}{s}
\right\rfloor.
$$

첫 위치를 포함해야 하므로 output size는

$$
\boxed{
H_{out}
=
\left\lfloor
\frac{H_{in}+2p-k}{s}
\right\rfloor
+1
}
$$

이다.

stride로 정확히 나누어떨어지는 경우에는 floor를 생략하여

$$
H_{out}
=
\frac{H_{in}+2p-k}{s}+1
$$

로 쓴다.

### 10.7 Same convolution padding

Stride가 $s=1$이고 input과 output 크기를 같게 만들고 싶으면

$$
H_{out}=H_{in}
$$

이어야 한다.

공식에 대입하면

$$
H_{in}
=
H_{in}+2p-k+1.
$$

양변에서 $H_{in}$을 빼면

$$
0
=
2p-k+1.
$$

따라서

$$
2p=k-1,
$$

$$
\boxed{
p=\frac{k-1}{2}
}
$$

이다.

일반 stride $s$까지 포함한 "same" convention을 $H_{out}=H_{in}/s$로 해석하면 강의분석 파일의 정리처럼

$$
2p=k-s
$$

가 된다. 반대로 $s\ne1$인데도 문자 그대로 $H_{out}=H_{in}$을 요구하면 padding은 $H_{in}$에도 의존하므로, 시험에서는 어떤 "same" 정의를 쓰는지 먼저 확인해야 한다.

### 10.8 Pooling

Average pooling은 평균을 내므로 linear이다.

예를 들어 2개 입력의 average pooling은

$$
f(x_1,x_2)
=
\frac{x_1+x_2}{2}
$$

이고 matrix로

$$
\begin{bmatrix}
\frac{1}{2} & \frac{1}{2}
\end{bmatrix}
\begin{bmatrix}
x_1\\x_2
\end{bmatrix}
$$

처럼 쓸 수 있다.

반면 max pooling은 linear가 아니다. 반례를 보이면 된다.

$$
x=(1,0),
\qquad
u=(0,1).
$$

그러면

$$
\max(x)=1,
\qquad
\max(u)=1,
$$

따라서

$$
\max(x)+\max(u)=2.
$$

하지만

$$
x+u=(1,1),
$$

$$
\max(x+u)=1.
$$

따라서

$$
\max(x+u)
\ne
\max(x)+\max(u).
$$

Additivity가 깨지므로 max pooling은 linear가 아니다.

### 10.9 출제 가능성이 높은 이유

9주차 전체가 CNN을 "linear algebra의 가장 중요한 정리, linear transformation = matrix"에 연결하는 방식으로 구성되어 있다. 또한 실제 퀴즈가 1D convolution 계산, matrix 구성, max pooling linearity, AlexNet shape 계산으로 되어 있어 계산 문제 출제 가능성이 매우 높다.

### 10.10 예상문제 9

**문제:** Kernel $w=[1,2]$를 사용하여 $x=[1,2,3,4,5,6,7]^T$에 대한 valid 1D convolution을 계산하고, 이에 대응하는 matrix $A\in\mathbb{R}^{6\times7}$를 구하라.

**예상 답안:**

출력은

$$
y_i=x_i+2x_{i+1}.
$$

따라서

$$
y
=
\begin{bmatrix}
1+2\cdot2\\
2+2\cdot3\\
3+2\cdot4\\
4+2\cdot5\\
5+2\cdot6\\
6+2\cdot7
\end{bmatrix}
=
\begin{bmatrix}
5\\
8\\
11\\
14\\
17\\
20
\end{bmatrix}.
$$

Matrix는

$$
A
=
\begin{bmatrix}
1 & 2 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 2 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 2 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 2 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 2 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 2
\end{bmatrix}.
$$

### 10.11 예상문제 10

**문제:** Convolution matrix에서 sparse와 weight sharing이 무엇인지 위 $A$를 이용해 설명하라.

**예상 답안:**

Sparse는 matrix의 대부분 entry가 0이라는 뜻이다. 위 $A$는 각 행에 두 개의 nonzero entry만 있다. 이는 convolution이 local patch만 보기 때문이다.

Weight sharing은 같은 weight가 여러 위치에서 반복되는 것이다. 위 $A$에서는 $1,2$가 각 행마다 한 칸씩 이동하며 반복된다. 이는 같은 kernel을 입력의 모든 위치에 적용하기 때문이다.

따라서 convolution은 일반 matrix multiplication보다 가능한 matrix 형태가 제한된다. 이것이 CNN의 inductive bias다.

### 10.12 예상문제 11

**문제:** $H_{in}=227$, kernel size $k=11$, stride $s=4$, padding $p=0$인 convolution의 output size를 계산하라. 이 계산이 AlexNet Conv1과 어떻게 연결되는지 설명하라.

**예상 답안:**

$$
H_{out}
=
\frac{H_{in}+2p-k}{s}+1.
$$

대입하면

$$
H_{out}
=
\frac{227+0-11}{4}+1
=
\frac{216}{4}+1
=
54+1
=
55.
$$

따라서 AlexNet의 첫 convolution은 spatial size를

$$
227\times227
\rightarrow
55\times55
$$

로 바꾼다. output channel이 96이면 결과 shape은

$$
55\times55\times96
$$

이다.

### 10.13 예상문제 12

**문제:** Max pooling이 linear transformation인지 판별하고, 아니라면 반례를 제시하라.

**예상 답안:**

Max pooling은 linear가 아니다. 예를 들어

$$
x=(1,0),
\qquad
u=(0,1)
$$

이면

$$
f(x)=1,
\qquad
f(u)=1,
$$

따라서

$$
f(x)+f(u)=2.
$$

하지만

$$
x+u=(1,1),
$$

$$
f(x+u)=1.
$$

따라서

$$
f(x+u)\ne f(x)+f(u).
$$

Additivity가 성립하지 않으므로 max pooling은 linear transformation이 아니다.

---

## 11. 통합형 예상문제

아래 문제들은 개념을 여러 주차에 걸쳐 연결해야 하는 고난도 예상문제다.

### 통합문제 A - Prior에서 CNN까지

**문제:** Restricted prior, linear regression, CNN convolution은 모두 hypothesis space restriction의 예라고 볼 수 있다. 세 경우 각각에서 어떤 공간이 어떻게 제한되는지 설명하고, 이 제한이 inductive bias와 어떤 관련이 있는지 서술하라.

**답안 골격:**

1. Restricted prior:
   $$
   \theta\in\left[\frac{1}{2}-a,\frac{1}{2}+a\right]
   $$
   만 허용한다. Scalar parameter space를 제한한다.

2. Linear regression:
   모든 함수 $h:X\to Y$가 아니라
   $$
   h(x)=ax+b
   $$
   꼴의 함수만 허용한다. Function space를 선형 함수공간으로 제한한다.

3. CNN:
   모든 matrix $A$가 아니라 sparse와 weight sharing을 만족하는 convolution matrix만 허용한다. Linear transformation space를 제한한다.

공통적으로 가능한 hypothesis의 집합을 줄인다. 이는 데이터가 적을 때 generalization에 도움을 줄 수 있지만, 잘못된 prior이면 표현력이 부족해질 수 있다.

### 통합문제 B - Likelihood에서 Optimization까지

**문제:** Gaussian regression에서 시작하여 MSE loss를 얻고, 이를 empirical risk로 쓴 뒤, parameterized model $h_\theta$에 대해 gradient descent update를 쓰라.

**답안 골격:**

Gaussian likelihood:

$$
y_i\mid x_i,h_\theta
\sim
\mathcal{N}(h_\theta(x_i),\sigma^2).
$$

NLL:

$$
\text{NLL}(\theta)
=
\sum_i
\left[
\frac{(y_i-h_\theta(x_i))^2}{2\sigma^2}
+C
\right].
$$

Empirical risk:

$$
\hat{L}_S(\theta)
=
\frac{1}{n}
\sum_i
(y_i-h_\theta(x_i))^2.
$$

Gradient descent:

$$
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta \hat{L}_S(\theta_t).
$$

### 통합문제 C - CNN과 Bitter Lesson

**문제:** CNN과 Transformer를 inductive bias 관점에서 비교하고, 이 비교가 Bitter Lesson과 어떻게 연결되는지 설명하라.

**답안 골격:**

CNN은 이미지에 대해 locality와 translation invariance를 강하게 가정한다. 따라서 strong inductive bias를 가진다.

Transformer는 attention을 통해 모든 token이 서로 영향을 줄 수 있게 하므로 Markov model보다 약한 prior를 가진다. CNN보다도 domain-specific locality bias가 약하다.

Bitter Lesson은 사람이 넣은 prior knowledge보다 scale과 computation을 활용하는 general method가 장기적으로 더 강하다는 주장이다. 따라서 데이터와 compute가 많아질수록 strong hand-crafted prior보다 weak prior 모델이 유리해질 수 있다.

---

## 12. 출제 가능성별 공부 우선순위

### 1순위 - 반드시 손으로 유도할 것

아래는 시험지에 그대로 나와도 이상하지 않은 항목이다.

1. Restricted prior PDF와 MAP boundary 계산
2. Gaussian mean MLE
3. Linear regression $a^*=\frac{\sum x_iy_i}{\sum x_i^2}$
4. Matrix linear regression $w^*=(X^TX)^{-1}X^Ty$
5. NLL = ERM 유도
6. Gaussian NLL = MSE 유도
7. Categorical NLL = Cross-Entropy 유도
8. Newton's method 2차 근사 유도
9. Softmax derivative
10. Softmax + CE gradient $\frac{\partial L}{\partial z}=p-e_y$
11. 1D convolution 계산과 matrix 구성
12. 2D convolution matrix 구성
13. Output size formula 유도
14. Max pooling non-linearity 반례

### 2순위 - 정의와 비교를 정확히 외울 것

1. Hypothesis space restriction
2. Inductive bias
3. Function hypothesis
4. Linear, nonlinear basis, parametrized feature extractor, neural network의 prior 강도 비교
5. XOR problem의 의미
6. Universal approximation의 의미와 한계
7. GD, SGD, momentum, Adam의 차이
8. Sparse와 weight sharing
9. Locality와 translation invariance
10. Feature detector와 inner product similarity
11. Average pooling과 max pooling 차이
12. CNN과 Transformer prior 비교

### 3순위 - 사례형으로 준비할 것

1. AlexNet layer shape 계산
2. ZFNet feature visualization의 의미
3. Hubel-Wiesel 실험과 feature detector 연결
4. Markov language model과 Transformer 비교
5. Set permutation invariance 예시

---

## 13. 최종 예상 시험지 구성

실제 시험이 "문제 수는 약간 적을 수 있음"이라고 되어 있으므로, 긴 문제 4-6개로 묶일 가능성이 있다. 가장 그럴듯한 구성은 다음과 같다.

### 예상 구성 1 - 확률모델과 ERM

1. Restricted prior의 PDF를 쓰고 MAP을 구하라.
2. Gaussian likelihood에서 MSE를 유도하라.
3. Categorical likelihood에서 cross-entropy를 유도하라.
4. 위 두 loss가 NLL/ERM과 어떻게 연결되는지 설명하라.

### 예상 구성 2 - Optimization과 Backprop

1. Newton's method가 $L$의 2차 근사 minimum임을 보여라.
2. GD와 SGD update rule을 쓰고 차이를 설명하라.
3. Softmax Jacobian을 구하라.
4. $L=-\log p_y$일 때 $\frac{\partial L}{\partial z}=p-e_y$를 유도하라.

### 예상 구성 3 - CNN

1. 1D convolution을 직접 계산하라.
2. 표준기저를 이용해 convolution matrix를 구성하라.
3. 이 matrix에서 sparse와 weight sharing을 표시하라.
4. Convolution이 linear transformation임을 증명하라.
5. Max pooling이 linear가 아님을 반례로 보여라.
6. Output size formula를 유도하고 AlexNet shape을 계산하라.

### 예상 구성 4 - 통합 서술

1. Restricted prior, linear model, CNN을 hypothesis space restriction 관점에서 비교하라.
2. CNN과 Transformer를 inductive bias 관점에서 비교하고 Bitter Lesson과 연결하라.
3. Linear model에서 deep network로 갈수록 prior가 어떻게 변하는지 설명하라.

---

## 14. 마지막 체크리스트

시험 직전에는 아래 질문에 답할 수 있으면 된다.

1. Prior가 0이면 왜 posterior도 0인가?
2. MAP과 MLE는 언제 달라지는가?
3. $\theta$ hypothesis와 function hypothesis $h$의 차이는 무엇인가?
4. Gaussian likelihood에서 왜 squared loss가 나오는가?
5. Categorical likelihood에서 왜 cross-entropy가 나오는가?
6. NLL과 ERM은 왜 같은 optimization 문제인가?
7. Empirical distribution $P_S$를 수식으로 쓸 수 있는가?
8. Linear regression closed form을 미분으로 유도할 수 있는가?
9. Newton update를 2차 Taylor approximation에서 유도할 수 있는가?
10. GD, SGD, Adam의 update intuition을 설명할 수 있는가?
11. Softmax Jacobian을 성분별로 유도할 수 있는가?
12. Softmax + CE에서 gradient가 $p-e_y$가 되는 이유를 설명할 수 있는가?
13. Convolution이 linear transformation인 이유를 additivity와 homogeneity로 보일 수 있는가?
14. 1D convolution matrix를 표준기저 없이도 바로 쓸 수 있는가?
15. 2D convolution matrix에서 어떤 entry가 kernel weight인지 찾을 수 있는가?
16. Sparse와 weight sharing을 matrix에서 표시할 수 있는가?
17. Output size formula를 유도할 수 있는가?
18. Same padding 조건을 유도할 수 있는가?
19. Max pooling이 linear가 아님을 반례로 보일 수 있는가?
20. CNN의 locality와 translation invariance가 왜 inductive bias인지 설명할 수 있는가?

---

## 15. 한 줄 결론

가장 유력한 기말고사 핵심은 "확률모델에서 loss가 나오고, loss를 parameter space에서 optimization하며, backprop으로 gradient를 구하고, CNN은 convolution이라는 제한된 linear transformation으로 inductive bias를 넣는다"는 전체 흐름이다. 이 흐름을 수식으로 유도할 수 있는지 묻는 문제가 가장 많이 나올 가능성이 높다.

---
title: "딥러닝 중간고사 최종 제출본"
slug: 00-midterm-exam-master-final-merged
order: 2
---

# 딥러닝 중간고사 최종 제출본

> 기준: 교재 + 교수님 강의 스크립트 통합본  
> 용도: 중간고사 범위를 빠짐없이 커버하면서도, 교수님이 실제 수업에서 강조한 논리와 표현으로 바로 답안을 구성할 수 있게 만든 최종 1권  
> 범위: Week 1 ~ Week 7, 교재 기준 핵심 페이지 `pp. 033-292`

---

## 0. 이 최종본을 어떻게 읽어야 하나

### 한 줄 결론
이 시험은 "Learning을 이해하려면 Maximum Likelihood와 Prior를 이해해야 한다"는 교수님 프레임 위에, `선형대수 → 미분 → 확률 → 베이즈 → 손실함수 → 역전파`를 연결해서 쓰는 시험이다.

### 이 최종본의 통합 원칙
- `교재 기준 완전 범위`는 [00_midterm_exam_master_guide.md](/Users/jeongseongchae/dev/university/deep_learning/final/deep_learing_prev_exam/codex/00_midterm_exam_master_guide.md)의 장점을 유지
- `교수님 실제 수업 표현`은 [00_midterm_exam_master_guide_v2_script_based.md](/Users/jeongseongchae/dev/university/deep_learning/final/deep_learing_prev_exam/codex/00_midterm_exam_master_guide_v2_script_based.md)의 장점을 유지
- 이 문서는 둘을 합쳐서 `실제 제출/실전 복습용 기준본` 1개로 만든 버전

### 교수님이 실제로 보는 핵심
- Learning = belief update
- posterior를 크게 하는 것이 진짜 목표
- MAP가 상위 개념이고, MLE는 uniform prior를 둔 MAP의 특수한 경우
- Gaussian/Bernoulli가 probability 언어의 핵심 예제
- Gaussian regression에서 NLL이 MSE로 떨어지는 연결을 이해해야 함
- CE, KL, entropy의 연결을 설명할 수 있어야 함
- 답보다 과정, 수식보다 논리

### 시험 답안 공통 프레임
1. 최적화 목표를 먼저 적는다.
2. 필요한 가정(i.i.d., Gaussian, Bernoulli, prior 형태)을 먼저 쓴다.
3. 수식을 한 줄씩 전개한다.
4. 각 줄 사이의 이유를 글로 설명한다.
5. 마지막에 "이 결과가 의미하는 바"를 데이터/가설/학습 관점으로 번역한다.

### 시험 TIP
- 시험 직전 요약에서 많은 문제가 직접 또는 변형 형태로 나온다.
- Summary는 꼭 참석하는 편이 좋다. 힌트가 많다.
- 알고리즘 간 비교 문제가 중요하다.
- Preliminary 파트가 자주 나온다.
- 슬라이드의 증명과 formula가 거의 그대로 출제될 수 있다.
- 큰 문제 5개 + 소문항 여러 개 구조를 예상하는 것이 안전하다.
- 짧게 정의만 쓰면 부분점수가 약하다.
- `정의 → 증명/유도 → 응용` 순서로 답안을 써야 한다.

---

## 1. 교수님 수업의 진짜 메인 줄기

> 스크립트 근거: `1week.txt`, `3week.txt`, `4week.txt`

### 한 줄 결론
교수님은 Learning을 "좋은 hypothesis를 찾는 것"으로 보고, 그 기준을 posterior의 크기로 설명한다.

### 핵심 문장
> Learning은 belief update이고, posterior를 크게 하는 것이 MAP이며, MLE는 uniform prior를 둔 MAP의 특수한 경우다.

### 이 문장을 시험 언어로 번역하면
- hypothesis \(h\) 또는 parameter \(\theta\)가 있다.
- data \(E\)가 들어온다.
- prior \(p(h)\)가 있다.
- likelihood \(p(E\mid h)\)가 있다.
- posterior \(p(h\mid E)\)가 나온다.
- 우리는 posterior가 가장 큰 hypothesis를 고른다.

### 기호 해체

| 기호 | 이름 | 뜻 | 시스템 비유 |
|---|---|---|---|
| \(h\) | hypothesis | 후보 가설/모델 | 운영 정책 후보 |
| \(\theta\) | parameter | 모델 설정값 | 시스템 튜닝값 |
| \(E\) | evidence / data | 관측 데이터 | 서비스 로그 |
| \(p(h)\) | prior | 사전 믿음 | 도메인 지식 |
| \(p(E\mid h)\) | likelihood | 가설이 맞을 때 데이터가 나올 확률 | 정책이 맞을 때 현재 로그가 나올 가능성 |
| \(p(h\mid E)\) | posterior | 데이터 이후 믿음 | 로그 반영 후 정책 신뢰도 |

### 교수님 스타일 핵심
- "왜 independence를 썼는가"
- "왜 identical distribution이 필요한가"
- "왜 log를 취했는가"
- "왜 미분해서 0으로 두는가"
- "uniform prior면 왜 MAP=MlE인가"

### 쉽게 설명하면
러닝은 그냥 숫자 맞추기가 아니라, 가능한 설명 후보들 중에서 "지금 데이터까지 봤을 때 가장 믿을 만한 설명"을 고르는 과정이다.

### 남에게 설명하는 한 문장
"이 수업에서 학습은 데이터를 본 뒤 hypothesis에 대한 믿음을 업데이트해서 posterior가 가장 큰 가설을 고르는 것으로 설명된다."

---

## 2. 범위 전체 지도

### 범위 커버리지 매트릭스

| 주차 | 교재 페이지 | 핵심 개념 | 시험에서 해야 할 것 |
|---|---:|---|---|
| W1 | 033-040 | 수업 로드맵, 표기법, learning/ML/prior 프레임 | 전체 그림 설명 |
| W2 | 041-072 | 선형대수, 고유값, 대각화, SVD | 고유값/고유벡터/랭크/근사 해석 |
| W3 | 073-112 | 미분, Jacobian, Hessian, softmax 미분 | 체인룰, 미분, 역전파 계산 |
| W4 | 153-196 | 확률, 조건부확률, 분포, 기대값/분산 | 분포 계산, 모멘트 계산 |
| W5 | 230-255 | Bayesian Probability, MLE, MAP, KL | posterior, MAP, KL 관계 설명 |
| W6 | 273-285 | classification vs regression, 손실함수 | Gaussian NLL → MSE 유도 |
| W7 | 293-301 | 선형회귀, 정규방정식, 릿지 | OLS, ridge, MAP 연결 |

### 범위 누락 방지 체크리스트
- [x] Uniform
- [x] Normal / Gaussian
- [x] Poisson
- [x] 기대값 / 분산 / 모멘트
- [x] Bernoulli
- [x] Bayes theorem
- [x] MLE / MAP
- [x] prior \( \theta^m \), \( \theta^m(1-\theta)^m \)
- [x] MSE / NLL / CE / KL
- [x] KL 비음수
- [x] Gaussian KL
- [x] 고유값 / 대각화 / quadratic form
- [x] Jacobian / chain rule / backprop
- [x] softmax Jacobian
- [x] learning rate 조건
- [x] average pooling

---

## 3. 확률의 언어

> 교재 근거: `page_153`, `page_156`, `page_160`  
> 스크립트 근거: `3week.txt`

### 한 줄 결론
확률의 언어를 모르면 MLE도 MAP도 못 쓴다.

### sample space와 event
- sample space: 가능한 모든 outcome의 집합
- event: sample space의 부분집합

### 조건부 확률
\[
P(E\mid F)=\frac{P(E\cap F)}{P(F)}
\]

### 중1 설명
- 전체 경우의 수를 보던 걸,
- 이제 "F가 이미 일어났다"는 조건 안에서 다시 계산하는 것이다.

### 기호 해체

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(E\) | 관심 사건 | 우리가 알고 싶은 일 |
| \(F\) | 조건 사건 | 이미 일어났다고 가정한 일 |
| \(E\cap F\) | 교집합 | 둘 다 일어나는 경우 |
| \(P(F)\) | 새 표본공간 크기 | 조건 후 분모 |

### 알고리즘 순서
1. 원래 sample space를 본다.
2. 조건 \(F\)가 참이라고 선언한다.
3. 표본공간을 \(F\)로 줄인다.
4. 그 안에서 \(E\cap F\) 비율을 계산한다.

### 핵심:
- conditional probability는 sample space 축소다.

---

## 4. 교수님이 계속 쓰는 두 분포: Bernoulli와 Gaussian

> 스크립트 근거: `2week.txt`, `3week.txt`, `4week.txt`

### 한 줄 결론
Bernoulli는 classification intuition의 출발점이고, Gaussian은 regression/MSE intuition의 출발점이다.

### 4-1. Bernoulli Distribution

\[
P(X=x\mid\theta)=\theta^x(1-\theta)^{1-x},\qquad x\in\{0,1\}
\]

#### 중1 설명
- 결과가 둘뿐인 분포다.
- 동전 앞/뒤, 성공/실패, fire/not fire 같은 경우다.

#### 핵심:
- sample space 크기가 2
- parameter \(\theta\) 하나로 완전히 설명 가능

### 4-2. Gaussian Distribution

\[
\mathcal N(x;\mu,\sigma^2)=
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac12\left(\frac{x-\mu}{\sigma}\right)^2
\right)
\]

#### 교수님 식 핵심 표현
- exponential
- minus
- square

#### 왜 중요한가
- log를 취하면 exponential이 사라진다
- minus를 붙이면 minimization으로 바뀐다
- square 구조가 남아서 MSE가 나온다

#### 핵심:
- Gaussian + log + minus = square loss 구조

---

## 5. 베이즈 정리

> 교재 근거: `page_160`, `page_230`  
> 스크립트 근거: `3week.txt`

### 한 줄 결론
posterior는 likelihood와 prior의 곱에 비례한다.

\[
p(h\mid e)=\frac{p(e\mid h)p(h)}{p(e)}
\]

### 각 항의 의미
- \(e\): evidence / data
- \(h\): hypothesis
- \(p(h)\): prior
- \(p(e\mid h)\): likelihood
- \(p(h\mid e)\): posterior
- \(p(e)\): \(h\)에 대해 상수

### Line-by-Line
\[
p(h\mid e)=\frac{p(e\cap h)}{p(e)}
\]
- 조건부 확률 정의

\[
p(e\mid h)=\frac{p(e\cap h)}{p(h)}
\Rightarrow p(e\cap h)=p(e\mid h)p(h)
\]
- 교집합을 likelihood와 prior로 다시 쓴다

\[
p(h\mid e)=\frac{p(e\mid h)p(h)}{p(e)}
\]

### 로그 버전
\[
\log p(h\mid e)=\log p(e\mid h)+\log p(h)-\log p(e)
\]

### 핵심:
- Posterior ∝ Likelihood × Prior

### 주의:
- \(p(e)\)는 정규화 상수다.
- argmax에서는 보통 무시된다.

---

## 6. MLE: 데이터만 믿는 방법

> 교재 근거: `page_242`, `page_245`  
> 스크립트 근거: `3week.txt`, `4week.txt`

### 한 줄 결론
MLE는 데이터를 가장 잘 설명하는 parameter를 찾는 방법이고, 데이터가 적을 때는 극단적으로 흔들릴 수 있다.

### Goal
관측 데이터만 보고 \(\theta\)를 추정하는 것.

### Tool / Algorithm
\[
\hat\theta_{\text{MLE}}=\arg\max_\theta p(\text{data}\mid \theta)
\]

### 왜 i.i.d.가 중요한가
- independence: 곱으로 분해하기 위해
- identically distributed: 모두 같은 \(\theta\)에서 나왔음을 보장하기 위해

### Line-by-Line: Bernoulli MLE

데이터: \(n\)번 시행, \(k\)번 성공

\[
p(\text{data}\mid\theta)=\theta^k(1-\theta)^{n-k}
\]
- i.i.d.라서 각 샘플 확률의 곱

\[
\log L(\theta)=k\log\theta+(n-k)\log(1-\theta)
\]
- 곱을 합으로 바꿔서 미분 쉽게 함

\[
\frac{d}{d\theta}\log L(\theta)=\frac{k}{\theta}-\frac{n-k}{1-\theta}
\]

\[
\frac{k}{\theta}-\frac{n-k}{1-\theta}=0
\]
- 극값 후보

\[
k(1-\theta)=(n-k)\theta
\]

\[
k=n\theta
\Rightarrow
\hat\theta_{\text{MLE}}=\frac{k}{n}
\]

### 교수님이 실제로 강조한 문제점
- \(n=3, k=3\)이면
\[
\hat\theta_{\text{MLE}}=1
\]
- 즉 "이 동전은 계속 앞면만 나온다"는 너무 극단적인 결론

### 백엔드/시스템 비유
- 로그 3개만 보고 운영 규칙을 "이 패턴만 영원히 온다"로 고정하는 것과 비슷하다.

### 시험장에서 써야 하는 문장
- "i.i.d. 가정에 의해 likelihood를 곱으로 쓸 수 있습니다."
- "곱을 합으로 바꾸기 위해 log를 취합니다."
- "극값 후보를 찾기 위해 미분하여 0으로 둡니다."

### 핵심:
- MLE는 데이터 중심
- 데이터가 많을 때 강력

### 주의:
- 적은 데이터에서 과적합 가능

---

## 7. MAP: prior를 반영한 진짜 목표

> 교재 근거: `page_242`, `page_245`  
> 스크립트 근거: `3week.txt`, `4week.txt`

### 한 줄 결론
교수님 관점에서는 "우리가 진짜 하고 싶은 것"은 posterior를 최대화하는 MAP다.

### Tool / Algorithm
\[
\hat\theta_{\text{MAP}}=\arg\max_\theta p(\theta\mid \text{data})
\]

\[
\log p(\theta\mid \text{data})
=
\log p(\text{data}\mid \theta)+\log p(\theta)+C
\]

### 교수님 핵심 문장
- MLE는 prior가 없는 게 아니라 uniform prior를 둔 MAP다.
- 모든 학습은 본질적으로 MAP 관점에서 보면 자연스럽다.

### 7-1. Uniform prior

\[
p(\theta)=\text{constant}
\Rightarrow
\log p(\theta)=\text{constant}
\]

따라서
\[
\arg\max_\theta \log p(\theta\mid \text{data})
=
\arg\max_\theta \log p(\text{data}\mid \theta)
\]

\[
\text{MAP}=\text{MLE}
\]

### 7-2. Informative prior \(p(\theta)\propto \theta(1-\theta)\)

\[
\log p(\theta)=\log\theta+\log(1-\theta)+C
\]

\[
\log p(\theta\mid \text{data})
=(k+1)\log\theta+(n-k+1)\log(1-\theta)+C
\]

\[
\frac{k+1}{\theta}-\frac{n-k+1}{1-\theta}=0
\]

\[
(k+1)(1-\theta)=(n-k+1)\theta
\]

\[
k+1=(n+2)\theta
\]

\[
\hat\theta_{\text{MAP}}=\frac{k+1}{n+2}
\]

예를 들어 \(n=3,k=3\)이면
\[
\hat\theta_{\text{MAP}}=\frac45=0.8
\]

MLE의 1보다 덜 극단적이다.

### prior update 해석
- prior \(p(\theta)\)와 likelihood \(p(\text{data}\mid\theta)\)를 곱해서 posterior \(p(\theta\mid\text{data})\)를 만든다.
- 즉 기존 믿음이 데이터 관측 후 믿음으로 업데이트된다.
- MAP는 업데이트된 posterior에서 최고점만 뽑는 것이다.

### 7-3. Strong prior \(p(\theta)\propto \theta^m(1-\theta)^m\)

\[
\log p(\theta)=m\log\theta+m\log(1-\theta)+C
\]

\[
\log p(\theta\mid \text{data})
=(k+m)\log\theta+(n-k+m)\log(1-\theta)+C
\]

\[
\frac{k+m}{\theta}-\frac{n-k+m}{1-\theta}=0
\]

\[
(k+m)(1-\theta)=(n-k+m)\theta
\]

\[
k+m=(n+2m)\theta
\]

\[
\hat\theta_{\text{MAP}}=\frac{k+m}{n+2m}
\]

\[
m\to\infty \Rightarrow \hat\theta_{\text{MAP}}\to \frac12
\]

### 의미
- prior를 강하게 주면 hypothesis space가 좁아진다
- expressivity가 낮아진다
- inductive bias가 강해진다

### 핵심:
- MAP = likelihood + prior
- uniform prior면 MAP = MLE
- strong MAP는 데이터보다 knowledge를 더 믿는 방식

### 주의:
- MAP가 항상 더 좋은 건 아니다
- 문제마다 data vs knowledge trade-off가 다르다
- \(\theta\in[0,1]\) 경계와 미분 불가능/정의 불가 지점을 반드시 확인해야 한다
- \(\log \theta\), \(\log(1-\theta)\)가 등장하면 \(\theta=0,1\)은 별도 체크가 필요하다

---

## 8. Knowledge vs Data Trade-off

> 스크립트 근거: `4week.txt`

### 한 줄 결론
교수님이 실제로 강조한 비교는 "MLE vs MAP 공식 차이"보다 `Data reliance vs Prior knowledge reliance`다.

| 관점 | ML | Strong MAP |
|---|---|---|
| 의존성 | 데이터 | prior knowledge |
| hypothesis space | 넓음 | 좁음 |
| expressivity | 높음 | 낮음 |
| inductive bias | 약함 | 강함 |
| 데이터 적을 때 | 불리할 수 있음 | 유리할 수 있음 |
| 데이터 많을 때 | 유리 | 불리할 수 있음 |

### 모델 연결
- Linear model: 비교적 강한 prior
- CNN: 이미지 특화 prior
- Transformer: 더 약한 prior, 더 많은 데이터 필요

### 핵심:
- prior를 많이 준다는 말 = inductive bias를 많이 준다는 말
- hypothesis space를 줄인다는 말 = 표현 가능한 영역을 줄인다는 말

### 남에게 설명하는 한 문장
"MLE와 MAP 차이는 공식을 하나 더 더하는 정도가 아니라, 데이터를 더 믿을지 지식을 더 믿을지의 설계 철학 차이다."

---

## 9. Uniform / Gaussian / Poisson / 기대값 / 모멘트

> 교재 근거: `page_153`, `page_156`, `page_173`~`page_180`, `page_196`

### 9-1. Uniform

\[
f_X(x)=
\begin{cases}
\frac1{b-a}, & a\le x\le b\\
0, & \text{otherwise}
\end{cases}
\]

\[
E[X]=\int_a^b x\cdot \frac1{b-a}\,dx=\frac{a+b}{2}
\]

\[
\operatorname{Var}(X)=\frac{(b-a)^2}{12}
\]

#### 적분 기호 해체

| 기호 | 의미 | 비유 |
|---|---|---|
| \(\int_a^b\) | \(a\)부터 \(b\)까지 다 훑어 더함 | for loop |
| \(x\) | 현재 값 | 현재 레코드 |
| \(f(x)\) | 가중치 | weight |
| \(dx\) | 미세 간격 | micro-step |

### 9-2. Gaussian

\[
\mathcal N(x;\mu,\sigma^2)=
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac12\left(\frac{x-\mu}{\sigma}\right)^2
\right)
\]

#### 평균과 분산
\[
E[X]=\mu,\qquad \operatorname{Var}(X)=\sigma^2
\]

표준정규 \(Z\sim \mathcal N(0,1)\)에 대해
- \(E[Z]=0\)
- \(E[Z^{2n-1}]=0\)
- \(E[Z^{2n}]=(2n-1)!!\)
- \(E[Z^2]=1\)
- \(E[Z^4]=3\)

### 9-3. Poisson

\[
P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!}
\]

- count 데이터 모델
- 평균 = 분산 = \(\lambda\)

### 9-4. pdf와 기대값 계산 템플릿

\[
E[X]=\int_{-\infty}^{\infty} x f_X(x)\,dx
\]

\[
E[g(X)]=\int_{-\infty}^{\infty} g(x) f_X(x)\,dx
\]

#### 계산 순서
1. pdf \(f_X(x)\)를 확인한다.
2. 적분 구간을 정확히 잡는다.
3. \(x f_X(x)\) 또는 \(g(x)f_X(x)\)를 만든다.
4. 적분해 기대값을 구한다.

### 9-5. 여러 분포 평균 비교

| 분포 | 평균 | 분산 |
|---|---|---|
| Uniform \([a,b]\) | \((a+b)/2\) | \((b-a)^2/12\) |
| Normal \(\mathcal N(\mu,\sigma^2)\) | \(\mu\) | \(\sigma^2\) |
| Poisson \(\text{Poi}(\lambda)\) | \(\lambda\) | \(\lambda\) |

### 핵심:
- Uniform: 모든 값이 같은 밀도
- Gaussian: 평균 근처에 몰림
- Poisson: 이벤트 개수

---

## 10. Gaussian NLL → MSE

> 교재 근거: `page_285`  
> 스크립트 근거: `4week.txt`

### 한 줄 결론
Regression에서 MSE를 쓰는 이유는 Gaussian noise를 가정했을 때 NLL과 같은 문제가 되기 때문이다.

### Tool / Algorithm
\[
y_i=h(x_i)+\varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,\sigma^2)
\]

### Line-by-Line
\[
\text{NLL}(h)=-\sum_{i=1}^n \log p(y_i\mid x_i,h)
\]

\[
=-\sum_{i=1}^n \log \mathcal N(y_i;h(x_i),\sigma^2)
\]

\[
=\sum_{i=1}^n \frac{1}{2\sigma^2}(y_i-h(x_i))^2 + C
\]

\[
=\frac{n}{2\sigma^2}\text{MSE}(h)+C
\]

따라서
\[
\arg\min_h \text{NLL}(h)=\arg\min_h \text{MSE}(h)
\]

### 교수님식 압축
- likelihood를 키운다
- negative log-likelihood를 줄인다
- MSE를 줄인다

이 셋은 같은 문제다.

### 시험장에서 써야 하는 문장
- "Gaussian noise 가정 하에서 negative log-likelihood를 전개하면 squared error 항만 남습니다."
- "상수항은 argmin에 영향을 주지 않으므로 MSE minimization과 동치입니다."

### 핵심:
- Gaussian regression → NLL → MSE

---

## 11. Entropy / Cross Entropy / KL Divergence

> 교재 근거: `page_254`, `page_255`  
> 스크립트 근거: `4week.txt`

### 한 줄 결론
Cross entropy는 KL divergence와 entropy의 합이고, \(P=Q\)일 때 KL이 0이 된다.

\[
H(P)=-\sum_x P(x)\log P(x)
\]

\[
CE(P,Q)=-\sum_x P(x)\log Q(x)
\]

\[
KL(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
\]

### Line-by-Line
\[
KL(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
\]

\[
=\sum_x P(x)\log P(x)-\sum_x P(x)\log Q(x)
\]

\[
=-H(P)+CE(P,Q)
\]

따라서
\[
CE(P,Q)=KL(P\|Q)+H(P)
\]

### 왜 중요한가
- \(H(P)\)는 데이터가 고정되면 상수
- 그래서 CE를 줄이는 것은 KL을 줄이는 것과 같다

### KL 비음수
\[
KL(P\|Q)\ge 0
\]

### Gaussian KL
\[
P=\mathcal N(\mu_1,\sigma^2),\quad
Q=\mathcal N(\mu_2,\sigma^2)
\Rightarrow
KL(P\|Q)=\frac{(\mu_1-\mu_2)^2}{2\sigma^2}
\]

### 일반 KL 계산 템플릿

#### discrete case
\[
KL(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
\]

#### continuous case
\[
KL(P\|Q)=\int p(x)\log\frac{p(x)}{q(x)}\,dx
\]

#### 계산 순서
1. \(P\)와 \(Q\)를 명시한다.
2. support가 같은지 확인한다.
3. \(\log(P/Q)\) 또는 \(\log(p/q)\)를 만든다.
4. \(P\) 또는 \(p\)로 가중하여 합/적분한다.

#### 주의
- \(Q(x)=0\)인데 \(P(x)>0\)이면 KL은 발산할 수 있다.
- KL은 비대칭이다.

### 핵심:
- CE = KL + H
- \(P=Q\)일 때 KL = 0
- 동일 분산 Gaussian에선 KL이 평균 차이 제곱형

---

## 12. 선형대수: 고유값, 대각화, quadratic form

> 교재 근거: `page_055`~`page_061`  
> 스크립트 근거: `2week.txt`

### 한 줄 결론
고유값은 큰 것이 중요하고, 작은 것은 덜 중요하다는 교수님 설명은 사실 "시스템의 중요한 축"을 본다는 뜻이다.

### 고유값 / 고유벡터
\[
Av=\lambda v
\]

\[
\det(A-\lambda I)=0
\]

### 대각화
\[
A=PDP^{-1}
\]

### quadratic form
\[
f(x)=x^TAx
\]

### 왜 중요한가
- 이미지 압축: 큰 eigenvalue부터 남기면 원본을 잘 근사
- PageRank: 중요한 페이지를 고유벡터로 찾음
- 최적화: Hessian 고유값이 학습 안정성 결정

### 핵심:
- 큰 고유값 방향이 중요
- 작은 고유값은 덜 중요
- 대각화는 꼬인 시스템을 독립 엔진으로 분리하는 것

---

## 13. Jacobian, softmax, backprop

> 교재 근거: `page_094`, `page_097`, `page_103`  
> 스크립트 근거: `2week.txt`

### 한 줄 결론
복잡한 함수는 선형으로 근사되고, vector→vector 미분은 Jacobian이며, 그 대표 계산 예제가 softmax 미분이다.

### Jacobian
- vector → vector 미분 결과는 행렬

### Softmax
\[
p_i=\frac{e^{g_i}}{\sum_j e^{g_j}}
\]

\[
\frac{\partial p_i}{\partial g_i}=p_i(1-p_i)
\]

\[
\frac{\partial p_i}{\partial g_j}=-p_ip_j\qquad (i\neq j)
\]

### 행렬 형태
\[
\frac{\partial p}{\partial g}=\operatorname{diag}(p)-pp^T
\]

### softmax의 \(a, W, b\) 미분 템플릿

logit을
\[
g=Wa+b
\]
라고 두면

\[
\frac{\partial L}{\partial a}=W^T\frac{\partial L}{\partial g}
\]

\[
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial g}\, a^T
\]

\[
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial g}
\]

즉 softmax 자체 미분과 affine layer 미분을 chain rule로 연결하면 \(a,W,b\)에 대한 미분이 나온다.

### \(-\log \sigma(Ax+b)\) 미분
\[
L=-\log \sigma(z),\qquad z=Ax+b
\]

\[
\frac{dL}{dz}=\sigma(z)-1
\]

\[
\frac{\partial L}{\partial A}=(\sigma(z)-1)x^T
\]

\[
\frac{\partial L}{\partial x}=A^T(\sigma(z)-1)
\]

\[
\frac{\partial L}{\partial b}=\sigma(z)-1
\]

### 평균/분산 유도 기본식

\[
E[X]=\sum_x x p(x)\quad \text{또는}\quad E[X]=\int x f(x)\,dx
\]

\[
\operatorname{Var}(X)=E[X^2]-(E[X])^2
\]

### backprop 핵심
- forward cache 저장
- 출력 오차부터 역방향 전파
- 각 층 가중치 gradient는 보통 `오차 × 입력` 외적 형태

### 핵심:
- softmax derivative는 꼭 계산 가능해야 함
- Jacobian은 역전파의 핵심 언어

---

## 14. Gradient Descent와 learning rate 조건

> 교재 근거: `page_060`, `page_061`

### 한 줄 결론
GD는 가장 가파른 축의 고유값에 맞춰 보폭을 정해야 안정적이다.

\[
f(w)=\frac12 w^TAw,\qquad \nabla f(w)=Aw
\]

\[
w_{t+1}=w_t-\eta Aw_t=(I-\eta A)w_t
\]

고유축으로 보면
\[
y_{t+1}^{(i)}=(1-\eta\lambda_i)y_t^{(i)}
\]

수렴 조건은
\[
|1-\eta\lambda_i|<1
\]

모든 \(i\)에 대해 성립하려면
\[
0<\eta<\frac{2}{\lambda_{\max}(A)}
\]

### 핵심:
- learning rate 상한은 최대 고유값이 정함

### 주의:
- \(\lambda_{\min}\)가 아니다

### Quadratic 문제에서 자주 쓰는 문장
- "이 함수는 quadratic form이므로 고유벡터 축으로 좌표변환하면 각 방향이 분리됩니다."
- "가장 큰 고유값 방향이 가장 가파르므로 learning rate 상한은 \(\lambda_{\max}\)가 결정합니다."

---

## 15. Average Pooling과 행렬 표현

### 한 줄 결론
pooling도 결국 선형대수로 쓰면 행렬 곱이다.

\[
x=
\begin{bmatrix}
x_1\\x_2\\x_3\\x_4
\end{bmatrix},
\quad
y=
\begin{bmatrix}
\frac{x_1+x_2}{2}\\
\frac{x_3+x_4}{2}
\end{bmatrix}
\]

\[
y=Px,\qquad
P=
\begin{bmatrix}
1/2&1/2&0&0\\
0&0&1/2&1/2
\end{bmatrix}
\]

### 핵심:
- average pooling = local average
- matrix form으로 쓰면 구조가 명확해진다

---

## 16. 선형회귀, 정규방정식, 릿지

> 교재 근거: `page_293`, `page_296`, `page_301`

### 선형회귀
\[
f_w(x)=x^Tw+b
\]

설계 행렬로 쓰면
\[
f_w(X)=Xw
\]

### 정규방정식
\[
L(w)=\frac12\|Xw-y\|^2
\]

\[
\nabla_w L=X^T(Xw-y)=0
\]

\[
X^TX\hat w=X^Ty
\]

### 릿지 회귀
\[
L_\lambda(w)=\frac12\|Xw-y\|^2+\frac12\lambda\|w\|^2
\]

\[
\hat w_\lambda=(X^TX+\lambda I)^{-1}X^Ty
\]

### MAP 연결
- Gaussian prior를 두면 ridge가 MAP로 해석된다.

### 핵심:
- OLS는 데이터 적합만 본다
- ridge는 weight를 작게 유지하는 prior를 넣은 것

---

## 17. 실전 예상문제

### 문제 1. Bernoulli MLE
- \(n,k\)가 주어졌을 때 MLE를 유도하라
- 반드시 i.i.d., log, 미분 이유를 서술하라

### 문제 2. MAP with \(p(\theta)\propto \theta(1-\theta)\)
- posterior를 전개하고 MLE와 비교하라

### 문제 3. MAP with \(p(\theta)\propto \theta^m(1-\theta)^m\)
- \(m\to\infty\) 의미를 strong prior 관점에서 설명하라

### 문제 4. Gaussian NLL → MSE
- Gaussian noise 가정 하 NLL을 전개하라

### 문제 5. \(CE = KL + H\)
- 정의에서 직접 유도하라

### 문제 6. softmax Jacobian
- \(i=j\), \(i\neq j\)를 나누어 유도하라

### 문제 7. \(-\log \sigma(Ax+b)\) 미분
- \(A,x,b\) 각각에 대한 gradient와 차원을 쓰라

### 문제 8. \(0<\eta<2/\lambda_{\max}(A)\)
- quadratic loss에서 GD 수렴 조건을 증명하라

### 문제 9. average pooling matrix form
- pooling을 행렬 \(P\)로 표현하라

### 문제 10. 고유값/대각화
- 2x2 행렬을 대각화하고 의미를 설명하라

### 문제 11. Quadratic 문제
- \(f(x)=x^TAx\) 또는 \(f(w)=\frac12 w^TAw\) 꼴에서
  - gradient를 구하라
  - eigenvalue 관점에서 의미를 설명하라
  - GD 수렴 조건과 연결하라

---

## 18. 시험 직전 백지유도 체크리스트

- [ ] posterior, likelihood, prior를 말로 구분 가능
- [ ] MLE는 uniform prior를 둔 MAP라고 설명 가능
- [ ] Bernoulli MLE를 끝까지 유도 가능
- [ ] \(p(\theta)\propto \theta(1-\theta)\) MAP 가능
- [ ] \(p(\theta)\propto \theta^m(1-\theta)^m\) MAP 가능
- [ ] strong prior의 의미를 hypothesis space / expressivity / inductive bias로 설명 가능
- [ ] Gaussian NLL → MSE를 줄마다 전개 가능
- [ ] Normal distribution의 평균, 분산을 바로 쓸 수 있음
- [ ] \(E[Z^{2n}]\), \(E[Z^{2n-1}]\)를 설명 가능
- [ ] Poisson 평균, 분산을 바로 쓸 수 있음
- [ ] \(CE = KL + H\)를 정의에서 바로 유도 가능
- [ ] 일반 KL 계산 템플릿을 쓸 수 있음
- [ ] softmax 미분을 \(diag(p)-pp^T\)로 쓸 수 있음
- [ ] softmax의 \(a,W,b\) 미분을 chain rule로 연결 가능
- [ ] \(-\log\sigma(Ax+b)\) 미분 가능
- [ ] \(0<\eta<2/\lambda_{\max}(A)\)를 고유축으로 설명 가능
- [ ] average pooling을 행렬로 변환 가능
- [ ] 고유값/고유벡터/대각화 계산 가능
- [ ] quadratic 문제를 eigenvalue 관점으로 설명 가능

---

## 19. 최종 압축

### 쉽게 설명하면
이 시험은 확률 언어로 hypothesis를 세우고, prior와 data를 결합해 learning을 설명하고, Gaussian/Bernoulli에서 손실함수를 끌어내고, 선형대수와 미분으로 학습 계산을 완성하는 시험이다.

### 남에게 설명하는 한 문장
"교수님 수업의 핵심은 learning을 MAP 관점에서 이해하는 것이고, MLE는 그 특수한 경우이며, Gaussian과 Bernoulli를 통해 손실함수와 추정을 연결하는 것이다."

### 핵심 킬러 요약
- Learning = belief update
- posterior maximization = MAP
- MLE = uniform prior를 둔 MAP
- strong prior = strong inductive bias
- Gaussian regression → NLL → MSE
- CE = KL + H
- softmax Jacobian과 backprop은 반드시 계산
- GD 안정성은 \(\lambda_{\max}\)가 결정

### 주의
- 공식만 쓰고 논리를 안 쓰면 감점
- i.i.d. 가정을 안 쓰면 likelihood 전개가 무너짐
- prior를 단순 보정항으로만 쓰지 말고 hypothesis space/inductive bias와 연결해서 써야 함

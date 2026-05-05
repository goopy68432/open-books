---
title: "딥러닝 중간고사 마스터 가이드 v2"
slug: 00-midterm-exam-master-guide-v2-script-based
order: 4
---

# 딥러닝 중간고사 마스터 가이드 v2

> 기준: 교재 + 교수님 실제 강의 스크립트(`/DL/script/1week.txt` ~ `4week.txt`)  
> 목적: 교수님이 실제로 수업에서 반복한 논리, 표현, 강조점을 반영해 중간고사 대비용으로 다시 재구성한 버전  
> 범위: 중간고사 기준 `Week 1 ~ Week 7`의 핵심 축. 특히 실제 강의 스크립트에서 반복된 `Maximum Likelihood`, `Prior`, `MAP`, `belief update`, `hypothesis space`, `inductive bias`, `Gaussian/Bernoulli`, `MSE/NLL/KL`를 중심으로 정리

---

## 0. 교수님 수업의 진짜 메인 줄기

### 한 줄 결론
이 수업의 핵심은 "러닝을 이해하려면 `Maximum Likelihood`와 `Prior` 두 개를 이해해야 한다"는 것이다.

### 스크립트 기준으로 다시 말하면
교수님은 1주차부터 이렇게 프레임을 잡고 있다.

- Learning을 이해하려면 두 가지가 중요하다.
  - `Maximum Likelihood`
  - `Prior`
- 이 둘을 이해하면 다음이 연결된다.
  - Bayesian Probability
  - MAP
  - MSE / NLL
  - KL divergence
  - hypothesis space
  - inductive bias
  - CNN / Transformer 같은 모델이 왜 다른지

즉, 시험은 단순히 공식을 묻는 것이 아니라 아래 문장을 이해했는지를 본다.

> "Learning은 belief update이고, 그 update를 수학적으로 쓰면 posterior가 나오고, posterior를 크게 하는 것이 MAP이며, MLE는 uniform prior를 둔 MAP의 특수한 경우다."

### 교수님 스타일 핵심
- 답보다 과정
- 수식보다 논리
- "independence를 어디서 썼는가", "identically distributed를 어디서 썼는가", "왜 log를 취했는가", "왜 미분해서 0으로 두는가"를 말로 설명해야 함
- 시험 문제는 영어일 수 있지만 채점은 논리 전개를 본다

### 시험 답안 기본 프레임
1. 문제에서 무엇을 최적화하는지 먼저 적는다.
2. 필요한 가정(i.i.d., Gaussian, Bernoulli, prior 형태)을 선언한다.
3. 수식을 한 줄씩 전개한다.
4. 각 줄마다 "왜 이 줄이 되는지"를 써준다.
5. 최종 해의 의미를 데이터/가설/학습 관점으로 번역한다.

---

## 1. 교수님이 보는 Learning의 정의

> 스크립트 근거: `1week.txt`, `4week.txt`

### 한 줄 결론
Learning은 데이터로 인해 hypothesis에 대한 믿음이 업데이트되는 과정이다.

### Goal
"학습"이 단순히 숫자를 맞추는 것이 아니라, hypothesis space 안에서 더 믿을 만한 가설을 찾는 과정임을 이해하는 것.

### Tool / Algorithm
- hypothesis \(h\)
- data \(E\)
- posterior \(p(h\mid E)\)
- MAP: posterior를 최대화

### 중1 설명
- 여러 개의 "가능한 답안"이 있다고 생각하면 된다.
- 데이터를 보기 전에는 뭐가 맞는지 잘 모른다.
- 데이터를 보고 나면 어떤 답안은 더 믿게 되고, 어떤 답안은 덜 믿게 된다.
- 이 믿음이 바뀌는 과정이 학습이다.

### 중2 설명
- 동전이 있다고 하자.
- "이 동전은 공평하다", "이 동전은 앞면이 잘 나온다", "이 동전은 조작됐다" 같은 여러 가설이 있다.
- 실제로 동전을 던져 보면 믿음이 바뀐다.

### 중3 설명
- hypothesis는 후보 모델이다.
- 데이터가 들어오면 후보 모델들의 점수가 바뀐다.
- 점수가 제일 높은 후보를 고르는 것이 학습의 핵심이다.

### 고1 설명
- 출력 \(y\)를 예측하는 함수 \(h:X\to Y\)를 생각한다.
- 학습은 가능한 함수들 중에서 가장 좋은 하나를 찾는 것이다.

### 고2 설명
- "가장 좋다"는 건 보통 posterior가 가장 큰 것을 뜻한다.
\[
h^*=\arg\max_h p(h\mid E)
\]

### 고3 설명
- 베이지안 관점에서 probability는 단순히 사건의 비율이 아니라 "belief의 정도"로도 해석된다.
- 즉 learning은 belief update다.

### 대학수학 연결
- Bayesian learning
- posterior inference
- inductive bias selection

### 기호 해체

| 기호 | 이름 | 뜻 | 시스템 비유 |
|---|---|---|---|
| \(h\) | hypothesis | 후보 모델 | 운영 정책 버전 |
| \(E\) | evidence / data | 관측 데이터 | 서비스 로그 |
| \(p(h)\) | prior | 사전 믿음 | 배포 전 도메인 지식 |
| \(p(h\mid E)\) | posterior | 데이터 후 믿음 | 로그 반영 후 정책 신뢰도 |

### 백엔드/시스템 설계 비유
- prior: "이 서비스는 원래 이런 패턴일 것이다"라는 운영자의 초기 가정
- data: 실제 관측된 요청/오류 로그
- posterior: 로그를 본 뒤 갱신된 운영 판단

### 핵심:
- learning = belief update
- 좋은 hypothesis = posterior가 큰 hypothesis

### 주의:
- frequentist probability와 Bayesian probability를 섞어 쓰면 안 된다.
- \(p(h)\)는 "가설에 대한 믿음"이다. 이벤트 빈도와 다른 개념이다.

### 남에게 설명하는 한 문장
"학습은 후보 가설들에 대한 믿음을 데이터로 업데이트해서 가장 믿을 만한 가설을 고르는 과정이다."

---

## 2. Frequentist vs Bayesian

> 스크립트 근거: `3week.txt`

### 한 줄 결론
Frequentist는 사건의 상대 빈도를 보고, Bayesian은 hypothesis에 대한 믿음의 정도를 본다.

### Goal
왜 \(p(\theta)\)가 "동전 파라미터에 대한 믿음"으로 해석될 수 있는지 이해하는 것.

### Tool / Algorithm
- Frequentist: event probability
- Bayesian: hypothesis belief

### 중1 설명
- Frequentist: "앞면이 얼마나 자주 나오나?"
- Bayesian: "이 동전이 공평할 것 같나?"

### 중2 설명
- 둘 다 probability라는 말을 쓰지만 대상이 다르다.

### 중3 설명
- Frequentist는 데이터 생성 빈도를 본다.
- Bayesian은 가설의 신뢰도를 본다.

### 고1 설명
- \(p(\theta)\)는 사건이 아니라 parameter hypothesis에 대한 분포일 수 있다.

### 핵심:
- Bayesian에서는 \(\theta\)도 random object처럼 다룬다.

### 주의:
- \(p(\theta)\)를 event frequency처럼 해석하면 안 된다.

---

## 3. 확률의 언어: sample space, event, conditional probability

> 스크립트 근거: `3week.txt`

### 한 줄 결론
확률의 언어를 모르면 베이즈도, MLE도, MAP도 이해할 수 없다.

### Goal
MLE/MAP 전개에 필요한 가장 기초 문법을 확보하는 것.

### 핵심 개념
- sample space: 가능한 outcome의 전체 집합
- event: sample space의 부분집합
- conditional probability: 조건이 주어졌을 때 줄어든 sample space 안에서 다시 계산한 확률

### 조건부 확률 공식
\[
P(E\mid F)=\frac{P(E\cap F)}{P(F)}
\]

### 중1 설명
- 전체 경우의 수 중에서 보던 걸,
- 이제 "F가 이미 일어났다"는 상황 안에서 다시 계산하는 것이다.

### 기호 해체

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(E\) | event | 관심 사건 |
| \(F\) | condition event | 조건 사건 |
| \(E\cap F\) | 교집합 | 둘 다 일어남 |
| \(P(F)\) | 조건 집합 크기 | 새 표본공간 크기 |

### 알고리즘 순서
1. 원래 sample space를 본다.
2. 조건 \(F\)가 참이라고 선언한다.
3. 표본공간을 \(F\)로 줄인다.
4. 그 안에서 \(E\cap F\)의 비율을 계산한다.

### 핵심:
- conditional probability는 sample space 축소다.

---

## 4. Bernoulli와 Gaussian: 교수님이 반복한 두 분포

> 스크립트 근거: `2week.txt`, `3week.txt`, `4week.txt`

### 한 줄 결론
교수님은 probability 파트에서 특히 Bernoulli와 Gaussian을 중심으로 사고하라고 반복했다.

### 왜 이 둘이 중요한가
- Bernoulli: classification과 coin-flip intuition의 출발점
- Gaussian: regression, MSE, NLL, CLT의 출발점

---

### 4-1. Bernoulli Distribution

#### 한 줄 결론
베르누이는 sample space 크기가 2인 가장 단순하고 중요한 분포다.

#### Goal
MLE, MAP, Bayesian update를 가장 단순한 예제로 설명하는 것.

#### Tool / Algorithm
\[
P(X=x\mid \theta)=\theta^x(1-\theta)^{1-x},\qquad x\in\{0,1\}
\]

#### 중1 설명
- 앞면/뒷면, 성공/실패, fire/not fire처럼 결과가 둘뿐인 분포다.

#### 중2 설명
- \(\theta\) 하나만 알면 완전히 설명된다.

#### 중3 설명
- \(x=1\)이면 \(\theta\)
- \(x=0\)이면 \(1-\theta\)

#### 기호 해체

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(X\) | 랜덤변수 | 동전 결과 |
| \(x\) | 실제 값 | 0 또는 1 |
| \(\theta\) | parameter | 앞면 확률 |

#### 핵심:
- 가장 단순한 random distribution
- MLE/MAP 연습의 기본 모델

---

### 4-2. Gaussian Distribution

#### 한 줄 결론
가우시안은 "exponential, minus, square" 구조 때문에 log를 취하면 제곱오차 구조가 남는다.

#### Goal
왜 regression과 MSE가 Gaussian으로부터 나오는지 이해하는 것.

#### Tool / Algorithm
\[
\mathcal N(x;\mu,\sigma^2)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac12\left(\frac{x-\mu}{\sigma}\right)^2
\right)
\]

#### 교수님 표현 그대로 핵심
- Exponential
- Minus
- Square

#### 왜 중요하냐
- log를 취하면 exponential이 사라진다
- minus를 붙이면 minimization 문제로 바뀐다
- square만 남아서 MSE와 연결된다

#### 중1 설명
- 평균 근처가 가장 가능성이 높고, 멀수록 급격히 덜 가능하다.

#### 중2 설명
- 멀어진 정도를 제곱해서 벌점으로 준다.

#### 핵심:
- Gaussian + log + minus = square loss

---

## 5. Bayes Theorem

> 스크립트 근거: `3week.txt`

### 한 줄 결론
베이즈 정리는 posterior가 likelihood와 prior의 곱에 비례한다는 사실이다.

### Tool / Algorithm
\[
p(h\mid e)=\frac{p(e\mid h)p(h)}{p(e)}
\]

### 각 항의 의미
- \(e\): 데이터, observation, evidence
- \(h\): hypothesis
- \(p(h)\): prior
- \(p(e\mid h)\): likelihood
- \(p(h\mid e)\): posterior
- \(p(e)\): 정규화 상수

### Line-by-Line 주석
\[
p(h\mid e)=\frac{p(e\cap h)}{p(e)}
\]
- 조건부 확률 정의

\[
p(e\mid h)=\frac{p(e\cap h)}{p(h)}
\Rightarrow
p(e\cap h)=p(e\mid h)p(h)
\]
- 다시 조건부 확률 정의를 뒤집어 씀

\[
p(h\mid e)=\frac{p(e\mid h)p(h)}{p(e)}
\]
- 교집합 표현을 대입

### 로그 버전
\[
\log p(h\mid e)=\log p(e\mid h)+\log p(h)-\log p(e)
\]

### 핵심:
- Posterior ∝ Likelihood × Prior

### 주의:
- \(p(e)\)는 \(h\)에 대해 상수다.
- argmax에서는 보통 무시할 수 있다.

---

## 6. MLE: 데이터만 믿는 방법

> 스크립트 근거: `3week.txt`, `4week.txt`

### 한 줄 결론
MLE는 데이터를 가장 잘 설명하는 parameter를 찾는 방법이고, 데이터가 적을 때는 너무 극단적일 수 있다.

### Goal
동전 예제에서 데이터 비율 \(k/n\)이 왜 MLE가 되는지 완전히 이해하는 것.

### Tool / Algorithm
\[
\hat\theta_{\text{MLE}}=\arg\max_\theta p(\text{data}\mid \theta)
\]

### 왜 i.i.d.가 중요한가
- independence: 곱으로 분해하기 위해
- identically distributed: 모두 같은 \(\theta\)에서 나왔다는 가정

### Line-by-Line: Bernoulli MLE

데이터: \(n\)번 던져서 \(k\)번 앞면

\[
p(\text{data}\mid \theta)=\theta^k(1-\theta)^{n-k}
\]
- 독립이라서 곱
- 같은 \(\theta\)라서 동일한 형태 반복

\[
\log L(\theta)=k\log\theta+(n-k)\log(1-\theta)
\]
- 곱을 합으로 변환

\[
\frac{d}{d\theta}\log L(\theta)=\frac{k}{\theta}-\frac{n-k}{1-\theta}
\]

\[
\frac{k}{\theta}-\frac{n-k}{1-\theta}=0
\]
- 극값 후보 찾기

\[
k(1-\theta)=(n-k)\theta
\]

\[
k=n\theta
\]

\[
\hat\theta_{\text{MLE}}=\frac{k}{n}
\]

### 중1 설명
- 관측된 앞면 비율이 곧 가장 그럴듯한 확률이다.

### 교수님이 실제로 강조한 문제점
- \(n=3, k=3\)이면
\[
\hat\theta_{\text{MLE}}=1
\]
- 즉 "앞으로도 계속 앞면만 나올 것"처럼 너무 극단적으로 예측한다.

### 백엔드/시스템 설계 비유
- 로그 3개만 보고 운영 정책을 즉시 "항상 이 패턴만 온다"로 바꾸는 것과 비슷하다.
- 데이터가 적으면 과적합된 운영 판단이 된다.

### 핵심:
- MLE는 데이터에 강하게 의존한다.
- 데이터가 많을 때는 강력하다.

### 주의:
- 데이터가 적을 때 과적합 위험

### 남에게 설명하는 한 문장
"MLE는 관측된 데이터 비율을 그대로 믿는 방법이라 적은 데이터에서는 극단적인 결론으로 치우칠 수 있다."

---

## 7. MAP: prior를 반영한 진짜 목표

> 스크립트 근거: `3week.txt`, `4week.txt`

### 한 줄 결론
교수님 관점에서 "우리가 진짜 하고 싶은 것"은 posterior를 최대화하는 MAP이고, MLE는 그 특수한 경우다.

### Goal
MAP가 MLE보다 더 상위 개념이라는 교수님 프레임을 이해하는 것.

### Tool / Algorithm
\[
\hat\theta_{\text{MAP}}=\arg\max_\theta p(\theta\mid \text{data})
\]

\[
\log p(\theta\mid \text{data})
=
\log p(\text{data}\mid \theta)+\log p(\theta)+C
\]

### 교수님 표현 핵심
- "MLE는 prior가 없는 게 아니라 uniform prior를 둔 MAP다."
- "모든 학습은 본질적으로 MAP로 생각하는 게 자연스럽다."

---

### 7-1. Uniform Prior의 경우

\[
p(\theta)=\text{constant}
\]

그러면
\[
\log p(\theta)=\text{constant}
\]

그래서
\[
\arg\max_\theta \log p(\theta\mid \text{data})
=
\arg\max_\theta \left[
\log p(\text{data}\mid \theta)+\text{constant}
\right]
\]

\[
=\arg\max_\theta \log p(\text{data}\mid \theta)
\]

즉
\[
\text{MAP}=\text{MLE}
\]

### 핵심:
- MLE는 "prior 없음"이 아니라 "uniform prior"

---

### 7-2. Informative Prior \(p(\theta)\propto \theta(1-\theta)\)

\[
\log p(\theta)=\log\theta+\log(1-\theta)+C
\]

posterior log는
\[
\log p(\theta\mid \text{data})
=
k\log\theta+(n-k)\log(1-\theta)
\]
\[
\quad +\log\theta+\log(1-\theta)+C
\]

\[
=(k+1)\log\theta+(n-k+1)\log(1-\theta)+C
\]

미분하면
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

### 예시
\[
n=3,\quad k=3
\]

\[
\hat\theta_{\text{MLE}}=1,\qquad
\hat\theta_{\text{MAP}}=\frac45=0.8
\]

### 의미
- MLE의 1보다 덜 극단적
- 데이터와 prior를 함께 고려

---

### 7-3. Strong Prior \(p(\theta)\propto \theta^m(1-\theta)^m\)

#### 한 줄 결론
strong MAP는 "동전은 0.5 근처일 것"이라는 믿음을 강하게 넣는 방식이다.

#### Tool / Algorithm
\[
p(\theta)\propto \theta^m(1-\theta)^m
\]

#### 로그 prior
\[
\log p(\theta)=m\log\theta+m\log(1-\theta)+C
\]

#### posterior log
\[
\log p(\theta\mid \text{data})
=(k+m)\log\theta+(n-k+m)\log(1-\theta)+C
\]

#### 미분
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

#### \(m\to\infty\) 해석
\[
\hat\theta_{\text{MAP}}\to \frac12
\]

### 교수님 식 해석
- prior knowledge를 엄청 크게 준 것
- 데이터 weight를 거의 무시하는 것
- \(1/2\) 근처만 가능한 hypothesis space로 제한한 것

### 중1 설명
- "나는 이 동전이 무조건 거의 공평하다고 믿어"를 엄청 강하게 넣은 상태다.

### 시스템 비유
- 운영 정책이 너무 강경해서, 로그가 아무리 들어와도 기존 가정을 거의 안 바꾸는 상태

### 핵심:
- strong MAP는 데이터 적을 때 유리할 수 있다
- 데이터 많을 때는 실제 패턴을 못 따라갈 수 있다

### 주의:
- MAP가 무조건 MLE보다 좋은 것은 아니다
- 문제마다 trade-off가 다르다

---

## 8. Knowledge vs Data Trade-off

> 스크립트 근거: `4week.txt`

### 한 줄 결론
교수님이 실제로 강조한 비교는 "MLE vs MAP" 자체보다 `Data reliance vs Prior knowledge reliance`다.

### 교수님 비교표를 시험형으로 다시 정리

| 관점 | ML | Strong MAP |
|---|---|---|
| 의존성 | 데이터 | prior knowledge |
| hypothesis space | 넓음 | 좁음 |
| expressivity | 높음 | 낮음 |
| inductive bias | 약함 | 강함 |
| 데이터 적을 때 | 불리할 수 있음 | 유리할 수 있음 |
| 데이터 많을 때 | 유리 | 불리할 수 있음 |

### 한 줄로 외우기
- ML: 넓고 자유롭고 데이터 중심
- Strong MAP: 좁고 제한적이고 지식 중심

### 왜 이게 딥러닝 모델로 이어지나
- Linear model: 강한 prior
- CNN: 이미지 특화 prior
- Transformer: 약한 prior, 데이터 대량 필요

### 핵심:
- prior를 많이 준다는 말 = inductive bias가 강하다는 말
- hypothesis space를 줄인다는 말 = expressivity를 줄인다는 말

### 남에게 설명하는 한 문장
"MLE와 MAP의 차이는 단순한 공식 차이가 아니라, 데이터를 더 믿을지 기존 지식을 더 믿을지의 시스템 설계 선택이다."

---

## 9. Gaussian, CLT, Regression, MSE

> 스크립트 근거: `4week.txt`

### 한 줄 결론
Regression에서 MSE를 쓰는 이유는 noise를 Gaussian으로 가정했을 때 NLL이 MSE와 같은 꼴로 떨어지기 때문이다.

### Goal
교수님이 실제로 설명한 "Gaussian distribution과 regression" 연결을 시험 답안 수준으로 정리하는 것.

### Tool / Algorithm
\[
y=h(x)+\varepsilon,\qquad \varepsilon\sim \mathcal N(0,\sigma^2)
\]

### CLT 직관
- 많은 독립 요인의 합은 정규분포에 가까워진다.
- 그래서 실제 오차를 Gaussian으로 두는 가정이 자주 등장한다.

### NLL 전개
\[
p(y_i\mid x_i,h)=\mathcal N(y_i;h(x_i),\sigma^2)
\]

\[
\text{NLL}(h)
=-\sum_{i=1}^n \log p(y_i\mid x_i,h)
\]

가우시안 로그를 열면
\[
\text{NLL}(h)=
\sum_{i=1}^n \frac{1}{2\sigma^2}(y_i-h(x_i))^2 + C
\]

\[
=\frac{n}{2\sigma^2}\text{MSE}(h)+C
\]

### 교수님 스타일 한 줄 해석
- likelihood를 키우는 것
- negative log-likelihood를 줄이는 것
- MSE를 줄이는 것

이 세 개는 같은 문제다.

### 핵심:
- Gaussian noise
- log likelihood
- minus
- square
- MSE

### 주의:
- regression과 classification을 섞지 말 것
- 여기서는 regression 쪽 설명이다

---

## 10. Entropy, Cross Entropy, KL Divergence

> 스크립트 근거: `4week.txt`

### 한 줄 결론
Cross entropy는 KL divergence와 entropy의 합이며, \(P=Q\)일 때 KL이 0이 된다.

### Goal
교수님이 말한 "데이터 분포와 가장 가까운 hypothesis를 찾는다"는 말을 수식으로 이해하는 것.

### Tool / Algorithm
\[
CE(P,Q)=KL(P\|Q)+H(P)
\]

### 기호 해체

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(P\) | 데이터 분포 | 실제 세계 |
| \(Q\) | 모델 분포 | 우리가 세운 hypothesis |
| \(H(P)\) | entropy | 데이터 자체의 불확실성 |
| \(KL(P\|Q)\) | KL divergence | 모델이 데이터와 얼마나 다른가 |

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
- 따라서 CE를 줄이는 것은 KL을 줄이는 것과 같다
- 즉 모델 분포를 데이터 분포에 가깝게 만드는 것

### 핵심:
- \(P=Q\)이면 KL = 0
- KL은 두 분포 간 차이 측정

### 주의:
- KL은 대칭이 아니다

---

## 11. Classification vs Regression

> 스크립트 근거: `4week.txt`

### 한 줄 결론
분류와 회귀의 차이는 출력 \(y\)의 성질, 즉 이산이냐 연속이냐에 있다.

### Goal
문제 타입에 따라 어떤 분포/손실이 자연스러운지 구분하는 것.

### Tool / Algorithm
- Classification: \(y\)가 discrete
- Regression: \(y\)가 continuous

### 스크립트 기준 연결
- Bernoulli → classification intuition
- Gaussian → regression intuition

### 핵심:
- classification은 category
- regression은 real-valued output

---

## 12. 선형대수와 미분이 왜 앞에서 필요한가

> 스크립트 근거: `1week.txt`, `2week.txt`

### 한 줄 결론
교수님은 선형대수와 미적분을 "함수를 이해하기 위한 준비 작업"으로 본다.

### 핵심 흐름
- Intelligence를 함수로 본다
- 복잡한 함수는 선형으로 근사한다
- 선형 근사는 matrix로 표현된다
- vector → vector 미분은 Jacobian이다
- AI는 matrix multiplication + 조금의 nonlinearity라고 봐도 된다

### 꼭 알아야 할 선형대수 포인트
- basis
- dimension
- rank-nullity
- eigenvalue / eigenvector
- SVD

### 꼭 알아야 할 미적분 포인트
- gradient
- Jacobian
- chain rule
- softmax derivative

---

## 13. Softmax 미분

> 스크립트 근거: `2week.txt`

### 한 줄 결론
softmax는 점수 벡터를 확률분포로 바꾸는 함수이며, 미분은 self-term과 cross-term으로 나뉜다.

### Tool / Algorithm
\[
p_i=\frac{e^{g_i}}{\sum_j e^{g_j}}
\]

### 결과
\[
\frac{\partial p_i}{\partial g_i}=p_i(1-p_i)
\]

\[
\frac{\partial p_i}{\partial g_j}=-p_ip_j \qquad (i\neq j)
\]

### 교수님 식 해석
- softmax는 벡터를 확률분포로 바꾼다
- attention, image classification 등에 널리 쓰인다
- 벡터→벡터 미분도 계산 가능하다는 대표 시연 예제다

### 핵심:
- diagonal: \(p_i(1-p_i)\)
- off-diagonal: \(-p_ip_j\)

---

## 14. 시험장에서 바로 써야 하는 문장들

### MLE 문제에서
- "i.i.d. 가정에 의해 likelihood를 각 샘플 likelihood의 곱으로 쓸 수 있습니다."
- "곱을 합으로 바꾸고 미분 계산을 쉽게 하기 위해 log를 취합니다."
- "극값 후보를 찾기 위해 log-likelihood를 미분하여 0으로 둡니다."

### MAP 문제에서
- "posterior maximization이 진짜 목표이며, MLE는 uniform prior를 둔 MAP의 특수한 경우입니다."
- "prior는 hypothesis space를 제한하고 inductive bias를 부여합니다."

### Gaussian-MSE 문제에서
- "Gaussian noise 가정 하에서 negative log-likelihood를 전개하면 squared error 항만 남습니다."
- "상수항은 최적화 해에 영향을 주지 않으므로 MSE minimization과 동치입니다."

### KL 문제에서
- "cross entropy는 KL divergence와 entropy의 합이며, 데이터 분포가 고정되면 entropy는 상수입니다."

---

## 15. 교수님 기준 최종 압축

### 한 줄 결론
중간고사에서 가장 중요한 것은 `ML과 Prior`, 더 정확히는 `MAP 관점에서 learning 전체를 보는 것`이다.

### 시험 핵심 연결고리
1. probability 언어를 익힌다  
2. Bernoulli와 Gaussian을 이해한다  
3. Bayes theorem으로 posterior를 쓴다  
4. posterior를 최대화하면 MAP다  
5. uniform prior면 MLE다  
6. Gaussian이면 NLL이 MSE가 된다  
7. CE와 KL이 연결된다  
8. 이 전체가 learning의 수학적 본질이다

### 쉽게 설명하면
교수님은 "학습이란 결국 가설에 대한 믿음을 데이터로 업데이트하는 것"이라고 보고, 그걸 동전 예제에서 MLE/MAP로 보여준 뒤, Gaussian과 MSE, KL과 CE까지 연결한다.

### 남에게 설명하는 한 문장
"이 수업은 probability로 hypothesis를 표현하고, prior와 data의 균형으로 learning을 이해하는 수업이며, MLE는 MAP의 특수한 경우라고 보는 게 교수님 스타일이다."

### 핵심 킬러 요약
- MLE보다 MAP가 상위 개념이다
- ML = uniform prior를 둔 MAP
- strong prior = strong inductive bias
- hypothesis space가 좁아지면 expressivity가 낮아진다
- Gaussian regression → NLL → MSE
- CE = KL + H
- 시험은 공식 암기가 아니라 논리 설명을 본다

### 주의
- "MLE가 진짜 목표"라고만 쓰면 교수님 프레임과 어긋난다
- prior를 단순 보정항으로만 쓰면 약하다
- hypothesis space, inductive bias, expressivity를 연결해서 써야 한다

### 연구 포인트
- CNN과 Transformer의 차이를 prior / inductive bias 관점에서 해석
- Bayesian learning과 modern deep learning의 연결
- empirical risk, NLL, KL의 더 큰 그림


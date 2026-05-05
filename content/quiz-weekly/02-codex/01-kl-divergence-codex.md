---
title: "KL Divergence 학습 자료"
slug: kl-divergence-codex
order: 1
---

---
title: "KL Divergence 학습 자료 - Codex"
course: "딥러닝이론"
created: 2026-05-04
tags:
  - deep-learning
  - kl-divergence
  - cross-entropy
  - nll
  - mse
  - mle
  - quiz
---

# KL Divergence 학습 자료

> [!summary] 결론
> KL Divergence 관련 수업은 진행되었다. 교수님은 KL Divergence의 정의, 성질, Cross-Entropy와의 관계, Empirical distribution과 Model distribution의 KL 최소화가 MLE/NLL 최소화와 연결된다는 점을 설명했다.  
> 다만 두 Gaussian 사이의 KL Divergence를 처음부터 끝까지 적분으로 직접 계산하는 풀이는 수업 중 완전 전개하지 않고, "직접 풀어보라"는 형태로 넘긴 것으로 정리하는 것이 정확하다.  
> 반대로 Gaussian likelihood에서 NLL이 MSE로 바뀌는 유도는 중요도 10/10 수준으로 강조된 핵심 유도이다.

---

## 0. 수업 자료 근거 정리

| 항목 | 판단 | 근거 |
|---|---:|---|
| KL Divergence 정의 수업 | 진행함 | [04_4주차_분석.md](/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/04_4주차_분석.md:40) |
| KL의 성질: 비음수, 같으면 0, 비대칭 | 진행함 | [04_4주차_분석.md](/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/04_4주차_분석.md:41) |
| Empirical distribution과 model distribution의 KL 최소화가 MLE와 연결 | 진행함 | [04_4주차_분석.md](/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/04_4주차_분석.md:43) |
| Cross-Entropy와 KL 관계 | 진행함 | [lecture_script_complete_study_guide.md](/Users/jeongseongchae/dev/university/deep_learning/docs/lecture_script_complete_study_guide.md:1370) |
| 두 Gaussian KL 직접 계산 | 직접 풀이 권유, 수업 중 완전 풀이는 생략 | [4주차.md](/Users/jeongseongchae/dev/university/deep_learning/docs/QUIZ/4주차.md:24) |
| 공식 Quiz 8의 요구: Gaussian KL 단순화 + MSE 연결 | 공식 퀴즈 자료에 있음 | [공식퀴즈_종합.md](/Users/jeongseongchae/dev/university/deep_learning/docs/QUIZ/공식퀴즈_종합.md:150) |
| NLL에서 MSE가 나오는 유도 | 매우 중요, 명시적 강조 | [04_4주차_분석.md](/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/04_4주차_분석.md:65) |
| NLL은 ERM으로 연결됨 | 진행함 | [lecture_script_complete_study_guide.md](/Users/jeongseongchae/dev/university/deep_learning/docs/lecture_script_complete_study_guide.md:543) |
| 기말 대비 핵심 축: NLL, ERM, Gaussian likelihood, MSE | 출제 가능성 높음 | [기말고사_60퍼센트_범위_출제예상_심층보고서.md](/Users/jeongseongchae/dev/university/deep_learning/docs/강의분석/기말고사_60퍼센트_범위_출제예상_심층보고서.md:280) |

---

## 1. 이 주제의 시험 핵심

시험에서 필요한 답은 단순히 다음 식 하나를 외우는 것이 아니다.

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\mathbb{E}_{x\sim P}
\left[
\log P(x)-\log Q(x)
\right]
$$

교수님이 보려는 것은 아래 연결 체인이다.

$$
\boxed{
\text{Data distribution}
\to
\text{Empirical distribution}
\to
D_{\mathrm{KL}}(\hat P\Vert Q_\theta)
\to
\text{Cross-Entropy}
\to
\text{NLL}
\to
\text{MLE}
\to
\text{Gaussian assumption}
\to
\text{MSE}
}
$$

즉 답안은 다음 질문에 모두 답해야 한다.

| 질문 | 답안에 들어가야 할 핵심 |
|---|---|
| KL Divergence는 무엇인가? | 두 분포 $P$, $Q$의 log-ratio를 $P$ 기준으로 평균낸 값 |
| 왜 항상 0 이상인가? | Jensen 부등식 또는 Gibbs inequality로 증명 |
| 왜 Cross-Entropy와 연결되는가? | $D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P)$ |
| 왜 Cross-Entropy 최소화가 KL 최소화인가? | $P$가 고정이면 $H(P)$가 상수 |
| 왜 NLL과 연결되는가? | empirical distribution에서의 cross-entropy가 평균 NLL |
| 왜 MLE와 연결되는가? | likelihood 최대화는 negative log-likelihood 최소화 |
| 왜 MSE가 나오는가? | Gaussian likelihood에 $-\log$를 취하면 square term만 남음 |
| Gaussian KL과 MSE는 어떻게 닮았는가? | 같은 분산이면 $D_{\mathrm{KL}}=\frac{(\mu_1-\mu_2)^2}{2\sigma^2}$ |

---

## 2. 가장 기초부터: 확률분포란 무엇인가

확률분포는 "각 경우가 얼마나 자주 일어나는지"를 숫자로 나눠 놓은 것이다.

예를 들어 주사위가 공정하면

$$
P(1)=P(2)=P(3)=P(4)=P(5)=P(6)=\frac{1}{6}
$$

이다. 여기서 모든 확률을 더하면 1이 되어야 한다.

$$
\sum_{x}P(x)=1
$$

왜 1인가? 가능한 모든 경우를 다 합치면 "반드시 어떤 일 하나는 일어난다"는 뜻이기 때문이다.

확률분포 $P$와 $Q$가 있다고 하자.

| 기호 | 의미 |
|---|---|
| $P$ | 진짜 분포, 데이터 분포, target distribution |
| $Q$ | 모델이 만든 분포, approximation |
| $P(x)$ | 사건 $x$가 진짜로 나올 확률 |
| $Q(x)$ | 모델이 사건 $x$에 부여한 확률 |

딥러닝 학습은 결국 $Q_\theta$가 $P$를 잘 따라가도록 parameter $\theta$를 조정하는 과정이다.

$$
Q_\theta \approx P
$$

KL Divergence는 이 "얼마나 비슷한가"를 재는 대표적인 도구다.

---

## 3. 기댓값: 확률을 곱한 평균

중학교 산술에서 평균은 다음처럼 계산한다.

$$
\text{평균}
=
\frac{\text{값}_1+\text{값}_2+\cdots+\text{값}_n}{n}
$$

확률에서는 모든 값이 똑같이 자주 나오지 않는다. 그래서 각 값에 "그 값이 나올 확률"을 곱해서 평균낸다.

$$
\mathbb{E}_{x\sim P}[f(x)]
=
\sum_x P(x)f(x)
$$

기호를 해체하면 다음과 같다.

| 기호 | 뜻 |
|---|---|
| $\mathbb{E}$ | expectation, 기댓값, 확률 가중 평균 |
| $x\sim P$ | $x$를 분포 $P$에서 뽑는다는 뜻 |
| $f(x)$ | $x$가 나왔을 때 계산할 값 |
| $\sum_x$ | 가능한 모든 $x$에 대해 더한다는 뜻 |
| $P(x)f(x)$ | 값 $f(x)$에 그 값이 나올 확률 $P(x)$를 곱함 |

KL Divergence도 기댓값이다. 어떤 값을 평균내는가? 다음 log-ratio를 평균낸다.

$$
\log\frac{P(x)}{Q(x)}
$$

---

## 4. KL Divergence의 정의

이산 확률분포에서 KL Divergence는 다음과 같이 정의한다.

$$
\boxed{
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}
}
$$

기댓값 형태로 쓰면 다음과 같다.

$$
\boxed{
D_{\mathrm{KL}}(P\Vert Q)
=
\mathbb{E}_{x\sim P}
\left[
\log\frac{P(x)}{Q(x)}
\right]
}
$$

로그의 나눗셈 법칙을 쓰면

$$
\log\frac{P(x)}{Q(x)}
=
\log P(x)-\log Q(x)
$$

이므로

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\mathbb{E}_{x\sim P}
\left[
\log P(x)-\log Q(x)
\right]
$$

가 된다. 강의 자료에 나온 형태가 바로 이것이다.

### Line-by-Line 해체

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

| 부분 | 의미 | 왜 필요한가 |
|---|---|---|
| $D_{\mathrm{KL}}$ | KL Divergence | 두 분포 차이를 재는 양 |
| $(P\Vert Q)$ | 방향이 $P$에서 $Q$로 감 | 일반적으로 $D_{\mathrm{KL}}(P\Vert Q)\neq D_{\mathrm{KL}}(Q\Vert P)$ |
| $\sum_x$ | 모든 가능한 $x$에 대해 더함 | 분포 전체를 비교해야 하므로 |
| $P(x)$ | 진짜 분포의 확률 | 진짜로 자주 나오는 곳을 더 중요하게 평가 |
| $\log\frac{P(x)}{Q(x)}$ | true/model의 log 비율 | 모델이 true에 비해 과소평가했는지 과대평가했는지 측정 |

### 직관

만약 어떤 $x$에 대해 $P(x)$는 큰데 $Q(x)$가 작으면,

$$
\frac{P(x)}{Q(x)}
$$

가 커진다. 그러면

$$
\log\frac{P(x)}{Q(x)}
$$

도 커진다. 즉 진짜로 자주 나오는 사건을 모델이 낮은 확률로 예측하면 큰 벌점을 받는다.

---

## 5. KL Divergence의 성질

### 5.1 KL은 일반적인 거리(distance)가 아니다

일반적인 거리라면 다음이 성립해야 한다.

$$
d(a,b)=d(b,a)
$$

예를 들어 서울에서 부산까지의 거리는 부산에서 서울까지의 거리와 같다.

하지만 KL Divergence는 일반적으로

$$
D_{\mathrm{KL}}(P\Vert Q)
\neq
D_{\mathrm{KL}}(Q\Vert P)
$$

이다. 그래서 엄밀히는 distance가 아니라 divergence라고 부른다.

### 5.2 KL은 항상 0 이상이다

핵심 성질은 다음이다.

$$
\boxed{
D_{\mathrm{KL}}(P\Vert Q)\ge 0
}
$$

또한

$$
\boxed{
D_{\mathrm{KL}}(P\Vert Q)=0
\iff
P=Q
}
$$

이다.

### 5.3 비음수성 증명

이 증명은 시험에서 "수학적 유도의 논리적 완결성"을 보여주기 좋다.

이산분포 $p_i$, $q_i$를 생각하자. 단, $p_i>0$인 곳에서는 $q_i>0$라고 가정한다. 만약 $p_i>0$인데 $q_i=0$이면 모델이 실제로 나오는 사건에 확률 0을 준 것이므로 KL은 무한대가 된다.

정의에서 시작한다.

$$
D_{\mathrm{KL}}(p\Vert q)
=
\sum_i p_i\log\frac{p_i}{q_i}
$$

나눗셈을 뒤집으면 마이너스가 붙는다.

$$
\log\frac{p_i}{q_i}
=
-\log\frac{q_i}{p_i}
$$

따라서

$$
D_{\mathrm{KL}}(p\Vert q)
=
-\sum_i p_i\log\frac{q_i}{p_i}
$$

여기서 함수

$$
f(t)=-\log t
$$

는 볼록 함수다. 이유는 2차 미분이 양수이기 때문이다.

$$
f''(t)=\frac{1}{t^2}>0
$$

볼록 함수에 Jensen 부등식을 적용하면

$$
\sum_i p_i f(a_i)
\ge
f\left(\sum_i p_i a_i\right)
$$

이다. 여기서

$$
a_i=\frac{q_i}{p_i}
$$

로 두면

$$
-\sum_i p_i\log\frac{q_i}{p_i}
\ge
-\log\left(
\sum_i p_i\frac{q_i}{p_i}
\right)
$$

이다. 안쪽을 정리한다.

$$
\sum_i p_i\frac{q_i}{p_i}
=
\sum_i q_i
$$

$q$는 확률분포이므로 모든 확률을 더하면 1이다.

$$
\sum_i q_i=1
$$

따라서

$$
D_{\mathrm{KL}}(p\Vert q)
\ge
-\log 1
=
0
$$

결론은

$$
\boxed{
D_{\mathrm{KL}}(p\Vert q)\ge 0
}
$$

이다.

등호가 성립하려면 Jensen 부등식의 등호 조건에 의해

$$
\frac{q_i}{p_i}
$$

가 모든 $i$에서 같은 상수여야 한다. 즉

$$
q_i=cp_i
$$

이다. 양변을 모든 $i$에 대해 더하면

$$
\sum_i q_i
=
c\sum_i p_i
$$

이고 둘 다 확률분포이므로

$$
1=c\cdot 1
$$

따라서

$$
c=1
$$

이다. 결국

$$
q_i=p_i
$$

이므로

$$
\boxed{
D_{\mathrm{KL}}(p\Vert q)=0
\iff
p=q
}
$$

이다.

---

## 6. Entropy와 Cross-Entropy

### 6.1 Entropy

Entropy는 한 분포 자체의 불확실성을 재는 값이다.

$$
\boxed{
H(P)
=
-\sum_x P(x)\log P(x)
}
$$

기댓값 형태로 쓰면

$$
H(P)
=
-\mathbb{E}_{x\sim P}[\log P(x)]
$$

이다.

### 6.2 Cross-Entropy

Cross-Entropy는 진짜 분포 $P$를 기준으로, 모델 분포 $Q$가 부여한 log probability의 평균 벌점이다.

$$
\boxed{
H(P,Q)
=
-\sum_x P(x)\log Q(x)
}
$$

기댓값 형태로 쓰면

$$
H(P,Q)
=
-\mathbb{E}_{x\sim P}[\log Q(x)]
$$

이다.

### 6.3 KL과 Cross-Entropy의 관계 유도

KL 정의에서 출발한다.

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_x P(x)\log\frac{P(x)}{Q(x)}
$$

로그의 나눗셈 법칙을 쓴다.

$$
\log\frac{P(x)}{Q(x)}
=
\log P(x)-\log Q(x)
$$

대입하면

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_x P(x)
\left[
\log P(x)-\log Q(x)
\right]
$$

분배법칙으로 나눈다.

$$
D_{\mathrm{KL}}(P\Vert Q)
=
\sum_x P(x)\log P(x)
-
\sum_x P(x)\log Q(x)
$$

Entropy 정의에 의해

$$
H(P)
=
-\sum_x P(x)\log P(x)
$$

이므로

$$
\sum_x P(x)\log P(x)
=
-H(P)
$$

Cross-Entropy 정의에 의해

$$
H(P,Q)
=
-\sum_x P(x)\log Q(x)
$$

이므로

$$
-\sum_x P(x)\log Q(x)
=
H(P,Q)
$$

따라서

$$
D_{\mathrm{KL}}(P\Vert Q)
=
-H(P)+H(P,Q)
$$

즉

$$
\boxed{
D_{\mathrm{KL}}(P\Vert Q)
=
H(P,Q)-H(P)
}
$$

양변을 바꾸면

$$
\boxed{
H(P,Q)
=
H(P)+D_{\mathrm{KL}}(P\Vert Q)
}
$$

이다.

### 왜 Cross-Entropy 최소화가 KL 최소화인가

학습할 때 진짜 데이터 분포 $P$는 고정되어 있다. 모델이 바꾸는 것은 $Q_\theta$다.

$$
H(P,Q_\theta)
=
H(P)+D_{\mathrm{KL}}(P\Vert Q_\theta)
$$

여기서 $H(P)$는 $\theta$와 무관한 상수다. 상수는 최소화 위치를 바꾸지 않는다.

따라서

$$
\arg\min_\theta H(P,Q_\theta)
=
\arg\min_\theta D_{\mathrm{KL}}(P\Vert Q_\theta)
$$

이다.

이것이 강의의 핵심 문장이다.

$$
\boxed{
\text{Cross-Entropy 최소화}
\equiv
\text{KL Divergence 최소화}
\quad
\text{if }P\text{ is fixed}
}
$$

---

## 7. Empirical Distribution과 MLE 연결

### 7.1 Empirical Distribution

현실에서는 진짜 분포 $P$를 모른다. 우리가 가진 것은 데이터셋뿐이다.

$$
D=\{z_1,z_2,\dots,z_n\}
$$

여기서 supervised learning이면 보통

$$
z_i=(x_i,y_i)
$$

로 생각하면 된다.

Empirical distribution은 데이터 하나하나에 같은 확률질량을 주는 분포다.

$$
\boxed{
\hat P_D(z)
=
\frac{1}{n}
\sum_{i=1}^{n}
\delta(z-z_i)
}
$$

기호를 해체하면 다음과 같다.

| 기호 | 의미 |
|---|---|
| $\hat P_D$ | 데이터셋 $D$로 만든 empirical distribution |
| $\frac{1}{n}$ | 데이터 $n$개가 모두 같은 가중치 |
| $\delta(z-z_i)$ | $z=z_i$일 때만 질량을 주는 delta |
| $\sum_{i=1}^n$ | 모든 데이터 포인트를 모음 |

직관적으로는 데이터셋 자체를 하나의 확률분포처럼 보는 것이다.

### 7.2 Empirical distribution에서 Cross-Entropy 계산

모델 분포를 $q_\theta(z)$라고 하자.

Empirical distribution과 model distribution 사이의 Cross-Entropy는

$$
H(\hat P_D,q_\theta)
=
-\mathbb{E}_{z\sim \hat P_D}
\left[
\log q_\theta(z)
\right]
$$

이다.

Empirical distribution에서의 기댓값은 sample average가 된다.

$$
\mathbb{E}_{z\sim \hat P_D}
\left[
\log q_\theta(z)
\right]
=
\frac{1}{n}
\sum_{i=1}^{n}
\log q_\theta(z_i)
$$

따라서

$$
H(\hat P_D,q_\theta)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log q_\theta(z_i)
$$

양변에 $n$을 곱하면

$$
nH(\hat P_D,q_\theta)
=
-
\sum_{i=1}^{n}
\log q_\theta(z_i)
$$

오른쪽은 Negative Log-Likelihood다.

$$
\boxed{
\mathrm{NLL}(\theta)
=
-
\sum_{i=1}^{n}
\log q_\theta(z_i)
=
nH(\hat P_D,q_\theta)
}
$$

### 7.3 NLL, Cross-Entropy, KL, MLE의 동치

앞에서

$$
H(\hat P_D,q_\theta)
=
H(\hat P_D)
+
D_{\mathrm{KL}}(\hat P_D\Vert q_\theta)
$$

임을 보였다.

$H(\hat P_D)$는 데이터셋이 고정되면 상수다. 따라서

$$
\arg\min_\theta H(\hat P_D,q_\theta)
=
\arg\min_\theta D_{\mathrm{KL}}(\hat P_D\Vert q_\theta)
$$

이다.

또한

$$
\mathrm{NLL}(\theta)
=
nH(\hat P_D,q_\theta)
$$

이고 $n>0$은 양의 상수이므로

$$
\arg\min_\theta \mathrm{NLL}(\theta)
=
\arg\min_\theta H(\hat P_D,q_\theta)
$$

이다.

마지막으로 MLE는 likelihood를 최대화한다.

$$
\hat\theta_{\mathrm{MLE}}
=
\arg\max_\theta
\prod_{i=1}^{n}q_\theta(z_i)
$$

로그는 증가함수이므로 likelihood 최대화와 log-likelihood 최대화는 같다.

$$
\arg\max_\theta
\prod_{i=1}^{n}q_\theta(z_i)
=
\arg\max_\theta
\sum_{i=1}^{n}\log q_\theta(z_i)
$$

여기에 마이너스를 붙이면 최대화 문제가 최소화 문제로 바뀐다.

$$
\arg\max_\theta
\sum_{i=1}^{n}\log q_\theta(z_i)
=
\arg\min_\theta
\left[
-
\sum_{i=1}^{n}\log q_\theta(z_i)
\right]
$$

따라서 전체 연결은 다음과 같다.

$$
\boxed{
\mathrm{MLE}
\equiv
\mathrm{NLL\ minimization}
\equiv
\mathrm{Cross\text{-}Entropy\ minimization}
\equiv
D_{\mathrm{KL}}(\hat P_D\Vert q_\theta)\ \mathrm{minimization}
}
$$

말로 쓰면 다음이다.

> MLE는 "데이터가 만든 empirical distribution"과 "모델이 만든 distribution" 사이의 KL Divergence를 줄이는 과정이다.

---

## 8. NLL에서 MSE가 나오는 유도

이 부분이 수업에서 가장 강하게 강조된 핵심 유도다.

### 8.1 문제 설정

회귀 문제에서 데이터가 다음과 같이 주어졌다고 하자.

$$
D=\{(x_i,y_i)\}_{i=1}^{n}
$$

모델 $h_\theta$는 입력 $x_i$에 대해 예측값을 낸다.

$$
\hat y_i=h_\theta(x_i)
$$

Gaussian likelihood 가정은 다음과 같다.

$$
\boxed{
y_i\mid x_i,\theta
\sim
\mathcal{N}
\left(
h_\theta(x_i),\sigma^2
\right)
}
$$

뜻은 다음이다.

| 부분 | 의미 |
|---|---|
| $y_i\mid x_i,\theta$ | 입력 $x_i$와 parameter $\theta$가 주어졌을 때 출력 $y_i$ |
| $\mathcal{N}$ | Gaussian, Normal distribution |
| $h_\theta(x_i)$ | Gaussian의 평균, 즉 모델 예측값 |
| $\sigma^2$ | noise variance, 고정된 상수라고 가정 |

즉 모델은 이렇게 말한다.

> "정답 $y_i$는 내 예측값 $h_\theta(x_i)$ 주변에서 Gaussian noise를 가지고 흔들린다."

### 8.2 Gaussian PDF

Gaussian density는 다음이다.

$$
p(y_i\mid x_i,\theta)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right)
$$

핵심 구조는 교수님이 강조한 것처럼

$$
\boxed{
\text{exp}+\text{minus}+\text{square}
}
$$

이다.

| 구조 | 식 안의 위치 | 의미 |
|---|---|---|
| square | $\left(y_i-h_\theta(x_i)\right)^2$ | 예측과 정답의 차이를 제곱 |
| minus | $-\frac{(\cdot)^2}{2\sigma^2}$ | 멀어질수록 확률이 작아짐 |
| exp | $\exp(\cdot)$ | density를 양수로 만듦 |

### 8.3 IID likelihood

데이터가 IID라고 가정하면 전체 likelihood는 각 sample likelihood의 곱이다.

$$
p(D\mid\theta)
=
\prod_{i=1}^{n}
p(y_i\mid x_i,\theta)
$$

왜 곱인가? 독립인 사건들이 동시에 일어날 확률은 각 확률을 곱하기 때문이다.

Gaussian PDF를 대입하면

$$
p(D\mid\theta)
=
\prod_{i=1}^{n}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right)
$$

### 8.4 NLL 정의

Negative Log-Likelihood는 다음이다.

$$
\mathrm{NLL}(\theta)
=
-\log p(D\mid\theta)
$$

왜 로그를 취하는가?

$$
\log(ab)=\log a+\log b
$$

이므로 곱을 합으로 바꿀 수 있기 때문이다. 합은 미분과 계산이 쉽다.

왜 negative를 붙이는가? MLE는 likelihood를 최대화하지만, 딥러닝 최적화는 보통 loss를 최소화하는 형태로 쓰기 때문이다.

$$
\arg\max_\theta p(D\mid\theta)
=
\arg\min_\theta -\log p(D\mid\theta)
$$

### 8.5 NLL 계산을 한 줄씩 전개

정의에서 시작한다.

$$
\mathrm{NLL}(\theta)
=
-
\log
\prod_{i=1}^{n}
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right)
$$

로그의 곱 법칙을 쓴다.

$$
=
-
\sum_{i=1}^{n}
\log
\left[
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right)
\right]
$$

로그 안의 곱을 다시 합으로 분리한다.

$$
=
-
\sum_{i=1}^{n}
\left[
\log
\frac{1}{\sqrt{2\pi\sigma^2}}
+
\log
\exp
\left(
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right)
\right]
$$

여기서

$$
\log\exp(a)=a
$$

를 적용한다. 그러면 exp가 사라진다.

$$
=
-
\sum_{i=1}^{n}
\left[
\log
\frac{1}{\sqrt{2\pi\sigma^2}}
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right]
$$

첫 번째 로그를 정리한다.

$$
\log
\frac{1}{\sqrt{2\pi\sigma^2}}
=
-
\log\sqrt{2\pi\sigma^2}
$$

따라서

$$
=
-
\sum_{i=1}^{n}
\left[
-
\log\sqrt{2\pi\sigma^2}
-
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right]
$$

바깥의 마이너스를 안으로 분배한다.

$$
\mathrm{NLL}(\theta)
=
\sum_{i=1}^{n}
\left[
\log\sqrt{2\pi\sigma^2}
+
\frac{
\left(y_i-h_\theta(x_i)\right)^2
}{
2\sigma^2
}
\right]
$$

상수항과 제곱항을 분리한다.

$$
\mathrm{NLL}(\theta)
=
n\log\sqrt{2\pi\sigma^2}
+
\frac{1}{2\sigma^2}
\sum_{i=1}^{n}
\left(y_i-h_\theta(x_i)\right)^2
$$

### 8.6 Argmin 동치

$\sigma^2$가 고정되어 있으면

$$
n\log\sqrt{2\pi\sigma^2}
$$

는 $\theta$와 무관한 상수다. 또한

$$
\frac{1}{2\sigma^2}
$$

는 양의 상수다. 상수를 더하거나 양의 상수를 곱해도 최소점은 바뀌지 않는다.

따라서

$$
\arg\min_\theta \mathrm{NLL}(\theta)
=
\arg\min_\theta
\sum_{i=1}^{n}
\left(y_i-h_\theta(x_i)\right)^2
$$

양의 상수 $\frac{1}{n}$을 곱해도 최소점은 바뀌지 않는다.

$$
\arg\min_\theta
\sum_{i=1}^{n}
\left(y_i-h_\theta(x_i)\right)^2
=
\arg\min_\theta
\frac{1}{n}
\sum_{i=1}^{n}
\left(y_i-h_\theta(x_i)\right)^2
$$

오른쪽이 MSE다.

$$
\boxed{
\mathrm{MSE}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(y_i-h_\theta(x_i)\right)^2
}
$$

최종 결론은 다음이다.

$$
\boxed{
\arg\min_\theta \mathrm{NLL}(\theta)
=
\arg\min_\theta \mathrm{MSE}(\theta)
}
$$

말로 쓰면 다음이다.

> 회귀에서 MSE를 쓰는 것은 임의의 선택이 아니라, 출력 noise가 고정분산 Gaussian이라고 가정했을 때의 NLL 최소화와 동치이기 때문이다.

---

## 9. 두 Gaussian 사이의 KL Divergence 직접 계산

공식 퀴즈에서 주어진 식은 다음이다.

$$
D_{\mathrm{KL}}
\left(
\mathcal{N}(\mu_1,\sigma_1^2)
\Vert
\mathcal{N}(\mu_2,\sigma_2^2)
\right)
=
\log\frac{\sigma_2}{\sigma_1}
+
\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
-
\frac{1}{2}
$$

아래는 이 식이 왜 나오는지 처음부터 유도한 것이다.

### 9.1 두 분포 정의

첫 번째 분포를 $p$라고 하자.

$$
p(x)
=
\frac{1}{\sqrt{2\pi}\sigma_1}
\exp
\left(
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
\right)
$$

두 번째 분포를 $q$라고 하자.

$$
q(x)
=
\frac{1}{\sqrt{2\pi}\sigma_2}
\exp
\left(
-
\frac{(x-\mu_2)^2}{2\sigma_2^2}
\right)
$$

기호 의미는 다음이다.

| 기호 | 의미 |
|---|---|
| $\mu_1$ | 첫 번째 Gaussian의 평균 |
| $\mu_2$ | 두 번째 Gaussian의 평균 |
| $\sigma_1^2$ | 첫 번째 Gaussian의 분산 |
| $\sigma_2^2$ | 두 번째 Gaussian의 분산 |
| $\sigma_1,\sigma_2$ | 표준편차, 분산의 제곱근 |

### 9.2 KL 정의에서 시작

연속형 KL 정의는 다음이다.

$$
D_{\mathrm{KL}}(p\Vert q)
=
\mathbb{E}_{x\sim p}
\left[
\log p(x)-\log q(x)
\right]
$$

즉 $p$에서 $x$를 뽑는다고 생각하고, 그때의 log-density 차이를 평균낸다.

### 9.3 $\log p(x)$ 계산

$$
\log p(x)
=
\log
\left[
\frac{1}{\sqrt{2\pi}\sigma_1}
\exp
\left(
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
\right)
\right]
$$

로그의 곱 법칙으로 분리한다.

$$
=
\log
\frac{1}{\sqrt{2\pi}\sigma_1}
+
\log
\exp
\left(
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
\right)
$$

$\log\exp(a)=a$를 적용한다.

$$
\log p(x)
=
-
\log(\sqrt{2\pi}\sigma_1)
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
$$

### 9.4 $\log q(x)$ 계산

같은 방식으로

$$
\log q(x)
=
-
\log(\sqrt{2\pi}\sigma_2)
-
\frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

이다.

### 9.5 $\log p(x)-\log q(x)$ 계산

두 식을 뺀다.

$$
\begin{aligned}
\log p(x)-\log q(x)
&=
\left[
-
\log(\sqrt{2\pi}\sigma_1)
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
\right]\\
&\quad
-
\left[
-
\log(\sqrt{2\pi}\sigma_2)
-
\frac{(x-\mu_2)^2}{2\sigma_2^2}
\right]
\end{aligned}
$$

마이너스를 분배한다.

$$
\log p(x)-\log q(x)
=
-
\log(\sqrt{2\pi}\sigma_1)
+
\log(\sqrt{2\pi}\sigma_2)
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
+
\frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

로그 상수를 정리한다.

$$
-
\log(\sqrt{2\pi}\sigma_1)
+
\log(\sqrt{2\pi}\sigma_2)
=
\log
\frac{\sqrt{2\pi}\sigma_2}{\sqrt{2\pi}\sigma_1}
=
\log\frac{\sigma_2}{\sigma_1}
$$

따라서

$$
\log p(x)-\log q(x)
=
\log\frac{\sigma_2}{\sigma_1}
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
+
\frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

### 9.6 $p$에 대한 기댓값을 취한다

KL 정의에 의해

$$
D_{\mathrm{KL}}(p\Vert q)
=
\mathbb{E}_{x\sim p}
\left[
\log\frac{\sigma_2}{\sigma_1}
-
\frac{(x-\mu_1)^2}{2\sigma_1^2}
+
\frac{(x-\mu_2)^2}{2\sigma_2^2}
\right]
$$

기댓값의 선형성을 쓰면

$$
D_{\mathrm{KL}}(p\Vert q)
=
\log\frac{\sigma_2}{\sigma_1}
-
\frac{1}{2\sigma_1^2}
\mathbb{E}_{x\sim p}
\left[
(x-\mu_1)^2
\right]
+
\frac{1}{2\sigma_2^2}
\mathbb{E}_{x\sim p}
\left[
(x-\mu_2)^2
\right]
$$

첫 번째 기댓값은 분산 정의 그대로다.

$$
\mathbb{E}_{x\sim p}
\left[
(x-\mu_1)^2
\right]
=
\sigma_1^2
$$

따라서 둘째 항은

$$
-
\frac{1}{2\sigma_1^2}\sigma_1^2
=
-
\frac{1}{2}
$$

이다.

### 9.7 $\mathbb{E}_{p}[(x-\mu_2)^2]$ 계산

이 부분이 직접 계산의 핵심이다.

다음처럼 쪼갠다.

$$
x-\mu_2
=
(x-\mu_1)+(\mu_1-\mu_2)
$$

왜 이렇게 하는가? $x$는 $p=\mathcal{N}(\mu_1,\sigma_1^2)$에서 뽑히므로, $x-\mu_1$의 평균과 분산은 쉽게 알 수 있기 때문이다.

제곱한다.

$$
\begin{aligned}
(x-\mu_2)^2
&=
\left[
(x-\mu_1)+(\mu_1-\mu_2)
\right]^2\\
&=
(x-\mu_1)^2
+2(x-\mu_1)(\mu_1-\mu_2)
+(\mu_1-\mu_2)^2
\end{aligned}
$$

기댓값을 취한다.

$$
\begin{aligned}
\mathbb{E}_{p}[(x-\mu_2)^2]
&=
\mathbb{E}_{p}[(x-\mu_1)^2]\\
&\quad
+2(\mu_1-\mu_2)\mathbb{E}_{p}[x-\mu_1]\\
&\quad
+(\mu_1-\mu_2)^2
\end{aligned}
$$

각 항을 계산한다.

$$
\mathbb{E}_{p}[(x-\mu_1)^2]=\sigma_1^2
$$

그리고 평균 정의에 의해

$$
\mathbb{E}_{p}[x-\mu_1]
=
\mathbb{E}_{p}[x]-\mu_1
=
\mu_1-\mu_1
=
0
$$

따라서 가운데 항은 사라진다.

$$
2(\mu_1-\mu_2)\mathbb{E}_{p}[x-\mu_1]
=
0
$$

결국

$$
\boxed{
\mathbb{E}_{p}[(x-\mu_2)^2]
=
\sigma_1^2+(\mu_1-\mu_2)^2
}
$$

### 9.8 최종 공식

앞의 결과를 모두 대입한다.

$$
\begin{aligned}
D_{\mathrm{KL}}(p\Vert q)
&=
\log\frac{\sigma_2}{\sigma_1}
-
\frac{1}{2}
+
\frac{1}{2\sigma_2^2}
\left[
\sigma_1^2+(\mu_1-\mu_2)^2
\right]\\
&=
\log\frac{\sigma_2}{\sigma_1}
+
\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
-
\frac{1}{2}
\end{aligned}
$$

따라서

$$
\boxed{
D_{\mathrm{KL}}
\left(
\mathcal{N}(\mu_1,\sigma_1^2)
\Vert
\mathcal{N}(\mu_2,\sigma_2^2)
\right)
=
\log\frac{\sigma_2}{\sigma_1}
+
\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
-
\frac{1}{2}
}
$$

---

## 10. 공식 Quiz 8 단순화

공식 퀴즈의 조건은

$$
\sigma_1^2=\sigma_2^2=1
$$

이다. 따라서

$$
\sigma_1=1,\qquad \sigma_2=1
$$

이고

$$
\log\frac{\sigma_2}{\sigma_1}
=
\log 1
=
0
$$

이다.

일반식을 다시 쓴다.

$$
D_{\mathrm{KL}}
=
\log\frac{\sigma_2}{\sigma_1}
+
\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}
-
\frac{1}{2}
$$

값을 대입한다.

$$
D_{\mathrm{KL}}
=
0
+
\frac{1+(\mu_1-\mu_2)^2}{2}
-
\frac{1}{2}
$$

분수를 나눈다.

$$
D_{\mathrm{KL}}
=
\frac{1}{2}
+
\frac{(\mu_1-\mu_2)^2}{2}
-
\frac{1}{2}
$$

$\frac{1}{2}$와 $-\frac{1}{2}$가 상쇄된다.

$$
\boxed{
D_{\mathrm{KL}}
=
\frac{(\mu_1-\mu_2)^2}{2}
}
$$

더 일반적으로 같은 분산 $\sigma^2$이면

$$
\boxed{
D_{\mathrm{KL}}
\left(
\mathcal{N}(\mu_1,\sigma^2)
\Vert
\mathcal{N}(\mu_2,\sigma^2)
\right)
=
\frac{(\mu_1-\mu_2)^2}{2\sigma^2}
}
$$

이 식은 평균 차이가 커질수록 KL이 제곱으로 커진다는 뜻이다.

---

## 11. Gaussian KL과 MSE 연결

공식 Quiz 8의 두 번째 요구는 다음 연결을 설명하는 것이다.

$$
\boxed{
\text{CE}
\to
\text{NLL}
\to
\text{Gaussian assumption}
\to
\text{MSE}
}
$$

### 11.1 KL 관점의 연결

모델이 입력 $x$에 대해 평균 $\mu_\theta(x)$를 예측한다고 하자.

$$
q_\theta(y\mid x)
=
\mathcal{N}
\left(
\mu_\theta(x),1
\right)
$$

target을 평균 $y$이고 분산 1인 Gaussian으로 생각하면

$$
p(y\mid x)
=
\mathcal{N}(y,1)
$$

이다. 두 분포의 분산이 같고 1이므로 앞에서 구한 특수식을 적용한다.

$$
D_{\mathrm{KL}}
\left(
\mathcal{N}(y,1)
\Vert
\mathcal{N}(\mu_\theta(x),1)
\right)
=
\frac{(y-\mu_\theta(x))^2}{2}
$$

여러 sample에 대해 더하면

$$
\sum_{i=1}^{n}
D_{\mathrm{KL}}
\left(
\mathcal{N}(y_i,1)
\Vert
\mathcal{N}(\mu_\theta(x_i),1)
\right)
=
\frac{1}{2}
\sum_{i=1}^{n}
\left(
y_i-\mu_\theta(x_i)
\right)^2
$$

평균을 취하면

$$
\frac{1}{n}
\sum_{i=1}^{n}
D_{\mathrm{KL}}
\left(
\mathcal{N}(y_i,1)
\Vert
\mathcal{N}(\mu_\theta(x_i),1)
\right)
=
\frac{1}{2}
\mathrm{MSE}
$$

따라서

$$
\boxed{
\text{같은 분산 Gaussian 사이의 KL 최소화}
\equiv
\text{MSE 최소화}
}
$$

이다.

### 11.2 NLL 관점의 더 표준적인 연결

엄밀한 supervised regression에서는 target 하나 $y_i$를 "평균이 $y_i$인 Gaussian 분포"로 보는 설명은 직관용이다. 표준적인 유도는 다음이다.

$$
y_i\mid x_i,\theta
\sim
\mathcal{N}
\left(
\mu_\theta(x_i),1
\right)
$$

그러면 한 sample의 NLL은

$$
\begin{aligned}
-\log q_\theta(y_i\mid x_i)
&=
-\log
\left[
\frac{1}{\sqrt{2\pi}}
\exp
\left(
-
\frac{(y_i-\mu_\theta(x_i))^2}{2}
\right)
\right]\\
&=
\log\sqrt{2\pi}
+
\frac{(y_i-\mu_\theta(x_i))^2}{2}
\end{aligned}
$$

첫 항은 상수이므로 최소화에는 영향을 주지 않는다.

$$
\arg\min_\theta
\sum_{i=1}^{n}
-\log q_\theta(y_i\mid x_i)
=
\arg\min_\theta
\sum_{i=1}^{n}
\left(
y_i-\mu_\theta(x_i)
\right)^2
$$

따라서

$$
\boxed{
\text{Gaussian NLL 최소화}
\equiv
\text{MSE 최소화}
}
$$

이다.

### 11.3 두 설명의 관계

두 설명은 서로 충돌하지 않는다.

| 관점 | 무엇을 보여주는가 |
|---|---|
| KL 관점 | 같은 분산 Gaussian 분포끼리 비교하면 평균 차이 제곱이 KL이 됨 |
| NLL 관점 | Gaussian likelihood에 $-\log$를 취하면 squared error가 됨 |
| 통합 관점 | NLL은 empirical distribution과 model distribution 사이의 Cross-Entropy이고, 이는 KL 최소화와 동치 |

시험 답안에서는 아래 문장이 가장 안전하다.

> Cross-Entropy는 $H(P,Q)=H(P)+D_{\mathrm{KL}}(P\Vert Q)$이므로 $P$가 고정이면 CE 최소화와 KL 최소화가 같다. 데이터의 empirical distribution에 대한 CE는 평균 NLL이고, MLE는 NLL 최소화와 같다. 회귀에서 $q_\theta(y\mid x)=\mathcal{N}(\mu_\theta(x),\sigma^2)$를 가정하면 NLL은 상수와 양의 상수배를 제외하고 $\sum_i(y_i-\mu_\theta(x_i))^2$가 되어 MSE 최소화와 동치다.

---

## 12. 교수님이 직접 풀어준 것과 넘긴 것

정확히 구분하면 다음이다.

| 내용 | 수업 진행 여부 | 답안에 반영할 표현 |
|---|---:|---|
| KL Divergence의 정의 | 진행함 | "KL은 $D_{\mathrm{KL}}(P\Vert Q)=E_P[\log P-\log Q]$로 정의된다." |
| KL의 성질 | 진행함 | "$D_{\mathrm{KL}}\ge 0$, 같으면 0, 일반적으로 비대칭이다." |
| KL과 Cross-Entropy 관계 | 진행함 | "$D_{\mathrm{KL}}=H(P,Q)-H(P)$." |
| Empirical distribution과 model distribution의 KL 최소화 | 진행함 | "MLE는 empirical distribution과 model distribution의 KL을 줄이는 과정이다." |
| 두 Gaussian KL 직접 적분 계산 | 직접 풀어보라고 권유, 시간 관계상 생략 | "공식은 주어질 수 있으나, 유도 과정을 준비해야 한다." |
| NLL에서 MSE가 나오는 유도 | 강하게 진행함 | "Gaussian likelihood + log + minus를 통해 square만 남아 MSE가 된다." |

따라서 "KL Divergence 수업이 있었는가?"에 대한 답은 "있었다"가 맞다.

하지만 "두 Gaussian KL을 칠판에서 처음부터 끝까지 직접 계산해 주었는가?"에 대한 답은 "직접 계산은 권유하고 넘어간 것으로 보는 것이 정확하다"가 맞다.

---

## 13. Quiz 8 추천 답안

### 문제

1변량 Gaussian 분포의 KL Divergence가 다음과 같이 주어졌다고 하자.

$$
D_{\mathrm{KL}}
\left(
\mathcal{N}(\mu_1,\sigma_1^2)
\Vert
\mathcal{N}(\mu_2,\sigma_2^2)
\right)
=
\frac{1}{2}
\left[
2\log\frac{\sigma_2}{\sigma_1}
+
\frac{\sigma_1^2}{\sigma_2^2}
+
\frac{(\mu_1-\mu_2)^2}{\sigma_2^2}
-
1
\right]
$$

1. $\sigma_1^2=\sigma_2^2=1$일 때 단순화하라.
2. 이 결과가 딥러닝 학습에서 Cross-Entropy, NLL, Gaussian assumption, MSE와 어떻게 이어지는지 설명하라.

### 추천 답안

$\sigma_1^2=\sigma_2^2=1$이면 $\sigma_1=\sigma_2=1$이다. 따라서

$$
\log\frac{\sigma_2}{\sigma_1}
=
\log 1
=
0
$$

이고

$$
\frac{\sigma_1^2}{\sigma_2^2}
=
1
$$

이다. 주어진 식에 대입하면

$$
\begin{aligned}
D_{\mathrm{KL}}
&=
\frac{1}{2}
\left[
2\cdot 0
+
1
+
(\mu_1-\mu_2)^2
-
1
\right]\\
&=
\frac{1}{2}
(\mu_1-\mu_2)^2
\end{aligned}
$$

따라서

$$
\boxed{
D_{\mathrm{KL}}
=
\frac{(\mu_1-\mu_2)^2}{2}
}
$$

이다.

이 결과는 MSE와 직접 연결된다. Cross-Entropy는

$$
H(P,Q)
=
H(P)+D_{\mathrm{KL}}(P\Vert Q)
$$

이고, 학습에서 true distribution $P$는 고정되어 있으므로 $H(P)$는 상수다. 따라서

$$
\arg\min_\theta H(P,Q_\theta)
=
\arg\min_\theta D_{\mathrm{KL}}(P\Vert Q_\theta)
$$

이다.

데이터셋으로 만든 empirical distribution $\hat P_D$에 대해 Cross-Entropy를 쓰면

$$
H(\hat P_D,q_\theta)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log q_\theta(y_i\mid x_i)
$$

이고, 오른쪽은 평균 NLL이다. 따라서 Cross-Entropy 최소화는 NLL 최소화와 연결된다.

회귀 문제에서 모델 출력 분포를

$$
q_\theta(y\mid x)
=
\mathcal{N}(\mu_\theta(x),1)
$$

로 가정하면, 한 sample의 NLL은

$$
-\log q_\theta(y_i\mid x_i)
=
C
+
\frac{(y_i-\mu_\theta(x_i))^2}{2}
$$

가 된다. 여기서 $C$는 $\theta$와 무관한 상수다. 따라서 전체 NLL 최소화는

$$
\sum_{i=1}^{n}
(y_i-\mu_\theta(x_i))^2
$$

를 최소화하는 것과 같고, 평균을 취하면 MSE다.

결론적으로

$$
\boxed{
\text{CE 최소화}
\equiv
\text{KL 최소화}
\equiv
\text{NLL 최소화}
\equiv
\text{Gaussian 가정하의 MSE 최소화}
}
$$

이다. 회귀 문제에서 MSE loss를 쓰는 통계적 정당화는 "출력 noise가 등분산 Gaussian"이라는 likelihood 가정에서 나온다.

---

## 14. 시험장에서 바로 쓰는 압축 답안

아래는 시간이 부족할 때 쓰는 버전이다.

```text
KL divergence is defined by
D_KL(P||Q)=E_{x~P}[log P(x)-log Q(x)].
It satisfies D_KL(P||Q)>=0 and equals 0 iff P=Q, but it is not symmetric.

Since
D_KL(P||Q)=H(P,Q)-H(P),
we have H(P,Q)=H(P)+D_KL(P||Q).
When P is fixed, H(P) is constant, so minimizing cross-entropy is equivalent
to minimizing KL divergence.

For empirical data distribution \hat P_D,
H(\hat P_D,q_\theta)=-(1/n) sum_i log q_\theta(y_i|x_i),
so cross-entropy minimization is average NLL minimization.
MLE is also NLL minimization because maximizing likelihood is equivalent
to minimizing negative log-likelihood.

If q_\theta(y|x)=N(\mu_\theta(x),\sigma^2) with fixed variance, then
-log q_\theta(y_i|x_i)
= C + (y_i-\mu_\theta(x_i))^2/(2\sigma^2).
Constants and positive scaling do not change the argmin, so NLL minimization
is equivalent to MSE minimization.
```

---

## 15. 자주 틀리는 지점

| 실수 | 왜 틀렸는가 | 올바른 표현 |
|---|---|---|
| KL을 그냥 거리라고만 씀 | KL은 비대칭이라 엄밀한 distance가 아님 | "거리처럼 해석할 수 있지만 divergence이다." |
| $D_{\mathrm{KL}}(P\Vert Q)=D_{\mathrm{KL}}(Q\Vert P)$라고 씀 | 방향이 다르면 값이 달라질 수 있음 | 순서를 반드시 유지한다. |
| Cross-Entropy와 KL이 항상 같다고 씀 | $H(P,Q)=H(P)+KL$이므로 상수항 차이가 있음 | $P$가 고정일 때 최소화가 동치라고 써야 함 |
| NLL과 MSE가 숫자로 완전히 같다고 씀 | 상수항과 $\frac{1}{2\sigma^2}$ 배율이 있음 | "argmin이 같다" 또는 "상수/양의 상수배를 제외하고 같다" |
| Gaussian variance 고정 가정을 빠뜨림 | $\sigma^2$도 학습하면 상수항이 아니게 됨 | "$\sigma^2$ fixed"를 명시 |
| IID 가정을 빠뜨림 | likelihood를 곱으로 쓰는 근거가 사라짐 | "IID이므로 joint likelihood는 product"라고 씀 |
| $\log\exp(a)=a$ 단계를 생략 | NLL→MSE의 핵심 조작이 보이지 않음 | exp가 log로 사라진다고 명시 |
| Quiz 8에서 $\sigma^2=1$인데 $\sigma=0$으로 착각 | 분산이 1이면 표준편차도 1 | $\sigma_1=\sigma_2=1$ |

---

## 16. 공부 순서

1. 먼저 KL 정의를 외우지 말고 "log-ratio의 $P$ 기준 평균"이라고 이해한다.
2. 다음으로 $D_{\mathrm{KL}}=H(P,Q)-H(P)$를 직접 전개한다.
3. $P$가 고정이면 CE 최소화와 KL 최소화가 왜 같은지 말로 설명한다.
4. empirical distribution에서 Cross-Entropy가 평균 NLL이 되는 식을 쓴다.
5. MLE가 NLL 최소화와 같은 이유를 로그와 마이너스로 설명한다.
6. Gaussian likelihood에 $-\log$를 취해서 MSE가 나오는 유도를 손으로 3번 쓴다.
7. 마지막으로 두 Gaussian KL 공식에서 $\sigma_1^2=\sigma_2^2=1$을 대입해 $\frac{(\mu_1-\mu_2)^2}{2}$가 나오는 계산을 확인한다.

---

## 17. 한 문장 최종 요약

$$
\boxed{
\text{딥러닝에서 loss를 최소화한다는 것은}
\quad
\text{경우에 따라 empirical distribution과 model distribution의 KL을 줄이는 것이며,}
\quad
\text{Gaussian 회귀에서는 그 NLL/KL 구조가 MSE로 나타난다.}
}
$$


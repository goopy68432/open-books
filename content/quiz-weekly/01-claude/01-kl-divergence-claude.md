---
title: "KL Divergence 학습 자료 — 4주차 핵심 유도"
slug: kl-divergence-claude
order: 1
---

# KL Divergence 학습 자료 — 4주차 핵심 유도

> **본 자료의 핵심**: KL Divergence라는 추상 개념이 어떻게 **MLE → NLL → MSE** 라는 실용적 손실함수로 자연스럽게 이어지는지를, 4주차 강의에서 교수님이 강조하신 유도 흐름 그대로 처음부터 끝까지 재현합니다.
>
> 교수님 직접 강조 (라인 489~493):
> > "라이크리오드를 키우는 게 NLL이랑 연결되고, NLL은 MSE랑 연결되어 있다. ... 이걸 전체를 다 얘기하시면 중간고사 때까지 준비가 되실 거고, 한번 여기까지의 내용을 ... **여러분 다 이해를 하셔야 돼요**"

---

## 목차

1. [큰 그림 — 왜 이 학습이 중요한가](#1-큰-그림--왜-이-학습이-중요한가)
2. [수학 준비 0장 — 기호 풀이](#2-수학-준비-0장--기호-풀이)
3. [KL Divergence 개념 (수업에서 다룬 부분)](#3-kl-divergence-개념-수업에서-다룬-부분)
4. [Cross Entropy = Entropy + KL Divergence (분해 증명)](#4-cross-entropy--entropy--kl-divergence-분해-증명)
5. [KL ≥ 0 증명 (Gibbs Inequality)](#5-kl--0-증명-gibbs-inequality)
6. [데이터 분포와 KL — 왜 KL 최소화가 학습인가](#6-데이터-분포와-kl--왜-kl-최소화가-학습인가)
7. [⭐ 핵심: MLE → NLL → MSE 완전 유도](#7--핵심-mle--nll--mse-완전-유도)
8. [두 가우시안의 KL Divergence 직접 계산](#8-두-가우시안의-kl-divergence-직접-계산)
9. [σ² = 1 단순화 + MSE/2 도출](#9-σ--1-단순화--mse2-도출)
10. [전체 연결망 한 장 정리 — MAP↔MLE↔NLL↔MSE↔KL↔CE](#10-전체-연결망-한-장-정리)
11. [시험 대비 핵심 질문 5가지](#11-시험-대비-핵심-질문-5가지)

---

## 1. 큰 그림 — 왜 이 학습이 중요한가

이 한 장의 흐름이 머릿속에 들어오면 4주차 모든 내용이 손에 잡힙니다.

$$
\underbrace{\text{Bayes}}_{\text{posterior}} \;\Rightarrow\; \underbrace{\text{MAP}}_{\arg\max p(h\mid D)} \;\Rightarrow\; \underbrace{\text{MLE}}_{\text{prior=uniform}} \;\Rightarrow\; \underbrace{\text{NLL}}_{-\log L} \;\Rightarrow\; \underbrace{\text{MSE}}_{\text{Gaussian likelihood}}
$$

그리고 옆길로,

$$
\underbrace{\text{KL Divergence}}_{D_{KL}(p\Vert q)} \;=\; \underbrace{\text{Cross Entropy}}_{H(p,q)} - \underbrace{\text{Entropy}}_{H(p)}
$$

이 두 가닥이 만나서 **"학습 = empirical 분포와 모델 분포 사이 KL을 최소화 = MLE 수행 = NLL 최소화 = (Gaussian 가정 시) MSE 최소화"** 라는 결론이 나옵니다.

교수님 발언 (라인 481):
> "MSE 줄이는 게 NLL을 줄이는 것이고, NLL을 줄이는 건 likelihood를 키우는 것이죠. 이렇게 다 연결되어 있습니다."

---

## 2. 수학 준비 0장 — 기호 풀이

### 2.1 기댓값 $\mathbb{E}$

기호 $\mathbb{E}_{x \sim p}[f(x)]$ 는 "$x$ 를 분포 $p$ 에서 뽑았을 때 $f(x)$ 의 평균값" 입니다.

이산 분포라면:
$$
\mathbb{E}_{x \sim p}[f(x)] = \sum_{x} p(x) \cdot f(x)
$$

연속 분포라면:
$$
\mathbb{E}_{x \sim p}[f(x)] = \int p(x) \cdot f(x) \, dx
$$

**왜 이게 평균인가?** 각 $x$ 가 나올 확률 $p(x)$ 를 가중치로 곱해서 다 더하면 그게 가중평균이고, 무한히 많이 뽑아 평균을 내면 이 값으로 수렴 (큰 수의 법칙). 그래서 "평균"입니다.

### 2.2 확률밀도 $p(x)$, 분포 $p$

$p$ 는 분포 (어떤 값이 얼마나 자주 나오는지의 패턴), $p(x)$ 는 그 분포에서 $x$ 위치의 밀도값입니다.

### 2.3 로그 $\log$

본 자료에서 $\log$ 는 항상 자연로그 $\ln$ (밑 $e$) 를 의미합니다. 머신러닝 관습입니다.

핵심 성질:
- $\log(ab) = \log a + \log b$ (곱을 합으로)
- $\log(a^n) = n \log a$ (지수를 앞으로)
- $\log(e^x) = x$ (지수와 로그는 서로 풀어준다)
- 단조증가: $a < b \Leftrightarrow \log a < \log b$ (양수 한정)

**왜 ML에서 로그를 그렇게 쓰나?** 곱이 합으로 바뀌면 미분이 쉬워지고, $e^{(\cdots)}$ 형태가 정리되며, 단조성 덕분에 $\arg\max f = \arg\max \log f$ 가 됩니다.

### 2.4 Likelihood $L(\theta) = p(D \mid \theta)$

데이터 $D$ 가 모수 $\theta$ 의 모델에서 나왔다고 가정할 때, 이 데이터가 관측될 확률입니다. 데이터는 **고정**, $\theta$ 는 **변수** 로 보는 관점입니다.

### 2.5 정규분포(가우시안) PDF

$$
\mathcal{N}(y; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\!\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)
$$

기호 풀이:
- $\mu$ : 평균 (분포의 중심)
- $\sigma^2$ : 분산 (퍼짐 정도)
- $\frac{1}{\sqrt{2\pi\sigma^2}}$ : 전체 적분이 1이 되도록 맞추는 정규화 상수
- $\exp(-\frac{(y-\mu)^2}{2\sigma^2})$ : $y$ 가 $\mu$ 에서 멀어질수록 빠르게 작아지는 종 모양

### 2.6 $\arg\max$, $\arg\min$

$\arg\max_x f(x)$ : "$f$ 를 가장 크게 만드는 $x$ 값". 함숫값 자체가 아니라 **위치** 를 찾는다는 뜻입니다.

---

## 3. KL Divergence 개념 (수업에서 다룬 부분)

### 3.1 정의 (교수님 라인 303)

> "케일 다이먼스 K-difference 라고 하는 것은 ... $x$ 라고 하는 것을 $p$ 에서 샘플링해서 이 값의 평균이 되는 거예요. $\log p(x)$ 빼기 $\log q(x)$ 입니다."

수식으로 옮기면:

$$
\boxed{\;D_{KL}(p \Vert q) = \mathbb{E}_{x \sim p}\!\left[\log p(x) - \log q(x)\right] = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]\;}
$$

**기호 한 줄씩 해체**:
- $D_{KL}$ : Kullback–Leibler divergence 의 약자.
- $p \Vert q$ : "기준 분포 $p$ 에 대해 $q$ 가 얼마나 다른가". **이 순서는 비대칭**입니다 ($p \Vert q \neq q \Vert p$). 그래서 KL 은 진짜 거리(metric)가 아니라 "거리 같은 양(divergence)" 이라 부릅니다.
- $\mathbb{E}_{x \sim p}$ : 평균을 내는 분포는 $p$. ($q$ 가 아니라 $p$ 라는 점이 중요)
- $\log p(x) - \log q(x)$ : 같은 점 $x$ 에서 두 분포가 차지하는 밀도의 로그차.

이산형으로 풀어쓰면:
$$
D_{KL}(p \Vert q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

연속형:
$$
D_{KL}(p \Vert q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

### 3.2 두 가지 핵심 성질 (교수님 라인 305~307)

**성질 1: $p = q$ 이면 $D_{KL} = 0$**

증명:
$$
D_{KL}(p \Vert p) = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{p(x)}\right] = \mathbb{E}_{x \sim p}[\log 1] = \mathbb{E}_{x \sim p}[0] = 0
$$

각 등호 이유:
- 1단계: 분모 분자가 모두 $p(x)$ 가 되므로 분수는 1.
- 2단계: $\log 1 = 0$ (자연로그의 정의).
- 3단계: 0의 평균은 0.

**성질 2: $D_{KL}(p \Vert q) \geq 0$ (항상 음이 아님)**

증명은 [5절](#5-kl--0-증명-gibbs-inequality) 에서 자세히. 결과만 미리 말하면 두 분포가 같을 때만 등호.

### 3.3 직관 — "분포 간의 거리"

교수님 라인 317:
> "케일 다이먼스라고 하는 것은 분포 간의 거리를 재는구나, 이 개념 좀 상상하시면 됩니다."

엄밀히는 거리 함수의 공리(대칭성, 삼각부등식)를 만족하지 않지만, **"두 분포가 다를수록 큰 양수, 같으면 0"** 이라는 점에서 거리의 역할을 합니다.

---

## 4. Cross Entropy = Entropy + KL Divergence (분해 증명)

이 분해는 4주차 모든 논의의 출발점입니다.

### 4.1 Entropy 정의

$$
H(p) := -\mathbb{E}_{x \sim p}[\log p(x)] = -\sum_x p(x) \log p(x)
$$

분포 $p$ 자체가 가진 "불확실성의 양". $p$ 만 의존, $q$ 와 무관.

### 4.2 Cross Entropy 정의

$$
H(p, q) := -\mathbb{E}_{x \sim p}[\log q(x)] = -\sum_x p(x) \log q(x)
$$

"$p$ 가 진짜 분포일 때, $q$ 라는 잘못된 모델로 코딩하면 평균 몇 비트(또는 nat) 가 드는가"의 정보이론 해석.

### 4.3 분해 증명

$D_{KL}$ 정의에서 출발:

$$
\begin{aligned}
D_{KL}(p \Vert q) &= \mathbb{E}_{x \sim p}\!\left[\log p(x) - \log q(x)\right] \\
&= \mathbb{E}_{x \sim p}[\log p(x)] - \mathbb{E}_{x \sim p}[\log q(x)] \\
&= -H(p) - (-H(p, q)) \\
&= H(p, q) - H(p)
\end{aligned}
$$

각 등호 이유:
- 1행: KL 정의 그대로.
- 2행: 기댓값의 선형성 ($\mathbb{E}[X - Y] = \mathbb{E}[X] - \mathbb{E}[Y]$).
- 3행: $\mathbb{E}_{x \sim p}[\log p(x)] = -H(p)$ (Entropy 정의), $\mathbb{E}_{x \sim p}[\log q(x)] = -H(p,q)$ (Cross Entropy 정의).
- 4행: 부호 정리.

따라서:

$$
\boxed{\;H(p, q) = H(p) + D_{KL}(p \Vert q)\;}
$$

### 4.4 학습 관점에서 이 분해의 의미

- $H(p)$ 는 "데이터 자체의 본질적 불확실성" 으로 모델 $q$ 와 무관 → **상수**.
- 따라서 모델 $q$ 를 학습할 때:
$$
\arg\min_q H(p, q) = \arg\min_q D_{KL}(p \Vert q)
$$
- **CE 최소화 ⟺ KL 최소화**. 두 손실은 학습 관점에서 동일.

---

## 5. KL ≥ 0 증명 (Gibbs Inequality)

교수님이 라인 307에서 "$KL$ 은 0 이상의 값을 가집니다" 라고 단언만 하셨지만, 시험에서 증명을 요구할 수 있으므로 직접 증명합니다.

### 5.1 보조정리: $\log x \leq x - 1$ (Jensen 또는 미분으로)

함수 $g(x) = x - 1 - \log x$ 의 미분:
$$
g'(x) = 1 - \frac{1}{x}
$$
- $x = 1$ 에서 $g'(1) = 0$.
- $x < 1$ 에서 $g'(x) < 0$ → 감소.
- $x > 1$ 에서 $g'(x) > 0$ → 증가.

따라서 $g$ 는 $x = 1$ 에서 최솟값 $g(1) = 0$. 즉:
$$
g(x) \geq 0 \;\Leftrightarrow\; \log x \leq x - 1 \quad (\text{등호는 } x = 1)
$$

### 5.2 본 정리 증명

부호를 바꿔 $-D_{KL}$ 부터 위로 한정:
$$
-D_{KL}(p \Vert q) = \mathbb{E}_{x \sim p}\!\left[\log \frac{q(x)}{p(x)}\right]
$$

5.1 보조정리에 $x = q(x)/p(x)$ 대입 ($p(x) > 0$ 인 $x$ 에 한정):
$$
\log \frac{q(x)}{p(x)} \leq \frac{q(x)}{p(x)} - 1
$$

양변에 $p(x)$ 곱하고 $x$ 에 대해 합하면:
$$
\sum_x p(x) \log \frac{q(x)}{p(x)} \leq \sum_x p(x) \left(\frac{q(x)}{p(x)} - 1\right) = \sum_x q(x) - \sum_x p(x) = 1 - 1 = 0
$$

따라서:
$$
-D_{KL}(p \Vert q) \leq 0 \;\Leftrightarrow\; D_{KL}(p \Vert q) \geq 0 \quad \blacksquare
$$

등호는 모든 $x$ 에서 $q(x)/p(x) = 1$ 즉 $p = q$ 일 때만.

---

## 6. 데이터 분포와 KL — 왜 KL 최소화가 학습인가

교수님 라인 309~313 핵심 메시지:
> "이 데이터를 가장 잘 설명하는 분포를 찾고 싶은 거예요. 이쪽에다가 이 데이터의 분포를 넣습니다. 데이터 분포는 empirical distribution 이에요. 이 empirical distribution을 여기다가 넣어놓고 그거랑 가장 가까운 분포를 찾는데 ..."

### 6.1 Empirical Distribution

데이터 $\{x^{(1)}, x^{(2)}, \ldots, x^{(n)}\}$ 가 주어졌을 때 경험적 분포는:

$$
\hat{p}_{\text{data}}(x) = \frac{1}{n} \sum_{i=1}^{n} \delta(x - x^{(i)})
$$

각 데이터 포인트에 똑같이 $1/n$ 의 질량을 둔 분포 ($\delta$ 는 디랙 델타로, "그 점에만 모든 질량이 모여 있다" 는 뜻).

### 6.2 KL 최소화 = Cross Entropy 최소화 = MLE

목적 함수:

$$
\min_q D_{KL}(\hat{p}_{\text{data}} \Vert q_\theta)
$$

[4.4](#44-학습-관점에서-이-분해의-의미) 결과에 의해 $H(\hat{p}_{\text{data}})$ 가 상수이므로:

$$
\arg\min_\theta D_{KL}(\hat{p}_{\text{data}} \Vert q_\theta) = \arg\min_\theta H(\hat{p}_{\text{data}}, q_\theta)
$$

Cross entropy 를 풀어쓰면:
$$
H(\hat{p}_{\text{data}}, q_\theta) = -\sum_x \hat{p}_{\text{data}}(x) \log q_\theta(x) = -\frac{1}{n} \sum_{i=1}^{n} \log q_\theta(x^{(i)})
$$

즉:
$$
\arg\min_\theta H(\hat{p}_{\text{data}}, q_\theta) = \arg\min_\theta \left[-\frac{1}{n}\sum_{i=1}^{n} \log q_\theta(x^{(i)})\right] = \arg\max_\theta \sum_{i=1}^{n} \log q_\theta(x^{(i)})
$$

마지막 식의 $\sum_i \log q_\theta(x^{(i)})$ 가 바로 **log-likelihood** 입니다 (i.i.d 가정 하).

**결론**:
$$
\boxed{\;\text{KL 최소화} \;\Leftrightarrow\; \text{CE 최소화} \;\Leftrightarrow\; \text{NLL 최소화} \;\Leftrightarrow\; \text{MLE}\;}
$$

이 사슬이 4주차 강의의 척추입니다.

---

## 7. ⭐ 핵심: MLE → NLL → MSE 완전 유도

> 교수님이 4주차에서 가장 강조하신 부분 (라인 461~519, 중요도 10/10).

### 7.1 문제 설정 (회귀)

데이터: $\{(x^{(i)}, y^{(i)})\}_{i=1}^{n}$, supervised learning.

가설: 함수 $h$ 가 입력 $x$ 를 받아 출력 $y$ 를 예측. 단, **관측치는 잡음을 포함** 한다고 가정:
$$
y = h(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

기호 풀이:
- $h$ : 우리가 찾고 싶은 함수 (가설).
- $\varepsilon$ : 잡음. 평균 0, 분산 $\sigma^2$ 인 정규분포에서 독립적으로 뽑힘.

이는 곧:
$$
y \mid x, h \sim \mathcal{N}(h(x), \sigma^2)
$$

즉 $x, h$ 가 주어졌을 때 $y$ 의 조건부 분포는 평균 $h(x)$, 분산 $\sigma^2$ 의 정규분포.

교수님 표현 (라인 471~473):
> "$x$ 가 주어졌으니까 평균은 $h(x)$ 고 분산이 $\sigma^2$ 입니다. $y$ 가 여기서 나오는 거죠."

### 7.2 Likelihood 작성

i.i.d. 가정 하 전체 likelihood:
$$
L(h) = p(\{y^{(i)}\} \mid \{x^{(i)}\}, h) = \prod_{i=1}^{n} p(y^{(i)} \mid x^{(i)}, h)
$$

각 항은 가우시안 PDF:
$$
p(y^{(i)} \mid x^{(i)}, h) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y^{(i)} - h(x^{(i)}))^2}{2\sigma^2}\right)
$$

### 7.3 Log-Likelihood

곱이 합으로:
$$
\log L(h) = \sum_{i=1}^{n} \log p(y^{(i)} \mid x^{(i)}, h)
$$

각 항을 전개:
$$
\log p(y^{(i)} \mid x^{(i)}, h) = \log \frac{1}{\sqrt{2\pi\sigma^2}} + \log \exp\!\left(-\frac{(y^{(i)} - h(x^{(i)}))^2}{2\sigma^2}\right)
$$

$\log \exp(z) = z$ 이용:
$$
= -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y^{(i)} - h(x^{(i)}))^2}{2\sigma^2}
$$

각 등호 이유:
- $\log(ab) = \log a + \log b$ 로 분리.
- $\log\frac{1}{\sqrt{2\pi\sigma^2}} = -\log\sqrt{2\pi\sigma^2} = -\frac{1}{2}\log(2\pi\sigma^2)$.
- $\log e^z = z$ 로 지수 풀이.

전체 합:
$$
\log L(h) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y^{(i)} - h(x^{(i)}))^2
$$

### 7.4 Negative Log-Likelihood (NLL)

부호 뒤집기:
$$
\text{NLL}(h) = -\log L(h) = \underbrace{\frac{n}{2}\log(2\pi\sigma^2)}_{\text{상수 (}h\text{ 무관)}} + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y^{(i)} - h(x^{(i)}))^2
$$

### 7.5 NLL 최소화 = MSE 최소화

상수항은 $h$ 와 무관하므로 $\arg\min$ 에서 무시 가능. 양수 $\frac{1}{2\sigma^2}$ 도 최소화 위치에 영향 없음:

$$
\arg\min_h \text{NLL}(h) = \arg\min_h \sum_{i=1}^{n}(y^{(i)} - h(x^{(i)}))^2
$$

$\frac{1}{n}$ 로 나누어 평균화하면:
$$
\boxed{\;\arg\min_h \text{NLL}(h) = \arg\min_h \underbrace{\frac{1}{n}\sum_{i=1}^{n}(y^{(i)} - h(x^{(i)}))^2}_{\text{MSE}(h)}\;}
$$

### 7.6 교수님이 한 줄로 요약하신 핵심 (라인 481)

> "**MSE 줄이는 것이 NLL 줄이는 것이고, NLL 줄이는 것은 likelihood 키우는 것이죠. 이렇게 다 연결되어 있습니다.**"

### 7.7 유도 흐름 다이어그램

$$
\underbrace{y \mid x, h \sim \mathcal{N}(h(x), \sigma^2)}_{\text{Gaussian likelihood 가정}}
\;\xrightarrow{\text{i.i.d.}}\;
L(h) = \prod_i p(y^{(i)}\mid x^{(i)}, h)
$$

$$
\;\xrightarrow{\log}\; \log L(h) \;\xrightarrow{-1\text{ 곱하기}}\; \text{NLL}(h)
$$

$$
\;\xrightarrow{\text{상수 제거}}\; \frac{1}{2\sigma^2}\sum_i (y^{(i)} - h(x^{(i)}))^2
$$

$$
\;\xrightarrow{\frac{1}{n}\text{ 곱하기}}\; \boxed{\text{MSE}(h)}
$$

### 7.8 왜 MSE 가 "당연한" 회귀 손실인가의 답

MSE 는 임의로 정해진 손실이 아니라, **"잡음이 가우시안이라는 가정 하에서 likelihood 를 최대화하는 자연스러운 손실함수"** 입니다.

교수님 라인 485:
> "이게 이제 가오스가 가오시안 디스플레이션이라고 불리는 이유입니다. 가오스가 ... 외성인데 얘의 궤도를 예측하는 걸 했었어요. 궤도는 위치죠. 위치니까 리그레이션 문제입니다. ... 그때 사용했던 방법이 앞에서 설명드렸던 그 방법이라서 가우시안 디스플레이션이라고 썼던 거에요."

즉 가우스가 19개의 천체 관측 데이터로 궤도를 추정할 때 정확히 이 유도를 사용했고, 그 결과 정규분포가 "가우시안" 이름을 얻었다는 역사적 일화입니다.

---

## 8. 두 가우시안의 KL Divergence 직접 계산

> 교수님은 "시간 관계상 넘어가겠습니다" (라인 319) 로 결과만 보여주셨지만, 시험 출제 가능성이 있어 완전 유도합니다.

### 8.1 설정

두 1차원 가우시안:
$$
p(x) = \mathcal{N}(x; \mu_1, \sigma_1^2), \qquad q(x) = \mathcal{N}(x; \mu_2, \sigma_2^2)
$$

목표: $D_{KL}(p \Vert q) = \mathbb{E}_{x \sim p}[\log p(x) - \log q(x)]$

### 8.2 단계별 유도

**Step 1.** 각 로그 풀이:
$$
\log p(x) = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x - \mu_1)^2}{2\sigma_1^2}
$$
$$
\log q(x) = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x - \mu_2)^2}{2\sigma_2^2}
$$

**Step 2.** 차이:
$$
\log p(x) - \log q(x) = -\frac{1}{2}\log\frac{\sigma_1^2}{\sigma_2^2} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

상수 정리:
$$
-\frac{1}{2}\log\frac{\sigma_1^2}{\sigma_2^2} = \frac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} = \log\frac{\sigma_2}{\sigma_1}
$$

**Step 3.** $\mathbb{E}_{x \sim p}$ 적용. 두 가지 보조 사실 사용:

(a) $\mathbb{E}_{x \sim p}[(x - \mu_1)^2] = \sigma_1^2$ (분산의 정의)

(b) $\mathbb{E}_{x \sim p}[(x - \mu_2)^2]$ 계산. $(x - \mu_2)^2 = ((x - \mu_1) + (\mu_1 - \mu_2))^2$ 전개:
$$
= (x - \mu_1)^2 + 2(x - \mu_1)(\mu_1 - \mu_2) + (\mu_1 - \mu_2)^2
$$

기댓값:
$$
\mathbb{E}_{x \sim p}[(x - \mu_2)^2] = \sigma_1^2 + 2 \cdot 0 \cdot (\mu_1 - \mu_2) + (\mu_1 - \mu_2)^2 = \sigma_1^2 + (\mu_1 - \mu_2)^2
$$

(중간항이 0인 이유: $\mathbb{E}_{x \sim p}[x - \mu_1] = 0$.)

**Step 4.** 종합:
$$
\begin{aligned}
D_{KL}(p \Vert q) &= \log\frac{\sigma_2}{\sigma_1} - \frac{\mathbb{E}[(x-\mu_1)^2]}{2\sigma_1^2} + \frac{\mathbb{E}[(x-\mu_2)^2]}{2\sigma_2^2} \\
&= \log\frac{\sigma_2}{\sigma_1} - \frac{\sigma_1^2}{2\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} \\
&= \log\frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}
\end{aligned}
$$

따라서 최종 공식:

$$
\boxed{\;D_{KL}\!\left(\mathcal{N}(\mu_1, \sigma_1^2) \,\Vert\, \mathcal{N}(\mu_2, \sigma_2^2)\right) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}\;}
$$

### 8.3 일관성 검증

$p = q$ ($\mu_1 = \mu_2$, $\sigma_1 = \sigma_2$):
$$
D_{KL} = \log 1 + \frac{\sigma^2 + 0}{2\sigma^2} - \frac{1}{2} = 0 + \frac{1}{2} - \frac{1}{2} = 0 \;\checkmark
$$

성질 1과 부합.

---

## 9. σ² = 1 단순화 + MSE/2 도출

### 9.1 단순화

$\sigma_1^2 = \sigma_2^2 = 1$ 대입:
$$
\begin{aligned}
D_{KL} &= \log\frac{1}{1} + \frac{1 + (\mu_1 - \mu_2)^2}{2 \cdot 1} - \frac{1}{2} \\
&= 0 + \frac{1}{2} + \frac{(\mu_1 - \mu_2)^2}{2} - \frac{1}{2} \\
&= \boxed{\frac{(\mu_1 - \mu_2)^2}{2}}
\end{aligned}
$$

### 9.2 MSE 와의 직접 연결

학습 시나리오 (4주차 회귀 설정 [7절](#7--핵심-mle--nll--mse-완전-유도)) 에 대입:
- "true 분포" 를 $p = \mathcal{N}(y, 1)$ ($\mu_1 = y$, 등분산 1)
- "모델 분포" 를 $q = \mathcal{N}(\mu_\theta(x), 1)$ ($\mu_2 = \mu_\theta(x)$, 등분산 1)

그러면:
$$
D_{KL}(p \Vert q) = \frac{(y - \mu_\theta(x))^2}{2} = \frac{1}{2} \cdot \text{(squared error)}
$$

전체 데이터셋에 평균:
$$
\frac{1}{n}\sum_{i=1}^{n} D_{KL}(p^{(i)} \Vert q^{(i)}) = \frac{1}{2n}\sum_i (y^{(i)} - \mu_\theta(x^{(i)}))^2 = \frac{1}{2}\text{MSE}
$$

### 9.3 결론

$$
\boxed{\;\text{KL 최소화 (등분산 가우시안)} \;\Leftrightarrow\; \text{MSE 최소화}\;}
$$

이것이 7절의 가우시안 NLL → MSE 와 동일한 결과를 KL 경유 경로로 다시 얻은 것입니다. **두 경로 모두 같은 결론** 에 도달한다는 점이 시험에서 보여줄 수 있는 가장 강력한 통찰입니다.

---

## 10. 전체 연결망 한 장 정리

```
                    [Bayes Theorem]
                         │
                    p(h|D) ∝ p(D|h)·p(h)
                         │
                  ┌──────┴──────┐
                 MAP           MLE  (uniform prior)
                  │             │
                  │             │   ─── log + (-1) ───>  NLL
                  │                                       │
                  │                                       ↓
                  │           [Gaussian likelihood 가정]
                  │                                       │
                  │                                       ↓
                  │                                     MSE
                  │
                  │
[KL Divergence] ─── = ──> [Cross Entropy] - [Entropy]
       │                         │
       │                         │  (분류: softmax + one-hot true → CE Loss)
       │                         │
       │   true=N(y,1), model=N(μ,1)
       │   ────────────────────────>  (μ-y)²/2  =  MSE/2
       │
       └─── always ≥ 0, =0 iff p=q (Gibbs)
```

핵심 등가성 사슬:
$$
\arg\min_\theta D_{KL}(\hat{p}_{\text{data}} \Vert q_\theta)
\;=\; \arg\min_\theta H(\hat{p}_{\text{data}}, q_\theta)
\;=\; \arg\min_\theta \text{NLL}(\theta)
\;=\; \arg\max_\theta L(\theta)
\;\overset{\text{Gauss}}{=}\; \arg\min_\theta \text{MSE}(\theta)
$$

---

## 11. 시험 대비 핵심 질문 5가지

### Q1. KL Divergence 의 정의와 두 가지 핵심 성질은?

**답안 작성 모범**:
정의:
$$
D_{KL}(p \Vert q) = \mathbb{E}_{x \sim p}\!\left[\log\frac{p(x)}{q(x)}\right]
$$

성질:
1. $D_{KL}(p \Vert q) \geq 0$ (Gibbs inequality, 등호는 $p = q$).
2. $p = q$ 이면 $D_{KL} = 0$ (자명).
3. 비대칭: $D_{KL}(p \Vert q) \neq D_{KL}(q \Vert p)$ — 거리 함수 아님.

### Q2. Cross Entropy 와 KL 의 관계를 증명하라.

[4.3](#43-분해-증명) 의 4단계 증명을 그대로 작성. 핵심 결과:
$$
H(p, q) = H(p) + D_{KL}(p \Vert q)
$$

학습 시 $H(p)$ 가 상수이므로 CE 최소화 = KL 최소화.

### Q3. 회귀 문제에서 MSE Loss 가 왜 자연스러운가?

**답안 작성 모범**:
관측 모델 $y = h(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ 가정. 그러면:
$$
y \mid x, h \sim \mathcal{N}(h(x), \sigma^2)
$$

i.i.d. 데이터의 NLL:
$$
\text{NLL}(h) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y^{(i)} - h(x^{(i)}))^2
$$

상수항 제거 후:
$$
\arg\min_h \text{NLL}(h) = \arg\min_h \sum_{i=1}^{n}(y^{(i)} - h(x^{(i)}))^2 = \arg\min_h \text{MSE}(h)
$$

따라서 MSE = "가우시안 잡음 가정 하에서 likelihood 를 최대화하는 손실".

### Q4. 두 가우시안의 KL 일반식을 유도하라.

[8.2](#82-단계별-유도) 의 4-step 유도를 그대로 작성. 결과:
$$
D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \Vert \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

### Q5. KL 최소화가 MLE 와 같음을 증명하라.

empirical distribution $\hat{p}_{\text{data}}(x) = \frac{1}{n}\sum_i \delta(x - x^{(i)})$ 정의.

$$
\arg\min_\theta D_{KL}(\hat{p}_{\text{data}} \Vert q_\theta) \overset{(a)}{=} \arg\min_\theta H(\hat{p}_{\text{data}}, q_\theta) \overset{(b)}{=} \arg\max_\theta \sum_{i=1}^{n}\log q_\theta(x^{(i)})
$$

각 등호 이유:
- (a) $H(\hat{p}_{\text{data}})$ 는 $\theta$ 무관 상수.
- (b) Cross entropy 풀어쓰기 + 부호 뒤집기.

마지막 식이 정확히 log-likelihood 이므로 **KL 최소화 = MLE**.

---

## 부록: 강의 인용 모음

라인 번호는 `/Users/jeongseongchae/dev/university/deep_learning/docs/수업_스크립트/딥러닝이론-4주차.md` 기준.

| 라인 | 인용 | 학습 포인트 |
|------|------|-------------|
| 303 | "케일 다이먼스 K-difference라고 하는 것은 ... $\log p(x)$ 빼기 $\log q(x)$ 입니다." | KL 정의 |
| 305 | "이렇게 쪼개집니다. 크로스 엔트로피랑 그냥 엔트로피 차이 이렇게 되구요." | $H(p,q) = H(p) + KL$ |
| 307 | "p와 q가 같으면 0이 됩니다. ... 얘는 0 이상의 값을 가집니다." | 두 핵심 성질 |
| 309~313 | "데이터 분포 (empirical) 를 넣어놓고 ... 가장 가까운 분포를 찾는데" | KL 최소화 = 학습 |
| 317 | "케일 다이먼스라고 하는 것은 분포 간의 거리를 재는구나" | 직관: 거리 |
| 319 | "이것도 직접 좀 해보시는 게 좋은데 시간 관계상 넘어가겠습니다" | 두 가우시안 KL 자율 풀이 |
| 461~475 | "$y$ 가 가우시안 ... 로그 취하면 exponential 사라지죠 ... 마이너스 붙여서 NLL" | 가우시안 likelihood → NLL |
| 479 | "제곱오차의 합이 ... MSE 라고 부릅니다" | NLL → MSE |
| 481 | "MSE 줄이는 게 NLL 줄이는 것이고, NLL 줄이는 건 likelihood 키우는 것" | ⭐ 핵심 사슬 |
| 485 | "이게 가오스가 가우시안 디스플레이션이라고 불리는 이유입니다" | 역사적 일화 |
| 489~491 | "NLL이 MSE 랑 연결, K-Divergence 와도 연결, Empirical Risk 와도 연결" | 전체 망 |
| 493 | "**여러분 다 이해를 하셔야 돼요**" | 시험 강조 |

---

**이 자료의 사용법**:
1. 처음 읽을 때는 [1절](#1-큰-그림--왜-이-학습이-중요한가) 큰 그림과 [10절](#10-전체-연결망-한-장-정리) 다이어그램을 먼저 보고 골격 잡기.
2. [7절](#7--핵심-mle--nll--mse-완전-유도) 을 손으로 직접 따라 유도해보기 (시험 핵심).
3. [4절](#4-cross-entropy--entropy--kl-divergence-분해-증명), [8절](#8-두-가우시안의-kl-divergence-직접-계산), [9절](#9-σ--1-단순화--mse2-도출) 은 백지에서 재현 가능할 때까지 반복.
4. [11절](#11-시험-대비-핵심-질문-5가지) 5문항을 시험 직전 자가테스트로.

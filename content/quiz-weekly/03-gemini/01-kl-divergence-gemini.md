---
title: "딥러닝 이론 핵심 학습 자료: KL Divergence와 손실 함수의 대통합"
slug: kl-divergence-gemini
order: 1
---

# 딥러닝 이론 핵심 학습 자료: KL Divergence와 손실 함수의 대통합

본 자료는 수업 시간에 매우 중요하게 다뤄진(중요도 8/10 ~ 10/10) **KL Divergence의 개념**과 그것이 어떻게 **Cross-Entropy, MLE, 그리고 최종적으로 MSE(평균제곱오차)로 연결되는지**에 대한 완벽한 수학적 유도와 직관적 이해를 돕기 위해 작성되었습니다.

---

## 1. KL Divergence (쿨백-라이블러 발산) 기본 개념 (중요도 8/10)

### 1.1 개념 및 정의
KL Divergence는 두 확률 분포 $P$와 $Q$가 **얼마나 다르게 생겼는지(거리)**를 측정하는 수학적 척도입니다. 딥러닝에서는 보통 $P$를 '실제 정답 데이터의 분포(Empirical Distribution)', $Q$를 '인공지능 모델의 예측 분포(Model Distribution)'로 둡니다.

**수학적 정의:**
$$D_{\mathrm{KL}}(P \parallel Q) = \mathbb{E}_{x \sim P} \left[ \log P(x) - \log Q(x) \right]$$

로그의 성질을 이용해 위 식을 두 가지 엔트로피의 결합으로 쪼개어 표현할 수 있습니다.
$$D_{\mathrm{KL}}(P \parallel Q) = -H(P) + H(P, Q)$$
- $H(P) = -\mathbb{E}_{x \sim P}[\log P(x)]$: **Entropy (엔트로피)** - 정답 분포 자체의 내재적인 불확실성
- $H(P, Q) = -\mathbb{E}_{x \sim P}[\log Q(x)]$: **Cross-Entropy (크로스 엔트로피)** - 정답 분포 $P$를 모델 $Q$의 시각으로 바라봤을 때의 놀람의 정도

### 1.2 주요 성질
1. **항상 0 이상 ($D_{\mathrm{KL}} \ge 0$)**: 거리를 의미하므로 음수가 될 수 없습니다. (Jensen's Inequality를 통해 증명 가능)
2. **동일할 때 0 ($P = Q \implies D_{\mathrm{KL}} = 0$)**: 두 분포가 완벽히 겹치면 거리는 0이 됩니다.
3. **비대칭성 ($D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P)$)**: 완전한 물리적 거리가 아니라 '발산(Divergence)'이라고 부르는 이유입니다.

---

## 2. 두 분포 간의 거리를 줄이는 것 = MLE = Cross-Entropy 최소화

딥러닝의 궁극적인 학습 목표는 모델의 예측 분포 $Q_\theta$를 정답 데이터의 분포 $P$와 최대한 똑같이 만드는 것입니다. 즉, **KL Divergence를 최소화**하는 최적의 파라미터 $\theta$를 찾는 것입니다.

$$ \arg\min_{\theta} D_{\mathrm{KL}}(P \parallel Q_\theta) = \arg\min_{\theta} \left( H(P, Q_\theta) - H(P) \right) $$

이 수식의 전개에서 딥러닝 최적화의 놀라운 연결고리가 탄생합니다.

1. **$H(P)$는 무시된다**: 정답 데이터의 분포 $P$는 이미 세상에 주어져 있는 고정된 값입니다. 모델 파라미터 $\theta$와 무관하므로 상수 취급되어 미분 및 최소화 과정에서 사라집니다.
2. **KL 최소화 = Cross-Entropy 최소화**: 결국 최소화해야 할 대상은 $H(P, Q_\theta)$뿐입니다. 즉, KL 발산을 줄이는 고상한 통계적 노력은, 단순히 **Cross-Entropy를 최소화**하는 연산과 완벽하게 동치입니다.
3. **Cross-Entropy = Negative Log-Likelihood (NLL)**:
   $H(P, Q_\theta) \approx -\frac{1}{N} \sum_{i=1}^N \log Q_\theta(y_i \mid x_i)$
   이 식은 주어진 데이터의 우도(Likelihood)에 로그를 씌우고 마이너스를 붙인 NLL의 표본 평균과 완전히 똑같습니다. 

**[핵심 연결 고리 요약]**
> KL Divergence 최소화 $\Longleftrightarrow$ Cross-Entropy 최소화 $\Longleftrightarrow$ MLE (Maximum Likelihood Estimation) 달성 $\Longleftrightarrow$ NLL (Negative Log-Likelihood) 최소화

---

## 3. Negative Log-Likelihood (NLL) ↔ MSE 연결 (중요도 10/10)

분류(Classification) 문제에서는 모델의 출력에 Softmax를 씌워 Cross-Entropy를 그대로 손실 함수로 씁니다. 그렇다면 수치를 예측하는 **회귀(Regression) 문제에서는 왜 오차를 제곱하는 MSE(평균제곱오차)를 쓸까요?** 교수님이 10점 만점으로 강조하신 "가우시안 가정 기반의 NLL 유도" 증명이 바로 여기에 있습니다.

**가정: 모델의 오차는 정규분포(Gaussian)를 따른다.**
모델의 예측값을 $h(x)$, 실제 값을 $y$라고 할 때, 오차 $\epsilon = y - h(x)$ 가 평균 0, 분산 $\sigma^2$인 가우시안 정규 분포를 따른다고 가정해 봅시다. (자연계 수많은 잡음 요인들의 합은 정규분포에 수렴한다는 중심극한정리(CLT)에 기반합니다.)

### 3.1 가우시안 우도 함수 (Gaussian Likelihood)
데이터가 주어졌을 때 이 모델이 뱉어낼 확률(우도)은 정규분포 공식에 의해 다음과 같이 정의됩니다.
$$ P(y_i \mid x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(y_i - h(x_i))^2}{2\sigma^2} \right) $$

### 3.2 NLL 유도 과정
학습은 우도를 최대화(MLE)하는 것이며, 이는 우도에 마이너스 로그($-\log$)를 씌운 NLL을 최소화하는 것과 같습니다. 위의 식에 $-\log$를 씌워 전개해 보겠습니다.

$$ \text{NLL} = -\log P(y_i \mid x_i, \theta) = -\log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) - \log \left( \exp \left( -\frac{(y_i - h(x_i))^2}{2\sigma^2} \right) \right) $$

- 첫 번째 항 $-\log(\frac{1}{\sqrt{2\pi\sigma^2}})$는 학습시킬 파라미터 $\theta$와 무관한 **상수(Constant)**이므로 최적화 미분 시 날아갑니다.
- 두 번째 항에서 $\log$와 $\exp$가 만나 서로 상쇄됩니다. 마이너스 기호 역시 앞의 마이너스와 만나 양수($+$)가 됩니다.

결국 파라미터 최적화를 위한 NLL 식은 아래와 같이 매우 단순하게 정리됩니다.
$$ \arg\min_{\theta} \text{NLL}(\theta) = \arg\min_{\theta} \sum_{i=1}^N \frac{(y_i - h(x_i))^2}{2\sigma^2} $$

### 3.3 결론: NLL이 곧 MSE다!
분모의 $2\sigma^2$ 역시 변하지 않는 단순한 상수 배율이므로, 최소화 문제에서는 이를 무시할 수 있습니다. 전체 데이터 N개에 대해 평균을 내면 남는 것은 오직 오차의 제곱합입니다.

$$ \text{최종 최소화 목표} = \frac{1}{N} \sum_{i=1}^N (y_i - h(x_i))^2 = \mathbf{MSE} $$

### 🌟 딥러닝 손실 함수의 마스터 키 (대통합 결론)
교수님이 수업 중 **"Loss Function이 먼저가 아니다, NLL이 먼저다"**라고 역설하신 이유가 바로 이 증명에 있습니다.
1. **분류(Classification)**: 데이터 분포가 이항/다항(Bernoulli/Multinomial) 분포를 따른다고 가정하면, NLL을 전개했을 때 **Cross-Entropy Loss**가 튀어나옵니다.
2. **회귀(Regression)**: 데이터 오차가 정규 분포(Gaussian)를 따른다고 가정하면, NLL을 전개했을 때 수학적으로 완벽하게 **MSE Loss**가 튀어나옵니다.

즉, 우리가 딥러닝에서 무심코 사용하던 MSE와 Cross-Entropy는 모두 **"KL Divergence(두 분포 간 거리)를 좁히기 위해 최대우도(MLE)를 찾아가는 NLL이라는 단 하나의 뿌리"**에서 뻗어나온 필연적인 결과물입니다!

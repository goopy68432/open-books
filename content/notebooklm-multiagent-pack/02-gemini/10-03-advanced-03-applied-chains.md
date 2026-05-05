---
title: "Level 3: Advanced - 손실 함수로의 응용 (Applied Chains)"
slug: 03-advanced-03-applied-chains
order: 10
---

# Level 3: Advanced - 손실 함수로의 응용 (Applied Chains)

> **학습 목표:** 딥러닝에서 당연하게 쓰는 손실 함수들(MSE, Cross-Entropy)과 정규화(L2)가 도대체 어디서 뚝 떨어진 것인지, 확률 모델(Gaussian, Bernoulli, MAP)을 통해 수학적으로 증명합니다.

---

## 1. 정규분포(Gaussian) 모델링이 MSE를 만든다

**주장:** 오차(Noise)가 정규분포를 따른다고 가정하고 최대우도추정(MLE)을 하면, 결과적으로 평균제곱오차(MSE)를 최소화하는 것과 완벽히 똑같다.

**유도 체인:**
1. **모델 가정:** 실제 정답 $y_i$ 는 인공지능의 예측값 $f(x_i)$ 에다가 알 수 없는 노이즈 $\epsilon_i$ 가 낀 것입니다. 즉, $y_i = f(x_i) + \epsilon_i$.
2. **정규분포 가정:** 이 노이즈 $\epsilon_i$ 가 평균이 0이고 분산이 $\sigma^2$ 인 정규분포 $\mathcal{N}(0, \sigma^2)$ 를 따른다고 가정합시다.
   * 그러면 $y_i$ 도 평균이 $f(x_i)$ 인 정규분포 $\mathcal{N}(f(x_i), \sigma^2)$ 를 따르게 됩니다.
3. **Likelihood 작성 (i.i.d):**
   * $L = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - f(x_i))^2}{2\sigma^2} \right)$
4. **NLL (Negative Log Likelihood) 변환:**
   * $-\log L = - \sum_{i=1}^n \left[ \log(\text{상수}) - \frac{(y_i - f(x_i))^2}{2\sigma^2} \right]$
   * $= \text{상수} + \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - f(x_i))^2$
5. **결론:**
   * NLL을 최소화하는 파라미터를 찾는 과정에서, '상수' 부분은 어차피 파라미터와 무관하여 미분하면 0이 되어 날아갑니다.
   * 결국 남는 것은 **$\sum_{i=1}^n (y_i - f(x_i))^2$ (MSE, 오차의 제곱합)을 최소화**하는 문제로 귀결됩니다.

---

## 2. 베르누이(Bernoulli) 모델링이 Cross-Entropy를 만든다

**주장:** 데이터가 0 또는 1의 이진 분류 문제(Bernoulli)일 때 MLE를 수행하면, 딥러닝의 이진 교차 엔트로피(BCE, Binary Cross-Entropy) 손실 함수가 그대로 유도된다.

**유도 체인:**
1. **모델 가정:** 인공지능의 출력 $p_i$ 는 1(개)이 될 확률을 의미합니다. 정답 $y_i$ 는 1 또는 0입니다.
2. **베르누이 PMF:** $p(y_i) = p_i^{y_i} (1-p_i)^{1-y_i}$
3. **Likelihood (i.i.d):** $L = \prod_{i=1}^n p_i^{y_i} (1-p_i)^{1-y_i}$
4. **NLL (Negative Log Likelihood):**
   * $-\log L = - \sum_{i=1}^n [ y_i \log(p_i) + (1-y_i) \log(1-p_i) ]$
5. **결론:**
   * 이 수식 자체가 바로 정보이론에서 말하는 **Cross-Entropy 공식과 100% 일치**합니다. (Pytorch의 `BCELoss`가 바로 이것입니다!)

---

## 3. 정규분포 사전확률(Prior)이 L2 정규화를 만든다 (MAP)

**주장:** 가중치(Weight, $w$)가 정규분포를 따른다는 선입견(Prior)을 주고 MAP 추론을 하면, 자연스럽게 손실 함수 뒤에 꼬리표로 붙는 L2 Regularization(가중치 감쇠, Weight Decay) 항이 유도된다.

**유도 체인:**
1. **베이즈 정리:** $\log \text{Posterior} = \log \text{Likelihood} + \log \text{Prior}$
2. **Likelihood 부분:** 위 1번 증명에서 보았듯, Gaussian 노이즈 가정 하에서 Negative Log Likelihood는 **MSE** 가 됩니다.
3. **Prior 부분:**
   * 가중치 $w$가 $0$ 근처에 모여있을 것이라는 선입견, 즉 $w \sim \mathcal{N}(0, \lambda^{-1})$ 을 가정합니다. (정확히는 분산의 역수가 정규화 강도와 비례합니다)
   * $\log \text{Prior} = \log \left( c \cdot \exp(-\frac{\lambda}{2} \|w\|^2) \right) = \text{상수} - \frac{\lambda}{2} \|w\|^2$
   * 여기에 음수(Negative)를 취하면 $+\frac{\lambda}{2} \|w\|^2$ 가 됩니다.
4. **결론:**
   * 음의 로그 사후확률 (Negative Log Posterior)을 최소화하는 MAP 문제는:
   * **$\text{Loss} = \text{MSE}(y, f(x)) + \frac{\lambda}{2} \|w\|^2$** 가 됩니다.
   * 즉, L2 정규화($\lambda \|w\|^2$) 항은 엔지니어들이 억지로 만들어낸 것이 아니라, **"가중치는 0 주변의 정규분포일 거야"라는 베이지안 선입견(Prior)의 수학적 발현**입니다.

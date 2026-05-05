---
title: "📝 [V3] Model Answer - Q3. 손실함수와 KL Divergence (Loss & Information Theory)"
slug: 03-model-answer-q3
order: 4
---

# 📝 [V3] Model Answer - Q3. 손실함수와 KL Divergence (Loss & Information Theory)

**[평가 기준]**
* $KL \ge 0$ 증명 시 젠센 부등식을 역수와 함께 정확히 적용했는가?
* 가우시안 분산이 동일할 때, 기댓값 연산자 $\mathbb{E}$ 내부의 변수 치환을 제대로 수행했는가?
* MSE ↔ MLE ↔ NLL ↔ KL의 연결 관계(파이프라인)를 상속 개념으로 논리적으로 서술했는가?

---

## (a) Jensen 부등식을 이용한 $KL(P||Q) \ge 0$ 증명

**[정의]**
$$KL(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$$

**[증명 과정]**
1. **역수 취하기 (부호 반전):**
   $\log$ 안의 분수를 뒤집어 앞에 마이너스($-$)를 붙입니다.
   $= - \int P(x) \log \frac{Q(x)}{P(x)} dx = - \mathbb{E}_{x \sim P} \left[ \log \frac{Q(x)}{P(x)} \right]$
2. **젠센 부등식(Jensen's Inequality) 적용:**
   $\log$ 함수는 위로 볼록(Concave)하므로, $\mathbb{E}[\log(X)] \le \log(\mathbb{E}[X])$ 가 성립합니다.
   하지만 식 앞에 음수($-$)가 있으므로 부등호 방향이 반전되어 $\ge$ 가 됩니다.
   $\ge - \log \left( \mathbb{E}_{x \sim P} \left[ \frac{Q(x)}{P(x)} \right] \right)$
3. **기댓값 전개 및 약분 (Garbage Collection):**
   $\mathbb{E}_{x \sim P} \left[ \frac{Q(x)}{P(x)} \right] = \int P(x) \frac{Q(x)}{P(x)} dx = \int Q(x) dx = 1$
   (어떤 확률분포든 전체 구간 적분 합은 무조건 1입니다.)
4. **결론 도출:**
   $\ge - \log(1) = 0$
   $$\therefore KL(P || Q) \ge 0$$

---

## (b) 두 가우시안 분포 간의 KL Divergence 계산

**[조건]** $P \sim \mathcal{N}(\mu_1, \sigma^2), \quad Q \sim \mathcal{N}(\mu_2, \sigma^2)$

**[증명 전개]**
$$KL(P || Q) = \mathbb{E}_{x \sim P} [\log P(x) - \log Q(x)]$$
1. **분산 동일 조건으로 상수항 상쇄:**
   가우시안의 앞부분 상수 $\log(\frac{1}{\sqrt{2\pi\sigma^2}})$ 는 두 분포 모두 동일하므로 뺄셈에서 날아갑니다. 지수부의 $\exp$ 는 $\log$ 와 상쇄됩니다.
   $= \mathbb{E}_{x \sim P} \left[ -\frac{(x-\mu_1)^2}{2\sigma^2} + \frac{(x-\mu_2)^2}{2\sigma^2} \right]$
2. **제곱식 전개 및 $x^2$ 항 소거:**
   $= \mathbb{E}_{x \sim P} \left[ \frac{2x(\mu_1 - \mu_2) + \mu_2^2 - \mu_1^2}{2\sigma^2} \right]$
3. **기댓값 치환 (킬러 포인트):**
   $\mathbb{E}_{x \sim P}[x]$ 에 의해 변수 $x$ 는 정답 분포 $P$ 의 평균인 $\mu_1$ 으로 치환됩니다.
   $= \frac{2\mu_1(\mu_1 - \mu_2) + \mu_2^2 - \mu_1^2}{2\sigma^2}$
4. **결론 도출:**
   $= \frac{2\mu_1^2 - 2\mu_1\mu_2 + \mu_2^2 - \mu_1^2}{2\sigma^2} = \frac{\mu_1^2 - 2\mu_1\mu_2 + \mu_2^2}{2\sigma^2}$
   $$\therefore KL(P || Q) = \frac{(\mu_1 - \mu_2)^2}{2\sigma^2}$$

---

## (c) 전체 관계 증명 (MSE ↔ NLL ↔ KL)

이 문제는 딥러닝 손실 함수의 설계 아키텍처를 묻는 핵심 문제입니다.

**[1. Gaussian 가정 $\rightarrow$ NLL 최소화 $\rightarrow$ MSE 도출]**
*   **가정:** 오차가 정규분포 $\mathcal{N}(0, \sigma^2)$ 를 따른다고 가정. $y \sim \mathcal{N}(\hat{y}, \sigma^2)$.
*   **NLL 도출:** $NLL(\theta) = - \sum \log P(y_i | x_i) = - \sum \left[ \log(\text{상수}) - \frac{(y_i - \hat{y}_i)^2}{2\sigma^2} \right]$
*   **상수 제거:** 최적화(Optimization) 시 파라미터 $\theta$ 와 무관한 $\log(\text{상수})$ 항은 날아가고, 분모의 $2\sigma^2$ 은 스케일 상수이므로 무시됩니다.
*   **결과:** $NLL \propto \sum (y_i - \hat{y}_i)^2 = MSE$. 즉, 가우시안 에러를 가정하고 NLL을 최소화하는 로직은 MSE를 최소화하는 것과 완벽히 일치합니다.

**[2. KL 최소화 $\leftrightarrow$ NLL 최소화 연결]**
*   $KL(P || Q) = \sum P \log P - \sum P \log Q = - H(P) + H(P, Q)$
*   **해석:** 정답 분포의 엔트로피 $-H(P)$ 는 파라미터 튜닝으로 바꿀 수 없는 고정된 상수(노이즈)입니다. 따라서 이 상수를 제거하면 $Q$ 에 대한 미분 최적화 시 영향을 미치지 않습니다.
*   **결과:** 남은 항은 크로스 엔트로피 $H(P, Q)$ 이며, 이는 곧 NLL입니다. 따라서 **"KL Divergence 최소화 $\equiv$ NLL(Cross-Entropy) 최소화"** 라는 등식이 성립합니다.

---

## (d) 일반 KL Divergence 이산형 데이터 계산

**[조건]** $P(X)=[0.4, 0.6]$, $Q(X)=[0.5, 0.5]$

**[계산 전개]**
$$KL(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$
$$= 0.4 \log\left(\frac{0.4}{0.5}\right) + 0.6 \log\left(\frac{0.6}{0.5}\right)$$
$$= 0.4 \log(0.8) + 0.6 \log(1.2)$$
*   시스템적 의미: 완벽한 5:5 밸런스 서버($Q$)로 4:6 편향 트래픽($P$)을 처리할 때 낭비되는 정보량(비효율)을 계산한 수치입니다. (머신러닝에서는 보통 밑이 $e$ 인 자연로그를 사용합니다.)
---
title: "[파트 4] KL Divergence (쿨백-라이블러 발산) - 데이터 분포 간의 거리 측정 로직"
slug: 04-kl-divergence
order: 4
---

# [파트 4] KL Divergence (쿨백-라이블러 발산) - 데이터 분포 간의 거리 측정 로직

🔥 **한 줄 결론**
👉 "KL Divergence는 두 확률 분포가 '얼마나 다르게 생겼는지'를 측정하며, 정답 분포 $P$를 기준으로 내 모델 $Q$가 유발하는 정보량의 낭비를 합산하는 비대칭 함수다."

---

## 1. KL Divergence 일반 계산 (기출: 이산형 데이터)

시험에서 확률 표가 주어지고 직접 수치를 계산하라는 문제가 출제됩니다.

**[예제: P(실제)와 Q(예측)가 주어졌을 때의 일반 KL 계산]**
- $P(X): [0.4, 0.6]$
- $Q(X): [0.5, 0.5]$

$$KL(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$$
$= 0.4 \log\left(\frac{0.4}{0.5}\right) + 0.6 \log\left(\frac{0.6}{0.5}\right)$
$= 0.4 \log(0.8) + 0.6 \log(1.2)$
*   **주의:** 계산 시 $\log$의 밑이 $2$인지 $e$인지 명시하세요 (일반적으로 정보이론에서는 2, 머신러닝 최적화에서는 $e$를 씁니다). 

---

## 2. 킬러 증명 1: KL Divergence는 항상 0 이상이다 ($KL \ge 0$)

반드시 **Jensen's Inequality (젠센 부등식)**을 사용하여 증명해야 합니다.

$$KL(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$$
1. **역수 취하기 (부호 반전):** $= - \int P(x) \log \frac{Q(x)}{P(x)} dx = - \mathbb{E}_{x \sim P} \left[ \log \frac{Q(x)}{P(x)} \right]$
2. **젠센 부등식 적용 (오목 함수 $\log$):** $\mathbb{E}[\log(X)] \le \log(\mathbb{E}[X])$
   앞에 마이너스($-$)가 붙으므로 부등호 방향 반전 $\ge$
   $\ge - \log \left( \mathbb{E}_{x \sim P} \left[ \frac{Q(x)}{P(x)} \right] \right)$
3. **기댓값 전개 및 약분:** $\mathbb{E}_{x \sim P} \left[ \frac{Q(x)}{P(x)} \right] = \int P(x) \frac{Q(x)}{P(x)} dx = \int Q(x) dx = 1$
4. **결론:** $\ge - \log(1) = 0 \quad \therefore KL(P || Q) \ge 0$

---

## 3. 킬러 증명 2: Gaussian 간 KL 계산

**조건:** $P \sim \mathcal{N}(\mu_1, \sigma^2), Q \sim \mathcal{N}(\mu_2, \sigma^2)$ (평균 다르고 분산 동일)

$$KL(P || Q) = \mathbb{E}_{x \sim P} [\log P(x) - \log Q(x)]$$
1. **가우시안 대입 및 상수 상쇄:** 분산이 같으므로 루트 상수항 $\log$ 부분이 뺄셈에서 날아감.
   $= \mathbb{E}_{x \sim P} \left[ -\frac{(x-\mu_1)^2}{2\sigma^2} + \frac{(x-\mu_2)^2}{2\sigma^2} \right]$
2. **제곱식 전개:** $x^2$ 항 날아감.
   $= \mathbb{E}_{x \sim P} \left[ \frac{2x(\mu_1 - \mu_2) + \mu_2^2 - \mu_1^2}{2\sigma^2} \right]$
3. **기댓값 치환 (핵심 단계):** $\mathbb{E}_{x \sim P}[x]$ 이므로 변수 $x$ 자리에 $P$의 평균인 $\mu_1$ 대입!
   $= \frac{2\mu_1(\mu_1 - \mu_2) + \mu_2^2 - \mu_1^2}{2\sigma^2} = \frac{\mu_1^2 - 2\mu_1\mu_2 + \mu_2^2}{2\sigma^2}$
4. **결론:** $\frac{(\mu_1 - \mu_2)^2}{2\sigma^2}$ (두 평균 간 MSE의 절반과 동일)

---

📌 **교수님 강조 주의사항**
*   **비대칭성 명시:** $KL(P\|Q) \neq KL(Q\|P)$ 임을 묻는 개념 문제가 소문항으로 출제될 수 있습니다.
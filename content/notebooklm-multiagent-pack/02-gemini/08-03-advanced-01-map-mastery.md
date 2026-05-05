---
title: "Level 3: Advanced - MAP 완벽 해부 (MAP Mastery)"
slug: 03-advanced-01-map-mastery
order: 8
---

# Level 3: Advanced - MAP 완벽 해부 (MAP Mastery)

> **학습 목표:** 베이지안 통계의 꽃인 MAP(Maximum A Posteriori) 기출문제 3종(대칭, 비대칭, 텐트)을 수식으로 완벽히 전개하고, 극한($m \to \infty$)이나 미분 불가능점에서 어떻게 대처하는지 체득합니다.

---

## MAP의 기본 공식 (베이즈 정리)

**$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$**
사후확률(최종 결론)은 우도(데이터)와 사전확률(선입견)의 곱에 비례합니다.
로그를 취하면: $\log \text{Posterior} = \log \text{Likelihood} + \log \text{Prior}$ 가 됩니다.

베르누이 분포 데이터 $k$번 성공, $n-k$번 실패의 로그 우도는 $k \log \theta + (n-k) \log(1-\theta)$ 입니다.

---

## 1. 기출문제: 대칭 Prior (Symmetric)

**문제:** 데이터가 주어졌을 때, 사전확률이 $p(\theta) \propto \theta^m (1-\theta)^m$ 이다. $m \to \infty$ 일 때 $\hat{\theta}_{MAP}$ 를 구하시오.

### Step-by-Step 유도
1. **Log Posterior 구성:**
   * $\log P(\theta | y) = \text{Likelihood} + \text{Prior} = [k \log \theta + (n-k) \log(1-\theta)] + [m \log \theta + m \log(1-\theta)]$
   * 묶으면: $\ell(\theta) = (k+m) \log \theta + (n-k+m) \log(1-\theta)$
2. **미분하여 0이 되는 지점 찾기:**
   * $\frac{\partial \ell}{\partial \theta} = \frac{k+m}{\theta} - \frac{n-k+m}{1-\theta} = 0$
   * $(k+m)(1-\theta) = (n-k+m)\theta$
   * $k+m - (k+m)\theta = (n-k+m)\theta \implies (n+2m)\theta = k+m$
   * $\hat{\theta}_{MAP} = \frac{k+m}{n+2m}$
3. **$m \to \infty$ 극한 취하기:**
   * $\lim_{m \to \infty} \frac{k+m}{n+2m} = \frac{m(k/m + 1)}{m(n/m + 2)}$
   * $m$이 무한대로 가면 $k/m, n/m$ 은 0이 되므로, 답은 **$\frac{1}{2}$** 입니다.
4. **직관 (왜?):**
   * 대칭 Prior는 $\theta=0.5$ 에서 가장 높은 뾰족한 산입니다. $m$이 무한히 커지면 "동전은 무조건 반반이야!" 라는 똥고집(선입견)이 무한히 강해져서, 실제 데이터가 어떠하든 결론은 0.5가 됩니다.

---

## 2. 기출문제: 비대칭 Prior (Asymmetric)

**문제:** 사전확률이 $p(\theta) \propto \theta^m$ 이다. $m \to \infty$ 일 때 $\hat{\theta}_{MAP}$ 를 구하시오.

### Step-by-Step 유도
1. **Log Posterior 구성:**
   * $\ell(\theta) = [k \log \theta + (n-k) \log(1-\theta)] + [m \log \theta]$
   * 묶으면: $\ell(\theta) = (k+m) \log \theta + (n-k) \log(1-\theta)$
2. **미분하여 0이 되는 지점 찾기:**
   * $\frac{\partial \ell}{\partial \theta} = \frac{k+m}{\theta} - \frac{n-k}{1-\theta} = 0$
   * $(k+m)(1-\theta) = (n-k)\theta \implies \hat{\theta}_{MAP} = \frac{k+m}{n+m}$
3. **$m \to \infty$ 극한 취하기:**
   * $\lim_{m \to \infty} \frac{k+m}{n+m} = \mathbf{1}$
4. **직관 (왜?):**
   * $\theta^m$ 은 $\theta$가 1에 가까울수록 기하급수적으로 커집니다. 선입견이 "무조건 $\theta=1$이야!" 라고 강하게 주장하므로, 데이터 무시하고 1이 됩니다.

---

## 3. 기출문제: 텐트 모양 Prior (Tent Prior)

**문제:** (이미지 첨부 문제) 총 던진 횟수 $n=5$, 성공 $k=4$, 사전확률의 강도 $m=2$ 또는 $m=6$ 이다.
사전확률 $p_m(\theta)$는 $|\theta - 0.5| \le \frac{1}{m}$ 구간에서 $m - m^2|\theta - 0.5|$ 이고, 그 외는 0이다.
이때 $m=2$와 $m=6$ 각각에 대해 $\hat{\theta}_{MAP}$를 구하시오.

### Step-by-Step 유도 (킬러 문항)

이 문제는 Prior가 0.5에서 뾰족한 삼각형(텐트) 모양입니다. 절댓값 기호 때문에 0.5에서 미분이 불가능하므로, 기계적인 미분=0 공식이 통하지 않습니다.

1. **데이터만 봤을 때 (MLE):**
   * $\hat{\theta}_{MLE} = 4/5 = 0.8$ 입니다. 즉 Likelihood는 0.8에서 가장 높습니다.
2. **$m=2$ 일 때 분석:**
   * 텐트의 밑변 구간: $|\theta - 0.5| \le 1/2 \implies \theta \in [0, 1]$ (전체 구간)
   * 0.5를 기준으로 왼쪽, 오른쪽으로 식을 나누어 미분해봐야 합니다. 
   * 최댓값은 0.8(MLE 위치)과 0.5(Prior 정점) 사이 어딘가에서 타협이 일어납니다. 복잡한 미분을 피하고, 부분 구간의 증감을 분석하여 실제 시험에서는 특정 값을 정확히 구하기보다 극대점이 형성되는 원리를 서술하는 것이 핵심입니다.
3. **$m=6$ 일 때 분석:**
   * 텐트의 밑변 구간: $|\theta - 0.5| \le 1/6 \implies \theta \in [0.333, 0.667]$
   * 텐트 밖은 사전확률이 0입니다. 사후확률 = (우도 $\times$ 0) = 0 이 되어버립니다.
   * MLE 값인 0.8은 텐트 바깥(0.667보다 오른쪽)에 있습니다!
   * Likelihood는 0.8을 향해 계속 증가하고 싶은데, 0.667을 넘는 순간 Prior에 의해 낭떠러지(0)로 떨어집니다.
   * 따라서 가장 높은 곳은 경계선인 **$\hat{\theta}_{MAP} = 0.667 (즉, \frac{4}{6} = \frac{2}{3})$** 이 됩니다. 미분해서 0이 되는 곳이 아니라 끄트머리(Boundary)가 정답이 되는 전형적인 함정 문제입니다.

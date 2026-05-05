---
title: "[파트 2] MLE / MAP / Bayesian - 백엔드 시스템 관점의 파라미터 튜닝"
slug: 02-mle-map
order: 2
---

# [파트 2] MLE / MAP / Bayesian - 백엔드 시스템 관점의 파라미터 튜닝

🔥 **한 줄 결론**
👉 "MLE는 '현재 들어온 데이터만' 믿고 모델을 세팅하는 방법이고, MAP는 '기존의 경험(Prior)'까지 섞어서 모델을 세팅하는 방법이며, 이 과정은 끝없는 Prior Update의 반복이다."

💡 **쉽게 설명하면**
- **MLE (Maximum Likelihood Estimation):** 오늘 트래픽 로그만 보고 타임아웃 초수($\theta$) 세팅. (데이터 의존도 100%)
- **MAP (Maximum A Posteriori):** 오늘 로그 + 우리 회사의 10년치 노하우(Prior)를 섞어서 세팅.
- **Bayesian Update:** 오늘의 MAP 결과(Posterior)가 내일의 베스트 프랙티스(새로운 Prior)가 되는 캐시 업데이트 로직.

---\n
## 1. 확률적 추정의 본질 (Goal vs. Tool)
*   **Goal:** 현재 시스템 상태(데이터 $D$)를 가장 잘 설명하는 최적의 파라미터 $\theta$ 찾기.
*   **Tool:** 연산 오버플로우 방지를 위해 $\log$를 씌워 곱셈을 덧셈으로 변환 후, 미분($\frac{\partial}{\partial \theta} = 0$)으로 최적점 엑세스.

---\n
## 2. MLE (최대우도추정) 완벽 해부

$$\hat{\theta}_{MLE} = \arg\max_{\theta} P(D|\theta) = \arg\max_{\theta} \prod_{i=1}^n P(x_i|\theta)$$

**[로그 변환과 NLL 도출 (기출 필수)]**
컴퓨터의 Underflow 에러를 막기 위해 $\log$를 씌웁니다.
$$\ell(\theta) = \log P(D|\theta) = \sum_{i=1}^n \log P(x_i|\theta)$$
$$NLL(\theta) = - \sum_{i=1}^n \log P(x_i|\theta)$$
*   **Likelihood:** 가설 $\theta$가 주어졌을 때 데이터 $D$가 나올 확률.
*   **Posterior Estimator:** 데이터를 다 보고 난 후 최종적으로 확정된 파라미터 함수.

---\n
## 3. MAP와 Prior 업데이트 (기출 킬러 문항)

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} \left[ P(D|\theta) \cdot P(\theta) \right]$$

### 🔥 기출 1: Prior 형태 $\theta^m$ 단독 케이스
시험에서 $P(\theta) \propto \theta^m$ 인 경우의 MAP 추정치를 구하라고 나옵니다.
1. **Objective Function:** $J(\theta) = (N_1 \log \theta + N_0 \log(1-\theta)) + m \log \theta$
   (성공 횟수 $N_1$, 실패 횟수 $N_0$ 인 베르누이 데이터 가정)
2. **항 묶기:** $J(\theta) = (N_1 + m) \log \theta + N_0 \log(1-\theta)$
3. **미분:** $\frac{\partial J}{\partial \theta} = \frac{N_1 + m}{\theta} - \frac{N_0}{1-\theta} = 0$
4. **결과 도출:** $\hat{\theta}_{MAP} = \frac{N_1 + m}{N_1 + N_0 + m}$
   *   **해석:** 전체 성공 횟수에만 $m$개의 가상 데이터를 밀어넣는 튜닝 로직입니다.

### 🔥 기출 2: Prior 형태 $\theta^m (1-\theta)^m$ 케이스
1. **Objective Function:** $J(\theta) = (N_1 + m) \log \theta + (N_0 + m) \log(1-\theta)$
2. **결과 도출:** $\hat{\theta}_{MAP} = \frac{N_1 + m}{N_1 + N_0 + 2m}$
   *   **해석:** 성공과 실패 양쪽에 모두 $m$개씩 가상 캐시를 넣어 극단적 추정(Overfitting)을 방지합니다.

### 🔥 기출 3: Prior Update (사후 확률의 사전 확률화)
베이지안 딥러닝의 핵심은 **순차적 데이터 학습(Sequential Learning)**입니다.
*   $t=1$ 시점에 데이터 $D_1$을 보고 구한 Posterior $P(\theta|D_1)$는 데이터베이스에 저장됩니다.
*   $t=2$ 시점에 새로운 데이터 $D_2$가 들어오면, 어제 저장해둔 $P(\theta|D_1)$이 오늘의 **Prior**로 작동합니다.
*   **수식:** $P(\theta | D_1, D_2) \propto P(D_2 | \theta) \cdot P(\theta | D_1)$
*   **백엔드 비유:** 모델 파라미터를 처음부터 재학습(Full Batch)하지 않고, 이전 가중치를 베이스라인으로 삼아 새로 들어온 로그만 반영하는 **온라인 업데이트(Online Update) 아키텍처**입니다.

---\n
## 4. 특수 케이스: 미분 불가능 지점 고려 문제
단순히 $\frac{\partial}{\partial \theta} = 0$ 으로 풀면 0점 처리되는 기출 유형입니다.

*   **균등 분포 $U(0, \theta)$의 MLE:**
    *   $\log P(D|\theta) = -n \log \theta$ (단, 모든 $x_i \le \theta$)
    *   **해결 로직:** $-n \log \theta$를 최대화하려면 $\theta$를 최대한 작게 만들어야 합니다. 하지만 조건(`if x_i <= theta`)이 있으므로 $\theta$는 데이터 중 가장 큰 값 이상이어야 합니다.
    *   **결론:** $\hat{\theta}_{MLE} = \max(x_1, x_2, \dots, x_n)$
*   **라플라스 분포 (절댓값):** 미분 대신 "정렬 알고리즘(Sort) 후 중앙값(Median) 선택" 사용.

---\n
📌 **교수님 강조 주의사항 (시험 TIP)**
*   **증명 과정 필수:** MAP 수식 전개 시 "Assume Bernoulli distribution for data and apply log" 등 가정과 변환 과정을 영어로 짧게 적어주세요.
*   **Prior의 물리적 의미:** $m$이 커질수록 $\theta \rightarrow 0.5$로 수렴함을 보이며, "Prior knowledge dominates the data" 라고 서술하면 완벽합니다.
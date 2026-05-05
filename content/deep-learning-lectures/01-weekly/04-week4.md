---
title: "딥러닝이론 4주차 분석"
slug: week4
order: 4
---

# 딥러닝이론 4주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-4주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### MLE / MAP / Strong-MAP — 3가지 비교 프레임
- **정의:** 동전 던지기 framework에서 hypothesis $\theta$를 찾는 세 가지 방법.
  - **MLE / Uniform-prior MAP:** $\theta^*=k/n$. Prior 무시.
  - **Mid MAP** ($P(\theta) \propto \theta(1-\theta)$, $M=1$): $\theta^*=(k+1)/(n+2)$.
  - **Strong MAP** ($M\to\infty$): $\theta^*=1/2$. 데이터 무시.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]
  - "이 세 가지를 이제 비교를 하려고 합니다... 이 흐름을 다 이해해야 됩니다."

### Knowledge vs Data 축
- **정의:** Deduction 쪽엔 knowledge, induction 쪽엔 data. MAP는 prior knowledge에 의존, MLE는 data에만 의존.
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 첫 시간부터 이어지는 framing]

### Inductive Bias / Hypothesis Space
- **정의:** Prior knowledge를 model architecture에 강제로 넣어주는 것. Hypothesis space $\mathcal{H}$가 작아지는 것 = inductive bias가 강한 것.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 7주차/9주차에서 핵심 키워드]

### Linear Model vs Neural Network — Inductive Bias 차이
- **정의:**
  - Linear model: 강한 prior (X-Y가 선형 관계). 데이터 적을 때 유리.
  - Neural network: 약한 prior, expressive. 데이터 많을 때 유리.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### CNN — Domain-specific Inductive Bias
- **정의:** 이미지 특화 NN. 1980s LeCun, 2012 AlexNet. 강한 prior(이미지)을 넣어 데이터 적어도 잘함.
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 9주차에서 본격]

### Transformer — Low Inductive Bias
- **정의:** Expressive 매우 높음, prior knowledge 거의 없음. 데이터 많을 때 더 좋음.
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 7주차 재등장]

### KL Divergence
- **정의:** $D_{KL}(P\|Q) = E_{x\sim P}[\log P(x) - \log Q(x)] = -H(P) + H(P,Q)$.
- **성질:** $\ge 0$, $P=Q$일 때 0. 분포 간 "거리"처럼 해석 (asymmetric).
- **응용:** Empirical distribution과 model distribution의 KL을 최소화 = MLE와 연결.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Empirical Distribution
- **정의:** $P_S(x) = \frac{1}{n}\sum_{i=1}^n \delta(x-x_i)$. 데이터 자체를 분포로 표현.
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 8주차 ERM에서 핵심]

### Cross-Entropy / Entropy
- **정의:** $H(P,Q) = -E_{x\sim P}[\log Q(x)]$, $H(P) = -E_{x\sim P}[\log P(x)]$. $D_{KL}(P\|Q) = H(P,Q) - H(P)$.
- **응용:** Classification loss.
- **중요도:** ★★★★★★★★ (8/10) [추론 보충: 6주차/7주차 로스 함수]

### Hypothesis = Function (Classification/Regression)
- **정의:** 동전 던지기에서 $\theta$ (스칼라)였던 hypothesis가 supervised learning에서는 함수 $h: X\to Y$.
- **분류:** $Y$가 discrete = classification, $Y$가 연속 = regression.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Regression — Squared Error의 Gaussian 기원
- **정의:** $y = h(x) + \epsilon$, $\epsilon\sim\mathcal{N}(0,\sigma^2)$ 가정 → MLE = MSE 최소화.
- **CLT 정당화:** 고려 못한 수많은 변수($x_{11},\dots,x_{10^6}$)의 합 → Gaussian noise.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 가오스 이름 붙은 이유, 핵심]

### Negative Log-Likelihood (NLL) ↔ MSE 연결
- **정의:** Gaussian likelihood + log + minus → square = MSE.
- **유도:** $-\log P(y|x,h) = \frac{1}{2\sigma^2}(y - h(x))^2 + C$.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심 통합 시각]

### CLT (Central Limit Theorem) — 진짜 의미
- **정의:** $X_i\sim P$ (어떤 분포), $S_n = \sum X_i / n$이라 하면 $S_n$의 분포는 $n\to\infty$에서 normal에 수렴. **샘플 분포가 아니라 샘플 평균의 분포가 normal.**
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 학생들의 오해 직접 정정]

### Gaussian의 이름 유래
- **정의:** Gauss가 처음 발견한 게 아니지만 (De Moivre가 70년 먼저), 외성 궤도 예측(regression 문제)에 19개 데이터로 least squares 적용 → Gaussian distribution과 연결시켜 사용. 그래서 그 이름이 붙음.
- **중요도:** ★★★★★ (5/10) [명시적: 1주차에서부터 약속한 이야기]

### IID Assumption (Regression)
- **정의:** $x_i$들이 어떤 분포에서 IID, $y_i = h(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0,\sigma^2)$ IID.
- **중요도:** ★★★★★★★ (7/10)

### NLL ↔ Empirical Risk
- **정의:** NLL을 평균 내면 empirical risk가 됨. $L_S(h) = \frac{1}{n}\sum \ell(h, x_i, y_i)$. MSE는 squared loss를 사용한 ERM의 특수 케이스.
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 6,7,8주차에서 계속 등장]

---

## 🔢 2. 수식 풀이 및 증명

### General-$M$ MAP (지난주 숙제 마무리)
**문제/목표:** $P(\theta) \propto \theta^M(1-\theta)^M$일 때 $\theta^*_{MAP}$.
**단계별 풀이:**
1. Log prior = $M\log\theta + M\log(1-\theta) + C$.
2. Log posterior 미분 = 0:
$\frac{k+M}{\theta} - \frac{n-k+M}{1-\theta} = 0$.
3. $\theta^*_{MAP} = \frac{k+M}{n+2M}$.
4. $M\to\infty$: $\to 1/2$.

**중요도:** ★★★★★★★★★ (9/10) [명시적: 직접 보드에 풀이]

### Linear Regression의 MLE = MSE 유도
**문제/목표:** $y_i \sim \mathcal{N}(h(x_i), \sigma^2)$일 때 NLL = MSE를 보임.
**단계별 풀이:**
1. $-\log P(D|h) = -\sum_i \log P(y_i|x_i, h)$ (IID).
2. $P(y_i|x_i,h) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_i-h(x_i))^2}{2\sigma^2}\right)$.
3. $-\log$: exponential 사라짐, minus도 짝지어 사라짐.
4. 결과: $\sum_i \frac{(y_i - h(x_i))^2}{2\sigma^2} + C$.
5. $\sigma$ 무시 + 평균 → $\frac{1}{n}\sum_i (y_i - h(x_i))^2 = \text{MSE}$.

**결론:** Gaussian likelihood + NLL = MSE. **MLE 최소화 = MSE 최소화.**
**중요도:** ★★★★★★★★★★ (10/10) [명시적: 가장 중요한 유도]

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| MLE↔MAP↔Strong-MAP 비교 | 10 | 통합 시각 | "이 흐름을 다 이해해야 됩니다" |
| NLL↔MSE 유도 (Gaussian) | 10 | 정리 | "이게 다 이렇게 시작한 겁니다" |
| Inductive Bias (Hypothesis Space) | 10 | 개념 | 7,9주차 핵심 키워드 |
| Hypothesis = Function | 9 | 개념 | "$\theta$에서 함수로 확장" |
| CLT의 진짜 의미 | 9 | 개념 | 학생 오답 직접 정정 |
| Knowledge↔Data 축 | 9 | 프레임 | 1주차부터 일관 |
| Empirical Distribution | 7 | 정의 | ERM에서 핵심 도구 |
| KL Divergence | 8 | 정의 | 분포간 거리 |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. General-$M$ MAP 풀기
**문제:**
> "$P(\theta)\propto\theta^M(1-\theta)^M$일 때 $\theta^*_{MAP}$를 구하고 $M\to\infty$를 보내라."

**트리거 발언:** "여기다 그대로 써줄거에요... 어떻게 되는지 좀 보시고 답을 얻으세요."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Log posterior 미분 = 0:
$\frac{k+M}{\theta} = \frac{n-k+M}{1-\theta}$.
2. $(k+M)(1-\theta) = (n-k+M)\theta$.
3. $\theta^*_{MAP} = \frac{k+M}{n+2M}$.

**정답:** $\theta^* = (k+M)/(n+2M)$, $M\to\infty$이면 $\to 1/2$.

**해설:** $M$은 prior strength 파라미터. $M=0$이면 MLE, $M\to\infty$이면 데이터 완전 무시. 이를 통해 prior의 strength continuum을 시각화.

</details>

### Q2. MLE / Mid-MAP / Strong-MAP 비교 직관
**문제:**
> "Maximum Likelihood만, MAP만, Strong MAP, 셋 중에서 뭐가 더 좋을 것 같으세요?"

**트리거 발언:** "본인 취향... 직관적으로 어떻게 보세요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. **답은 없다 — 문제마다 다름.**
2. 데이터 적음 + 어려운 task → Strong prior (MAP)이 좋음.
3. 데이터 많음 + 쉬운 task → MLE이 좋음.
4. CNN(2012)의 성공 = 데이터 적었을 때 strong prior(이미지) 효과.
5. Transformer/ChatGPT(2020s) = 데이터 많아지자 weak prior 더 좋음.

**정답:** Task difficulty와 data amount의 상대 관계로 결정.

**해설:** AI 역사 자체가 prior strength의 oscillation. Linear model→NN→CNN→Transformer로 가며 prior가 약해지는 흐름이지만 항상 옳지 않고 상황에 따라 다름.

</details>

### Q3. KL divergence 직관 (생략된 직접 풀이)
**문제:**
> "두 가오시안 분포 사이의 KL divergence를 계산하면, 평균이 같고 분산이 같으면 0이 되고, 평균이 떨어질수록 거리가 멀어지는 것을 확인하라."

**트리거 발언:** "이것도 저희가 시간이 되면 하고 싶은데... 직접 해보시기 바랍니다"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $D_{KL}(\mathcal{N}(\mu_1,\sigma^2)\|\mathcal{N}(\mu_2,\sigma^2)) = \frac{(\mu_1-\mu_2)^2}{2\sigma^2}$.
2. $\mu_1=\mu_2$이면 0, 멀어질수록 quadratic하게 증가.

**정답:** Gaussian의 KL은 평균 차이의 제곱에 비례.

**해설:** KL을 두 분포 간 거리처럼 직관적으로 사용 가능 (특수 케이스). 일반적으로는 비대칭이지만 같은 family의 Gaussian이면 평균 차이의 quadratic.

</details>

---

## 📎 5. 기타 참고사항

- **행정 (시험 안내):** 다음 주 같은 시간 오프라인 시험. 퀴즈는 기말까지 가지고 있다가 시험 직전 제출. 메일/게시판/카톡 문의 안 받음, 직접 질문만. 객관식 X, 주관식 — 답보다 과정 평가. 문제는 영어, 번역 가능. 수식만 적으면 의미 없고 논리 과정 서술 필요.
- **시험 범위:** 오늘 배운 것까지. 녹화본 + 실시간 다룬 내용. 7강(이번 주차) 녹화본 더 자세히 보기 권장.
- **녹화 정책:** 실시간 녹화는 업로드 안 함. "직접 들으시면 되지 충분"이라는 입장.
- 학생 출장 사례 처리: 한국 도착 시간에 맞춰 온라인 진행.
- 라이클리우드 풀이 후 잠시 옆길 — "데이터에 너무 의존" 문제 의식 환기.
- **Knowledge–Data 표** 수업 중 칠판 전개:
  - MLE: prior 없음, 데이터에만 의존, expressivity 높음, hypothesis space 넓음, 데이터 많을 때 좋음
  - Strong MAP: prior 강함, knowledge에 의존, expressivity 낮음, hypothesis space 좁음, 데이터 적을 때 좋음
- 동전 1000번 던져 1000번 다 head 사례 — Strong-MAP은 0.5라고 답하는 부조리.
- "Inductive bias", "weight", "flexibility", "expressiveness" 표현이 모두 같은 prior 강도와 연결됨.
- Naive class-feature 등 어려운 부분은 "지금 다루기 너무 어려워서 직관만"으로 스킵.
- Cross-entropy loss 나오는 classification 쪽은 "기회 되면 다음에"로 스킵.
- 마지막 정리: 7강 강의자료의 KL/Cross-Entropy/Empirical Risk 까지 가는 전체 그림 보여주고, 리그레션이 핵심이라는 점 재강조.
- 시험 출장으로 못 오는 두 명만 메일로 알려달라고 요청.


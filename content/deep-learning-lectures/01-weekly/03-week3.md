---
title: "딥러닝이론 3주차 분석"
slug: week3
order: 3
---

# 딥러닝이론 3주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-3주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.
> **이 주차는 수업 전체에서 "가장 중요한 날"로 명시되었음.**

---

## 📘 1. 개념 및 정의

### Sample Space / Event / Probability
- **정의:**
  - Sample space: 가능한 모든 outcome의 집합.
  - Event: sample space의 부분집합.
  - Probability: event에 값(0~1)을 부여.
- **중요도:** ★★★★★ (5/10) [추론 보충: 복습 차원]

### Conditional Probability / Independence
- **정의:** $P(E|F) = P(E\cap F)/P(F)$. Independence: $P(E\cap F) = P(E)P(F)$ — 곱으로 표현되는 것이 핵심.
- **중요도:** ★★★★★★★ (7/10) [명시적: 곱이라는 점 강조]

### Bernoulli Distribution
- **정의:** Sample space 크기 = 2 (가장 간단한 random distribution). $P(H)=\theta$, $P(T)=1-\theta$. 스칼라 $\theta$ 하나로 분포 결정.
- **중요도:** ★★★★★★★★ (8/10) [명시적: "가장 간단해서 베르눌리부터 시작"]

### Gaussian (Normal) Distribution — 3가지 핵심 성질
- **정의:** $\mathcal{N}(\mu, \sigma^2)$. PDF에 **(1) exponential, (2) minus, (3) square** 세 가지 핵심 요소.
- **핵심:** $-\log \mathcal{N}(x|\mu,\sigma^2)$ 하면 exponential 사라지고, minus 사라지고, **square만 남음**. 이게 MSE loss의 기원.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: "꼭 기억하세요. exponential, minus, square"]

### Bayesian vs Frequentist Probability
- **정의:**
  - Frequentist: event에 대한 relative frequency.
  - Bayesian: hypothesis에 대한 **degree of belief** (믿음의 정도).
- **맥락:** "Bayesian 관점을 이해해야 learning이라는 개념을 이해할 수 있다."
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 수업 전체에서 가장 중요]

### Hypothesis $h$ / $\theta$
- **정의:** 가설. 동전의 경우 $\theta\in[0,1]$ 각각이 하나의 hypothesis. 일반적으로 모델/함수.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Bayes' Theorem
- **정의:** $P(H|E) = \frac{P(E|H) P(H)}{P(E)}$.
- **풀어 쓰면:** Posterior $\propto$ Likelihood $\times$ Prior. 그래서 $\arg\max_H P(H|E) \equiv \arg\max_H P(E|H)P(H)$ (분모 무시).
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 수업의 핵심 도구]

### Prior $P(H)$
- **정의:** 데이터를 보기 전 hypothesis에 대한 믿음의 정도.
- **표기:** **파란색**으로 표시.
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 색상 코딩으로 강조]

### Posterior $P(H|E)$
- **정의:** 데이터 $E$를 관측한 후 hypothesis $H$에 대한 믿음의 정도. **이게 우리가 진짜 보고 싶은 것.**
- **표기:** **빨간색.**
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]

### Likelihood $P(E|H)$
- **정의:** Hypothesis $H$가 참이라고 가정했을 때 데이터 $E$가 나타날 확률. **데이터에 대한 확률처럼 보이지만 실제로는 $H$의 함수**로 본다.
- **표기:** **초록색.**
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]

### Belief Update
- **정의:** Prior → (관측 데이터) → Posterior로 믿음이 갱신되는 과정. 이게 곧 **learning**.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]
  - "Belief update를 하는 것이 우리가 learning이라고 하는 것과 연결시키려고 합니다."

### IID (Independent and Identically Distributed)
- **정의:** 데이터들이 서로 독립이고 같은 분포에서 나옴. 동전 던지기 1회와 2회의 결과가 서로 영향 없고 분포가 동일.
- **중요도:** ★★★★★★★★ (8/10) [명시적: 곱 표현의 정당화]

### Maximum Likelihood Estimation (MLE)
- **정의:** $\theta^*_{ML} = \arg\max_\theta P(E|\theta)$. 동전 $n$번 던져 $k$번 head가 나오면 $\theta^* = k/n$.
- **문제점:** 데이터에만 의존. $n=k=3$이면 항상 head 나온다고 결론 — 데이터 적을 때 위험.
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 풀이 과정과 단점 모두 강조]

### Maximum A Posteriori (MAP)
- **정의:** $\theta^*_{MAP} = \arg\max_\theta P(\theta|E) = \arg\max_\theta [\log P(E|\theta) + \log P(\theta)]$.
- **MLE = uniform prior일 때의 MAP.** 즉 MLE는 MAP의 special case로 이해해야 함.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]

### Different Priors for $\theta$
- **정의:** 다양한 prior 가능: Uniform, $\theta(1-\theta)$ (0.5에서 peak인 quadratic), $\theta^M(1-\theta)^M$ ($M\to\infty$이면 0.5에 spike).
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### "Strong" Prior로서의 $M\to\infty$
- **정의:** $P(\theta) \propto \theta^M(1-\theta)^M$, $M$이 커질수록 0.5 근처에 집중. 데이터를 무시하고 0.5라고 결론.
- **중요도:** ★★★★★★★★ (8/10) [추론 보충: 4주차에서 ML/MAP 비교 핵심 예시로 재등장]

---

## 🔢 2. 수식 풀이 및 증명

### Bayes' Theorem 증명
**문제/목표:** $P(H|E) = P(E|H)P(H)/P(E)$ 증명.
**단계별 풀이:**
1. Conditional probability 정의: $P(H|E) = P(H\cap E)/P(E)$.
2. $P(E|H) = P(E\cap H)/P(H)$, 따라서 $P(E\cap H) = P(E|H)P(H)$.
3. 1번 분자에 대입: $P(H|E) = P(E|H)P(H)/P(E)$. ∎

**중요도:** ★★★★★★★★ (8/10) [명시적: 3분 시간 주고 증명]

### Maximum Likelihood for Bernoulli ($n$번 던져 $k$번 head)
**문제/목표:** $\theta^*_{ML}$ 도출.
**단계별 풀이:**
1. **IID 사용:** $P(D|\theta) = \prod_{i=1}^n P(x_i|\theta)$.
2. **Identical 사용:** $\prod = \theta^k (1-\theta)^{n-k}$.
3. **Log-likelihood:** $\log P(D|\theta) = k\log\theta + (n-k)\log(1-\theta)$.
4. 미분: $\frac{\partial}{\partial\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta}$.
5. Set to 0: $k(1-\theta) = (n-k)\theta \Rightarrow k = n\theta$.
6. $\theta^*_{ML} = k/n$.

**결론:** $n=k=3$이면 $\theta^*=1$ → "계속 head만 나올 거다" — 너무 극단적.
**중요도:** ★★★★★★★★★★ (10/10) [명시적: 칠판 풀이 + 시험 단골]

### MAP with Prior $P(\theta) \propto \theta(1-\theta)$
**문제/목표:** $\theta^*_{MAP}$ 도출.
**단계별 풀이:**
1. $\log P(\theta) = \log\theta + \log(1-\theta) + C$.
2. Log posterior $\propto k\log\theta + (n-k)\log(1-\theta) + \log\theta + \log(1-\theta)$
   $= (k+1)\log\theta + (n-k+1)\log(1-\theta) + C$.
3. 미분 = 0: $\frac{k+1}{\theta} - \frac{n-k+1}{1-\theta} = 0$.
4. $\theta^*_{MAP} = \frac{k+1}{n+2}$.

**결론:** $n=k=3$이면 $\theta^*_{MAP} = 4/5 = 0.8$ — 덜 극단적.
**중요도:** ★★★★★★★★★ (9/10) [명시적: 3분 풀이 + 핵심 비교]

### MAP with Prior $P(\theta) \propto \theta^M(1-\theta)^M$, $M\to\infty$
**문제/목표:** Strong prior에서 $\theta^*_{MAP}$ 거동.
**단계별 풀이:**
1. Log prior = $M\log\theta + M\log(1-\theta) + C$.
2. Log posterior 미분 = 0: $\frac{k+M}{\theta} = \frac{n-k+M}{1-\theta}$.
3. $\theta^*_{MAP} = \frac{k+M}{n+2M}$.
4. $M\to\infty$: $\theta^* \to 1/2$ (데이터 무시).

**결론:** $M=0$ → MLE. $M=1$ → 0.4 정도. $M\to\infty$ → 항상 0.5.
**중요도:** ★★★★★★★★★ (9/10) [명시적: 다음 주 숙제로 지정]

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| Bayesian probability (degree of belief) | 10 | 개념 | "이걸 이해하는 것이 가장 중요" |
| Posterior $P(H\mid E)$ | 10 | 개념 | "우리가 진짜 봐야 하는 것" |
| Bayes Theorem | 10 | 정리 | 직접 증명 + 색상 코드 |
| MLE = Uniform-prior MAP | 10 | 통합 시각 | "거꾸로 이해해야 한다" |
| Gaussian의 exp/minus/square | 10 | 성질 | "꼭 기억하세요" |
| MLE의 데이터 의존 단점 | 9 | 비판 | $\theta^*=1$ 결론의 부조리 |
| Prior의 해석 (믿음의 정도) | 9 | 개념 | 직접 그래프 그리게 함 |
| Belief Update = Learning | 10 | 통합 | "이게 learning이다" |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. Coin's $\theta$ 분포 직접 그리기
**문제:**
> "동전을 보여드릴 때 이 동전의 $\theta$ 값이 얼마일 것 같냐를 0~1 사이의 분포로 그려보세요. 더 그럴듯하다고 생각하면 높게, 덜 그럴 것 같으면 낮게."

**트리거 발언:** "각자 한번 해보실 수 있어요? 사진을 찍어서 올려주시거나..."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. 500원 동전이라면 $\theta\approx 0.5$에서 peak인 종 모양 분포가 자연스럽다.
2. 더 강한 confidence가 있다면 좁은 분포, 약하다면 넓은 분포.

**정답:** 0.5에서 가장 높고 양쪽으로 부드럽게 감소하는 unimodal 분포.

**해설:** 이게 곧 prior $P(\theta)$의 시각화. Bayesian에서 분포는 hypothesis에 대한 믿음의 정도를 표현.

</details>

### Q2. 3번 던져 3번 head 본 후 분포 갱신
**문제:**
> "3번 던졌더니 3번 다 앞면이 나왔다는 정보를 알려드렸을 때 여러분 생각이 어떻게 달라졌는지 다른 색깔로 그려주세요."

**트리거 발언:** "그 위에다가 다른 색깔을 써서 하나를 더 그려볼 거에요"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Posterior는 prior에 likelihood($\theta^3$, 단조증가)를 곱한 형태.
2. Peak가 prior peak (0.5)와 likelihood peak (1) 사이로 이동, 일반적으로 0.5보다 약간 오른쪽.

**정답:** 원래 0.5에서 peak였던 분포가 약간 오른쪽으로 치우치고, 0.6~0.7 근처에 새 peak가 형성됨.

**해설:** 이게 belief update의 시각화. $P(H|E) \propto P(E|H)P(H) = \theta^3 \cdot P(\theta)$.

</details>

### Q3. Bayes Theorem 증명
**문제:**
> "$P(H|E) = P(E|H)P(H)/P(E)$를 증명하라. 정의만 사용하면 됩니다."

**트리거 발언:** "3분 드릴테니까 한번 해볼까요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Conditional probability 정의: $P(H|E) = P(H\cap E)/P(E)$.
2. 같은 정의: $P(E|H) = P(E\cap H)/P(H) \Rightarrow P(E\cap H) = P(E|H)P(H)$.
3. 1번에 대입: $P(H|E) = P(E|H)P(H)/P(E)$. ∎

**정답:** 위 3단계.

**해설:** 식 자체는 쉽지만 각 항의 의미(prior, likelihood, posterior)가 핵심. "분모는 $H$와 무관한 상수"라는 점이 MAP를 단순화하는 키.

</details>

### Q4. MLE for Bernoulli — $\theta^*_{ML}$ 도출
**문제:**
> "Log-likelihood가 $k\log\theta + (n-k)\log(1-\theta)$일 때 가장 좋은 $\theta$를 찾아라."

**트리거 발언:** "여기서 4분 드릴테니까 어떤 세타가 가장 좋은 세타인지 여기서부터 찾아보시기 바랍니다"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $\theta$로 미분: $\frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$.
2. 통분: $k(1-\theta) - (n-k)\theta = 0$.
3. 전개: $k - k\theta - n\theta + k\theta = 0 \Rightarrow k = n\theta$.

**정답:** $\theta^*_{ML} = k/n$.

**해설:** Log를 취해 곱을 합으로 바꾸고 미분=0을 풀면 단순한 비율이 나옴. 이게 MLE의 출발점.

</details>

### Q5. MAP with $P(\theta) \propto \theta(1-\theta)$
**문제:**
> "이번에는 prior $P(\theta) = 6\theta(1-\theta)$일 때 $\theta^*_{MAP}$를 구하라."

**트리거 발언:** "3분 드리고 그래서 이거를 제일 크게 하는 세타는 뭔지 이제 찾아보시겠어요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Log posterior = $k\log\theta + (n-k)\log(1-\theta) + \log\theta + \log(1-\theta) + C$.
2. = $(k+1)\log\theta + (n-k+1)\log(1-\theta) + C$.
3. 미분 = 0 → $\theta^*_{MAP} = \frac{k+1}{n+2}$.

**정답:** $\theta^*_{MAP} = (k+1)/(n+2)$.

**해설:** Prior이 $\theta(1-\theta)$ 꼴이면 분자/분모에 1씩 더해진 효과 — pseudo-count 1을 양쪽에 추가한 것과 동치 (Laplace smoothing).

</details>

### Q6. MAP with $P(\theta) \propto \theta^M(1-\theta)^M$, $M\to\infty$
**문제:**
> "$M\to\infty$로 보내면 $\theta^*_{MAP}$는?"

**트리거 발언:** "다음주에 그 녹화본을 보시면서 한번 채워보시기 바랍니다"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $\theta^*_{MAP} = (k+M)/(n+2M)$.
2. $M\to\infty$: 분자/분모 모두 $M$에 dominated, $\to 1/2$.

**정답:** $\theta^*_{MAP} \to 1/2$ (데이터 무시).

**해설:** Strong prior 극단 — 어떤 데이터를 보든 무조건 0.5로 결론. $n=k=1000$이어도 0.5라고 답함. Prior의 위험성.

</details>

---

## 📎 5. 기타 참고사항

- 행정: 메일 문의 안 받음, 게시판으로. 공결 조건은 스프레드시트 참고. 출장 등으로 시간이 안 맞을 경우 같은 시간 온라인 진행.
- 중간고사 일정 안내, 퀴즈 LMS 제출 방법.
- 실수: 수업 중간에 화이트보드 모드를 켜려다가 실패. "쓸 수 있으면 좋겠는데... 안되겠어 잠시만요." 학생들이 각자 그리고 사진/카메라로 공유하는 방식으로 진행.
- 시험 과정 vs 답: "답만 적으면 점수 없고 과정을 봐야 한다."
- "베이지안적 관점은 데이터가 적거나 불확실성을 표현할 때 쓰는 건가요?"라는 학생 질문에 "그건 나중에 연결되지만 지금은 hypothesis에 대한 거다라는 식으로 이해하라"고 답함.
- $P(\theta)$가 마이너스/100만이어도 상관없음 — frequentist와 달리 0~1로 제한할 필요 없음.
- "디덕션적 reasoning이 자연스럽다. 베이지안은 처음에 이질감이 있다 — 같이 해보면서 익숙해져라."
- 프리퀀시스트 관점은 이벤트($P(E)$)가 중심, 베이지안은 hypothesis($P(H)$)가 중심.
- 다음 주 권장 학습: 7강 녹화본을 다시 듣기 (오늘 못 따라온 부분 + MAP 이해). 8강(이번 시간) 녹화본도 따로 올리지 않음 — 이번 라이브 수업이 7강과 같은 주제를 다시 다룬 것.
- 마지막 멘트: "지금까지 한 거를 다음 주에 비슷하게 다시 보는 식으로 구성했습니다."


---
title: "03. 완벽 답안 — 텐트 prior"
slug: perfect-answer
order: 3
---

# 03. 완벽 답안 — 텐트 prior

이 문제는 가장 어려우므로 **부분점수 사냥**에 집중. 완전한 답이 안 나와도 절차를 정확히 적으면 점수.

---

### [문제] (n=5, k=4) 텐트 prior $p_m(\theta) = m - m^2|\theta - 0.5|$ (정의역 내) 일 때, m=2와 m=6의 MAP.

### [공통 설정]

- Likelihood: $L(\theta) = \theta^4(1-\theta)$
- Posterior: $p(\theta|D) \propto L(\theta) \cdot p_m(\theta)$
- 절댓값으로 인해 $\theta = 0.5$에서 미분 불가능 → **두 영역으로 분리**해 분석.

---

### [Case 1: m = 2]

prior 정의역: $\theta \in [0, 1]$.

**영역 (a) $\theta \in [0, 0.5]$:** $p_2 = 4\theta$, $f_a = 4\theta^5(1-\theta)$.
$$\frac{d}{d\theta}\log f_a = \frac{5}{\theta} - \frac{1}{1-\theta} = 0 \Rightarrow \theta = 5/6.$$
$5/6 > 0.5$이므로 영역 (a) 밖. 영역 (a) 내에서는 단조증가.

**영역 (b) $\theta \in [0.5, 1]$:** $p_2 = 4 - 4\theta$, $f_b = 4\theta^4(1-\theta)^2$.
$$\frac{d}{d\theta}\log f_b = \frac{4}{\theta} - \frac{2}{1-\theta} = 0 \Rightarrow \theta = 2/3.$$
영역 (b) 내. 부호 분석으로 정점.

**비교:** $f_a(0.5) = 1/16$, $f_b(2/3) = 64/729 \approx 0.088 > 1/16$. → 2/3이 최댓값.

$$\boxed{\hat{\theta}_{\text{MAP}}^{(m=2)} = \frac{2}{3}.}$$

---

### [Case 2: m = 6]

prior 정의역: $\theta \in [1/3, 2/3]$.

**영역 (a) $\theta \in [1/3, 0.5]$:** $p_6 = 36\theta - 12$.

$\frac{d}{d\theta}\log f_a = 0$ 풀이 → $9\theta^2 - 10\theta + 2 = 0$, $\theta = (5 \pm \sqrt{7})/9 \approx 0.262, 0.849$. 모두 영역 (a) 밖. 부호 분석으로 영역 (a)에서 단조증가.

**영역 (b) $\theta \in [0.5, 2/3]$:** $p_6 = 24 - 36\theta$.

$\frac{d}{d\theta}\log f_b = 0$ 풀이 → $18\theta^2 - 25\theta + 8 = 0$, $\theta = 0.5, 8/9$. 영역 (b) 내부에 임계점 없음, 영역 (b)에서 단조감소.

**결합:** 영역 (a)에서 증가, 영역 (b)에서 감소 → **미분 불가능 점 $\theta = 0.5$가 정점**.

$$\boxed{\hat{\theta}_{\text{MAP}}^{(m=6)} = \frac{1}{2}.}$$

---

### [직관 정리]

m=2일 때는 prior 폭이 충분히 넓어서 likelihood의 영향이 prior 대칭성을 이기고 0.5보다 큰 쪽(2/3)에 정점이 형성된다. m=6일 때는 prior가 너무 좁고 첨예해 데이터를 거의 무시하고 prior의 정점 0.5에 머무른다. ∎

---

## 채점 포인트 (총 100%)

| 항목 | 배점 |
|------|------|
| 미분 불가능 → 영역 분리 인식 | 20% |
| Case 1 영역 (a), (b) 식 분리 | 10% |
| Case 1 임계점 풀이 (영역 (b) → 2/3) | 15% |
| Case 1 비교 후 결론 (2/3) | 10% |
| Case 2 정의역 [1/3, 2/3] 인식 | 15% |
| Case 2 두 영역 단조성 분석 | 15% |
| Case 2 정점 0.5 결론 | 10% |
| 직관 한 줄 | 5% |

**부분점수 사냥 팁:**
- 절댓값 → 영역 분리 한 줄만 적어도 20%
- prior 식을 정의역에 맞춰 분리해 적으면 추가 10%
- 정확한 풀이 못 해도 "단조증가/감소 분석으로 0.5 또는 경계가 후보" 적으면 부분 점수

---

## 다음

[`04-mastery-quiz.md`](./04-mastery-quiz.md)

---
title: "04. 마스터리 퀴즈 — 텐트 prior"
slug: mastery-quiz
order: 4
---

# 04. 마스터리 퀴즈 — 텐트 prior

## 문제 1: 텐트 prior 정규화 검증

m=2 prior $p_2(\theta) = 2 - 4|\theta - 0.5|$ on $[0,1]$의 적분이 1임을 보여라.

<details><summary>풀이</summary>

대칭성으로 $[0, 0.5]$ 적분의 2배:
$$\int_0^{0.5} (2 - 4(0.5 - \theta))\,d\theta = \int_0^{0.5} (4\theta)\,d\theta = 2\theta^2\Big|_0^{0.5} = 0.5$$

전체: $0.5 \times 2 = 1$ ✓
</details>

## 문제 2: m=10 일 때 MAP 예상

n=5, k=4, m=10일 때 $\hat{\theta}_{\text{MAP}}$는 (정성적)?

<details><summary>풀이</summary>

m=10이면 prior 정의역 $[0.4, 0.6]$. likelihood MLE = 0.8 → 정의역 밖. m=6과 비슷하게 **0.5가 정점** (미분 불가능 점 또는 좁은 prior 우세).
</details>

## 문제 3: n=10, k=5, m=2 (균형 데이터)

<details><summary>풀이</summary>

likelihood: $\theta^5(1-\theta)^5$. MLE = 0.5 (대칭).

prior도 0.5 정점. → 둘 다 0.5에서 정점 → MAP = 0.5.
</details>

## 문제 4: 미분 불가능 점에서 좌·우미분이 둘 다 + (양수)이면?

<details><summary>풀이</summary>

좌측에서 증가, 우측에서도 증가 → 그 점은 정점 아님. 우측에서 더 큰 값.
</details>

## 문제 5: prior 정의역이 좁아질 때의 일반 원리

<details><summary>풀이</summary>

prior 정의역이 좁아지면 (m 큼):
- likelihood MLE가 정의역 밖이면 → 경계 또는 정점
- 정의역 내부에 likelihood가 단조면 → 경계
- prior 정점이 likelihood 형상을 압도하면 → prior 정점 (0.5)
</details>

---

## 시험 직전 체크
- [ ] 절댓값 → 영역 분리 자동 반응?
- [ ] 정의역 [0.5 - 1/m, 0.5 + 1/m] 즉답?
- [ ] 미분 불가능 점이 후보임을 인식?
- [ ] 부분점수 사냥 (영역 분리만 적어도 점수) 전략?

[`../08-softmax/00-overview.md`](../08-softmax/00-overview.md)

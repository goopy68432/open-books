---
title: "기출 7번 — 텐트 prior MAP (이미지 문제)"
slug: 07-map-tent-00-overview
order: 41
---

# 기출 7번 — 텐트 prior MAP (이미지 문제)

## 문제 원문 (이미지)

prior 형태 (n=5, k=4):
$$p_m(\theta) = \begin{cases} 0 & |\theta - 0.5| \geq 1/m \\ m - m^2|\theta - 0.5| & |\theta - 0.5| \leq 1/m \end{cases}$$

m = 2 또는 m = 6일 때 각각 $\hat{\theta}_{\text{MAP}}$를 구하라.

(주의: 위 prior가 $p(\theta) \propto \theta^m$ 가정에서 유도된 사후분포의 근사 형태로 주어짐)

---

## 출제 의도

1. **미분 불가능 점**(절댓값) 처리 능력
2. **prior의 정의역 제약** (0이 되는 구간)
3. **likelihood vs prior 균형** 분석
4. **m에 따라 정의역이 좁아지는** 효과 인식

이 문제는 **가장 까다롭다**. 부분점수 사냥 전략 필요.

---

## 5분 핵심 답

**m = 2:**
- prior 정의역: $\theta \in [0, 1]$ (전체)
- prior 그래프: 0.5 정점, 0과 1에서 0
- likelihood: $\theta^4(1-\theta)$
- log posterior 미분 분석:
  - $\theta < 0.5$ 영역과 $\theta > 0.5$ 영역으로 나눠 각각 미분
  - 한 영역에서 단조증가 다른 영역에서 단조감소면 정점 0.5
  - 또는 내부에 극값 존재
- **답:** 미분 가능한 두 부분에서 각각 미분=0 풀이 후 비교
  - 일반적 답안: $\theta^* = 0.5$ 부근 (정확값은 풀이 참조)

**m = 6:**
- prior 정의역: $\theta \in [0.5 - 1/6, 0.5 + 1/6] = [1/3, 2/3]$
- 이 구간 밖에서 posterior = 0
- MLE는 0.8 → 정의역 밖 → 경계 또는 내부 극값
- 좁은 구간에서 likelihood 단조 → 경계값 후보

자세한 풀이는 [`02-derivation.md`](./02-derivation.md).

---

## 학습 자료

| 파일 | 내용 |
|------|-----|
| [`01-concept.md`](./01-concept.md) | 텐트 함수, 미분 불가능 점 |
| [`02-derivation.md`](./02-derivation.md) | m=2, m=6 케이스별 자세한 풀이 |
| [`03-perfect-answer.md`](./03-perfect-answer.md) | 답안 |
| [`04-mastery-quiz.md`](./04-mastery-quiz.md) | 변형 |

## 관련 사전지식
- [`../00-prerequisites/03-derivative-101.md`](../00-prerequisites/03-derivative-101.md) — 미분
- [`../00-prerequisites/05-integral-101.md`](../00-prerequisites/05-integral-101.md) — 적분 (정규화 검증)

## 출제 변형
- 다른 형태의 prior (사다리꼴 등)
- 정의역이 likelihood MLE를 포함/배제할 때

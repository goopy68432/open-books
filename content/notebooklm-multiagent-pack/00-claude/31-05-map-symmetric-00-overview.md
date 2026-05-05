---
title: "기출 5번 — MAP, 대칭 prior $p(\\theta) \\propto \\theta^m(1-\\theta)^m$"
slug: 05-map-symmetric-00-overview
order: 31
---

# 기출 5번 — MAP, 대칭 prior $p(\theta) \propto \theta^m(1-\theta)^m$

## 문제 원문

기출 4번 ($y_i \sim$ Bern(θ) i.i.d, $k = \sum y_i$) 상황에서, **prior** $p(\theta) \propto \theta^m(1-\theta)^m$ 일 때 $m \to \infty$이면 $\hat{\theta}_{\text{MAP}} = ?$

---

## 출제 의도

1. **베이즈 정리**: posterior ∝ likelihood × prior
2. **MAP과 MLE의 차이**: 데이터 + 사전믿음 결합
3. **prior가 강해질 때의 극한 행동**: 데이터를 무시
4. **m→∞ 극한 분석** 능력

---

## 5분 핵심 답

**유도:** posterior ∝ $\theta^{k+m}(1-\theta)^{n-k+m}$
**미분=0:** $\hat{\theta}_{\text{MAP}} = \dfrac{k+m}{n+2m}$
**m → ∞:** $\hat{\theta}_{\text{MAP}} \to \dfrac{1}{2}$

**직관:** "0.5에 대한 사전 확신이 무한히 강해지면, 데이터를 무시하고 0.5"

---

## 학습 자료

| 파일 | 내용 |
|------|-----|
| [`01-concept.md`](./01-concept.md) | MAP, Beta prior, 대칭성 |
| [`02-derivation.md`](./02-derivation.md) | 6단계 풀이 |
| [`03-perfect-answer.md`](./03-perfect-answer.md) | 답안 형식 |
| [`04-mastery-quiz.md`](./04-mastery-quiz.md) | 변형 |

## 관련
- 기출 4번 ([`../04-mle-bernoulli/`](../04-mle-bernoulli/)) — 우도함수
- 베이즈 정리 ([`../10-ten-proofs/03-bayes.md`](../10-ten-proofs/03-bayes.md))

---
title: '기출 6번 — MAP, 비대칭 prior $p(\theta) \propto \theta^m$'
slug: overview
order: 0
---

# 기출 6번 — MAP, 비대칭 prior $p(\theta) \propto \theta^m$

## 문제

기출 4번 상황에서 prior $p(\theta) \propto \theta^m$일 때 $m \to \infty$이면 $\hat{\theta}_{\text{MAP}} = ?$

## 5분 핵심 답

posterior ∝ $\theta^{k+m}(1-\theta)^{n-k}$
미분=0 → $\hat{\theta}_{\text{MAP}} = (k+m)/(n+m)$
m → ∞ → $\hat{\theta}_{\text{MAP}} \to 1$

**직관:** "prior가 θ=1 쪽으로 강하게 쏠리면, m→∞에서 데이터 무시하고 1로 수렴"

## 학습 자료
- [`02-derivation.md`](./02-derivation.md) — 5단계 풀이
- [`03-perfect-answer.md`](./03-perfect-answer.md) — 답안

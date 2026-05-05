---
title: "03. 베이즈 정리"
slug: 10-ten-proofs-03-bayes
order: 60
---

# 03. 베이즈 정리

## 정리

$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$

또는 (연속형):
$$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}$$

## 유도

조건부확률 정의:
$$P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B | A) = \frac{P(A \cap B)}{P(A)}$$

두 식에서 $P(A \cap B) = P(B|A) P(A) = P(A|B) P(B)$.

따라서:
$$P(A | B) = \frac{P(B|A) P(A)}{P(B)}$$

## 응용 — MAP

데이터 D, 모수 θ:
$$p(\theta | D) = \frac{p(D|\theta) p(\theta)}{p(D)}$$

분모 $p(D)$는 θ에 무관 → MAP에서 무시:
$$p(\theta | D) \propto p(D|\theta) \cdot p(\theta) = \text{likelihood} \times \text{prior}$$

## 사후 분포의 정규화

$$p(D) = \int p(D|\theta) p(\theta) d\theta$$

(sum over all θ)

## 시험 답안용

> "베이즈 정리: $p(\theta|D) = p(D|\theta)p(\theta)/p(D)$. MAP에서는 $p(D)$ 무관 → posterior ∝ likelihood × prior."

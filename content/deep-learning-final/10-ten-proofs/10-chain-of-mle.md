---
title: "10. MLE 통일 7단계 — 모든 분포"
slug: chain-of-mle
order: 10
---

# 10. MLE 통일 7단계 — 모든 분포

> 베르누이, 정규, 푸아송, 지수 — **모두 같은 7단계**.

## 7단계 템플릿

| 단계 | 작업 | "왜?" |
|-----|------|------|
| 1 | 모델 명시 (i.i.d) | 곱셈 정당화 |
| 2 | 단일 pdf/pmf | 출발점 |
| 3 | 우도 = 곱 | i.i.d 가정 |
| 4 | 단순화 (지수합 등) | 형태 깔끔 |
| 5 | log → ℓ | 곱→합, 단조, 수치 |
| 6 | 미분=0 (페르마) | 극값 후보 |
| 7 | 풀이 + 2계 미분 검증 | 최댓값 확정 |

## 분포별 결과 비교표

| 분포 | pdf/pmf | $L(\theta)$ | $\hat{\theta}_{\text{MLE}}$ |
|------|---------|----------|--------------------------|
| Bern(θ) | $\theta^y(1-\theta)^{1-y}$ | $\theta^k(1-\theta)^{n-k}$ | $k/n$ |
| Bin(n, θ) | $\binom{n}{k}\theta^k(1-\theta)^{n-k}$ | 동일 | $k/n$ |
| Poisson(λ) | $e^{-\lambda}\lambda^y/y!$ | $e^{-n\lambda}\lambda^{\sum y}/\prod y!$ | $\bar{y}$ |
| Exp(λ) | $\lambda e^{-\lambda x}$ | $\lambda^n e^{-\lambda \sum x}$ | $1/\bar{x}$ |
| N(μ, σ²) (μ만) | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | (생략) | $\bar{x}$ |
| N(μ, σ²) (σ²만) | 동일 | (생략) | $\frac{1}{n}\sum (x_i - \bar{x})^2$ |

## 푸아송 MLE (예시)

**1.** $X_i \sim$ Poisson(λ) i.i.d.

**2.** pmf: $p(x|\lambda) = e^{-\lambda}\lambda^x/x!$

**3.** 곱:
$$L(\lambda) = \prod e^{-\lambda}\lambda^{x_i}/x_i! = e^{-n\lambda}\lambda^{\sum x_i}/\prod x_i!$$

**4.** 단순화 (이미 됨).

**5.** log:
$$\ell(\lambda) = -n\lambda + (\sum x_i)\log\lambda - \log\prod x_i!$$

**6.** 미분=0:
$$\ell'(\lambda) = -n + \sum x_i / \lambda = 0$$

**7.** 풀이:
$$\hat{\lambda}_{\text{MLE}} = \bar{x}$$

2계 미분: $-\sum x_i/\lambda^2 < 0$ → 최댓값. ∎

## 패턴 인식

**모든 i.i.d MLE의 결과 = 표본의 자연스러운 통계량**:
- 베르누이/이항: 표본 비율
- 정규/푸아송: 표본 평균
- 지수: 표본 평균의 역수

**즉시 답할 수 있는 직관**: 시험에서 "MLE 풀이"를 보자마자 "분포 → 표본 통계량" 매핑.

## 시험에서

> "MLE는 모든 분포에서 7단계 체인으로 풀린다. 각 단계마다 i.i.d, 로그, 페르마, 2계 미분의 이유를 명시하면 만점."

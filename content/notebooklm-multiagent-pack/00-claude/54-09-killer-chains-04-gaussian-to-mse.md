---
title: "04. Gaussian → MSE — 정규분포에서 MSE 손실 유도"
slug: 09-killer-chains-04-gaussian-to-mse
order: 54
---

# 04. Gaussian → MSE — 정규분포에서 MSE 손실 유도

> "MSE는 그냥 직관 아닌 데이터에 정규 잡음이라는 가정의 산물"

## 1. 모델 가정

회귀 문제에서:
$$y_i = f(x_i; \theta) + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2) \text{ i.i.d}$$

- $f$: 모델 (예: 신경망)
- $\theta$: 모수
- $\epsilon$: 가우스 잡음

이를 보면 $y_i \sim N(f(x_i; \theta), \sigma^2)$.

## 2. 우도

i.i.d 가정:
$$L(\theta) = \prod_{i=1}^n p(y_i | x_i, \theta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right)$$

## 3. 로그우도

$$\ell(\theta) = \sum_i \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right]$$

$$= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f(x_i; \theta))^2$$

## 4. NLL

$$\text{NLL}(\theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i (y_i - f(x_i; \theta))^2$$

첫 항은 θ 무관 → 최소화에서 무시.

## 5. MSE와 동치

$$\arg\min_\theta \text{NLL} = \arg\min_\theta \sum_i (y_i - f(x_i; \theta))^2$$

상수 $1/(2\sigma^2)$는 argmin에 영향 없음.

**즉, "MSE 최소화" = "정규 잡음 가정 하의 MLE"**.

## 6. 시험 답안 핵심 문장

> "잡음 $\epsilon_i \sim N(0, \sigma^2)$ i.i.d 가정 하에서 우도의 음의 로그를 취하면 NLL = const + $\frac{1}{2\sigma^2}\sum (y_i - f_i)^2$이다. θ 최적화에서 상수와 양의 스칼라는 영향이 없으므로 NLL 최소화는 MSE 최소화와 동치다. 따라서 **MSE는 가우스 잡음 가정의 자연스러운 결과**이며, 잡음 분포가 다르면 다른 손실이 나온다 (예: 라플라스 잡음 → MAE)."

## 7. 한 줄 요약

> "정규 잡음 → MLE → MSE 등가성. 다른 잡음 가정은 다른 손실."

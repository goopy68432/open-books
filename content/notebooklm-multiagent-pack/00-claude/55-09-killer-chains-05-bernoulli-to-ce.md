---
title: "05. Bernoulli → Cross Entropy — 이진 분류 손실"
slug: 09-killer-chains-05-bernoulli-to-ce
order: 55
---

# 05. Bernoulli → Cross Entropy — 이진 분류 손실

## 1. 모델

이진 분류:
$$y_i \in \{0, 1\}, \quad y_i \sim \text{Bern}(p_i), \quad p_i = \sigma(f(x_i; \theta))$$

- $\sigma$: sigmoid (또는 softmax c=2)
- $p_i$: 모델이 예측한 "1" 확률

## 2. 우도

$$L(\theta) = \prod_{i=1}^n p_i^{y_i}(1 - p_i)^{1 - y_i}$$

## 3. NLL = Binary Cross Entropy

$$\text{NLL} = -\log L = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

이게 **BCE 손실**:
$$\boxed{\text{BCE} = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]}$$

## 4. 다중 클래스 확장 (Categorical CE)

$y_i \in \{1, 2, \ldots, c\}$, one-hot 인코딩 $\mathbf{y}_i$:
$$\mathbf{p}_i = \text{softmax}(f(x_i; \theta))$$

$$\text{CE} = -\sum_{i=1}^n \sum_{k=1}^c y_{ik} \log p_{ik}$$

## 5. 시험 핵심 — Softmax + CE 합성 그래디언트

**놀라운 단순성:**
$$\frac{\partial \text{CE}}{\partial z_j} = p_j - y_j$$

**유도** (기출 8번 응용):

$\partial \log p_i/\partial z_j = \delta_{ij} - p_j$ (softmax log 미분, [`08-softmax/04`](../08-softmax/04-mastery-quiz.md) 문제 1)

$$\frac{\partial \text{CE}}{\partial z_j} = -\sum_i y_i \frac{\partial \log p_i}{\partial z_j} = -\sum_i y_i (\delta_{ij} - p_j)$$

$$= -y_j + p_j \sum_i y_i = p_j - y_j$$

(one-hot이라 $\sum y_i = 1$)

## 6. 의미

- 그래디언트 = "예측 - 정답"
- 신경망 학습이 이렇게 단순한 형태인 이유.

## 7. 한 줄 요약

> "베르누이 NLL이 BCE 손실. Softmax + CE 합성 미분 = $\mathbf{p} - \mathbf{y}$ (예측 - 정답)."

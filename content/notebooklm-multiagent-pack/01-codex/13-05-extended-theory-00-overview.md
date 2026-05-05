---
title: "확장 시험 범위 로드맵"
slug: 05-extended-theory-00-overview
order: 13
---

# 확장 시험 범위 로드맵

이번 확장 범위는 기존 `MLE/MAP/Softmax` 자료의 다음 단계입니다.

기존 자료가 "손실함수를 어떻게 유도하는가"였다면, 확장 자료는 다음 질문에 답합니다.

> 손실함수를 얻은 뒤, 깊은 신경망은 어떻게 미분하고, 어떻게 최적화하며, 왜 일반화되는가?

## 1. 전체 흐름

```text
활성화 함수
  ↓
순전파 계산 그래프
  ↓
역전파 = 체인 룰의 행렬형 적용
  ↓
Gradient Descent / SGD / Adam
  ↓
Vanishing / Exploding Gradient 문제
  ↓
Regularization / Dropout / BatchNorm
  ↓
Bias-Variance / Double Descent
  ↓
Cross Entropy = H(p) + KL(p||q)
```

## 2. 새 파일 구성

| 파일 | 핵심 |
|---|---|
| `01-activation-functions.md` | Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softplus의 식과 미분 |
| `02-backprop-computational-graph.md` | 계산 그래프와 행렬형 역전파 |
| `03-optimization-algorithms.md` | GD, SGD, Momentum, RMSProp, Adam |
| `04-vanishing-exploding-gradients.md` | 깊은 네트워크에서 기울기가 죽거나 폭발하는 수학적 원인 |
| `05-generalization-regularization.md` | Overfitting, L1/L2, Dropout, BatchNorm, Early stopping |
| `06-bias-variance-double-descent.md` | Bias-Variance 분해 증명과 Double Descent |
| `07-cross-entropy-kl.md` | Cross Entropy = Entropy + KL 분해 |
| `08-perfect-answer-templates.md` | 시험장 서술형 답안 템플릿 |

## 3. 우선순위

시험 가능성이 높은 순서:

1. Backpropagation
2. 활성화 함수 미분
3. Bias-Variance 분해 증명
4. Cross Entropy = $H(p)+KL(p\|q)$
5. Vanishing/Exploding Gradient
6. Adam/Momentum/RMSProp
7. Dropout/BatchNorm
8. Double Descent

## 4. 기존 자료와 연결

| 기존 주제 | 확장 주제 |
|---|---|
| Softmax 자코비안 | Softmax+CE 출력층 역전파 |
| NLL | 모든 손실함수의 최적화 대상 |
| 체인 룰 | Backpropagation |
| MAP + L2 | Regularization 이론 |
| KL nonnegative | Cross Entropy 분해 |
| Gaussian MSE | Bias-Variance 분해 |

## 5. 최종 도달 기준

아래를 노트 없이 설명할 수 있으면 확장 범위까지 충분합니다.

- ReLU가 왜 sigmoid보다 깊은 신경망에 유리한지 미분값 관점에서 설명한다.
- $\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})$를 유도한다.
- Momentum, RMSProp, Adam의 차이를 "방향 평균"과 "스케일 보정"으로 설명한다.
- 깊은 네트워크에서 gradient가 곱셈 체인 때문에 사라지거나 폭발함을 보인다.
- Dropout을 random subnetwork ensemble 관점으로 설명한다.
- BatchNorm의 계산식과 역할을 말한다.
- $E[(y-\hat f)^2]=Bias^2+Variance+Noise$를 증명한다.
- $H(p,q)=H(p)+KL(p\|q)$를 증명한다.

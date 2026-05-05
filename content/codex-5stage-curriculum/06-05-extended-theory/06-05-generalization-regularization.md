---
title: "05. 일반화와 정규화"
slug: 05-generalization-regularization
order: 6
---

# 05. 일반화와 정규화

학습 데이터에서 잘 맞는 것과 처음 보는 데이터에서 잘 맞는 것은 다릅니다. 일반화 이론은 이 차이를 다룹니다.

## 1. 기본 용어

| 용어 | 의미 |
|---|---|
| Training error | 학습 데이터에서의 오차 |
| Test error | 새로운 데이터에서의 오차 |
| Generalization gap | test error - training error |
| Underfitting | 모델이 너무 단순해 train/test 모두 오차가 큼 |
| Overfitting | train error는 작지만 test error가 큼 |
| Regularization | 과도하게 복잡한 해를 피하게 만드는 장치 |

## 2. Empirical Risk와 True Risk

진짜 데이터 분포 $P$에 대한 위험:

$$
L_P(h)=E_{(x,y)\sim P}[\ell(h(x),y)]
$$

학습 데이터 $S=\{(x_i,y_i)\}_{i=1}^n$에 대한 경험적 위험:

$$
L_S(h)=\frac1n\sum_{i=1}^n \ell(h(x_i),y_i)
$$

학습은 보통 $L_S$를 줄이지만, 우리가 원하는 것은 $L_P$가 작아지는 것입니다.

## 3. 정규화된 ERM

일반적인 정규화 목적함수:

$$
\min_\theta L_S(\theta)+\lambda C(\theta)
$$

$C(\theta)$는 모델 복잡도 penalty입니다.

| penalty | 이름 | 효과 |
|---|---|---|
| $\|\theta\|_2^2$ | L2 / weight decay | 큰 가중치 억제 |
| $\|\theta\|_1$ | L1 | sparse한 해 유도 |
| 조기 종료 | Early stopping | 너무 오래 학습해 과적합되는 것 방지 |
| 데이터 증강 | Data augmentation | 효과적 데이터 수 증가 |

## 4. L2와 MAP 복습

Gaussian prior:

$$
p(\theta)\propto \exp\left(-\frac{\|\theta\|^2}{2\tau^2}\right)
$$

MAP:

$$
\arg\min_\theta \{-\log p(D|\theta)-\log p(\theta)\}
$$

prior의 음의 로그:

$$
-\log p(\theta)=C+\frac{\|\theta\|^2}{2\tau^2}
$$

따라서:

$$
MAP=NLL+\lambda\|\theta\|^2
$$

## 5. Dropout

훈련 중 각 뉴런 또는 activation을 확률적으로 꺼버립니다.

마스크:

$$
m_i\sim Bern(1-p)
$$

inverted dropout:

$$
\tilde h_i=\frac{m_i}{1-p}h_i
$$

기댓값:

$$
E[\tilde h_i]
=E\left[\frac{m_i}{1-p}h_i\right]
=h_i
$$

즉 훈련 때 일부 뉴런을 꺼도 activation의 기대값이 유지됩니다.

해석:

- 매번 다른 subnetwork를 학습합니다.
- 많은 subnetworks를 ensemble하는 효과가 있습니다.
- 특정 뉴런끼리만 강하게 의존하는 co-adaptation을 줄입니다.

시험 문장:

> Dropout은 학습 중 뉴런을 확률적으로 제거하여 매 mini-batch마다 다른 subnetwork를 학습시키는 방법이다. 이는 많은 모델의 ensemble과 비슷한 효과를 내며, 뉴런 간 복잡한 co-adaptation을 줄여 일반화를 돕는다.

## 6. Batch Normalization

미니배치 $B=\{x^{(1)},\ldots,x^{(m)}\}$에 대해 각 feature 차원별 평균과 분산:

$$
\mu_B=\frac1m\sum_{i=1}^m x^{(i)}
$$

$$
\sigma_B^2=\frac1m\sum_{i=1}^m (x^{(i)}-\mu_B)^2
$$

정규화:

$$
\hat x^{(i)}=\frac{x^{(i)}-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}
$$

scale/shift:

$$
y^{(i)}=\gamma\odot \hat x^{(i)}+\beta
$$

$\gamma,\beta$는 학습되는 파라미터입니다. 정규화만 하면 표현력이 제한될 수 있으므로 다시 scale과 shift를 허용합니다.

## 7. BatchNorm은 왜 작동하는가

전통적 설명:

- 각 층 입력 분포가 훈련 중 계속 바뀌는 internal covariate shift를 줄인다.

현대적/실전적 설명:

- 최적화 지형을 더 부드럽게 만든다.
- learning rate에 덜 민감하게 만든다.
- gradient 흐름을 안정화한다.
- mini-batch noise 때문에 약한 regularization 효과가 있다.

시험에서는 두 관점을 같이 쓰는 것이 안전합니다.

## 8. Dropout vs BatchNorm

| 항목 | Dropout | BatchNorm |
|---|---|---|
| 핵심 | 뉴런을 무작위 제거 | activation 분포 정규화 |
| 주효과 | regularization | optimization 안정화 |
| 확률성 | 마스크 샘플링 | batch statistics |
| 추론 시 | 모든 뉴런 사용, 스케일 조정 | running mean/variance 사용 |
| 해석 | ensemble | smoother landscape |

## 9. 시험 답안 문장

> 일반화는 학습 데이터의 경험적 위험이 아니라 실제 데이터 분포에서의 위험을 낮추는 문제다. 모델이 너무 단순하면 bias가 커져 underfitting이 되고, 너무 복잡하면 train error는 낮지만 test error가 커져 overfitting이 된다. 정규화는 $L_S(\theta)+\lambda C(\theta)$ 형태로 모델 복잡도에 penalty를 주어 과적합을 줄인다. Dropout은 무작위 subnetwork ensemble 효과로, BatchNorm은 activation을 batch 통계로 정규화해 최적화 지형을 부드럽게 하고 gradient 흐름을 안정화한다.

## 10. 연습

1. Dropout에서 inverted scaling을 쓰면 왜 기댓값이 유지되는지 보이라.
2. BatchNorm의 $\gamma,\beta$가 왜 필요한지 설명하라.
3. L2 regularization과 Gaussian prior의 관계를 다시 유도하라.
4. Dropout과 BatchNorm의 차이를 regularization/optimization 관점으로 구분하라.

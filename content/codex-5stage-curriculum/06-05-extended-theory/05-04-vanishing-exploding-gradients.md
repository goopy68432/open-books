---
title: "04. Vanishing / Exploding Gradient"
slug: 04-vanishing-exploding-gradients
order: 5
---

# 04. Vanishing / Exploding Gradient

깊은 네트워크의 역전파는 여러 층의 미분값을 곱하는 구조입니다. 이 곱이 반복되면 0으로 사라지거나 무한히 커질 수 있습니다.

## 1. 핵심 원인: 곱셈 체인

단순한 합성함수:

$$
h_L=f_L(f_{L-1}(\cdots f_1(x)))
$$

체인 룰:

$$
\frac{\partial h_L}{\partial x}
=
\prod_{l=1}^L f_l'(h_{l-1})
$$

각 항의 절댓값이 평균적으로 1보다 작으면:

$$
\left|\frac{\partial h_L}{\partial x}\right|\approx c^L,\quad 0<c<1
$$

깊이가 커질수록 0에 가까워집니다. 이것이 vanishing gradient입니다.

반대로 평균적으로 1보다 크면:

$$
c^L,\quad c>1
$$

기하급수적으로 커집니다. 이것이 exploding gradient입니다.

## 2. 신경망에서의 형태

역전파:

$$
\delta^{(l)}
=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})
$$

초기층까지 펼치면 여러 $W^T$와 $\sigma'$가 계속 곱해집니다.

$$
\delta^{(1)}
\approx
\left(\prod_{l=2}^L (W^{(l)})^T D^{(l-1)}\right)\delta^{(L)}
$$

여기서 $D^{(l)}=\operatorname{diag}(\sigma'(z^{(l)}))$입니다.

즉 gradient의 크기는 가중치 행렬의 스케일과 활성화 미분값의 스케일에 의해 결정됩니다.

## 3. Sigmoid가 위험한 이유

Sigmoid:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))\le\frac14
$$

10층만 지나도 활성화 미분 부분만 보면:

$$
\left(\frac14\right)^{10}\approx 9.5\times 10^{-7}
$$

초기층은 거의 학습하지 못합니다.

## 4. Tanh도 완전한 해결은 아님

Tanh:

$$
\tanh'(x)=1-\tanh^2(x)
$$

$x$가 0 근처이면 미분이 1에 가깝지만, 큰 양수/음수에서는 포화되어 0에 가까워집니다.

## 5. ReLU가 완화하는 이유

ReLU:

$$
ReLU'(x)=1\quad (x>0)
$$

양수 영역에서는 활성화 미분이 1이므로 sigmoid처럼 매번 0.25 이하를 곱하는 문제가 없습니다.

하지만 음수 영역에서는:

$$
ReLU'(x)=0
$$

이므로 dying ReLU 문제가 생길 수 있습니다.

## 6. Exploding Gradient

가중치 행렬의 spectral norm이 크거나 recurrent 구조에서 같은 행렬을 반복해서 곱하면 gradient가 커질 수 있습니다.

단순화:

$$
\delta^{(0)}\approx (W^T)^L\delta^{(L)}
$$

$W$의 큰 고유값/특이값이 1보다 크면:

$$
\|(W^T)^L\|\approx \|W\|^L
$$

이 되어 gradient가 폭발합니다.

## 7. 해결책

| 문제 | 해결 |
|---|---|
| Sigmoid saturation | ReLU, Leaky ReLU, GELU |
| 가중치 스케일 문제 | Xavier/He initialization |
| exploding gradient | gradient clipping |
| 깊은 네트워크의 전달 문제 | residual connection |
| activation 분포 변화 | BatchNorm, LayerNorm |

## 8. 초기화와 연결

가중치가 너무 작으면 activation과 gradient가 작아집니다. 너무 크면 폭발합니다.

Xavier 초기화는 tanh류에서 분산을 유지하려고:

$$
Var(W)\approx\frac{2}{n_{in}+n_{out}}
$$

He 초기화는 ReLU가 절반 정도를 0으로 만들기 때문에:

$$
Var(W)\approx\frac{2}{n_{in}}
$$

를 사용합니다.

## 9. 시험 답안 문장

> 깊은 신경망의 역전파는 각 층의 가중치 행렬과 활성화 함수의 도함수를 반복적으로 곱하는 구조이다. 이 곱의 평균 크기가 1보다 작으면 gradient가 깊이에 따라 지수적으로 작아져 vanishing gradient가 발생하고, 1보다 크면 exploding gradient가 발생한다. Sigmoid는 미분값이 최대 1/4라 깊은 네트워크에서 gradient를 빠르게 줄인다. ReLU, 적절한 초기화, BatchNorm, residual connection, gradient clipping 등이 이를 완화한다.

## 10. 연습

1. $\sigma'(x)\le1/4$가 vanishing gradient와 어떻게 연결되는지 설명하라.
2. $W$를 반복해서 곱하는 구조에서 exploding gradient가 생기는 이유를 고유값/특이값 관점으로 설명하라.
3. ReLU가 vanishing을 완화하지만 dying ReLU를 만들 수 있는 이유를 설명하라.
4. Xavier와 He 초기화의 차이를 설명하라.

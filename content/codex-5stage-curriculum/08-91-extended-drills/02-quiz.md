---
title: "확장 범위 훈련 문제"
slug: quiz
order: 2
---

# 확장 범위 훈련 문제

## A. 활성화 함수

1. $\sigma(x)=1/(1+e^{-x})$의 미분이 $\sigma(x)(1-\sigma(x))$임을 증명하라.
2. $Softplus(x)=\log(1+e^x)$의 미분을 구하라.
3. ReLU가 $x=0$에서 미분 불가능하다는 것을 좌미분/우미분으로 설명하라.
4. Sigmoid와 Tanh가 vanishing gradient를 일으키는 이유를 미분값 관점에서 설명하라.
5. Leaky ReLU가 dying ReLU를 완화하는 이유를 설명하라.

## B. Backpropagation

1. $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$, $a^{(l)}=\sigma(z^{(l)})$에서 $\delta^{(l)}$를 정의하라.
2. $\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})$를 index notation으로 유도하라.
3. $\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^T$의 shape을 설명하라.
4. softmax+CE 출력층에서 $\delta^{(L)}=p-y$가 되는 이유를 설명하라.

## C. 최적화

1. GD와 SGD의 차이를 설명하라.
2. Momentum이 진동을 줄이는 이유를 설명하라.
3. RMSProp이 adaptive learning rate를 만든다는 말의 의미를 설명하라.
4. Adam의 $m_t$, $v_t$, bias correction을 쓰라.
5. AdamW가 Adam과 weight decay를 분리한다는 뜻을 설명하라.

## D. 기울기 소실/폭발

1. 깊은 네트워크에서 gradient가 곱셈 체인이 되는 이유를 설명하라.
2. Sigmoid의 미분값 최대가 $1/4$라는 사실이 왜 위험한지 설명하라.
3. exploding gradient를 spectral norm 또는 고유값 관점에서 설명하라.
4. Xavier 초기화와 He 초기화의 차이를 설명하라.
5. gradient clipping은 어떤 문제를 해결하는가?

## E. 일반화와 정규화

1. empirical risk와 true risk의 차이를 설명하라.
2. Dropout에서 inverted scaling을 쓰면 왜 activation 기대값이 유지되는지 보이라.
3. BatchNorm의 $\mu_B$, $\sigma_B^2$, $\hat x$, $y$ 식을 쓰라.
4. BatchNorm의 internal covariate shift 설명과 smoother landscape 설명을 비교하라.
5. L2 regularization이 Gaussian prior의 MAP과 연결되는 과정을 설명하라.

## F. Bias-Variance와 정보이론

1. Bias-Variance 분해를 증명하라.
2. 교차항이 0이 되는 이유를 각각 설명하라.
3. Double Descent가 전통적 U자형 test error와 어떻게 다른지 설명하라.
4. $H(p,q)=H(p)+KL(p\|q)$를 증명하라.
5. Cross Entropy 최소화가 KL 최소화와 동치인 이유를 설명하라.

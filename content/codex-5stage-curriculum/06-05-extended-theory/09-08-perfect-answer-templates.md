---
title: "08. 확장 범위 시험 답안 템플릿"
slug: 08-perfect-answer-templates
order: 9
---

# 08. 확장 범위 시험 답안 템플릿

이 문서는 시험장에서 바로 쓸 수 있는 서술형 뼈대입니다.

## 1. 활성화 함수

문제:

> 활성화 함수가 필요한 이유와 sigmoid/ReLU의 장단점을 설명하라.

답안:

> 활성화 함수는 신경망에 비선형성을 부여한다. 만약 활성화 함수가 선형이면 여러 층의 합성 $W_2(W_1x+b_1)+b_2$는 다시 하나의 선형변환으로 합쳐지므로 깊이를 쌓아도 표현력이 증가하지 않는다. Sigmoid는 $\sigma(x)=1/(1+e^{-x})$이고 미분은 $\sigma'(x)=\sigma(x)(1-\sigma(x))$이다. 이 미분값은 최대 $1/4$라 깊은 네트워크에서 곱해지면 vanishing gradient를 일으킬 수 있다. ReLU는 $ReLU(x)=\max(0,x)$이고 양수 영역에서 미분이 1이므로 이를 완화하지만, 음수 영역에서는 미분이 0이라 dying ReLU 문제가 생길 수 있다.

## 2. Backpropagation

문제:

> 체인 룰을 이용해 L층 신경망의 역전파 공식을 설명하라.

답안:

> $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$, $a^{(l)}=\sigma(z^{(l)})$라 하자. $\delta^{(l)}=\partial L/\partial z^{(l)}$로 정의하면, 체인 룰에 의해 은닉층 오차는
> $$
> \delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})
> $$
> 로 뒤에서 앞으로 전파된다. 또한 $z^{(l)}$는 $W^{(l)}$에 선형으로 의존하므로
> $$
> \frac{\partial L}{\partial W^{(l)}}=\delta^{(l)}(a^{(l-1)})^T,\quad
> \frac{\partial L}{\partial b^{(l)}}=\delta^{(l)}
> $$
> 이다. 따라서 역전파는 계산 그래프를 역순으로 따라가며 체인 룰을 행렬 형태로 적용하는 알고리즘이다.

## 3. Optimizer

문제:

> SGD, Momentum, RMSProp, Adam의 차이를 설명하라.

답안:

> Gradient Descent는 $\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$로 손실이 가장 빠르게 증가하는 gradient의 반대 방향으로 이동한다. SGD는 전체 데이터 대신 mini-batch gradient를 사용해 계산을 빠르게 하고 noise를 도입한다. Momentum은 $m_t=\beta m_{t-1}+(1-\beta)g_t$처럼 gradient의 이동평균을 사용해 관성을 부여하므로 일관된 방향은 가속하고 진동은 줄인다. RMSProp은 gradient 제곱의 이동평균으로 $g_t/(\sqrt{s_t}+\epsilon)$ 형태의 좌표별 adaptive learning rate를 만든다. Adam은 이 둘을 결합해 1차 모멘트와 2차 모멘트를 모두 사용한다.

## 4. Vanishing / Exploding Gradient

문제:

> 깊은 신경망에서 기울기 소실과 폭발이 발생하는 수학적 원인을 설명하라.

답안:

> 역전파는 각 층의 가중치 행렬과 활성화 함수 도함수를 반복적으로 곱한다. 단순히 보면 $\delta^{(1)}$은 여러 $(W^{(l)})^T$와 $\operatorname{diag}(\sigma'(z^{(l)}))$의 곱으로 표현된다. 이 곱의 평균 크기가 1보다 작으면 깊이에 따라 gradient가 지수적으로 0에 가까워져 vanishing gradient가 발생하고, 1보다 크면 exploding gradient가 발생한다. Sigmoid는 미분값이 최대 $1/4$라 vanishing을 일으키기 쉽다. 해결책으로 ReLU/GELU, Xavier/He 초기화, BatchNorm, residual connection, gradient clipping 등이 있다.

## 5. Dropout

문제:

> Dropout의 원리와 regularization 효과를 설명하라.

답안:

> Dropout은 학습 중 각 뉴런의 출력을 확률적으로 0으로 만드는 방법이다. 마스크 $m_i\sim Bern(1-p)$를 두고 inverted dropout에서는 $\tilde h_i=m_i h_i/(1-p)$로 스케일링한다. 그러면 $E[\tilde h_i]=h_i$가 되어 추론 시 activation scale을 유지할 수 있다. 매 학습 step마다 다른 subnetwork를 학습하는 효과가 있으므로 ensemble과 유사하며, 특정 뉴런들 사이의 co-adaptation을 줄여 일반화를 돕는다.

## 6. Batch Normalization

문제:

> Batch Normalization의 수식과 역할을 설명하라.

답안:

> BatchNorm은 mini-batch의 평균과 분산을 이용해 activation을 정규화한다. $\mu_B=\frac1m\sum_i x^{(i)}$, $\sigma_B^2=\frac1m\sum_i(x^{(i)}-\mu_B)^2$라 하면, $\hat x^{(i)}=(x^{(i)}-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$로 정규화하고 $y^{(i)}=\gamma\hat x^{(i)}+\beta$로 scale과 shift를 학습한다. 이는 내부 공변량 변화 감소라는 동기로 제안되었고, 실제로는 최적화 지형을 부드럽게 하고 learning rate 민감도를 낮추며 gradient 흐름을 안정화하는 효과가 있다.

## 7. Bias-Variance

문제:

> $E[(y-\hat f)^2]=Bias^2+Variance+Noise$를 증명하라.

답안:

> $y=f+\epsilon$, $E[\epsilon]=0$, $Var(\epsilon)=\sigma^2$라 하자. 데이터셋 $D$에 대해 학습된 예측을 $\hat f_D$, 평균 예측을 $\bar f=E_D[\hat f_D]$라 두면,
> $$
> y-\hat f_D=(f-\bar f)+(\bar f-\hat f_D)+\epsilon
> $$
> 이다. 양변을 제곱하고 기댓값을 취하면 세 제곱항과 교차항이 생긴다. 교차항은 $E[\bar f-\hat f_D]=0$, $E[\epsilon]=0$, 잡음과 데이터셋의 독립성 때문에 모두 0이다. 따라서
> $$
> E[(y-\hat f_D)^2]=(\bar f-f)^2+E[(\hat f_D-\bar f)^2]+\sigma^2
> $$
> 이고 이는 $Bias^2+Variance+Noise$이다.

## 8. Cross Entropy = H + KL

문제:

> $H(p,q)=H(p)+KL(p\|q)$를 증명하고 의미를 설명하라.

답안:

> 정의에서 $H(p,q)=-\sum_i p_i\log q_i$이다. 여기에 $\log p_i$를 더하고 빼면
> $$
> H(p,q)=\sum_i p_i\log\frac{p_i}{q_i}-\sum_i p_i\log p_i
> =KL(p\|q)+H(p)
> $$
> 이다. 학습에서는 진짜 분포 $p$가 고정되어 있으므로 $H(p)$는 상수이다. 따라서 Cross Entropy를 최소화하는 것은 모델 분포 $q_\theta$와 진짜 분포 $p$ 사이의 $KL(p\|q_\theta)$를 최소화하는 것과 동치이다.

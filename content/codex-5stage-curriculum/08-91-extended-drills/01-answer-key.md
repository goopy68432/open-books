---
title: "확장 범위 답안 키"
slug: answer-key
order: 1
---

# 확장 범위 답안 키

## A. 활성화 함수

1. $\sigma(x)=(1+e^{-x})^{-1}$이므로
   $$
   \sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}
   =\sigma(x)(1-\sigma(x)).
   $$

2. 
   $$
   \frac{d}{dx}\log(1+e^x)=\frac{e^x}{1+e^x}=\sigma(x).
   $$

3. ReLU의 좌미분은 0, 우미분은 1이므로 $x=0$에서 미분 불가능하다. 구현에서는 subgradient를 관례적으로 둔다.

4. Sigmoid는 $\sigma'(x)\le1/4$, tanh는 포화 영역에서 $\tanh'(x)\to0$이다. 깊은 네트워크에서 이런 값들이 반복 곱해져 gradient가 0에 가까워진다.

5. Leaky ReLU는 음수 영역에서도 미분값 $\alpha>0$를 남기므로 gradient가 완전히 0이 되는 것을 완화한다.

## B. Backpropagation

1. 
   $$
   \delta^{(l)}:=\frac{\partial L}{\partial z^{(l)}}.
   $$

2. 
   $$
   \frac{\partial L}{\partial z^{(l)}_j}
   =
   \sum_k
   \frac{\partial L}{\partial z^{(l+1)}_k}
   \frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j}
   $$
   이고
   $$
   \frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j}
   =W^{(l+1)}_{kj}\sigma'(z^{(l)}_j).
   $$
   따라서 행렬형으로
   $$
   \delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)}).
   $$

3. $\delta^{(l)}$가 $n_l\times1$, $a^{(l-1)}$가 $n_{l-1}\times1$이면 외적은 $n_l\times n_{l-1}$이다. 이는 $W^{(l)}$의 shape과 같다.

4. Softmax 자코비안 $p_i(\delta_{ij}-p_j)$와 CE 미분 $-y_i/p_i$를 체인 룰로 곱하면 $p_j-y_j$가 남는다.

## C. 최적화

1. GD는 전체 데이터 gradient를 사용하고, SGD는 한 샘플 또는 mini-batch gradient를 사용한다. SGD는 빠르고 noise가 있지만 진동한다.

2. Momentum은 gradient의 이동평균을 사용하므로 일관된 방향은 누적되고, 부호가 자주 바뀌는 진동 방향은 평균화되어 줄어든다.

3. RMSProp은 $s_t=\rho s_{t-1}+(1-\rho)g_t^2$로 gradient 제곱 평균을 추정하고 $g_t/(\sqrt{s_t}+\epsilon)$로 나눈다. 따라서 좌표별 step 크기가 조정된다.

4. 
   $$
   m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
   v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
   $$
   $$
   \hat m_t=m_t/(1-\beta_1^t),\quad
   \hat v_t=v_t/(1-\beta_2^t)
   $$
   $$
   \theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
   $$

5. AdamW는 weight decay를 adaptive gradient 안에 섞지 않고, 파라미터를 직접 감소시키는 별도 항으로 적용한다.

## D. 기울기 소실/폭발

1. 역전파는 각 층의 Jacobian을 계속 곱하는 체인 룰 구조이기 때문이다.

2. $(1/4)^L$처럼 깊이에 따라 지수적으로 작아질 수 있다.

3. 반복 곱해지는 행렬의 spectral norm 또는 큰 고유값/특이값이 1보다 크면 $\|W^L\|$이 커져 gradient가 폭발한다.

4. Xavier는 tanh류에서 fan-in/fan-out을 모두 고려해 $2/(n_{in}+n_{out})$를 쓰고, He는 ReLU의 절반 비활성화를 고려해 $2/n_{in}$을 쓴다.

5. Gradient clipping은 exploding gradient가 너무 큰 update를 만드는 것을 막는다.

## E. 일반화와 정규화

1. Empirical risk는 학습 샘플 평균 손실이고, true risk는 실제 데이터 분포에 대한 기대 손실이다.

2. $m_i\sim Bern(1-p)$이면
   $$
   E\left[\frac{m_i}{1-p}h_i\right]=h_i.
   $$

3. 
   $$
   \mu_B=\frac1m\sum_i x^{(i)},\quad
   \sigma_B^2=\frac1m\sum_i(x^{(i)}-\mu_B)^2
   $$
   $$
   \hat x^{(i)}=\frac{x^{(i)}-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\quad
   y^{(i)}=\gamma\hat x^{(i)}+\beta.
   $$

4. Internal covariate shift 설명은 층 입력 분포 변화를 줄인다는 초기 동기이고, smoother landscape 설명은 최적화 지형을 부드럽게 해 gradient 흐름과 learning rate 안정성을 높인다는 관점이다.

5. Gaussian prior의 음의 로그가 $\|\theta\|^2/(2\tau^2)$가 되므로 MAP 목적함수는 NLL + L2 penalty가 된다.

## F. Bias-Variance와 정보이론

1. $\bar f=E_D[\hat f_D]$를 두고
   $$
   y-\hat f_D=(f-\bar f)+(\bar f-\hat f_D)+\epsilon
   $$
   로 분해한 뒤 제곱하고 기댓값을 취한다. 교차항이 0이 되어 $Bias^2+Variance+Noise$가 남는다.

2. $E[\bar f-\hat f_D]=0$, $E[\epsilon]=0$, 그리고 테스트 잡음 $\epsilon$은 학습 데이터와 독립이기 때문이다.

3. 전통적 관점은 모델 복잡도 증가에 따라 test error가 U자형이지만, Double Descent는 interpolation threshold 이후 overparameterized 영역에서 test error가 다시 감소한다.

4. 
   $$
   H(p,q)=-\sum_i p_i\log q_i
   =\sum_i p_i\log\frac{p_i}{q_i}-\sum_i p_i\log p_i
   =KL(p\|q)+H(p).
   $$

5. $p$는 고정된 진짜 분포이므로 $H(p)$는 상수다. 따라서 $H(p,q_\theta)$ 최소화와 $KL(p\|q_\theta)$ 최소화는 같은 최적점을 가진다.

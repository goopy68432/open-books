---
title: "07. Cross Entropy = Entropy + KL"
slug: 07-cross-entropy-kl
order: 8
---

# 07. Cross Entropy = Entropy + KL

Cross Entropy는 단순한 분류 손실이 아니라, 모델 분포 $q$가 진짜 분포 $p$에서 얼마나 벗어났는지를 재는 정보이론적 목적함수입니다.

## 1. Entropy

정의:

$$
H(p)=-\sum_i p_i\log p_i
$$

의미:

진짜 분포 $p$ 자체가 가진 불확실성입니다.

예:

- 한 클래스가 확률 1이면 불확실성 0
- 모든 클래스가 균일하면 불확실성 큼

## 2. Cross Entropy

정의:

$$
H(p,q)=-\sum_i p_i\log q_i
$$

의미:

진짜 분포는 $p$인데, 모델이 $q$라고 믿고 코딩하거나 예측할 때의 평균 손실입니다.

분류에서 one-hot 정답 $y$와 softmax 예측 $p_\theta$가 있으면:

$$
L=-\sum_i y_i\log p_{\theta,i}
$$

이것이 categorical cross entropy입니다.

## 3. KL divergence

정의:

$$
KL(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}
$$

의미:

$q$가 $p$를 얼마나 잘못 표현하는지의 차이입니다.

성질:

$$
KL(p\|q)\ge0
$$

그리고 $p=q$일 때 0입니다.

## 4. 핵심 분해

증명할 것:

$$
\boxed{
H(p,q)=H(p)+KL(p\|q)
}
$$

증명:

$$
H(p,q)
=-\sum_i p_i\log q_i
$$

$\log q_i$에 $\log p_i$를 끼워 넣습니다.

$$
=-\sum_i p_i\log q_i+\sum_i p_i\log p_i-\sum_i p_i\log p_i
$$

앞의 두 항을 묶으면:

$$
=\sum_i p_i\log\frac{p_i}{q_i}-\sum_i p_i\log p_i
$$

따라서:

$$
=KL(p\|q)+H(p)
$$

## 5. 머신러닝에서의 의미

학습 데이터의 진짜 분포 $p$는 고정되어 있습니다. 따라서 $H(p)$는 모델 파라미터 $\theta$와 무관한 상수입니다.

$$
\min_\theta H(p,q_\theta)
=
\min_\theta \{H(p)+KL(p\|q_\theta)\}
$$

상수 $H(p)$는 최적화에 영향을 주지 않으므로:

$$
\min_\theta H(p,q_\theta)
\equiv
\min_\theta KL(p\|q_\theta)
$$

즉 Cross Entropy를 최소화하는 것은 모델 분포 $q_\theta$를 진짜 분포 $p$에 가깝게 만드는 것입니다.

## 6. NLL과의 연결

one-hot 정답 $y$에서 정답 클래스가 $c$라면:

$$
y_c=1,\quad y_i=0\ (i\ne c)
$$

Cross Entropy:

$$
H(y,p_\theta)=-\sum_i y_i\log p_{\theta,i}
=-\log p_{\theta,c}
$$

즉 정답 클래스의 negative log-likelihood입니다.

따라서:

```text
Categorical NLL = Cross Entropy = H(p) + KL(p||q)
```

## 7. Bernoulli BCE와의 연결

이진 분류:

$$
y\in\{0,1\},\quad q=P(y=1|x)
$$

Bernoulli likelihood:

$$
p(y|q)=q^y(1-q)^{1-y}
$$

NLL:

$$
-\log p(y|q)
=-[y\log q+(1-y)\log(1-q)]
$$

이것이 Binary Cross Entropy입니다.

## 8. 시험 답안 문장

> Cross Entropy는 $H(p,q)=-\sum_i p_i\log q_i$로 정의된다. 여기에 $\log p_i$를 더하고 빼면 $H(p,q)=\sum_i p_i\log(p_i/q_i)-\sum_i p_i\log p_i=KL(p\|q)+H(p)$가 된다. 학습에서 진짜 분포 $p$는 고정되어 있으므로 $H(p)$는 상수이고, Cross Entropy 최소화는 $KL(p\|q_\theta)$ 최소화와 동치이다. one-hot 정답에서는 CE가 정답 클래스의 negative log-likelihood와 같다.

## 9. 연습

1. $H(p,q)=H(p)+KL(p\|q)$를 정의에서 증명하라.
2. one-hot 정답에서 CE가 $-\log q_c$가 됨을 보여라.
3. Bernoulli NLL이 BCE와 같음을 보여라.
4. 왜 CE 최소화와 KL 최소화가 같은 최적점을 갖는지 설명하라.

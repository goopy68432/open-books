---
title: "01. 활성화 함수"
slug: 05-extended-theory-01-activation-functions
order: 14
---

# 01. 활성화 함수

활성화 함수는 신경망에 비선형성을 넣는 장치입니다. 활성화 함수가 없으면 여러 층을 쌓아도 결국 하나의 선형변환으로 합쳐집니다.

## 1. 왜 비선형성이 필요한가

두 층 네트워크가 모두 선형이면:

$$
f(x)=W_2(W_1x+b_1)+b_2
=(W_2W_1)x+(W_2b_1+b_2)
$$

즉 깊은 모델이 아니라 다시 하나의 선형 모델입니다. 따라서 깊이의 표현력을 얻으려면 중간에 비선형 활성화 함수가 필요합니다.

## 2. Sigmoid

정의:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

범위: $(0,1)$

미분:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

유도:

$$
\sigma(x)=(1+e^{-x})^{-1}
$$

$$
\sigma'(x)=-(1+e^{-x})^{-2}(-e^{-x})
=\frac{e^{-x}}{(1+e^{-x})^2}
$$

$$
=\frac{1}{1+e^{-x}}-\frac{1}{(1+e^{-x})^2}
=\sigma(x)-\sigma(x)^2
$$

단점:

$$
0<\sigma'(x)\le \frac14
$$

깊은 층에서 이 값이 계속 곱해지면 gradient가 빠르게 작아집니다.

## 3. Tanh

정의:

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

범위: $(-1,1)$

미분:

$$
\tanh'(x)=1-\tanh^2(x)
$$

Sigmoid보다 평균이 0에 가까워 학습이 조금 낫지만, 큰 $|x|$에서 미분이 0에 가까워지는 문제는 같습니다.

## 4. ReLU

정의:

$$
ReLU(x)=\max(0,x)
=
\begin{cases}
x,&x>0\\
0,&x\le0
\end{cases}
$$

미분:

$$
ReLU'(x)=
\begin{cases}
1,&x>0\\
0,&x<0
\end{cases}
$$

$x=0$에서는 미분 불가능합니다. 구현에서는 보통 subgradient로 0 또는 1 중 하나를 택합니다. 시험에서는 "$x=0$에서 미분 불가능하지만 실제 구현에서는 관례적으로 subgradient를 둔다"고 쓰면 충분합니다.

장점:

- 양수 영역에서 미분이 1이라 vanishing gradient를 완화합니다.
- 계산이 매우 단순합니다.

단점:

- 음수 영역에서는 gradient가 0입니다.
- 한 뉴런이 계속 음수만 내면 학습이 멈추는 dying ReLU가 생길 수 있습니다.

## 5. Leaky ReLU

정의:

$$
LeakyReLU(x)=
\begin{cases}
x,&x>0\\
\alpha x,&x\le0
\end{cases}
\quad 0<\alpha\ll1
$$

미분:

$$
LeakyReLU'(x)=
\begin{cases}
1,&x>0\\
\alpha,&x<0
\end{cases}
$$

음수 영역에도 작은 gradient를 남겨 dying ReLU를 완화합니다.

## 6. GELU

정의:

$$
GELU(x)=x\Phi(x)
$$

$\Phi(x)$는 표준정규분포 CDF입니다.

미분:

$$
GELU'(x)=\Phi(x)+x\phi(x)
$$

$\phi(x)$는 표준정규분포 pdf입니다.

직관:

ReLU처럼 단순히 $x>0$이면 통과시키는 것이 아니라, 입력이 양수일 확률에 따라 부드럽게 통과시킵니다. Transformer 계열에서 자주 사용됩니다.

## 7. Softplus

정의:

$$
Softplus(x)=\log(1+e^x)
$$

미분:

$$
Softplus'(x)=\frac{e^x}{1+e^x}=\sigma(x)
$$

ReLU의 매끄러운 근사로 볼 수 있습니다.

## 8. 한 표로 정리

| 함수 | 식 | 미분 | 핵심 단점 |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | $\sigma(1-\sigma)$ | saturating, vanishing |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $1-\tanh^2(x)$ | saturating, vanishing |
| ReLU | $\max(0,x)$ | $1_{x>0}$ | dying ReLU |
| Leaky ReLU | $x$ or $\alpha x$ | $1$ or $\alpha$ | $\alpha$ 선택 필요 |
| GELU | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ | 계산 복잡 |
| Softplus | $\log(1+e^x)$ | $\sigma(x)$ | 큰 음수에서 gradient 작음 |

## 9. 시험 답안 문장

> 활성화 함수는 신경망에 비선형성을 부여한다. 활성화가 없으면 여러 선형층의 합성은 다시 하나의 선형층으로 합쳐져 깊이의 의미가 사라진다. Sigmoid와 tanh는 미분값이 1보다 작고 포화 구간에서 0에 가까워 vanishing gradient를 일으킬 수 있다. ReLU는 양수 영역에서 미분값이 1이라 이를 완화하지만, 음수 영역에서는 gradient가 0이 되어 dying ReLU 문제가 생길 수 있다.

## 10. 연습

1. Sigmoid 미분을 직접 유도하라.
2. Softplus의 미분이 sigmoid임을 보여라.
3. ReLU의 $x=0$에서 미분 가능성을 논하라.
4. 왜 tanh가 sigmoid보다 hidden layer에서 더 선호되었는지 설명하라.
5. GELU의 미분을 곱의 미분으로 유도하라.

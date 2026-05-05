---
title: "중급 3: Softmax와 Cross Entropy"
slug: 02-intermediate-03-softmax-and-loss
order: 8
---

# 중급 3: Softmax와 Cross Entropy

softmax 문제는 계산만 보면 길지만 구조는 단순합니다. 분자와 분모가 모두 $z_j$에 의존하므로 케이스를 나누면 됩니다.

## 1. softmax 정의

$$
p_i=\frac{e^{z_i}}{S},\quad S=\sum_{k=1}^c e^{z_k}
$$

역할:

| 기호 | 의미 |
|---|---|
| $z_i$ | class $i$의 점수 |
| $e^{z_i}$ | 양수화된 점수 |
| $S$ | 전체 점수 합 |
| $p_i$ | class $i$의 확률 |

## 2. 자코비안 원소 계산

구하려는 것:

$$
\frac{\partial p_i}{\partial z_j}
$$

분모:

$$
\frac{\partial S}{\partial z_j}=e^{z_j}
$$

### 케이스 1: $i=j$

$$
\frac{\partial p_i}{\partial z_i}
=\frac{e^{z_i}S-e^{z_i}e^{z_i}}{S^2}
=\frac{e^{z_i}}{S}\left(1-\frac{e^{z_i}}{S}\right)
$$

결론:

$$
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)
$$

해석: 자기 점수를 올리면 자기 확률은 증가하지만, 이미 $p_i$가 1에 가까우면 증가 여지가 작습니다.

### 케이스 2: $i\ne j$

분자 $e^{z_i}$는 $z_j$와 무관하므로 미분하면 0입니다.

$$
\frac{\partial p_i}{\partial z_j}
=\frac{0\cdot S-e^{z_i}e^{z_j}}{S^2}
=-p_ip_j
$$

해석: 남의 점수를 올리면 내 확률은 감소합니다.

## 3. 한 줄 통합

크로네커 델타:

$$
\delta_{ij}=
\begin{cases}
1,&i=j\\
0,&i\ne j
\end{cases}
$$

통합식:

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
$$

행렬식:

$$
J=\operatorname{diag}(p)-pp^T
$$

## 4. 검산: 행 합

각 행의 합:

$$
\sum_j \frac{\partial p_i}{\partial z_j}
=\sum_j p_i(\delta_{ij}-p_j)
=p_i-p_i\sum_jp_j
=0
$$

의미: softmax 확률들의 합이 1이라는 제약이 유지됩니다.

## 5. Cross Entropy와 결합

one-hot 정답 $y$에 대해:

$$
L=-\sum_i y_i\log p_i
$$

체인 룰:

$$
\frac{\partial L}{\partial z_j}
=\sum_i \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}
$$

각 항:

$$
\frac{\partial L}{\partial p_i}=-\frac{y_i}{p_i}
$$

대입:

$$
\frac{\partial L}{\partial z_j}
=\sum_i\left(-\frac{y_i}{p_i}\right)p_i(\delta_{ij}-p_j)
$$

$$
=\sum_i(-y_i\delta_{ij}+y_ip_j)
=-y_j+p_j\sum_i y_i
$$

one-hot이면 $\sum_i y_i=1$:

$$
\frac{\partial L}{\partial z_j}=p_j-y_j
$$

## 6. 왜 중요한가

딥러닝 분류 모델에서 마지막 층의 그래디언트가 "예측 확률 - 정답"이라는 단순한 형태가 됩니다. 이 결과는 softmax 자코비안과 cross entropy가 잘 맞물리기 때문에 생깁니다.

## 7. 중급 훈련

1. $c=3$, $p=(0.2,0.5,0.3)^T$일 때 자코비안 행렬을 직접 써라.
2. 위 자코비안의 각 행 합이 0인지 확인하라.
3. 정답이 class 2일 때 $\nabla_z L$을 구하라.
4. $p_i(1-p_i)$가 항상 0 이상인 이유를 설명하라.

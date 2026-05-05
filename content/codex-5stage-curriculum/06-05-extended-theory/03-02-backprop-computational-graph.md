---
title: "02. Backpropagation과 계산 그래프"
slug: 02-backprop-computational-graph
order: 3
---

# 02. Backpropagation과 계산 그래프

역전파는 새로운 미분법이 아닙니다. 체인 룰을 계산 그래프에 대해 뒤에서 앞으로 효율적으로 적용한 알고리즘입니다.

## 1. 한 층의 순전파

층 $l$에서:

$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
$$

$$
a^{(l)}=\sigma(z^{(l)})
$$

입력은 $a^{(0)}=x$, 출력은 $a^{(L)}=\hat y$입니다.

## 2. 계산 그래프

```text
x = a^(0)
   → z^(1)=W^(1)a^(0)+b^(1)
   → a^(1)=σ(z^(1))
   → z^(2)=W^(2)a^(1)+b^(2)
   → a^(2)=σ(z^(2))
   → ...
   → a^(L)
   → Loss L
```

순전파에서는 $z^{(l)}$, $a^{(l)}$를 저장합니다. 역전파에서 이 값들이 필요하기 때문입니다.

## 3. 핵심 기호: delta

정의:

$$
\delta^{(l)}:=\frac{\partial L}{\partial z^{(l)}}
$$

즉 $l$층 pre-activation $z^{(l)}$에 대한 손실의 기울기입니다.

## 4. 출력층 delta

일반형:

$$
\delta^{(L)}
=\nabla_{a^{(L)}}L\odot\sigma'(z^{(L)})
$$

단, softmax + cross entropy 출력층에서는 훨씬 단순해집니다.

$$
\delta^{(L)}=p-y
$$

이 결과는 기존 `02-intermediate/03-softmax-and-loss.md`의 softmax 자코비안 유도와 연결됩니다.

## 5. 은닉층 delta 전파

목표:

$$
\delta^{(l)}=\frac{\partial L}{\partial z^{(l)}}
$$

다음 층의 식:

$$
z^{(l+1)}=W^{(l+1)}a^{(l)}+b^{(l+1)}
$$

$$
a^{(l)}=\sigma(z^{(l)})
$$

체인 룰:

$$
\frac{\partial L}{\partial z^{(l)}_j}
=
\sum_k
\frac{\partial L}{\partial z^{(l+1)}_k}
\frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j}
$$

그런데:

$$
z^{(l+1)}_k=\sum_i W^{(l+1)}_{ki}\sigma(z^{(l)}_i)+b^{(l+1)}_k
$$

따라서:

$$
\frac{\partial z^{(l+1)}_k}{\partial z^{(l)}_j}
=W^{(l+1)}_{kj}\sigma'(z^{(l)}_j)
$$

대입:

$$
\delta^{(l)}_j
=
\sigma'(z^{(l)}_j)\sum_k W^{(l+1)}_{kj}\delta^{(l+1)}_k
$$

행렬형:

$$
\boxed{
\delta^{(l)}
=
(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})
}
$$

여기서 전치행렬이 등장하는 이유는 뒤층의 오차를 앞층의 각 노드로 다시 분배하기 때문입니다.

## 6. 가중치와 편향의 gradient

$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$이므로:

$$
\boxed{
\frac{\partial L}{\partial W^{(l)}}=\delta^{(l)}(a^{(l-1)})^T
}
$$

$$
\boxed{
\frac{\partial L}{\partial b^{(l)}}=\delta^{(l)}
}
$$

미니배치에서는 각 샘플의 gradient를 평균냅니다.

## 7. Backprop 4줄 요약

| 단계 | 식 |
|---|---|
| 출력층 오차 | $\delta^{(L)}=\nabla_{a^{(L)}}L\odot\sigma'(z^{(L)})$ |
| 은닉층 오차 | $\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})$ |
| 가중치 gradient | $\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^T$ |
| 편향 gradient | $\partial L/\partial b^{(l)}=\delta^{(l)}$ |

## 8. 체인 룰을 행렬 형태로 본다는 뜻

스칼라 체인 룰:

$$
\frac{dL}{dx}=\frac{dL}{du}\frac{du}{dx}
$$

벡터 체인 룰:

$$
\nabla_xL
=
\left(\frac{\partial u}{\partial x}\right)^T\nabla_uL
$$

역전파는 이 벡터-야코비안 곱을 레이어마다 반복합니다.

## 9. 시험 답안 문장

> 역전파는 손실에서 시작해 계산 그래프를 역순으로 따라가며 체인 룰을 적용하는 알고리즘이다. 각 층에서 $\delta^{(l)}=\partial L/\partial z^{(l)}$를 정의하면, 은닉층 오차는 $\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})$로 재귀적으로 계산된다. 이후 $z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$이므로 $\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^T$이다. 따라서 backprop은 체인 룰을 행렬 형태로 효율적으로 적용한 것이다.

## 10. 연습

1. 2층 신경망의 forward 식을 쓰고 모든 $z,a$를 표시하라.
2. $\delta^{(l)}$ 식에서 왜 $(W^{(l+1)})^T$가 나오는지 index notation으로 설명하라.
3. $\partial L/\partial W^{(l)}$의 shape이 $W^{(l)}$와 같은지 확인하라.
4. softmax+CE에서 출력층 $\delta$가 왜 $p-y$인지 다시 유도하라.

---
title: "02. 왜 로그를 취하는가 — 3대 이유"
slug: 09-killer-chains-02-why-log
order: 52
---

# 02. 왜 로그를 취하는가 — 3대 이유

> 답안에 매번 등장. **3가지 이유 모두** 적어야 만점.

## 이유 1: 곱 → 합 (미분 단순화)

i.i.d 가정 하의 우도:
$$L(\theta) = \prod_{i=1}^n p(y_i | \theta)$$

이건 **곱**. 곱의 미분은:
$$L'(\theta) = \sum_i \left(\prod_{j \neq i} p(y_j|\theta)\right) p'(y_i|\theta)$$

→ **n개의 곱이 n개**! 매우 복잡.

로그를 취하면:
$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(y_i | \theta)$$

미분:
$$\ell'(\theta) = \sum_{i=1}^n \frac{p'(y_i|\theta)}{p(y_i|\theta)}$$

**합의 합**으로 매우 단순.

## 이유 2: 단조성 (argmax 보존)

로그 함수는 **단조증가**:
$$x_1 < x_2 \Rightarrow \log x_1 < \log x_2$$

따라서:
$$\arg\max_\theta L(\theta) = \arg\max_\theta \log L(\theta)$$

**최댓값 위치**가 보존됨. 즉, $\log L$을 최대화한 θ는 $L$을 최대화한 θ와 **정확히 같다**.

### 증명 (시험 답안용 한 줄)

$\theta^* = \arg\max L$이면 $L(\theta^*) \geq L(\theta) \forall \theta$. 단조성으로 $\log L(\theta^*) \geq \log L(\theta) \forall \theta$. 따라서 $\theta^* = \arg\max \log L$.

## 이유 3: 수치 안정성 (underflow 방지)

n이 크고 각 확률이 작을 때:

$$L = p_1 \cdot p_2 \cdots p_n$$

각 $p_i \approx 0.01$이고 n=100이면:
$$L \approx 10^{-200}$$

→ 컴퓨터에서 **0으로 표현됨** (double precision은 $\sim 10^{-308}$이 한계, 그러나 곱 누적 중간에 underflow).

로그 취하면:
$$\log L = \sum \log p_i \approx -200 \cdot \log 10 \approx -460$$

→ 안전한 큰 음수.

## 시험 답안 표준 3줄 문장

답안에 **반드시** 적기:

> "(1) i.i.d 가정으로 우도가 곱 형태이고, 로그는 곱을 합으로 바꿔 미분을 단순화한다.
> (2) 로그는 단조증가 함수이므로 $\arg\max L = \arg\max \log L$이 보장된다.
> (3) 작은 확률의 곱은 컴퓨터에서 underflow를 일으키나, 로그를 취하면 수치적으로 안정하다.
> 이 세 이유로 로그우도 $\ell = \log L$을 다룬다."

## 한 줄 요약

> "로그는 곱을 합으로, 작은 수를 안전한 음수로, argmax를 보존하면서 바꾼다."

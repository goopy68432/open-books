---
title: "06. Bias-Variance 분해와 Double Descent"
slug: 06-bias-variance-double-descent
order: 7
---

# 06. Bias-Variance 분해와 Double Descent

Bias-Variance 분해는 일반화 이론의 대표 증명입니다. 시험에서는 분해 공식을 외우는 것보다, 중간에 평균 예측값을 더하고 빼는 과정을 재현해야 합니다.

## 1. 설정

진짜 함수:

$$
y=f(x)+\epsilon
$$

잡음:

$$
E[\epsilon]=0,\quad Var(\epsilon)=\sigma^2
$$

학습 데이터셋 $D$로 학습한 모델:

$$
\hat f_D(x)
$$

데이터셋이 바뀌면 $\hat f_D$도 바뀌므로, $\hat f_D(x)$는 $D$에 대한 확률변수입니다.

## 2. 평균 예측

고정된 테스트 점 $x$에서:

$$
\bar f(x)=E_D[\hat f_D(x)]
$$

즉 여러 학습 데이터셋으로 모델을 여러 번 학습했을 때 평균 예측입니다.

## 3. 분해 공식

테스트 예측 오차:

$$
E_{D,\epsilon}[(y-\hat f_D(x))^2]
$$

결과:

$$
\boxed{
E[(y-\hat f_D(x))^2]
=
(\bar f(x)-f(x))^2
+
E_D[(\hat f_D(x)-\bar f(x))^2]
+
\sigma^2
}
$$

즉:

$$
\text{Error}=Bias^2+Variance+Noise
$$

## 4. 증명

$x$를 고정하고 $f=f(x)$, $\hat f=\hat f_D(x)$, $\bar f=E_D[\hat f_D(x)]$로 씁니다.

$$
y=f+\epsilon
$$

시작:

$$
E[(y-\hat f)^2]
=E[((f+\epsilon)-\hat f)^2]
$$

$\bar f$를 더하고 뺍니다.

$$
=E[((f-\bar f)+(\bar f-\hat f)+\epsilon)^2]
$$

제곱 전개:

$$
=E[(f-\bar f)^2]
+E[(\bar f-\hat f)^2]
+E[\epsilon^2]
$$

$$
+2E[(f-\bar f)(\bar f-\hat f)]
+2E[(f-\bar f)\epsilon]
+2E[(\bar f-\hat f)\epsilon]
$$

교차항은 모두 0입니다.

첫 번째 교차항:

$$
E[(f-\bar f)(\bar f-\hat f)]
=(f-\bar f)E[\bar f-\hat f]
$$

그런데:

$$
E[\bar f-\hat f]=\bar f-E[\hat f]=0
$$

두 번째:

$$
E[(f-\bar f)\epsilon]=(f-\bar f)E[\epsilon]=0
$$

세 번째는 $\epsilon$이 학습 데이터와 독립이고 평균 0이므로:

$$
E[(\bar f-\hat f)\epsilon]=E[\bar f-\hat f]E[\epsilon]=0
$$

남는 항:

$$
(f-\bar f)^2+E[(\bar f-\hat f)^2]+\sigma^2
$$

따라서:

$$
Bias^2+Variance+Noise
$$

입니다.

## 5. 의미

| 항 | 의미 |
|---|---|
| Bias | 평균 예측이 진짜 함수에서 얼마나 벗어나는가 |
| Variance | 데이터셋이 바뀔 때 예측이 얼마나 흔들리는가 |
| Noise | 데이터 자체의 제거 불가능한 잡음 |

일반적인 전통 관점:

- 모델이 너무 단순하면 bias가 큽니다.
- 모델이 너무 복잡하면 variance가 큽니다.
- 최적 복잡도는 두 항의 합을 적절히 줄이는 지점입니다.

## 6. Underfitting과 Overfitting

Underfitting:

- 모델 표현력이 부족합니다.
- train error도 높고 test error도 높습니다.
- bias가 큽니다.

Overfitting:

- 학습 데이터에 너무 맞춥니다.
- train error는 낮지만 test error가 높습니다.
- variance가 큽니다.

## 7. Double Descent

전통적 Bias-Variance 관점은 모델 복잡도가 증가하면 test error가 U자 형태를 보인다고 설명합니다.

현대 딥러닝에서는 종종 다음 현상이 관찰됩니다.

1. 모델 복잡도 증가 초반: test error 감소
2. interpolation threshold 근처: test error 증가
3. 더 큰 overparameterized 영역: test error가 다시 감소

이것을 Double Descent라고 부릅니다.

```text
test error
   │\          /\
   │ \        /  \____
   │  \______/        
   └──────────────────── model size
       under   interp   over
```

핵심 직관:

- 파라미터 수가 데이터 수 근처일 때는 데이터를 겨우 맞추며 variance가 커질 수 있습니다.
- 훨씬 큰 모델에서는 많은 해 중 SGD가 비교적 단순한 해, 예를 들어 작은 norm의 해를 찾는 implicit regularization이 작동할 수 있습니다.

## 8. 시험 답안 문장

> Bias-Variance 분해는 테스트 제곱오차를 평균 예측의 오차, 학습 데이터 변화에 따른 예측의 흔들림, 제거 불가능한 잡음으로 나누는 공식이다. $\bar f=E_D[\hat f_D]$를 더하고 빼면 $y-\hat f=(f-\bar f)+(\bar f-\hat f)+\epsilon$이고, 제곱 후 기댓값을 취하면 교차항은 평균 0과 독립성 때문에 사라진다. 따라서 오차는 $Bias^2+Variance+Noise$로 분해된다. Double Descent는 현대 과파라미터 모델에서 interpolation 이후 test error가 다시 감소하는 현상으로, SGD의 implicit regularization과 연결해 설명할 수 있다.

## 9. 연습

1. 분해 증명에서 왜 $\bar f$를 더하고 빼는지 설명하라.
2. 교차항 3개가 각각 왜 0인지 쓰라.
3. underfitting과 overfitting을 bias/variance 관점으로 설명하라.
4. Double Descent가 전통적 U자 곡선과 어떻게 다른지 설명하라.

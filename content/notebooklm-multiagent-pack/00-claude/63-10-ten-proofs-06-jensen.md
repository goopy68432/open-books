---
title: "06. Jensen 부등식"
slug: 10-ten-proofs-06-jensen
order: 63
---

# 06. Jensen 부등식

## 정리

$f$가 **볼록함수**(convex)이면:
$$f(E[X]) \leq E[f(X)]$$

오목(concave)이면 부등호 반대.

## 직관

볼록함수: 그래프가 위로 볼록 (예: $f(x) = x^2$).
- 두 점의 평균에서 함수값 ≤ 함수값들의 평균.

```
       f(x)
        │       ╱
        │      ╱  ← f(E[X])
        │   ╱
        │ E[f(X)]
        └────────── x
            ↑
          E[X]
```

## 증명 (이산형, 두 점)

$X = a$ 확률 $p$, $X = b$ 확률 $1-p$.

$E[X] = pa + (1-p)b$.

볼록 정의: $f(pa + (1-p)b) \leq p f(a) + (1-p) f(b) = E[f(X)]$. ∎

(일반 분포는 측도론적 확장)

## 응용

### KL ≥ 0
$\log$는 오목 → $\log E \geq E \log$ → KL = E log(p/q) ≥ 0.

### 분산 ≥ 0
$x^2$는 볼록 → $(E[X])^2 \leq E[X^2]$ → $\text{Var}[X] = E[X^2] - (E[X])^2 \geq 0$.

## 시험 답안용

> "Jensen: 볼록 f에 대해 $f(E[X]) \leq E[f(X)]$. KL ≥ 0, 분산 ≥ 0 등 핵심 부등식의 출발점."

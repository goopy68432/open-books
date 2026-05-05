---
title: '01. 비대칭 prior $\theta^m$의 의미'
slug: concept
order: 1
---

# 01. 비대칭 prior $\theta^m$의 의미

## 1. prior 형태

$$p(\theta) \propto \theta^m, \quad \theta \in [0, 1]$$

대칭 prior와 달리 $1-\theta$ 항이 **없다**. 즉:
- $\theta = 0$에서 prior = 0 (절대 0이 아니라고 사전 가정)
- $\theta = 1$에서 prior 최대값

```
   p(θ)
    │
    │            ╱
    │          ╱    ← m이 클수록 더 가파르게 1로 쏠림
    │        ╱
    │     ╱
    │__╱_________
    └──┼─────┼──── θ
       0     1
```

## 2. m별 모양

- m = 0: 균일 (정보 없음)
- m = 1: 선형 증가 ($p \propto \theta$)
- m = 2: 2차 함수 ($p \propto \theta^2$)
- m → ∞: θ = 1에 디랙 델타

## 3. 직관

**"이 동전은 사기로 무조건 앞면 (θ=1)에 가깝다는 강한 사전 믿음"**

m이 클수록 이 믿음이 강해짐. m → ∞이면 어떤 데이터(앞면 0번이라도)에도 θ = 1을 우긴다.

## 다음

[`02-derivation.md`](./02-derivation.md)

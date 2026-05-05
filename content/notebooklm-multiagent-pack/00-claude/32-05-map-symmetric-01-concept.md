---
title: "01. MAP — 사전 믿음 + 데이터"
slug: 05-map-symmetric-01-concept
order: 32
---

# 01. MAP — 사전 믿음 + 데이터

---

## 1. MLE vs MAP — 한 줄 비교

| | MLE | MAP |
|---|-----|-----|
| 최대화 대상 | $L(\theta) = p(D\|\theta)$ | $p(\theta\|D) \propto p(D\|\theta) p(\theta)$ |
| 사전믿음 | 없음 | prior $p(\theta)$로 표현 |
| 결과 | 데이터만 반영 | 데이터 + 사전믿음 결합 |

---

## 2. 베이즈 정리

$$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}$$

용어:
- **Posterior** $p(\theta|D)$: 데이터 본 후 θ에 대한 분포
- **Likelihood** $p(D|\theta)$: θ가 주어졌을 때 데이터 확률 (= 우도)
- **Prior** $p(\theta)$: 데이터 보기 전 θ에 대한 사전 분포
- **Evidence** $p(D)$: 정규화 상수 (θ 무관)

### MAP의 핵심

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta | D) = \arg\max_\theta p(D|\theta) p(\theta)$$

(p(D)는 θ 무관 → 무시)

→ **likelihood × prior** 곱을 최대화.

---

## 3. 이번 문제의 prior

$$p(\theta) \propto \theta^m (1-\theta)^m$$

이건 **Beta(m+1, m+1)** 분포.

### 대칭성

$\theta = 0.5$ 기준 좌우 **대칭**:
- 0과 1 양 끝에서 prior = 0
- 0.5에서 정점

```
   p(θ)
    │      ╱╲     ← 정점 0.5
    │     ╱  ╲
    │    ╱    ╲
    │   ╱      ╲
    │__╱        ╲__
    └──┼─────┼────── θ
       0   0.5     1
```

### m이 클수록 첨예

m이 커지면 0.5 주변이 더 좁고 높음.
- m = 0: prior = 1 (균일, 정보 없음) → MAP = MLE
- m = 1: 약한 정보
- m → ∞: 0.5에 디랙 델타 (확신!)

---

## 4. 직관 — 왜 m → ∞이면 0.5?

**비유:** 친구가 "이 동전은 100% 공정해 (앞면 0.5)"라고 매우 강하게 주장. 데이터 100번 던져 70번 앞면이 나와도, 친구의 주장이 너무 강하면 우리도 "데이터가 우연히 그렇게 나왔겠지" 생각.

→ **prior 무한 강함 = 데이터 무시 = 0.5**

---

## 5. 풀이 미리보기

```
posterior ∝ likelihood × prior
         = θ^k (1-θ)^(n-k) × θ^m (1-θ)^m
         = θ^(k+m) (1-θ)^(n-k+m)

log posterior = (k+m) log θ + (n-k+m) log(1-θ)

미분=0:
(k+m)/θ - (n-k+m)/(1-θ) = 0
→ θ̂_MAP = (k+m)/(n+2m)

m → ∞: 분자/분모를 m으로 나누면
   = (k/m + 1)/(n/m + 2) → (0+1)/(0+2) = 1/2
```

---

## 다음

[`02-derivation.md`](./02-derivation.md)

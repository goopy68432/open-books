---
title: "딥러닝 이론 기말 — 시험 직전 빠른 복습 시트"
slug: 04-quick-review-sheet
order: 8
---

# 딥러닝 이론 기말 — 시험 직전 빠른 복습 시트

> 시험 30분 전에 이것만 훑어보라

---

## 1. 유도 체인 한눈에 보기

```
[회귀]  Gaussian Noise → i.i.d. → log → NLL = MSE + const → MSE 최소화
[분류]  Categorical    → i.i.d. → log → NLL = CE          → CE 최소화
[정규화] Gaussian Prior → log    → ‖θ‖² 항 추가           → L2 Reg = Weight Decay = Ridge
[소프트] 제약 최적화    → Lagrangian → ∂L/∂p=0 → softmax(z/τ)
```

**공통 패턴**: 확률 가정 → i.i.d.(곱) → log(합) → 미분=0(Fermat) → 해

---

## 2. "왜" 한 줄 모음 (채점 포인트)

| 단계 | "왜?" |
|------|-------|
| i.i.d. 가정 | 결합확률을 곱으로 분해하기 위해 |
| log 변환 | 곱→합, 수치안정, 단조증가(argmax 불변) |
| 미분=0 | Fermat 정리: 내부 극값의 필요조건 |
| NLL (부호반전) | 최대화 → 최소화 (관례, optimizer 호환) |
| √D_Q 스케일링 | 내적 분산이 D_Q에 비례 → softmax 포화 방지 |
| λ=1/(2σ_p²) | Prior 분산이 작을수록 정규화 강함 |
| CLT | 독립 원인의 합 → 가우시안 가정의 정당화 |

---

## 3. 핵심 수식 10개

| # | 이름 | 수식 |
|---|------|------|
| 1 | MSE 유도 | NLL(Gaussian) = $\frac{1}{2\sigma^2}\sum(y_i-h(x_i))^2$ |
| 2 | CE 유도 | NLL(Categorical) = $-\sum\log[h(x)]_{y_i}$ |
| 3 | MAP→L2 | $\min[\text{Loss} + \lambda\|\theta\|^2]$, $\lambda=1/(2\sigma_p^2)$ |
| 4 | Softmax | $p_i = \exp(z_i/\tau) / \sum_j\exp(z_j/\tau)$ |
| 5 | Softmax Jacobian | $\partial p/\partial z = \text{diag}(p) - pp^\top$ |
| 6 | Rank-Nullity | $n = \text{rank}(A) + \text{nullity}(A)$ |
| 7 | SVD | $A = U\Sigma V^\top = \sum\sigma_i u_i v_i^\top$ |
| 8 | Bayes | $P(H|E) = P(E|H)P(H)/P(E)$ |
| 9 | KL≥0 | Jensen on $-\log$ (볼록) |
| 10 | Normal Eq | $\hat{w} = (X^\top X)^{-1}X^\top y$ |

---

## 4. 헷갈리는 쌍 비교

| 비교 | A | B |
|------|---|---|
| MLE vs MAP | $\max \log P(D|\theta)$ | $\max [\log P(D|\theta) + \log P(\theta)]$ |
| CE vs MSE | Categorical 가정 | Gaussian 가정 |
| L1 vs L2 | Laplace Prior, sparsity | Gaussian Prior, shrinkage |
| Rank vs Nullity | 살아남는 차원 | 죽는 차원 |
| 고유값 vs 특이값 | 정방행렬만 | 임의 행렬 가능 |
| JVP vs VJP | Forward mode | Backward mode (역전파) |
| Entropy vs CE | 자기 분포의 불확실성 | 다른 분포로 코딩하는 비용 |
| Prior vs Posterior | 데이터 전 믿음 | 데이터 후 갱신된 믿음 |

---

## 5. 모델 비교 (시험 단골)

```
Inductive Bias 강도:  Linear > CNN > MLP > Transformer
데이터 적을 때 유리:  Linear > CNN > MLP > Transformer
데이터 많을 때 유리:  Transformer > MLP > CNN > Linear
표현력(Expressivity): Transformer > MLP > CNN > Linear
```

**CNN**: 이미지 특화 Prior (locality, translation equivariance)
**Transformer**: 약한 Prior, 대량 데이터 필요 (ViT: JFT-300M)

---

## 6. 함정 문제 대비

**Q**: "MSE를 쓰는 것은 어떤 가정인가?"
**A**: 노이즈가 가우시안이라는 가정. CLT에 의해 정당화.

**Q**: "Uniform Prior의 MAP는?"
**A**: MLE와 동일. $\log P(\theta) = \text{const}$이므로 사라짐.

**Q**: "KL(p||q) = 0이면?"
**A**: p = q (거의 확실하게). Gibbs' inequality의 등호 조건.

**Q**: "Softmax의 입력을 상수 c만큼 shift하면?"
**A**: 출력 불변. $\exp(z_i+c)/\sum\exp(z_j+c) = \exp(z_i)/\sum\exp(z_j)$. (수치안정성 트릭)

**Q**: "데이터가 무한히 많으면 MAP는?"
**A**: MLE에 수렴. Prior 항이 $O(1)$이고 Likelihood 항이 $O(n)$이므로.

**Q**: "대칭행렬의 고유벡터가 직교하는 이유?"
**A**: $(\lambda_1-\lambda_2)u_1^\top u_2 = 0$이고 $\lambda_1 \neq \lambda_2$이면 $u_1^\top u_2 = 0$.

---

## 7. 시험 전략

1. **유도 문제**: 먼저 결과를 적고, 역순으로 "왜" 체인을 채워라
2. **각 단계에 왜**: "i.i.d.이므로", "Fermat에 의해", "log는 단조증가이므로"
3. **시간 배분**: 유도 문제 (60%) → 개념 문제 (30%) → 검산 (10%)
4. **부분 점수**: 모르겠으면 아는 단계까지라도 적어라 (과정 점수)
5. **통합 관점**: "결국 이것은 NLL의 한 형태입니다"와 같은 연결을 보여줘라

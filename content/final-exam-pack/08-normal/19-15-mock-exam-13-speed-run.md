---
title: "딥러닝 이론 모의고사 #13 — 스피드런 (30분 단답형)"
slug: 15-mock-exam-13-speed-run
order: 19
---

# 딥러닝 이론 모의고사 #13 — 스피드런 (30분 단답형)

> 30분 / 50점 / 빠르게 핵심만 답하는 연습

---

## 1. [5점] i.i.d. 가정이 MLE 유도에 필요한 이유를 한 문장으로 쓰시오.
**답**: 결합확률을 개별확률의 곱으로 분해하기 위해 (곱이 되어야 로그 취했을 때 합이 됨).

## 2. [5점] 로그를 취하는 이유 3가지를 나열하시오.
**답**: (1) 곱→합 변환, (2) 수치 언더플로 방지, (3) 로그는 단조증가이므로 argmax 불변.

## 3. [5점] Fermat 정리를 한 문장으로 쓰시오.
**답**: 함수의 내부 극값에서는 반드시 미분값이 0이다 (미분=0은 극값의 필요조건).

## 4. [5점] MSE를 쓴다 = 어떤 확률 가정?
**답**: 노이즈가 가우시안 분포를 따른다는 가정.

## 5. [5점] CE를 쓴다 = 어떤 확률 가정?
**답**: 출력이 카테고리(Categorical) 분포를 따른다는 가정.

## 6. [5점] MAP = MLE + ?
**답**: MAP = MLE + log Prior = Loss + Regularization.

## 7. [5점] Gaussian Prior → ? Regularization
**답**: L2 Regularization (= Weight Decay = Ridge).

## 8. [5점] Laplace Prior → ? Regularization
**답**: L1 Regularization (= LASSO → Sparsity 유도).

## 9. [5점] $KL(p\|q) = CE(p,q) - ?$
**답**: $H(p)$ (엔트로피). $p$ 고정이면 CE 최소화 = KL 최소화.

## 10. [5점] Softmax Jacobian = ?
**답**: $\text{diag}(p) - pp^\top$

---

# 보너스 20문 (각 2.5점)

## 11. Rank-Nullity: $n = ?$
**답**: $\text{rank}(A) + \text{nullity}(A)$

## 12. 고유값 정의: $Av = ?$
**답**: $\lambda v$ ($v \neq 0$)

## 13. SVD: $A = ?$
**답**: $U\Sigma V^\top$

## 14. 정규방정식: $\hat{w} = ?$
**답**: $(X^\top X)^{-1}X^\top y$

## 15. 베이즈 정리: $P(H|E) = ?$
**답**: $P(E|H)P(H)/P(E)$

## 16. Newton's Method: $x_{n+1} = ?$
**답**: $x_n - f(x_n)/f'(x_n)$

## 17. Gradient = ? 방향
**답**: 가장 가파른 상승(steepest ascent) 방향

## 18. 대칭행렬의 고유벡터는 서로 ?
**답**: 직교(orthogonal)

## 19. PSD 행렬의 고유값은 모두 ?
**답**: 비음수 ($\geq 0$)

## 20. $\text{Tr}(AB) = \text{Tr}(?)$
**답**: $\text{Tr}(BA)$ (순환 성질)

## 21. VJP가 역전파에 적합한 이유?
**답**: Loss가 스칼라이므로 1회 backward로 모든 파라미터 그래디언트 계산 가능.

## 22. CLT = ?
**답**: i.i.d. 합의 표본평균 분포가 정규분포로 수렴.

## 23. $\sqrt{D_Q}$ 스케일링 이유?
**답**: 내적 분산이 $D_Q$에 비례 → softmax 포화 방지.

## 24. Temperature $\tau \to 0$이면 softmax → ?
**답**: argmax (one-hot 벡터)

## 25. Temperature $\tau \to \infty$이면 softmax → ?
**답**: Uniform 분포

## 26. Linear Model의 inductive bias는?
**답**: 높음 (선형 관계 가정)

## 27. Transformer의 inductive bias는?
**답**: 낮음 (약한 Prior → 대량 데이터 필요)

## 28. det(A) = 0이면 A는 ?
**답**: 비가역(singular). Rank < n.

## 29. $e_{y_i}^\top \log h(x_i)$의 역할?
**답**: 정답 클래스의 로그 확률만 선택 (인덱싱).

## 30. 볼록 함수에서 임계점 = ?
**답**: 전역 최소점 (Global minimum)

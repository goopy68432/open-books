---
title: "09. SVD와 PCA"
slug: 11-extra-topics-09-svd-pca
order: 77
---

# 09. SVD와 PCA

> 고유값 분해의 일반화. 차원 축소·압축의 수학적 기초.

---

## 1. SVD (Singular Value Decomposition)

임의의 행렬 $A \in \mathbb{R}^{m \times n}$에 대해:
$$A = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: 직교 ($U^T U = I$)
- $V \in \mathbb{R}^{n \times n}$: 직교
- $\Sigma \in \mathbb{R}^{m \times n}$: 대각, 비음수 (특이값 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)

### 직관

"임의 변환 = 회전 ($V^T$) + 확대/축소 ($\Sigma$) + 다른 회전 ($U$)"

---

## 2. 고유값 분해 vs SVD

| 분해 | 대상 | 식 |
|-----|-----|-----|
| 고유값 분해 | 정사각, 대각화 가능 | $A = P \Lambda P^{-1}$ |
| 스펙트럴 (대칭) | 정사각, 대칭 | $A = Q \Lambda Q^T$ (Q 직교) |
| **SVD** | **임의 행렬** | $A = U \Sigma V^T$ |

**SVD는 가장 일반적.**

---

## 3. SVD 계산

$A^T A$는 대칭 + 양반정치 → 스펙트럴 정리 적용:
$$A^T A = V \Lambda V^T$$

여기서 $\Lambda = \Sigma^T \Sigma$ (특이값의 제곱).

비슷하게 $A A^T = U \Lambda' U^T$.

특이값 $\sigma_i = \sqrt{\lambda_i}$.

---

## 4. PCA (Principal Component Analysis)

### 문제
n개 데이터 $\{x_1, \ldots, x_n\} \subseteq \mathbb{R}^d$. **분산을 최대로 보존**하는 저차원 사영을 찾고 싶음.

### 풀이

1. **중심화:** $\tilde{x}_i = x_i - \bar{x}$
2. **공분산 행렬:**
$$C = \frac{1}{n} \sum_i \tilde{x}_i \tilde{x}_i^T = \frac{1}{n} \tilde{X}^T \tilde{X}$$
3. **고유값 분해:** $C = V \Lambda V^T$ (대칭 → 스펙트럴)
4. **상위 k개 고유벡터** = 주성분 (principal components)
5. 사영: $z_i = V_k^T \tilde{x}_i$ (k차원)

### SVD 관점

$\tilde{X} = U \Sigma V^T$ (n×d 데이터 행렬의 SVD).

$C = (1/n) V \Sigma^2 V^T$로 같은 V 등장.

**주성분 = $\tilde{X}$의 우특이벡터 (V의 열).**

---

## 5. PCA의 정당성 (분산 최대화)

**정리:** k차원 사영 중 분산 보존 최대인 것은 공분산 행렬의 상위 k 고유벡터로 구성한 부분공간.

### 1차원 증명 스케치

$z = w^T \tilde{x}$, $\|w\| = 1$. 분산:
$$\text{Var}(z) = w^T C w$$

최대화: 라그랑주 $L = w^T C w - \lambda(w^T w - 1)$.
$\nabla L = 2Cw - 2\lambda w = 0 \Rightarrow Cw = \lambda w$ (고유값 방정식).

분산 = $w^T C w = w^T \lambda w = \lambda$ → 최대 고유값 = 최대 분산. ∎

---

## 6. 응용

### 데이터 압축
원본 $A$ (rank r) → 상위 k 특이값만 사용:
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T \quad (k < r)$$

→ Eckart-Young 정리: $A_k$가 rank-k 근사 중 최적 (Frobenius/Spectral norm).

### 신경망 가중치 압축
W를 SVD → 작은 특이값 잘라 파라미터 절감 (low-rank factorization).

### 노이즈 제거
작은 특이값 = 잡음 → 절단 (truncate)

---

## 7. SVD 핵심 식 정리

| 양 | 식 |
|----|-----|
| Frobenius norm | $\|A\|_F^2 = \sum \sigma_i^2$ |
| Spectral norm | $\|A\|_2 = \sigma_1$ (최대 특이값) |
| Pseudo-inverse | $A^+ = V \Sigma^+ U^T$ |
| Rank | $\text{rank}(A) = $ (0 아닌 특이값 수) |
| Range/Null | Range = span(U의 첫 r 열), Null = span(V의 마지막 (n-r) 열) |

---

## 8. 시험 답안 — PCA 유도 (1차원)

### [문제] 데이터 $\tilde{X}$의 1차원 PCA가 공분산 행렬의 최대 고유벡터임을 보여라.

### [풀이]

$z_i = w^T \tilde{x}_i$ (1차원 사영, $\|w\| = 1$).

샘플 분산:
$$\text{Var}(z) = \frac{1}{n}\sum_i (w^T \tilde{x}_i)^2 = w^T \left(\frac{1}{n}\sum \tilde{x}_i \tilde{x}_i^T\right) w = w^T C w$$

라그랑주: $L = w^T C w - \lambda(w^T w - 1)$.

$\nabla_w L = 2Cw - 2\lambda w = 0 \Rightarrow Cw = \lambda w$.

→ w는 C의 고유벡터, λ는 고유값.

$\text{Var}(z) = w^T C w = \lambda w^T w = \lambda$.

**분산 최대 ⇔ λ 최대 ⇔ w는 최대 고유값의 고유벡터.** ∎

---

## 9. 한 줄 요약

> "SVD: 임의 행렬 = 회전·확대·회전. PCA: 공분산 행렬의 최대 고유벡터로 분산 최대 보존 사영."

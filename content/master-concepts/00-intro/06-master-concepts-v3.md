---
title: "딥러닝 이론 마스터 컨셉 가이드 v3 — 작년 시험 영역 보강판"
slug: master-concepts-v3
order: 6
---

# 딥러닝 이론 마스터 컨셉 가이드 v3 — 작년 시험 영역 보강판

> **Version 3** — 작년 시험 100% 출제 정보 반영. v2 본문에 작년 출제 토픽 8개 영역을 깊이 보강.
>
> **v2 대비 변경점:**
> - **§3 Convex** 4유형 증명 추가
> - **§4 KL/Information Theory** 정규분포 KL 명시 계산
> - **§5 GD 수렴 조건** $\eta < 2/\lambda_{\max}$ 증명
> - **§7 Convolution** 출력 크기 + matrix 형태
> - **§8 Pooling** matrix 표현
> - **§9 Markov / DAG** (신규)
> - **§10 모델 비교 표** (신규)
> - **§11 PyTorch 흐름** (신규)
> - **§12 손실함수 통합 (MSE/CE/KL/NLL)** 강화

---

## 0. 강의의 메타 메시지 (v2와 동일, 요약)

1. **Deduction vs Induction** — AI 역사에서 인덕션 승, 시험은 deduction
2. **Tango (MLE + Prior)** — 강의 두 축
3. **답만 적으면 0점** — 채점 철학
4. **행렬은 마음의 고향** — 모든 것이 행렬

(자세한 인용은 `MASTER-CONCEPTS_v2.md` §0 참조)

---

## 1. 수학적 기초 — 핵심 요약

### 1.1 선형대수 (★★★★ — 작년 GD 수렴 조건과 연결)

**핵심 도구:**
- 내적, 행렬곱
- Range/Null/Rank-Nullity
- Eigenvalue/SVD/Spectral 정리
- **★ 직교 대각화** (GD 수렴 증명에서 사용)

### 1.2 미적분 + Linear Approximation

핵심:
- 체인 룰
- 가우스 적분 $\sqrt{2\pi}$
- Newton's Method (Linear Approx 본질)
- Jacobian (Softmax 자코비안)

### 1.3 확률·통계

- Bernoulli, Gauss, Uniform, Poisson 분포
- 베이즈 정리 + Bayesian vs Frequentist
- Chebyshev, Hoeffding 부등식
- 중심극한정리

---

## 2. 통계적 추정 (MLE + MAP) — v2와 동일

(자세한 풀이는 `EXAM-BANK_v3.md` Tier 2)

---

## 3. ★★★★★ Convex Function — 작년 1번 출제

### 3.1 정의
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \quad \forall x, y, \lambda \in [0,1]$$

### 3.2 4유형 증명 (★★★★★)

#### 유형 1: $f(x) = x^2$ convex

**정의로:**
$$f(\lambda x + (1-\lambda)y) - [\lambda f(x) + (1-\lambda)f(y)] = -\lambda(1-\lambda)(x-y)^2 \leq 0 \checkmark$$

**2차 미분:** $f'' = 2 > 0 \checkmark$

#### 유형 2: $f(x) = -\log x$ convex ($x > 0$)

**2차 미분:** $f'' = 1/x^2 > 0 \checkmark$

**Jensen 직접:** AM-GM 부등식 사용:
$$\lambda x_1 + (1-\lambda)x_2 \geq x_1^\lambda x_2^{1-\lambda}$$
$$-\log(\lambda x_1 + (1-\lambda)x_2) \leq -\lambda\log x_1 - (1-\lambda)\log x_2$$

#### 유형 3: 일반 함수 판별 (Hessian)

다변수 $f$ convex ⇔ Hessian $H \succeq 0$ (PSD).

**예시:** $f(x_1, x_2) = x_1^2 + x_2^2$:
$$H = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}, \quad \mathbf{v}^T H \mathbf{v} = 2(v_1^2 + v_2^2) \geq 0$$

#### 유형 4: Convex Set

$$\forall x, y \in C, \lambda \in [0,1]: \lambda x + (1-\lambda)y \in C$$

**예시:**
- ✅ $\mathbb{R}^n$, $\{x : \Vert x\Vert  \leq 1\}$ (ball), hyperplane
- ❌ 도넛, $\{x : \Vert x\Vert  = 1\}$ (sphere surface)

### 3.3 Convex 함수의 핵심 성질

| 성질 | 의미 |
|------|------|
| Jensen 부등식 | $f(E[X]) \leq E[f(X)]$ |
| 임계점 = 전역 최솟값 | 페르마 풀이가 보장됨 |
| Sublevel set convex | $\{x : f(x) \leq c\}$ convex |

### 3.4 자주 쓰이는 Convex 함수

| 함수 | Convex? | 비고 |
|------|--------|-----|
| $x^2$, $\Vert x\Vert ^p$ ($p \geq 1$) | ✅ |  |
| $e^x$ | ✅ |  |
| $-\log x$ ($x > 0$) | ✅ | KL 증명에 사용 |
| $\log x$ | ❌ (오목) |  |
| $\Vert x\Vert $ | ✅ (절댓값) |  |
| MSE 손실 | ✅ | Linear regression |
| BCE 손실 | ✅ (in p) | Logistic regression |
| 신경망 손실 | ❌ (일반적) | Non-convex |

---

## 4. ★★★★★ Information Theory + KL Divergence — 작년 2번, 4번 출제

### 4.1 Entropy
$$H(p) = -\sum_i p_i \log p_i = E_p[-\log p(X)]$$

### 4.2 Cross Entropy
$$H(p, q) = -\sum_i p_i \log q_i$$

### 4.3 KL Divergence
$$\text{KL}(p \Vert  q) = \sum_i p_i \log \frac{p_i}{q_i}$$

**핵심 분해:**
$$\boxed{H(p, q) = H(p) + \text{KL}(p \Vert  q)}$$

### 4.4 KL ≥ 0 증명 (Jensen 사용)

$\log$ 오목 → $E[\log Y] \leq \log E[Y]$.

$Y = q/p$, $X \sim p$:
$$-\text{KL}(p\Vert q) = E_p[\log(q/p)] \leq \log E_p[q/p] = \log 1 = 0$$
$$\therefore \text{KL}(p\Vert q) \geq 0 \quad \blacksquare$$

### 4.5 ★ 정규분포 KL 계산 (작년 직접 출제)

$P \sim N(\mu_1, 1)$, $Q \sim N(\mu_2, 1)$:

$$\log\frac{p(x)}{q(x)} = -\frac{(x-\mu_1)^2}{2} + \frac{(x-\mu_2)^2}{2} = (\mu_1 - \mu_2)x - \frac{\mu_1^2 - \mu_2^2}{2}$$

$$\text{KL}(P\Vert Q) = E_p[\log(p/q)] = (\mu_1-\mu_2)\mu_1 - \frac{\mu_1^2-\mu_2^2}{2} = \frac{1}{2}(\mu_1 - \mu_2)^2$$

$$\boxed{\text{KL}(P \Vert  Q) = \frac{1}{2}(\mu_1 - \mu_2)^2}$$

### 4.6 일반 정규분포 KL

$P \sim N(\mu_1, \sigma_1^2), Q \sim N(\mu_2, \sigma_2^2)$:
$$\text{KL}(P\Vert Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 4.7 KL과 손실함수 (★★★★ 작년)

$$\text{NLL} = \text{CE} = H(p) + \text{KL}(p \Vert q)$$

p 고정 → CE 최소화 ⇔ KL 최소화 ⇔ NLL 최소화.

---

## 5. ★★★★★ Bias-Variance + GD 수렴 — 작년 3번, 4번 출제

### 5.1 Bias-Variance 분해
$$E[(y_0 - \hat{f})^2] = (\bar{f} - f)^2 + E[(\bar{f} - \hat{f})^2] + \sigma^2$$

(증명은 `EXAM-BANK_v3.md` Tier 1-3 참조)

### 5.2 ★ Gradient Descent 수렴 조건 — $\eta < 2/\lambda_{\max}(A)$

**Quadratic loss:** $L(\theta) = \frac{1}{2}\theta^T A \theta - b^T\theta$, $A$ symmetric PSD.

**오차 재귀:** $e_{t+1} = (I - \eta A)e_t$.

**고유분해:** $A = Q\Lambda Q^T$, $\tilde{e} = Q^T e$:
$$L(\theta_{t+1}) - L^* = \frac{1}{2}\sum_i \lambda_i (1-\eta\lambda_i)^2 \tilde{e}_{t,i}^2$$

**감소 조건:** $|1 - \eta\lambda_i| < 1$ for all i:
$$0 < \eta < \frac{2}{\lambda_{\max}}$$

(증명은 `EXAM-BANK_v3.md` Tier 1-4)

### 5.3 일반 최적화 도구

- **GD**: $\theta_{t+1} = \theta_t - \eta\nabla L(\theta_t)$
- **SGD**: 미니배치 그래디언트
- **Momentum**: $v_t = \beta v_{t-1} + \nabla L$
- **Adam**: 적응적 학습률

---

## 6. 신경망 기초 — 활성화 + Backprop + Softmax

(v2와 동일, 자세한 건 `MASTER-CONCEPTS_v2.md` §5)

### 6.1 Softmax 자코비안

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j), \quad J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### 6.2 Backpropagation 4단계 식

1. $\delta^{(L)} = \nabla L \odot \sigma'(z^{(L)})$
2. $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$
3. $\partial L/\partial W^{(l)} = \delta^{(l)}(a^{(l-1)})^T$
4. $\partial L/\partial b^{(l)} = \delta^{(l)}$

### 6.3 ★ 행렬 미분 — $-\log\sigma(Ax+b)$ (작년 영역)

$z = Ax + b$, $p = \sigma(z)$, $L = -\log p$:
$$\frac{\partial L}{\partial z} = p - 1 \quad (\text{단순 형태})$$

연쇄:
- $\partial L/\partial A = (p-1) x^T$
- $\partial L/\partial x = A^T(p-1)$
- $\partial L/\partial b = (p-1)$

---

## 7. ★★★★★ CNN — Convolution + Pooling (작년 5번 + 7번 출제)

### 7.1 Convolution 정의

**1D, stride 1, padding 0:**
$$y_i = \sum_{j=1}^k w_j x_{i+j-1}, \quad i = 1, \ldots, n-k+1$$

### 7.2 ★ 출력 크기 공식 (시험 직접 출제)

$$\boxed{H_{\text{out}} = \frac{H_{\text{in}} - k + 2p}{s} + 1}$$

- $H_{\text{in}}$: 입력
- $k$: 커널
- $p$: padding
- $s$: stride

**예: $H_{\text{in}}=7, k=3, p=1, s=2$ → $H_{\text{out}}=4$**

### 7.3 Convolution Matrix (Toeplitz)

$\mathbf{x} = (x_1, x_2, x_3, x_4)$, $\mathbf{w} = (w_1, w_2, w_3)$:
$$\mathbf{y} = \begin{pmatrix} w_1 & w_2 & w_3 & 0 \\ 0 & w_1 & w_2 & w_3 \end{pmatrix} \mathbf{x}$$

→ Toeplitz matrix.

### 7.4 ★ Average Pooling Matrix (작년 출제)

kernel=2, stride=2:
$$P = \frac{1}{2}\begin{pmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 \end{pmatrix}$$

**Max Pooling은 비선형 → 매트릭스 X**.

### 7.5 Inductive Bias

- Translation Equivariance
- Parameter Sharing
- Local Receptive Field

---

## 8. ★★★★ RNN / LSTM (변형 출제)

### 8.1 RNN 기본
$$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

### 8.2 LSTM 게이트
$$f_t, i_t, o_t \text{ (gates)}, \quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \quad h_t = o_t \odot \tanh(c_t)$$

### 8.3 BPTT
시간 펼침 + Backprop.

---

## 9. ★★★★★ Markov Chain / DAG (작년 6번 출제, 신규)

### 9.1 Markov Property
$$P(X_{t+1} | X_1, \ldots, X_t) = P(X_{t+1} | X_t)$$

### 9.2 Joint Probability
$$P(X_1, \ldots, X_T) = P(X_1)\prod_{t=2}^T P(X_t | X_{t-1})$$

### 9.3 DAG (Bayesian Network)
$$P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))$$

**예시 DAG:** $A \to B, A \to C, B \to D, C \to D$:
$$P(A, B, C, D) = P(A)P(B|A)P(C|A)P(D|B,C)$$

### 9.4 조건부확률 3가지 유형
- 직접 transition: $P(X_{t+1} | X_t)$
- Backward (Bayes 사용): $P(X_1 | X_T)$
- 임의 시점: Transition matrix powers

---

## 10. ★★★★★ 모델 비교 — Markov vs RNN vs Transformer (작년 7번, 신규)

| 측면 | Markov | RNN | Transformer |
|------|--------|-----|-------------|
| **Inductive Bias** | 매우 강함 | 강함 | 약함 |
| **State** | 마지막만 | hidden 압축 | 모든 토큰 |
| **의존성** | 1차 | 무한 (이론) | 모든 위치 |
| **데이터** | 적음 | 중간 | 매우 많음 |
| **병렬화** | - | 어려움 | 쉬움 |
| **장기 의존성** | 약함 | Vanishing | 강함 |

### 10.1 Inductive Bias의 의미

> Inductive Bias 강함 → 적은 데이터로 일반화
> 약함 → 많은 데이터 필요

### 10.2 Transformer의 핵심 — Self-Attention
$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$$

→ 모든 위치 직접 연결, 병렬 가능.

---

## 11. ★★★ PyTorch 코드 흐름 (작년 8번, 신규)

### 11.1 표준 학습 루프 5단계
```python
for input, target in dataset:
    optimizer.zero_grad()        # 1. Gradient 초기화
    output = model(input)         # 2. Forward
    loss = loss_fn(output, target)  # 3. Loss 계산
    loss.backward()               # 4. Backward (autograd)
    optimizer.step()              # 5. Parameter 업데이트
```

### 11.2 각 단계 설명

| # | 함수 | 역할 |
|---|------|------|
| 1 | `optimizer.zero_grad()` | 누적 grad 제거 |
| 2 | `model(input)` | forward pass (예측) |
| 3 | `loss_fn(...)` | loss 스칼라 계산 |
| 4 | `loss.backward()` | 자동 미분 → 모든 grad 계산 |
| 5 | `optimizer.step()` | $\theta \leftarrow \theta - \eta \nabla L$ |

---

## 12. ★★★★★ 손실함수 통합 (MSE / NLL / CE / KL) — 작년 출제

### 12.1 통합 흐름도

```
            분포 가정
               ↓
         Likelihood (i.i.d → 곱)
               ↓ -log
              NLL
       ────┼──────────┼────
       ↓        ↓
  [Gauss noise]   [Categorical]
       ↓        ↓
      MSE       CE = H(p) + KL(p||q)
                     ↓
              CE 최소화 = KL 최소화 (p 고정)
```

### 12.2 핵심 등가성

- **MSE = NLL** under Gauss noise
- **CE = NLL** for Categorical
- **CE = H(p) + KL** (분해)
- **CE 최소화 = KL 최소화** (p 고정 시)

### 12.3 시험 답안 표준 한 줄

> "MSE/CE/NLL/KL은 분포 가정 하에서 모두 동치. 가우스 가정 → MSE, 카테고리컬 → CE = H(p)+KL이며, 진짜 분포 p 고정 시 CE 최소화 = KL 최소화."

---

## 13. ★★★ 생성 모델 — VAE / Diffusion NLL 유도

### 13.1 VAE ELBO

$$\log p(x) = \log \int p(x|z)p(z)dz$$

Jensen + 변분분포 $q(z|x)$:
$$\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\Vert p(z)) = \text{ELBO}$$

**해석:**
- 첫 항: 재구성 (likelihood)
- 둘째 항: KL 정규화 (latent를 prior에 가깝게)

### 13.2 Diffusion Model

**Forward:** $q(x_t | x_{t-1}) = N(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$
**Reverse 학습:** $p_\theta(x_{t-1} | x_t) = N(\mu_\theta, \Sigma_\theta)$

**Loss (간단화):**
$$L = E[\Vert \epsilon - \epsilon_\theta(x_t, t)\Vert ^2]$$

(noise 예측 신경망)

---

## 14. ★ 약어·기호 사전

(v2와 동일, 생략)

---

## 15. v3 통합 메시지

### 작년 100% 출제 영역 (Tier 1)
1. Convex 4유형
2. KL Divergence (정규분포 계산)
3. Bias-Variance 분해
4. GD 수렴 $\eta < 2/\lambda_{\max}$
5. Convolution + 출력 크기 + matrix
6. Markov Chain / DAG
7. 모델 비교 (Markov/RNN/Transformer)
8. PyTorch 5단계
9. Pooling Matrix
10. MSE/CE/NLL/KL 통합 관계

### 올해 기출 (Tier 2)
8문제 7단계 체인 (final-fire/ 참조)

### 변형 대비 (Tier 3)
다양한 분포, 행렬 미분, VAE/Diffusion

---

## 부록: 자료 매핑

| 본 가이드 § | EXAM-BANK_v3 | final-fire |
|----------|-------------|----------|
| §3 Convex | T1-1 | `11-extra-topics/06` |
| §4 KL | T1-2 | `10-ten-proofs/07` |
| §5.1 BV | T1-3 | `11-extra-topics/03` |
| §5.2 GD | T1-4 | (신규) |
| §7 CNN | T1-5, T1-9 | (신규) |
| §9 Markov | T1-6 | (신규) |
| §10 모델 | T1-7 | (신규) |
| §11 PyTorch | T1-8 | (신규) |
| §12 손실 통합 | T1-10 | `09-killer-chains/` |

---

**작성:** 2026-04-26 (v3)
**기반:** 작년 시험 5장 이미지 + 강의 스크립트 + v2 본문
**연관 자료:** `EXAM-BANK_v3.md` (메인 풀이집), `STRATEGY_v3.md` (전략)

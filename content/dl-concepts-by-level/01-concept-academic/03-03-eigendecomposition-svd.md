---
title: "제3장: 고유값 분해와 특이값 분해(SVD)"
slug: 03-eigendecomposition-svd
order: 3
---

# 제3장: 고유값 분해와 특이값 분해(SVD)

> **선수 과목**: 2장 행렬과 선형 공간 (rank, 치역, 영공간)
> **후속 연결**: 4장 노름/대각합 (spectral norm, Frobenius norm), 13장 정규화

---

## 1. 동기부여 및 개요

고유값 분해와 SVD는 행렬의 **내부 구조를 해부하는 도구**이다. PCA, 추천 시스템, 이미지 압축, Google PageRank, 그리고 딥러닝의 LoRA, Spectral Normalization 등의 수학적 기반이 된다.

핵심 아이디어는 단순하다: **복잡한 변환을 "회전 → 스케일링 → 회전"으로 분해**하는 것이다.

```
가역 행렬 (TFAE) ──> 고유값/고유벡터 ──> 고유값 분해 A = UΛU^T
     │                                        │
직교행렬, PSD/PD ────────────────────────> SVD: A = UΣV^T
                                                │
                                    저랭크 근사, Power Method, PageRank
```

---

## 2. 가역 행렬의 동치 조건 (TFAE)

**Theorem 2.1 (Invertible Matrix Theorem).** 정사각 행렬 $A \in \mathbb{R}^{n \times n}$에 대해 다음은 모두 동치이다:

1. $A$가 가역이다 ($\exists A^{-1}$)
2. $\text{rank}(A) = n$ (full rank)
3. $\ker(A) = \{\mathbf{0}\}$
4. $Ax = b$가 유일한 해를 가진다
5. $A$의 모든 고유값이 $0$이 아니다
6. $A$의 열들이 선형독립이다
7. $\det(A) \neq 0$

**직관**: 행렬식이 0이 아니라는 것은 행렬이 공간을 "찌그러뜨려서 차원을 줄이지 않는다"는 뜻이다.

역행렬 계산의 시간복잡도는 $O(n^3)$이다 (LU 분해 기반). 실무에서는 역행렬을 명시적으로 구하지 않고 $Ax = b$를 직접 푼다 (`np.linalg.solve`).

> **주의**: $\det(A) \neq 0$이어도 매우 작은 고유값이 있으면 수치적으로 불안정하다(ill-conditioned). 조건수 $\kappa(A) = |\lambda_{\max}/\lambda_{\min}|$이 클수록 위험하다.

---

## 3. 고유값과 고유벡터

### 3.1 정의

**Definition 3.1.** 정사각 행렬 $A \in \mathbb{R}^{n \times n}$에 대해, $Av = \lambda v$를 만족하는 영이 아닌 벡터 $v$를 **고유벡터(eigenvector)**, 스칼라 $\lambda$를 **고유값(eigenvalue)**이라 한다. 보통 $\|v\| = 1$로 정규화한다.

### 3.2 특성방정식

$$Av = \lambda v \iff (A - \lambda I)v = \mathbf{0}$$

$v \neq \mathbf{0}$인 해가 존재하려면:

$$\det(A - \lambda I) = 0 \quad \text{(특성방정식)}$$

**예시**: $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$이면 $\det \begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0$이므로 $\lambda_1 = 3, \lambda_2 = 2$.

### 3.3 기하학적 의미

행렬 $A$는 벡터를 고유벡터 방향으로 $\lambda$배만큼 **늘이거나 줄이는** 변환이다. 고유값이 음수이면 방향이 뒤집힌다.

### 3.4 딥러닝에서의 의미

Hessian 행렬의 고유값 스펙트럼은 손실 함수의 곡률을 나타낸다:
- 가장 큰 고유값 $\lambda_{\max}$가 학습률 상한을 결정: $\eta < 2/\lambda_{\max}$
- 이것이 **Edge of Stability** 현상의 핵심 (Cohen et al., 2021)

```python
import numpy as np
A = np.array([[3, 1], [0, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)
# 검증: Av = λv
for i in range(len(eigenvalues)):
    v, lam = eigenvectors[:, i], eigenvalues[i]
    assert np.allclose(A @ v, lam * v)
```

> **주의**: 고유벡터는 유일하지 않다. 상수를 곱해도 여전히 고유벡터이며, 부호 모호성(sign ambiguity)이 있다.

---

## 4. 직교행렬, PSD, PD

### 4.1 직교행렬

**Definition 4.1 (직교행렬).** $U \in \mathbb{R}^{n \times n}$이 직교행렬(orthogonal matrix)이라 함은 $U^\top U = UU^\top = I$이다.

**핵심 성질**: 길이 보존 $\|Ux\| = \|x\|$. 기하학적으로 회전(rotation) 또는 반사(reflection)에 해당한다.

### 4.2 양의 준정부호/정부호 행렬

**Definition 4.2.** 대칭 행렬 $A$에 대해:
- **PSD (양의 준정부호)**: $v^\top Av \geq 0, \; \forall v \neq \mathbf{0}$. 표기: $A \succeq 0$. 모든 고유값 $\geq 0$.
- **PD (양의 정부호)**: $v^\top Av > 0, \; \forall v \neq \mathbf{0}$. 표기: $A \succ 0$. 모든 고유값 $> 0$.

**Proposition 4.1.** $BB^\top$와 $B^\top B$ (Gram matrix)는 항상 PSD이다.

*증명.* $v^\top (BB^\top) v = (B^\top v)^\top (B^\top v) = \|B^\top v\|^2 \geq 0$. $\square$

**Proposition 4.2.** $BB^\top + \lambda I$ ($\lambda > 0$)는 항상 PD이다.

이 성질은 **Ridge regression**에서 $(X^\top X + \lambda I)$의 역행렬이 항상 존재함을 보장하며, 정규화의 수학적 근거이다.

### 4.3 이차형식과 볼록성

이차형식 $f(x) = x^\top Ax$에서:
- $A \succ 0$ → $f$는 강볼록(strictly convex), 유일한 최솟값 존재
- $A \succeq 0$ → $f$는 볼록, 최솟값이 유일하지 않을 수 있음
- 음의 고유값 존재 → 안장점(saddle point) 존재

> **주의**: PSD/PD는 **대칭 행렬**에 대해서만 정의된다. 비대칭 행렬 $A$의 이차형식은 $v^\top \frac{A+A^\top}{2} v$로 대칭 부분만 기여한다.

---

## 5. 고유값 분해 (Spectral Decomposition)

### 5.1 정리

**Theorem 5.1 (Spectral Theorem).** 실수 대칭 행렬 $A \in \mathbb{R}^{n \times n}$은 항상 다음과 같이 분해된다:

$$A = U\Lambda U^\top = \sum_{i=1}^{n} \lambda_i u_i u_i^\top$$

여기서 $U = [u_1 \; \cdots \; u_n]$은 직교행렬 ($U^\top U = I$), $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

### 5.2 기하학적 해석

고유값 분해는 세 단계의 변환이다:
1. $U^\top$: 원래 좌표를 고유벡터 좌표로 회전
2. $\Lambda$: 각 고유벡터 방향으로 스케일링
3. $U$: 다시 원래 좌표로 역회전

### 5.3 이차형식의 단순화

$$f(x) = x^\top Ax = x^\top U\Lambda U^\top x = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2$$

여기서 $y = U^\top x$. 좌표 변환을 통해 교차항이 사라지고 각 변수가 독립적으로 분리된다.

등고선 $f(x) = c$는 타원(또는 쌍곡선)이며, 타원의 축은 고유벡터 방향이고 축의 길이는 $1/\sqrt{\lambda_i}$에 비례한다.

### 5.4 고유값의 성질

| 행렬 연산 | 고유값 변화 | 고유벡터 |
|-----------|-----------|---------|
| $cA$ | $c\lambda$ | 동일 |
| $A^2$ | $\lambda^2$ | 동일 |
| $A^{-1}$ | $1/\lambda$ | 동일 |
| $A + cI$ | $\lambda + c$ | 동일 |
| $A^\top$ | 동일 | 일반적으로 다름 (대칭이면 동일) |
| $AB$ | 일반적으로 예측 불가 | -- |

> **딥러닝 연결**: Weight decay ($+\lambda I$)는 고유값을 $\lambda_i + \lambda$로 이동시켜 작은 고유값을 키우고, 조건수 $\kappa = \lambda_{\max}/\lambda_{\min}$을 줄인다.

```python
import numpy as np
A = np.array([[5, 1], [1, 3]])  # 대칭 행렬
eigenvalues, U = np.linalg.eigh(A)  # eigh: 대칭 전용 (더 안정적)
Lambda = np.diag(eigenvalues)

# 분해 검증
assert np.allclose(A, U @ Lambda @ U.T)
assert np.allclose(U.T @ U, np.eye(2))  # 직교성

# 외적 합 형태
A_sum = sum(eigenvalues[i] * np.outer(U[:, i], U[:, i]) for i in range(2))
assert np.allclose(A, A_sum)
```

> **주의**: 모든 정사각 행렬이 고유값 분해 가능한 것은 아니다. $n$개의 선형독립 고유벡터가 필요하다. 대칭 행렬은 Spectral Theorem에 의해 항상 가능하지만, 일반 행렬은 불가능할 수 있다 (defective matrix). 일반 행렬에는 SVD를 사용한다.

---

## 6. 특이값 분해 (SVD)

### 6.1 정리

**Theorem 6.1 (SVD).** **임의의** 행렬 $A \in \mathbb{R}^{m \times n}$은 다음과 같이 분해된다:

$$A = U\Sigma V^\top = \sum_{i=1}^{\min(m,n)} \sigma_i u_i v_i^\top$$

여기서:
- $U \in \mathbb{R}^{m \times m}$: 직교행렬 (좌특이벡터)
- $V \in \mathbb{R}^{n \times n}$: 직교행렬 (우특이벡터)
- $\Sigma \in \mathbb{R}^{m \times n}$: 대각 원소 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ (특이값)
- $Av_j = \sigma_j u_j$: 우특이벡터가 좌특이벡터로 매핑됨

### 6.2 고유값 분해와의 관계

| 비교 항목 | 고유값 분해 | SVD |
|-----------|-----------|-----|
| 적용 대상 | 대칭 정사각 행렬 | **임의의** $m \times n$ 행렬 |
| 형태 | $A = U\Lambda U^\top$ | $A = U\Sigma V^\top$ |
| 좌/우 행렬 | 같음 ($U = U$) | 다름 ($U \neq V$) |
| 값 | 고유값 (음수 가능) | 특이값 (항상 $\geq 0$) |

**관계**: $A^\top A = V\Sigma^\top \Sigma V^\top$ → $V$의 열은 $A^\top A$의 고유벡터, $\sigma_i^2$이 고유값.

### 6.3 기하학적 해석

SVD는 모든 선형 변환을 세 단계로 분해한다:
1. $V^\top$: 입력 공간에서 회전
2. $\Sigma$: 각 축 방향으로 스케일링 (차원 변환 포함)
3. $U$: 출력 공간에서 회전

```python
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2

U, sigma, Vt = np.linalg.svd(A, full_matrices=True)

# 복원 검증
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, sigma)
assert np.allclose(A, U @ Sigma @ Vt)

# 외적 합 형태
A_sum = sum(sigma[i] * np.outer(U[:, i], Vt[i, :]) for i in range(len(sigma)))
assert np.allclose(A, A_sum)
```

---

## 7. 저랭크 근사

### 7.1 Eckart-Young-Mirsky 정리

**Theorem 7.1.** $A = \sum_{i=1}^r \sigma_i u_i v_i^\top$ (rank $r$)의 최적 rank-$k$ 근사 ($k < r$)는:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top$$

근사 오차: $\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$

이것이 Frobenius norm 기준으로 rank $k$ 이하의 모든 행렬 중 $A$에 가장 가까운 행렬이다.

### 7.2 압축률 분석

원본 $m \times n$ 행렬: $mn$개의 숫자 저장.
Rank-$k$ 근사: $k(m + n + 1)$개의 숫자 저장 ($k$개의 $u_i, v_i, \sigma_i$).

특이값이 급격히 감소하는 경우 소수의 $k$로도 원본에 가까운 근사가 가능하다.

### 7.3 딥러닝에서의 저랭크 근사

| 응용 | 설명 |
|------|------|
| **LoRA** (Hu et al., 2021) | $W_{\text{new}} = W_0 + BA$ ($B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$) |
| **모델 압축** | 학습된 가중치의 SVD → 상위 $k$개 특이값만 유지 |
| **Spectral Normalization** (Miyato et al., 2018) | $W / \sigma_1(W)$로 최대 특이값 정규화 → GAN 안정화 |

> **주의**: 특이값이 천천히 감소하면(노이즈가 많은 데이터) 저랭크 근사의 품질이 나쁘다. 또한 SVD 자체의 계산 비용이 $O(\min(m,n) \cdot mn)$이므로 대규모 행렬에서는 randomized SVD를 사용한다.

---

## 8. Power Method와 PageRank

### 8.1 Power Method

행렬 $A$에 벡터 $u$를 반복적으로 곱하면, 결과가 **최대 고유값의 고유벡터** 방향으로 수렴한다:

$$u_{t+1} = \frac{Au_t}{\|Au_t\|}$$

수렴 속도는 spectral gap $\gamma = (\lambda_1 - \lambda_2)/\lambda_1$에 의해 결정된다:

$$\text{error} \leq O\left(\left(\frac{\lambda_2}{\lambda_1}\right)^t\right)$$

### 8.2 PageRank (Brin & Page, 1998)

Google 행렬 $G = \alpha M + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^\top$에서 ($\alpha \approx 0.85$):

$$\pi_{t+1} = G\pi_t$$

PageRank 벡터 $\pi$는 $G$의 **고유값 1에 대응하는 고유벡터**(= 정상 분포)이며, Power Method로 계산한다.

Damping factor $\alpha < 1$은 $G$를 원시(primitive) 행렬로 만들어 Perron-Frobenius 정리의 조건을 보장하고, 유일한 정상 분포의 존재를 보장한다.

```python
import numpy as np

# Power Method: 최대 고유값
def power_method(A, num_iter=100):
    n = A.shape[0]
    u = np.random.randn(n)
    u /= np.linalg.norm(u)
    for _ in range(num_iter):
        u = A @ u
        u /= np.linalg.norm(u)
    return u @ A @ u, u  # (고유값, 고유벡터)

# PageRank
P = np.array([[0, 0, 1, 0.5], [1/3, 0, 0, 0],
              [1/3, 0.5, 0, 0.5], [1/3, 0.5, 0, 0]])
alpha = 0.85
n = P.shape[0]
G = alpha * P + (1 - alpha) / n * np.ones((n, n))

pi = np.ones(n) / n
for _ in range(100):
    pi = G @ pi
print(f"PageRank: {pi}, 합: {pi.sum():.4f}")
```

### 8.3 Spectral Normalization과의 연결

Spectral Normalization (Miyato et al., 2018)은 Power Iteration을 1-2회만 수행하여 최대 특이값을 근사적으로 구하고, GAN 판별자의 가중치를 $W/\sigma_1(W)$로 정규화한다. 이는 Lipschitz 조건을 강제하여 학습을 안정화한다.

---

## 9. 딥러닝 적용 요약

| 수학 개념 | 딥러닝 응용 | 중요도 |
|-----------|------------|--------|
| 고유값/고유벡터 $Av = \lambda v$ | Hessian 분석, PCA, 학습률 결정 | 핵심 |
| PSD/PD | 볼록 최적화, 정규화, Gram matrix | 핵심 |
| Spectral 분해 $A = U\Lambda U^\top$ | PCA, loss landscape 분석 | 핵심 |
| SVD $A = U\Sigma V^\top$ | LoRA, 모델 압축, Spectral Norm | 핵심 |
| 저랭크 근사 | LoRA, 추천 시스템 | 핵심 |
| 이차형식 기하학 | 조건수, Adam의 적응적 학습률 | 중요 |
| Power Method | Spectral Norm 계산, PageRank, GNN | 중요 |
| 가역 조건 (TFAE) | 학습 안정성, 수치적 조건 | 기초 |

---

## 10. 흔한 오해와 주의점

1. **"역행렬이 존재하면 수치적으로 안전하다"** → 조건수가 크면 불안정. $\kappa(A) = |\lambda_{\max}/\lambda_{\min}|$.
2. **"고유벡터는 유일하다"** → 상수배, 부호 모호성이 있고, 중복 고유값에서는 고유공간 전체가 고유벡터.
3. **"모든 정사각 행렬이 고유값 분해 가능하다"** → 대칭 행렬만 보장. 일반 행렬은 SVD를 사용.
4. **"SVD는 정사각 행렬에만 적용된다"** → SVD의 핵심 강점이 **임의의** $m \times n$ 행렬에 적용 가능하다는 것.
5. **"PSD는 대칭이 아니어도 된다"** → PSD/PD는 대칭 행렬에 대해서만 정의됨.
6. **"타원 등고선의 넓은 방향이 중요하다"** → 넓은 방향은 고유값이 **작은** 방향. 좁은 방향(큰 고유값)이 최적화에서 더 민감.
7. **"Power Method는 모든 고유값을 구한다"** → 최대 고유값만 직접 계산. 나머지는 deflation 필요.
8. **"SVD 압축은 항상 효과적이다"** → 특이값의 감소 속도(spectral decay)에 의존.

---

## 11. 핵심 요약

1. **고유값 분해**: 대칭 행렬을 "회전 → 스케일 → 역회전"으로 분해. $A = U\Lambda U^\top$.
2. **SVD**: **모든** 행렬을 "회전 → 스케일(+차원변환) → 회전"으로 분해. $A = U\Sigma V^\top$.
3. **저랭크 근사**: 큰 특이값 몇 개만으로 원래 행렬을 잘 근사. LoRA, 이미지 압축의 기반.
4. **PD/PSD**: 고유값의 부호가 이차형식의 모양(볼록/안장점)을 결정. 최적화 지형의 열쇠.
5. **Power Method**: 반복 곱셈으로 최대 고유벡터를 찾음. Spectral Normalization의 계산 방법.
6. **조건수**: $\kappa = \lambda_{\max}/\lambda_{\min}$. 클수록 최적화가 어렵고, 정규화가 이를 줄인다.

> **한 문장 정리**: 행렬의 고유값/특이값은 그 행렬이 각 방향으로 벡터를 얼마나 늘이고 줄이는지를 나타내며, 이 스펙트럼이 딥러닝의 학습 동역학, 모델 압축, 안정성의 수학적 핵심이다.

---

## 참고문헌

- Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press. Chapter 2.
- Strang, G. (2019). *Linear Algebra and Its Applications*, 5th ed.
- Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Miyato, T. et al. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.
- Brin, S. & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine.
- Cohen, J. et al. (2021). Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability. *ICLR*.

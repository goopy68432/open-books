---
title: "제4장: 노름, 대각합, 연립방정식"
slug: 04-norms-trace-linear-equations
order: 4
---

# 제4장: 노름, 대각합, 연립방정식

> **선수 과목**: 2장 행렬과 선형 공간, 3장 고유값 분해와 SVD
> **후속 연결**: 5장 미적분과 행렬 미적분, 13장 정규화

---

## 1. 동기부여 및 개요

벡터와 행렬의 **"크기"를 재는 방법**을 알면, 딥러닝의 손실 함수(MSE, MAE)부터 정규화(L1, L2, Spectral Norm)까지 모든 것이 하나의 수학적 틀로 연결된다.

본 장에서는 노름(크기 측정), 대각합(구조적 요약), 연립방정식 풀이(유사역행렬), 그리고 선형 사영(최소제곱법)을 다룬다.

```
벡터 노름 (L1, L2, Linf) ──> 행렬 노름 (Nuclear, Frobenius, Spectral)
                                         │
대각합 (Trace) ──> Hutchinson 추정        │
                                         │
연립방정식 Ax=b ──> 유사역행렬 A+ ──> 선형 사영/최소제곱법
```

---

## 2. 벡터 노름

### 2.1 $L_p$ 노름

**Definition 2.1.** 벡터 $v \in \mathbb{R}^n$의 $L_p$ 노름 ($p \geq 1$):

$$\|v\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}$$

| 노름 | 정의 | 직관 | 딥러닝 응용 |
|------|------|------|------------|
| $\|v\|_1$ | $\sum_i |v_i|$ | 맨해튼 거리 | L1 정규화 (희소성 유도), MAE 손실 |
| $\|v\|_2$ | $\sqrt{\sum_i v_i^2}$ | 유클리드 거리 | L2 정규화 (weight decay), MSE 손실 |
| $\|v\|_\infty$ | $\max_i |v_i|$ | 최대 성분 | Gradient clipping, $L_\infty$ 공격 |
| $\|v\|_0$ | $\sum_i \mathbf{1}(v_i \neq 0)$ | 비영 원소 개수 | 희소성 측정 (엄밀히 노름 아님) |

> **주의**: $L_0$ "노름"은 삼각부등식을 만족하지 않아 수학적으로 노름이 아니다. 관례적으로만 "노름"이라 부른다.

### 2.2 노름의 공리

함수 $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}_{\geq 0}$이 노름이 되려면:
1. **양정치성**: $\|v\| \geq 0$, 등호 $\iff v = \mathbf{0}$
2. **동차성**: $\|cv\| = |c| \cdot \|v\|$
3. **삼각부등식**: $\|v + w\| \leq \|v\| + \|w\|$

```python
import numpy as np
v = np.array([3, 4])
print(f"L1: {np.linalg.norm(v, 1)}")      # 7.0
print(f"L2: {np.linalg.norm(v, 2)}")      # 5.0
print(f"Linf: {np.linalg.norm(v, np.inf)}") # 4.0
```

---

## 3. 행렬 노름

### 3.1 세 가지 핵심 행렬 노름

$A$의 특이값을 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$이라 하면:

| 행렬 노름 | 정의 | 특이값 표현 | 딥러닝 응용 |
|-----------|------|-----------|------------|
| **Nuclear (trace) norm** $\|A\|_*$ | $\sum_i \sigma_i$ | $\|\sigma\|_1$ | 저랭크 행렬 완성, 추천 시스템 |
| **Frobenius norm** $\|A\|_F$ | $\sqrt{\sum_{i,j} a_{ij}^2}$ | $\|\sigma\|_2$ | Weight decay: $\lambda\|W\|_F^2$ |
| **Spectral (operator) norm** $\|A\|_\sigma$ | $\sup_{x \neq 0} \frac{\|Ax\|}{\|x\|}$ | $\sigma_{\max} = \|\sigma\|_\infty$ | Spectral Normalization (GAN) |

### 3.2 노름 간 부등식

$$\|A\|_\sigma \leq \|A\|_F \leq \|A\|_*$$

이 부등식은 항상 성립하며, 특이값 벡터의 $L_\infty \leq L_2 \leq L_1$ 부등식에 대응한다.

### 3.3 딥러닝에서의 정규화와 노름

| 정규화 기법 | 수식 | 효과 |
|------------|------|------|
| L2 정규화 (weight decay) | $\lambda\|W\|_F^2$ | 가중치를 골고루 작게 유지 |
| L1 정규화 | $\lambda\|W\|_1$ | 희소성(sparsity) 유도 |
| Nuclear norm 정규화 | $\lambda\|W\|_*$ | 저랭크 구조 유도 |
| Spectral Normalization | $W / \|W\|_\sigma$ | Lipschitz 제약 → GAN 안정화 |

```python
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])
sigma = np.linalg.svd(A, compute_uv=False)

print(f"Nuclear norm: {np.sum(sigma):.4f}")            # ||σ||_1
print(f"Frobenius norm: {np.linalg.norm(A, 'fro'):.4f}") # ||σ||_2
print(f"Spectral norm: {np.linalg.norm(A, 2):.4f}")     # σ_max
```

**예제**: $A = \text{diag}(3, 1, 0)$일 때:
- $\|A\|_* = 3 + 1 + 0 = 4$
- $\|A\|_F = \sqrt{9 + 1 + 0} = \sqrt{10}$
- $\|A\|_\sigma = 3$

> **주의**: Frobenius 노름은 "행렬 원소에 대한 L2 노름"이자 "특이값 벡터에 대한 L2 노름"이다. 벡터의 L2 노름과 혼동하지 말 것. 또한 Spectral 노름은 최대 **특이값**이지, 최대 **고유값**이 아니다.

---

## 4. Power Method

### 4.1 알고리즘

**최대 고유값** (대칭 행렬 $A$):

$$u_{t+1} = \frac{Au_t}{\|Au_t\|}, \quad \lambda = u_t^\top A u_t$$

**최대 특이값** (일반 행렬 $A$):

$$v \leftarrow \frac{Au}{\|Au\|}, \quad u \leftarrow \frac{A^\top v}{\|A^\top v\|}, \quad \sigma = v^\top Au$$

### 4.2 수렴 분석

오차의 기대값은 지수적으로 감소한다:

$$\mathbb{E}[\text{err}(\xi_t)] \leq \sqrt{2n} \left(\frac{\lambda_2}{\lambda_1}\right)^t$$

수렴 속도는 **spectral gap** $\gamma = (\lambda_1 - \lambda_2)/\lambda_1$에 의해 결정된다. 두 번째로 큰 고유값이 첫 번째와 가까울수록 수렴이 느리다.

```python
import numpy as np

def power_method_singular(A, num_iter=100):
    """최대 특이값과 좌/우 특이벡터"""
    m, n = A.shape
    u = np.random.randn(n); u /= np.linalg.norm(u)
    for _ in range(num_iter):
        v = A @ u; v /= np.linalg.norm(v)
        u = A.T @ v; u /= np.linalg.norm(u)
    return v @ A @ u, u, v

A = np.array([[4, 1], [1, 3]])
sigma, u, v = power_method_singular(A)
print(f"Power Method: {sigma:.4f}, NumPy: {np.linalg.svd(A)[1][0]:.4f}")
```

### 4.3 딥러닝 연결

- **Spectral Normalization**: Power iteration 1-2회로 $\sigma_{\max}$를 근사 → $O(mn)$의 저비용 정규화
- **두 번째 고유값**: deflation $A' = A - \lambda_1 u_1 u_1^\top$ 후 재적용

> **주의**: Power Method는 최대 고유값만 직접 구한다. 초기 벡터가 최대 고유벡터와 직교하면(확률 0이지만 이론적으로) 수렴하지 않는다.

---

## 5. 대각합 (Trace)

### 5.1 정의와 핵심 성질

**Definition 5.1.** 정사각 행렬 $A \in \mathbb{R}^{n \times n}$의 대각합:

$$\text{Tr}(A) = \sum_{i=1}^n a_{ii}$$

| 성질 | 수식 | 비고 |
|------|------|------|
| 고유값 합 | $\text{Tr}(A) = \sum_i \lambda_i$ | 대각 원소의 합 = 고유값의 합 |
| 전치 불변 | $\text{Tr}(A^\top) = \text{Tr}(A)$ | |
| 선형성 | $\text{Tr}(A + B) = \text{Tr}(A) + \text{Tr}(B)$ | |
| **순환 성질** | $\text{Tr}(AB) = \text{Tr}(BA)$ | $AB \neq BA$여도 성립! |
| Frobenius 연결 | $\|A\|_F^2 = \text{Tr}(A^\top A)$ | |
| 이차형식 | $x^\top Ax = \text{Tr}(Axx^\top)$ | 기대값 계산의 핵심 |

> **주의**: 순환 성질은 **순환(cyclic) 치환**만 허용한다: $\text{Tr}(ABC) = \text{Tr}(BCA) = \text{Tr}(CAB)$. 하지만 $\text{Tr}(ABC) \neq \text{Tr}(ACB)$ (일반적으로). 순서를 뒤집는 것은 안 되고, **순환 이동**만 된다.

### 5.2 이차형식과 기대값

$$x^\top Ax = \text{Tr}(x^\top Ax) = \text{Tr}(Axx^\top)$$

이 변환이 핵심적인 이유: 기대값과 결합하면

$$\mathbb{E}[x^\top Ax] = \text{Tr}(A\,\mathbb{E}[xx^\top])$$

### 5.3 Hutchinson Trace 추정

**대규모 행렬**의 trace를 대각 원소에 직접 접근하지 않고 추정하는 방법:

$\mathbb{E}[xx^\top] = I$ (예: $x \sim \mathcal{N}(0, I)$ 또는 Rademacher 분포)를 이용하면:

$$\text{Tr}(A) = \mathbb{E}[x^\top Ax]$$

**Hutchinson 추정기**: $s$개의 랜덤 벡터로 Monte Carlo 추정:

$$\text{Tr}(A) \approx \frac{1}{s}\sum_{i=1}^s x_i^\top A x_i$$

분산: $\text{Var}[x^\top Ax] = 2\|A\|_F^2$ (정규분포). $s \geq 8/\epsilon$ 샘플이면 높은 확률로 상대오차 $\leq \epsilon$.

```python
import numpy as np

def hutchinson_trace(A, num_samples=1000):
    n = A.shape[0]
    estimates = [x @ A @ x for x in (np.random.randn(n) for _ in range(num_samples))]
    return np.mean(estimates)

A = np.diag([1, 2, 3, 4, 5])
print(f"정확한 Trace: {np.trace(A)}")           # 15
print(f"Hutchinson: {hutchinson_trace(A, 5000):.2f}")  # ≈ 15
```

### 5.4 딥러닝 연결

- **Hessian trace 추정**: 손실 함수의 곡률(curvature) 파악에 사용
- **Fisher Information Matrix**: Natural Gradient에서 $\text{Tr}(F)$ 추정
- **Frobenius norm 추정**: $\mathbb{E}[\|Ax\|^2] = \text{Tr}(A^\top A) = \|A\|_F^2$

> **주의**: Hutchinson 추정기의 핵심은 $A$의 원소에 직접 접근하지 않고 **행렬-벡터 곱 $Ax$만으로** trace를 추정한다는 것이다. 또한 $\mathbb{E}[xx^\top] = I$를 만족하는 분포에서 샘플링해야 한다.

---

## 6. 연립방정식 $Ax = b$

### 6.1 해의 분류

$A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$, $b \in \mathbb{R}^m$에 대해:

| 조건 | 해의 상태 | 전략 |
|------|----------|------|
| $b \in \text{im}(A)$, $\text{rank}(A) = n$ | 유일한 해 | $x = A^{-1}b$ |
| $b \in \text{im}(A)$, $\text{rank}(A) < n$ | 무한히 많은 해 | 최소노름해 $x^* = A^+b$ |
| $b \notin \text{im}(A)$ | 해 없음 | 최소제곱해 $x^* = A^+b$ |

### 6.2 실무적 해법

역행렬을 명시적으로 계산하지 않는다! `np.linalg.solve(A, b)` (내부적으로 LU 분해)를 사용한다.

```python
import numpy as np
A = np.array([[1, 2, -1], [2, -2, 4], [-1, 0.5, -1]])
b = np.array([1, -2, 0])
x = np.linalg.solve(A, b)
assert np.allclose(A @ x, b)
```

---

## 7. Moore-Penrose 유사역행렬

### 7.1 정의

**Definition 7.1.** $A \in \mathbb{R}^{m \times n}$에 대해 $A^+$는 다음 네 조건을 **동시에** 만족하는 유일한 행렬이다:

1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^\top = AA^+$ (대칭)
4. $(A^+A)^\top = A^+A$ (대칭)

### 7.2 SVD를 통한 계산

$A = \sum_i \sigma_i u_i v_i^\top$이면:

$$A^+ = \sum_{\sigma_i \neq 0} \frac{1}{\sigma_i} v_i u_i^\top$$

### 7.3 주요 성질

| 성질 | 수식 |
|------|------|
| 전치 | $(A^\top)^+ = (A^+)^\top$ |
| 등가 표현 | $A^+ = (A^\top A)^+ A^\top = A^\top(AA^\top)^+$ |
| 정사각 가역 | $A^+ = A^{-1}$ |
| 멱등성 | $(A^+A)^2 = A^+A$ |
| 영공간 | $\ker(A) = \text{im}(I - A^+A)$ |

### 7.4 최소노름 최소제곱해

**Theorem 7.1.** $x^* = A^+b$는 $\|Ax - b\|$를 최소화하는 모든 해 중에서 $\|x\|$가 최소인 해이다.

해가 존재할 때 일반해는:

$$x = A^+b + (I - A^+A)w \quad (\forall w)$$

$A^+Ax^*$와 $(I - A^+A)w$가 직교하므로, 피타고라스 정리에 의해 $\|w\| = 0$일 때 $\|x\|$가 최소이다.

```python
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2, overdetermined
b = np.array([1, 2, 3])

# Moore-Penrose 유사역행렬
A_pinv = np.linalg.pinv(A)
x = A_pinv @ b  # 최소제곱해

# Moore-Penrose 조건 검증
assert np.allclose(A @ A_pinv @ A, A)
assert np.allclose(A_pinv @ A @ A_pinv, A_pinv)
assert np.allclose((A @ A_pinv).T, A @ A_pinv)
assert np.allclose((A_pinv @ A).T, A_pinv @ A)
```

> **주의**: $A^+A \neq I$ (일반적으로). $A^+A = I$가 되려면 $A$의 열들이 선형독립이어야 한다.

---

## 8. 선형 사영과 최소제곱법

### 8.1 열공간으로의 직교 사영

**Definition 8.1.** 벡터 $b$를 $A$의 열공간 $\text{im}(A)$로 사영:

$$\text{Proj}(b; A) = AA^+b$$

이는 $\text{im}(A)$에서 $b$에 가장 가까운 점이다.

### 8.2 최소제곱법과의 동치성

$$x^* = \arg\min_x \|Ax - b\|^2 \implies Ax^* = \text{Proj}(b; A) = AA^+b$$

잔차(residual) $r = b - Ax^*$는 $A$의 열공간에 직교한다: $r \in \ker(A^\top)$.

### 8.3 벡터 사영

벡터 $a$로의 사영:

$$\text{Proj}(b; a) = \frac{a^\top b}{\|a\|^2} a$$

```python
import numpy as np

# 벡터 사영
a = np.array([1, 0, 0], dtype=float)
b = np.array([3, 4, 5], dtype=float)
proj = (a @ b / (np.linalg.norm(a)**2)) * a  # [3, 0, 0]

# 열공간 사영
A = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)  # xy 평면
b = np.array([3, 4, 5], dtype=float)
proj_col = A @ np.linalg.pinv(A) @ b  # [3, 4, 0]

# 최소제곱법
A2 = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
b2 = np.array([1, 2, 2], dtype=float)
x_ls = np.linalg.lstsq(A2, b2, rcond=None)[0]
print(f"최소제곱 해: {x_ls}")
```

### 8.4 딥러닝 연결

- **선형 회귀**: $\theta^* = (X^\top X)^{-1}X^\top y = X^+y$
- **Attention 메커니즘**: value 벡터의 가중합은 사영의 일종
- **Implicit regularization**: 과매개변수화된 모델에서 SGD가 찾는 해가 왜 최소노름해에 가까운지가 현대 이론의 핵심 질문

---

## 9. 딥러닝 적용 요약

| 수학 개념 | 딥러닝 활용 | 구체적 사례 |
|-----------|------------|------------|
| L2 노름 | MSE 손실, Weight decay | $\|y - \hat{y}\|_2^2$, $\lambda\|W\|_F^2$ |
| L1 노름 | MAE 손실, 희소 정규화 | Lasso |
| Spectral 노름 | GAN 안정화, Lipschitz 제약 | Spectral Normalization |
| Nuclear 노름 | 저랭크 완성, 추천 시스템 | Matrix completion |
| Trace | Hessian 곡률 추정 | Fisher information |
| Hutchinson 추정 | 대규모 Hessian trace | 2차 최적화 |
| 유사역행렬 $A^+$ | 최소제곱 해, 선형 회귀 | Normal equation |
| 선형 사영 | Attention, 차원 축소 | Q/K/V projection |
| Power Method | Spectral norm 계산 | GAN 판별자 정규화 |

---

## 10. 흔한 오해와 주의점

1. **"L0 노름은 노름이다"** → 삼각부등식 불만족. 수학적으로 노름이 아니다.
2. **"Frobenius 노름 = L2 노름"** → 적용 대상이 다르다. Frobenius는 행렬, L2는 벡터.
3. **"Spectral 노름 = 최대 고유값"** → 최대 **특이값**이다. 고유값과 특이값은 다르다.
4. **"$\text{Tr}(ABC) = \text{Tr}(CBA)$이므로 아무 순서나 됨"** → 순환 치환만 가능. 뒤집기 불가.
5. **"$A^+A = I$"** → 일반적으로 성립하지 않음. $A^+A$는 행공간으로의 직교 사영.
6. **"역행렬을 구해서 곱하면 됨"** → 수치적으로 `np.linalg.solve`가 더 안정적이고 빠르다.
7. **"Hutchinson은 아무 랜덤 벡터나 됨"** → $\mathbb{E}[xx^\top] = I$를 만족하는 분포 필요 (정규 또는 Rademacher).
8. **"최소제곱해는 유일하다"** → $\ker(A) \neq \{0\}$이면 무한히 많다. $A^+b$는 그 중 최소노름.

---

## 11. 핵심 요약

1. **노름**은 크기를 재는 자이다. 어떤 자를 쓰느냐(L1, L2, Spectral)에 따라 정규화의 성질이 달라진다.
2. **행렬의 세 노름** ($\|A\|_*, \|A\|_F, \|A\|_\sigma$)은 모두 특이값으로 표현되며, 각각 $L_1, L_2, L_\infty$에 대응한다.
3. **Trace**는 대각합 = 고유값의 합이며, 순환 성질 $\text{Tr}(AB) = \text{Tr}(BA)$는 기대값 계산의 핵심 도구이다.
4. **Hutchinson 추정**: $\mathbb{E}[x^\top Ax] = \text{Tr}(A)$ 한 줄의 등식으로 대규모 trace를 추정한다.
5. **유사역행렬 $A^+$**: 역행렬이 없어도 최선의 근사해를 제공한다. $A^+b$는 최소노름 최소제곱해이다.
6. **선형 사영**: $\text{Proj}(b; A) = AA^+b$는 열공간에서 $b$에 가장 가까운 점이며, 최소제곱법의 기하학적 본질이다.

> **한 문장 정리**: 노름은 딥러닝의 정규화를, trace는 곡률 추정을, 유사역행렬은 최적 근사해를 제공하며, 이 세 도구가 딥러닝의 학습과 안정성의 수학적 기반을 형성한다.

---

## 참고문헌

- Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press. Chapter 2.
- Miyato, T. et al. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*.
- Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Martens, J. & Grosse, R. (2015). Optimizing Neural Networks with Kronecker-factored Approximate Curvature. *ICML*.

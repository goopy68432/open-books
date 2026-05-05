---
title: "교수 직강 수식 풀이·증명 모음"
slug: professor-proofs
order: 7
---

# 교수 직강 수식 풀이·증명 모음

> **출처:** 한양대 이성윤 교수 「딥러닝」 강의 1·2·3·4·6·7·8주차 강의 녹취록
>
> **목적:** 교수가 강의실에서 **칠판/슬라이드에 직접 풀어 보여준 수식**과 **단계별 증명**만을 추출. 교수가 한 학생들과의 문답·중간 단계를 그대로 보존.
>
> **사용법:** 시험 답안에서 그대로 인용하거나, 답안 작성 시 "교수님은 이렇게 푸셨다"의 기준점.

---

## 목차

1. [Newton's Method — 루트7 계산 (2주차)](#1-newtons-method--루트7-계산-2주차)
2. [선형변환 ↔ 행렬 동치 증명 (1주차)](#2-선형변환--행렬-동치-증명-1주차)
3. [Range/Null 계산 + Rank-Nullity 검증 (2주차)](#3-rangenull-계산--rank-nullity-검증-2주차)
4. [Vector·Matrix 미분 4종 (2주차)](#4-vectormatrix-미분-4종-2주차)
5. [Softmax 자코비안 직접 유도 (2주차 끝)](#5-softmax-자코비안-직접-유도-2주차-끝)
6. [Bayes Theorem 1줄 증명 (3주차)](#6-bayes-theorem-1줄-증명-3주차)
7. [Bernoulli MLE 7단계 풀이 (3주차)](#7-bernoulli-mle-7단계-풀이-3주차)
8. [MAP under Uniform Prior = MLE 증명 (3주차)](#8-map-under-uniform-prior--mle-증명-3주차)
9. [MAP with Beta(2,2) Prior 풀이 (3-4주차)](#9-map-with-beta22-prior-풀이-3-4주차)
10. [MAP with θ^m(1-θ)^m Prior, m→∞ 풀이 (4주차)](#10-map-with-θm1-θm-prior-m-풀이-4주차)
11. [Tent Prior MAP — 영역 분리 풀이 (6주차, 시험 7번)](#11-tent-prior-map--영역-분리-풀이-6주차-시험-7번)
12. [ERM = NLL 등가성 증명 (7주차)](#12-erm--nll-등가성-증명-7주차)
13. [Gauss → MSE 유도 (3·7주차)](#13-gauss--mse-유도-37주차)
14. [Bernoulli → BCE 유도 (4·7주차)](#14-bernoulli--bce-유도-47주차)
15. [Newton's Method = L의 2차 근사 (8주차)](#15-newtons-method--l의-2차-근사-8주차)

---

## 1. Newton's Method — 루트7 계산 (2주차)

### 교수의 질문
> *"루트 7이 어느 정도 될까요? 이거 실제로 계산하려고 하면 컴퓨터는 뭘 하고 있을까요?"*

### 무엇을 증명/풀이?
$\sqrt{7}$을 직접 계산하는 알고리즘 (Newton's Method).

### 단계별 풀이

**1단계: 문제 변환**
$$\sqrt{7} = x \iff x^2 = 7 \iff x^2 - 7 = 0$$

→ $f(x) = x^2 - 7$의 양수 zero를 찾는 문제.

**2단계: 직관 — 2차 함수 → 1차 함수로 근사**
> *"이거의 zero를 찾기가 너무 어려워서, 1차 함수로 바꿉니다. 그게 핵심입니다."*

시작점 $x_0 = 3$. 그 점에서 접선:
$$y - f(x_0) = f'(x_0)(x - x_0)$$

여기서 $f(3) = 2$, $f'(x) = 2x$, $f'(3) = 6$이므로:
$$y - 2 = 6(x - 3) \quad \Rightarrow \quad y = 6x - 16$$

**3단계: 접선의 zero**
$$6x - 16 = 0 \quad \Rightarrow \quad x_1 = \frac{16}{6} = \frac{8}{3} \approx 2.667$$

**4단계: 일반화 (반복)**

$x_t$에서:
$$x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)} = x_t - \frac{x_t^2 - 7}{2x_t} = \frac{x_t}{2} + \frac{7}{2x_t}$$

| t | $x_t$ |
|---|-------|
| 0 | 3 |
| 1 | 8/3 ≈ 2.667 |
| 2 | 2.6458... |
| 3 | 2.6457513... |
| 4 | 2.6457513110... (8자리 정확) |

### 어디에 어떻게 왜 사용?
- **모든 최적화의 근본** (Gradient Descent의 일반화)
- 신경망 학습의 본질: "복잡한 함수를 1차로 근사 → 0 찾기"
- 8주차 강의에서 다시 등장: "f가 ∇L이면 → L에 대해 2차 근사"

---

## 2. 선형변환 ↔ 행렬 동치 증명 (1주차)

### 교수의 메시지
> *"우리가 아는 것은 행렬뿐이다."*

### 무엇을 증명?
"선형변환 = 행렬"이라는 동치성.

### 단계별 풀이

**증명 (← 방향):** 모든 행렬 $A: \mathbb{R}^n \to \mathbb{R}^m$은 선형변환.

$L_A(\mathbf{x}) := A\mathbf{x}$로 정의:
$$A(\mathbf{u} + \mathbf{v}) = A\mathbf{u} + A\mathbf{v} \checkmark$$
$$A(c\mathbf{v}) = cA\mathbf{v} \checkmark$$

→ 행렬 곱셈의 분배법칙으로 자명.

**증명 (→ 방향):** 모든 선형변환 $T: \mathbb{R}^n \to \mathbb{R}^m$은 행렬로 표현.

표준기저 $\mathbf{e}_1, \ldots, \mathbf{e}_n$에 대해 $T(\mathbf{e}_i)$를 계산. 임의 $\mathbf{x} = \sum x_i \mathbf{e}_i$:
$$T(\mathbf{x}) = T\left(\sum_i x_i \mathbf{e}_i\right) = \sum_i x_i T(\mathbf{e}_i)$$

이는 행렬 $A = [T(\mathbf{e}_1), T(\mathbf{e}_2), \ldots, T(\mathbf{e}_n)]$로:
$$T(\mathbf{x}) = A\mathbf{x}$$

→ 행렬의 각 열에 $T(\mathbf{e}_i)$를 채우면 됨.

### 어디에 어떻게 왜 사용?
- 미분도 행렬 (Jacobian)
- 신경망의 한 층 = 선형변환 + 비선형 활성화
- "복잡한 것을 단순한 행렬로 근사" 사상

---

## 3. Range/Null 계산 + Rank-Nullity 검증 (2주차)

### 교수의 예시
$A = (1, 1)$ (1×2 행렬), $\mathbf{x} = (x_1, x_2)^T$.

### 단계별 풀이

**Range 계산:**
$$A\mathbf{x} = x_1 + x_2$$

$x_1 = 0$ 고정, $x_2$를 임의 실수로 → $x_1 + x_2 \in \mathbb{R}$ 전체.
$$\mathcal{R}(A) = \mathbb{R} \Rightarrow \text{rank}(A) = 1$$

**Null 계산:**
$A\mathbf{x} = 0 \Leftrightarrow x_1 + x_2 = 0 \Leftrightarrow x_2 = -x_1$.

해의 일반형: $\mathbf{x} = t(1, -1)^T$, $t \in \mathbb{R}$.
$$\mathcal{N}(A) = \{t(1,-1)^T : t \in \mathbb{R}\} \Rightarrow \text{nullity}(A) = 1$$

**Rank-Nullity 검증:**
$$\text{rank} + \text{nullity} = 1 + 1 = 2 = n \checkmark$$

### 다른 예시 (2주차 슬라이드 문제)
$A = \text{diag}(4, 3, 2, 1, 0)$ (5×5 대각행렬).

$A\mathbf{x} = (4x_1, 3x_2, 2x_3, x_4, 0)$

- Range: 마지막 성분만 0인 벡터들 → 4차원 → rank = 4
- Null: $x_1 = x_2 = x_3 = x_4 = 0$, $x_5$ 자유 → 1차원 → nullity = 1
- 검증: 4 + 1 = 5 = n ✓

### 어디에 어떻게 왜 사용?
- **Fundamental Theorem of Linear Algebra의 핵심**
- SVD 이해의 기초
- 신경망 가중치 행렬의 표현력 분석

---

## 4. Vector·Matrix 미분 4종 (2주차)

### 교수의 메시지
> *"미분이라는 것은 매트릭스를 얻는다는 거예요."*

### 4가지 케이스

#### Case 1: Scalar to Scalar (일반)
$f(x) = x^2 \Rightarrow f'(x) = 2x$

#### Case 2: Vector to Scalar (Scalar input → Vector output)
$\mathbf{v}(x) = (1, x, x^2)^T$를 $x$로 미분:
$$\frac{d\mathbf{v}}{dx} = (0, 1, 2x)^T$$

성분별 미분.

#### Case 3: Scalar to Vector (Vector input → Scalar output)
$f(\mathbf{v}) = v_1 + v_2^2 + 2v_3$를 $\mathbf{v}$로 미분:
$$\nabla_\mathbf{v} f = \left(\frac{\partial f}{\partial v_1}, \frac{\partial f}{\partial v_2}, \frac{\partial f}{\partial v_3}\right)^T = (1, 2v_2, 2)^T$$

#### Case 4: Vector to Vector (Jacobian)
$\mathbf{u}: \mathbb{R}^n \to \mathbb{R}^m$:
$$J = \frac{\partial \mathbf{u}}{\partial \mathbf{v}} = \begin{pmatrix}
\frac{\partial u_1}{\partial v_1} & \cdots & \frac{\partial u_1}{\partial v_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial u_m}{\partial v_1} & \cdots & \frac{\partial u_m}{\partial v_n}
\end{pmatrix}$$

m×n 행렬.

### 어디에 어떻게 왜 사용?
- Backpropagation의 핵심
- Softmax 자코비안 (다음 절)
- 모든 신경망 학습의 미분 규칙

---

## 5. Softmax 자코비안 직접 유도 (2주차 끝)

### 교수의 도전 과제
> *"4분 드릴 테니까 직접 풀어보세요. p_1을 g_1으로 미분한 거랑 p_1을 g_2로 미분한 거 두 개만이라도 해보세요."*

### 무엇을 증명?
$$\frac{\partial p_i}{\partial g_j} = ?$$

여기서 $p_i = \frac{e^{g_i}}{S}$, $S = \sum_k e^{g_k}$.

### 단계별 풀이

**Case 1: i = j (대각, $p_1$을 $g_1$으로)**

분수의 미분 (몫 규칙):
$$\frac{\partial p_1}{\partial g_1} = \frac{(\partial e^{g_1}/\partial g_1) \cdot S - e^{g_1} \cdot (\partial S/\partial g_1)}{S^2}$$

분자 미분:
- $\partial e^{g_1}/\partial g_1 = e^{g_1}$
- $\partial S/\partial g_1 = e^{g_1}$ (S에서 $k=1$ 항만 살아남음)

대입:
$$= \frac{e^{g_1} \cdot S - e^{g_1} \cdot e^{g_1}}{S^2} = \frac{e^{g_1}(S - e^{g_1})}{S^2}$$

분리:
$$= \frac{e^{g_1}}{S} \cdot \frac{S - e^{g_1}}{S} = p_1 \cdot (1 - p_1)$$

**일반화:** $\boxed{\dfrac{\partial p_i}{\partial g_i} = p_i(1 - p_i)}$

**Case 2: i ≠ j (비대각, $p_1$을 $g_2$로)**

분자 미분이 달라짐:
- $\partial e^{g_1}/\partial g_2 = 0$ ($g_1$은 $g_2$ 무관)
- $\partial S/\partial g_2 = e^{g_2}$

$$\frac{\partial p_1}{\partial g_2} = \frac{0 \cdot S - e^{g_1} \cdot e^{g_2}}{S^2} = -\frac{e^{g_1}}{S} \cdot \frac{e^{g_2}}{S} = -p_1 p_2$$

**일반화:** $\boxed{\dfrac{\partial p_i}{\partial g_j} = -p_i p_j \quad (i \neq j)}$

### 통합 (Kronecker Delta)
$$\boxed{\frac{\partial p_i}{\partial g_j} = p_i(\delta_{ij} - p_j)}$$

### 행렬 형태
$$J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### 어디에 어떻게 왜 사용?
> *"Attention에서도 Softmax가 쓰이고, Image Classification에서도 Softmax가 쓰이는데, 그 함수가 다 미분이 됩니다. 이게 중요합니다."*

- **Attention 메커니즘 (Transformer)**: $\text{softmax}(QK^T/\sqrt{d_k})$
- **분류 문제 출력층** (Cross Entropy와 합성)
- **시험 8번 직접 출제**

### Cross Entropy + Softmax 합성 (보너스)
$L = -\sum_i y_i \log p_i$일 때:
$$\frac{\partial L}{\partial g_j} = p_j - y_j$$
("예측 - 정답"의 놀라운 단순성)

---

## 6. Bayes Theorem 1줄 증명 (3주차)

### 교수의 도전
> *"3분 드릴 테니 베이즈 정리 증명해보세요. 정의만 사용하면 됩니다."*

### 무엇을 증명?
$$p(H|E) = \frac{p(E|H)p(H)}{p(E)}$$

### 단계별 증명

**1단계: 조건부확률 정의 두 번 사용**
$$p(H|E) = \frac{p(H \cap E)}{p(E)} \quad \cdots (1)$$
$$p(E|H) = \frac{p(H \cap E)}{p(H)} \quad \cdots (2)$$

**2단계: 식 (2)에서 분자 추출**
$$p(H \cap E) = p(E|H) \cdot p(H)$$

**3단계: 식 (1)에 대입**
$$p(H|E) = \frac{p(E|H) p(H)}{p(E)} \quad \blacksquare$$

### 어디에 어떻게 왜 사용?
- **MAP 추정의 출발**
- 분모 $p(E)$는 $H$와 무관 → MAP에서는 무시 가능
- $\hat{H}_{\text{MAP}} = \arg\max p(E|H)p(H) = \arg\max [\text{likelihood} \times \text{prior}]$
- VAE의 ELBO 유도, Diffusion 등 모든 베이지안 추론

### 학생들이 가장 헷갈려한 부분
> *"분모 $p(E)$는 왜 무시하나요?"*
>
> 답: "데이터가 주어졌을 때 고정된 상수. $\arg\max_H$에 영향 없음."

---

## 7. Bernoulli MLE 7단계 풀이 (3주차)

### 교수의 메시지
> *"이게 모든 분포의 MLE를 통일하는 체인입니다."*

### 무엇을 증명/풀이?
$y_i \sim \text{Bern}(\theta)$ i.i.d, k = 앞면 개수일 때 $\hat{\theta}_{\text{MLE}}$.

### 7단계 풀이 (★★★★★)

**1단계: 모델 명시**
$$y_i \sim \text{Bern}(\theta), \quad i = 1, \ldots, n, \quad \text{i.i.d}$$

**2단계: 단일 pmf**
$$p(y|\theta) = \theta^y(1-\theta)^{1-y}, \quad y \in \{0, 1\}$$

검증:
- $y=1 \Rightarrow \theta^1 (1-\theta)^0 = \theta$ ✓
- $y=0 \Rightarrow \theta^0 (1-\theta)^1 = 1-\theta$ ✓

**3단계: i.i.d → 곱 (Independence 사용)**
> *"독립의 정의는 곱이라는 거 강조를 드립니다."*
$$L(\theta) = p(y_1, \ldots, y_n | \theta) = \prod_{i=1}^n p(y_i | \theta) = \prod_i \theta^{y_i}(1-\theta)^{1-y_i}$$

**4단계: 단순화 (지수의 합)**
$\sum y_i = k$, $\sum (1-y_i) = n-k$이므로:
$$L(\theta) = \theta^k (1-\theta)^{n-k}$$

**5단계: 로그 (★ 핵심 단계)**
> *"왜 로그? 곱→합으로 미분 단순, log 단조 → argmax 보존, 수치 안정성."*
$$\ell(\theta) = \log L(\theta) = k \log \theta + (n-k) \log(1-\theta)$$

**6단계: 미분 = 0 (페르마 정리)**
$$\frac{d\ell}{d\theta} = \frac{k}{\theta} - \frac{n-k}{1-\theta} = 0$$

**7단계: 풀이 (대수 계산)**
$$\frac{k}{\theta} = \frac{n-k}{1-\theta}$$
$$k(1-\theta) = (n-k)\theta$$
$$k - k\theta = n\theta - k\theta$$
$$k = n\theta$$
$$\boxed{\hat{\theta}_{\text{MLE}} = \frac{k}{n}}$$

**8단계 (검증): 2계 미분**
$$\frac{d^2\ell}{d\theta^2} = -\frac{k}{\theta^2} - \frac{n-k}{(1-\theta)^2} < 0$$

→ ℓ은 오목 → 임계점이 전역 최댓값.

### 교수의 통찰 (직접 인용)
> *"3번 던져서 3번 앞면 나왔어요. MLE는 1이라고 답합니다. 데이터에만 의존하고 있어요. 시행 횟수가 적을 때 안 좋습니다."*

### 어디에 어떻게 왜 사용?
- **모든 분포의 MLE의 템플릿** (정규, 푸아송, 지수 모두 같은 7단계)
- 머신러닝 학습 알고리즘의 출발점
- **시험 4번 직접 출제**

---

## 8. MAP under Uniform Prior = MLE 증명 (3주차)

### 교수의 통찰
> *"맥시멈 라이클리오드는 사실 'Prior가 Uniform일 때의 MAP'이다 — 이렇게 거꾸로 이해하라."*

### 무엇을 증명?
$\hat{\theta}_{\text{MAP}}$ under uniform prior $= \hat{\theta}_{\text{MLE}}$.

### 증명

prior $p(\theta) = 1$ (uniform on $[0,1]$).

log posterior:
$$\log p(\theta|D) = \underbrace{\log L(\theta)}_{\ell(\theta)} + \underbrace{\log p(\theta)}_{\log 1 = 0} - \underbrace{\log p(D)}_{\text{const}}$$

상수항은 미분하면 0:
$$\frac{d}{d\theta} \log p(\theta|D) = \frac{d\ell}{d\theta}$$

→ MLE 풀이와 동일. $\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{MLE}} = k/n$. $\blacksquare$

### 어디에 어떻게 왜 사용?
- **MLE와 MAP의 통합 이해**
- "지식이 없을 때(uniform) MAP = MLE"
- 정규화의 베이지안 해석의 출발점

---

## 9. MAP with Beta(2,2) Prior 풀이 (3-4주차)

### 무엇을 증명/풀이?
prior $p(\theta) \propto \theta(1-\theta)$일 때 $\hat{\theta}_{\text{MAP}}$.

### 단계별 풀이

**1단계: log prior**
$$\log p(\theta) = \log \theta + \log(1-\theta) + \text{const}$$

**2단계: log posterior**
$$\log p(\theta|D) = \underbrace{k\log\theta + (n-k)\log(1-\theta)}_{\text{log likelihood}} + \underbrace{\log\theta + \log(1-\theta)}_{\text{log prior}} + C$$

$$= (k+1)\log\theta + (n-k+1)\log(1-\theta) + C$$

**3단계: 미분 = 0**
$$\frac{k+1}{\theta} - \frac{n-k+1}{1-\theta} = 0$$
$$(k+1)(1-\theta) = (n-k+1)\theta$$
$$k+1 = (n+2)\theta$$
$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+1}{n+2}}$$

### 직관 (교수의 설명)
> *"가상의 데이터를 1번 더 본 것처럼 작동. 앞면 1번, 뒷면 1번을 미리 더해준 효과. 데이터에 덜 의존."*

### 검증: 3번 던져서 3번 앞면 (n=3, k=3)
- MLE: 3/3 = **1.0** (극단적!)
- MAP (Beta(2,2)): (3+1)/(3+2) = **0.8** (덜 극단적, 더 합리적)

### 어디에 어떻게 왜 사용?
- **Laplace's Rule of Succession** (라플라스의 후계의 법칙)
- 작은 데이터에서 **smoothing** (Add-1 smoothing의 출처)
- NLP의 단어 빈도 추정 등에 활용

---

## 10. MAP with θ^m(1-θ)^m Prior, m→∞ 풀이 (4주차)

### 무엇을 증명/풀이?
prior $p(\theta) \propto \theta^m(1-\theta)^m$일 때 $\hat{\theta}_{\text{MAP}}$, 그리고 $m \to \infty$ 극한.

### 단계별 풀이

**1단계: log prior**
$$\log p(\theta) = m\log\theta + m\log(1-\theta) + C$$

**2단계: log posterior**
$$\log p(\theta|D) = (k+m)\log\theta + (n-k+m)\log(1-\theta) + C$$

**3단계: 미분 = 0**
$$\frac{k+m}{\theta} = \frac{n-k+m}{1-\theta}$$
$$(k+m)(1-\theta) = (n-k+m)\theta$$
$$k+m = (n+2m)\theta$$
$$\boxed{\hat{\theta}_{\text{MAP}} = \frac{k+m}{n+2m}}$$

**4단계: 극한 분석**

분자/분모를 m으로 나눔:
$$\hat{\theta}_{\text{MAP}} = \frac{k/m + 1}{n/m + 2}$$

$m \to \infty$이면 $k/m \to 0$, $n/m \to 0$:
$$\boxed{\lim_{m \to \infty} \hat{\theta}_{\text{MAP}} = \frac{0 + 1}{0 + 2} = \frac{1}{2}}$$

### 검증 (3가지 극한)

| m | 결과 | 해석 |
|---|------|-----|
| 0 | k/n (MLE) | prior 약함, 데이터만 |
| 1 | (k+1)/(n+2) | 균형 |
| ∞ | 1/2 | prior 강함, 데이터 무시 |

### 교수의 통찰 (직접 인용)
> *"M이 무한대로 보낸다는 것은 prior가 0.5에 디랙 델타로 수렴 — '나는 이 동전이 무조건 0.5라고 강하게 믿는다'. 그러면 1000번 던져 1000번 앞면이 나와도 0.5라고 답한다."*

### 어디에 어떻게 왜 사용?
- **시험 5번 직접 출제** (대칭 prior, m → ∞)
- **MLE vs Strong MAP 균형 이해**
- 정규화 강도 조절의 베이지안 해석

---

## 11. Tent Prior MAP — 영역 분리 풀이 (6주차, 시험 7번)

### 무엇을 증명/풀이?
prior $p_m(\theta) = m - m^2|\theta - 0.5|$ for $|\theta - 0.5| \leq 1/m$, 0 otherwise.

### 단계별 풀이 (m=2 케이스)

**1단계: prior 형태 확인**
- 정의역: $|\theta - 0.5| \leq 1/2 \Rightarrow \theta \in [0, 1]$
- 영역 (a) $\theta \in [0, 0.5]$: $p_2(\theta) = 2 - 4(0.5 - \theta) = 4\theta$
- 영역 (b) $\theta \in [0.5, 1]$: $p_2(\theta) = 2 - 4(\theta - 0.5) = 4 - 4\theta$

**2단계: posterior in 영역 (a)**

$L(\theta) = \theta^4(1-\theta)$ (n=5, k=4 가정)

$$f_a(\theta) = \theta^4(1-\theta) \cdot 4\theta = 4\theta^5(1-\theta)$$

$$\log f_a = \log 4 + 5\log\theta + \log(1-\theta)$$

미분:
$$\frac{d}{d\theta}\log f_a = \frac{5}{\theta} - \frac{1}{1-\theta}$$

= 0: $5(1-\theta) = \theta$ → $\theta = 5/6 \approx 0.833$

→ 영역 (a) [0, 0.5] 밖! 영역 (a) 내에서는 단조증가 → **0.5에서 최댓값**.

**3단계: posterior in 영역 (b)**

$$f_b(\theta) = \theta^4(1-\theta) \cdot (4-4\theta) = 4\theta^4(1-\theta)^2$$

미분:
$$\frac{d}{d\theta}\log f_b = \frac{4}{\theta} - \frac{2}{1-\theta} = 0$$

$$4(1-\theta) = 2\theta \Rightarrow \theta = \frac{2}{3}$$

영역 (b) [0.5, 1] **내**에 위치 → 정점 $\theta = 2/3$.

**4단계: 두 영역 비교**

영역 (a) 최댓값: $f_a(0.5) = 4(0.5)^5(0.5) = 1/16$
영역 (b) 최댓값: $f_b(2/3) = 4(2/3)^4(1/3)^2 = 64/729 \approx 0.0878$

$1/16 = 0.0625 < 0.0878$ → **(b)가 더 큼**.

$$\boxed{\hat{\theta}_{\text{MAP}}^{(m=2)} = \frac{2}{3}}$$

### 교수의 통찰
> *"영역 바깥에서는 prior = 0이라 log prior = -∞, posterior도 -∞. 따라서 그 영역은 후보가 될 수 없습니다."*

### 어디에 어떻게 왜 사용?
- **시험 7번 직접 출제** (가장 어려운 문제)
- 절댓값 함수의 미분 불가능 점 처리
- prior가 실제로 도메인 제약을 주는 경우 (정규화 함수의 support 이해)

---

## 12. ERM = NLL 등가성 증명 (7주차)

### 무엇을 증명?
$$\hat{R}(h) = \mathbb{E}_{x \sim p_S}[\ell(h(x), y)] \overset{?}{=} -\frac{1}{n}\sum_i \log p(y_i|h)$$

### 단계별 증명

**1단계: Empirical distribution 정의**

n개 데이터 $\{x_1, \ldots, x_n\}$:
$$p_S(x) = \frac{1}{n}\sum_{i=1}^n \delta(x - x_i)$$

**2단계: Expectation 전개**
$$\mathbb{E}_{x \sim p_S}[f(x)] = \int p_S(x) f(x)\,dx = \frac{1}{n}\sum_i \int \delta(x - x_i) f(x)\,dx = \frac{1}{n}\sum_i f(x_i)$$

**3단계: ERM과 NLL 연결**

ERM:
$$\hat{R}(h) = \frac{1}{n}\sum_i \ell(h(x_i), y_i)$$

여기서 $\ell(h(x), y) := -\log p(y|x, h)$로 정의하면:
$$\hat{R}(h) = -\frac{1}{n}\sum_i \log p(y_i|x_i, h)$$

NLL은 $-\log L = -\sum_i \log p(y_i|x_i, h)$이므로:
$$\hat{R}(h) = \frac{1}{n}\text{NLL}(h)$$

→ argmin은 동일 (1/n은 상수배). $\blacksquare$

### 교수의 핵심 메시지
> *"Loss Function이 먼저가 아니에요. NLL이 먼저예요. 그러니까 Squared Loss를 왜 쓰는지, Cross Entropy를 왜 쓰는지에 대한 답은 ─ 어떤 distribution을 hypothesis로 주느냐에 따라 결정됩니다."*

### 어디에 어떻게 왜 사용?
- **머신러닝 학습 = NLL 최소화 = ERM**의 통합 관점
- 임의 손실 함수의 정당화
- 통계학과 ML의 연결고리

---

## 13. Gauss → MSE 유도 (3·7주차)

### 교수의 메시지
> *"가우시안 3가지 핵심: Exponential, Minus, Square. 음의 로그를 취하면 Exp와 Minus가 사라지고 제곱만 남습니다."*

### 무엇을 증명?
가우스 잡음 가정 → MSE 손실.

### 단계별 증명

**1단계: 모델 가정**
$y_i = f(x_i; \theta) + \epsilon_i$, $\epsilon_i \sim N(0, \sigma^2)$ i.i.d.

→ $y_i | x_i, \theta \sim N(f(x_i; \theta), \sigma^2)$.

**2단계: 우도**
$$L(\theta) = \prod_i p(y_i | x_i, \theta) = \prod_i \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right)$$

**3단계: 로그 우도 — Exp와 Minus가 사라짐**
$$\ell(\theta) = -\frac{n}{2}\log(2\pi\sigma^2) - \sum_i \frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}$$

**4단계: NLL — 부호 반전, 제곱만 남음**
$$\text{NLL}(\theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i (y_i - f_i)^2$$

**5단계: argmin (상수와 양의 스칼라 무시)**
$$\arg\min_\theta \text{NLL}(\theta) = \arg\min_\theta \sum_i (y_i - f(x_i; \theta))^2 = \arg\min_\theta \text{MSE}$$

### 교수의 통찰
> *"MSE는 가우스 잡음 가정의 자연스러운 결과. 다른 잡음 분포면 다른 손실. 라플라스 잡음 → MAE."*

### 어디에 어떻게 왜 사용?
- **회귀 문제의 모든 MSE 정당화**
- "왜 MSE인가?"에 대한 정답
- 정규화 추가 → MAP → L2 정규화 (Ridge)

---

## 14. Bernoulli → BCE 유도 (4·7주차)

### 무엇을 증명?
베르누이 가정 → Binary Cross Entropy 손실.

### 단계별 증명

**1단계: 모델**
$y_i \in \{0, 1\}$, $y_i \sim \text{Bern}(p_i)$, $p_i = \sigma(f(x_i; \theta))$.

**2단계: 우도**
$$L(\theta) = \prod_i p_i^{y_i}(1-p_i)^{1-y_i}$$

**3단계: NLL**
$$\text{NLL} = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

**4단계: BCE 인식**
이게 정확히 **Binary Cross Entropy**:
$$\boxed{\text{BCE} = -\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]}$$

### 다중 클래스 (Categorical → CE)
$y$ one-hot, $\mathbf{p} = \text{softmax}(z)$:
$$\text{CE} = -\sum_i \sum_k y_{ik}\log p_{ik}$$

### Sigmoid + BCE 그래디언트 (보너스)
$\partial L/\partial z = p - y$ ("예측 - 정답")

### 어디에 어떻게 왜 사용?
- **이진 분류의 모든 BCE 정당화**
- 다중 분류의 CE 일반화
- 신경망 학습의 표준 손실

---

## 15. Newton's Method = L의 2차 근사 (8주차)

### 무엇을 증명?
"f의 zero finding은 L의 2차 근사 + 그래디언트 = 0"임을.

### 단계별 증명

**1단계: f가 L의 도함수일 때**
$$f(x) = L'(x)$$

**2단계: f의 1차 근사 (Newton 적용)**
$$f(x) \approx f(x_t) + f'(x_t)(x - x_t)$$

f를 zero로:
$$f(x_t) + f'(x_t)(x - x_t) = 0 \Rightarrow x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)}$$

**3단계: L의 2차 근사 (Taylor)**
$$L(x) \approx L(x_t) + L'(x_t)(x - x_t) + \frac{1}{2}L''(x_t)(x - x_t)^2$$

L의 그래디언트 = 0:
$$L'(x_t) + L''(x_t)(x - x_t) = 0$$
$$x_{t+1} = x_t - \frac{L'(x_t)}{L''(x_t)}$$

**4단계: 두 식 일치 확인**
$f(x) = L'(x)$이면 $f'(x) = L''(x)$이므로:
$$x_{t+1} = x_t - \frac{f(x_t)}{f'(x_t)} = x_t - \frac{L'(x_t)}{L''(x_t)}$$

→ 동일! **Newton's Method = L의 2차 근사 + zero gradient.**

### 교수의 통찰
> *"f를 zero finding한다는 것은 L 입장에서는 quadratic approximation을 한 후 그래디언트가 0인 위치를 찾는다는 거예요."*

### Gradient Descent와의 관계
GD는 1차 근사만:
$$x_{t+1} = x_t - \eta L'(x_t)$$
($\eta$는 고정 학습률, $1/L''$의 근사)

Newton은 2차 정보까지 사용 → 더 빠른 수렴 (단, $L''$ 계산 비싼).

### 어디에 어떻게 왜 사용?
- **모든 최적화의 근본 이해**
- Gradient Descent → Newton → Gauss-Newton → Levenberg-Marquardt
- 뉴럴넷 2차 최적화 (Hessian 기반) 이해

---

## 부록 A. 수식 풀이의 8대 패턴 (교수가 반복 사용)

교수가 모든 풀이에서 반복하는 8개 패턴:

1. **i.i.d → 곱**: 결합확률 분해
2. **로그 → 합**: 곱셈을 덧셈으로
3. **NLL → 부호 반전**: 최대화를 최소화로
4. **분수 미분**: 몫의 규칙
5. **케이스 분리**: i=j vs i≠j
6. **분모 통분**: 미분 후 정리
7. **m으로 나누기**: 극한 분석
8. **영역 분리**: 절댓값 처리

→ 답안 작성 시 이 8개 중 어느 패턴을 쓰는지 명시.

---

## 부록 B. 강의 시간 분배

| 주차 | 풀이 시간 (강의 중 비중) | 핵심 풀이 |
|-----|---------------------|---------|
| 1주차 | ~5% | (도입, 풀이 적음) |
| 2주차 | ~30% | Newton's Method, Eigenvalue, Softmax 자코비안 |
| 3주차 | ~50% | **Bayes Theorem, MLE 7단계** ★★★★★ |
| 4주차 | ~40% | **MAP 변형, m→∞** ★★★★ |
| 6주차 | ~30% | **Tent Prior** (시험 7번 ★) |
| 7주차 | ~25% | ERM=NLL, Gauss→MSE |
| 8주차 | ~30% | Newton 재방문, Backprop 준비 |

---

## 부록 C. 시험 답안 작성 시 활용

각 풀이는 **시험 답안에서 그대로 인용 가능**. 단, 풀이 단계마다:

1. **정의/가정 명시** (1단계)
2. **각 단계의 "왜?" 한 문장**
3. **수식만 나열 ≠ 답안** (논리 서술 필수)
4. **검증 단계** (2계 미분, 검증 등)

→ 자세한 답안 형식은 [`MASTER-CONCEPTS_v2.md`](./MASTER-CONCEPTS_v2.md) §12 참조.

---

**작성:** 2026-04-26
**기반:** 1·2·3·4·6·7·8주차 강의 녹취록 직접 인용
**참조:** [`MASTER-CONCEPTS_v2.md`](./MASTER-CONCEPTS_v2.md), [`LEARNING-MAP_v2.md`](./LEARNING-MAP_v2.md)

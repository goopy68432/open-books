---
title: "딥러닝 이론 마스터 컨셉 가이드"
slug: master-concepts
order: 4
---

# 딥러닝 이론 마스터 컨셉 가이드

> **출처:** 한양대 이성윤 교수 「딥러닝」 강의 (742장 슬라이드, 2025년 11월 28일 업데이트)
> **목표:** 시험 대비를 넘어, **이 과목 전체가 가르치고자 하는 것**을 단일 문서로 마스터.

---

## 0. 과목의 메시지 — 이 과목이 가르치고자 하는 것

이 강의는 단순히 "딥러닝 모델을 어떻게 쓰는가"가 아니라, **딥러닝이 왜·어떻게 작동하는가**를 수학적으로 연역해내는 사고법을 가르칩니다. 강의 내내 반복되는 3대 메시지:

### 메시지 ① 딥러닝 = 선형대수 + 미적분 + 확률의 조합
- 한 분야가 아니라 세 분야의 정수가 결합된 결과
- 신경망 한 층 = 선형대수 (W·x + b)
- 학습 = 미적분 (gradient descent)
- 손실 함수 = 확률 (NLL = -log p(D|θ))

### 메시지 ② "왜?"를 글로 설명할 수 있어야 한다
교수님 채점 철학: **"답만 적으면 0점"**.
모든 유도 단계마다 "왜 이 단계를 밟는가"를 한국어 문장으로 명시할 수 있어야 한다. 예: "i.i.d 가정으로 곱 → 로그 단조성으로 argmax 보존 → 페르마 정리로 미분=0 후보".

### 메시지 ③ i.i.d → 로그 → 미분=0 체인의 보편성
모든 분포의 MLE/MAP은 동일한 7단계 체인으로 풀린다 (베르누이·정규·푸아송·지수 모두). 이 체인 하나로 출제 35%를 커버.

```
   y_i ~ Distribution(θ) i.i.d
            ↓ (i.i.d → 곱)
   L(θ) = ∏ p(y_i|θ)
            ↓ (log → 합 + argmax 보존 + 수치 안정)
   ℓ(θ) = ∑ log p(y_i|θ)
            ↓ (× -1 → 손실 최소화 표준)
   NLL(θ) = -ℓ(θ)
            ↓ (페르마 정리)
   dℓ/dθ = 0
            ↓ (풀이 + 2계 미분 검증)
   θ̂_MLE
```

---

## 1. 수학적 기초

### 1.1 선형대수 (page_021–150)

#### 벡터·행렬 기본 연산
- **내적:** $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_i u_i v_i$
- **노름:** $\|\mathbf{v}\| = \sqrt{\mathbf{v}^T \mathbf{v}}$
- **행렬 곱:** $(AB)_{ij} = \sum_k A_{ik} B_{kj}$
- **전치:** $(AB)^T = B^T A^T$ (순서 뒤집힘)

#### 핵심 성질 (slide 97 page_110)
$$AB \neq BA \quad (\text{교환법칙 없음})$$
$$AB = 0 \nRightarrow A = 0 \text{ or } B = 0 \quad (\text{영인자 존재})$$
$$A^2 = 0 \nRightarrow A = 0$$
$$\|AB\|_F = \|BA\|_F \quad (\text{Frobenius norm은 보존})$$

#### Range, Null, Rank-Nullity (slide 41)
- **Range** $\mathcal{R}(A) = \text{im}(A) = \{A\mathbf{v} : \mathbf{v} \in \mathbb{R}^n\}$ (column space)
- **Null** $\mathcal{N}(A) = \ker(A) = \{\mathbf{v} : A\mathbf{v} = \mathbf{0}\}$
- **Rank-Nullity 정리:**
$$\dim(\mathcal{R}(A)) + \dim(\mathcal{N}(A)) = n$$

#### 행렬식과 역행렬
$$\det\begin{pmatrix}a & b \\ c & d\end{pmatrix} = ad - bc, \quad A^{-1} = \frac{1}{\det A}\begin{pmatrix}d & -b \\ -c & a\end{pmatrix}$$
- $A^{-1}$ 존재 ⇔ $\det(A) \neq 0$
- $\det(AB) = \det(A)\det(B)$

#### 고유값·고유벡터 (eigendecomposition, slide 60 page_070)
$$A\mathbf{v} = \lambda \mathbf{v}, \quad \mathbf{v} \neq \mathbf{0}$$

특성방정식: $\det(A - \lambda I) = 0$.

**Power method 직관 (page_070):** $A^t \mathbf{u}_0 = \sum_i \lambda_i^t (\mathbf{v}_i^T \mathbf{u}_0) \mathbf{v}_i$. 큰 t에서 최대 고유벡터 방향이 우세:
$$\tan^2(\mathbf{v}_1, \mathbf{u}_t) \leq \tan^2(\mathbf{v}_1, \mathbf{u}_0) \left(\frac{\lambda_2}{\lambda_1}\right)^{2t}$$

#### 스펙트럴 정리 (대칭행렬)
$A^T = A$ (대칭) ⇒ 모든 고유값 실수 + 직교 고유벡터 기저 ⇒
$$A = Q \Lambda Q^T$$
여기서 $Q$는 직교, $\Lambda$는 대각.

#### SVD (Singular Value Decomposition)
임의 $A \in \mathbb{R}^{m\times n}$:
$$A = U \Sigma V^T$$
- $U, V$ 직교, $\Sigma$ 대각 (특이값 $\sigma_1 \geq \cdots \geq 0$).
- $A^TA = V\Lambda V^T$, $\Lambda = \Sigma^T\Sigma$ → $\sigma_i = \sqrt{\lambda_i}$.

**Eckart-Young 정리:** Rank-k 근사 최적: $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.

#### Moore-Penrose Pseudoinverse (slide 80 page_080)
$$A^+ = V \Sigma^+ U^T$$
- 정규방정식 해: $\hat{\mathbf{x}} = A^+ \mathbf{b}$ for $A\mathbf{x} = \mathbf{b}$.

---

### 1.2 미적분과 최적화 (page_100–160)

#### 미분의 정의·핵심 공식
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

| 함수 | 도함수 |
|------|--------|
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\log x$ | $1/x$ |
| $f(g(x))$ | $f'(g(x)) \cdot g'(x)$ (체인 룰) |

#### 페르마 정리 (1차 조건)
미분 가능 함수 $f$의 내부 극값 $a$에서 $f'(a) = 0$.

**볼록(concave)이면 임계점 = 전역 최댓값.**

#### Lagrange Multiplier (slide ~110, page_120)
제약 $g(\mathbf{x}) = 0$ 하에서 $f(\mathbf{x})$ 최적화:
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$
$$\nabla_{\mathbf{x}} \mathcal{L} = 0, \quad g(\mathbf{x}) = 0$$

PCA·SVM 유도의 핵심 도구.

#### 적분과 부분적분
$$\int_a^b f(x)\,dx = F(b) - F(a)$$
$$\int u\,dv = uv - \int v\,du \quad (\text{부분적분})$$

#### 가우스 적분 (시험 단골)
$$\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$$

**증명 (극좌표):** $I^2 = \iint e^{-(x^2+y^2)/2}\,dx\,dy = \int_0^{2\pi}\int_0^\infty e^{-r^2/2} r\,dr\,d\phi = 2\pi$.

#### 미분방정식 기초 (page_140)
- 분리변수: $x' = x^2$ → $\int dx/x^2 = \int dt$ → $-1/x = t + C$
- 특이점: $t = 1/x_0$에서 발산

#### 볼록성 (Convexity, slide ~330 page_372)
$f$ 볼록 ⇔ $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$
⇔ $f'' \geq 0$ (1변수)
⇔ Hessian PSD (다변수)

**Jensen 부등식:** 볼록 $f$에 대해 $f(E[X]) \leq E[f(X)]$.

---

### 1.3 확률·통계 (page_160–230)

#### 기본 공리와 포함배제 (slide 147 page_160)
$$|E_1 \cup E_2| = |E_1| + |E_2| - |E_1 \cap E_2|$$
$$p(E_1 \cup E_2) = p(E_1) + p(E_2) - p(E_1 \cap E_2)$$
$$p(E^c) = 1 - p(E)$$
$$p(E_1 \uplus E_2) = p(E_1) + p(E_2) \quad (\text{서로소})$$

#### 조건부확률·독립
$$p(A|B) = \frac{p(A \cap B)}{p(B)}$$
- 독립: $p(A \cap B) = p(A)p(B) \Leftrightarrow p(A|B) = p(A)$.

#### 베이즈 정리 (slide ~250)
$$p(\theta | D) = \frac{p(D|\theta) p(\theta)}{p(D)} \propto p(D|\theta) p(\theta)$$

#### 확률변수와 분포

**베르누이 Bern(θ):**
$$p(y|\theta) = \theta^y (1-\theta)^{1-y}, \quad y \in \{0,1\}$$
$$E[y] = \theta, \quad \text{Var}[y] = \theta(1-\theta)$$

**이항 Binomial(n, θ):**
$$p(k) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$$
$$E[K] = n\theta, \quad \text{Var}[K] = n\theta(1-\theta)$$

**균일 Uniform[a,b] (slide ~190):**
$$p(x) = \frac{1}{b-a}, \quad x \in [a,b]$$
$$E[X] = \frac{a+b}{2}, \quad \text{Var}[X] = \frac{(b-a)^2}{12}$$

**정규 N(μ, σ²) (slide ~180 page_180):**
$$p(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- 표준정규의 모멘트: $E[X^{2n}] = (2n-1)!!$, 홀수차 = 0.
- $E[X^2]=1, E[X^4]=3, E[X^6]=15$.

**푸아송 Poisson(λ):**
$$p(y) = \frac{e^{-\lambda} \lambda^y}{y!}, \quad E[Y] = \text{Var}[Y] = \lambda$$

#### Chebyshev·Hoeffding 부등식 (slide 200 page_220)

**Chebyshev:** $X$가 평균 $\mu$, 분산 $\sigma^2$일 때:
$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

**Hoeffding (i.i.d 유계 표본):** $X_i \in [a, b]$, $\bar{X} = \frac{1}{n}\sum X_i$:
$$P(|\bar{X} - E[\bar{X}]| \geq t) \leq 2\exp\left(-\frac{2nt^2}{(b-a)^2}\right)$$

표본평균이 기댓값에 **지수적으로 빠르게** 수렴 → 학습이론의 기초.

#### 중심극한정리 (CLT)
$X_1, \ldots, X_n$ i.i.d, 평균 $\mu$, 분산 $\sigma^2$:
$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1)$$

→ 대량 표본의 평균은 분포에 무관하게 정규에 수렴.

#### 기댓값·분산
$$E[X] = \int x p(x)\,dx, \quad \text{Var}[X] = E[X^2] - (E[X])^2$$
- 선형성: $E[aX + b] = aE[X] + b$
- 독립이면 $E[XY] = E[X]E[Y]$

---

## 2. 통계적 추정과 학습 (page_230–320)

### 2.1 Bayesian vs Frequentist (slide 226 page_240)

| | Bayesian | Frequentist |
|---|---|---|
| 확률 해석 | belief (신념) | frequency (빈도) |
| θ | 확률변수 | 고정 미지수 |
| 목적 | 사후분포 $p(\theta|D)$ | 점추정 $\hat{\theta}$ |
| 결과 | MAP, Bayesian inference | MLE |

**Laplace 빈도주의 정의:** $p(E) = |E|/|S|$ (모든 결과 동등)

### 2.2 MLE — 최대우도추정

**정의:** $\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(D|\theta) = \arg\max_\theta L(\theta)$

**7단계 체인:**
1. **모델 명시** ($y_i \sim$ Distribution(θ) i.i.d)
2. **단일 pdf/pmf**
3. **i.i.d → 곱**: $L(\theta) = \prod p(y_i|\theta)$
4. **단순화** (지수합 등)
5. **로그**: $\ell(\theta) = \log L$ — 곱→합, argmax 보존, 수치 안정
6. **미분=0** (페르마)
7. **풀이 + 2계 미분 검증**

**베르누이 결과:** $\hat{\theta}_{\text{MLE}} = k/n$.
**정규 (μ만):** $\hat{\mu} = \bar{x}$.
**정규 (σ²만):** $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$.
**푸아송:** $\hat{\lambda} = \bar{x}$.
**지수:** $\hat{\lambda} = 1/\bar{x}$.

→ **모든 MLE 결과 = 표본의 자연스러운 통계량.**

### 2.3 NLL = Negative Log-Likelihood

$$\text{NLL}(\theta) = -\log L(\theta) = -\sum_i \log p(y_i|\theta)$$

**왜 음수?**
- ML 표준은 손실 **최소화** → max → min 변환
- 그래디언트 하강과 호환

**NLL 미분 = MLE 풀이.**

### 2.4 MAP — 사전 + 데이터

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta|D) = \arg\max_\theta p(D|\theta) p(\theta)$$

**베르누이 + Beta(a+1, b+1) prior 일반식:**
$$\hat{\theta}_{\text{MAP}} = \frac{k+a}{n+a+b}$$

- 균일 prior (a=b=0): MAP = MLE = k/n
- 대칭 prior $\theta^m(1-\theta)^m$: MAP = (k+m)/(n+2m), m→∞이면 1/2
- 비대칭 $\theta^m$: MAP = (k+m)/(n+m), m→∞이면 1

### 2.5 Cross Entropy = Bernoulli NLL

$$\text{NLL} = -\sum_i [y_i\log p_i + (1-y_i)\log(1-p_i)] = \text{BCE}$$

다중 클래스 (one-hot $y$, $p = \text{softmax}(z)$):
$$\text{CE} = -\sum_i \sum_k y_{ik} \log p_{ik}$$

### 2.6 KL Divergence와 정보이론

**Entropy:** $H(p) = -\sum p_i \log p_i$
**Cross Entropy:** $H(p, q) = -\sum p_i \log q_i$
**KL:**
$$\text{KL}(p \| q) = \sum p_i \log\frac{p_i}{q_i} \geq 0$$

**핵심 분해:** $H(p, q) = H(p) + \text{KL}(p \| q)$.

**증명 (KL ≥ 0, Jensen):**
$\log$ 오목 → $-\text{KL} = E_p[\log(q/p)] \leq \log E_p[q/p] = \log 1 = 0$.

### 2.7 Gauss → MSE 등가성 (slide ~270 page_290)

$y_i = f(x_i; \theta) + \epsilon_i$, $\epsilon \sim N(0, \sigma^2)$ i.i.d일 때:
$$\text{NLL}(\theta) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum(y_i - f_i)^2$$

θ 무관 항 무시 → **MSE 최소화**와 동치.

### 2.8 MAP → L2 정규화

가중치 prior $\mathbf{w} \sim N(\mathbf{0}, \tau^2 I)$일 때:
$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min[\text{NLL} + \frac{1}{2\tau^2}\|\mathbf{w}\|^2]$$

→ NLL + L2 정규화 = **L2의 베이지안 해석**.

| Prior | 정규화 |
|-------|-------|
| Gauss $N(0, \tau^2)$ | L2: $\lambda\|\mathbf{w}\|^2$ |
| Laplace | L1: $\lambda\|\mathbf{w}\|_1$ |
| 균일 | 없음 (=MLE) |

---

## 3. 고전 ML 이론 (page_310–360)

### 3.1 Empirical Risk Minimization (ERM)

**진짜 위험:** $R(h) = E_{(x,y) \sim P}[\ell(h(x), y)]$ — 알 수 없음 (분포 P 모름)
**경험적 위험:** $\hat{R}(h) = \frac{1}{n}\sum_i \ell(h(x_i), y_i)$

ERM: $\hat{h} = \arg\min_h \hat{R}(h)$.

### 3.2 Linear Models

$$h(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$$

**Logistic Regression:** $p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$, BCE 손실로 학습.

**LDA vs QDA (slide 309 page_350):**
- LDA: 공분산 동일 가정 → 선형 결정경계
- QDA: 클래스별 공분산 → 2차(곡선) 결정경계

### 3.3 Linear Separability와 SVM (slide ~315–330)

**Linearly separable:** ∃ $\mathbf{w}, b$ s.t. $y_i(\mathbf{w}^T\mathbf{x}_i + b) > 0 \forall i$.

**Margin:** $\gamma = \min_i \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|}$.

**SVM 목적:** Margin 최대화
$$\max_{\mathbf{w}, b} \min_i \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

**Hard-margin (Lagrange):**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.}\; y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$

---

## 4. 최적화 (page_360–440)

### 4.1 Gradient Descent

$$\theta^{(t+1)} = \theta^{(t)} - \eta \nabla L(\theta^{(t)})$$

**수렴 조건 (L-smooth + 볼록):** $\eta \leq 1/L$이면 $L(\theta^{(t)}) - L^* \leq \frac{\|\theta^{(0)} - \theta^*\|^2}{2\eta t}$ → $O(1/t)$.

**Strongly convex (μ-strong):** 선형 수렴
$$L(\theta^{(t)}) - L^* \leq (1 - \mu/L)^t [L^{(0)} - L^*]$$

### 4.2 SGD + Momentum + Adam

**SGD:**
$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \frac{1}{|B|}\sum_{i \in B} \nabla L_i(\theta)$$

**Momentum:**
$$v^{(t+1)} = \beta v^{(t)} + \nabla L, \quad \theta^{(t+1)} = \theta^{(t)} - \eta v^{(t+1)}$$

**Adam:**
$$m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1)\nabla L$$
$$v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2)(\nabla L)^2$$
$$\hat{m}^{(t)} = m^{(t)}/(1-\beta_1^t), \quad \hat{v}^{(t)} = v^{(t)}/(1-\beta_2^t)$$
$$\theta^{(t+1)} = \theta^{(t)} - \frac{\eta}{\sqrt{\hat{v}} + \epsilon}\hat{m}$$

기본값: $\beta_1=0.9, \beta_2=0.999, \eta=10^{-3}, \epsilon=10^{-8}$.

### 4.3 PyTorch 학습 루프 (page_440)
```python
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
```

### 4.4 표본평균 수렴 (slide ~370 page_410)

$\bar{X}_n$의 표준오차: $\sigma/\sqrt{n}$.
- 95% 신뢰구간 ≈ $\bar{X} \pm 1.96 \sigma/\sqrt{n} \approx \pm 2 \sigma/\sqrt{n}$.

→ "샘플 더 모으면 $\sqrt{n}$ 비율로 정확해진다."

---

## 5. 신경망 기초 (page_310–500)

### 5.1 Perceptron (Rosenblatt 1958, slide 316 page_357)
$$y = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$$
- S→A→R 토폴로지 (Sensors → Association → Response)
- 학습 가능 가중치는 A→R (Rosenblatt 가정)

### 5.2 Universal Approximation Theorem (Cybenko 1989)
1개 은닉층 + 비선형 활성화로 임의 연속함수 근사:
$$g(\mathbf{x}) = \sum_{i=1}^N \alpha_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)$$
$\sup_{\mathbf{x}} |f(\mathbf{x}) - g(\mathbf{x})| < \epsilon$.

**한계:** 존재성만 보장 — 효율성, 학습 가능성은 별개.

### 5.3 활성화 함수 (slide 338 page_338)

| 함수 | 정의 | 미분 |
|------|-----|-----|
| Sigmoid σ | $1/(1+e^{-x})$ | $\sigma(1-\sigma)$ |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $1 - \tanh^2$ |
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ |
| Leaky ReLU | (위) | 1 또는 α |
| GELU | $x \Phi(x)$ | $\Phi(x) + x\phi(x)$ |
| Softplus | $\log(1+e^x)$ | $\sigma(x)$ |

**Sigmoid 미분 유도:** $e^{-x} = (1+e^{-x}) - 1$ 분해 → $\sigma'(x) = \sigma(x) - \sigma(x)^2 = \sigma(1-\sigma)$.

### 5.4 Backpropagation (4단계 식)

층 $l$: $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$, $a^{(l)} = \sigma(z^{(l)})$.

1. **출력층:** $\delta^{(L)} = \nabla_{a^{(L)}} L \odot \sigma'(z^{(L)})$
2. **역전파:** $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$
3. **가중치:** $\partial L/\partial W^{(l)} = \delta^{(l)} (a^{(l-1)})^T$
4. **편향:** $\partial L/\partial b^{(l)} = \delta^{(l)}$

**Softmax + CE 그래디언트:** $\partial L/\partial z_j = p_j - y_j$ ("예측 - 정답").

### 5.5 Softmax 자코비안

$p_i = e^{z_i}/\sum_j e^{z_j}$일 때:
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$
$$J = \frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### 5.6 Initialization

**Xavier (sigmoid/tanh):** $\text{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$
**He (ReLU):** $\text{Var}(w) = 2/n_{\text{in}}$ (절반 죽임 보정)

### 5.7 Vanishing/Exploding Gradient

깊은 망에서 $\delta^{(l)}$이 지수적으로 곱해지며 $\sigma' < 1$ 누적 → 0. 해결책:
- ReLU 활성화
- Xavier/He 초기화
- Batch/Layer Normalization
- Residual Connection (ResNet)
- LSTM 게이트

### 5.8 Grandmother Cell (slide 492 page_540)
"Jennifer Aniston 뉴런" — 한 뉴런이 한 개념을 표현한다는 가설. 분산 표현(distributed representation) 논의의 출발점.

---

## 6. 일반화 이론 (page_440–510)

### 6.1 Bias-Variance Tradeoff (slide ~395 page_435 부근)

테스트점 $x_0$의 예측 오차:
$$E[(y_0 - \hat{f})^2] = \underbrace{(\bar{f} - f)^2}_{\text{Bias}^2} + \underbrace{E[(\bar{f} - \hat{f})^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

**유도 트릭:** $\bar{f}$ 더하고 빼기, 교차항 3개 모두 0.

### 6.2 Train vs Test Error (slide ~400 page_460)
- 모델 복잡도 ↑ → Training error ↓
- 일정 지점부터 Test error ↑ (overfitting)

### 6.3 Double Descent (slide ~395 page_435)

전통적 U곡선과 달리, over-parameterized 영역에서 test error가 **다시 감소**:
```
오차 ↑   /\         /
        /  \      /
       /    \   /
     _/      \_/
    under  inter  over
```
**이유:** 과대 매개변수 모델은 implicit regularization으로 매끄러운 해를 찾음.

### 6.4 Overfitting 통념 도전 (slide 422 page_470)
> "통계·ML 교과서에서 모든 훈련 예제에 완벽히 맞는 estimate는 종종 overfitting의 예시로 제시된다... However, **this is not true**."

→ 현대 딥러닝은 보간(interpolation)에 도달해도 일반화 잘 됨.

### 6.5 Sharpness (slide 458)
> "Sharpness는 generalization과 잘 상관되지 않는다."
- Flat minima vs sharp minima 논쟁
- Out-of-distribution 오차에서는 sharp가 더 나을 수도 (반전)

### 6.6 Regularization 종류
- **L2 (weight decay):** $+\lambda \|\mathbf{w}\|^2$, Gauss prior 등가
- **L1:** $+\lambda \|\mathbf{w}\|_1$, 희소성, Laplace prior 등가
- **Dropout:** 학습 시 뉴런 무작위 제거
- **Batch Normalization:** 미니배치 정규화
- **Data Augmentation:** 입력 다양성

---

## 7. CNN / Computer Vision (page_480–540)

### 7.1 합성곱 (Convolution)

$$(I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n)$$

- Filter/Kernel: 학습 가능한 가중치
- **Parameter sharing**: 같은 필터를 모든 위치에 적용 → 평행이동 등변성

### 7.2 Pooling, Stride, Padding
- **Max Pooling:** 영역 내 최댓값 → 공간 축소 + 작은 이동에 강건
- **Stride:** 필터 이동 간격
- **Padding:** 경계 0 채우기 (출력 크기 유지)

### 7.3 Receptive Field
한 출력 뉴런이 보는 입력 영역. 깊어질수록 넓어짐.

### 7.4 응용 (slide 490 page_490)
- **Classification:** ImageNet, ResNet
- **Object Detection:** YOLO, R-CNN
- **Segmentation:** U-Net, Mask R-CNN

---

## 8. RNN / Sequence Models (page_510–540)

### 8.1 RNN 기본
$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)$$
$$y_t = W_{hy} h_t$$

**유형 (slide 530):**
- Seq2Vec (분류)
- Vec2Seq (캡션 생성)
- Seq2Seq (번역)

### 8.2 LSTM 게이트
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t]) \quad (\text{forget})$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t]) \quad (\text{input})$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t]) \quad (\text{output})$$
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t])$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

**해결한 문제:** Vanishing gradient — cell state $c_t$가 곱이 아닌 덧셈으로 전파.

### 8.3 BPTT (Backpropagation Through Time)
RNN을 시간에 따라 펼친 다음 backprop. $h_t$의 그래디언트가 모든 이전 시점 의존.

### 8.4 Image Captioning (slide 520)
CNN으로 이미지 → 특징 벡터 → LSTM 디코더 → 캡션 단어 생성.

---

## 9. Attention과 Transformer (page_540–580)

### 9.1 Self-Attention (Q, K, V)

입력 $X \in \mathbb{R}^{N \times d}$:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

### 9.2 Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**왜 $\sqrt{d_k}$?** $QK^T$의 분산이 $d_k$ → softmax 입력이 너무 커지지 않도록 정규화.

### 9.3 Multi-Head Attention (slide 539)
$$X' = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$
$$\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$$

각 head가 다른 관계 학습. 일반적으로 $h=8$.

### 9.4 Masked MHA (Decoder, slide 552 page_552)
$$[QK^T]_{ij} = \begin{cases} q_i^T k_j & j \leq i \\ -\infty & j > i \end{cases}$$
$\Rightarrow A_{ij} = \begin{cases} \cdots & j \leq i \\ 0 & j > i \end{cases}$ (자동회귀)

### 9.5 Transformer 블록 (slide 532 page_580)
- Input/Output Embedding
- Positional Encoding
- Multi-Head Attention
- Add & Norm (Residual + LayerNorm)
- Feed Forward (2-layer MLP)
- **Decoder:** Masked MHA + Cross Attention

---

## 10. 현대 LLM·Vision 응용 (page_580–620)

### 10.1 Transfer Learning vs Domain Adaptation (page_640)

| | Transfer Learning | Domain Adaptation |
|---|---|---|
| 입력 | $\mathcal{X}_s \approx \mathcal{X}_t$ | $\mathcal{X}_s \neq \mathcal{X}_t$ |
| 라벨 | $\mathcal{Y}_s \neq \mathcal{Y}_t$ | $\mathcal{Y}_s = \mathcal{Y}_t$ |

**Domain Adversarial Learning:**
$$\min_{\theta, \theta_y} \max_{\theta_d} \hat{E}[-L_d(\cdots) + L_y(\cdots)]$$
- Feature extractor가 도메인 분류기를 속이도록 학습 → 도메인 불변 feature.

### 10.2 CLIP (slide 612 page_660)

이미지·텍스트 contrastive pre-training:
$$L_{ij} = \cos(I_i, T_j) = \frac{f_I(x_i)^T f_T(y_j)}{\|f_I\|\|f_T\|}$$
$$\mathcal{L} = \sum_i -\log \text{softmax}(L_{i,:})_i + \sum_j -\log \text{softmax}(L_{:,j})_j$$

400M 이미지-텍스트 쌍으로 사전학습 → zero-shot 분류.

### 10.3 LLM 동향 (slide 570 page_570)
- Chain of Thought (CoT)
- 비전-언어 모델 (VLM)
- Scaling Laws
- RLHF, Instruction Tuning

---

## 11. 생성 모델 (page_620–700)

### 11.1 GAN (Goodfellow 2014, slide 620 page_620)

**Minimax 목적:**
$$\min_G \max_D V(D, G) = E_{x \sim p_{\text{data}}}[\log D(x)] + E_{z \sim p_z}[\log(1 - D(G(z)))]$$

- Generator G: 노이즈 z → 가짜 데이터
- Discriminator D: 진짜/가짜 분류
- 균형: $D^* = \frac{p_{\text{data}}}{p_{\text{data}} + p_g}$, 최적해에서 $p_g = p_{\text{data}}$.

### 11.2 VAE (slide 646 page_646)

**구조:** Encoder $q_\phi(z|x) \to z$, Decoder $p_\theta(x|z) \to x'$.

**ELBO 유도:**
$$\log p_\theta(x) \geq E_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$
$$\equiv \text{ELBO}(\phi, \theta; x)$$

- 첫 항: 재구성 손실 (likelihood)
- 둘째 항: KL 정규화 (latent를 prior에 가깝게)

**Reparameterization Trick:**
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim N(0, I)$$
→ $z$를 stochastic node 대신 deterministic (gradient 통과 가능).

### 11.3 Diffusion Model (slide ~590 page_652, 700)

**Forward process** (data → noise):
$$q(z_t | z_{t-1}) = N(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t I)$$
$$z_T \sim N(0, I)$$

**Reverse process** (noise → data, 학습 목표):
$$p_\theta(z_{t-1} | z_t) = N(z_{t-1}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))$$

**ELBO 유도:** VAE와 비슷, 시간에 걸쳐 KL 합:
$$L = \sum_t E_q [\text{KL}(q(z_{t-1}|z_t, x) \| p_\theta(z_{t-1}|z_t))] + \cdots$$

**핵심 단순화 (DDPM):** Noise 예측 신경망 $\epsilon_\theta(z_t, t)$로 학습.

---

## 12. 핵심 정리·증명 색인

### 10대 핵심 증명 (`final-fire/10-ten-proofs/` 참조)
1. **Rank-Nullity:** $\dim(\mathcal{R}) + \dim(\mathcal{N}) = n$
2. **고유값 독립:** 서로 다른 λ ⇒ 고유벡터 선형독립
3. **베이즈 정리:** $p(\theta|D) = p(D|\theta)p(\theta)/p(D)$
4. **Softmax 자코비안:** $J = \text{diag}(p) - pp^T$
5. **가우스 적분:** $\int e^{-x^2/2} = \sqrt{2\pi}$ (극좌표)
6. **Jensen 부등식:** 볼록 ⇒ $f(E[X]) \leq E[f(X)]$
7. **KL ≥ 0:** Jensen + log 오목
8. **Cauchy-Schwarz:** $|⟨u,v⟩| \leq \|u\|\|v\|$ (판별식 ≤ 0)
9. **Spectral 정리:** 대칭 ⇒ 직교 고유벡터 기저
10. **MLE 통일 7단계:** 모든 분포 동일 패턴

### 핵심 등가성 표
| 분포 가정 | NLL 단순화 결과 |
|---------|---------------|
| Gauss noise | MSE 손실 |
| Bernoulli output | BCE 손실 |
| Categorical output | Cross Entropy |
| Laplace noise | MAE 손실 |
| Gauss prior on w | L2 정규화 |
| Laplace prior on w | L1 정규화 |

---

## 13. 약어·기호 사전

| 기호 | 의미 |
|------|-----|
| $\sum, \prod$ | 합, 곱 |
| $\int, \partial$ | 적분, 편미분 |
| $\propto$ | 비례 |
| $\sim$ | "~분포를 따른다" |
| $\hat{\theta}$ | 추정값 |
| $\delta_{ij}$ | 크로네커 델타 |
| $\nabla$ | 그래디언트 |
| $\odot$ | 원소별 곱 (Hadamard) |
| $\|\cdot\|, \|\cdot\|_F$ | 노름, Frobenius 노름 |
| MLE/MAP | 최대우도/최대사후 추정 |
| NLL | 음의 로그우도 |
| BCE/CE | (Binary) Cross Entropy |
| ELBO | Evidence Lower Bound |
| KL | Kullback-Leibler divergence |
| MHA | Multi-Head Attention |
| BPTT | Backprop Through Time |
| UAT | Universal Approximation Theorem |

---

## 14. 강의가 가르치고자 하는 통합 메시지 (재강조)

1. **수학적 기초 → ML 기초 → 딥러닝 → 생성 모델**의 일관된 흐름
2. 모든 손실 함수는 **분포 가정**에서 유도된다 (NLL이 통일 도구)
3. 모든 정규화는 **prior**에서 유도된다 (MAP 관점)
4. 모든 학습은 **i.i.d → 로그 → 미분=0** 체인으로 풀린다
5. 신경망은 **선형대수(층) + 미적분(학습) + 확률(손실)**의 조합
6. 일반화는 **Bias-Variance**로 분해되지만 현대 딥러닝은 이를 초월(Double Descent)
7. Attention은 분류뿐 아니라 **시퀀스 모델링의 표준**
8. 생성 모델은 **likelihood / ELBO / minimax** 세 가지 관점

이 8가지 통합 메시지를 답안 어디에든 인용할 수 있어야 한다.

---

## 15. 학습 자료 매핑 (`final-fire/` 폴더와의 대응)

| 본 가이드 섹션 | final-fire 참조 |
|--------------|---------------|
| §1 수학 기초 | `00-prerequisites/` |
| §2 추정 (MLE, MAP, NLL) | `04-mle-bernoulli/`, `05-07-map-*/`, `09-killer-chains/07-nll-master.md` |
| §2.6 KL/CE | `09-killer-chains/05-bernoulli-to-ce.md`, `11-extra-topics/04-information-theory.md` |
| §2.7 Gauss→MSE | `09-killer-chains/04-gaussian-to-mse.md` |
| §2.8 MAP→L2 | `09-killer-chains/06-map-to-l2.md` |
| §4 최적화 | `11-extra-topics/07-optimization.md` |
| §5 신경망 | `11-extra-topics/01-activations.md`, `02-backpropagation.md`, `08-initialization.md` |
| §5.5 Softmax 미분 | `08-softmax/`, `10-ten-proofs/04-softmax-derivative.md` |
| §6 일반화 | `11-extra-topics/03-bias-variance.md` |
| §10대 증명 | `10-ten-proofs/` 전체 |
| §13 학습 전략 | `99-strategy/` |

이 마스터 가이드는 **과목 전체의 큰 그림**을, `final-fire/`는 **시험 답안 작성**을 보완한다.

---

**작성:** 2026-04-26
**기반 슬라이드:** 742장 (한양대 이성윤 교수, 2025-11-28 update)
**다음:** [`LEARNING-MAP.md`](./LEARNING-MAP.md) — 학습 순서 + Mermaid 관계도

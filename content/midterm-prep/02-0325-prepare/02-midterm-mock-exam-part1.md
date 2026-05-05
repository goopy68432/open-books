---
title: "Deep Learning Theory - Midterm Mock Exam (Part 1: Questions 1-10)"
slug: midterm-mock-exam-part1
order: 2
---

# Deep Learning Theory - Midterm Mock Exam (Part 1: Questions 1-10)

> **과목**: 딥러닝 이론 (Deep Learning Theory)
> **범위**: 선형대수, 미적분, 확률/통계, 최적화 기초
> **출제 스타일**: 증명/유도 중심, 과정 평가, "Show that", "Derive", "Explain why"
> **작성일**: 2026-03-25

---

## 문제 1 (Cross-Entropy Loss & Inner Product)

### [EN] Problem Statement

Let $\mathbf{e}_y = [0, 1, 0]$ be the one-hot encoded label vector for a 3-class classification problem, where the true class is $y = 2$ (0-indexed: class 1). Let the model output after softmax be $\mathbf{h}(x) = [0.1, 0.7, 0.2]$.

**(a)** Compute the cross-entropy loss for this single sample:

$$L = -\mathbf{e}_y^\top \log \mathbf{h}(x)$$

Show all steps explicitly.

**(b)** Explain mathematically why the inner product $\mathbf{e}_y^\top \log \mathbf{h}(x)$ selects only the log-probability of the correct class. Generalize this to the $C$-class case.

**(c)** What happens to the CE loss as $h_y(x) \to 0$? As $h_y(x) \to 1$? Explain the intuition.

### [KR] 문제

$\mathbf{e}_y = [0, 1, 0]$을 3-클래스 분류 문제의 원-핫 인코딩 레이블 벡터라 하자 (정답 클래스: $y = 2$, 0-인덱스 기준 클래스 1). 소프트맥스 이후 모델 출력이 $\mathbf{h}(x) = [0.1, 0.7, 0.2]$이다.

**(a)** 이 단일 샘플에 대한 크로스 엔트로피 손실을 계산하라:

$$L = -\mathbf{e}_y^\top \log \mathbf{h}(x)$$

모든 단계를 명시적으로 보여라.

**(b)** 내적 $\mathbf{e}_y^\top \log \mathbf{h}(x)$가 왜 정답 클래스의 로그 확률만을 선택하는지 수학적으로 설명하라. 이를 $C$-클래스 경우로 일반화하라.

**(c)** $h_y(x) \to 0$일 때와 $h_y(x) \to 1$일 때 CE 손실은 어떻게 되는가? 직관적으로 설명하라.

### 출제 의도

이 문제는 딥러닝에서 가장 기본이 되는 손실 함수인 크로스 엔트로피(CE)의 수학적 구조를 이해하는지 평가한다. 단순히 "CE 공식을 안다"가 아니라, **왜 원-핫 벡터와의 내적이 정답 클래스만 골라내는지**, 그 메커니즘을 벡터 연산 관점에서 설명할 수 있는지를 본다. 이는 분류 모델의 학습 원리를 이해하는 첫걸음이며, 이후 소프트맥스 미분, 역전파 등으로 연결되는 핵심 개념이다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**먼저 등장하는 기호들을 하나씩 알아보자.**

- **벡터**: 숫자들을 나란히 적어놓은 것이다. 예를 들어 $[0, 1, 0]$은 세 칸짜리 벡터.
- **원-핫(one-hot) 벡터**: 딱 한 칸만 1이고 나머지는 모두 0인 벡터. "정답이 뭐냐"를 알려주는 역할.
  - $\mathbf{e}_y = [0, 1, 0]$ → "정답은 두 번째 칸(클래스 1)이야!"
- **모델 출력 $\mathbf{h}(x)$**: 모델이 "각 클래스일 확률이 얼마나 되나" 추측한 값.
  - $[0.1, 0.7, 0.2]$ → "클래스 0일 확률 10%, 클래스 1일 확률 70%, 클래스 2일 확률 20%"
- **$\log$**: 여기서는 자연로그(ln)를 의미. "큰 수는 크게, 작은 수는 아주 작게(마이너스로)" 바꿔주는 함수.
- **내적(inner product)**: 같은 위치의 숫자끼리 곱한 뒤 다 더하는 것.

---

**(a) 크로스 엔트로피 손실 계산하기**

**1단계: $\log \mathbf{h}(x)$ 계산**

각 원소에 로그를 씌운다:

$$\log \mathbf{h}(x) = [\log(0.1),\ \log(0.7),\ \log(0.2)]$$

계산기를 두드려보면:
- $\log(0.1) = \ln(0.1) \approx -2.3026$
- $\log(0.7) = \ln(0.7) \approx -0.3567$
- $\log(0.2) = \ln(0.2) \approx -1.6094$

따라서:

$$\log \mathbf{h}(x) \approx [-2.3026,\ -0.3567,\ -1.6094]$$

**2단계: 내적 계산 ($\mathbf{e}_y^\top \log \mathbf{h}(x)$)**

내적이란? 같은 자리끼리 곱해서 더하는 것!

$$\mathbf{e}_y^\top \log \mathbf{h}(x) = 0 \times (-2.3026) + 1 \times (-0.3567) + 0 \times (-1.6094)$$

하나씩 계산:
- 첫 번째 칸: $0 \times (-2.3026) = 0$
- 두 번째 칸: $1 \times (-0.3567) = -0.3567$
- 세 번째 칸: $0 \times (-1.6094) = 0$

합계: $0 + (-0.3567) + 0 = -0.3567$

**3단계: 앞에 마이너스 붙이기**

$$L = -\mathbf{e}_y^\top \log \mathbf{h}(x) = -(-0.3567) = 0.3567$$

$$\boxed{L \approx 0.3567}$$

---

**(b) 왜 정답 클래스만 골라지는가?**

비유로 설명하자. 원-핫 벡터는 **"스포트라이트"** 같은 것이다.

세 명이 무대에 서 있다고 상상해보자:
- 1번 자리: 클래스 0의 로그 확률 = $-2.3026$
- 2번 자리: 클래스 1의 로그 확률 = $-0.3567$ ← 정답!
- 3번 자리: 클래스 2의 로그 확률 = $-1.6094$

원-핫 벡터 $[0, 1, 0]$은 "2번 자리에만 스포트라이트를 켜라!"라는 명령이다.

내적할 때:
- 0이 곱해진 자리 → 아무리 큰 값이어도 0이 됨 (스포트라이트 꺼짐)
- 1이 곱해진 자리 → 그 값이 그대로 살아남음 (스포트라이트 켜짐)

그래서 내적 결과는 **정답 클래스의 로그 확률 하나만** 남게 된다!

---

**(c) 극단적인 경우**

- $h_y(x) \to 0$ (모델이 정답 확률을 0%에 가깝게 예측):
  - $\log(0^+) \to -\infty$ 이므로 $L \to +\infty$
  - 의미: "완전히 틀렸으니 벌(손실)이 무한대!"

- $h_y(x) \to 1$ (모델이 정답 확률을 100%에 가깝게 예측):
  - $\log(1) = 0$ 이므로 $L \to 0$
  - 의미: "완벽하게 맞췄으니 벌이 0!"

#### 🟡 Level 2: 고등학생 눈높이

**(a) CE 손실 계산**

$$L = -\mathbf{e}_y^\top \log \mathbf{h}(x) = -\sum_{c=0}^{C-1} (e_y)_c \cdot \log h_c(x)$$

$\mathbf{e}_y = [0,1,0]$이므로 $(e_y)_c$는 $c=1$일 때만 1, 나머지는 0:

$$L = -(0 \cdot \log 0.1 + 1 \cdot \log 0.7 + 0 \cdot \log 0.2) = -\log 0.7 = -\ln 0.7 \approx 0.3567$$

**(b) 내적의 선택 효과**

일반적으로 $C$-클래스에서 원-핫 벡터 $\mathbf{e}_y$는:

$$(e_y)_c = \begin{cases} 1 & \text{if } c = y \\ 0 & \text{if } c \neq y \end{cases}$$

이를 크로네커 델타 $\delta_{cy}$로 쓸 수 있다. 따라서:

$$\mathbf{e}_y^\top \log \mathbf{h}(x) = \sum_{c=0}^{C-1} \delta_{cy} \cdot \log h_c(x) = \log h_y(x)$$

합 안에서 $\delta_{cy} = 1$인 항은 $c = y$ 하나뿐이므로, 전체 합이 $\log h_y(x)$ 하나로 축약된다.

$$\therefore\ L = -\log h_y(x)$$

이것이 **Negative Log-Likelihood (NLL)** 와 동치인 이유이기도 하다.

**(c) 극한 분석**

$\log$ 함수의 성질:
- $\lim_{p \to 0^+} (-\log p) = +\infty$: 모델이 정답에 0 확률을 배정하면 손실이 발산 → 강한 페널티
- $\lim_{p \to 1} (-\log p) = 0$: 완벽한 예측 시 손실 0

이는 $-\log p$가 $(0, 1]$에서 단조감소하는 볼록(convex) 함수이기 때문이다.

#### 🔴 Level 3: 대학생 눈높이

**(a)** 계산은 자명하다: $L = -\mathbf{e}_y^\top \log \mathbf{h}(x) = -\log h_y(x) = -\ln 0.7 \approx 0.3567$.

**(b)** 원-핫 벡터는 표준 기저 벡터 $\mathbf{e}_y \in \mathbb{R}^C$이다. 내적 $\langle \mathbf{e}_y, \mathbf{v} \rangle = v_y$는 벡터 $\mathbf{v}$의 $y$-번째 좌표를 추출하는 **projection(사영)**이다. 따라서:

$$L_{\text{CE}}(\mathbf{h}, y) = -\langle \mathbf{e}_y, \log \mathbf{h} \rangle = -\log h_y$$

이는 정보이론적으로 true distribution $p = \mathbf{e}_y$ (degenerate)와 predicted distribution $q = \mathbf{h}(x)$ 사이의 크로스 엔트로피 $H(p, q) = -\sum_c p_c \log q_c$와 정확히 일치한다.

**(c)** $-\log(\cdot)$는 proper scoring rule이며 $h_y \in (0, 1]$에서 strictly convex하다. $h_y \to 0$이면 loss가 발산하여 모델이 정답 클래스에 0 확률을 배정하는 것을 강력히 억제한다. 이 성질은 수치적으로 log-sum-exp trick으로 다루며, label smoothing은 이 극단적 gradient를 완화하는 정규화 기법이다.

---

## 문제 2 (Eigenvalue Computation)

### [EN] Problem Statement

Given the matrix:

$$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

**(a)** Find all eigenvalues of $A$.

**(b)** For each eigenvalue, find the corresponding eigenvector.

**(c)** Verify your answer by explicitly computing $A\mathbf{v} = \lambda \mathbf{v}$ for each eigenpair.

**(d)** Is $A$ diagonalizable? If so, write $A = PDP^{-1}$.

### [KR] 문제

다음 행렬이 주어졌다:

$$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

**(a)** $A$의 모든 고유값(eigenvalue)을 구하라.

**(b)** 각 고유값에 대응하는 고유벡터(eigenvector)를 구하라.

**(c)** 각 고유값-고유벡터 쌍에 대해 $A\mathbf{v} = \lambda \mathbf{v}$를 직접 계산하여 답을 검증하라.

**(d)** $A$는 대각화 가능한가? 가능하다면 $A = PDP^{-1}$을 써라.

### 출제 의도

고유값/고유벡터는 딥러닝의 여러 곳에서 등장한다: PCA(주성분 분석), 헤시안 행렬의 고유값 분석(학습률 결정, saddle point 판별), 순환 신경망의 기울기 폭발/소실 분석 등. 이 문제는 고유값 분해의 **계산 과정**을 완벽히 수행할 수 있는지, 그리고 결과를 **검증하는 습관**이 있는지를 평가한다. 상삼각 행렬이므로 고유값은 대각 원소이지만, 그 이유를 설명할 수 있어야 한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**기호 설명부터 시작!**

- **행렬(matrix)**: 숫자를 직사각형으로 배열한 것. $A$는 2행 2열(2x2) 행렬.
- **고유값(eigenvalue) $\lambda$**: 행렬 $A$가 어떤 벡터 $\mathbf{v}$에 작용했을 때, 그 벡터의 **방향은 안 바뀌고 크기만 $\lambda$배 되는** 특별한 숫자.
- **고유벡터(eigenvector) $\mathbf{v}$**: 위의 조건을 만족하는 특별한 벡터. 즉, $A\mathbf{v} = \lambda \mathbf{v}$.

비유: 바람이 부는 방향이 있다고 하자. 대부분의 깃발은 바람에 의해 방향이 바뀌지만, 바람과 **같은 방향**을 가리키는 깃발은 방향이 안 바뀌고 세기만 바뀐다. 이 깃발의 방향이 고유벡터, 바람의 세기가 고유값!

---

**(a) 고유값 구하기**

핵심 공식: $\det(A - \lambda I) = 0$

여기서 $I$는 단위행렬 $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$이고, $\det$는 행렬식이다.

**1단계: $A - \lambda I$ 계산**

$$A - \lambda I = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix} - \lambda \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 - \lambda & 1 \\ 0 & 2 - \lambda \end{pmatrix}$$

**2단계: 행렬식(determinant) 계산**

2x2 행렬 $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$의 행렬식 = $ad - bc$.

$$\det(A - \lambda I) = (3 - \lambda)(2 - \lambda) - (1)(0)$$
$$= (3 - \lambda)(2 - \lambda) - 0$$
$$= (3 - \lambda)(2 - \lambda)$$

전개:
$$= 3 \times 2 - 3 \times \lambda - \lambda \times 2 + \lambda \times \lambda$$
$$= 6 - 3\lambda - 2\lambda + \lambda^2$$
$$= \lambda^2 - 5\lambda + 6$$

**3단계: $= 0$으로 놓고 풀기**

$$\lambda^2 - 5\lambda + 6 = 0$$

인수분해: $(\lambda - 3)(\lambda - 2) = 0$

$$\boxed{\lambda_1 = 3,\quad \lambda_2 = 2}$$

> **관찰**: 고유값이 대각 원소 3과 2 그 자체다! 이것은 우연이 아니다. 상삼각행렬(대각선 아래가 모두 0인 행렬)의 고유값은 항상 대각 원소이다. 왜냐하면 $\det(A - \lambda I)$에서 아래 삼각 부분의 0 때문에 행렬식이 대각 원소의 곱 $(3-\lambda)(2-\lambda)$이 되기 때문이다.

---

**(b) 고유벡터 구하기**

**$\lambda_1 = 3$일 때:**

$(A - 3I)\mathbf{v} = \mathbf{0}$을 풀자.

$$A - 3I = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}$$

$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}$로 놓으면:

$$\begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

행렬 곱을 풀면:
- 1행: $0 \cdot v_1 + 1 \cdot v_2 = 0$ → $v_2 = 0$
- 2행: $0 \cdot v_1 + (-1) \cdot v_2 = 0$ → $v_2 = 0$ (같은 조건)

$v_1$은 아무 값이나 가능! $v_1 = 1$로 놓으면:

$$\boxed{\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}}$$

**$\lambda_2 = 2$일 때:**

$(A - 2I)\mathbf{v} = \mathbf{0}$을 풀자.

$$A - 2I = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$$

$$\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

- 1행: $v_1 + v_2 = 0$ → $v_1 = -v_2$
- 2행: $0 = 0$ (항상 참, 조건 없음)

$v_2 = 1$로 놓으면 $v_1 = -1$:

$$\boxed{\mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}}$$

---

**(c) 검증**

**$\lambda_1 = 3$, $\mathbf{v}_1 = (1, 0)^\top$ 검증:**

좌변: $A\mathbf{v}_1 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \cdot 1 + 1 \cdot 0 \\ 0 \cdot 1 + 2 \cdot 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix}$

우변: $\lambda_1 \mathbf{v}_1 = 3 \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 3 \\ 0 \end{pmatrix}$

좌변 = 우변 ✓

**$\lambda_2 = 2$, $\mathbf{v}_2 = (-1, 1)^\top$ 검증:**

좌변: $A\mathbf{v}_2 = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} -1 \\ 1 \end{pmatrix} = \begin{pmatrix} 3(-1) + 1(1) \\ 0(-1) + 2(1) \end{pmatrix} = \begin{pmatrix} -2 \\ 2 \end{pmatrix}$

우변: $\lambda_2 \mathbf{v}_2 = 2\begin{pmatrix} -1 \\ 1 \end{pmatrix} = \begin{pmatrix} -2 \\ 2 \end{pmatrix}$

좌변 = 우변 ✓

---

**(d) 대각화**

고유벡터가 2개이고 서로 독립이므로 대각화 가능!

$$P = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix},\quad D = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$$

$$A = PDP^{-1}$$

#### 🟡 Level 2: 고등학생 눈높이

**(a)** $A$가 상삼각행렬이므로 특성방정식은:

$$\det(A - \lambda I) = (3-\lambda)(2-\lambda) = 0 \implies \lambda_1 = 3,\ \lambda_2 = 2$$

**(b)**

$\lambda_1 = 3$: $\text{null}(A - 3I) = \text{null}\begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}$. 조건: $v_2 = 0$. 기저: $\mathbf{v}_1 = (1, 0)^\top$.

$\lambda_2 = 2$: $\text{null}(A - 2I) = \text{null}\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$. 조건: $v_1 = -v_2$. 기저: $\mathbf{v}_2 = (-1, 1)^\top$.

**(c)** 직접 행렬-벡터 곱으로 $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$ 확인. (위 Level 1 참조)

**(d)** 서로 다른 고유값 2개 → 고유벡터가 선형독립 → 대각화 가능.

$$A = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}^{-1}$$

$P^{-1} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$이므로 검증: $PDP^{-1} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix} = A$ ✓

#### 🔴 Level 3: 대학생 눈높이

상삼각행렬의 고유값은 대각 원소 $\{3, 2\}$. 서로 다른 고유값이므로 자동으로 대각화 가능하다.

$$\text{Eigenspace}(\lambda=3) = \text{span}\{(1,0)^\top\},\quad \text{Eigenspace}(\lambda=2) = \text{span}\{(-1,1)^\top\}$$

딥러닝 관점에서: 가중치 행렬 $W$의 고유값 크기가 1보다 크면 기울기 폭발(gradient explosion), 1보다 작으면 기울기 소실(gradient vanishing)이 발생한다. RNN에서 $W^t$의 행동은 $PD^tP^{-1}$로 분석되며, 이때 $D^t$의 대각 원소 $\lambda_i^t$가 시간 $t$에 따라 폭발/소실 여부를 결정한다.

---

## 문제 3 (Derivative from Definition)

### [EN] Problem Statement

Using the **limit definition of the derivative**:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

Derive that $\frac{d}{dx}(x^3) = 3x^2$.

Show **every algebraic step**, including the full expansion of $(x+h)^3$.

### [KR] 문제

**도함수의 극한 정의**를 사용하여:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

$\frac{d}{dx}(x^3) = 3x^2$임을 유도하라.

$(x+h)^3$의 완전한 전개를 포함하여 **모든 대수적 단계를 보여라**.

### 출제 의도

미분의 정의에서부터 출발하여 결과를 유도하는 능력을 평가한다. 딥러닝에서 역전파(backpropagation)는 본질적으로 연쇄 법칙을 통한 미분의 연속이다. 미분이 "기울기"라는 기하학적 의미를 가지며, 극한 과정을 통해 정확한 값을 얻는다는 것을 이해해야 한다. 이 문제는 또한 대수적 조작 능력과 극한의 기본 성질을 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**미분이 뭐야?**

어떤 함수의 그래프가 있다고 하자. 그래프 위의 한 점에서 접선(딱 한 점만 스치듯 지나가는 직선)의 기울기를 구하는 것이 미분이다.

기울기를 어떻게 구하나? 두 점을 잡아서 "(y 변화량) / (x 변화량)"을 구한 다음, 두 점을 점점 가까이 붙이면 된다!

- 한 점: $(x,\ f(x)) = (x,\ x^3)$
- 가까운 점: $(x+h,\ f(x+h)) = (x+h,\ (x+h)^3)$
- 두 점 사이 기울기: $\frac{(x+h)^3 - x^3}{h}$
- $h \to 0$으로 보내면? → 접선의 기울기 = 미분!

---

**자, 이제 계산하자!**

**1단계: $(x+h)^3$ 전개**

$(x+h)^3 = (x+h)(x+h)(x+h)$

먼저 $(x+h)(x+h)$부터:
$$x \cdot x + x \cdot h + h \cdot x + h \cdot h = x^2 + xh + xh + h^2 = x^2 + 2xh + h^2$$

이제 $(x^2 + 2xh + h^2)(x+h)$:
$$= x^2 \cdot x + x^2 \cdot h + 2xh \cdot x + 2xh \cdot h + h^2 \cdot x + h^2 \cdot h$$
$$= x^3 + x^2h + 2x^2h + 2xh^2 + xh^2 + h^3$$
$$= x^3 + 3x^2h + 3xh^2 + h^3$$

$$\boxed{(x+h)^3 = x^3 + 3x^2h + 3xh^2 + h^3}$$

**2단계: $f(x+h) - f(x)$ 계산**

$$f(x+h) - f(x) = (x+h)^3 - x^3$$
$$= (x^3 + 3x^2h + 3xh^2 + h^3) - x^3$$
$$= 3x^2h + 3xh^2 + h^3$$

**3단계: $h$로 나누기**

$$\frac{f(x+h) - f(x)}{h} = \frac{3x^2h + 3xh^2 + h^3}{h}$$

$h$가 공통 인수! 각 항에서 $h$를 하나씩 빼자:

$$= \frac{h(3x^2 + 3xh + h^2)}{h} = 3x^2 + 3xh + h^2$$

**4단계: $h \to 0$ 극한 취하기**

$$\lim_{h \to 0} (3x^2 + 3xh + h^2)$$

$h$가 0으로 가면:
- $3x^2$ → $3x^2$ (변화 없음, $h$가 없으니까)
- $3xh$ → $3x \cdot 0 = 0$
- $h^2$ → $0^2 = 0$

$$= 3x^2 + 0 + 0 = 3x^2$$

$$\boxed{\frac{d}{dx}(x^3) = 3x^2}$$

#### 🟡 Level 2: 고등학생 눈높이

$f(x) = x^3$에 대해:

$$f'(x) = \lim_{h \to 0} \frac{(x+h)^3 - x^3}{h}$$

이항정리에 의해 $(x+h)^3 = x^3 + 3x^2h + 3xh^2 + h^3$이므로:

$$f'(x) = \lim_{h \to 0} \frac{3x^2h + 3xh^2 + h^3}{h} = \lim_{h \to 0}(3x^2 + 3xh + h^2) = 3x^2$$

일반화: 같은 방법으로 $f(x) = x^n$에 대해 $(x+h)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} h^k$를 사용하면:

$$\frac{(x+h)^n - x^n}{h} = nx^{n-1} + \binom{n}{2}x^{n-2}h + \cdots + h^{n-1}$$

$h \to 0$이면 첫 항만 살아남아 $\frac{d}{dx}(x^n) = nx^{n-1}$.

#### 🔴 Level 3: 대학생 눈높이

이항정리를 적용하면 즉시:

$$\frac{(x+h)^3 - x^3}{h} = 3x^2 + O(h) \xrightarrow{h \to 0} 3x^2$$

더 일반적으로 power rule $\frac{d}{dx}x^n = nx^{n-1}$은 이항정리의 직접적 귀결이다. 딥러닝에서 이 결과는 다항식 활성화 함수의 미분이나, Taylor 전개를 통한 활성화 함수의 로컬 근사($\text{ReLU} \approx$ piecewise linear) 분석에 활용된다. 자동 미분(autograd)은 이러한 기본 미분 규칙들의 조합으로 작동한다.

---

## 문제 4 (Partial Derivative and Gradient)

### [EN] Problem Statement

Let $f(x, y) = x^2 y + 3xy^2 - 2x + 5$.

**(a)** Compute the partial derivatives $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$.

**(b)** Write the gradient vector $\nabla f(x, y)$.

**(c)** Evaluate the gradient at the point $(1, -1)$.

**(d)** Explain geometrically what the gradient vector represents. Why is it important in gradient descent?

### [KR] 문제

$f(x, y) = x^2 y + 3xy^2 - 2x + 5$로 정의하자.

**(a)** 편미분 $\frac{\partial f}{\partial x}$와 $\frac{\partial f}{\partial y}$를 구하라.

**(b)** 그래디언트 벡터 $\nabla f(x, y)$를 써라.

**(c)** 점 $(1, -1)$에서 그래디언트를 계산하라.

**(d)** 그래디언트 벡터가 기하학적으로 무엇을 의미하는지 설명하라. 왜 경사 하강법에서 중요한가?

### 출제 의도

편미분과 그래디언트는 딥러닝 학습 알고리즘의 핵심이다. 경사 하강법(Gradient Descent)은 $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$로 파라미터를 업데이트하는데, 이 그래디언트가 "어느 방향으로 가면 손실이 가장 빨리 증가하는가"를 알려주는 벡터라는 것을 이해해야 한다. 편미분 계산 능력 + 기하학적 해석 능력을 동시에 평가한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**편미분이 뭐야?**

보통 미분은 변수가 하나: $f(x) = x^2$이면 $f'(x) = 2x$.

그런데 $f(x, y)$처럼 변수가 두 개면? **하나만 변하게 하고 나머지는 고정!**

- $\frac{\partial f}{\partial x}$: "$y$는 그냥 숫자 취급하고 $x$로만 미분해라"
- $\frac{\partial f}{\partial y}$: "$x$는 그냥 숫자 취급하고 $y$로만 미분해라"

비유: 산에서 동서 방향 기울기와 남북 방향 기울기를 따로 재는 것!

---

**(a) $\frac{\partial f}{\partial x}$ 구하기**

$f(x, y) = x^2 y + 3xy^2 - 2x + 5$

$y$를 상수 취급하고 $x$로 미분한다. 각 항을 하나씩:

- $x^2 y$: $y$는 상수, $x^2$를 $x$로 미분 → $2x$. 결과: $2xy$
- $3xy^2$: $3y^2$는 상수, $x$를 $x$로 미분 → $1$. 결과: $3y^2$
- $-2x$: $x$로 미분 → $-2$
- $5$: 상수 미분 → $0$

$$\boxed{\frac{\partial f}{\partial x} = 2xy + 3y^2 - 2}$$

**$\frac{\partial f}{\partial y}$ 구하기**

$x$를 상수 취급하고 $y$로 미분:

- $x^2 y$: $x^2$는 상수, $y$를 $y$로 미분 → $1$. 결과: $x^2$
- $3xy^2$: $3x$는 상수, $y^2$를 $y$로 미분 → $2y$. 결과: $6xy$
- $-2x$: $y$가 없으므로 → $0$
- $5$: → $0$

$$\boxed{\frac{\partial f}{\partial y} = x^2 + 6xy}$$

---

**(b) 그래디언트 벡터**

그래디언트 = 편미분들을 벡터로 묶은 것:

$$\nabla f(x, y) = \begin{pmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{pmatrix} = \begin{pmatrix} 2xy + 3y^2 - 2 \\ x^2 + 6xy \end{pmatrix}$$

---

**(c) 점 $(1, -1)$에서 계산**

$x = 1$, $y = -1$을 대입:

$$\frac{\partial f}{\partial x}\bigg|_{(1,-1)} = 2(1)(-1) + 3(-1)^2 - 2 = -2 + 3 - 2 = -1$$

$$\frac{\partial f}{\partial y}\bigg|_{(1,-1)} = (1)^2 + 6(1)(-1) = 1 - 6 = -5$$

$$\boxed{\nabla f(1, -1) = \begin{pmatrix} -1 \\ -5 \end{pmatrix}}$$

---

**(d) 기하학적 의미**

산에서 서 있다고 상상하자. 그래디언트 벡터는:

1. **방향**: "이 방향으로 가면 가장 가파르게 올라간다" (가장 빠르게 $f$가 증가하는 방향)
2. **크기**: 얼마나 가파른지 (기울기의 세기)

경사 하강법에서는 **손실을 줄이고 싶으니까**, 그래디언트의 **반대 방향**으로 간다!

$$\theta_{새} = \theta_{현재} - \eta \cdot \nabla f$$

여기서 $\eta$는 학습률(한 번에 얼마나 갈지). 마이너스가 붙어있으니 가장 빠르게 내려가는 방향!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 다변수 함수의 편미분. 표준적인 계산:

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 - 2, \quad \frac{\partial f}{\partial y} = x^2 + 6xy$$

**(b)** $\nabla f = \left(\frac{\partial f}{\partial x},\ \frac{\partial f}{\partial y}\right)^\top$

**(c)** $(1, -1)$에서: $\nabla f = (-1, -5)^\top$.

$\|\nabla f\| = \sqrt{1 + 25} = \sqrt{26} \approx 5.10$이므로 이 점에서의 최대 증가율은 약 5.10이다.

**(d)** 방향 도함수(directional derivative)의 관점에서, 단위 벡터 $\mathbf{u}$ 방향으로의 변화율은:

$$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\theta$$

이 값은 $\theta = 0$ (즉, $\mathbf{u}$가 그래디언트 방향)일 때 최대. 따라서 **그래디언트 방향 = 함수 최대 증가 방향**.

경사 하강법은 $-\nabla f$ 방향(최대 감소 방향)으로 이동하여 손실을 최소화한다.

#### 🔴 Level 3: 대학생 눈높이

편미분 계산은 자명하다. 핵심은 기하학적 해석이다.

그래디언트 $\nabla f \in \mathbb{R}^n$은 함수 $f$의 level set $\{x : f(x) = c\}$에 **직교하는 벡터**이며, steepest ascent 방향을 가리킨다. 이는 일차 테일러 근사 $f(x + \delta) \approx f(x) + \nabla f^\top \delta$에서 $\|\delta\| = \epsilon$ 제약 하에 $\nabla f^\top \delta$를 최대화하는 $\delta$가 $\delta \propto \nabla f$임을 Cauchy-Schwarz 부등식으로 증명할 수 있다.

딥러닝에서의 함의: SGD, Adam 등 모든 일차(first-order) 최적화 알고리즘은 그래디언트 정보에 기반한다. 그래디언트가 0인 점(critical point)은 local min, local max, 또는 saddle point이며, 이를 구분하기 위해 Hessian(이차 미분)이 필요하다.

---

## 문제 5 (Conditional Probability & Bayes' Theorem)

### [EN] Problem Statement

In a dataset of emails:
- 40% are spam: $P(\text{Spam}) = 0.4$
- Among spam emails, 90% contain the word "free": $P(\text{Free} \mid \text{Spam}) = 0.9$
- Among non-spam emails, 5% contain "free": $P(\text{Free} \mid \text{Not Spam}) = 0.05$

**(a)** Compute $P(\text{Free})$, the overall probability that an email contains "free". Use the law of total probability.

**(b)** If an email contains "free", what is the probability it is spam? Derive using Bayes' theorem, showing every step.

**(c)** Explain intuitively why the answer is much higher than the base rate of 40%.

### [KR] 문제

이메일 데이터셋에서:
- 40%가 스팸: $P(\text{Spam}) = 0.4$
- 스팸 중 90%가 "free"라는 단어 포함: $P(\text{Free} \mid \text{Spam}) = 0.9$
- 정상 이메일 중 5%가 "free" 포함: $P(\text{Free} \mid \text{Not Spam}) = 0.05$

**(a)** 전체 확률의 법칙을 사용하여 "free"를 포함할 전체 확률 $P(\text{Free})$를 구하라.

**(b)** "free"를 포함하는 이메일이 스팸일 확률을 베이즈 정리로 유도하라. 모든 단계를 보여라.

**(c)** 답이 기저율 40%보다 훨씬 높은 이유를 직관적으로 설명하라.

### 출제 의도

베이즈 정리는 딥러닝과 머신러닝의 이론적 기반이다. MAP(Maximum A Posteriori) 추정, 베이지안 신경망, 나이브 베이즈 분류기 등에 직접 사용된다. "사전 확률에서 데이터(증거)를 반영하여 사후 확률을 업데이트한다"는 개념을 정확히 이해하고 계산할 수 있는지 평가한다. 또한 전체 확률의 법칙(law of total probability)이 베이즈 정리의 분모를 계산하는 데 필수적이라는 점을 알아야 한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**확률 기초부터!**

비유: 학교에 1000명의 학생이 있다고 하자.

- 400명이 스팸 이메일을 보내는 그룹 (40%)
- 600명이 정상 이메일을 보내는 그룹 (60%)

"free"라는 단어를 사용하는 사람:
- 스팸 그룹 400명 중 90% = $400 \times 0.9 = 360$명
- 정상 그룹 600명 중 5% = $600 \times 0.05 = 30$명

---

**(a) "free"를 포함하는 이메일의 전체 비율**

"free"를 쓰는 총 인원 = $360 + 30 = 390$명

$1000$명 중 $390$명:

$$P(\text{Free}) = \frac{390}{1000} = 0.39$$

수식으로 쓰면 (전체 확률의 법칙):

$$P(\text{Free}) = P(\text{Free} \mid \text{Spam}) \cdot P(\text{Spam}) + P(\text{Free} \mid \text{Not Spam}) \cdot P(\text{Not Spam})$$
$$= 0.9 \times 0.4 + 0.05 \times 0.6$$
$$= 0.36 + 0.03 = 0.39$$

---

**(b) "free"를 포함하는 이메일이 스팸일 확률**

"free"를 쓰는 390명 중에서 스팸인 사람은 360명:

$$P(\text{Spam} \mid \text{Free}) = \frac{360}{390} = \frac{36}{39} = \frac{12}{13} \approx 0.923$$

이것을 **베이즈 정리** 공식으로 쓰면:

$$P(\text{Spam} \mid \text{Free}) = \frac{P(\text{Free} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{Free})}$$

대입:

$$= \frac{0.9 \times 0.4}{0.39} = \frac{0.36}{0.39} \approx 0.923$$

$$\boxed{P(\text{Spam} \mid \text{Free}) \approx 92.3\%}$$

---

**(c) 왜 40%에서 92%로 뛰었을까?**

핵심: "free"라는 단어는 **스팸의 강력한 신호**이기 때문!

- 스팸 중 "free" 사용: 90% (매우 높음)
- 정상 중 "free" 사용: 5% (매우 낮음)

이 격차가 크면 클수록, "free"를 봤을 때 "이건 스팸이다!"라는 확신이 강해진다.

쉬운 비유: 기침하는 사람이 감기일 확률. 평소 감기 걸릴 확률이 10%여도, 감기 환자의 95%가 기침하고 건강한 사람은 5%만 기침한다면, 기침하는 사람을 보면 "감기겠구나" 하고 생각하게 되는 것과 같다!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 전체 확률의 법칙 (Law of Total Probability):

$$P(F) = P(F|S)P(S) + P(F|S^c)P(S^c) = 0.9 \times 0.4 + 0.05 \times 0.6 = 0.39$$

**(b)** 베이즈 정리:

$$P(S|F) = \frac{P(F|S)P(S)}{P(F)} = \frac{0.36}{0.39} = \frac{12}{13} \approx 92.3\%$$

**(c)** 우도비(likelihood ratio) 관점:

$$\frac{P(F|S)}{P(F|S^c)} = \frac{0.9}{0.05} = 18$$

"free"가 나타나면 스팸일 가능성이 비스팸일 가능성의 18배. 사전 오즈(prior odds) $\frac{P(S)}{P(S^c)} = \frac{0.4}{0.6} = \frac{2}{3}$에 우도비를 곱하면:

$$\text{사후 오즈} = \frac{2}{3} \times 18 = 12$$

$$P(S|F) = \frac{12}{12 + 1} = \frac{12}{13} \approx 92.3\%$$

#### 🔴 Level 3: 대학생 눈높이

베이즈 정리의 로그 오즈(log-odds) 형태:

$$\log \frac{P(S|F)}{P(S^c|F)} = \log \frac{P(S)}{P(S^c)} + \log \frac{P(F|S)}{P(F|S^c)}$$

$$= \log \frac{2}{3} + \log 18 = -0.405 + 2.890 = 2.485$$

이 형태는 로지스틱 회귀(logistic regression)의 선형 모델과 정확히 대응한다: 각 feature의 log-likelihood ratio가 log-odds에 **가산적(additive)으로** 기여한다. 나이브 베이즈 분류기는 이 구조를 features의 조건부 독립 가정 하에 일반화한 것이다. 딥러닝의 softmax 출력층도 본질적으로 log-odds를 모델링한다.

---

## 문제 6 (MLE for Bernoulli Distribution)

### [EN] Problem Statement

You flip a (possibly biased) coin $n$ times independently. You observe $k$ heads and $(n-k)$ tails. Let $p$ be the probability of heads.

**(a)** Write the likelihood function $L(p)$ for the observed data.

**(b)** Write the log-likelihood function $\ell(p) = \log L(p)$.

**(c)** Derive the MLE estimate $\hat{p}$ by differentiating $\ell(p)$, setting the derivative to zero, and solving.

**(d)** Verify that this is indeed a maximum (not a minimum) using the second derivative test.

**(e)** Why is MLE important for training neural networks?

### [KR] 문제

(편향될 수 있는) 동전을 독립적으로 $n$번 던진다. $k$번 앞면, $(n-k)$번 뒷면이 나왔다. $p$를 앞면 확률이라 하자.

**(a)** 관측된 데이터에 대한 우도 함수(likelihood function) $L(p)$를 써라.

**(b)** 로그 우도 함수 $\ell(p) = \log L(p)$를 써라.

**(c)** $\ell(p)$를 미분하고 0으로 놓아 MLE 추정량 $\hat{p}$를 유도하라.

**(d)** 이차 도함수 판정법으로 이것이 최대인지(최소가 아닌지) 확인하라.

**(e)** MLE가 신경망 학습에 왜 중요한지 설명하라.

### 출제 의도

MLE(최대우도추정)는 딥러닝 학습의 이론적 토대이다. 신경망의 손실 함수(CE loss, MSE loss)는 모두 특정 확률 분포 가정 하에서의 **음의 로그 우도(negative log-likelihood)**와 동치이다. 이 문제는 가장 단순한 분포(베르누이)에서 MLE 유도 과정을 완전히 수행할 수 있는지 확인한다. "왜 log를 취하는가", "왜 곱이 합이 되는가"를 이해하는 것이 핵심.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**우도(Likelihood)가 뭐야?**

동전을 5번 던져서 앞앞뒤앞뒤(HHTHT)가 나왔다고 하자.

만약 이 동전의 앞면 확률이 $p$라면, 이 결과가 나올 확률은 얼마일까?

- 앞(H): 확률 $p$
- 앞(H): 확률 $p$
- 뒤(T): 확률 $(1-p)$
- 앞(H): 확률 $p$
- 뒤(T): 확률 $(1-p)$

각 던지기가 독립이므로 전체 확률 = 다 곱하기:

$$p \times p \times (1-p) \times p \times (1-p) = p^3 \times (1-p)^2$$

이것이 **우도 함수**: "만약 $p$가 이 값이라면, 관측 데이터가 나올 확률은 얼마인가?"

---

**(a) 우도 함수**

$n$번 중 $k$번 앞면:

$$L(p) = p^k \cdot (1-p)^{n-k}$$

(앞면 $k$번 → $p$를 $k$번 곱하고, 뒷면 $(n-k)$번 → $(1-p)$를 $(n-k)$번 곱한다)

---

**(b) 로그 우도**

왜 로그를 취할까? **곱셈을 덧셈으로** 바꿔주니까 미분이 쉬워진다!

$$\ell(p) = \log L(p) = \log(p^k \cdot (1-p)^{n-k})$$

로그의 성질: $\log(A \times B) = \log A + \log B$, $\log(A^n) = n \log A$

$$\ell(p) = k \log p + (n-k) \log(1-p)$$

---

**(c) MLE 유도**

"$\ell(p)$를 최대로 만드는 $p$는?" → 미분해서 0으로 놓자!

$$\frac{d\ell}{dp} = \frac{k}{p} + (n-k) \cdot \frac{-1}{1-p}$$

(왜? $\frac{d}{dp}\log p = \frac{1}{p}$이고, $\frac{d}{dp}\log(1-p) = \frac{1}{1-p} \cdot (-1) = \frac{-1}{1-p}$)

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

이제 풀자:

$$\frac{k}{p} = \frac{n-k}{1-p}$$

양변에 $p(1-p)$를 곱하면:

$$k(1-p) = (n-k)p$$

전개:

$$k - kp = np - kp$$

양변에서 $-kp$가 소거:

$$k = np$$

$$\boxed{\hat{p} = \frac{k}{n}}$$

직관적으로 당연하다: 10번 던져서 3번 앞면이면 $\hat{p} = 3/10 = 0.3$!

---

**(d) 최대인지 확인**

이차 도함수를 구하자:

$$\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2}$$

$p \in (0, 1)$이고 $k \geq 0$, $n - k \geq 0$이므로:

- $-\frac{k}{p^2} \leq 0$
- $-\frac{n-k}{(1-p)^2} \leq 0$

합도 $\leq 0$! 이차 도함수가 음수 → **아래로 볼록** → 극대점이 맞다! ✓

---

**(e) 신경망과의 연결**

신경망을 학습시킬 때 하는 일 = "손실 함수를 최소화"
- 크로스 엔트로피 손실 최소화 = **로그 우도 최대화**와 같은 말!
- 즉, 신경망 학습은 MLE를 하는 것이다!

더 정확히: 분류 문제에서 소프트맥스 출력에 CE 손실을 쓰는 것은, 카테고리컬 분포를 가정한 MLE이다.

#### 🟡 Level 2: 고등학생 눈높이

**(a)** i.i.d. 베르누이 시행의 결합 확률:

$$L(p) = \prod_{i=1}^n p^{x_i}(1-p)^{1-x_i} = p^k(1-p)^{n-k}$$

**(b)** $\ell(p) = k\ln p + (n-k)\ln(1-p)$

**(c)** Score function:

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0 \implies k(1-p) = (n-k)p \implies \hat{p} = \frac{k}{n}$$

**(d)** $\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0$ for $0 < k < n$. 로그 우도 함수가 강오목(strictly concave)이므로 유일한 전역 최대.

**(e)** 신경망의 파라미터 $\theta$에 대해 데이터 $\mathcal{D} = \{(x_i, y_i)\}$가 주어졌을 때:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_i \log p(y_i | x_i; \theta)$$

이것에 $-1$을 곱하면 NLL 손실의 최소화가 된다. 분류(카테고리컬 분포) → CE 손실, 회귀(가우시안 분포) → MSE 손실.

#### 🔴 Level 3: 대학생 눈높이

베르누이 MLE $\hat{p} = k/n$은 지수족(exponential family)의 충분통계량(sufficient statistic)에 기반한 MLE의 특수한 경우이다.

Fisher 정보량: $I(p) = -\mathbb{E}\left[\frac{d^2\ell}{dp^2}\right] = \frac{n}{p(1-p)}$.

MLE의 점근적 성질: $\hat{p} \xrightarrow{d} \mathcal{N}\left(p, \frac{1}{I(p)}\right) = \mathcal{N}\left(p, \frac{p(1-p)}{n}\right)$.

딥러닝 관점: SGD로 NLL을 최소화하는 것이 근사적 MLE이다. 배치 크기, 학습률, 정규화(weight decay = MAP with Gaussian prior)는 모두 이 MLE 프레임워크의 변형으로 이해할 수 있다. KL divergence $D_{\text{KL}}(p_{\text{data}} \| p_\theta)$의 최소화와 MLE는 동치이다.

---

## 문제 7 (Singular Value Decomposition)

### [EN] Problem Statement

**(a)** Explain what the Singular Value Decomposition (SVD) $A = U\Sigma V^\top$ represents. Specifically describe the role of $U$, $\Sigma$, and $V$.

**(b)** Let $A$ be a $3 \times 2$ matrix with singular values $\sigma_1 = 5$ and $\sigma_2 = 1$. Write $A = U\Sigma V^\top$ in terms of the column vectors $\mathbf{u}_1, \mathbf{u}_2$ and $\mathbf{v}_1, \mathbf{v}_2$.

**(c)** Construct the best rank-1 approximation $A_1$ of $A$. Write its formula. Explain why this minimizes $\|A - A_1\|_F$ among all rank-1 matrices.

**(d)** What is the Frobenius norm of the approximation error $\|A - A_1\|_F$?

**(e)** How does this relate to image compression or dimensionality reduction in deep learning?

### [KR] 문제

**(a)** 특이값 분해(SVD) $A = U\Sigma V^\top$가 무엇을 의미하는지 설명하라. $U$, $\Sigma$, $V$ 각각의 역할을 서술하라.

**(b)** $A$가 특이값 $\sigma_1 = 5$, $\sigma_2 = 1$을 가진 $3 \times 2$ 행렬이라 하자. 열벡터 $\mathbf{u}_1, \mathbf{u}_2$와 $\mathbf{v}_1, \mathbf{v}_2$를 사용하여 $A = U\Sigma V^\top$를 써라.

**(c)** $A$의 최적 랭크-1 근사 $A_1$을 구성하라. 공식을 써라. 이것이 왜 모든 랭크-1 행렬 중 $\|A - A_1\|_F$를 최소화하는지 설명하라.

**(d)** 근사 오차의 프로베니우스 노름 $\|A - A_1\|_F$는 얼마인가?

**(e)** 이것이 딥러닝의 이미지 압축 또는 차원 축소와 어떻게 관련되는가?

### 출제 의도

SVD는 선형대수의 핵심 분해이며, 딥러닝에서 차원 축소(PCA의 기반), 가중치 행렬 압축(LoRA의 이론적 기반), 추천 시스템(잠재 요인 모델), 수치 안정성 분석 등에 광범위하게 사용된다. "왜 가장 큰 특이값부터 취하면 최적 근사가 되는가?"라는 Eckart-Young 정리의 직관을 이해하는 것이 핵심이다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**SVD가 뭔데?**

행렬은 "변환(transform)"을 의미한다. 어떤 도형에 행렬을 곱하면 도형이 늘어나거나 회전하거나 한다.

SVD는 이 변환을 **3단계로 분해**하는 것이다:

$$A = U \Sigma V^\top$$

비유: 춤 동작을 분해하는 것과 같다!
1. $V^\top$ (첫 번째 회전): "먼저 이 방향으로 돌아서"
2. $\Sigma$ (늘이기/줄이기): "이 방향으로 이만큼 늘리고, 저 방향으로 이만큼 줄여"
3. $U$ (두 번째 회전): "마지막으로 이 방향으로 돌아"

각 부분의 역할:
- $V$: 입력 공간의 특별한 방향들 (정규직교 행렬, $V^\top V = I$)
- $\Sigma$: 각 방향으로 얼마나 늘리는지 (대각행렬, 특이값 $\sigma_1 \geq \sigma_2 \geq \cdots$)
- $U$: 출력 공간의 특별한 방향들 (정규직교 행렬, $U^\top U = I$)

---

**(b) 외적 형태로 쓰기**

$A$는 $3 \times 2$이므로:

$$A = U\Sigma V^\top = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top + \sigma_2 \mathbf{u}_2 \mathbf{v}_2^\top$$

$$= 5 \cdot \mathbf{u}_1 \mathbf{v}_1^\top + 1 \cdot \mathbf{u}_2 \mathbf{v}_2^\top$$

여기서 $\mathbf{u}_i \mathbf{v}_i^\top$는 **랭크-1 행렬** (외적, outer product)이다.

이것이 무슨 뜻인가? "행렬 $A$는 두 장의 겹쳐진 사진이다."
- 첫 번째 사진: $\mathbf{u}_1 \mathbf{v}_1^\top$ (밝기 = 5, 매우 중요!)
- 두 번째 사진: $\mathbf{u}_2 \mathbf{v}_2^\top$ (밝기 = 1, 덜 중요)

---

**(c) 최적 랭크-1 근사**

가장 "밝은" 사진 하나만 남기면:

$$\boxed{A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top = 5 \cdot \mathbf{u}_1 \mathbf{v}_1^\top}$$

왜 이것이 최적인가? Eckart-Young 정리에 의해, $A$에서 가장 큰 특이값에 대응하는 성분을 남기면 오차가 최소가 된다. 직관적으로, 가장 "중요한"(크기가 큰) 성분부터 취하는 것이 효율적이다.

---

**(d) 오차**

버린 부분의 특이값들의 제곱합의 제곱근:

$$\|A - A_1\|_F = \sqrt{\sigma_2^2} = \sigma_2 = 1$$

일반적으로 랭크-$r$ 근사의 오차: $\|A - A_r\|_F = \sqrt{\sigma_{r+1}^2 + \sigma_{r+2}^2 + \cdots}$

---

**(e) 딥러닝 연결**

이미지 압축: 이미지를 행렬로 보고 SVD를 적용하면, 큰 특이값 몇 개만 남겨도 원본과 비슷한 이미지를 얻을 수 있다. 저장 공간 절약!

- 원래: $m \times n$개의 숫자를 저장
- 랭크-$r$ 근사: $r(m + n + 1)$개만 저장 ($\mathbf{u}_i, \mathbf{v}_i, \sigma_i$ 각각)

#### 🟡 Level 2: 고등학생 눈높이

**(a)** SVD: 임의의 $m \times n$ 행렬 $A$에 대해 $A = U\Sigma V^\top$, 여기서:
- $U \in \mathbb{R}^{m \times m}$: 좌특이벡터, 정규직교 ($U^\top U = I$)
- $\Sigma \in \mathbb{R}^{m \times n}$: 특이값 대각행렬, $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: 우특이벡터, 정규직교 ($V^\top V = I$)

기하학적으로 $A$는 $V^\top$(회전) → $\Sigma$(스케일) → $U$(회전) 변환의 합성.

**(b)** 외적 분해: $A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ ($r = \text{rank}(A) \leq 2$).

**(c)** Eckart-Young-Mirsky 정리: 랭크-$k$ 행렬 중 프로베니우스 노름으로 $A$에 가장 가까운 것은 $A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$이다.

직관: 각 $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$는 "에너지" $\sigma_i^2$를 가진다. 에너지가 가장 큰 것부터 남기면 보존 에너지가 최대.

**(d)** $\|A - A_1\|_F^2 = \sigma_2^2 = 1$이므로 $\|A - A_1\|_F = 1$.

보존율: $\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2} = \frac{25}{26} \approx 96.2\%$

**(e)** PCA는 데이터 공분산 행렬의 SVD이며, 딥러닝에서 LoRA(Low-Rank Adaptation)는 대규모 사전학습 모델의 가중치 업데이트를 $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$)로 저랭크 근사하여 효율적으로 파인튜닝한다.

#### 🔴 Level 3: 대학생 눈높이

SVD는 행렬의 정준 분해(canonical decomposition)이다. $A^\top A$의 고유벡터 = $V$의 열, $AA^\top$의 고유벡터 = $U$의 열, 특이값 = $\sqrt{\lambda_i(A^\top A)}$.

Eckart-Young 정리의 증명 스케치: $\|A - B\|_F^2$에서 $B$가 랭크-$k$이면, $B$의 열공간이 $A$의 상위 $k$개 좌특이벡터가 span하는 부분공간과 일치할 때 최적. 이는 Courant-Fischer minimax 정리의 귀결.

딥러닝에서의 활용:
- **Weight compression**: SVD로 큰 FC layer를 두 개의 작은 layer로 분해 ($W \approx U_r \Sigma_r V_r^\top$)
- **LoRA**: $W + \Delta W$에서 $\Delta W = BA$로 저랭크 업데이트. SVD가 최적 저랭크 근사를 보장하므로 이론적 기반 제공
- **Spectral norm regularization**: $\sigma_1(W)$을 제한하여 Lipschitz 상수 제어 → GAN 안정화 (Spectral Normalization)

---

## 문제 8 (Chain Rule and Backpropagation)

### [EN] Problem Statement

Let $y = f(g(x))$ where $g(x) = 3x^2 + 1$ and $f(u) = \ln(u)$.

**(a)** Compute $\frac{dy}{dx}$ using the chain rule. Show each step clearly.

**(b)** Verify your answer by first computing $y = \ln(3x^2 + 1)$ and differentiating directly.

**(c)** Draw a computation graph for this function. Label each node with its value and derivative.

**(d)** Explain how the chain rule is the mathematical foundation of backpropagation in neural networks. Use this specific example to illustrate forward pass and backward pass.

### [KR] 문제

$y = f(g(x))$이고, $g(x) = 3x^2 + 1$, $f(u) = \ln(u)$이다.

**(a)** 연쇄 법칙(chain rule)을 사용하여 $\frac{dy}{dx}$를 구하라. 각 단계를 명확히 보여라.

**(b)** $y = \ln(3x^2 + 1)$을 직접 미분하여 답을 검증하라.

**(c)** 이 함수의 계산 그래프(computation graph)를 그려라. 각 노드에 값과 미분을 표시하라.

**(d)** 연쇄 법칙이 신경망의 역전파(backpropagation)의 수학적 기반인 이유를 설명하라. 이 예제를 사용하여 순전파(forward pass)와 역전파(backward pass)를 설명하라.

### 출제 의도

연쇄 법칙(Chain Rule)은 역전파 알고리즘의 수학적 핵심이다. 딥러닝 모델은 수백 개의 함수가 합성된 것이며, 각 파라미터에 대한 손실의 기울기를 구하려면 연쇄 법칙을 반복 적용해야 한다. 이 문제는 단순한 2-함수 합성에서 연쇄 법칙을 적용하고, 이를 계산 그래프와 역전파로 확장하는 사고 과정을 평가한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**연쇄 법칙(Chain Rule)이 뭐야?**

비유: 기차가 "서울 → 대전 → 부산"으로 간다고 하자.
- 서울→대전 속도: 시속 100km
- 대전→부산 속도: 시속 50km

전체 변화율은? 각 구간의 변화율을 **곱하는** 것!

수학으로 쓰면: $y$가 $u$의 함수이고, $u$가 $x$의 함수이면:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

"$y$가 $x$에 대해 얼마나 변하나" = "$y$가 $u$에 대해 얼마나 변하나" × "$u$가 $x$에 대해 얼마나 변하나"

---

**(a) 연쇄 법칙으로 $\frac{dy}{dx}$ 구하기**

**1단계: 각 부분의 미분 구하기**

$u = g(x) = 3x^2 + 1$이므로:
$$\frac{du}{dx} = 6x$$
($3x^2$를 $x$로 미분하면 $6x$, 상수 $1$은 사라짐)

$y = f(u) = \ln(u)$이므로:
$$\frac{dy}{du} = \frac{1}{u}$$

**2단계: 곱하기!**

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = \frac{1}{u} \cdot 6x = \frac{6x}{u}$$

$u = 3x^2 + 1$을 대입:

$$\boxed{\frac{dy}{dx} = \frac{6x}{3x^2 + 1}}$$

---

**(b) 직접 미분으로 검증**

$y = \ln(3x^2 + 1)$

"밖의 함수 미분 × 안의 함수 미분":
$$\frac{dy}{dx} = \frac{1}{3x^2 + 1} \cdot 6x = \frac{6x}{3x^2 + 1}$$

(a)와 같은 결과! ✓

---

**(c) 계산 그래프**

$x = 2$일 때를 예로 들어보자:

```
[순전파 (Forward Pass): 왼쪽 → 오른쪽]

x = 2  →  u = g(x) = 3(2)² + 1 = 13  →  y = ln(13) ≈ 2.565

[역전파 (Backward Pass): 오른쪽 → 왼쪽]

dy/dy = 1  ←  dy/du = 1/13  ←  du/dx = 12

dy/dx = dy/du × du/dx = (1/13) × 12 = 12/13 ≈ 0.923
```

더 자세히:

```
노드:        x=2      →      u=13      →      y=2.565
                g(x)=3x²+1        f(u)=ln(u)

순전파:       2    ───→      13     ───→     2.565
역전파:    12/13   ←───     1/13    ←───      1
           (dy/dx)         (dy/du)          (dy/dy)
```

---

**(d) 역전파와의 연결**

신경망은 이런 구조이다:

$$\text{입력} \xrightarrow{\text{층 1}} \xrightarrow{\text{층 2}} \cdots \xrightarrow{\text{층 L}} \text{손실}$$

각 층은 하나의 함수이다. 전체 신경망은 함수의 합성:

$$L = f_L(f_{L-1}(\cdots f_2(f_1(x; \theta_1); \theta_2) \cdots; \theta_L))$$

파라미터 $\theta_i$에 대한 손실의 기울기를 구하려면 연쇄 법칙!

$$\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_{i+1}}{\partial f_i} \cdot \frac{\partial f_i}{\partial \theta_i}$$

**순전파**: $x$부터 시작해서 $y$(손실)까지 값을 계산 (앞에서 뒤로)
**역전파**: $y$부터 시작해서 $x$(파라미터)까지 기울기를 계산 (뒤에서 앞으로)

위의 예에서:
- 순전파: $x=2$ → $u=13$ → $y=2.565$
- 역전파: $\frac{dy}{dy}=1$ → $\frac{dy}{du}=\frac{1}{13}$ → $\frac{dy}{dx}=\frac{12}{13}$

역전파가 효율적인 이유: 각 노드에서 **로컬 기울기**만 계산하고, 뒤에서 오는 기울기와 곱하면 된다. 전체를 한 번에 미분할 필요가 없다!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 합성 함수의 미분:

$$\frac{dy}{dx} = \frac{df}{du}\bigg|_{u=g(x)} \cdot \frac{dg}{dx} = \frac{1}{3x^2+1} \cdot 6x = \frac{6x}{3x^2+1}$$

**(b)** 동일한 결과 확인.

**(c)** 계산 그래프에서 각 노드는 (값, 로컬 기울기)를 저장:
- $x$ 노드: 값 $x$
- $u$ 노드: 값 $3x^2+1$, 로컬 기울기 $\frac{du}{dx} = 6x$
- $y$ 노드: 값 $\ln u$, 로컬 기울기 $\frac{dy}{du} = \frac{1}{u}$

역전파: 출력에서 입력으로, 로컬 기울기를 체인으로 곱해나간다.

**(d)** 다변수 연쇄 법칙: $z = f(g(x, y), h(x, y))$인 경우:

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial g}\frac{\partial g}{\partial x} + \frac{\partial z}{\partial h}\frac{\partial h}{\partial x}$$

신경망에서 하나의 변수가 여러 경로로 출력에 영향을 주면, 각 경로의 기울기를 **합산**한다. 이것이 역전파에서 기울기가 분기점에서 더해지는 이유이다.

#### 🔴 Level 3: 대학생 눈높이

연쇄 법칙은 역전파의 수학적 등가물이며, 계산적으로는 reverse-mode automatic differentiation이다. 이의 시간 복잡도는 $O(1)$ × (forward pass 비용)으로, 파라미터 수에 무관하게 한 번의 backward pass로 모든 기울기를 계산할 수 있다.

이는 forward-mode AD ($O(n)$ × forward pass, $n$ = 파라미터 수)와 대조적이며, 딥러닝에서 파라미터 수가 수십억에 달하므로 reverse-mode가 필수적이다.

Jacobian $J_g \in \mathbb{R}^{m \times n}$과 $J_f \in \mathbb{R}^{p \times m}$에 대해:

$$J_{f \circ g} = J_f \cdot J_g$$

역전파는 이 곱을 **오른쪽에서 왼쪽으로** (vector-Jacobian product, VJP) 수행하여 효율성을 달성한다.

---

## 문제 9 (Convex Function)

### [EN] Problem Statement

**(a)** Using the definition of convexity:

$$f(\lambda a + (1-\lambda)b) \leq \lambda f(a) + (1-\lambda)f(b), \quad \forall \lambda \in [0, 1]$$

Prove that $f(x) = x^2$ is convex.

**(b)** Provide a geometric interpretation of this inequality.

**(c)** Explain why convexity is important for optimization in machine learning. What guarantee does convexity provide that general non-convex functions do not?

### [KR] 문제

**(a)** 볼록 함수의 정의를 사용하여:

$$f(\lambda a + (1-\lambda)b) \leq \lambda f(a) + (1-\lambda)f(b), \quad \forall \lambda \in [0, 1]$$

$f(x) = x^2$이 볼록(convex)임을 증명하라.

**(b)** 이 부등식의 기하학적 해석을 제시하라.

**(c)** 볼록성이 머신러닝 최적화에서 왜 중요한지 설명하라. 볼록 함수가 일반적인 비볼록 함수에 비해 보장하는 것은 무엇인가?

### 출제 의도

볼록성(convexity)은 최적화 이론의 핵심 개념이다. 볼록 함수에서는 모든 극소점이 전역 최소점이라는 보장이 있어, 경사 하강법이 전역 최적해에 수렴함을 보장할 수 있다. 딥러닝의 손실 함수는 일반적으로 비볼록이지만, 볼록 최적화의 개념과 도구(학습률 이론, 수렴 속도 분석 등)를 이해하는 것이 비볼록 문제를 다루는 출발점이다. 정의를 사용한 증명 능력을 직접 평가한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**볼록 함수가 뭐야?**

$y = x^2$ 그래프를 상상하자. U자 모양이다 (아래로 볼록한 그릇).

이 그릇 위의 아무 두 점을 잡고 직선(줄)으로 연결하면, 그 줄은 항상 그릇 **위에** 있거나 그릇에 닿는다. 절대 그릇 안쪽(아래)으로 파고들지 않는다.

이것을 수학으로 쓰면:
- 그릇 위의 두 점: $(a, a^2)$와 $(b, b^2)$
- 두 점을 잇는 줄 위의 점: $\lambda$로 섞기 → 높이 = $\lambda a^2 + (1-\lambda)b^2$
- 그릇 자체의 높이: $(\lambda a + (1-\lambda)b)^2$

볼록 = "줄의 높이 $\geq$ 그릇의 높이"

---

**(a) 증명**

보여야 할 것: $(\lambda a + (1-\lambda)b)^2 \leq \lambda a^2 + (1-\lambda)b^2$

**전략: 우변 - 좌변 $\geq 0$임을 보이자!**

$$\text{우변} - \text{좌변} = \lambda a^2 + (1-\lambda)b^2 - (\lambda a + (1-\lambda)b)^2$$

좌변의 제곱을 전개:

$$(\lambda a + (1-\lambda)b)^2 = \lambda^2 a^2 + 2\lambda(1-\lambda)ab + (1-\lambda)^2 b^2$$

대입:

$$= \lambda a^2 + (1-\lambda)b^2 - \lambda^2 a^2 - 2\lambda(1-\lambda)ab - (1-\lambda)^2 b^2$$

각 항을 정리:
- $a^2$ 항: $\lambda - \lambda^2 = \lambda(1 - \lambda)$
- $b^2$ 항: $(1-\lambda) - (1-\lambda)^2 = (1-\lambda)(1 - (1-\lambda)) = (1-\lambda)\lambda = \lambda(1-\lambda)$
- $ab$ 항: $-2\lambda(1-\lambda)$

$$= \lambda(1-\lambda)a^2 + \lambda(1-\lambda)b^2 - 2\lambda(1-\lambda)ab$$

$\lambda(1-\lambda)$로 묶으면:

$$= \lambda(1-\lambda)(a^2 - 2ab + b^2)$$

$$= \lambda(1-\lambda)(a - b)^2$$

이제 확인:
- $\lambda \in [0, 1]$이므로 $\lambda \geq 0$
- $1 - \lambda \geq 0$
- $(a - b)^2 \geq 0$ (제곱은 항상 0 이상!)

따라서:

$$\lambda(1-\lambda)(a-b)^2 \geq 0$$

$$\therefore\ \lambda f(a) + (1-\lambda)f(b) - f(\lambda a + (1-\lambda)b) \geq 0$$

$$\boxed{f(\lambda a + (1-\lambda)b) \leq \lambda f(a) + (1-\lambda)f(b) \quad \checkmark}$$

---

**(b) 기하학적 해석**

$y = x^2$ 그래프 위에 두 점 $(a, a^2)$와 $(b, b^2)$를 찍자.

이 두 점을 잇는 **직선(할선, secant line)** 위의 임의의 점은:

$$\text{직선의 높이} = \lambda a^2 + (1-\lambda)b^2$$

같은 $x$-좌표에서의 **곡선의 높이**는:

$$\text{곡선의 높이} = (\lambda a + (1-\lambda)b)^2$$

볼록 부등식은 "**곡선이 항상 직선 아래(또는 위에 닿음)**"을 뜻한다.

---

**(c) 머신러닝에서의 중요성**

볼록 함수에서 경사 하강법을 하면:
- **모든 극소점 = 전역 최소점**: 골짜기가 하나뿐! 어디서 출발해도 같은 곳에 도착.
- **수렴 보장**: 적절한 학습률이면 반드시 최적해에 도달.

비볼록 함수(실제 딥러닝 손실 함수)에서는:
- 극소점이 여러 개 → 어디에 빠질지 모름
- 안장점(saddle point) 존재 → 기울기 0이지만 최소점 아님
- 수렴 보장 없음

그래서 볼록 최적화를 먼저 이해하고, 비볼록 문제를 다루는 전략(학습률 스케줄링, 모멘텀, Adam 등)을 배우는 것이다!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 우변 - 좌변:

$$\lambda a^2 + (1-\lambda)b^2 - [\lambda a + (1-\lambda)b]^2 = \lambda(1-\lambda)(a-b)^2 \geq 0$$

$\lambda \in [0,1]$이므로 $\lambda(1-\lambda) \geq 0$이고 $(a-b)^2 \geq 0$.  $\square$

**대안적 증명 (이차 도함수 이용)**: $f''(x) = 2 > 0$ for all $x$. 이차 도함수가 항상 양수이면 볼록.

**(b)** Jensen의 부등식의 특수한 경우: 볼록 함수에서 "함수의 기댓값 $\geq$ 기댓값의 함수". 기하학적으로는 할선이 곡선 위에 위치.

**(c)** 볼록 최적화의 핵심 보장:
1. 모든 지역 최소 = 전역 최소
2. GD 수렴 속도: $L$-smooth convex → $O(1/T)$, $\mu$-strongly convex → $O(e^{-\mu T/L})$
3. 딥러닝 손실은 비볼록이지만, 최근 연구에서 overparameterized network의 손실 landscape는 "거의 볼록"에 가깝다는 결과들이 있음 (loss surface에 spurious local minima가 적음)

#### 🔴 Level 3: 대학생 눈높이

$f(x) = x^2$의 볼록성은 $f'' = 2 > 0$으로 즉시 확인된다. 정의를 사용한 증명은 $\lambda(1-\lambda)(a-b)^2 \geq 0$으로 귀결.

볼록성의 등가 조건들:
1. $f(\lambda a + (1-\lambda)b) \leq \lambda f(a) + (1-\lambda)f(b)$ (정의)
2. $f(y) \geq f(x) + \nabla f(x)^\top (y-x)$ (first-order condition: 접선 아래에 곡선 없음)
3. $\nabla^2 f(x) \succeq 0$ (second-order: Hessian이 양반정치)

딥러닝에서의 함의:
- 단일 뉴런 + MSE + 선형 활성화 → 볼록 (선형 회귀)
- ReLU 네트워크 → 비볼록이지만, 충분히 넓은(overparameterized) 네트워크는 NTK(Neural Tangent Kernel) regime에서 loss landscape가 거의 볼록
- Batch normalization, skip connection 등은 loss landscape를 더 "볼록에 가깝게" 만드는 효과가 있다는 경험적 증거가 있음

---

## 문제 10 (Gaussian Distribution and MSE Loss)

### [EN] Problem Statement

Assume that the target variable $y$ given input $x$ follows a Gaussian (Normal) distribution:

$$y \mid x \sim \mathcal{N}(h(x),\ \sigma^2)$$

where $h(x)$ is the model's prediction and $\sigma^2$ is a known, fixed variance.

**(a)** Write the probability density function (PDF) of $y$ given $x$.

**(b)** For a dataset $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ with i.i.d. samples, write the likelihood function $L(h)$.

**(c)** Write the log-likelihood function $\ell(h)$.

**(d)** Show that maximizing the log-likelihood is equivalent to minimizing the MSE loss:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - h(x_i))^2$$

Clearly indicate at each step where the Gaussian assumption is used.

### [KR] 문제

입력 $x$가 주어졌을 때 목표 변수 $y$가 가우시안(정규) 분포를 따른다고 가정하자:

$$y \mid x \sim \mathcal{N}(h(x),\ \sigma^2)$$

여기서 $h(x)$는 모델의 예측값이고 $\sigma^2$는 알려진 고정 분산이다.

**(a)** $y$의 조건부 확률밀도함수(PDF)를 써라.

**(b)** i.i.d. 샘플 $\{(x_1, y_1), \ldots, (x_n, y_n)\}$에 대한 우도 함수 $L(h)$를 써라.

**(c)** 로그 우도 함수 $\ell(h)$를 써라.

**(d)** 로그 우도를 최대화하는 것이 MSE 손실을 최소화하는 것과 동치임을 보여라. 각 단계에서 가우시안 가정이 어디에 사용되었는지 명시하라.

### 출제 의도

이 문제는 딥러닝에서 "왜 회귀 문제에 MSE 손실을 쓰는가?"에 대한 확률론적 정당성을 유도하는 것이다. 단순히 "MSE 공식을 안다"가 아니라, **가우시안 노이즈 가정 → 우도 함수 → 로그 → 상수 제거 → MSE**라는 인과 체인을 완전히 이해하고 있는지를 평가한다. 이는 문제 6(MLE for Bernoulli → CE loss와의 연결)과 짝을 이루는 문제로, "확률 분포 가정이 손실 함수를 결정한다"는 핵심 원리를 양쪽에서 확인한다.

### 풀이 (Solutions)

#### 🟢 Level 1: 중학생 눈높이

**가우시안(정규) 분포가 뭐야?**

"평균 근처에 데이터가 많고, 멀어질수록 데이터가 적은" 종 모양 분포.

예: 반 학생들의 키. 평균이 170cm이면 165~175cm 근처에 학생이 많고, 150cm이나 190cm은 드물다.

$\mathcal{N}(h(x), \sigma^2)$의 의미:
- $h(x)$ = 평균 (모델이 예측한 값)
- $\sigma^2$ = 분산 (데이터가 평균에서 얼마나 퍼지는지)
- "실제 값 $y$는 모델 예측 $h(x)$ 주변에서 종 모양으로 분포한다"

---

**(a) 확률밀도함수 (PDF)**

$$\text{[가우시안 가정 사용]} \quad p(y \mid x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - h(x))^2}{2\sigma^2}\right)$$

각 부분의 의미:
- $\frac{1}{\sqrt{2\pi\sigma^2}}$: 정규화 상수 (전체 확률이 1이 되도록)
- $\exp(-\cdots)$: 종 모양을 만드는 부분
- $(y - h(x))^2$: 실제값과 예측값의 차이의 제곱 → 멀수록 확률 작아짐!

---

**(b) 우도 함수**

데이터가 $n$개 있고, 각각 독립(i.i.d.)이면, 전체 우도 = 각 데이터의 확률을 다 곱하기:

$$L(h) = \prod_{i=1}^{n} p(y_i \mid x_i) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right)$$

(여기서 "독립" 가정이 곱셈을 가능하게 한다!)

---

**(c) 로그 우도**

로그를 취하면 곱셈이 덧셈으로!

$$\ell(h) = \log L(h) = \sum_{i=1}^{n} \log\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right)\right]$$

각 항을 분리:

$$= \sum_{i=1}^{n} \left[\log\frac{1}{\sqrt{2\pi\sigma^2}} + \log\exp\left(-\frac{(y_i - h(x_i))^2}{2\sigma^2}\right)\right]$$

$\log\exp(A) = A$ (로그와 지수는 역함수!):

$$= \sum_{i=1}^{n} \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - h(x_i))^2}{2\sigma^2}\right]$$

정리:

$$\ell(h) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - h(x_i))^2$$

---

**(d) 로그 우도 최대화 = MSE 최소화 증명!**

$\ell(h)$를 최대화하고 싶다. 식을 다시 보자:

$$\ell(h) = \underbrace{-\frac{n}{2}\log(2\pi\sigma^2)}_{\text{상수! } h\text{와 무관}} - \frac{1}{2\sigma^2}\underbrace{\sum_{i=1}^{n}(y_i - h(x_i))^2}_{\text{이 부분만 } h\text{에 의존}}$$

$h$를 조절해서 $\ell(h)$를 최대화하려면:
- 첫 번째 항은 상수이므로 무시
- 두 번째 항 앞에 마이너스가 있으니, $\sum(y_i - h(x_i))^2$를 **최소화**해야 전체가 **최대화**됨

$$\arg\max_h\ \ell(h) = \arg\min_h\ \sum_{i=1}^{n}(y_i - h(x_i))^2$$

양변을 $n$으로 나눠도 최적해는 동일:

$$= \arg\min_h\ \frac{1}{n}\sum_{i=1}^{n}(y_i - h(x_i))^2 = \arg\min_h\ \text{MSE}$$

$$\boxed{\text{가우시안 가정 하에서, MLE} = \text{MSE 최소화}}$$

**가우시안 가정이 사용된 곳 정리:**
1. **(a)에서**: PDF의 형태가 $\exp(-(y-\mu)^2/2\sigma^2)$ → 이것 때문에 $(y - h(x))^2$가 등장
2. **(b)에서**: i.i.d. 가정 → 곱셈이 가능
3. **(c)→(d)에서**: $\exp$에 로그를 취하면 제곱항이 남음 → 이것이 MSE의 출처

만약 가우시안이 아니라 **라플라스 분포**를 가정하면? $|y - h(x)|$가 남아서 **MAE (Mean Absolute Error)** 손실이 된다!

#### 🟡 Level 2: 고등학생 눈높이

**(a)** 가우시안 PDF: $p(y|x) = (2\pi\sigma^2)^{-1/2}\exp\left(-\frac{(y-h(x))^2}{2\sigma^2}\right)$

**(b)** i.i.d. 가정에 의한 결합 우도:

$$L(h) = \prod_{i=1}^n (2\pi\sigma^2)^{-1/2}\exp\left(-\frac{(y_i-h(x_i))^2}{2\sigma^2}\right)$$

**(c)** 로그 우도:

$$\ell(h) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - h(x_i))^2$$

**(d)** $h$에 대해 최적화할 때 상수항 $-\frac{n}{2}\ln(2\pi\sigma^2)$와 양의 스칼라 $\frac{1}{2\sigma^2}$는 argmax/argmin에 영향을 주지 않는다:

$$\hat{h}_{\text{MLE}} = \arg\max_h \ell(h) = \arg\min_h \sum_{i=1}^n (y_i - h(x_i))^2 = \arg\min_h\ n \cdot \text{MSE}$$

핵심: **가우시안의 지수 부분에 있는 $(y - \mu)^2$이 MSE를 낳는다.** 분포 가정을 바꾸면 손실 함수도 바뀐다:

| 분포 가정 | 지수/핵심 항 | 대응 손실 |
|-----------|-------------|-----------|
| 가우시안 $\mathcal{N}$ | $(y-\mu)^2$ | MSE |
| 라플라스 | $|y-\mu|$ | MAE |
| 카테고리컬(소프트맥스) | $\log p_y$ | Cross-Entropy |

#### 🔴 Level 3: 대학생 눈높이

가우시안 노이즈 모델 $y = h(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 하에서:

$$\hat{h}_{\text{MLE}} = \arg\max_h \ell(h) = \arg\min_h \frac{1}{2\sigma^2}\|y - h(x)\|_2^2$$

이는 최소제곱법(OLS)의 확률론적 정당화이다.

확장:
- **이분산성(heteroscedastic)**: $\sigma^2(x)$가 $x$에 의존하면, 손실에 가중치 $1/\sigma^2(x_i)$가 붙어 **가중 최소제곱법(WLS)**이 된다.
- **정규화(regularization)**: MLE 대신 MAP(Maximum A Posteriori)를 하면, $p(h) \propto \exp(-\lambda\|h\|^2)$ (가우시안 사전분포) 가정 시 $L_2$ 정규화(weight decay)가 도출된다.
- **Aleatoric vs Epistemic uncertainty**: $\sigma^2$를 모델이 출력하도록 학습시키면 aleatoric uncertainty를 추정할 수 있다. 이때 NLL은:

$$\ell = -\sum_i \left[\frac{(y_i - \mu_i)^2}{2\sigma_i^2} + \frac{1}{2}\ln\sigma_i^2\right]$$

이 경우 $\sigma_i^2$도 학습 가능하며, 단순 MSE와는 달리 불확실성이 큰 샘플의 영향을 줄인다.

---

> **End of Part 1 (Questions 1-10)**
> 다음: Part 2 (Questions 11-20)에서는 소프트맥스 미분, 행렬 곱의 미분, 정보 이론, 최적화 알고리즘 등을 다룹니다.

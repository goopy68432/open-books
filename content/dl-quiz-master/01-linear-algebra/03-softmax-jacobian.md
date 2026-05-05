---
title: 'Quiz 3 — Softmax Jacobian'
description: 'softmax의 Jacobian = diag(p) - pp^T 유도'
draft: false
---

## 0. 한 줄 요약

$p_i = e^{z_i}/\sum_k e^{z_k}$ 의 편미분은 케이스에 따라
$$
\frac{\partial p_i}{\partial z_j}=\begin{cases}p_i(1-p_i),& i=j\\ -p_i p_j,& i\neq j\end{cases}\qquad\Longleftrightarrow\qquad J=\frac{\partial p}{\partial z}=\mathrm{diag}(p)-p p^\top.
$$
유도 도구는 **몫의 미분 + 합 안에서 한 항만 살아남는 미분** 두 가지뿐. 이 결과는 분포 $p$ 의 공분산 행렬과 정확히 같으며, Quiz 10 의 backprop 결과 $\partial L/\partial z=p-e_y$ 의 기둥이다.

---

## 1. 문제 (출제 형태 그대로)

$z\in\mathbb{R}^n$ 에 대해 softmax 출력
$$
p_i \;=\; \frac{e^{z_i}}{\sum_{k=1}^{n} e^{z_k}},\qquad i=1,\ldots,n
$$
의 편미분 $\dfrac{\partial p_i}{\partial z_j}$ 를 모든 $(i,j)$ 쌍에 대해 구하라. 또한 그 결과를 행렬 (Jacobian) 형태로 정리하라.

---

## 2. 출제 의도와 시험 가치

Softmax 는 분류 신경망의 *출력층 표준* 함수이다. 학습은 backpropagation 으로 일어나고, backprop 의 첫 단계는 정확히 이 Jacobian. 이 문제는:

1. **분수형 합성함수**의 편미분을 정확히 다룰 수 있는가 (몫의 미분, 합 안의 단일 항).
2. **인덱스 분기 (case split)** 가 자연스럽게 발생하는 상황을 의식적으로 처리할 수 있는가 ($i=j$ vs $i\neq j$).
3. 결과를 *행렬 형태* 로 닫을 수 있는가 ($\mathrm{diag}(p)-pp^\top$).
4. 결과의 **정성적 성질** 을 읽을 수 있는가: 합 보존 ($\sum_i J_{ij}=0$), 대칭, 공분산.

또한 Quiz 10 (backprop) 에서 cross entropy 와 결합되어 결과가 $p-e_y$ 로 폭발적으로 단순화되는 *마법의 출발점* 이 본 퀴즈이다.

---

## 3. 사전 개념

### 3.1 기호 풀이

| 기호 | 의미 |
|---|---|
| $z=(z_1,\ldots,z_n)^\top$ | 신경망의 *logit* (raw score) 벡터 |
| $e^{z_i}$ 또는 $\exp(z_i)$ | 자연지수 — 항상 양수 |
| $S\equiv\sum_{k=1}^n e^{z_k}$ | softmax 의 분모 (정규화 합) |
| $p_i$ | softmax 출력의 $i$ 번째 성분, $0<p_i<1$, $\sum_i p_i=1$ |
| $\partial/\partial z_j$ | $z_j$ 에 대한 편미분 (다른 $z_k$ 는 상수 취급) |
| $\delta_{ij}$ | Kronecker delta: $i=j$ 면 1, 아니면 0 |
| $\mathrm{diag}(p)$ | 대각 성분이 $p_1,\ldots,p_n$ 인 $n\times n$ 대각행렬 |
| $pp^\top$ | 외적 (outer product). $(pp^\top)_{ij}=p_i p_j$. |

### 3.2 정의/공식 정리

**Def (Softmax).**
$$
p_i = \mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_k e^{z_k}},\qquad p=(p_1,\ldots,p_n)^\top.
$$
정규화 성질: $\sum_i p_i = \dfrac{\sum_i e^{z_i}}{\sum_k e^{z_k}}=1$, 모든 $p_i>0$.

**공식 (몫의 미분).**
$$
\frac{\partial}{\partial x}\left(\frac{u(x)}{v(x)}\right)=\frac{u'(x) v(x) - u(x) v'(x)}{v(x)^2}.
$$

**공식 (지수 미분).** $\frac{\partial}{\partial z_j}e^{z_i}=\begin{cases}e^{z_j},& i=j\\ 0,& i\neq j\end{cases}=\delta_{ij}\,e^{z_i}.$

**공식 (분모 미분).**
$$
\frac{\partial S}{\partial z_j}=\frac{\partial}{\partial z_j}\sum_{k} e^{z_k}=\sum_k \delta_{jk}\,e^{z_k}=e^{z_j}.
$$
*해설*: 합 안의 항 중 $z_j$ 를 포함하는 항은 *오직* $k=j$ 뿐이라 단일 항만 살아남는다.

---

## 4. 풀이 (모든 단계, 모든 등호 근거 명시)

표기 단순화: $u_i\equiv e^{z_i}$, $S\equiv\sum_k e^{z_k}$. 그러면 $p_i = u_i/S$.

또한
- $\partial u_i/\partial z_j = \delta_{ij}\,u_i$ (지수 미분).
- $\partial S/\partial z_j = u_j$ (분모 미분).

### 4.1 Case 1: $i=j$ (대각 성분)

**Step 1.** 몫의 미분 적용.
$$
\frac{\partial p_i}{\partial z_i}
= \frac{\partial}{\partial z_i}\frac{u_i}{S}
\stackrel{(\text{quotient})}{=}
\frac{\dfrac{\partial u_i}{\partial z_i}\cdot S \;-\; u_i\cdot \dfrac{\partial S}{\partial z_i}}{S^2}.
$$
*등호 근거*: 몫의 미분 공식.

**Step 2.** 항별 대입.
$$
= \frac{u_i\cdot S \;-\; u_i\cdot u_i}{S^2}
= \frac{u_i}{S} - \frac{u_i^2}{S^2}.
$$
*등호 근거*: $\partial u_i/\partial z_i=u_i$, $\partial S/\partial z_i=u_i$ (위 공식 두 줄).

**Step 3.** $u_i/S = p_i$ 로 묶기.
$$
= p_i - p_i^2 \;=\; \boxed{\,p_i(1-p_i)\,}.
$$
*등호 근거*: softmax 정의 $p_i = u_i/S$.

### 4.2 Case 2: $i\neq j$ (비대각 성분)

**Step 1.** 몫의 미분.
$$
\frac{\partial p_i}{\partial z_j}
= \frac{\dfrac{\partial u_i}{\partial z_j}\cdot S - u_i\cdot \dfrac{\partial S}{\partial z_j}}{S^2}.
$$

**Step 2.** $i\neq j$ 라 분자의 첫 항 0.
$$
= \frac{0\cdot S - u_i\cdot u_j}{S^2}=-\frac{u_i u_j}{S^2}.
$$
*등호 근거*: $\partial u_i/\partial z_j=\delta_{ij}u_i=0$ ($i\neq j$).

**Step 3.** $u_i/S=p_i$, $u_j/S=p_j$ 로 묶기.
$$
=-\frac{u_i}{S}\cdot\frac{u_j}{S}=\boxed{\,-p_i p_j\,}.
$$

### 4.3 통합 (Kronecker delta 형)

두 케이스를 한 식으로:
$$
\boxed{\;\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij}-p_j) = p_i\delta_{ij} - p_i p_j.\;}
$$

### 4.4 행렬 (Jacobian) 형태

$J_{ij} = \partial p_i/\partial z_j$ 로 정의. 위 결과를 행렬로 적으면:
- 대각 항: $J_{ii} = p_i - p_i^2$. 이는 $\mathrm{diag}(p)$ 의 대각 $p_i$ 에서 $(pp^\top)_{ii}=p_i^2$ 를 뺀 것.
- 비대각 항: $J_{ij} = -p_i p_j$. 이는 $\mathrm{diag}(p)$ 의 비대각 0 에서 $(pp^\top)_{ij}=p_i p_j$ 를 뺀 것.

따라서
$$
\boxed{\;J = \frac{\partial p}{\partial z} = \mathrm{diag}(p) - p p^\top \in\mathbb{R}^{n\times n}.\;}
$$

---

## 5. 검증

### 5.1 합 보존: $\sum_i J_{ij}=0$

$\sum_i p_i = 1$ 은 $z$ 와 무관한 상수. 양변을 $z_j$ 로 미분:
$$
\sum_i \frac{\partial p_i}{\partial z_j} = \frac{\partial}{\partial z_j}\,1 = 0.
$$
직접 대입으로도 확인:
$$
\sum_i p_i(\delta_{ij}-p_j) = p_j - p_j\underbrace{\sum_i p_i}_{=1}=p_j-p_j=0.\quad\checkmark
$$

### 5.2 대칭성: $J_{ij}=J_{ji}$

대각: $J_{ii}=p_i(1-p_i)$ (자기 자신).
비대각: $J_{ij}=-p_i p_j = -p_j p_i = J_{ji}$. $\checkmark$

따라서 $J$ 는 *대칭행렬*. 이는 $J$ 가 *분포 $p$ 의 공분산 행렬* 이라는 사실과 일치 — 공분산은 항상 대칭.

### 5.3 작은 수치 예: $n=2$

$z=(z_1,z_2)$, 예: $p=(0.7, 0.3)$.
- $J_{11}=0.7\cdot 0.3=0.21$.
- $J_{22}=0.3\cdot 0.7=0.21$.
- $J_{12}=J_{21}=-0.7\cdot 0.3=-0.21$.
- 행렬: $\begin{pmatrix}0.21 & -0.21\\ -0.21 & 0.21\end{pmatrix}$. 행 합 $=0$. $\checkmark$ rank $=1$ (한 행이 다른 행의 $-$).

### 5.4 공분산 해석

$X$ 가 분포 $p$ 를 따르는 categorical 확률변수 ($P(X=i)=p_i$). 그러면
$$
\mathrm{Cov}(X)_{ij} = \mathbb{E}[\mathbf{1}\{X=i\}\mathbf{1}\{X=j\}] - p_i p_j = \delta_{ij}p_i - p_i p_j.
$$
정확히 $J$ 와 일치. ⇒ **softmax Jacobian = categorical 분포의 공분산 행렬.** 따라서 $J$ 는 항상 *positive semi-definite* (rank $n-1$ — 합 0 제약 때문).

---

## 6. 일반화·통찰

### 6.1 Vector–Jacobian Product (VJP) 트릭

Backprop 에서는 보통 *왼쪽 곱* $g^\top J$ 만 필요 (전체 $J$ 를 메모리에 저장하지 않음).
$$
g^\top J = g^\top(\mathrm{diag}(p)-pp^\top) = (g\odot p)^\top - (g^\top p)\,p^\top.
$$
(여기서 $\odot$ 는 element-wise 곱.)

특히 $g=\partial L/\partial p$ 가 cross entropy 의 미분이라면 닫힘이 *극도로* 단순해진다 — Quiz 10 에서 $p-e_y$ 로 결말.

### 6.2 합 보존의 의미

$\sum_i \partial p_i/\partial z_j = 0$ 이라는 사실은 "$z_j$ 를 흔들어도 출력 확률들은 *재분배* 될 뿐 총합 1 이 보존된다" 의 미분 표현. softmax 가 *제로섬* 게임처럼 작동.

### 6.3 Numerical stability — log-sum-exp

큰 $z$ 에서 $e^{z_i}$ 가 오버플로우. 실전에서는 $z_i \mapsto z_i - \max_k z_k$ 로 시프트:
$$
p_i = \frac{e^{z_i - m}}{\sum_k e^{z_k - m}},\qquad m=\max_k z_k.
$$
시프트는 $z_j$ 미분에 영향 없음 (분자·분모 모두 같은 $-m$ 곱 → $1$).

### 6.4 Temperature scaling

$\tilde p_i = e^{z_i/T}/\sum_k e^{z_k/T}$ 일 때 Jacobian 은 $\frac{1}{T}(\mathrm{diag}(\tilde p)-\tilde p\tilde p^\top)$ — 같은 형태에 $1/T$ 스케일. $T\to 0$ 이면 argmax 로 수렴, $T\to\infty$ 이면 균등분포.

### 6.5 GLM 일반론

지수족 (exponential family) + canonical link 결합에서 *모든* 출력함수의 Jacobian 이 분포의 공분산 행렬과 일치. softmax–categorical 은 그 가장 흔한 사례.

---

## 7. 시험 출제 변형 5가지

### 변형 1. $n=2$ 직접 미분

> $p_1 = e^{z_1}/(e^{z_1}+e^{z_2})$ 의 모든 편미분.

*풀이.* $p_1 = \sigma(z_1-z_2)$ (sigmoid). $\partial p_1/\partial z_1 = p_1(1-p_1)$, $\partial p_1/\partial z_2=-p_1(1-p_1)$. $p_2=1-p_1$ 이라 $\partial p_2/\partial z_j$ 는 부호 바뀜.

### 변형 2. $\sum_i \partial p_i/\partial z_j$ 의 값

*답.* 0 (정규화 제약의 미분).

### 변형 3. Cross entropy 와 결합

> $L=-\sum_i y_i \log p_i$ 에서 $\partial L/\partial z_j$ 를 구하라 ($y$ 는 원핫).

*풀이.* $\partial L/\partial p_i = -y_i/p_i$. Chain rule:
$$
\frac{\partial L}{\partial z_j}=\sum_i \frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}=\sum_i\left(-\frac{y_i}{p_i}\right)p_i(\delta_{ij}-p_j) = -\sum_i y_i(\delta_{ij}-p_j) = -y_j + p_j\sum_i y_i = p_j - y_j.
$$
즉 $\partial L/\partial z = p - y$. (Quiz 10 의 결과 재유도.)

### 변형 4. Jacobian 의 rank

> $J=\mathrm{diag}(p)-pp^\top$ 의 rank 는?

*풀이.* $J\cdot \mathbf{1}=p-p\cdot 1=0$ 이라 $\mathbf 1$ 은 $J$ 의 null vector. 따라서 nullity $\ge 1$ ⇒ rank $\le n-1$. 한편 $p_i>0$ 모두 양이라 다른 방향들은 모두 nonzero ⇒ 정확히 rank $n-1$.

### 변형 5. PSD 증명

> $J$ 가 양의 준정부호임을 보여라.

*풀이.* 임의 $v\in\mathbb{R}^n$ 에 대해
$$
v^\top J v = v^\top \mathrm{diag}(p) v - (v^\top p)^2 = \sum_i p_i v_i^2 - \left(\sum_i p_i v_i\right)^2 = \mathrm{Var}_{p}(v).
$$
분산은 항상 $\ge 0$. ⇒ $J\succeq 0$. (Cauchy–Schwarz 의 분포 버전.)

---

## 8. 백지 재현 체크리스트

1. [ ] Softmax 정의 $p_i=e^{z_i}/\sum_k e^{z_k}$ 를 적을 수 있다.
2. [ ] 몫의 미분 공식을 외워 적을 수 있다.
3. [ ] $\partial S/\partial z_j = e^{z_j}$ 임을 한 줄 근거 (단일 항 생존) 와 함께 도출.
4. [ ] $\partial e^{z_i}/\partial z_j = \delta_{ij}\,e^{z_i}$ 도출.
5. [ ] Case $i=j$ 의 몫 미분을 정확히 적용해 $p_i(1-p_i)$ 도출.
6. [ ] Case $i\neq j$ 도 마찬가지로 $-p_i p_j$ 도출.
7. [ ] 두 케이스를 $p_i(\delta_{ij}-p_j)$ 한 식으로 통합.
8. [ ] 행렬 $J=\mathrm{diag}(p)-pp^\top$ 형태로 닫음.
9. [ ] 합 보존 $\sum_i J_{ij}=0$ 검증.
10. [ ] 대칭 $J_{ij}=J_{ji}$ 확인.
11. [ ] 공분산 해석 (categorical 분포) 한 줄.
12. [ ] $J\succeq 0$ 의 분산 해석.
13. [ ] Quiz 10 에서 cross entropy 와 결합되어 $p-e_y$ 가 됨을 안다.

---

## 9. 핵심 공식 카드

```
[Softmax]
  p_i = e^{z_i} / Σ_k e^{z_k}        (S ≡ Σ_k e^{z_k})

[기본 미분]
  ∂(e^{z_i})/∂z_j = δ_ij · e^{z_i}
  ∂S/∂z_j         = e^{z_j}
  몫의 미분:  (u/v)' = (u'v − u v')/v²

[Jacobian 성분]
  ∂p_i/∂z_j = p_i (δ_ij − p_j)
            = { p_i(1−p_i)   i=j
                −p_i p_j      i≠j

[행렬 형]
  J = diag(p) − p pᵀ   ∈ R^{n×n}

[성질]
  Σ_i J_ij = 0   (열 합 0; 정규화 제약 미분)
  J = Jᵀ          (대칭)
  J = Cov(p)      (categorical 공분산)
  J ⪰ 0,  rank J = n−1

[VJP]
  gᵀ J = (g⊙p)ᵀ − (gᵀp) pᵀ
  g = ∂L/∂p (CE 와 결합) ⇒ ∂L/∂z = p − y     (Quiz 10)
```

---

## 10. 다른 퀴즈와의 연결

- **Quiz 10 (Backprop):** 본 퀴즈의 $J$ 가 cross entropy gradient $-1/p_y\cdot e_y^\top$ 와 곱해져 $p-e_y$ 로 단순화. 출력층 backward 의 *닫힘 공식* — Q3 없이는 Q10 못 푼다.
- **Quiz 8 (KL → MSE):** softmax + cross entropy 가 categorical NLL = KL 임을 보였듯, 본 퀴즈는 softmax 의 *국지 미분* 측면을 책임. Q8 이 *손실 정당화*, Q3 가 *그 손실의 미분 메커니즘*.
- **Quiz 5/6/7 (MLE/MAP):** 베르누이 = 2-class softmax 의 특수 사례. $p_1=\theta, p_2=1-\theta$ 로 두면 본 퀴즈의 $J$ 는 $\theta(1-\theta)$ — Bernoulli Fisher information 과 일치.
- **공분산 / Fisher 정보:** $J$ = categorical 의 공분산 행렬 = log-likelihood 의 (음) Hessian 의 한 형태. 통계학 전체로 확장.
- **Numerical stability:** softmax 구현시 max-shift trick 이 필요 — 본 퀴즈가 미분 측면이라면 그건 *순방향* 측면이지만, 학습 안정성에 모두 영향.

---
title: "딥러닝 중간고사 모의시험 (30문항)"
slug: midterm-mock-exam-kr
order: 2
---

# 딥러닝 중간고사 모의시험 (30문항)
## 이성윤 교수 - 한양대학교
### 시험 특성: 영어 출제, 과정 중심 평가, 연역 → 논리 → 풀이

---

## 교수 분석

**이성윤 교수의 출제 스타일:**
- 서울대 수학 박사 — 극도로 엄밀한 수학적 추론을 요구
- 최종 답보다 **연역과 논리적 과정**을 중시
- "수식 하나가 그림 천 장의 가치" 철학
- 명확한 근거를 동반한 단계별 유도를 기대
- **완전한 풀이 과정**을 요구하는 퀴즈형 문제
- "어떻게"가 아닌 "왜"를 강조

---

# 파트 1: 선형대수 (Q1-Q8)

---

## 문제 1. 선형변환 검증

**$L : \mathbb{R}^2 \to \mathbb{R}^2$가 $L(v) = v + \begin{pmatrix} 1 \\ 0 \end{pmatrix}$로 정의될 때, $L$은 선형변환인가? 완전한 증명 또는 반례를 제시하여 답을 정당화하라.**

### 풀이

**[대학 수준]**

선형변환은 모든 $u, v \in \mathbb{R}^n$과 $a \in \mathbb{R}$에 대해 다음 두 성질을 만족해야 한다:
1. $L(v + u) = L(v) + L(u)$ (가법성)
2. $L(av) = aL(v)$ (동차성)

성질 (2) 확인: $a = 0$, $v = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$로 놓자.

$L(0 \cdot v) = L(\begin{pmatrix} 0 \\ 0 \end{pmatrix}) = \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

$0 \cdot L(v) = 0 \cdot \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

$L(0 \cdot v) \neq 0 \cdot L(v)$이므로, $L$은 선형변환이 **아니다**. $\square$

**[고등학교 수준]**

선형변환은 반드시 원점을 지나야 한다: $L(0) = 0$. 그런데 $L(0) = 0 + \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \neq 0$. 따라서 $L$은 선형이 아니다.

**[중학교 수준]**

선형변환은 원점을 고정하는 "직선 규칙"이라고 생각하면 된다. 영(아무것도 없는 것)을 입력하면 영이 나와야 한다. 그런데 이 함수는 모든 것에 $(1, 0)$을 더하므로, 영조차 $(1, 0)$이 된다. 이는 규칙을 어기는 것이므로 선형변환이 아니다.

---

## 문제 2. 랭크-퇴화차수 정리 적용

**$A \in \mathbb{R}^{3 \times 5}$이고 $\text{rank}(A) = 2$일 때, $\text{null}(A)$(영공간의 차원)를 구하라. 랭크와 퇴화차수의 관계를 설명하라.**

### 풀이

**[대학 수준]**

랭크-퇴화차수 정리에 의해: $A \in \mathbb{R}^{m \times n}$에 대해 $n = \text{rank}(A) + \text{null}(A)$.

여기서 $n = 5$, $\text{rank}(A) = 2$.

$\text{null}(A) = n - \text{rank}(A) = 5 - 2 = 3$

이는 $\mathcal{N}(A) = \ker(A) = \{v \in \mathbb{R}^5 : Av = 0\}$의 차원이 3임을 의미한다. 즉, $\mathbb{R}^5$의 3차원 부분공간이 $A$에 의해 영으로 사상된다.

또한: $\text{rank}(A) \leq \min(m, n) = \min(3, 5) = 3$이고, $\text{rank}(A) = 2 \leq 3$이므로 이를 만족한다. $\square$

**[고등학교 수준]**

행렬 $A$는 5개의 열(5개의 입력)을 가진다. 랭크-퇴화차수 정리: 열의 수 = 랭크 + 퇴화차수. 따라서 퇴화차수 = 5 - 2 = 3. 이는 입력 공간에서 3개의 독립적인 방향이 행렬에 의해 영으로 "압축"됨을 의미한다.

**[중학교 수준]**

행렬을 5개의 입력 슬롯을 가진 기계라고 상상하자. 랭크 = 2는 실제로 2개의 독립적인 출력만 생성됨을 의미한다. 나머지 5 - 2 = 3개의 입력 방향은 낭비된다(아무것도 생성하지 않는다). 그 "낭비된" 수가 퇴화차수 = 3이다.

---

## 문제 3. 고유값의 성질

**$A$가 고유값 $\lambda_1 = 3$과 $\lambda_2 = -1$을 가진 대칭행렬일 때, 다음의 고유값을 결정하라: (a) $A^2$, (b) $A^{-1}$ (존재하는 경우), (c) $A + 2I$, (d) 스칼라 $c$에 대한 $cA$.**

### 풀이

**[대학 수준]**

$Av = \lambda v$이면:

(a) $A^2 v = A(Av) = A(\lambda v) = \lambda(Av) = \lambda^2 v$.
따라서 $A^2$의 고유값: $\lambda_1^2 = 9$, $\lambda_2^2 = 1$.

(b) $\lambda_1 = 3 \neq 0$이고 $\lambda_2 = -1 \neq 0$이므로, $A^{-1}$이 존재한다.
$Av = \lambda v$로부터: $v = \lambda A^{-1} v$, 따라서 $A^{-1} v = \frac{1}{\lambda} v$.
$A^{-1}$의 고유값: $1/3$과 $-1$.

(c) $(A + 2I)v = Av + 2v = \lambda v + 2v = (\lambda + 2)v$.
고유값: $3 + 2 = 5$와 $-1 + 2 = 1$.

(d) $(cA)v = c(Av) = c\lambda v$.
고유값: $3c$와 $-c$.

모든 경우에서 고유벡터는 동일하게 유지된다. $\square$

**[고등학교 수준]**

규칙: $\lambda$가 $A$의 고유값이면:
- $A^2$의 고유값은 $\lambda^2$ → $9, 1$
- $A^{-1}$의 고유값은 $1/\lambda$ → $1/3, -1$
- $A + 2I$의 고유값은 $\lambda + 2$ → $5, 1$
- $cA$의 고유값은 $c\lambda$ → $3c, -c$

**[중학교 수준]**

고유값은 "늘이기 비율"과 같다. 원래 늘이기가 3이면, 기계를 제곱하면 $3 \times 3 = 9$만큼 늘어난다. 역기계($A^{-1}$)는 $1/3$만큼 늘인다. $2I$를 더하면 각 늘이기 비율에 2를 더한다. 기계에 $c$를 곱하면 각 늘이기에 $c$를 곱한다.

---

## 문제 4. 외적으로서의 행렬 곱셈

**$A = \begin{pmatrix} a_1^\top \\ a_2^\top \end{pmatrix}$이고 $B = \begin{pmatrix} | & | \\ b_1 & b_2 \\ | & | \end{pmatrix}$일 때 ($a_i, b_j \in \mathbb{R}^2$), $AB$를 내적 $a_i^\top b_j$로 표현하라. 그리고 $\text{rank}(AB) < \min(\text{rank}(A), \text{rank}(B))$가 발생할 수 있는 경우를 설명하라.**

### 풀이

**[대학 수준]**

$AB = \begin{pmatrix} a_1^\top b_1 & a_1^\top b_2 \\ a_2^\top b_1 & a_2^\top b_2 \end{pmatrix}$

각 성분 $(AB)_{ij} = a_i^\top b_j$는 $A$의 $i$번째 행과 $B$의 $j$번째 열의 내적이다.

$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$임이 알려져 있다.

엄격한 부등식은 $\text{im}(B) \cap \ker(A) \neq \{0\}$일 때, 즉 $B$의 영이 아닌 출력이 $A$의 영공간에 놓일 때 발생한다.

예: $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$. 둘 다 랭크 1이지만, $AB = 0$으로 랭크 0이다. $\square$

**[고등학교 수준]**

행렬 곱셈 $AB$: 각 성분은 $A$의 행과 $B$의 열의 내적이다. $AB$의 랭크는 $B$의 출력이 $A$의 사각지대(영공간)에 "빠질" 때 어느 한쪽의 랭크보다 작아질 수 있다.

**[중학교 수준]**

두 개의 필터가 겹쳐있다고 생각하자. $B$가 먼저 필터링하고, $A$가 두 번째로 필터링한다. $B$의 출력이 정확히 $A$가 무시하는 것이면, 각 필터가 단독으로는 무언가를 하더라도 조합은 아무것도 생성하지 않는다.

---

## 문제 5. SVD 해석

**특이값 분해(SVD) $A = U \Sigma V^\top$가 기하학적으로 무엇을 의미하는지 자신의 말로 설명하라. $U$, $\Sigma$, $V$는 각각 무엇을 나타내는가?**

### 풀이

**[대학 수준]**

$A \in \mathbb{R}^{m \times n}$에 대해 $A = U \Sigma V^\top$이며:
- $V \in \mathbb{R}^{n \times n}$은 직교행렬: 열 $v_i$가 $\mathbb{R}^n$(입력 공간)의 정규직교 기저를 형성
- $U \in \mathbb{R}^{m \times m}$은 직교행렬: 열 $u_i$가 $\mathbb{R}^m$(출력 공간)의 정규직교 기저를 형성
- $\Sigma \in \mathbb{R}^{m \times n}$은 특이값 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$을 가진 대각행렬

기하학적으로, 임의의 선형 사상 $A$는 세 단계로 분해될 수 있다:
1. $V^\top$를 사용하여 입력 공간을 **회전/반사** (주축에 정렬)
2. 각 축을 $\sigma_i$만큼 **스케일링** (늘이거나 압축)
3. $U$를 사용하여 출력 공간을 **회전/반사** (출력 축에 정렬)

$A$의 랭크는 영이 아닌 특이값의 수와 같다. $\square$

**[고등학교 수준]**

SVD는 임의의 행렬 변환 = 회전 → 스케일링 → 회전이라고 말한다. $V$가 입력을 회전하고, $\Sigma$가 각 방향을 다른 양만큼 늘이고, $U$가 결과를 회전한다. 모든 복잡한 운동을 단순한 단계로 분해하는 것과 같다.

**[중학교 수준]**

고무판을 변형하는 것을 상상하자. SVD는 어떤 변형이든 다음으로 분해할 수 있다고 말한다: (1) 판을 돌리고, (2) 각 방향으로 더 넓게/좁게 늘이고, (3) 다시 돌린다. 이 세 가지 간단한 단계로 어떤 변형이든 재현할 수 있다.

---

## 문제 6. 의사역행렬과 $Ax = b$ 풀기

**$A \in \mathbb{R}^{m \times n}$이고 $m > n$(과결정 시스템)인 연립방정식 $Ax = b$를 고려하라. 왜 해가 존재하지 않을 수 있는지, 그리고 의사역행렬 $A^+$가 어떻게 "최선의" 근사해를 제공하는지 설명하라. 완전 열 랭크 $A$에 대한 $A^+$는 무엇인가?**

### 풀이

**[대학 수준]**

$m > n$이면 미지수보다 방정식이 많으므로, $b$가 $\text{im}(A)$에 놓이지 않을 수 있으며, 이는 $Ax = b$에 정확한 해가 없음을 의미한다.

의사역행렬은 최소제곱해를 제공한다: $x^* = A^+ b = \arg\min_x \|Ax - b\|^2$.

완전 열 랭크 $A$ (랭크 $= n$)에 대해 $A^\top A$는 가역이며:
$$A^+ = (A^\top A)^{-1} A^\top$$

유도: $L(x) = \|Ax - b\|^2 = (Ax - b)^\top(Ax - b)$를 최소화한다.
$\nabla_x L = 2A^\top(Ax - b) = 0 \Rightarrow A^\top A x = A^\top b \Rightarrow x = (A^\top A)^{-1} A^\top b$.

성질: $AA^+ b = \text{Proj}(b; \text{im}(A))$, $b$를 $A$의 열공간에 직교 투영한 것이다.

또한, $\ker(A^+A) = \ker(A)$이므로 $b \in \text{im}(A)$일 때 해가 정확함을 확인한다. $\square$

**[고등학교 수준]**

미지수보다 방정식이 더 많으면 모든 방정식을 정확히 만족시킬 수 없다. $A^+$는 $Ax$를 $b$에 가능한 한 가깝게 만드는 $x$를 찾는다(총 오차를 최소화). 완전 열 랭크의 경우: $A^+ = (A^\top A)^{-1} A^\top$.

**[중학교 수준]**

직선 위에 놓이지 않는 10개의 점을 통과하는 직선을 그리려 한다고 상상하자. 모든 점을 맞출 수는 없지만, 모든 점에 가능한 한 가까운 "최적 적합" 직선을 찾을 수 있다. 의사역행렬 공식이 이 최적 적합을 자동으로 찾아준다.

---

## 문제 7. 양의 정부호 행렬의 성질

**$A$가 대칭 양의 정부호(PD) 행렬일 때, $A$의 모든 고유값이 양수임을 증명하라. 그리고 $A$가 가역인 이유를 설명하라.**

### 풀이

**[대학 수준]**

**정의:** $A$가 PD $\Leftrightarrow$ 모든 $v \neq 0$에 대해 $v^\top A v > 0$.

**증명:** $\lambda$를 고유벡터 $v \neq 0$에 대한 고유값이라 하면, $Av = \lambda v$.
그러면 $v^\top A v = v^\top (\lambda v) = \lambda \|v\|^2$.
$A$가 PD이므로 $v^\top A v > 0$이고, $v \neq 0$이므로 $\|v\|^2 > 0$. 따라서 $\lambda > 0$.

**가역성:** 모든 고유값이 양수(따라서 영이 아님)이므로, $\det(A) = \prod_i \lambda_i > 0 \neq 0$. 따라서 $A$는 가역이다.

또는, $A^{-1} = \sum_{i=1}^n \frac{1}{\lambda_i} u_i u_i^\top$ (스펙트럼 분해)이고, $\lambda_i > 0$이므로 각 $\frac{1}{\lambda_i}$가 존재한다. $\square$

**[고등학교 수준]**

PD란 임의의 영이 아닌 $v$에 대해 $v^\top A v > 0$임을 의미한다. 고유벡터를 대입하면: $\lambda \|v\|^2 > 0$이므로 $\lambda > 0$. 모든 양의 고유값 → 행렬식 $\neq 0$ → 행렬은 가역이다.

**[중학교 수준]**

양의 정부호 행렬은 "그릇" 형태와 같다 — 모든 것이 위로 곡선을 그린다. 고유값은 각 방향에서 그릇이 얼마나 가파른지를 나타낸다. 항상 위로 곡선을 그리므로 모든 가파름(고유값)은 양수여야 하고, 어떤 방향도 평평하지(영이) 않으므로 행렬을 "되돌릴" 수(역행렬을 구할 수) 있다.

---

## 문제 8. 프로베니우스 노름과 트레이스

**임의의 행렬 $A \in \mathbb{R}^{m \times n}$에 대해 $\|A\|_F^2 = \text{Tr}(A^\top A)$임을 보이라. 또한 $x \sim \mathcal{N}(0, I)$일 때 $\mathbb{E}[\|Ax\|^2] = \text{Tr}(A^\top A)$임을 보이라.**

### 풀이

**[대학 수준]**

**파트 1:**
$\|A\|_F^2 = \sum_{i,j} a_{ij}^2$

$(A^\top A)_{jj} = \sum_{i=1}^m a_{ij}^2$이므로, $\text{Tr}(A^\top A) = \sum_{j=1}^n \sum_{i=1}^m a_{ij}^2 = \sum_{i,j} a_{ij}^2 = \|A\|_F^2$. $\square$

**파트 2:**
$\mathbb{E}[\|Ax\|^2] = \mathbb{E}[x^\top A^\top A x] = \mathbb{E}[\text{Tr}(x^\top A^\top A x)]$

$x^\top A^\top A x$는 스칼라이므로, $\text{Tr}(x^\top A^\top A x) = \text{Tr}(A^\top A x x^\top)$ (트레이스의 순환 성질).

$= \text{Tr}(A^\top A \cdot \mathbb{E}[x x^\top]) = \text{Tr}(A^\top A \cdot I) = \text{Tr}(A^\top A) = \|A\|_F^2$. $\square$

**[고등학교 수준]**

프로베니우스 노름은 "모든 성분의 제곱의 합"이다. $A^\top A$의 트레이스는 "$A^\top A$의 대각 성분의 합"이고, 각 대각 성분은 $A$의 한 열의 제곱의 합이다. 따라서 같은 것이다.

기댓값에 대해: $x$는 단위 공분산($\mathbb{E}[xx^\top] = I$)을 가진 확률변수이므로, 트레이스 트릭을 사용하여 순서를 바꾸면 확률성이 상쇄되어 $\text{Tr}(A^\top A)$를 얻는다.

**[중학교 수준]**

프로베니우스 노름은 행렬의 모든 수를 제곱하여 더한 것으로 행렬의 "전체 크기"를 측정한다. 트레이스 트릭은 수학적 지름길이다: 복잡한 확률 양을 직접 계산하는 대신 공식을 재배열하여 확률성이 깔끔하게 상쇄되도록 한다.

---

# 파트 2: 행렬 미적분 (Q9-Q13)

---

## 문제 9. 행렬 미적분의 연쇄 법칙

**$z = BAx$일 때 ($B \in \mathbb{R}^{p \times m}$, $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$), 연쇄 법칙을 사용하여 $\frac{\partial z}{\partial x}$를 계산하라. 직접 계산으로 검증하라.**

### 풀이

**[대학 수준]**

**방법 1 (연쇄 법칙):**
$y = Ax \in \mathbb{R}^m$으로 놓으면, $z = By$.

$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x} = B \cdot A = BA$

(여기서 $\frac{\partial z}{\partial y} = B \in \mathbb{R}^{p \times m}$이고 $\frac{\partial y}{\partial x} = A \in \mathbb{R}^{m \times n}$)

**방법 2 (직접 계산):**
$z = BAx$이고, $BA \in \mathbb{R}^{p \times n}$은 상수 행렬이다.

$\frac{\partial z}{\partial x} = BA \in \mathbb{R}^{p \times n}$

두 방법 모두 같은 결과를 주며, 연쇄 법칙을 확인한다. $\square$

야코비안 $\frac{\partial z}{\partial x} \in \mathbb{R}^{p \times n}$의 차원은 $d(z) \times d(x)$이다.

**[고등학교 수준]**

연쇄 법칙: "외부 함수의 도함수 × 내부 함수의 도함수." 여기서 외부 = $B$를 곱하기, 내부 = $A$를 곱하기. 결과: $B \times A = BA$.

**[중학교 수준]**

두 연산을 연달아 하면(먼저 $A$를 곱하고, 그다음 $B$를 곱하면), 합친 효과는 $BA$를 곱하는 것이다. 선형 연산의 도함수(변화율)는 그 연산 자체이다.

---

## 문제 10. 이차형식의 그래디언트

**$A$가 대칭행렬이고 $x \in \mathbb{R}^n$일 때, $\nabla_x (x^\top A x)$를 계산하라.**

### 풀이

**[대학 수준]**

$f(x) = x^\top A x = \sum_{i,j} a_{ij} x_i x_j$

**방법 1 (곱의 법칙):**
$\frac{\partial (x^\top A x)}{\partial x} = \frac{\partial (x^\top)}{\partial x} (Ax) + x^\top \frac{\partial (Ax)}{\partial x}$

행렬 미적분의 곱의 법칙을 사용하면:
$= (Ax)^\top + x^\top A = x^\top A^\top + x^\top A$

그래디언트를 얻기 위해 전치하면:
$\nabla_x (x^\top A x) = (A^\top + A)x$

대칭 $A$ ($A^\top = A$)에 대해:
$$\nabla_x (x^\top A x) = 2Ax$$

**방법 2 (직접):**
$f(x + \epsilon) = (x + \epsilon)^\top A(x + \epsilon) = x^\top Ax + x^\top A\epsilon + \epsilon^\top Ax + \epsilon^\top A\epsilon$
$= f(x) + (x^\top A + x^\top A^\top)\epsilon + O(\|\epsilon\|^2)$
$= f(x) + 2x^\top A \epsilon + O(\|\epsilon\|^2)$

따라서 $\nabla_x f = 2Ax$. $\square$

**[고등학교 수준]**

대칭행렬에 대해, $\nabla_x (x^\top A x) = 2Ax$. 이것은 $\frac{d}{dx}(ax^2) = 2ax$의 행렬 버전이라고 생각하면 된다.

**[중학교 수준]**

일반 수학에서 $f(x) = ax^2$이면 도함수는 $2ax$이다. 행렬 수학에서 $x^\top A x$는 $ax^2$의 "행렬 버전"이므로 도함수는 $2Ax$이다. 행렬 $A$가 수 $a$의 역할을 한다.

---

## 문제 11. 소프트맥스의 야코비안

**$z, p \in \mathbb{R}^C$에 대해 $p_i = \frac{\exp(z_i)}{\sum_{k=1}^C \exp(z_k)}$일 때, (a) $i = j$와 (b) $i \neq j$에 대해 $\frac{\partial p_i}{\partial z_j}$를 계산하라.**

### 풀이

**[대학 수준]**

$Z = \sum_{k=1}^C \exp(z_k)$로 놓자.

**(a) $i = j$:**
$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot Z - \exp(z_i) \cdot \exp(z_i)}{Z^2}$ (몫의 미분법)
$= \frac{\exp(z_i)}{Z} \cdot \frac{Z - \exp(z_i)}{Z}$
$= p_i(1 - p_i)$

**(b) $i \neq j$:**
$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot Z - \exp(z_i) \cdot \exp(z_j)}{Z^2}$
$= -\frac{\exp(z_i)}{Z} \cdot \frac{\exp(z_j)}{Z}$
$= -p_i p_j$

**간결한 형태:** $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$, 또는 행렬 형태로:
$$\frac{\partial p}{\partial z} = \text{diag}(p) - p p^\top$$

$\square$

**[고등학교 수준]**

$p_i = e^{z_i} / \sum e^{z_k}$에 몫의 미분법을 적용한다:
- 같은 인덱스 ($i = j$): $p_i(1 - p_i)$ — 로지스틱 함수의 도함수와 유사
- 다른 인덱스 ($i \neq j$): $-p_i p_j$ — $z_j$를 증가시키면 $p_i$가 감소

**[중학교 수준]**

소프트맥스는 숫자를 합이 1인 확률로 변환한다. 하나의 입력 $z_j$를 증가시키면, 그 확률 $p_j$는 올라가고($p_j(1-p_j)$만큼), 다른 모든 확률은 내려간다($p_i \times p_j$만큼). 총합은 여전히 1이다.

---

## 문제 12. $\|f(x)\|^2$의 도함수

**$f : \mathbb{R}^{d(x)} \to \mathbb{R}^{d(y)}$일 때, $\frac{\partial \|f(x)\|^2}{\partial x} = 2f(x)^\top \frac{\partial f(x)}{\partial x}$임을 보이라.**

### 풀이

**[대학 수준]**

$\|f(x)\|^2 = f(x)^\top f(x)$

행렬 도함수의 곱의 법칙을 사용하면:
$\frac{\partial [f(x)^\top f(x)]}{\partial x} = \frac{\partial f(x)^\top}{\partial x} f(x) + f(x)^\top \frac{\partial f(x)}{\partial x}$

$\frac{\partial f^\top}{\partial x} f = (f^\top \frac{\partial f}{\partial x})^\top$이고, 두 항이 서로의 전치이므로:

$\frac{\partial}{\partial x}\|f\|^2 = 2 f(x)^\top \frac{\partial f(x)}{\partial x}$

이것은 $1 \times d(x)$ 행벡터이다 ($x$에 대한 스칼라의 도함수).

차원 확인: $f(x)^\top$는 $1 \times d(y)$, $\frac{\partial f}{\partial x}$는 $d(y) \times d(x)$, 곱은 $1 \times d(x)$. $\checkmark$ $\square$

**[고등학교 수준]**

이것은 $\frac{d}{dx}[g(x)]^2 = 2g(x) \cdot g'(x)$의 행렬 버전이다. 스칼라 $g$를 벡터 $f$로, 스칼라 곱을 내적/행렬 곱으로 바꾼 것이다.

**[중학교 수준]**

기본 수학에서 $(g^2)' = 2g \cdot g'$이다. 벡터에도 같은 패턴이 적용된다: $f$의 "길이의 제곱"의 도함수는 $2 \times f \times (f\text{의 변화율})$이다.

---

## 문제 13. $z = BAx$에서 $\frac{\partial z}{\partial x}$ 계산

**$z = BAx$일 때 ($B \in \mathbb{R}^{p \times m}$, $A \in \mathbb{R}^{m \times n}$), $\frac{\partial z}{\partial x}$의 차원은? 명시적으로 작성하라.**

### 풀이

**[대학 수준]**

$z \in \mathbb{R}^p$, $x \in \mathbb{R}^n$이므로, $\frac{\partial z}{\partial x} \in \mathbb{R}^{p \times n}$ (분자 레이아웃 규약).

$z = (BA)x$이고 $BA \in \mathbb{R}^{p \times n}$.

$z$가 $x$에 대해 선형이므로: $\frac{\partial z}{\partial x} = BA$.

검증: $z_i = \sum_j (BA)_{ij} x_j$이므로, $\frac{\partial z_i}{\partial x_j} = (BA)_{ij}$, 이는 행렬 $BA$를 준다. $\square$

**[고등학교 수준]**

차원 = (출력 차원) × (입력 차원) = $p \times n$. 선형 함수 $Mx$의 야코비안은 그냥 $M$이다.

**[중학교 수준]**

기계가 입력에 어떤 수(예: 3)를 곱하면, "변화율"은 그냥 3이다. 마찬가지로, 행렬이 $x$에 $BA$를 곱하면, 변화율은 $BA$이다.

---

# 파트 3: 미적분 & 최적화 (Q14-Q18)

---

## 문제 14. 뉴턴의 방법

**$f(x)$의 영점을 찾기 위한 뉴턴의 방법을 유도하라. $x_0 = 2$에서 시작하여, $f(x) = x^2 - 3$의 근사근을 찾기 위해 한 번의 반복을 적용하라.**

### 풀이

**[대학 수준]**

**유도:** 점 $x_n$에서 $f$의 접선은:
$y = f'(x_n)(x - x_n) + f(x_n)$

접선이 x축과 만나는 점을 찾기 위해 $y = 0$으로 놓으면:
$0 = f'(x_n)(x_{n+1} - x_n) + f(x_n)$
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**적용:** $f(x) = x^2 - 3$, $f'(x) = 2x$.

$x_0 = 2$:
$x_1 = 2 - \frac{2^2 - 3}{2 \cdot 2} = 2 - \frac{1}{4} = 1.75$

확인: $f(1.75) = 3.0625 - 3 = 0.0625$ (0에 가까움, 실제 $\sqrt{3} \approx 1.732$).

**최적화와의 연결:** $f(x) = \nabla_x L(x)$이면, 뉴턴의 방법은:
$x_{n+1} = x_n - [H(x_n)]^{-1} \nabla L(x_n)$
여기서 $H$는 헤시안이다. $\square$

**[고등학교 수준]**

현재 추측에서 접선을 그리고, x축과 만나는 곳을 다음 추측으로 사용한다. 공식: $x_{n+1} = x_n - f(x_n)/f'(x_n)$. $x_0 = 2$일 때: $x_1 = 2 - 1/4 = 1.75$.

**[중학교 수준]**

$\sqrt{3}$을 찾기 위해 추측 2에서 시작한다. 함수 $x^2 - 3$은 얼마나 벗어났는지 알려준다($2^2 - 3 = 1$, 너무 크다). 기울기($2 \times 2 = 4$)를 사용하여 조정한다: $2 - 1/4 = 1.75$. 벌써 $\sqrt{3} \approx 1.732$에 매우 가깝다!

---

## 문제 15. 라그랑주 승수

**제약 조건 $x + y = 1$ 하에서 $f(x, y) = x^2 + y^2$를 최소화하라. 라그랑주 승수법을 사용하라. 완전한 유도를 제시하라.**

### 풀이

**[대학 수준]**

라그랑지안을 정의한다: $\mathcal{L}(x, y, \lambda) = x^2 + y^2 + \lambda(x + y - 1)$

모든 편미분을 0으로 놓으면:
$$\frac{\partial \mathcal{L}}{\partial x} = 2x + \lambda = 0 \quad \Rightarrow \quad x = -\lambda/2$$
$$\frac{\partial \mathcal{L}}{\partial y} = 2y + \lambda = 0 \quad \Rightarrow \quad y = -\lambda/2$$
$$\frac{\partial \mathcal{L}}{\partial \lambda} = x + y - 1 = 0$$

식 (1)과 (2)에서: $x = y = -\lambda/2$.
(3)에 대입: $-\lambda/2 + (-\lambda/2) = 1 \Rightarrow -\lambda = 1 \Rightarrow \lambda = -1$.

따라서 $x = y = 1/2$이고, $f(1/2, 1/2) = 1/4 + 1/4 = 1/2$.

**기하학적 해석:** 최적점에서 $\nabla f = (2x, 2y)$는 $\nabla g = (1, 1)$에 평행해야 한다. $(1/2, 1/2)$에서: $\nabla f = (1, 1) = -\lambda \cdot (1, 1)$이므로 $\lambda = -1$이 확인된다.

참고: 직선을 따라 $f$의 최댓값은 없다(무한히 증가한다). $\square$

**[고등학교 수준]**

라그랑주 승수법: 최적점에서 $f$의 그래디언트는 제약의 그래디언트에 평행해야 한다. $\mathcal{L} = x^2 + y^2 + \lambda(x + y - 1)$을 설정하고, 3개 방정식의 연립을 풀면 → $(x, y) = (1/2, 1/2)$, 최솟값 $= 1/2$.

**[중학교 수준]**

직선 $x + y = 1$ 위에서 원점에 가장 가까운 점을 찾고 싶다. 대칭에 의해 $x = y$인 곳이어야 한다. $x + y = 1$에서 $x = y = 1/2$. 거리의 제곱은 $1/4 + 1/4 = 1/2$.

---

## 문제 16. 제약 최적화로부터 소프트맥스 유도

**소프트맥스 함수 $p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$가 $\sum_i p_i = 1$, $p_i \geq 0$ 제약 하에서 $f(p) = \sum_i p_i z_i + \tau H(p)$를 최대화함으로써 도출됨을 보이라. 여기서 $H(p) = -\sum_i p_i \log p_i$는 엔트로피이다.**

### 풀이

**[대학 수준]**

라그랑지안을 설정한다:
$\mathcal{L}(p, \lambda) = \sum_i p_i z_i + \tau(-\sum_i p_i \log p_i) + \lambda(1 - \sum_i p_i)$

$= \sum_i [p_i z_i - \tau p_i \log p_i] + \lambda(1 - \sum_i p_i)$

$p_i$에 대해 미분하고 0으로 놓으면:
$\frac{\partial \mathcal{L}}{\partial p_i} = z_i - \tau(\log p_i + 1) - \lambda = 0$

$\Rightarrow \log p_i = \frac{z_i - \lambda - \tau}{\tau}$

$\Rightarrow p_i = \exp\left(\frac{z_i - \lambda - \tau}{\tau}\right) = \frac{1}{Z}\exp\left(\frac{z_i}{\tau}\right)$

여기서 $Z = \exp((\lambda + \tau)/\tau)$. 제약 $\sum_i p_i = 1$을 사용하면:

$Z = \sum_j \exp(z_j / \tau)$

따라서: $p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$

이것이 온도 $\tau$를 가진 소프트맥스이다. $\tau \to 0$이면 $p$는 argmax에 접근하고; $\tau \to \infty$이면 $p$는 균일 분포에 접근한다. $\square$

**[고등학교 수준]**

"높은 점수 옵션 선택"($\sum p_i z_i$)과 "불확실하게 유지"(엔트로피 $H$)를 균형잡는 확률 $p_i$를 찾고 싶다. 라그랑주 승수를 사용하고 로그 방정식을 풀면 소프트맥스 공식을 얻는다. 온도 $\tau$가 이 균형을 제어한다.

**[중학교 수준]**

소프트맥스는 최선의 옵션을 "부드럽게" 고르는 방법이다. 가장 높은 점수의 옵션을 선택하되 약간의 무작위성도 유지하고 싶다면, 소프트맥스가 정확히 그것을 한다. 수학은 이것이 결정적임과 불확실함 사이의 *최적* 균형임을 보여준다.

---

## 문제 17. 경사 하강법 업데이트

**$L(\theta)$의 1차 테일러 전개로부터 경사 하강법 업데이트 규칙을 유도하라. 학습률 $\eta$를 신중하게 선택해야 하는 이유를 설명하라.**

### 풀이

**[대학 수준]**

$\theta_t$에서의 1차 테일러 전개:
$L(\theta) \approx L(\theta_t) + \nabla L(\theta_t)^\top (\theta - \theta_t)$

$L$을 감소시키는 $\theta_{t+1}$을 선택하고자 한다. 가장 가파른 하강 방향은 $-\nabla L(\theta_t)$ (음의 그래디언트)이다.

그 방향으로 크기 $\eta$의 스텝을 밟으면:
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

그러면: $L(\theta_{t+1}) \approx L(\theta_t) - \eta \|\nabla L(\theta_t)\|^2 < L(\theta_t)$ ($\eta$가 충분히 작으면 감소가 보장된다).

**왜 $\eta$가 중요한가:**
- 너무 큼: 테일러 근사가 깨지고, $L$이 증가할 수 있음 (발산)
- 너무 작음: 매우 느린 수렴
- $L$-매끄러운 함수($\|\nabla^2 L\| \leq L$)에 대해, $\eta \leq 1/L$이면 수렴이 보장됨
- 뉴턴 방법과의 연결: $\eta = [H]^{-1}$이 "최적" 학습률 (2차 방법) $\square$

**[고등학교 수준]**

경사 하강법: 기울기 반대 방향으로 내려간다. 스텝 크기($\eta$)가 중요하다: 너무 크면 = 오버슈트, 너무 작으면 = 너무 느림. 업데이트는 $\theta_{\text{new}} = \theta_{\text{old}} - \eta \times \text{그래디언트}$.

**[중학교 수준]**

언덕 위에 있고 바닥에 도달하고 싶은데 눈을 가리고 있다고 상상하자. 발밑의 경사를 느끼고 내리막으로 한 걸음 내딛는다. "학습률"은 걸음의 크기이다. 너무 크면 계곡을 뛰어넘을 수 있고; 너무 작으면 영원히 걸린다.

---

## 문제 18. 미분방정식과 시그모이드

**ODE $x' = a(x - \alpha)(x - \beta)$ ($\alpha \neq \beta$)는 시그모이드 형태의 해를 갖는다. 해를 유도하고 시그모이드 형태가 나타나는 이유를 설명하라.**

### 풀이

**[대학 수준]**

변수를 분리한다: $\frac{1}{(x - \alpha)(x - \beta)} dx = a \, dt$

부분분수: $\frac{1}{(x-\alpha)(x-\beta)} = \frac{1}{\alpha - \beta}\left(\frac{1}{x-\alpha} - \frac{1}{x-\beta}\right)$

적분: $a t + C = \frac{1}{\alpha - \beta}[\log(x-\alpha) - \log(x-\beta)] = \frac{1}{\alpha - \beta}\log\frac{x-\alpha}{x-\beta}$

$(\alpha - \beta)at + C' = \log\frac{x - \alpha}{x - \beta}$

$\exp((\alpha-\beta)at + C') = \frac{x - \alpha}{x - \beta}$

$x$에 대해 풀면:
$$x(t) = \frac{\alpha - \beta}{1 - C\exp((\alpha - \beta)at)} + \beta$$

**왜 시그모이드인가:** 해는 두 고정점 $\alpha$와 $\beta$ 사이를 부드럽게 전이한다. 그 사이에서 성장은 처음에 지수적(로지스틱 방정식처럼)이다가 안정 평형에 접근하면서 포화된다. 이 S-곡선이 시그모이드 형태이다.

$\alpha = 1, \beta = 0, a = 1$일 때: $x(t) = \frac{1}{1 + e^{-t}} = \sigma(t)$, 표준 시그모이드. $\square$

**[고등학교 수준]**

이 방정식은 두 평형점($\alpha$와 $\beta$)을 갖는다. 하나는 안정적, 하나는 불안정적이다. 해는 S-형태(시그모이드)로 둘 사이를 부드럽게 전이한다. 성장이 처음에 빨라지다가 안정점에 접근하면서 느려지기 때문이다.

**[중학교 수준]**

작을 때는 빠르게 성장하지만 한계에 도달하면 멈추는 인구를 생각하자. S-형 곡선(시그모이드)은 느린 시작 → 빠른 성장 → 수평화를 보여준다. 공식은 이것이 수학적으로 어떻게 일어나는지 정확히 보여준다.

---

# 파트 4: 확률 (Q19-Q25)

---

## 문제 19. 베이즈 정리 적용

**한 질병 검사의 참양성률은 99%, 거짓양성률은 2%이다. 인구의 1%가 질병을 가지고 있다면, 양성 판정을 받은 사람이 실제로 질병을 가지고 있을 확률은 얼마인가? 단계별 추론을 보이라.**

### 풀이

**[대학 수준]**

$D$ = 질병 있음, $T^+$ = 양성 판정으로 놓자.

주어진 조건: $P(T^+ | D) = 0.99$, $P(T^+ | D^c) = 0.02$, $P(D) = 0.01$.

베이즈 정리에 의해:
$$P(D | T^+) = \frac{P(T^+ | D)P(D)}{P(T^+)}$$

$P(T^+) = P(T^+|D)P(D) + P(T^+|D^c)P(D^c)$
$= 0.99 \times 0.01 + 0.02 \times 0.99$
$= 0.0099 + 0.0198 = 0.0297$

$$P(D | T^+) = \frac{0.0099}{0.0297} = \frac{1}{3} \approx 0.333$$

**핵심 통찰:** 99%의 정확도에도 불구하고, 양성 판정은 질병 확률 ~33%만을 의미한다. 질병이 희귀(낮은 사전확률)하기 때문이다. 사전확률 $P(D)$가 사후확률에 극적으로 영향을 미친다. 이는 베이지안 추론에서 **사전확률이 중요한** 이유를 보여준다: $P(H) \to P(H|E)$. $\square$

**[고등학교 수준]**

10,000명 중: 100명이 질병 있음(99명이 양성 판정), 9,900명은 없음(198명이 실수로 양성 판정). 총 양성 = 297명. 그 중 실제로 질병이 있는 사람은 99명뿐. $99/297 = 1/3 \approx 33\%$.

**[중학교 수준]**

10,000명을 상상하자. 100명만 아프다. 검사는 그 중 99명을 정확히 잡아낸다. 하지만 198명의 건강한 사람도 잘못 양성으로 판정한다. 297명의 양성 중 실제로 아픈 사람은 99명뿐이다. 약 3명 중 1명 — 질병이 매우 희귀하기 때문에 놀랍지만 사실이다.

---

## 문제 20. 가우시안의 MLE

**$\mathcal{N}(\mu, \sigma^2)$로부터의 i.i.d. 표본 $x_1, \ldots, x_n$이 주어졌을 때, MLE 추정량 $\hat{\mu}_{ML}$과 $\hat{\sigma}^2_{ML}$을 유도하라. $\hat{\sigma}^2_{ML}$은 불편 추정량인가?**

### 풀이

**[대학 수준]**

로그우도:
$\ell(\mu, \sigma^2) = \log \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$
$= n \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$

**$\mu$에 대해:** $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$
$\Rightarrow \hat{\mu}_{ML} = \frac{1}{n}\sum_{i=1}^n x_i$ (표본 평균)

**$\sigma^2$에 대해:** $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0$
$\Rightarrow \hat{\sigma}^2_{ML} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$

**편향 확인:**
$\mathbb{E}[\hat{\mu}_{ML}] = \mu$ → **불편** $\checkmark$
$\mathbb{E}[\hat{\sigma}^2_{ML}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$ → **편향** $\times$

불편 추정량: $\hat{\sigma}^2_{unb} = \frac{n}{n-1}\hat{\sigma}^2_{ML} = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ $\square$

**[고등학교 수준]**

MLE = 관측 데이터를 가장 그럴듯하게 만드는 매개변수를 찾기. 가우시안의 경우: $\hat{\mu} = \text{표본 평균}$, $\hat{\sigma}^2 = n$으로 나눈 표본 분산. 분산 추정은 편향됨(평균적으로 약간 작음); $n$ 대신 $n-1$로 나누면 이것이 수정됨.

**[중학교 수준]**

데이터 포인트가 주어졌을 때, 평균의 최선 추측은 데이터의 실제 평균이다. 퍼짐의 최선 추측은 평균으로부터의 제곱 차이의 평균을 사용하지만, $n$으로 나누면 약간 너무 작은 답을 준다. $n-1$로 나누면 이것이 수정된다.

---

## 문제 21. MAP vs MLE: 동전 던지기

**동전을 $n = 3$번 던져 $k = 3$번 앞면이 나왔다. 사전분포 $p(\theta) \propto \theta(1 - \theta)$를 사용할 때, $\theta$(앞면 확률)의 MLE와 MAP 추정을 비교하라. 여기서 MAP가 왜 더 좋은지 설명하라.**

### 풀이

**[대학 수준]**

**MLE:** $\hat{\theta}_{ML} = k/n = 3/3 = 1$

이는 모든 미래 던지기가 앞면이 될 것이라 예측한다 — 분명히 불합리하다.

**MAP:** 사후확률 $\propto$ 우도 $\times$ 사전확률:
$p(\theta | \text{data}) \propto \theta^k (1-\theta)^{n-k} \cdot \theta(1-\theta) = \theta^{k+1}(1-\theta)^{n-k+1}$

이것은 베타 분포 $\text{Beta}(k+2, n-k+2)$이다.

MAP = 베타의 최빈값:
$\hat{\theta}_{MAP} = \frac{k+1}{n+2} = \frac{4}{5} = 0.8$

**MAP가 더 좋은 이유:**
- MLE는 뒷면에 확률 0을 부여하는데, 단 3번의 관찰로는 극단적임
- MAP는 극단값(0 또는 1)이 가능성이 낮다는 사전 믿음을 포함
- 사전분포 $\theta(1-\theta)$는 Beta(2,2) 사전분포로, 대칭이고 0.5에서 정점
- MAP는 정규화로 작용: 추정을 0.5 쪽으로 "수축"시킴
- $n \to \infty$이면, MAP $\to$ MLE (데이터가 사전확률을 지배) $\square$

**[고등학교 수준]**

MLE는 $\theta = 1$(항상 앞면)이라 하는데, 단 3번 던지기로는 잘못된 것 같다. MAP는 극단적 확률이 가능성이 낮다는 "상식적 사전분포"를 추가하여 $\theta = 0.8$을 준다 — 더 합리적이다. 데이터가 많을수록 → MAP와 MLE가 수렴.

**[중학교 수준]**

동전을 3번 던져 모두 앞면이면, MLE는 "이 동전은 항상 앞면이 나온다!"고 말한다. MAP는 "아마 대부분 앞면이지만, 너무 극단적이지는 말자"며 80%로 추측한다. 3번 던지기는 확신하기에 충분하지 않으므로 MAP가 더 합리적이다.

---

## 문제 22. 기댓값과 분산

**$X$와 $Y$가 독립인 확률변수이고 $\mathbb{E}[X] = 2$, $\text{Var}[X] = 3$, $\mathbb{E}[Y] = -1$, $\text{Var}[Y] = 4$일 때, $\mathbb{E}[3X - 2Y + 5]$와 $\text{Var}[3X - 2Y + 5]$를 계산하라.**

### 풀이

**[대학 수준]**

**기댓값 (선형성):**
$\mathbb{E}[3X - 2Y + 5] = 3\mathbb{E}[X] - 2\mathbb{E}[Y] + 5$
$= 3(2) - 2(-1) + 5 = 6 + 2 + 5 = 13$

**분산:**
$X, Y$가 독립이므로:
$\text{Var}[3X - 2Y + 5] = 3^2 \text{Var}[X] + (-2)^2 \text{Var}[Y] + 0$
$= 9(3) + 4(4) = 27 + 16 = 43$

사용된 핵심 성질:
- $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
- $\text{Var}[aX + b] = a^2 \text{Var}[X]$ (상수는 분산을 추가하지 않음)
- 독립인 경우: $\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y]$ $\square$

**[고등학교 수준]**

기댓값은 선형: 곱하고/더하기를 그대로 통과한다. 분산: 상수의 제곱이 앞에 붙고, 상수 +5는 사라지고(무작위성 없음), 독립인 분산은 더해진다. 결과: $\mathbb{E} = 13$, $\text{Var} = 43$.

**[중학교 수준]**

기댓값: 평균값으로 일반 방정식처럼 다룬다: $3 \times 2 - 2 \times (-1) + 5 = 13$. 분산: "퍼짐"을 측정한다. 3을 곱하면 퍼짐이 3배(분산 × 9), 마이너스 부호는 퍼짐에 영향 없고(여전히 × 4), 5를 더해도 퍼짐은 변하지 않는다. 총 퍼짐: $27 + 16 = 43$.

---

## 문제 23. 가우시안 분포와 CLT

**정규(가우시안) 분포가 "정규(normal)"라 불리는 두 가지 이유를 설명하라. 각각에 대한 수학적 정당화를 제시하라.**

### 풀이

**[대학 수준]**

**이유 1: 최대 엔트로피**

주어진 평균 $\mu$와 분산 $\sigma^2$을 가진 모든 분포 중, 가우시안 분포가 미분 엔트로피를 최대화한다:
$H(p) = -\int p(x) \log p(x) dx$

이는 제약 $\int p(x)dx = 1$, $\int x \cdot p(x)dx = \mu$, $\int (x-\mu)^2 p(x)dx = \sigma^2$을 가진 변분법 / 라그랑주 승수로 보일 수 있다. 가우시안은 평균과 분산 정보만 주어졌을 때 "가장 불확실한"(가장 편향이 적은) 분포이다.

**이유 2: 중심극한정리(CLT)**

평균 $\mu$와 유한 분산 $\sigma^2$을 가진 i.i.d. 확률변수 $X_1, \ldots, X_n$에 대해:
$$Z_n = \sqrt{n} \frac{\hat{\mu} - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

여기서 $\hat{\mu} = \frac{1}{n}\sum X_i$. 이는 원래 분포에 관계없이 많은 독립 확률변수의 합/평균이 가우시안에 수렴함을 의미한다. 이 편재성이 "정규"라는 이름을 부여한다. $\square$

**[고등학교 수준]**

1. **가장 불확실**: 평균과 분산만 주어졌을 때, 가우시안은 추가적인 가정을 가장 적게 하는 분포(최대 엔트로피).
2. **CLT**: *어떤* 분포로부터든 충분히 많은 무작위 수를 평균내면, 결과가 가우시안처럼 보인다. 그래서 자연 어디에나 나타나며 — "정규"인 것이다.

**[중학교 수준]**

1. 종 모양 곡선은 "가장 공정한" 형태이다: 평균과 퍼짐만 알면 가우시안이 가장 균형 잡힌 추측이다. 2. 많은 무작위한 것을 더하면(사람들의 키나 측정 오차처럼), 결과는 항상 종 모양 곡선처럼 보인다. 그래서 "정규"라 불린다 — 자연에서 가장 흔한 형태이다.

---

## 문제 24. 체비셰프 부등식

**표본 평균에 대한 체비셰프 부등식을 기술하라. $[0, 1]$에서의 i.i.d. 표본의 표본 평균이 참 평균의 0.1 이내에 있을 확률이 0.95 이상이려면 몇 개의 표본이 필요한지 결정하라.**

### 풀이

**[대학 수준]**

**체비셰프 부등식 (표본 평균):**
$P(X_i \in [a,b]) = 1$인 i.i.d. $X_1, \ldots, X_n$에 대해:

$$P(|\hat{\mu}_n - \mathbb{E}[X_1]| \geq \epsilon) \leq \frac{(b-a)^2}{4n\epsilon^2}$$

**적용:** $[a,b] = [0,1]$, $\epsilon = 0.1$, $P(\text{오차} \geq \epsilon) \leq 0.05$ (즉, $\delta = 0.05$)를 원한다.

$\frac{(1-0)^2}{4n(0.1)^2} \leq 0.05$

$\frac{1}{0.04n} \leq 0.05$

$n \geq \frac{1}{0.04 \times 0.05} = \frac{1}{0.002} = 500$

**최소 500개의 표본이 필요하다.**

호프딩 부등식(더 촘촘한 바운드)과 비교:
$2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right) \leq 0.05$

$n \geq \frac{(b-a)^2 \log(2/\delta)}{2\epsilon^2} = \frac{1 \cdot \log(40)}{0.02} \approx \frac{3.69}{0.02} \approx 185$

호프딩은 같은 보장에 ~185개 표본만 필요하다. $\square$

**[고등학교 수준]**

체비셰프: 표본 평균이 $\epsilon$ 이상 벗어날 확률은 최대 $(b-a)^2/(4n\epsilon^2)$. 이를 $\leq 0.05$로 놓고 $n$에 대해 풀면: $n \geq 500$. 호프딩 바운드가 더 촘촘하여 $n \geq 185$.

**[중학교 수준]**

0과 1 사이의 무작위 수의 평균을 추정하면서, 추정값이 실제 값의 0.1 이내에 있을 확률이 95%이려면, 최소 500개의 표본이 필요하다(기본 공식). 더 정교한 공식(호프딩)은 실제로 185개면 충분하다고 말한다.

---

## 문제 25. KL 발산과 교차 엔트로피

**KL 발산 $KL(p \| q)$와 교차 엔트로피 $CE(p, q)$를 정의하라. $H(p)$가 엔트로피일 때, $KL(p \| q) = CE(p, q) - H(p)$임을 보이라. KL 발산이 항상 비음인 이유는?**

### 풀이

**[대학 수준]**

**정의:**
- 엔트로피: $H(p) = -\sum_x p(x) \log p(x) = -\mathbb{E}_{x \sim p}[\log p(x)]$
- 교차 엔트로피: $CE(p, q) = -\sum_x p(x) \log q(x) = -\mathbb{E}_{x \sim p}[\log q(x)]$
- KL 발산: $KL(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$

**관계:**
$KL(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \sum_x p(x) [\log p(x) - \log q(x)]$
$= -H(p) + CE(p, q)$

따라서: $KL(p \| q) = CE(p, q) - H(p)$ $\square$

**비음수성 (깁스 부등식):**
젠센 부등식에 의해 ($-\log$가 볼록이므로):
$KL(p \| q) = -\mathbb{E}_p\left[\log \frac{q(x)}{p(x)}\right] \geq -\log \mathbb{E}_p\left[\frac{q(x)}{p(x)}\right] = -\log \sum_x p(x) \frac{q(x)}{p(x)} = -\log \sum_x q(x) = -\log 1 = 0$

등호는 $p = q$일 때만 성립한다. $\square$

**ML 맥락에서:** $\theta$에 대해 $CE(p_E, p_\theta)$를 최소화하는 것은 $H(p_E)$가 상수이므로 $KL(p_E \| p_\theta)$를 최소화하는 것과 동치이다. 이는 MLE와 동치이다.

**[고등학교 수준]**

KL 발산은 두 분포가 얼마나 다른지를 측정한다. 교차 엔트로피 빼기 엔트로피와 같다: $KL = CE - H$. $CE$가 항상 $H$ 이상이므로(잘못된 분포 $q$의 추가 비트를 사용), $KL \geq 0$.

**[중학교 수준]**

엔트로피는 분포 $p$의 메시지를 인코딩하는 "이상적인" 비용이다. 교차 엔트로피는 대신 분포 $q$의 인코딩을 사용하는 비용이다. KL 발산 = 잘못된 인코딩을 사용하는 *추가* 비용. 잘못된 코드를 사용하는 것이 이상적인 것보다 *더 좋을* 수 없으므로, $KL$은 항상 $\geq 0$이다.

---

# 파트 5: 베이지안 확률 & 정보이론 (Q26-Q30)

---

## 문제 26. 사전확률, 우도, 사후확률

**$\log p(H|E) = \log p(E|H) + \log p(H) - \log p(E)$ 프레임워크에서 각 항의 역할을 설명하라. 강의에서 "탱고는 둘이 추는 것"(ML과 Prior)이라고 하는 이유는?**

### 풀이

**[대학 수준]**

MAP 추정의 항들:
- $\log p(H|E)$: **사후확률** — 증거 $E$를 관찰한 후 가설 $H$에 대한 업데이트된 믿음
- $\log p(E|H)$: **로그우도** — $H$가 데이터를 얼마나 잘 설명하는지. 이것만 최대화 = MLE
- $\log p(H)$: **로그사전확률** — 데이터를 보기 전 $H$에 대한 믿음. 정규화로 작용
- $\log p(E)$: **로그증거** (주변 우도) — 정규화 상수, $H$와 독립

**"탱고는 둘이 추는 것":**

학습에는 둘 다 필요하다:
1. **최대우도** (데이터 기반): 가설이 관찰을 얼마나 잘 설명하는가? → 최적화 (손실)
2. **사전확률** (지식 기반): 선험적으로 무엇을 믿는가? → 정규화, 일반화

신경망 훈련에서: 손실 함수 = 음의 로그우도, 가중치 감쇠 = 가우시안 사전확률, 드롭아웃 = 암묵적 사전확률. 이 두 구성요소(데이터 적합 + 사전 지식 통합)의 상호작용이 좋은 일반화에 필수적이다.

ML 없이: 데이터로부터 학습 없음. 사전확률 없이: 과적합(3/3 앞면에서 MLE가 $\theta = 1$을 주는 동전 예시처럼). $\square$

**[고등학교 수준]**

MAP = ML + 사전확률. 우도가 데이터에 적합하고, 사전확률이 극단적 답을 방지한다. "탱고는 둘이 추는 것"처럼 — 두 파트너(데이터와 지식) 모두 좋은 학습에 필요하다.

**[중학교 수준]**

학습은 의견을 형성하는 것과 같다: 증거(데이터/우도)를 보되 상식(사전 지식)도 사용한다. 누군가 3번의 시험에서 모두 A를 받으면, 데이터는 "완벽한 학생"이라 하지만 상식은 "아마 매우 잘하지만, 반드시 완벽하지는 않다"고 한다. 둘 다 필요하다.

---

## 문제 27. 엔트로피 최대화

**유한 집합 $S$($|S| = K$) 위의 이산 균일분포가 $S$ 위의 모든 분포 중 엔트로피 $H(p) = -\sum_{i=1}^K p_i \log p_i$를 최대화함을 증명하라.**

### 풀이

**[대학 수준]**

라그랑주 승수를 사용한다. $g(p) = \sum_i p_i - 1 = 0$ 제약 하에서 $H(p) = -\sum_i p_i \log p_i$를 최대화한다.

$\mathcal{L}(p, \lambda) = -\sum_i p_i \log p_i + \lambda(1 - \sum_i p_i)$

$\frac{\partial \mathcal{L}}{\partial p_i} = -\log p_i - 1 - \lambda = 0$

$\Rightarrow \log p_i = -(1 + \lambda)$ (모든 $i$에 대해)

$\Rightarrow p_i = e^{-(1+\lambda)}$ = 모든 $i$에 대해 상수

$\sum p_i = 1$을 사용하면: $K \cdot p_i = 1 \Rightarrow p_i = 1/K$ (균일).

$H_{max} = -\sum_{i=1}^K \frac{1}{K} \log \frac{1}{K} = \log K$.

**KL을 이용한 대안 증명:** 임의의 분포 $p$에 대해, $u$를 균일분포라 하자.
$KL(p \| u) = \sum p_i \log \frac{p_i}{1/K} = \sum p_i \log p_i + \log K = -H(p) + \log K \geq 0$

따라서 $H(p) \leq \log K$이고, 등호는 $p = u$일 때만 성립한다. $\square$

**[고등학교 수준]**

도함수를 0으로 놓으면 모든 $p_i$가 같아야 함을 보여준다. 합이 1이므로 $p_i = 1/K$. 최대 엔트로피 = $\log K$. 또는: 균일분포로부터의 KL 발산 $\geq 0$이므로 엔트로피 $\leq \log K$.

**[중학교 수준]**

엔트로피는 "놀라움" 또는 "불확실성"을 측정한다. 모든 결과가 동일하게 일어날 때(공정한 주사위처럼) 가장 불확실하다. 어떤 결과에 편향이 생기면 더 예측 가능해진다(엔트로피 감소). 따라서 균일 분포 = 최대 불확실성 = 최대 엔트로피.

---

## 문제 28. 상호정보량

**상호정보량 $I(X; Y)$를 정의하고, $I(X; Y) = KL(p(x, y) \| p(x)p(y))$임을 보이라. $I(X; Y) = 0$은 무엇을 의미하는가?**

### 풀이

**[대학 수준]**

**정의:**
$I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$

$= \mathbb{E}_{(x,y) \sim p(x,y)}\left[\log \frac{p(x, y)}{p(x)p(y)}\right]$

이것은 KL 발산의 정의에 의해 정확히 $KL(p(x,y) \| p(x)p(y))$이다. $\square$

**대안적 표현:**
$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$

또한: $I(X; Y) = \mathbb{E}_{(x,y) \sim p(x,y)}\left[\log \frac{p(x|y)}{p(x)}\right]$

**$I(X; Y) = 0$의 의미:**
$KL(p(x,y) \| p(x)p(y)) = 0 \Leftrightarrow p(x,y) = p(x)p(y)$

즉, $X$와 $Y$는 **독립**이다. $Y$를 아는 것이 $X$에 대해 정보를 전혀 제공하지 않으며, 반대도 마찬가지이다.

$KL \geq 0$이므로, 항상 $I(X;Y) \geq 0$이며, 등호는 독립일 때만 성립한다. $\square$

**[고등학교 수준]**

상호정보량은 한 변수를 아는 것이 다른 변수에 대해 얼마나 알려주는지를 측정한다. 결합분포와 주변분포의 곱 사이의 KL 발산이다. $I = 0$은 변수가 독립임을 의미한다.

**[중학교 수준]**

상호정보량은 "X를 아는 것이 Y를 예측하는 데 도움이 되는가?"에 답한다. 그렇다면 $I > 0$. X를 안다고 Y에 대해 전혀 모른다면 $I = 0$(독립). 예: 기온과 아이스크림 매출은 높은 $I$를 갖고; 기온과 당신의 생일은 $I \approx 0$이다.

---

## 문제 29. 전체 기댓값의 법칙

**전체 기댓값의 법칙 $\mathbb{E}_X[X] = \mathbb{E}_Y[\mathbb{E}_{X|Y}[X|Y]]$를 기술하고 증명하라. $X | N \sim \mathcal{N}(N, 1)$이고 $N \sim \text{Poisson}(\lambda)$일 때, $\mathbb{E}[X]$를 계산하는 데 적용하라.**

### 풀이

**[대학 수준]**

**증명 (연속 경우):**
$\mathbb{E}_Y[\mathbb{E}[X|Y]] = \int \left(\int x \cdot p(x|y) dx \right) p(y) dy$
$= \int \int x \cdot p(x|y) \cdot p(y) \, dx \, dy$
$= \int \int x \cdot p(x, y) \, dx \, dy$
$= \int x \left(\int p(x, y) dy\right) dx$
$= \int x \cdot p(x) dx = \mathbb{E}[X]$ $\square$

**적용:**
$\mathbb{E}[X|N] = N$ ($\mathcal{N}(N, 1)$의 평균)

전체 기댓값의 법칙에 의해:
$\mathbb{E}[X] = \mathbb{E}_N[\mathbb{E}[X|N]] = \mathbb{E}_N[N] = \lambda$

(포아송의 $\mathbb{E}[N] = \lambda$이므로). $\square$

**[고등학교 수준]**

$\mathbb{E}[X]$를 구하려면: 먼저 각 $N$에 대해 $\mathbb{E}[X|N]$을 계산하고, $N$에 대해 평균을 낸다. 여기서 $\mathbb{E}[X|N] = N$이므로, $\mathbb{E}[X] = \mathbb{E}[N] = \lambda$.

**[중학교 수준]**

$X$의 평균을 구하려면: 먼저 가능한 각 $N$에 대해 $X$의 평균을 구하고, 그 평균들을(각 $N$이 얼마나 가능한지에 따라 가중하여) 평균낸다. $X$가 $N$을 중심으로 하고, $N$이 $\lambda$로 평균을 내므로, $X$도 $\lambda$로 평균을 낸다.

---

## 문제 30. 포아송 분포: 평균과 분산

**확률질량함수 $p(X = k) = \frac{\lambda^k \exp(-\lambda)}{k!}$ ($k = 0, 1, 2, \ldots$)인 포아송 분포의 평균과 분산을 계산하라. 힌트: $\exp(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$**

### 풀이

**[대학 수준]**

**평균:**
$\mathbb{E}[X] = \sum_{k=0}^\infty k \cdot \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=1}^\infty \frac{\lambda^k}{(k-1)!}$ ($k=0$ 항은 사라짐)

$j = k-1$로 놓으면:
$= e^{-\lambda} \sum_{j=0}^\infty \frac{\lambda^{j+1}}{j!} = \lambda e^{-\lambda} \sum_{j=0}^\infty \frac{\lambda^j}{j!} = \lambda e^{-\lambda} \cdot e^\lambda = \lambda$

**분산:** 먼저 $\mathbb{E}[X(X-1)]$을 계산한다:
$\mathbb{E}[X(X-1)] = \sum_{k=2}^\infty k(k-1) \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=2}^\infty \frac{\lambda^k}{(k-2)!}$

$j = k-2$로 놓으면:
$= e^{-\lambda} \lambda^2 \sum_{j=0}^\infty \frac{\lambda^j}{j!} = \lambda^2$

따라서: $\mathbb{E}[X^2] = \mathbb{E}[X(X-1)] + \mathbb{E}[X] = \lambda^2 + \lambda$

$\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \lambda^2 + \lambda - \lambda^2 = \lambda$

**결과:** 포아송의 경우, 평균 = 분산 = $\lambda$. $\square$

**[고등학교 수준]**

테일러 급수 $e^\lambda = \sum \lambda^k/k!$를 사용하여 합을 단순화한다. $j = k-1$로 치환하여 인덱스를 이동시킨다. 평균과 분산 모두 $\lambda$와 같다 — 포아송 분포의 고유한 성질이다.

**[중학교 수준]**

포아송 분포는 희귀 사건을 세는 데 사용된다(시간당 이메일 수 등). 평균은 $\lambda$이고, 퍼짐(분산)도 $\lambda$이다. 이것은 특별한 성질이다: 시간당 이메일 5개를 기대하면 퍼짐도 5이다. $e^x$의 무한 급수 공식을 사용하여 이를 증명한다.

---

# 다룬 주제 요약

| 파트 | 주제 | 문항 |
|------|------|------|
| 1 | 선형대수 (변환, 랭크, 고유값, SVD, PD 행렬) | Q1-Q8 |
| 2 | 행렬 미적분 (연쇄 법칙, 그래디언트, 야코비안, 소프트맥스) | Q9-Q13 |
| 3 | 미적분 & 최적화 (뉴턴, 라그랑주, 경사 하강, ODE) | Q14-Q18 |
| 4 | 확률 (베이즈, MLE, MAP, 기댓값, CLT, 부등식) | Q19-Q25 |
| 5 | 베이지안 확률 & 정보이론 (사전/사후확률, 엔트로피, KL, MI) | Q26-Q30 |

---

*이 모의시험은 이성윤 교수의 딥러닝 강의 슬라이드(3-265페이지)를 기반으로, 수학적 엄밀성, 연역적 추론, 단계별 풀이 과정에 대한 교수의 강조를 반영하여 준비되었습니다.*

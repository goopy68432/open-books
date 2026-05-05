---
title: "Level 2: Intermediate - 기본 유도 과정 (Basic Derivations)"
slug: 02-intermediate-01-basic-derivations
order: 4
---

# Level 2: Intermediate - 기본 유도 과정 (Basic Derivations)

> **학습 목표:** 기출문제 중 '고유값', '균일분포', '정규분포' 3가지의 수식 유도 과정을 비약 없이, 한 줄 한 줄 **왜(Why)** 그렇게 넘어가는지 이해하며 따라 써봅니다.

---

## 1. 고유값과 고유벡터 기출문제 유도

**문제:** $A = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ 의 고유값과 고유벡터를 구하시오.

### Step-by-Step 유도 체인

1. **정의에서 출발:** 고유값/고유벡터의 정의는 $Av = \lambda v$ 입니다.
   * **왜?** 행렬 $A$를 곱한 것($Av$)과 단순히 숫자 $\lambda$를 곱한 것($\lambda v$)이 같아지는 벡터 $v$를 찾기 위함입니다. 단, $v = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ 이면 의미가 없으므로 $v \neq 0$ 을 가정합니다.
2. **이항하여 묶기:** 우변을 좌변으로 넘기면 $Av - \lambda v = 0$ 이고, 묶으면 $(A - \lambda I)v = 0$ 이 됩니다.
   * **왜 $I$가 붙나요?** $A$는 행렬이고 $\lambda$는 숫자라 바로 뺄 수 없습니다. 그래서 단위행렬 $I$를 곱해 행렬 모양을 맞춰줍니다.
3. **특성방정식 세우기:** $v \neq 0$ 인 해를 가지려면, 앞에 있는 행렬 $(A - \lambda I)$ 의 행렬식(det)이 $0$이 되어야 합니다. 즉, $\det(A - \lambda I) = 0$.
   * **왜 행렬식이 0인가요?** 행렬식이 0이 아니면 역행렬이 존재해서 양변에 곱해버리면 $v = 0$이 되어버립니다. 우리는 $v \neq 0$인 답을 원하므로, 역행렬이 없어야(det=0) 합니다.
4. **고유값 계산:**
   * $A - \lambda I = \begin{bmatrix} -\lambda & 1 \\ 1 & -\lambda \end{bmatrix}$
   * $\det = (-\lambda)(-\lambda) - (1)(1) = \lambda^2 - 1 = 0$
   * 따라서 고유값은 **$\lambda = 1$ 또는 $\lambda = -1$** 입니다.
5. **고유벡터 계산:**
   * **$\lambda = 1$ 일 때:** $\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ $\implies -x+y=0 \implies x=y$. 크기를 1로 맞추면 $v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$
   * **$\lambda = -1$ 일 때:** $\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$ $\implies x+y=0 \implies x=-y$. 크기를 1로 맞추면 $v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$

---

## 2. 균일분포 기출문제 유도

**문제:** $X \sim \text{Uniform}[a, b]$ 일 때, 기댓값 $E(X)$ 와 분산 $Var(X)$ 를 구하시오.

### Step-by-Step 유도 체인

1. **확률밀도함수(PDF) 정의:** 구간 길이는 $(b-a)$ 이고, 넓이가 1이 되어야 하는 직사각형이므로 $p(x) = \frac{1}{b-a}$ 입니다.
2. **기댓값 $E(X)$ 구하기:**
   * $E(X) = \int_a^b x \cdot p(x) dx = \int_a^b x \cdot \frac{1}{b-a} dx$
   * **왜 $x$를 곱하나요?** 평균을 구하려면 (값 $\times$ 확률)을 다 더해야 하기 때문입니다.
   * 적분 계산: $\frac{1}{b-a} \left[ \frac{1}{2}x^2 \right]_a^b = \frac{1}{2(b-a)} (b^2 - a^2) = \frac{(b-a)(b+a)}{2(b-a)} = \mathbf{\frac{a+b}{2}}$
   * **직관:** 딱 $a$와 $b$의 한가운데(평균)가 나옵니다.
3. **분산 $Var(X)$ 구하기:**
   * $Var(X) = E(X^2) - \{E(X)\}^2$
   * 먼저 $E(X^2)$ 를 구합니다: $\int_a^b x^2 \frac{1}{b-a} dx = \frac{1}{b-a} \left[ \frac{1}{3}x^3 \right]_a^b = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}$
   * 이제 빼줍니다: $\frac{a^2 + ab + b^2}{3} - \frac{a^2 + 2ab + b^2}{4} = \mathbf{\frac{(b-a)^2}{12}}$

---

## 3. 정규분포 모멘트 기출문제 유도

**문제:** $X \sim \mathcal{N}(0, 1)$ 일 때, $E(X), E(X^2), E(X^3), E(X^4)$ 를 구하시오.

### Step-by-Step 유도 체인

*표준정규분포의 PDF:* $p(x) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{x^2}{2})$

1. **홀수 모멘트 ($E(X), E(X^3)$):**
   * $E(X) = \int_{-\infty}^{\infty} x \cdot p(x) dx = \mathbf{0}$
   * $E(X^3) = \int_{-\infty}^{\infty} x^3 \cdot p(x) dx = \mathbf{0}$
   * **왜 계산 없이 0인가요?** $p(x)$는 우함수(y축 대칭)인데, $x$나 $x^3$은 기함수(원점 대칭, 홀함수)입니다. 우함수 $\times$ 기함수 = 기함수입니다. 기함수를 $-\infty$부터 $\infty$까지 적분하면 왼쪽(-)과 오른쪽(+) 넓이가 똑같아 상쇄되므로 무조건 0입니다.
2. **$E(X^2)$ 구하기:**
   * $E(X^2) = \int_{-\infty}^{\infty} x^2 \cdot \frac{1}{\sqrt{2\pi}} \exp(-\frac{x^2}{2}) dx$
   * 부분적분을 사용해야 합니다. $\int u \cdot v' = u \cdot v - \int u' \cdot v$
   * $u = x$, $v' = x \exp(-\frac{x^2}{2})$ 로 둡니다. (그러면 $v = -\exp(-\frac{x^2}{2})$ 입니다)
   * 계산을 마치면 결국 전체 확률의 합인 $\int p(x)dx$ 와 같아져서 **$1$** 이 됩니다. (표준정규분포의 분산이 1이라는 사실로도 직관적 증명 가능)
3. **$E(X^4)$ 구하기:**
   * 똑같이 부분적분을 합니다. $u = x^3$, $v' = x \exp(-\frac{x^2}{2})$ 로 둡니다.
   * $u \cdot v$ 항은 양끝에서 0으로 날아가고, $- \int u' \cdot v$ 항만 남습니다.
   * $3 \int_{-\infty}^{\infty} x^2 p(x) dx = 3 \cdot E(X^2) = 3 \cdot 1 = \mathbf{3}$
   * **규칙성 발견:** 짝수 모멘트는 $E(X^{2n}) = (2n-1)!! = 1 \cdot 3 \cdot 5 \cdots (2n-1)$ 이 됩니다.

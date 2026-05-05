---
title: "[파트 7 & 8] Pooling(풀링)과 Quadratic(이차 형식) - 데이터 다운샘플링과 표면 스캔"
slug: 07-pooling-and-quadratic
order: 7
---

# [파트 7 & 8] Pooling(풀링)과 Quadratic(이차 형식) - 데이터 다운샘플링과 표면 스캔

🔥 **한 줄 결론**
👉 "Pooling은 고해상도 데이터를 다운샘플링하는 희소 행렬(Sparse Matrix) 곱셈이며, Quadratic은 모델 최적화를 위해 Loss 지형의 곡률(Hessian)을 스캔하는 3D 레이더다."

---

## 1. Average Pooling 계산 및 행렬(Matrix) 표현

**[예제: 4x4 행렬에 2x2 Filter, Stride=2 적용]**
입력 데이터 $x$가 $4 \times 4$ 크기(총 16개 요소)일 때, 출력 $y$는 $2 \times 2$ 크기(총 4개 요소)가 됩니다.

1. **수치 계산:**
   $\begin{bmatrix} 1 & 3 & 2 & 4 \ 5 & 7 & 6 & 8 \ 0 & 2 & 1 & 1 \ 4 & 6 & 3 & 3 \end{bmatrix} \xrightarrow{Avg Pooling} \begin{bmatrix} \frac{1+3+5+7}{4} & \frac{2+4+6+8}{4} \\ \frac{0+2+4+6}{4} & \frac{1+1+3+3}{4} \end{bmatrix} = \begin{bmatrix} 4 & 5 \ 3 & 2 \end{bmatrix}$

2. **행렬 $P$ 변환 ($y = Px$):**
   * 입력 $x$를 $16 \times 1$ 벡터로 폅니다(Flatten).
   * 출력 $y$를 $4 \times 1$ 벡터로 폅니다.
   * 행렬 $P$는 $4 \times 16$ 크기의 희소 행렬이 되며, 각 행(출력 요소)은 4개의 위치에만 $0.25$ 가중치를 가지고 나머지 12개는 $0$으로 채워집니다.

---

## 2. Quadratic Form (이차 형식) - Loss 지형 스캔

이차 형식 $x^T A x$는 모델 파라미터 $x$의 최적화(Gradient Descent)를 위해 지형을 스캔하는 가장 기초적인 함수입니다.

**[기출: 이차 함수의 최소화]**
$$f(x) = \frac{1}{2} x^T A x - b^T x + c$$
(단, $A$는 대칭 행렬, $A = A^T$)

1. **Gradient (미분):**
   $\nabla f(x) = \frac{1}{2}(A + A^T)x - b = Ax - b$
2. **최적점 (최소화):**
   $\nabla f(x) = 0 \Rightarrow Ax = b$
   * 시스템 비유: $Ax = b$ 선형 방정식을 푸는 것은, Loss 곡면의 가장 밑바닥(Gradient=0)을 찾는 최적화 과정과 완벽히 동일합니다.
3. **Hessian과 고유값의 조건:**
   행렬 $A$의 고유값이 모두 양수(Positive Definite)여야만 이 지형이 아래로 볼록(Convex)하여 학습이 정상적으로 수렴합니다. 하나라도 음수면 말안장(Saddle Point)에 빠져 에러가 발생합니다.

---

📌 **핵심 정리 (시험 대비)**
* Pooling 행렬 $P$를 직접 그리는 문제에서는, 차원(Shape) 매칭과 0의 배치 간격(Stride 영향)을 정확히 표시해야 합니다.
* Quadratic 수식 미분 시 $x^T A x$ 가 $2Ax$ 가 되는 것이 아니라 $(A+A^T)x$ 임을 명시하고, $A$가 대칭이므로 $2Ax$가 되어 앞에 있던 $\frac{1}{2}$과 약분됨을 서술하세요.
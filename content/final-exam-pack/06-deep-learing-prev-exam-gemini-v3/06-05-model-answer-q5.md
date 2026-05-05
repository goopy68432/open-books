---
title: "📝 [V3] Model Answer - Q5. 최적화 구조 (Learning Rate, Pooling, Quadratic Form)"
slug: 05-model-answer-q5
order: 6
---

# 📝 [V3] Model Answer - Q5. 최적화 구조 (Learning Rate, Pooling, Quadratic Form)

**[평가 기준]**
* 오차(Error) 방정식 $e_{t+1} = (I - \eta A)e_t$ 구조를 직접 유도했는가?
* Pooling 필터 $P$ 행렬 작성 시 차원 일치와 0(희소 행렬)의 간격을 정확히 서술했는가?
* Hessian의 고유값과 지형의 Convexity(양의 정부호) 관계를 시스템 안정성 관점에서 설명했는가?

---

## (a) 학습률(Learning Rate) 수렴 조건 증명

**[1. Gradient 도출 및 최적점 정의]**
2차 목적함수 $f(x) = \frac{1}{2} x^T A x - b^T x$ 의 미분값(기울기)은 $\nabla f(x) = A x - b$ 입니다.
최적점 $x^*$ 에서는 기울기가 0이 되므로, $b = A x^*$ 입니다.

**[2. Gradient Descent 룰과 오차(Error) 치환]**
가중치 업데이트 식: $x_{t+1} = x_t - \eta \nabla f(x_t) = x_t - \eta (A x_t - b)$
목적지와의 오차를 $e_t = x_t - x^*$ 로 정의하고 식 양변에서 $x^*$를 뺍니다.
$$x_{t+1} - x^* = x_t - x^* - \eta(A x_t - A x^*)$$
$$e_{t+1} = e_t - \eta A e_t = (I - \eta A)e_t$$
*   **시스템 의미:** 매 스텝의 오차는 이전 오차에 증폭 필터 $(I - \eta A)$ 를 통과시킨 결과입니다.

**[3. 고유값 기반 한계 속도 계산]**
루프가 무한히 돌 때 $e_t \rightarrow 0$ 으로 수렴하려면, 증폭 행렬의 고유값이 절대값 1보다 작아야 합니다.
행렬 $A$ 의 고유값을 $\lambda_i$ 라 하면, $(I - \eta A)$ 의 고유값은 $1 - \eta \lambda_i$ 입니다.
$$|1 - \eta \lambda_i| < 1 \quad \Rightarrow \quad -1 < 1 - \eta \lambda_i < 1$$
$$-2 < -\eta \lambda_i < 0 \quad \Rightarrow \quad 0 < \eta \lambda_i < 2$$
이 조건은 시스템 내의 **모든 고유값(가속 엔진)**에 대해 성립해야 하므로, 가장 큰 고유값 $\lambda_{max}(A)$ 에 의해 제한됩니다.
$$\eta \lambda_{max}(A) < 2 \quad \therefore \eta < \frac{2}{\lambda_{max}(A)}$$

---

## (b) Average Pooling 계산 및 행렬 표현 (Matrix Transformation)

**[1. 수치 계산 (다운샘플링)]**
2x2 Filter, Stride 2 로 4x4 행렬을 스캔하면, 겹치지 않는 4개의 블록(사분면)에 대해 평균을 냅니다.
*   1사분면: $(2+4+6+8) / 4 = 5$
*   2사분면: $(1+3+5+7) / 4 = 4$
*   3사분면: $(1+3+5+7) / 4 = 4$
*   4사분면: $(0+2+4+6) / 4 = 3$
**결과 행렬:** $\begin{bmatrix} 5 & 4 \ 4 & 3 \end{bmatrix}$

**[2. 행렬 $P$ 변환 ($y = Px$)]**
딥러닝 백엔드 GPU는 루프 대신 이 연산을 거대한 희소 행렬(Sparse Matrix) $P$ 의 곱셈으로 치환합니다.
*   **차원 (Shape):** 입력 $x$ 가 $16 \times 1$ 열벡터로 Flatten 되고, 출력 $y$ 가 $4 \times 1$ 벡터가 되므로, 필터 행렬 $P$ 는 **$4 \times 16$** 차원을 갖습니다.
*   **배치 규칙:** $P$ 의 각 행(Row)은 출력의 각 픽셀을 생성합니다. 한 행당 정확히 **4개의 위치에만 $0.25$(평균 가중치)**가 배치되며, 나머지 12개 인덱스는 연산에서 배제하기 위해 모두 **$0$** 으로 채워집니다. Stride가 2이므로, 가중치 $0.25$ 는 원본 행렬의 행 바꿈에 따라 인덱스 간격을 띄우고 배치됩니다.

---

## (c) 이차 형식(Quadratic Form)과 Hessian 행렬의 Convexity

**[정의 및 시스템적 의미]**
*   이차 형식 $x^T A x$ 는 손실 함수의 지형(Topology)이 볼록한지 오목한지를 스캔하는 수학적 레이더입니다.
*   행렬 $A$ (Hessian)는 이러한 공간의 곡률(휘어짐) 정보를 담고 있는 코어 DB입니다.

**[Positive Definite의 필요성]**
*   **Convexity 확보:** 행렬 $A$ 의 모든 고유값(Eigenvalue)이 양수($>0$)일 때, 이 행렬은 '양의 정부호(Positive Definite)'라고 불립니다. 이 상태여야만 전체 Loss 지형이 완벽한 밥그릇 모양(Strictly Convex)을 띠게 됩니다.
*   **Saddle Point(말안장점) 방지:** 만약 고유값 중 하나라도 음수가 섞여 있다면, 특정 방향으로는 지형이 솟아오르고 다른 방향으로는 파이는 'Saddle Point'가 발생합니다. 이 경우 Gradient가 0임에도 최솟값이 아니므로, Gradient Descent 학습 서버가 길을 잃고 멈추거나 발산하는 치명적인 에러가 발생합니다.

---
title: "📝 [V3] Model Answer - Q4. 행렬 미분과 역전파 (Matrix Calculus & Backprop)"
slug: 04-model-answer-q4
order: 5
---

# 📝 [V3] Model Answer - Q4. 행렬 미분과 역전파 (Matrix Calculus & Backprop)

**[평가 기준]**
* Sigmoid 미분을 $p(1-p)$ 형태로 정확히 유도하였는가?
* 행렬 미분 시 차원 매칭(Shape Matching) 원리를 서술하였는가?
* Softmax Jacobian 미분을 몫의 미분법과 조건 분기($i=j$, $i 
eq j$)를 통해 논리적으로 분해했는가?

---

## (a) Sigmoid + NLL의 미분과 Chain Rule (역전파)

**[조건]**
* 선형 변환: $z = Ax + b$
* 활성화 함수: $p = \sigma(z) = \frac{1}{1 + e^{-z}}$
* 손실 함수: $L = -\log(p)$ (단, 정답 $y=1$ 인 경우)

**[1. 로컬 그래디언트 $\frac{\partial L}{\partial z}$ 도출 (스칼라 체인룰)]**
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z}$$
1. **$\frac{\partial L}{\partial p}$ 계산:** $L = -\log(p)$ 를 미분하면 $\Rightarrow -\frac{1}{p}$
2. **$\frac{\partial p}{\partial z}$ 계산:** $\sigma(z)$ 의 미분은 $p(1-p)$ 형태를 띱니다.
3. **Chain Rule 결합:** $\frac{\partial L}{\partial z} = \left(-\frac{1}{p}\right) \cdot p(1-p) = -(1-p) = p - 1$
   * (정답 $y=1$ 이므로, 일반화하면 $p-y$ 형태의 오차값이 됨)

**[2. 파라미터 행렬 $\frac{\partial L}{\partial A}$ 로의 확장 (Shape Matching)]**
위에서 구한 에러 벡터를 $\delta = \frac{\partial L}{\partial z}$ (크기 $N \times 1$)라고 정의합니다.
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial A}$$
*   **시스템/차원 분석:** $z = Ax + b$ 에서, $A$ 행렬이 $N \times M$, 입력 $x$ 가 $M \times 1$ 이라고 할 때, 에러 $\delta$ 와 입력 $x$ 를 곱하여 원본 $A$ 의 차원($N \times M$)을 완벽히 복원해야 합니다.
*   **결론:** 에러 신호(N차원 열벡터)와 입력 데이터(M차원 열벡터)의 **외적(Outer Product)**을 수행합니다.

$$\therefore \frac{\partial L}{\partial A} = \delta x^T$$

---

## (b) Softmax 출력 벡터의 Jacobian 미분 증명

**[정의]** 다중 분류의 출력 인터페이스 $p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$
입력 $z_j$ 가 출력 $p_i$ 에 미치는 영향을 구하기 위해 $\frac{\partial p_i}{\partial z_j}$ 를 계산합니다. 몫의 미분법 $\left( \frac{f}{g} \right)' = \frac{f'g - fg'}{g^2}$ 를 사용합니다.

**Case 1: $i = j$ (자신의 입력이 자신의 출력에 미치는 영향)**
*   분자 $f = e^{z_i}$, 분모 $g = \sum_k e^{z_k}$.
*   분자 미분 $f' = e^{z_i}$ (자기 자신이므로 미분값이 살아남음).
*   분모 미분 $g' = e^{z_i}$ (합계 중 $z_i$ 항만 미분값이 살아남음).
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \sum e^{z_k} - e^{z_i} e^{z_i}}{(\sum e^{z_k})^2} = \frac{e^{z_i}}{\sum e^{z_k}} \cdot \frac{\sum e^{z_k} - e^{z_i}}{\sum e^{z_k}} = p_i (1 - p_i)$$
*   **응용(의미):** 내 입력 가중치가 커지면 내 확률도 커지는 '자기 강화' 구간.

**Case 2: $i \neq j$ (타인의 입력이 내 출력에 미치는 영향)**
*   분자 $f = e^{z_i}$ 를 $z_j$ 로 편미분하면 $0$ 이 됨 ($f' = 0$).
*   분모 미분 $g' = e^{z_j}$ (합계 중 $z_j$ 항만 살아남음).
$$\frac{\partial p_i}{\partial z_j} = \frac{0 - e^{z_i} e^{z_j}}{(\sum e^{z_k})^2} = -\frac{e^{z_i}}{\sum e^{z_k}} \cdot \frac{e^{z_j}}{\sum e^{z_k}} = -p_i p_j$$
*   **응용(의미):** 타인의 입력값이 커지면 내 확률은 무조건 깎이게 되는 확률의 'Zero-Sum' 파이 뺏기 구간.

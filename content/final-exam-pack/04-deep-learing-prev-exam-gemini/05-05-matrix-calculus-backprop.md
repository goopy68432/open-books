---
title: "[파트 5] 행렬 미분(Matrix Calculus)과 역전파(Backprop) - 데이터 파이프라인 역추적 로직"
slug: 05-matrix-calculus-backprop
order: 5
---

# [파트 5] 행렬 미분(Matrix Calculus)과 역전파(Backprop) - 데이터 파이프라인 역추적 로직

🔥 **한 줄 결론**
👉 "역전파는 에러(Loss)가 발생했을 때, Chain Rule을 이용해 연산 그래프를 거꾸로 타들어가며 각 파라미터($W, b$)의 책임(Gradient)을 계산하는 디버깅 과정이다."

---

## 1. 킬러 수식 1: Sigmoid + NLL의 미분

**[Forward 로직]** $z = Ax + b \rightarrow p = \sigma(z) \rightarrow L = -\log(p)$ (정답이 1인 경우)
**[목표]** $\frac{\partial L}{\partial z}$ 구하기
1. $L = -\log(p) \Rightarrow \frac{\partial L}{\partial p} = -\frac{1}{p}$
2. $p = \sigma(z) \Rightarrow \frac{\partial p}{\partial z} = p(1-p)$
3. **Chain Rule:** $\frac{\partial L}{\partial z} = \left(-\frac{1}{p}\right) \cdot p(1-p) = p - 1$
   * 백엔드 비유: (내 예측값 $p$) - (실제 정답 $1$) $\rightarrow$ 직관적인 에러 차이값 도출.

---

## 2. 킬러 수식 2: Softmax 미분 (Jacobian Matrix)

출력 벡터 $p_i = \frac{e^{z_i}}{\sum e^{z_k}}$ 를 입력 벡터 $z_j$ 로 편미분 (몫의 미분법).

**Case 1: $i = j$ (자신의 입력이 자신의 출력에 미치는 영향)**
$\frac{\partial}{\partial z_i} \left( \frac{e^{z_i}}{\sum e^{z_k}} \right) = \frac{e^{z_i} \sum e^{z_k} - e^{z_i} e^{z_i}}{(\sum e^{z_k})^2} = p_i (1 - p_i)$ (자기 강화 로직)

**Case 2: $i \neq j$ (타인의 입력이 내 출력에 미치는 영향)**
$\frac{\partial}{\partial z_j} \left( \frac{e^{z_i}}{\sum e^{z_k}} \right) = \frac{0 - e^{z_i} e^{z_j}}{(\sum e^{z_k})^2} =-p_i p_j$ (Zero-sum 파이 뺏기 로직)

---

## 3. 킬러 수식 3: 전체 Backprop과 파라미터(W, b) 미분

위에서 구한 에러 스칼라 $\frac{\partial L}{\partial z} = (p - y)$ 를 바탕으로 선형 계층 $z = Wx + b$ 의 파라미터를 업데이트합니다.

1. **에러 신호 도달:** $\delta = \frac{\partial L}{\partial z} = p - y$ (벡터 차원 $N \times 1$)
2. **편향(b) 미분:** $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} = \delta \cdot 1 = p - y$ (상수항이므로 에러 신호 그대로 흡수)
3. **가중치(W) 미분 (Shape Matching):** $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} x^T = \delta x^T$
   *   $W$가 $N \times M$ 차원이면, 에러 $\delta$ ($N \times 1$)와 입력 $x$ ($M \times 1$)를 **외적(Outer Product)**하여 차원을 맞춥니다.

---

## 4. 번외 킬러: 행렬 미분을 통한 가우시안 평균/분산 (MLE) 유도

데이터 $X$에 대해 정규분포를 가정하고 파라미터 $\mu, \sigma^2$을 미분으로 찾는 기출문제입니다.
$$NLL(\mu, \sigma^2) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum (x_i - \mu)^2$$

1. **$\\mu$로 편미분 (평균 유도):**
   $\frac{\partial NLL}{\partial \mu} = \frac{1}{2\sigma^2} \sum 2(x_i - \mu)(-1) = 0$
   $\sum x_i - n\mu = 0 \Rightarrow \hat{\mu} = \frac{1}{n}\sum x_i$ (표본 평균)

2. **$\\sigma^2$로 편미분 (분산 유도):**
   $\\sigma^2$를 하나의 변수 $v$로 보고 미분: $\frac{n}{2v} - \frac{1}{2v^2}\\sum(x_i - \mu)^2 = 0$
   $\frac{n}{v} = \frac{\\sum(x_i - \mu)^2}{v^2} \Rightarrow \hat{\\sigma}^2 = \frac{1}{n}\\sum (x_i - \hat{\\mu})^2$ (표본 분산)
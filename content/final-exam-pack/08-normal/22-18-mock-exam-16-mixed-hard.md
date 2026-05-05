---
title: "딥러닝 이론 모의고사 #16 — 종합 심화 (최종)"
slug: 18-mock-exam-16-mixed-hard
order: 22
---

# 딥러닝 이론 모의고사 #16 — 종합 심화 (최종)

> 배점 100점 / 8문제 / 이것만 풀면 A+ 확정

---

## 문제 1. [15점] 확률→Loss→정규화 대통합 유도

아래 세 경로를 **빠짐없이** 유도하시오 (각 단계에 "왜").

**(a)** [5점] Gaussian → MSE → Ridge (MAP with Gaussian Prior)
**(b)** [5점] Categorical → CE → CE + L2 Reg
**(c)** [5점] Laplace noise → MAE → LASSO (MAP with Laplace Prior)

---

## 문제 2. [12점] Softmax의 3가지 얼굴

**(a)** [4점] 확률 관점: softmax가 확률 분포의 조건을 만족하는 이유 (비음수, 합=1)
**(b)** [4점] 최적화 관점: 라그랑주 유도에서 softmax가 유일한 해인 이유
**(c)** [4점] 미분 관점: Softmax Jacobian $\text{diag}(p) - pp^\top$를 유도하고, CE + Softmax 조합의 그래디언트가 $p - e_y$ (예측-정답)로 간결하게 정리됨을 보이시오.

---

## 문제 3. [12점] 역전파의 수학적 완전 분해

3층 네트워크: $z_1 = W_1x$, $a_1 = \sigma(z_1)$, $z_2 = W_2a_1$, $a_2 = \text{softmax}(z_2)$, $L = \text{CE}(y, a_2)$

**(a)** [3점] $\frac{\partial L}{\partial z_2}$를 구하시오 (CE + Softmax의 결합).
**(b)** [3점] $\frac{\partial L}{\partial W_2}$를 구하시오.
**(c)** [3점] $\frac{\partial L}{\partial a_1}$을 구하시오 (upstream gradient 전파).
**(d)** [3점] $\frac{\partial L}{\partial W_1}$을 구하시오 ($\sigma'$ 포함).

---

## 문제 4. [12점] 베이지안의 깊은 이해

**(a)** [4점] 동전 100번 중 60번 앞면. MLE, MAP(Beta(5,5)), MAP(Beta(50,50))를 각각 구하시오.
**(b)** [4점] 세 추정값을 비교하고, Prior 강도에 따른 변화를 분석하시오.
**(c)** [4점] 데이터가 1000번 중 600번이면 세 값이 어떻게 변하는지 계산하고, "데이터가 많으면 Prior 무관"을 정량적으로 확인하시오.

---

## 문제 5. [12점] 정보이론의 실전 적용

**(a)** [4점] 학습이 잘 된 모델 $q^*$에서 $KL(p\|q^*) \approx 0$이다. 이것의 의미를 설명하시오.
**(b)** [4점] VAE에서 KL 항 $KL(q(z|x)\|p(z))$의 역할을 설명하시오 (p(z) = 표준 정규).
**(c)** [4점] Mutual Information $I(X;Y) = KL(p(X,Y)\|p(X)p(Y))$의 의미와, 이것이 0이면 X,Y가 독립임을 설명하시오.

---

## 문제 6. [12점] 행렬 분해와 딥러닝

**(a)** [4점] 가중치 행렬 $W$의 SVD $W = U\Sigma V^\top$에서 특이값이 작은 성분을 잘라내면 어떤 효과가 있는가? (모델 압축 관점)
**(b)** [4점] LoRA(Low-Rank Adaptation)가 $\Delta W = BA$ ($B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$)로 미세조정하는 이유를 SVD 관점에서 설명하시오.
**(c)** [4점] Attention의 $QK^\top$이 큰 행렬($N \times N$)인 이유와, 이를 줄이기 위한 Linear Attention의 아이디어를 서술하시오.

---

## 문제 7. [13점] 최적화의 난제

**(a)** [4점] 딥러닝의 Loss landscape가 비볼록인데도 SGD가 잘 작동하는 이유에 대한 가설 2가지를 제시하시오.

**(b)** [5점] 다음 최적화 방법의 수식과 차이를 쓰시오:
- SGD: $\theta \leftarrow \theta - \alpha\nabla L$
- SGD+Momentum: $v \leftarrow \beta v + \nabla L$, $\theta \leftarrow \theta - \alpha v$
- Adam: adaptive learning rate + momentum

**(c)** [4점] Learning Rate가 너무 크면/작으면 각각 어떤 문제가 생기는가?

---

## 문제 8. [12점] 최종 종합: "딥러닝은 왜 작동하는가?"

다음 관점에서 각 2-3줄로 답하시오.

**(a)** [3점] 표현력(Expressivity) 관점: Universal Approximation Theorem
**(b)** [3점] 최적화 관점: SGD + 비볼록 landscape에서의 학습
**(c)** [3점] 일반화 관점: 왜 과적합하지 않는가 (implicit regularization)
**(d)** [3점] 확률 관점: MLE/MAP 프레임워크와 Loss 함수의 관계

---
---

# 모범답안

## 답 2.
### (c) CE + Softmax 그래디언트
CE: $L = -e_y^\top \log p$, $p = \text{softmax}(z)$

$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial z}$

$\frac{\partial L}{\partial p_i} = -e_{y,i}/p_i$ (정답 클래스만 $-1/p_y$, 나머지 0)

$\frac{\partial p}{\partial z} = \text{diag}(p) - pp^\top$

결합하면: $\frac{\partial L}{\partial z_j} = \sum_i (-e_{y,i}/p_i)(p_i\delta_{ij} - p_ip_j)$
$= -e_{y,j} + p_j\sum_i e_{y,i} = -e_{y,j} + p_j = p_j - e_{y,j}$

$$\boxed{\frac{\partial L}{\partial z} = p - e_y}$$

놀라울 정도로 간결! 예측 - 정답. 정답이면 작은 그래디언트, 틀리면 큰 그래디언트.

## 답 3.
(a) $\partial L/\partial z_2 = a_2 - e_y$ (위의 CE+Softmax 결합)
(b) $\partial L/\partial W_2 = (a_2 - e_y) a_1^\top$ (외적)
(c) $\partial L/\partial a_1 = W_2^\top(a_2 - e_y)$ (upstream을 역방향 전파)
(d) $\partial L/\partial W_1 = [\text{diag}(\sigma'(z_1)) W_2^\top(a_2-e_y)] x^\top$

## 답 4.
### (a)
공식: $\theta_{MAP} = \frac{k+\alpha-1}{n+\alpha+\beta-2}$

| | MLE | MAP(Beta(5,5)) | MAP(Beta(50,50)) |
|--|-----|---------------|-----------------|
| $n=100,k=60$ | 0.600 | 64/108=0.593 | 109/198=0.551 |

### (c)
| | MLE | MAP(Beta(5,5)) | MAP(Beta(50,50)) |
|--|-----|---------------|-----------------|
| $n=1000,k=600$ | 0.600 | 604/1008=0.599 | 649/1098=0.591 |

$n=1000$이면 세 값 모두 0.6에 수렴. 차이가 $O(1/n)$으로 줄어듦. ✓

## 답 7.
(a) 가설: (1) 고차원에서 대부분의 saddle point는 탈출 가능 (부정 고유값 방향), (2) SGD의 노이즈가 sharp minima에서 탈출하여 flat minima(일반화 좋음)로 수렴 유도.

(c) 너무 크면: 발산 (overshooting), Loss가 진동하거나 증가.
너무 작으면: 수렴 매우 느림, local minima에 갇힘, 학습 시간 낭비.

## 답 8.
(a) UAT: 충분히 넓은 단층 네트워크가 연속함수를 임의 정밀도로 근사 가능. 깊이를 더하면 더 효율적 근사.
(b) Loss landscape는 비볼록이지만 SGD는 stochastic noise 덕분에 sharp minima 탈출 + flat minima 수렴. 실전에서 "나쁜" local minima는 드물다는 경험적 증거.
(c) SGD 자체가 implicit regularization (작은 lr + large batch ≈ noise → flat solution 선호). Dropout, weight decay도 explicit reg. 과파라미터화가 오히려 일반화에 도움 (double descent).
(d) Loss는 NLL의 특수 형태. 학습 = 모델 분포를 데이터 분포에 접근시키는 것 = KL 최소화. MAP 관점에서 정규화 = Prior. 모든 것이 확률론적 프레임워크 안에서 통합.

---
title: "딥러닝 이론 모의고사 #11 — 연결/통합 심화"
slug: 13-mock-exam-11-connection
order: 17
---

# 딥러닝 이론 모의고사 #11 — 연결/통합 심화

> 배점 100점 / 8문제 / 개념 간 연결고리를 묻는 A+ 결정 문제

---

## 문제 1. [15점] 대통합: NLL 프레임워크

**(a)** [5점] "모든 Loss 함수는 NLL의 특수한 경우"임을 MSE, CE, MAE에 대해 각각 보이시오.
- MSE ← ? 분포
- CE ← ? 분포
- MAE ← ? 분포

**(b)** [5점] 이 프레임워크에서 "Loss 함수 선택 = 확률 가정 선택"이 왜 깊은 통찰인지 서술하시오.

**(c)** [5점] 만약 노이즈가 Student-t 분포를 따른다면 Loss는 어떤 형태가 되겠는가? 이것이 이상치(outlier)에 강건한 이유를 heavy tail과 연결하여 설명하시오.

---

## 문제 2. [12점] SVD → PCA → 차원축소 → 딥러닝

**(a)** [4점] SVD $A = U\Sigma V^\top$에서 저랭크 근사가 최적(Eckart-Young)인 이유를 서술하시오.
**(b)** [4점] PCA가 데이터 공분산 행렬의 고유값 분해임을 보이고, SVD와의 관계를 설명하시오.
**(c)** [4점] 오토인코더(Autoencoder)가 비선형 PCA라고 불리는 이유를 설명하시오.

---

## 문제 3. [12점] 체인룰 → 역전파 → Gradient Flow

$L = L(f_3(f_2(f_1(x;\theta_1);\theta_2);\theta_3))$에서:

**(a)** [4점] $\frac{\partial L}{\partial \theta_1}$을 체인룰로 전개하시오.
**(b)** [4점] 각 야코비안 $\frac{\partial f_i}{\partial f_{i-1}}$의 스펙트럴 노름이 1보다 작으면 어떤 문제가 생기는가?
**(c)** [4점] Residual Connection $f_i(x) = x + g_i(x)$이 이 문제를 해결하는 이유를 야코비안으로 보이시오.

---

## 문제 4. [12점] Prior → 아키텍처 → 정규화

**(a)** [4점] "CNN의 구조(locality + weight sharing) = 이미지에 대한 Prior"임을 설명하시오.
**(b)** [4점] Transformer가 CNN보다 약한 Prior를 가진다는 것을 구체적으로 설명하시오.
**(c)** [4점] 이로부터 "데이터가 적으면 CNN(강한 Prior), 많으면 Transformer(약한 Prior)"가 유리한 이유를 MAP 관점에서 설명하시오.

---

## 문제 5. [12점] Attention = Weighted Average = Softmax

**(a)** [4점] Attention의 출력 $y = \sum_i a_i v_i$가 "학습된 가중 평균"인 이유를 설명하시오.
**(b)** [4점] 가중치 $a_i = \text{softmax}(q^\top k_i / \sqrt{d})$에서 softmax를 쓰는 이유를 확률 관점에서 설명하시오.
**(c)** [4점] Softmax가 라그랑주 유도의 해라는 사실과, Attention에서의 역할을 연결하여 설명하시오.

---

## 문제 6. [12점] 베이즈 → MAP → 정규화 → 일반화

아래의 논리 체인을 완성하시오:

```
과적합 문제 → (a) 해결 방법 → (b) 수학적 근거 → (c) 하이퍼파라미터 해석
```

**(a)** [4점] 정규화(Regularization)가 과적합을 방지하는 직관적 이유
**(b)** [4점] L2 Reg = Gaussian Prior의 MAP임을 1줄로 요약
**(c)** [4점] $\lambda$가 크면 → $\sigma_p^2$이 작으면 → Prior가 강하면 → 어떤 결과?

---

## 문제 7. [12점] 최적화 방법 비교

**(a)** [4점] Gradient Descent $\theta \leftarrow \theta - \alpha\nabla L$의 수학적 근거를 Taylor 1차로 설명하시오.
**(b)** [4점] Newton's Method $\theta \leftarrow \theta - H^{-1}\nabla L$이 GD보다 수렴이 빠른 이유를 2차 정보 활용 관점에서 설명하시오.
**(c)** [4점] 딥러닝에서 Newton을 쓰지 않는 이유(3가지)를 서술하시오.

---

## 문제 8. [13점] PageRank → 고유벡터 → Power Method

**(a)** [5점] 웹을 행렬로 모델링하는 방법과, PageRank가 주요 고유벡터인 이유를 설명하시오.
**(b)** [4점] Power Method의 알고리즘을 쓰고, 수렴 속도가 $O(|\lambda_2/\lambda_1|^t)$인 이유를 설명하시오.
**(c)** [4점] 이 아이디어가 딥러닝의 어떤 부분과 연결되는지 (예: spectral normalization, attention의 low-rank 근사) 서술하시오.

---
---

# 모범답안

## 답 1.
### (a)
- MSE ← Gaussian $\mathcal{N}(h(x),\sigma^2)$: NLL = $(y-h(x))^2/(2\sigma^2)$ + const
- CE ← Categorical $\text{Cat}(h(x))$: NLL = $-\log[h(x)]_y$
- MAE ← Laplace $\text{Lap}(h(x),b)$: NLL = $|y-h(x)|/b$ + const

### (c)
Student-t의 PDF: $(1 + (y-\mu)^2/\nu)^{-(\nu+1)/2}$
NLL $\propto (\nu+1)/2 \cdot \log(1 + (y-h(x))^2/\nu)$
이상치가 있으면 $(y-h(x))^2$이 크지만, log 안에 있으므로 증가율이 완만 (MSE의 제곱 증가보다 느림). Heavy tail = 이상치에 더 높은 확률 부여 → Loss가 덜 폭발.

## 답 3.
### (b)
$\|\frac{\partial f_i}{\partial f_{i-1}}\| < 1$이 L개 층에 걸쳐 곱해지면 $\|grad\| < \epsilon^L \to 0$ → **Gradient Vanishing**. 초기 층의 파라미터가 학습되지 않음.

### (c)
$f_i(x) = x + g_i(x)$의 야코비안: $\frac{\partial f_i}{\partial x} = I + \frac{\partial g_i}{\partial x}$
$g_i$의 야코비안이 작아도 $I$ 항이 그래디언트 1을 보장 → **gradient highway**.
전체: $\prod_i (I + J_{g_i})$에서 $I$를 전개하면 직접 경로(skip path) 존재.

## 답 6.
(a) 정규화는 "가중치 크기를 제한"하여 모델의 자유도를 줄임 → 학습 데이터에 과도하게 맞추는 것을 방지
(b) $\arg\min[\text{NLL} + (1/2\sigma_p^2)\|\theta\|^2] = \text{MAP with Gaussian Prior}$
(c) $\lambda$ 크면 → $\sigma_p^2$ 작음 → Prior가 $\theta=0$ 근처를 강하게 선호 → 가중치가 0에 가깝게 → 모델이 단순해짐 → 과적합 방지 but 과소적합(underfitting) 위험

## 답 7.
(a) $L(\theta+\delta) \approx L(\theta) + \nabla L^\top \delta$. $\delta = -\alpha\nabla L$이면 $\Delta L \approx -\alpha\|\nabla L\|^2 < 0$ → Loss 감소 보장 (학습률 충분히 작을 때).
(b) Newton은 $L(\theta+\delta) \approx L + \nabla L^\top\delta + \frac{1}{2}\delta^\top H\delta$의 2차 근사 최소점을 직접 찾음 → 곡률 고려 → 적절한 step size 자동 결정 → 수렴 빠름 (2차 수렴).
(c) (1) 헤시안 $H$ 계산이 $O(n^2)$ 메모리, $O(n^3)$ 연산 (n=파라미터 수, 수억), (2) 비볼록이므로 saddle point 문제 (Newton은 saddle로 수렴할 수 있음), (3) 미니배치에서 $H$ 추정이 불안정.

---
title: "딥러닝이론 8주차 분석"
slug: week8
order: 7
---

# 딥러닝이론 8주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-8주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### NLL → ERM 형식 정리
- **정의:** $\text{NLL}(h) = -\log P(D|h) \overset{IID}{=} \sum_i -\log P(x_i|h) = \sum_i \ell(x_i, h) = n \cdot \hat{L}_S(h)$.
- **표기:** $\hat{L}_S(h) = E_{x\sim P_S}[\ell(x, h)]$ — empirical distribution에서의 expected loss.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Parameter Space로의 환원
- **정의:** $h_\theta$ 함수 대신 $\theta\in\mathbb{R}^d$를 다룸. $\theta^* = \arg\min_\theta \hat{L}_S(\theta)$.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Newton's Method (재방문)
- **정의:** $f(x) = 0$을 찾는 iterative — $x_{t+1} = x_t - f(x_t)/f'(x_t)$. 1차 근사로 zero 찾기.
- **중요도:** ★★★★★★★ (7/10) [명시적: 2주차 재등장]

### $L$의 미분이 $f$일 때 — Newton's Method의 의미 변환
- **정의:** $f = L'$로 보면, $f$의 zero를 찾는 것 = $L$의 critical point (미분=0)를 찾는 것 = $L$의 minimum 찾기.
- **함의:** Newton's method는 $L$에 대해 **2차 근사**를 한 후 **그 minimum을 찾는** 방법. (선형근사를 미분한 것이기 때문)
- **중요도:** ★★★★★★★★★ (9/10) [명시적: 핵심 통찰]

### Linear Regression의 Closed Form
- **정의:** $\arg\min_w \frac{1}{2}\|Xw - y\|^2$. 미분 = 0: $X^TXw - X^Ty = 0$. $Ax=b$ 형태로 풀림.
- **중요도:** ★★★★★★★★ (8/10)

### Convex
- **정의:** 미분=0 → 최소값 보장. Linear regression의 quadratic loss는 convex이므로 미분=0으로 풀림.
- **중요도:** ★★★★★★ (6/10) [추론 보충: 깊게는 다루지 않음]

### Gradient Descent (GD)
- **정의:** $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$. Gradient 반대 방향으로 이동.
- **First-order method:** Gradient(1차 미분)만 사용.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심]

### Stochastic Gradient Descent (SGD)
- **정의:** Batch $B \subset S$ ($|B|$ = mini-batch size, e.g. 128)에서만 gradient 계산.
- **장점:** 계산 효율 + (이상하게도) **generalization 더 좋음**.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Gradient Flow (GF)
- **정의:** $\eta \to 0$ continuous version. 부드러운 곡선을 따라 내려감.
- **중요도:** ★★★★★★ (6/10) [명시적]

### Learning Rate $\eta$ — 큰가 작은가?
- **정의:** 작게 쓰면 정확한 근사 but 느림. 크게 쓰면 빠르지만 발산 위험.
- **놀라운 사실:** "왜 그런지 모르지만 알려진 바로는 **될 수 있으면 가장 크게 쓰는 게 좋다**."
- **중요도:** ★★★★★★★★ (8/10) [명시적: 직관 거꾸로]

### Momentum (Heavy-ball / NAG)
- **정의:** $m_t = \beta m_{t-1} + g_t$, $\theta_{t+1} = \theta_t - \eta m_t$. 이전 gradient들의 moving average.
- **직관:** 공이 굴러가는 관성.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Second-Order / Preconditioned Methods
- **정의:** $\theta_{t+1} = \theta_t - \eta H^{-1} g_t$ 형태 (Hessian inverse). 곡률 큰 방향은 적게, 작은 방향은 많이.
- **NN에서의 어려움:** Hessian이 $d^2$차원 ($d$=Million 이상). Inverse는 $O(d^3)$ — **불가능**.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### AdaGrad / RMSProp
- **정의:** Hessian을 $g^2$로 근사. $s_t = \sum_t g_t^2$ (AdaGrad) or 가중 합 (RMSProp). $\theta_{t+1} = \theta_t - \eta g_t/\sqrt{s_t + \epsilon}$.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Adam (2014)
- **정의:** Momentum + RMSProp 결합. **현재까지 LLM 학습의 default** (AdamW로 약간 개선).
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: "거의 default"]

### Backpropagation
- **정의:** Computational graph에서 chain rule로 gradient를 효율적으로 계산. Loss로부터 leaf node로 거슬러 올라가며 gradient 누적.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: "학습의 기본"]

### Computational Graph / Leaf Node
- **정의:** 연산을 노드로 표현. Input과 parameter가 leaf node, loss가 root.
- **중요도:** ★★★★★★★ (7/10)

### Chain Rule
- **정의:** $\frac{\partial L}{\partial w} = \prod_{\text{path}} \frac{\partial \text{output}}{\partial \text{input}}$. 모든 edge의 derivative를 곱해서 path를 따라감.
- **중요도:** ★★★★★★★★★★ (10/10)

---

## 🔢 2. 수식 풀이 및 증명

### Newton's Method가 $L$에 대해 2차 근사하는 것임을 보임
**문제/목표:** $f = L'$일 때 Newton step의 의미.
**단계별 풀이:**
1. $L$을 $\theta_t$ 점에서 2차 근사: $\hat{L}(\theta) = L(\theta_t) + L'(\theta_t)(\theta-\theta_t) + \frac{1}{2}L''(\theta_t)(\theta-\theta_t)^2$.
2. 이의 minimum: $\hat{L}'(\theta) = L'(\theta_t) + L''(\theta_t)(\theta-\theta_t) = 0$.
3. $\theta_{t+1} = \theta_t - L'(\theta_t)/L''(\theta_t) = \theta_t - f(\theta_t)/f'(\theta_t)$ (where $f=L'$).

**결론:** $L$의 2차 근사 minimum = $L'=f$의 zero. Newton's method = preconditioned gradient descent with $1/L''$.
**중요도:** ★★★★★★★★★ (9/10)

### Linear Regression Closed Form
**문제/목표:** $L(w) = \frac{1}{2}\|Xw - y\|^2$, $w^*$ 도출.
**단계별 풀이:**
1. 전개: $L = \frac{1}{2}(w^TX^TXw - 2w^TX^Ty + y^Ty)$.
2. $\partial L/\partial w = X^TXw - X^Ty = 0$.
3. $w^* = (X^TX)^{-1} X^T y$.

**결론:** Linear regression은 $Ax=b$ 형태로 closed form.
**중요도:** ★★★★★★★★ (8/10)

### Backprop 예시 — AlexNet-style 작은 그래프
**Setup:** $x \to z = w_1 x \to \tilde{z} = \text{ReLU}(z) \to g = w_2 \tilde{z} \to p = \text{softmax}(g) \to L = -\log p_y$.
**계산해야 하는 미분 (퀴즈로 풀이):**
1. $\partial L / \partial p_y = -1/p_y$.
2. $\partial p_y / \partial g$: Softmax derivative — $p_y(\delta_{yj} - p_j)$.
3. $\partial g / \partial \tilde{z} = w_2$ (또는 $\tilde{z}$인지 확인 필요).

**Chain Rule 적용:**
$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial p}\cdot\frac{\partial p}{\partial g}\cdot\frac{\partial g}{\partial w_2}$.

**중요도:** ★★★★★★★★★ (9/10) [명시적: 퀴즈 + 마지막 시간 핵심]

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| Backpropagation | 10 | 알고리즘 | "학습의 기본" |
| Chain Rule | 10 | 계산 도구 | 모든 미분의 핵심 |
| Adam Optimizer | 10 | 알고리즘 | "현재까지 LLM 학습 default" |
| GD Update Rule | 10 | 알고리즘 | "그래디언트 반대 방향" |
| Newton = 2차 근사 minimum | 9 | 통찰 | 직관 깊이 |
| SGD가 GD보다 좋음 (generalization) | 9 | 사실 | "왜 그런지 모름" |
| Learning Rate 큰 게 좋음 | 8 | 사실 | "직관과 반대" |
| Hessian 계산 불가 (d² 차원) | 8 | 한계 | "불가능 → preconditioning 근사" |
| Computational Graph | 7 | 도구 | Backprop 기반 |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. Empirical Distribution과 Empirical Risk 표현
**문제:**
> "Empirical distribution $P_S$를 어떻게 쓰는지 답해주세요."

**트리거 발언:** "어떻게 했을 수 있었을까요?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $P_S(x) = \frac{1}{n}\sum_{i=1}^n \delta(x - x_i)$.
2. $E_{x\sim P_S}[\ell(x,h)] = \frac{1}{n}\sum_i \ell(x_i, h)$.
3. 그러므로 NLL $\equiv$ $n\cdot$ ERM (상수 차이).

**정답:** Delta 함수의 균등 분포로 표현.

**해설:** Empirical distribution 위에서의 expectation은 그냥 sample mean. 이게 ERM의 정의와 부합.

</details>

### Q2. Newton's Method $L$ 에 대해 무엇을 하는가?
**문제:**
> "$f = L'$일 때 $\theta_t$ 업데이트 식을 $L$에 대한 식으로 바꿔서 써보세요."

**트리거 발언:** "한 6분 정도 드릴테니까 한번 해보세요"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. Newton 식: $\theta_{t+1} = \theta_t - f(\theta_t)/f'(\theta_t)$.
2. $f = L'$, $f' = L''$.
3. $\theta_{t+1} = \theta_t - L'(\theta_t)/L''(\theta_t)$.
4. 의미: $L$을 2차 근사하여 그 minimum으로 점프.

**정답:** $\theta_{t+1} = \theta_t - L'(\theta_t)/L''(\theta_t)$.

**해설:** $L$의 2차 근사는 1차 함수가 아니라 quadratic이므로 그것의 minimum은 미분=0인 점. 이는 곧 $L'$=$f$의 zero 찾기와 동치. Second-order method의 본질.

</details>

### Q3. Backprop의 3가지 미분 계산
**문제:**
> "그래프에서 다음 3개 미분을 직접 계산하세요: $\partial L/\partial p_y$, $\partial p_y/\partial g$, $\partial g/\partial w_2$."

**트리거 발언:** "이 3개만 미분을 계산하라는 겁니다... 직접 좀 해보시고 보시는 걸 아시고요."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. **$\partial L/\partial p_y$:** $L = -\log p_y$, $\partial L/\partial p_y = -1/p_y$.
2. **$\partial p/\partial g$:** Softmax Jacobian $J_{ij} = p_i(\delta_{ij} - p_j)$. $\partial p_y/\partial g_j = p_y(\delta_{yj} - p_j)$.
3. **$\partial g/\partial w_2$:** $g = w_2 \tilde{z}$이므로 $\partial g/\partial w_2 = \tilde{z}$ (vector outer product 형태로 매트릭스).

**정답:** 위 세 식.

**해설:** 이 3개를 chain rule로 곱하면 $\partial L/\partial w_2$. 마지막 한 곱(예: ReLU 미분)을 더해주면 $w_1$까지 도달. Backprop의 본질 = 각 edge 미분을 path 따라 곱하기.

</details>

### Q4. Linear Regression Closed Form
**문제:**
> "$L = \frac{1}{2}\|Xw - y\|^2$일 때 $w^*$를 구해라."

**트리거 발언:** 강의 중 직접 풀이 진행

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $\partial L/\partial w = X^TXw - X^Ty = 0$.
2. $X^TXw = X^Ty$ — $Aw = b$ 형태.
3. $w^* = (X^TX)^{-1}X^Ty$.

**정답:** $w^* = (X^TX)^{-1}X^Ty$.

**해설:** Quadratic이므로 convex, 미분=0이 곧 minimum. Linear regression만의 특혜이고 NN에서는 이렇게 풀 수 없어 GD 사용.

</details>

---

## 📎 5. 기타 참고사항

- 행정: 기말고사 5월 8일, 18:30~22:30, 502호 (수업 장소와 다름). 퀴즈 시작 시간 전까지 LMS 토론판 또는 직접 제출.
- 필기본은 매일 자동 동기화 (시간차 약 하루).
- "Optimization 측면에서 빠르게 하고 싶지만, optimizer 선택에 따라 generalization도 다름. 이건 직관과 다른 부분이 많아 수업에서 깊이는 안 다룸."
- Linear regression은 $Ax=b$로 풀리지만, deep NN은 closed form이 없어 GD가 유일한 방법.
- Bitter lesson과 일관: 스케일링이 옵티마이제이션의 성공을 견인.
- AdamW: Adam의 decoupled weight decay 버전. LLM 학습의 표준.
- Muon (최근 ~2024) — Adam 후의 새로운 옵티마이저, scaling용.
- Loss landscape의 shape (sharp vs flat) 시각화 — Hessian이 큰 곳/작은 곳 직관.
- ReLU 미분: 양수면 1, 음수면 0 (0에서는 정의 안 됨, 보통 0).
- Matrix 미분 — 2주차 softmax 미분 풀이가 여기서 활용됨.
- 다음 시간 예고: Image (CNN). Inductive bias 측면에서 다룸.
- Generalization은 이 수업에서 다루지 않음 — "MAP를 이해하면 generalization으로 연결된다"는 정도.


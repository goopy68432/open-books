---
title: "Level 2: Intermediate - 최적화와 역전파 수식 기초 (Derivations)"
slug: 04-backprop-derivations
order: 4
---

# Level 2: Intermediate - 최적화와 역전파 수식 기초 (Derivations)

> **학습 목표:** 직관적으로 이해한 '경사 하강법'과 '역전파'를 수식과 계산 그래프를 통해 한 줄 한 줄 유도하고, "왜 곱하는지"를 이해합니다.

---

## 1. 경사 하강법 (Gradient Descent) 수식

**공식:** 
$$ \theta_{t+1} = \theta_t - \eta \frac{\partial L}{\partial \theta} $$

### Step-by-Step 수식 읽기
1. **$\theta_t$ (현재 위치):** 현재 시간 $t$에서의 파라미터(가중치) 값입니다.
2. **$L$ (손실 함수):** 우리가 줄이고자 하는 오차(예: NLL, MSE, Cross-Entropy)입니다.
3. **$\frac{\partial L}{\partial \theta}$ (기울기, Gradient):** 파라미터 $\theta$가 아주 찔끔 변할 때, 손실 $L$이 얼마나 변하는지 나타냅니다.
   * 이 기울기 값이 **양수(+)** 라면: $\theta$를 늘리면 오차 $L$이 커집니다. 따라서 $\theta$를 **줄여야** 합니다.
   * 이 기울기 값이 **음수(-)** 라면: $\theta$를 늘리면 오차 $L$이 줄어듭니다. 따라서 $\theta$를 **늘려야** 합니다.
4. **마이너스 기호 ($-$):**
   * 위에서 설명했듯 기울기의 부호와 "정답으로 가는 방향"은 반대입니다. 그래서 항상 기울기를 빼주는($-$) 방향으로 이동합니다. 이를 "가파른 내리막으로 걷기"라고 부릅니다.
5. **$\eta$ (에타, Learning Rate):**
   * 기울기에 곱해주는 학습률(보폭)입니다. 보통 $0.01$이나 $0.001$ 같은 작은 숫자를 씁니다. 기울기 값이 엄청 크더라도 한 번에 훅 날아가지 않게 브레이크를 걸어줍니다.

---

## 2. 역전파의 핵심: 합성함수의 미분 (Chain Rule)

**"딥러닝은 거대한 합성함수다."**

딥러닝 모델은 입력 $X$가 들어가서 수많은 층(가중치 곱하기, 활성화 함수)을 거쳐 최종 출력 $Y$가 나오고, 이것이 정답과 비교되어 손실 $L$이 나옵니다.
수식으로 쓰면: $L = f(g(h(x, w_1), w_2), w_3)$ 형태의 거대한 양파 껍질 구조입니다.

가장 안쪽 껍질에 있는 가중치 $w_1$을 업데이트하려면, 제일 바깥 껍질인 $L$부터 차례대로 뚫고 들어가야 합니다. 이때 쓰는 것이 **연쇄 법칙(Chain Rule)** 입니다.

**공식:**
$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} $$

### Step-by-Step 유도 체인 (계산 그래프)
간단한 함수 $L = (x+y) \cdot z$ 가 있다고 합시다. $q = x+y$ 라고 치환하면, $L = q \cdot z$ 입니다.
만약 $x=-2, y=5, z=-4$ 라면, $q=3$ 이고 $L=-12$ 입니다.

목표: $x$를 조금 바꿨을 때 $L$이 얼마나 변하는지, 즉 $\frac{\partial L}{\partial x}$ 를 찾기!

1. **마지막 껍질 미분 (국소적 미분):**
   * $L$을 $q$에 대해 미분: $\frac{\partial L}{\partial q} = z = -4$
   * (의미: $q$가 1 늘어나면 $L$은 -4만큼 변한다)
2. **안쪽 껍질 미분:**
   * $q$를 $x$에 대해 미분: $q = x+y$ 이므로 $\frac{\partial q}{\partial x} = 1$
3. **Chain Rule (곱하기):**
   * $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial q} \cdot \frac{\partial q}{\partial x} = (-4) \cdot 1 = -4$
   * **결론:** $x$에 대한 기울기는 $-4$ 입니다! 

**딥러닝에서의 적용:**
수백 개의 층이 있어도 원리는 똑같습니다.
$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \text{출력층}} \times \frac{\partial \text{출력층}}{\partial \text{은닉층2}} \times \frac{\partial \text{은닉층2}}{\partial \text{은닉층1}} \times \frac{\partial \text{은닉층1}}{\partial w_1}$

컴퓨터는 앞에서부터 값을 한 번씩 다 계산해서 저장해두고(Forward Pass), 맨 뒤의 $L$부터 시작해서 미분값을 징검다리 건너듯 계속 **곱하면서** 앞으로 돌아옵니다(Backward Pass). 이것이 역전파(Backpropagation)의 전부입니다.

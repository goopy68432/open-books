---
title: "딥러닝이론 9주차 분석"
slug: week9
order: 8
---

# 딥러닝이론 9주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-9주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### 이미지 다루기 = Linear Transformation의 Restriction
- **정의:** 이미지 다룬다는 것은 입력에 매트릭스 곱을 적용하는 것이지만, **모든 매트릭스가 아니라 특정 매트릭스**(sparse + weight sharing)만 사용.
- **함의:** Hypothesis space restriction = inductive bias = MAP의 prior.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심 통합 시각]

### Inductive Bias as Human-Inserted Prior Knowledge
- **정의:** "이미지는 이래야 한다"는 사람의 가설을 architecture에 강제로 넣어줌. 실제 그래야 한다는 것은 아니고, 사람의 가정.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 7주차에서 이어짐]

### Bitter Lesson과의 연결
- **정의:** Sutton의 bitter lesson — 사람이 만든 prior < scaling. 따라서 데이터가 많아지면 inductive bias 줄이는 방향.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Feature Detector
- **정의:** 이미지에서 특정 feature가 있는지 없는지 찾는 것. Low-level (선, 점, 색)부터 high-level (눈, 강아지 얼굴)까지 다양한 레벨.
- **중요도:** ★★★★★★★★★ (9/10)

### Inner Product = Similarity Measure
- **정의:**
  - $\langle x, y\rangle = \|x\|\|y\|\cos\theta$ — 각도 측면에서 similar할수록 큼.
  - $\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2\langle x,y\rangle$ — 내적이 클수록 거리 작음.
- **함의:** Feature detection은 patch와 image의 inner product 계산. 매칭 = 유사도 = 내적.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Convolution Operation (1D, 2D)
- **정의:** Kernel을 input 위에서 sliding하면서 매 위치에서 inner product 계산. **Linear operation.**
- **표기:** Stride $s$, padding $p$, kernel size $k$.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]

### Convolution은 Linear Transformation
- **정의:** Convolution이 (1) additivity와 (2) homogeneity 만족 → linear transformation → 매트릭스로 표현 가능.
- **증명:** 직접 계산으로 확인 — $f(x+y) = f(x)+f(y)$, $f(ax) = af(x)$.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Convolution → Sparse + Weight-Sharing Matrix
- **정의:** Convolution에 대응되는 매트릭스를 표준기저 image로 구성하면:
  - **Sparse** (대부분 0)
  - **Weight sharing** (같은 weight가 여러 위치에 반복)
- **이게 두 가지 inductive bias.**
- **중요도:** ★★★★★★★★★★ (10/10) [명시적: 핵심 결론]

### Locality (인덕티브 바이어스 1)
- **정의:** Feature detection은 local 영역만 보면 됨 — 대각선 검출에 3×3이면 충분.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Translation Invariance (인덕티브 바이어스 2)
- **정의:** 같은 feature(눈)는 위치에 관계없이 같은 detector로 검출. → 같은 weight를 위치마다 공유.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]

### Stride / Padding / Same / Valid Convolution
- **정의:**
  - Stride $s$: kernel이 한 번에 몇 칸 이동.
  - Padding $p$: 입력 가장자리에 0 추가.
  - Valid convolution: padding 없음.
  - Same convolution: input과 output 차원 같도록 padding.
- **Same convolution 조건:** $2p = k - s$.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Output Size Formula
- **수식:** $H_{out} = \frac{H_{in} + 2p - k}{s} + 1$.
- **유도 직관:** 박스가 sweep하면서 몇 번 위치할 수 있는지.
- **중요도:** ★★★★★★★★ (8/10) [명시적: "외우라는 게 아니고 증명할 수 있어야"]

### Pooling (Max / Average)
- **정의:**
  - Max Pooling: 영역에서 max값 선택. **Non-linear.**
  - Average Pooling: 영역 평균. **Linear**, 따라서 매트릭스로 표현 가능.
- **중요도:** ★★★★★★★★ (8/10) [명시적]

### Channel
- **정의:** RGB 이미지는 3 channel. Conv layer는 $c_{in}$개 채널을 $c_{out}$개로 매핑 — 각 (in, out) 쌍마다 kernel 하나.
- **중요도:** ★★★★★★★ (7/10)

### AlexNet Architecture (2012)
- **정의:** 8 layers — Conv(11, s=4) → ReLU → MaxPool(3, s=2) → Conv → ... → Flatten → 3 FC → Softmax → 1000-class.
- **중요도:** ★★★★★★★ (7/10) [명시적: 직접 분석]

### Flatten
- **정의:** 마지막 conv 결과 ($6\times 6\times 256$) → 9216차원 vector로 펼침.
- **중요도:** ★★★★★★ (6/10)

### ZFNet의 Feature Visualization (2013)
- **정의:** AlexNet의 각 layer가 무엇을 detect하는지 시각화. Layer 1: 색/대각선/세로/가로. Layer 2: 동그라미, 노란색. Layer 3: 알파벳, 아쿠아. Layer 4: 강아지 얼굴, 다리. → 점점 high-level.
- **중요도:** ★★★★★★★★ (8/10) [명시적: 직관 핵심]

### Grandmother Cell / Hubel-Wiesel 실험
- **정의:** 고양이 visual cortex에서 가로/세로 feature를 detect하는 cell들이 있음을 발견 (Nobel 수상). NN의 feature detector가 이와 유사.
- **중요도:** ★★★★★★ (6/10) [명시적]

---

## 🔢 2. 수식 풀이 및 증명

### Convolution이 Linear Transformation임을 증명
**문제/목표:** 4가지 예시($x_1, x_2, x_1+x_2, 2x_1$ 등)에 대해 conv 수행 후 linearity 확인.
**단계별 풀이:**
1. $f(x_1)$, $f(x_2)$, $f(x_1+x_2)$ 각각 계산.
2. $f(x_1)+f(x_2) = f(x_1+x_2)$ ✓ (additivity).
3. $f(2x_1) = 2f(x_1)$ ✓ (homogeneity).

**결론:** Convolution은 linear transformation. 따라서 매트릭스 표현 가능.
**중요도:** ★★★★★★★★ (8/10) [명시적: 퀴즈]

### 1D Conv → Matrix Construction (7→6, kernel = (1,2))
**문제/목표:** Input 7차원, kernel size 2, stride 1, no padding. 대응 매트릭스 ($6\times 7$) 찾기.
**단계별 풀이:**
1. Domain의 표준기저 7개 ($e_1,\dots,e_7$)를 모두 conv 통과시킴.
2. 각 $f(e_i)$를 column으로 stack.
3. 결과: 매트릭스가 $\begin{pmatrix}1&2&0&0&0&0&0\\0&1&2&0&0&0&0\\0&0&1&2&0&0&0\\\vdots\end{pmatrix}$ — 각 행에 (1,2)가 한 칸씩 shift되며 반복.

**관찰:**
- **Sparse:** 대부분이 0.
- **Weight sharing:** 1과 2가 모든 행에 반복.

**중요도:** ★★★★★★★★★★ (10/10) [명시적]

### 2D Conv → Matrix (9→4, +-shape kernel)
**문제/목표:** $3\times 3$ input(9차원), $3\times 3$ kernel, valid → $1\times 1$ 출력? 강의에서는 specific 4-output 예시.
**단계별 풀이:**
1. 9개 표준기저를 conv 통과.
2. $4\times 9$ 매트릭스 구성.
3. Sparse + weight sharing 패턴 확인.

**중요도:** ★★★★★★★★ (8/10) [명시적: 퀴즈]

### Output Size Formula 증명
**문제/목표:** $H_{out} = \lfloor (H_{in} + 2p - k)/s\rfloor + 1$.
**단계별 풀이:**
1. Input width with padding: $H_{in} + 2p$.
2. Kernel이 마지막 위치까지 갈 수 있는 거리 = $H_{in} + 2p - k$ (마지막 kernel 길이 빼야 함).
3. Stride $s$로 나누면 이동 횟수 = $(H_{in} + 2p - k)/s$.
4. 첫 위치 포함하므로 +1.

**결론:** $H_{out} = (H_{in} + 2p - k)/s + 1$.
**중요도:** ★★★★★★★★★ (9/10) [명시적: "외우지 말고 유도하라"]

### Same Convolution Padding 조건
**문제/목표:** $H_{in} = H_{out}$이 되도록 $p$ 결정.
**단계별 풀이:**
1. $H_{in} = (H_{in} + 2p - k)/s + 1$.
2. $s = 1$ 가정: $0 = 2p - k + 1 \cdot 0$? 다시 풀면 $2p = k - 1$ if $s=1$.
3. 일반: $2p = k - s$ (강의에서 명시).

**중요도:** ★★★★★★★★ (8/10)

### Max Pooling은 Linear인가?
**문제/목표:** Counterexample로 non-linearity 확인.
**단계별 풀이:**
1. $f(x_1+x_2) \ne f(x_1)+f(x_2)$인 반례 찾기.
2. e.g. $x_1 = (1, 0)$, $x_2 = (0, 1)$. $\max(x_1) = 1$, $\max(x_2) = 1$, 합 = 2. 그러나 $\max(x_1+x_2) = \max(1,1) = 1 \ne 2$.

**결론:** Max pooling은 non-linear.
**중요도:** ★★★★★★★ (7/10) [명시적]

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| Conv = Sparse + Weight-Sharing 매트릭스 | 10 | 핵심 정리 | "두 개의 inductive bias 넣어준 것" |
| Locality + Translation Invariance | 10 | 핵심 IB | "두 가지 특징" 강조 |
| 이미지 처리 = Linear Transformation Restriction | 10 | 통합 시각 | 첫 시간부터 누적된 framing |
| Conv → Matrix 변환 (표준기저) | 9 | 방법 | 직접 풀이 |
| Output Size Formula | 9 | 정리 | "유도할 수 있어야" |
| Inner Product = Similarity | 9 | 개념 | 1주차에서 이어짐 |
| Feature Detector 위계 (low→high) | 8 | 개념 | ZFNet 시각화 |
| AlexNet 구조 분석 | 7 | 사례 | 직접 차원 계산 |
| Max Pooling Non-linearity | 7 | 정리 | 반례 |
| Bitter Lesson 재조명 | 8 | 철학 | "scaling 대 prior" |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. 1D Convolution 직접 계산 (4 cases)
**문제:**
> "$x = (0,1,2,3,4,5,6)$, $w=(1,2)$일 때 $w * x$를 계산. 그리고 다른 입력에 대해 3개 더 계산해서 무엇을 보여주는지 생각해보세요."

**트리거 발언:** "발음 3개에 대해서도 부러워 부터 죽여주길 바라겠습니다... 5분 정도?"

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $x_1 = (0,1,2,3,4,5,6)$, $w * x_1 = (2, 5, 8, 11, 14, 17)$ (각 위치에서 내적).
2. 다른 입력 + 그 입력 → 결과 더하면 $w * (x_1+x_2)$와 일치 (additivity).
3. $w * (2x_1) = 2(w*x_1)$ (homogeneity).

**정답:** Linearity 확인 — $f(x+y) = f(x)+f(y)$, $f(ax) = af(x)$.

**해설:** 4개 example을 고른 이유 = linearity 정의 두 조건을 모두 검증하기 위함. 따라서 conv는 linear transformation = matrix.

</details>

### Q2. Conv → Matrix (1D, 7→6)
**문제:**
> "위 conv에 대응되는 $6\times 7$ 매트릭스를 표준기저로 찾으세요."

**트리거 발언:** "Linear transformation이 주어졌고 매트릭스를 찾고 싶은 겁니다... 5분 드릴게요."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $e_1 = (1,0,0,0,0,0,0)^T$ → $f(e_1) = (1,0,0,0,0,0)^T$ (첫 칸만 1, 나머지 0과 inner product).
2. $e_2 = (0,1,0,...)^T$ → $f(e_2) = (2,1,0,0,0,0)^T$.
3. ... 9까지 ...
4. 7개의 column을 stack:
$\begin{pmatrix}1&2&0&0&0&0&0\\0&1&2&0&0&0&0\\0&0&1&2&0&0&0\\0&0&0&1&2&0&0\\0&0&0&0&1&2&0\\0&0&0&0&0&1&2\end{pmatrix}$.

**정답:** 위 매트릭스.

**해설:** 매트릭스가 (1) sparse, (2) weight-sharing임을 확인. 이게 conv의 두 inductive bias의 매트릭스 표현.

</details>

### Q3. 2D Conv → Matrix (9→4)
**문제:**
> "$3\times 3$ input, +자 모양 $3\times 3$ kernel일 때 대응 $4\times 9$ 매트릭스를 표준기저로 찾으세요."

**트리거 발언:** "이번에도 마찬가지로 어떤 weight가 share되었는지, locality가 어떤지 두 가지를 보세요."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. 9개 표준기저 ($e_1,\dots,e_9$ — 각각 한 위치만 1) 모두 conv.
2. Output 4차원 vector를 column으로 stack.
3. 패턴: kernel의 +자 모양이 각 column에 sparse하게 나타남, 행 간 weight sharing.

**정답:** $4\times 9$ sparse + weight-sharing matrix (정확한 형태는 직접 계산).

**해설:** 2D에서도 same principle. Convolution operation이 곧 특정 형태의 linear transformation = 특정 매트릭스 곱.

</details>

### Q4. Max Pooling은 Linear 인가?
**문제:**
> "$2\times 2$ Max Pooling이 linear transformation인지 보이고, 만약 그렇다면 매트릭스를 구하라."

**트리거 발언:** "리니어 트랜스포미션이다 아니다 얘기하긴 했지만 발레를 찾는 겁니다."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $x_1 = \begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}$, $x_2 = \begin{pmatrix}0 & 1 \\ 0 & 0\end{pmatrix}$.
2. $\max(x_1) = 1$, $\max(x_2) = 1$, 합 = 2.
3. $x_1 + x_2 = \begin{pmatrix}1 & 1 \\ 0 & 0\end{pmatrix}$, $\max(x_1+x_2) = 1$.
4. $1 \ne 2$ — additivity 위반.

**정답:** Max Pooling은 linear가 아니다.

**해설:** Counterexample로 non-linearity 보임. 반면 Average Pooling은 linear이므로 매트릭스 표현 가능. NN에서 non-linearity를 활성화 함수 외에 max pool로도 추가하는 방법.

</details>

### Q5. AlexNet Layer Shape 변환
**문제:**
> "AlexNet의 각 layer에서 입력→출력 차원이 어떻게 변하는지 직접 계산해보세요. (227×227×3 → 55×55×96 → ...)"

**트리거 발언:** "이 네 가지에 대해서 진짜 이렇게 되는지를 각자 시간 났을 때 해보시길 바랍니다."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. **Conv1:** 227×227×3, $k=11, s=4, p=0$. $H_{out} = (227 - 11)/4 + 1 = 54 + 1 = 55$. → 55×55×96.
2. **MaxPool1:** $k=3, s=2, p=0$. $H_{out} = (55-3)/2 + 1 = 27$. → 27×27×96.
3. **Conv2:** $k=5, s=1, p=2$. $H_{out} = (27 + 4 - 5)/1 + 1 = 27$. → 27×27×256.
4. **MaxPool2:** 13×13×256.
5. ... 결국 6×6×256 → flatten 9216 → 4096 → 4096 → 1000.

**정답:** Output size formula로 직접 계산.

**해설:** Output size가 점점 줄어들면서 (resolution↓) channel 수는 늘어남 (representation depth↑). 마지막에는 fully-connected layer로 classification.

</details>

---

## 📎 5. 기타 참고사항

- 행정: 기말고사 5월 8일, 18:30~22:30, 502호. 퀴즈는 시험 시작 전까지 제출. 두 가지 방법 (대면 종이 또는 LMS 토론판 PDF).
- 시험 범위: 전범위. 중간고사 이전 부분은 많이 안 나오지만, **저번 시험에 못 푼 것들에서 조금만 바꿔 출제 가능성** 강조 — 약 1/3.
- 시험 형식: 비슷, 문제 수는 약간 적을 수 있음.
- 마지막 소통: 컨볼루션 연산이 매트릭스 곱의 일부이고, 학습되는 값은 kernel의 weight (예: $a, b$). 원래 42개 parameter가 필요했던 자리에 2개만 학습하는 격.
- ZFNet 시각화 — AlexNet의 각 layer가 무엇을 보는지 보여주고, 그것에 기반해 다음 해 (2013) 우승.
- ImageNet 대회 history: AlexNet (2012) → ZFNet (2013) → GoogLeNet (2014) → ResNet (2015, 152 layer, human level 돌파).
- 강의 마지막: Linear algebra의 가장 중요한 정리 (linear transformation = matrix)를 그대로 conv에 적용한 것.
- Hubel-Wiesel 고양이 실험 일화 — 가로 막대를 보여주다 빼다 하며 시각피질의 발화를 발견.
- "패딩이 1이고 K가 1이어도 커지는 경우가 있다" — output이 input보다 큰 경우의 응용 (resolution upscaling, e.g. segmentation).
- "사실 잘 안 쓰지만 그런 경우도 있다."
- 컨볼루션이 일반 매트릭스 곱의 부분집합이라는 점, 그래서 hypothesis space가 작아진다는 결론을 다시 강조.
- 학생 질문 처리: "$H_{out}$이 input보다 커질 수 있나?" → 패딩이 매우 크면 가능, 보통은 안 씀. Resolution을 키우는 응용 (e.g., decoder)에서 사용.


---
title: "딥러닝이론 1주차 분석"
slug: week1
order: 1
---

# 딥러닝이론 1주차 분석

> 원본: docs/수업_스크립트/딥러닝이론-1주차.md
> 처리 원칙: 원문 누락 없음. 모든 내용을 5개 섹션 중 하나에 배치.

---

## 📘 1. 개념 및 정의

### Deduction (연역)
- **정의:** 참인 명제(theory)들로부터 논리적으로 새로운 hypothesis를 얻어내는 추론. "참 → 무조건 참"이 보장됨.
- **맥락:** 첫 시간에 인덕션과 짝을 이루어 도입. 디덕션은 "rule, logic, theorem, knowledge"와 연결되며 후에 symbolic AI로 발전.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]
  - 명시적 근거 인용: "두 개의 차이를 이해하는 게 놀랍게도 중간고사 전까지의 목표에 굉장히 중요합니다."

### Induction (귀납)
- **정의:** Observation을 먼저 하고 패턴을 보아 hypothesis를 세움. "참 → 참" 보장 X (틀릴 수 있음).
- **맥락:** "수학적 귀납법"은 사실 디덕션이며, 진짜 귀납은 $P(n) = n^2 - n + 41$ 소수 예시처럼 몇 개만 보고 일반화하는 사고. AI 역사에서 인덕션이 디덕션을 이김.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]
  - 명시적 근거: "AI 역사에서는 인덕션이 이겼습니다." 시험 문제는 인덕션으로 풀면 안 됨.

### Hypothesis / Data / Model
- **정의:** 데이터(=Observation/Experience), Hypothesis(=Model/함수). Induction은 데이터 → 모델 학습으로 대응.
- **맥락:** "데이터로부터 패턴을 보고 hypothesis를 세움"이 induction이고 그게 머신러닝.
- **중요도:** ★★★★★★★★ (8/10) [추론 보충: 전체 수업의 핵심 프레임. 2~9주차에서 계속 등장]

### Maximum Likelihood vs Prior (수업의 메인 두 축)
- **정의:** "P-text Tango" 그림에서 Learning은 ML과 Prior 두 가지로 이해됨.
- **맥락:** 이 두 개를 이해하는 것이 수업 전체의 목표라고 명시.
- **중요도:** ★★★★★★★★★★ (10/10) [명시적]
  - "MM이라고 써진 Maximum Likelihood랑 Prior라고 써진 이 부분을 이해하는게 이 수업의 목표입니다. 그 둘을 이해하는게 저의 목표이고 이 둘을 이해하는게 끝입니다."

### 이론적 이해와 공학적 산출물 (Steam Engine 비유)
- **정의:** 산업혁명 증기기관처럼 공학적 산출물이 이론보다 먼저. 이후 열역학 등 이론이 따라잡으면 내연기관·상대성이론 등 더 강력한 산출물이 나옴. 딥러닝 = 증기기관 단계.
- **맥락:** Ali Rahimi("Machine learning is alchemy") vs Yann LeCun 논쟁 소개.
- **중요도:** ★★★★★★★ (7/10) [추론 보충: 수업 동기/철학을 잡는 framing]

### Set / Cartesian Product / Function / 1-to-1 Function
- **정의:** Set은 unordered, distinct collection. $A \times B$는 ordered pair. Function은 입력 하나에 출력 하나. 1-to-1 (one-to-one, 단사)은 "two-to-two": 다른 입력은 다른 출력.
- **맥락:** Thing(대상)으로 Vector, Algorithm(다루기)으로 Function을 도입.
- **중요도:** ★★★★★★ (6/10) [추론 보충: 기초 정의]

### Vector / Inner Product
- **정의:** $n$-tuple, $\mathbb{R}^n$의 원소. 트랜스포즈 $T$로 가로/세로 표기. 내적 $\langle v, w\rangle = \sum v_i w_i = \|v\|\|w\|\cos\theta$. cos 값으로 유사도 측정.
- **맥락:** "대학원 면접 가면 항상 물어보는 것. 모르면 탈락."
- **중요도:** ★★★★★★★★ (8/10) [명시적]
  - "이너프로덕트가 뭔지 물어봅니다. 모르면 탈락"

### Linear Transformation (Matrix와의 동치)
- **정의:** 함수 $L$이 $L(v_1+v_2)=L(v_1)+L(v_2)$, $L(av)=aL(v)$ 두 조건을 만족하면 linear. (정확히는 affine 제외)
- **핵심 정리:** Linear Transformation $\Leftrightarrow$ Matrix. 매트릭스 곱은 linear transformation이고, 모든 linear transformation은 매트릭스로 표현됨.
- **맥락:** "매트릭스가 미분이니까" — 이후 모든 수업의 토대.
- **중요도:** ★★★★★★★★★ (9/10) [명시적]
  - "행렬은 마음의 고향이라는 뜻... 우리가 아는 것은 행렬 뿐"이라는 이인석 교수 인용.

### Probability — Frequentist vs Bayesian
- **정의:** Frequentist: 상대적 빈도(시행 → ∞일 때). Bayesian: hypothesis에 대한 믿음의 정도.
- **맥락:** 수업은 베이지안 관점이 핵심. 배운 확률은 frequentist이지만 처음에는 이질감 있음.
- **중요도:** ★★★★★★★★ (8/10) [추론 보충: 3주차에서 다시 깊이 다룸]

### AI / Machine Learning 역사적 키워드
- **정의:** AI = 1956 Dartmouth (John McCarthy). McCarthy/Minsky/Rochester/Shannon 4명이 organize. ML = 1959 Arthur Samuel. Perceptron, AlexNet(2012, 딥러닝 혁명 시작), ChatGPT(2022.11). NeurIPS, ICML, ICLR가 top 3 학회.
- **중요도:** ★★★★ (4/10) [추론 보충: 배경 지식, 시험 가능성 낮음]

### Stigler's Law of Eponymy
- **정의:** "어떤 과학적 발견도 원래 발견자의 이름을 따서 명명되지 않는다"는 법칙. Gaussian distribution도 De Moivre가 먼저 발견.
- **중요도:** ★★ (2/10) [추론 보충: 잡학]

---

## 🔢 2. 수식 풀이 및 증명

### 매트릭스가 Linear Transformation인 이유 (예시 풀이)
**문제/목표:** $A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$일 때, $A\cdot e_1$, $A\cdot e_2$ 등을 계산하여 linearity 확인 + Linear transformation에 대응되는 matrix 찾기.
**단계별 풀이:**
1. 표준기저 $e_1=(1,0,0)^T$, $e_2=(0,1,0)^T$, $e_3=(0,0,1)^T$를 모두 $A$에 곱한다.
2. $A e_1, A e_2, A e_3$를 column으로 차례로 박아넣으면 그것이 linear transformation에 대응되는 matrix.
3. 더해서 보내는 것 = 보내서 더하는 것, 스칼라배해서 보내는 것 = 보내서 스칼라배 하는 것이 일치함을 확인.

**결론:** Matrix와 Linear Transformation은 동치이며, 어떤 linear $T:\mathbb{R}^n\to\mathbb{R}^m$이 주어져도 $[T] = [Te_1 \mid Te_2 \mid \cdots \mid Te_n]$로 행렬 복원 가능.
**중요도:** ★★★★★★★★ (8/10) [명시적: 직접 칠판 풀이 + 퀴즈]

### Affine vs Linear
**문제/목표:** $f(x)=Ax+b$가 linear인지 판별.
**단계별 풀이:**
1. $L(0)=0$이어야 하지만 $A\cdot 0 + b = b \ne 0$ (일반적으로).
2. 따라서 linear가 아님. 정확한 명칭은 affine.

**결론:** $Ax+b$는 affine이며 흔히 linear로 부르지만 엄밀히는 다름.
**중요도:** ★★★★★★ (6/10) [명시적]

---

## ⚠️ 3. 중요도 강조 항목

| 항목 | ★ | 유형 | 교수님 발언 근거 |
|---|---|---|---|
| Deduction vs Induction 구분 | 9 | 개념 | "둘의 차이를 이해하는 게 중간고사 전까지 목표에 굉장히 중요" |
| Maximum Likelihood + Prior 두 축 | 10 | 개념 | "이 둘을 이해하는게 끝입니다" |
| Linear Transformation = Matrix | 9 | 정리 | "매트릭스 = 리니어 트랜스폼", "매트릭스가 미분" |
| Inner Product 정의 | 8 | 정의 | "대학원 면접가면 항상 물어보는데 모르면 탈락" |
| 함수의 정의 (입력 1 → 출력 1) | 7 | 개념 | 반복 강조 [추론 보충: 다음 주들에서 핵심] |
| Bayesian 관점의 probability | 8 | 개념 | [추론 보충: 3주차에서 핵심으로 재등장] |

---

## 📝 4. QUIZ (문제 + 모범 답안)

### Q1. Linearity 확인 — Matrix-Vector 곱이 linear인지
**문제:**
> "$A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$에 $(0,1,0)^T$을 보내고, $(1,0,0)^T$을 보내고, 두 개를 더한 $(1,1,0)^T$을 보내고, $(1,2,0)^T$을 보내는 게 어떻게 되는지 봐주실 수 있을까요? 퀴즈인데요."

**트리거 발언:** "퀴즈인데요. 태블릿이면 태블릿으로 푸시면 되고..."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $A(1,0,0)^T = (1,4)^T$
2. $A(0,1,0)^T = (2,5)^T$
3. $A(1,1,0)^T = (3,9)^T = (1,4)^T + (2,5)^T$ ✓ (additivity)
4. $A(1,2,0)^T = (5,14)^T = (1,4)^T + 2\cdot(2,5)^T$ ✓ (scalar 배)

**정답:** Additivity와 homogeneity가 성립하므로 $L_A(x)=Ax$는 linear transformation.

**해설:** Linearity의 두 조건을 직접 검증하는 절차를 보여주는 핵심 예시. 이 절차가 일반화되어 "matrix ↔ linear transformation" 동치를 얻음.

</details>

### Q2. Linear Transformation에 대응되는 Matrix 찾기
**문제:**
> "Linear transformation이 있는데 3차원에서 2차원으로 보내는데 $e_1, e_2, e_3$을 어떤 곳으로 보낸다. 이 linear transformation에 대응되는 matrix $A$를 찾아라."

**트리거 발언:** "두번째 문제를 보면..."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. 표준기저의 image $T(e_1), T(e_2), T(e_3)$를 column으로 stack.
2. $A = [T(e_1)\mid T(e_2)\mid T(e_3)]$.
3. 그러면 임의의 $x=(x_1,x_2,x_3)^T$에 대해 $Ax = x_1 T(e_1)+x_2 T(e_2)+x_3 T(e_3) = T(x)$.

**정답:** 첫 column에 $T(e_1)$, 둘째 column에 $T(e_2)$, 셋째에 $T(e_3)$를 박아넣은 $2\times 3$ matrix.

**해설:** "주어진 linear transformation → matrix" 구성을 표준기저 image로 환원하는 표준 기법. 이 풀이 자체가 곧 두 개념의 동치성 증명.

</details>

### Q3. 인덕션 vs 디덕션 직관 ($n^2-n+41$)
**문제:**
> "$P(n) = n^2 - n + 41$이 모든 자연수 $n$에 대해 소수인지 생각해보세요."

**트리거 발언:** "이거를 몇 개 더 해볼 수 있겠죠?... 잠깐 시간을 더 드리고 싶지만..."

<details>
<summary>📖 모범 답안 보기</summary>

**풀이:**
1. $n=0,1,2,\dots,39$까지 넣으면 모두 소수.
2. $n=40$ 또는 더 명확히 $n=41$을 넣으면 $41^2-41+41 = 41^2 = 1681 = 41\times 41$로 소수가 아님.

**정답:** 모든 자연수에 대해 소수가 아니다.

**해설:** 인덕션의 위험 — 39개를 검증해도 40번째에서 무너질 수 있음. "성급한 일반화"의 경계. 이 예시가 인덕션의 본질적 약점을 보여주는 핵심 예시.

</details>

---

## 📎 5. 기타 참고사항

- 강의 운영: 녹화본 + 실시간(7시 40분 줌 오픈). 출석 20%, 중간 30%, 퀴즈는 매시간 진행. 답을 얻는 것보다 시도/과정이 중요. 답만 적으면 0점.
- 기말고사 일정: 5월 5일이 휴일이라 5월 12일 또는 5월 8일 후보. 결국 5월 8일 오프라인 502호로 결정 (8주차 공지).
- 교수 소개: 한양대 인공지능대학원, 이전에는 KIAS(고등과학원, 아인슈타인 같은 IAS를 본떠 만든 곳)에서 AI for Natural Science 연구.
- 펜로즈/펜하이머/필즈 메달 농담, "테렌스 타오가 AI 수학 증명에 큰 관심" 등 도입부 잡담.
- Bertrand Russell 이발사 역설 언급 (집합 정의의 한계).
- Dijkstra: "긴 글보다 그림으로 설명하는 게 낫다" / Linus Pauling: 노벨 화학상+평화상 2회 수상자, Linus Torvalds 이름이 그에서 따옴, 비타민C 메가도즈 일화.
- 책: PML(Probabilistic Machine Learning, Murphy). 챕터 14 이후(이미지 등)는 다루지 않음. Karpathy 2016 강의 추천.
- AI 역사: Herbert Simon "20년 내 사람의 모든 일을 기계가 한다"(1965) → 실제 체스는 40년 걸림(Deep Blue 1997). Sebastian Bubeck "Nobody understands what's going on" (GPT 초기). Ali Rahimi NeurIPS 2017 test-of-time talk: "Machine learning is alchemy".
- Frequentist vs Bayesian 통계학과의 라이벌리: 박사 전 들었던 통계학 수업에서 "다른 진영 내용 나오면 필요없다 쓸모없다 하고 넘어감" 일화.
- John Backus, Marvin Minsky, Claude Shannon (정보과학의 아버지, 가장 임팩트 있는 석사논문 작성자) 언급.
- 강의 자료 위치 안내, 책 구매 불필요(드래프트 PDF 홈페이지에 있음).
- 데이터·논리지(knowledge) 두 진영 비교 표 prep.


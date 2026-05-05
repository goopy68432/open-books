---
title: 'Quiz 4 — Bayes Theorem 증명'
description: '조건부확률 정의로부터 4단계 증명'
draft: false
---

## 0. 한 줄 요약

조건부확률 정의 $P(A\mid B)=P(A\cap B)/P(B)$ 를 두 방향으로 적용하고, 교집합 확률 $P(H\cap E)$ 를 *공통 매개* 로 두 식을 연결하면
$$
P(H\mid E) = \frac{P(E\mid H)\,P(H)}{P(E)}.
$$
가정은 단 두 줄: $P(E)>0,\;P(H)>0$. 이 한 줄 정리는 MLE → MAP → Bayesian 추론 전체의 뿌리이며, MAP 에서 $P(E)$ 를 무시할 수 있는 정당화이기도 하다.

---

## 1. 문제 (출제 형태 그대로)

조건부확률의 정의만 사용하여 다음을 증명하라.
$$
P(H\mid E)=\frac{P(E\mid H)\,P(H)}{P(E)}.
$$
가정: $P(E)>0$ 그리고 $P(H)>0$.

---

## 2. 출제 의도와 시험 가치

베이즈 정리는 *공식 자체* 보다 *그 유도 과정*을 묻는다. 이유:

1. 학생이 조건부확률 정의를 *암기* 가 아니라 *조작 가능한 도구* 로 다루는지 확인.
2. "교집합은 대칭" 같은 *집합론적 자명함* 을 등호 한 줄로 명시할 수 있는지 확인.
3. 정리 자체는 매우 짧지만 *모든 가정* (정의 가능성, 분모가 0 이 아님) 을 명시하는 *수학적 엄밀성* 의 훈련.
4. MAP 추정 ($\arg\max_\theta P(D\mid \theta)P(\theta)$) 가 왜 $P(D)$ 를 무시해도 되는지의 정당화 (Quiz 6, 7, 9 의 토대).
5. 신경망 분류기, 진단 검사, 스팸 필터, 의료 통계 — 모든 *후방 추론* 의 형식적 출발점.

---

## 3. 사전 개념

### 3.1 기호 풀이

| 기호 | 의미 |
|---|---|
| $H$ | Hypothesis (가설). 예: "환자가 병에 걸렸다", "메일이 스팸이다", "모델 모수가 $\theta$ 다". |
| $E$ | Evidence (증거, 관측). 예: "검사 양성", "메일 본문에 'lottery' 가 있다", "데이터 $D$ 가 관측됨". |
| $P(A)$ | 사건 $A$ 의 확률, $0\le P(A)\le 1$. |
| $A\cap B$ | $A$ 와 $B$ 가 *동시에* 일어나는 사건 (교집합). |
| $P(A\cap B)$ | 두 사건의 결합확률. 흔히 $P(A,B)$ 로도 표기. |
| $P(A\mid B)$ | $B$ 가 일어났다는 조건 하의 $A$ 의 확률 (조건부확률). |
| $P(H)$ | *사전확률* (prior) — 증거를 보기 전의 믿음. |
| $P(E\mid H)$ | *가능도* (likelihood) — 가설이 사실일 때 증거가 나올 확률. |
| $P(E)$ | *증거* (evidence, marginal likelihood) — 증거 자체의 무조건 확률. |
| $P(H\mid E)$ | *사후확률* (posterior) — 증거를 본 후 갱신된 믿음. |

### 3.2 정의 정리

**Def (조건부확률).** $P(B)>0$ 일 때
$$
P(A\mid B)\;\equiv\;\frac{P(A\cap B)}{P(B)}.
$$
이는 *정의* 이지 정리가 아니다. 직관: "$B$ 의 세계로 좁혀 놓고, 그 안에서 $A$ 가 차지하는 비중".

**Def (집합 교집합의 대칭).** $A\cap B=B\cap A$ — 두 집합 모두 "양쪽 모두에 속하는 원소" 라는 동일한 의미. 따라서 $P(A\cap B)=P(B\cap A)$.

**파생 (곱셈규칙, product rule).** 정의를 $P(A\cap B)$ 에 대해 풀면
$$
P(A\cap B)=P(A\mid B)\,P(B).
$$
양 방향:
$$
P(H\cap E)=P(H\mid E)\,P(E)=P(E\mid H)\,P(H).
$$
이 두 표현이 같다는 사실 자체가 베이즈 정리의 *씨앗*.

### 3.3 사용할 정리/공식

**Kolmogorov 공리** (배경; 명시적으로 사용하지는 않음):
- $P(A)\ge 0$.
- $P(\Omega)=1$.
- 가산 가법성: 서로소이면 $P(\bigsqcup A_i)=\sum P(A_i)$.

**전확률 정리** (Bayes 정리의 분모 $P(E)$ 를 어떻게 계산하는가):
$$
P(E)=\sum_i P(E\mid H_i)P(H_i),\qquad \{H_i\}\text{ 가 표본공간의 분할일 때.}
$$
본 퀴즈의 증명에는 직접 쓰이지 않으나, MAP 에서 $P(E)$ 를 무시 가능한 이유를 설명할 때 등장.

---

## 4. 풀이 (모든 단계, 모든 등호 근거 명시)

### Step 1. 조건부확률의 정의를 양 방향으로 두 번 적는다

$P(E)>0$ 이라는 가정 하에 정의에 의해
$$
P(H\mid E)\;=\;\frac{P(H\cap E)}{P(E)}.\qquad\qquad (\star)
$$
*등호 근거*: 조건부확률 정의 (3.2), $A=H, B=E$.

$P(H)>0$ 이라는 가정 하에 정의에 의해
$$
P(E\mid H)\;=\;\frac{P(E\cap H)}{P(H)}.\qquad\qquad (\star\star)
$$
*등호 근거*: 조건부확률 정의, $A=E, B=H$.

### Step 2. 교집합의 대칭

$$
H\cap E = E\cap H \;\Longrightarrow\; P(H\cap E)=P(E\cap H).
$$
*등호 근거*: 집합론 (교집합의 가환성). 확률 $P$ 는 사건의 *함수* 이므로 사건이 같으면 확률값도 같다.

### Step 3. $(\star\star)$ 를 곱셈규칙 형태로 변형

$(\star\star)$ 의 양변에 $P(H)$ 곱:
$$
P(E\cap H)\;=\;P(E\mid H)\,P(H).\qquad\qquad (\heartsuit)
$$
*등호 근거*: 분수 식의 양변에 분모를 곱한 것. (이를 *product rule* 또는 *chain rule of probability* 라 부름.)

### Step 4. $(\heartsuit)$ 를 $(\star)$ 의 분자에 대입

$(\star)$ 의 분자 $P(H\cap E)=P(E\cap H)$ (Step 2) 이고, $(\heartsuit)$ 로 다시 $=P(E\mid H)P(H)$. 따라서
$$
P(H\mid E)\;=\;\frac{P(H\cap E)}{P(E)}\;\stackrel{\text{Step 2}}{=}\;\frac{P(E\cap H)}{P(E)}\;\stackrel{(\heartsuit)}{=}\;\boxed{\,\frac{P(E\mid H)\,P(H)}{P(E)}\,.\,}\qquad\blacksquare
$$

### 한 페이지 요약 (시험에서 쓸 형태)

> *증명.* $P(E),P(H)>0$. 정의에 의해
> $P(H\mid E)=P(H\cap E)/P(E)$ (정의 1) 이고
> $P(E\mid H)=P(E\cap H)/P(H)$ (정의 2).
> 교집합의 대칭으로 $P(H\cap E)=P(E\cap H)$.
> 정의 2 로부터 $P(E\cap H)=P(E\mid H)P(H)$ (곱셈규칙).
> 이를 정의 1 의 분자에 대입하면
> $P(H\mid E)=P(E\mid H)P(H)/P(E)$. ∎

---

## 5. 검증

### 5.1 의학 검사 수치 예 (직관 점검)

병의 유병률 $P(H)=0.01$, 검사 민감도 $P(E\mid H)=0.99$ (병이 있으면 99% 양성), 특이도 96% — 거짓 양성률 $P(E\mid \neg H)=0.04$.

전확률로 $P(E)=P(E\mid H)P(H)+P(E\mid \neg H)P(\neg H)=0.99\cdot 0.01+0.04\cdot 0.99=0.0099+0.0396=0.0495$.

베이즈로 사후
$$
P(H\mid E)=\frac{0.99\cdot 0.01}{0.0495}\approx 0.20.
$$
즉 양성 검사에도 실제로 병일 확률은 20%. "양성 = 거의 확실히 병" 이라는 *직관 오류* 를 베이즈가 교정. 정리의 *유효성* 이 수치적으로 확인된다.

### 5.2 대칭 형태로 일관성

$P(H\mid E)P(E) = P(E\mid H)P(H)$ — 양변이 모두 $P(H\cap E)$ 와 같다. 베이즈 정리는 사실 *이 한 줄을 다시 쓴 것*.

### 5.3 가정의 필수성

$P(E)=0$ 이면 $P(H\mid E)$ 의 정의 자체가 불가 (분모 0). 마찬가지로 $P(H)=0$ 이면 $(\star\star)$ 의 정의 불가. 즉 $P(E),P(H)>0$ 이 *증명 시작 자체* 의 전제. (실용적으로 데이터가 한 번이라도 관측됐다면 거의 항상 성립.)

---

## 6. 일반화·통찰

### 6.1 MAP 정당화 ($P(E)$ 를 무시 가능한 이유)

$\theta$ 에 대한 추론에서 베이즈 정리는
$$
P(\theta\mid D)=\frac{P(D\mid \theta)P(\theta)}{P(D)}.
$$
$P(D)$ 는 $\theta$ 와 무관한 *상수* 이므로 $\arg\max_\theta P(\theta\mid D)$ 를 구할 때 무시 가능:
$$
\arg\max_\theta P(\theta\mid D) = \arg\max_\theta P(D\mid \theta)P(\theta).
$$
이것이 **MAP (Maximum A Posteriori)** 추정의 정의이며, Quiz 6 (Beta MAP), Quiz 7 (Strong prior 극한), Quiz 9 (Triangular MAP) 모두 이 단순화 위에서 작동.

### 6.2 MLE 와의 관계

$P(\theta)$ 가 *균등* (uniform) 이면 $P(\theta)$ 도 상수 ⇒ MAP = MLE. 즉 **MLE 는 prior 가 무정보일 때의 MAP 의 특수 사례**.

### 6.3 전확률 정리와의 결합

실제 계산에서는 $P(E)$ 를 직접 알기 어려운데, 가설들이 표본공간을 분할한다면
$$
P(H_i\mid E)=\frac{P(E\mid H_i)P(H_i)}{\sum_j P(E\mid H_j)P(H_j)}.
$$
이것이 분류기 / Naive Bayes 의 정확한 후방 확률 식.

### 6.4 비율 형태 (odds form)

두 가설 $H_1, H_2$ 의 후방 비율:
$$
\frac{P(H_1\mid E)}{P(H_2\mid E)} = \frac{P(E\mid H_1)}{P(E\mid H_2)}\cdot\frac{P(H_1)}{P(H_2)}.
$$
$P(E)$ 가 자동 소거. *우도비 × 사전비 = 사후비*. 의학 진단의 *likelihood ratio* 이론.

### 6.5 Bayesian update 의 반복 적용

데이터가 *순차* 관측되면 어제의 사후가 오늘의 사전:
$$
P(\theta\mid D_1, D_2)\propto P(D_2\mid \theta)\,\underbrace{P(\theta\mid D_1)}_{\text{어제의 posterior}}.
$$
독립 관측 가정 하에 likelihood 가 곱해지고 사전이 *누적*됨. 온라인 학습의 토대.

### 6.6 신경망과의 연결

분류 신경망의 softmax 출력 $p_i = P(\text{class}=i\mid x;\theta)$ 는 *학습된* 사후. 학습은 베이즈 정리의 *근사* 라고 해석할 수 있다 (NLL 최소화가 KL 최소화이고, 그것이 사후분포 일치이기 때문 — Quiz 8 참조).

---

## 7. 시험 출제 변형 5가지

### 변형 1. 곱셈규칙 (chain rule) 증명

> $P(A\cap B)=P(A\mid B)P(B)$ 임을 보여라.

*풀이.* 정의 $P(A\mid B)=P(A\cap B)/P(B)$ 의 양변에 $P(B)$ 곱.

### 변형 2. 베이즈를 odds 형태로 변환

> $\frac{P(H\mid E)}{P(\neg H\mid E)}$ 가 $\frac{P(E\mid H)}{P(E\mid \neg H)}\cdot\frac{P(H)}{P(\neg H)}$ 임을 보여라.

*풀이.* 분자와 분모에 베이즈 정리 적용 후 $P(E)$ 가 약분.

### 변형 3. 의학 수치

> 병 유병률 0.001, 검사 민감도 99%, 거짓 양성률 5%. 양성일 때 병일 확률은?

*풀이.* $P(E)=0.99\cdot 0.001+0.05\cdot 0.999=0.00099+0.04995=0.05094$. $P(H\mid E)=0.99\cdot 0.001/0.05094\approx 0.0194$. 약 1.9%.

### 변형 4. 분모를 전확률로

> $\{H_1,\ldots,H_n\}$ 이 분할이라면 $P(H_i\mid E)$ 의 완전한 닫힌 형태를 적어라.

*풀이.*
$$
P(H_i\mid E)=\frac{P(E\mid H_i)P(H_i)}{\sum_{j=1}^n P(E\mid H_j)P(H_j)}.
$$

### 변형 5. MAP = MLE 가 되는 조건

> Prior 가 어떤 형태일 때 MAP 과 MLE 가 같아지나?

*풀이.* $P(\theta)$ 가 (지원되는 영역 위에서) *상수* — 즉 무정보 균등 prior. 이때 $\arg\max_\theta P(D\mid\theta)P(\theta)=\arg\max_\theta P(D\mid\theta)$.

---

## 8. 백지 재현 체크리스트

1. [ ] 조건부확률 정의 $P(A\mid B)=P(A\cap B)/P(B)$ 를 적을 수 있다.
2. [ ] 가정 $P(E)>0, P(H)>0$ 의 필요성을 설명할 수 있다.
3. [ ] 정의를 $A=H,B=E$ 로 한 번 적용 → $(\star)$.
4. [ ] 정의를 $A=E,B=H$ 로 한 번 적용 → $(\star\star)$.
5. [ ] 교집합의 대칭 $H\cap E=E\cap H$ 의 한 줄 근거.
6. [ ] $(\star\star)$ 양변에 $P(H)$ 곱해 곱셈규칙 도출.
7. [ ] 곱셈규칙을 $(\star)$ 분자에 대입.
8. [ ] 최종 식 $P(H\mid E)=P(E\mid H)P(H)/P(E)$ 를 깔끔히 적을 수 있다.
9. [ ] 전확률 정리로 $P(E)$ 를 분해할 수 있다.
10. [ ] MAP 에서 $P(E)$ 무시 가능한 이유 한 줄 설명.
11. [ ] MAP = MLE 가 되는 조건 (균등 prior).
12. [ ] 의학 검사 예의 직관 (양성 ≠ 거의 확실히 병).
13. [ ] Odds form 이 분모를 자동 소거함을 안다.

---

## 9. 핵심 공식 카드

```
[정의]
  P(A | B) = P(A ∩ B) / P(B)            (P(B) > 0)

[곱셈규칙 / chain rule]
  P(A ∩ B) = P(A | B) P(B) = P(B | A) P(A)

[Bayes 정리]
  P(H | E) = P(E | H) P(H) / P(E)
            posterior = (likelihood × prior) / evidence

[증명 4-단계]
  (1) 정의 1: P(H|E) = P(H∩E)/P(E)
  (2) 정의 2: P(E|H) = P(E∩H)/P(H)
  (3) 대칭:  P(H∩E) = P(E∩H)
  (4) 곱셈규칙으로 P(E∩H)=P(E|H)P(H) → 대입.

[전확률]
  P(E) = Σ_i P(E | H_i) P(H_i)         ({H_i} 가 분할)

[MAP]
  arg max_θ P(θ | D) = arg max_θ P(D | θ) P(θ)
       (P(D) 는 θ 와 무관한 상수)
  prior 균등 ⇒ MAP = MLE

[Odds form]
  P(H₁|E) / P(H₂|E) = [P(E|H₁)/P(E|H₂)] · [P(H₁)/P(H₂)]
       (likelihood ratio × prior odds)
```

---

## 10. 다른 퀴즈와의 연결

- **Quiz 5 (Bernoulli MLE):** 본 퀴즈의 *prior 가 균등* 인 특수 사례. $\theta^*=k/n$ 의 정당화는 베이즈 정리 → MAP → 균등 prior → MLE 의 경로.
- **Quiz 6 (Beta MAP):** 베이즈 정리에 Beta(2,2) prior 를 곱한 결과의 closed form $\theta^{**}=(k+1)/(n+2)$. 본 퀴즈의 $P(E)$ 무시가 정확히 작동하는 사례.
- **Quiz 7 (Strong Prior 극한):** prior 가 데이터를 압도하는 양상 — $M\to\infty$ 에서 사후가 prior 의 정점 $\theta=1/2$ 로 수렴. 베이즈 정리의 분자가 prior 항에 의해 지배되는 극한.
- **Quiz 9 (Triangular Prior MAP):** Hard support prior 가 데이터의 모드를 강제로 잘라내는 현상. 베이즈 정리 + support 제약.
- **Quiz 8 (KL → CE → NLL):** 분류·회귀 손실의 *우도가정* 정당화. NLL 최소화 = posterior maximization 의 한 형태.
- **Quiz 10 (Backprop):** softmax 출력 $p_i = P(\text{class}=i\mid x;\theta)$ 를 *학습된 사후* 로 해석할 때, 베이즈 정리가 *왜 이런 형태가 자연스러운지* 의 이론적 정당화.
- **개념적 위계:** Quiz 4 (Bayes) → Quiz 5 (MLE = uniform-prior MAP) → Quiz 6,7,9 (구체적 prior 의 MAP) — 본 퀴즈가 통계적 학습 전체의 *씨앗*.

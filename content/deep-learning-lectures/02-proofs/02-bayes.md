---
title: "01. 베이즈 정리 (Bayes' Theorem) — 완전 증명"
slug: bayes
order: 2
---

# 01. 베이즈 정리 (Bayes' Theorem) — 완전 증명

> **출제 근거**: 3주차 ★10, 퀴즈 9번 직접 출제 (\"Bayes Theorem 증명\")
> **시험 출제 방식**: \"Prove Bayes' Theorem and explain each step.\"

---

## 1. 왜 시험에 나오는가

- 강의 전체 framework의 출발점. MLE, MAP, NLL, MSE, Cross-Entropy 등 **모든 후속 개념의 뿌리**.
- 강의에서 \"외우지 말고 유도하라\"고 명시.
- 색상 코딩 (🔴 Posterior / 🟢 Likelihood / 🔵 Prior) 도입 지점.

---

## 2. 필요한 사전 수학 (중1 → 대학원)

### 2.1 [중1] 확률이란

확률 = \"전체 중에 그 사건이 차지하는 비율\".

$$
P(A) = \frac{A\text{가 일어나는 경우의 수}}{\text{전체 경우의 수}}
$$

예: 주사위에서 짝수가 나올 확률 $= 3/6 = 1/2$.

### 2.2 [중2] 교집합과 합집합

- $A \cap B$ = A와 B가 **동시에** 일어남
- $A \cup B$ = A 또는 B가 일어남

### 2.3 [고1] 조건부 확률 (Conditional Probability) — **핵심**

\"B가 이미 일어났다는 조건 하에서 A가 일어날 확률\":

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \quad (\text{단, } P(B) > 0)
$$

**기호 해체 (Line-by-Line):**

| 기호 | 읽는 법 | 의미 |
|------|--------|------|
| $P(A \mid B)$ | \"P of A given B\" | B가 일어났다는 가정 하의 A의 확률 |
| $P(A \cap B)$ | \"P of A intersect B\" | A와 B가 동시에 일어날 확률 (joint) |
| $P(B)$ | \"P of B\" | B의 (marginal) 확률 |
| $\mid$ | \"given\" / 조건부 표시 | \"~라고 가정할 때\" |

**왜 이렇게 정의하나?**

\"B가 일어났다\"는 정보를 알면, 우리의 \"전체 표본공간\"이 B로 줄어든다. 그 줄어든 표본공간 안에서 A가 일어난 비율이 바로 $P(A\mid B)$.

> 📝 직관: 전체 학생 100명 중 남학생 60명, 그 중 안경 쓴 사람 30명. \"남학생일 때 안경 쓸 확률\" = $30/60 = 0.5$. 이걸 식으로 = $P(\text{안경} \cap \text{남}) / P(\text{남}) = 0.3/0.6 = 0.5$.

### 2.4 [고2] Joint Probability의 대칭성

$A \cap B$ 와 $B \cap A$ 는 \"A와 B가 동시에 일어남\"으로 같은 사건:

$$
P(A \cap B) = P(B \cap A)
$$

이 한 줄이 베이즈 정리의 **모든 마법**.

---

## 3. 베이즈 정리 — Statement

$$
\boxed{\;
\underbrace{P(H \mid E)}_{\color{red}\text{Posterior}}
=
\frac{
\underbrace{P(E \mid H)}_{\color{green}\text{Likelihood}}
\cdot
\underbrace{P(H)}_{\color{blue}\text{Prior}}
}{
\underbrace{P(E)}_{\text{Evidence}}
}
\;}
$$

**기호 해체 (Line-by-Line):**

| 기호 | 의미 | 색 | 직관 |
|------|------|-----|------|
| $H$ | Hypothesis (가설) | — | 우리가 진실인지 알고 싶은 것. 예: \"동전이 공정하다\", $\theta = 0.5$ |
| $E$ | Evidence (증거, 데이터) | — | 우리가 관측한 것. 예: \"3번 던져서 3번 앞면\" |
| $P(H \mid E)$ | Posterior | 🔴 | **데이터 본 후** $H$를 얼마나 믿는가. ← 진짜 알고 싶은 것 |
| $P(E \mid H)$ | Likelihood | 🟢 | $H$가 참이라면 이 데이터가 나올 확률 |
| $P(H)$ | Prior | 🔵 | **데이터 보기 전** $H$를 얼마나 믿었는가 |
| $P(E)$ | Evidence (marginal) | — | $H$를 모를 때 데이터의 확률 (정규화 상수) |

> 💡 **핵심 통찰**: 이 식은 \"prior 믿음을 데이터로 update해서 posterior 믿음을 만드는 공식\". 이것이 강의에서 말한 **Belief Update = Learning**.

---

## 4. 증명 (Proof) — 두 가지 방식

### 4.1 증명 1: Joint의 대칭성 활용 (가장 표준)

#### Step 1: 조건부 확률 정의 두 번 쓰기

$H$ 기준 조건부 확률 정의 (식 ①):

$$
P(H \mid E) = \frac{P(H \cap E)}{P(E)}
\tag{1}
$$

**왜?** 2.3절 정의 그대로. \"E가 일어났을 때 H의 확률\" = (둘 다 일어남) / (E의 확률).

$E$ 기준 조건부 확률 정의 (식 ②):

$$
P(E \mid H) = \frac{P(E \cap H)}{P(H)}
\tag{2}
$$

**왜?** 똑같은 정의를 $E$와 $H$의 역할만 바꿔 적용.

#### Step 2: 식 ②에서 joint 분리

식 ② 양변에 $P(H)$를 곱:

$$
P(E \cap H) = P(E \mid H)\, P(H)
\tag{3}
$$

**왜?** 분모를 양변으로 옮긴 단순 산수. 분수의 분모를 곱하면 분자만 남는다 (중1 산수).

#### Step 3: 대칭성 적용

\"E와 H가 동시에 일어남\" = \"H와 E가 동시에 일어남\":

$$
P(H \cap E) = P(E \cap H)
\tag{4}
$$

**왜?** 사건의 교집합은 순서 무관 (2.4절).

#### Step 4: 식 ①에 대입

식 ①의 분자 $P(H \cap E)$를 식 (4)로 바꾸고, 다시 식 (3)으로 바꿈:

$$
P(H \mid E) = \frac{P(H \cap E)}{P(E)} = \frac{P(E \cap H)}{P(E)} = \frac{P(E \mid H)\, P(H)}{P(E)}
$$

#### Step 5: 결론

$$
\boxed{\;
P(H \mid E) = \frac{P(E \mid H)\, P(H)}{P(E)}
\;}
$$

증명 끝. ∎

---

### 4.2 증명 2: 곱셈 정리 (Product Rule) 두 번 쓰기 — **암기용**

조건부 확률 정의를 양변에 분모를 곱한 형태로 쓰면:

$$
P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)
\tag{*}
$$

이 두 표현이 같다는 것이 핵심:

$$
P(H \mid E) P(E) = P(E \mid H) P(H)
$$

양변을 $P(E)$로 나누면:

$$
P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)} \quad \blacksquare
$$

> ⏱️ 시험에서 시간이 없으면 이 (*) 한 줄을 적고 \"양변이 같으므로 정리하면\" 식으로 1줄에 끝낼 수 있음. 단, **각 항이 무엇인지** (Prior / Likelihood / Posterior) 라벨링은 반드시.

---

## 5. Evidence $P(E)$의 전개 (Total Probability)

$P(E)$ 는 보통 직접 알기 어렵고, hypothesis 들에 대해 marginalize:

### 5.1 이산 (Discrete) — 가설이 유한개

$$
P(E) = \sum_{i} P(E \mid H_i)\, P(H_i)
$$

**왜?** 전체 확률의 법칙. \"E\"는 어떤 가설 하에 일어나든 다 합치면 됨.

### 5.2 연속 (Continuous) — 가설이 연속 파라미터 $\theta$

$$
P(E) = \int P(E \mid \theta)\, P(\theta)\, d\theta
$$

**왜?** 합 → 적분.

### 5.3 시험에서의 트릭

$P(E)$ 는 $H$에 대해 **상수** (data만의 함수)이므로, posterior를 **최대화** 할 때는 무시 가능:

$$
\arg\max_H P(H \mid E) = \arg\max_H \; P(E \mid H) P(H)
$$

→ 이것이 다음 토픽 **MAP**의 출발점 (`03_MAP_일반화_유도.md`).

---

## 6. 모범 답안 작성 템플릿 (영어 출제 대비)

> **Q. Prove Bayes' Theorem.**

```
[Setup]
Let H be a hypothesis and E be an event (data).
Assume P(E) > 0 so that conditional probability is well-defined.

[Step 1 — Conditional probability definitions]
By the definition of conditional probability:
  P(H | E) = P(H ∩ E) / P(E)         ... (1)
  P(E | H) = P(E ∩ H) / P(H)         ... (2)

[Step 2 — Symmetry of joint]
Since H ∩ E and E ∩ H denote the same event,
  P(H ∩ E) = P(E ∩ H)                ... (3)

[Step 3 — Substitute (2) and (3) into (1)]
From (2): P(E ∩ H) = P(E | H) P(H)
Using (3) in (1):
  P(H | E) = P(E ∩ H) / P(E)
           = P(E | H) P(H) / P(E)

[Conclusion]
  P(H | E) = P(E | H) P(H) / P(E).        ∎

[Interpretation]
- P(H)        : prior belief about H before seeing data
- P(E | H)    : likelihood — probability of data assuming H
- P(H | E)    : posterior — updated belief after seeing data
- P(E)        : evidence (normalizer); marginal of data

This is the formal statement that *learning is belief update*:
prior × likelihood, normalized, gives the posterior.
```

> ✅ 이 템플릿을 그대로 외우지 말고, **각 줄에 \"왜\"를 붙여 말로 설명할 수 있을 때까지** 연습.

---

## 7. 자주 틀리는 함정

1. **분자/분모 헷갈림**: $P(E\mid H)$ 와 $P(H \mid E)$ 위치 혼동. → \"우리가 알고 싶은 게 분자 좌변\".
2. **대칭성 단계 생략**: $P(H \cap E) = P(E \cap H)$ 를 당연시하고 안 쓰면 감점 가능 (\"왜 같은가\" 한 줄 필요).
3. **$P(E) > 0$ 조건 생략**: 분모가 0이면 정의 안 됨 — 한 줄로 명시.
4. **Prior와 Posterior 라벨링 누락**: 식만 적으면 0점에 가까움. 각 항을 색/단어로 라벨.

---

## 8. 연결 개념 (다음 학습)

- → [02. MLE Bernoulli](02_MLE_Bernoulli_완전유도.md): Posterior 최대화에서 prior가 uniform일 때 = MLE
- → [03. MAP 일반화](03_MAP_일반화_유도.md): Posterior 최대화 일반형
- → [14. Hypothesis Space Restriction = MAP](14_Hypothesis_MAP.md): prior가 hypothesis space를 제한하는 시각

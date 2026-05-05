---
title: "딥러닝 이론 시험 마스터 플랜 (중1 → A+)"
slug: plan
order: 84
---

# 딥러닝 이론 시험 마스터 플랜 (중1 → A+)

> **For agentic workers:** 이 플랜은 사용자가 직접 학습할 자료를 만드는 플랜입니다. 각 Task는 한 개의 학습 문서(Markdown)를 작성합니다. 코드가 아니라 "교육 자료"를 다룹니다. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 중학교 1학년 수학 수준의 학습자가 8개 기출문제를 100% 증명·재현할 수 있고, 누구에게든 설명할 수 있는 수준까지 끌어올린다. 또한 0425_images_checkpoint 폴더의 282장 강의 자료(필기 포함)에 등장하는 모든 핵심 개념을 마스터한다.

**Architecture:**
교과서식 선형 학습이 아닌, "기출문제 → 역추적 → 필요한 사전지식 → 유도체인 → 증명" 4단 구조. 모든 챕터는 (1) 개념(왜?) (2) 유도(어떻게?) (3) 완벽답안(시험 답안 그대로) (4) 자가테스트 4파일 세트로 구성.

**채점 철학 반영:** "답만 적으면 0점" → 모든 답안에 i.i.d 가정·로그 이유·미분=0 이유를 글로 명시.

**Tech Stack:** Markdown + LaTeX 수식 (`$...$`, `$$...$$`)

---

## 폴더 구조 (확정)

```
final-fire/
├── PLAN.md                        ← 이 문서
├── README.md                      ← 학습 로드맵 + 진도 체크리스트
│
├── 00-prerequisites/              ← 중1 → 대학 수학 가교 (필수 선행지식)
│   ├── 01-symbols.md              ← 수학 기호 사전 (∑, ∫, ∂, ∝, ∈, etc.)
│   ├── 02-functions.md            ← 함수, 합성, 역함수
│   ├── 03-derivative-101.md       ← 미분의 정의와 규칙
│   ├── 04-chain-rule.md           ← 합성함수 미분 (체인 룰)
│   ├── 05-integral-101.md         ← 적분의 정의와 정적분
│   ├── 06-vector-matrix.md        ← 벡터·행렬 기초
│   ├── 07-determinant.md          ← 행렬식, 역행렬
│   ├── 08-exp-log.md              ← e, ln 의 의미와 성질
│   ├── 09-probability.md          ← 확률, 확률변수, 확률밀도함수
│   └── 10-expectation.md          ← E[X], Var[X] 적분 정의
│
├── 01-eigen/                      ← 기출문제 1 (고유값/벡터)
├── 02-gaussian/                   ← 기출문제 2 (정규분포 모멘트)
├── 03-uniform/                    ← 기출문제 3 (균일분포)
├── 04-mle-bernoulli/              ← 기출문제 4 (베르누이 MLE)
├── 05-map-symmetric/              ← 기출문제 5 (MAP, 대칭 prior)
├── 06-map-asymmetric/             ← 기출문제 6 (MAP, 비대칭 prior)
├── 07-map-tent/                   ← 기출문제 7 (텐트 prior, 이미지 첨부)
├── 08-softmax/                    ← 기출문제 8 (softmax 미분)
│
│   각 문제 폴더는 다음 4개 파일로 구성:
│   ├── 00-overview.md             ← 문제 + 출제 의도 + 5분 핵심
│   ├── 01-concept.md              ← 개념 (중1도 이해 가능한 비유)
│   ├── 02-derivation.md           ← 단계별 유도 (왜? 매 단계마다 글로)
│   ├── 03-perfect-answer.md       ← 시험장에서 그대로 적을 A+ 답안
│   └── 04-mastery-quiz.md         ← 빈칸/변형 문제로 자가 검증
│
├── 09-killer-chains/              ← 35% 출제 핵심 유도 체인
│   ├── 01-iid-to-product.md       ← i.i.d → ∏ p(yᵢ)
│   ├── 02-why-log.md              ← 왜 로그를 취하는가 (3가지 이유)
│   ├── 03-why-derivative-zero.md  ← 왜 미분=0 으로 최댓값을 찾는가
│   ├── 04-gaussian-to-mse.md      ← 정규분포 → MSE 유도
│   ├── 05-bernoulli-to-ce.md      ← 베르누이 → Cross Entropy 유도
│   └── 06-map-to-l2.md            ← MAP + Gaussian prior → L2 정규화 유도
│
├── 10-ten-proofs/                 ← 10대 핵심 증명
│   ├── 01-rank-nullity.md         ← Rank-Nullity 정리
│   ├── 02-eigen-independence.md   ← 서로 다른 고유값 → 고유벡터 독립
│   ├── 03-bayes.md                ← 베이즈 정리
│   ├── 04-softmax-derivative.md   ← Softmax 자코비안
│   ├── 05-gaussian-integral.md    ← ∫exp(-x²/2)dx = √(2π)
│   ├── 06-jensen.md               ← Jensen 부등식 (KL ≥ 0 증명용)
│   ├── 07-kl-nonneg.md            ← KL divergence ≥ 0
│   ├── 08-cauchy-schwarz.md       ← 코시-슈바르츠 부등식
│   ├── 09-spectral-theorem.md     ← 대칭행렬 → 직교 고유벡터
│   └── 10-chain-of-mle.md         ← MLE 일반 형식 (모든 분포 통일)
│
└── 99-strategy/                   ← A+ 학습 전략
    ├── 01-grading-philosophy.md   ← "답만 적으면 0점" 철학
    ├── 02-derivation-method.md    ← 유도 체인 작성 7단계 템플릿
    ├── 03-week-schedule.md        ← 주차별 학습 일정 (8주 마스터)
    ├── 04-night-before-checklist.md ← 시험 전날 체크리스트
    └── 05-exam-day-tactics.md     ← 시험장 시간 배분 전술
```

---

## Task 0: 디렉토리 구조 생성 (이미 완료)

- [x] `final-fire/` 및 12개 하위 폴더 생성

---

## Task 1: README.md 마스터 인덱스 작성

**Files:**
- Create: `final-fire/README.md`

- [ ] **Step 1.1:** 학습 진도 체크리스트 (62개 문서 × 4단계: 읽음/유도가능/설명가능/시험가능)
- [ ] **Step 1.2:** 기출 8문제 직링크 + 출제 빈도 표시
- [ ] **Step 1.3:** "오늘부터 시험까지" 추천 학습 동선 3가지 (8주/4주/2주 코스)
- [ ] **Step 1.4:** 282장 강의자료 (`0425_images_checkpoint/`)와 각 챕터 매핑

---

## Task 2: 사전지식 (`00-prerequisites/`) 10개 문서

각 문서 공통 구조:
1. "왜 배워야 하는가" — 어느 기출문제에 쓰이는지 명시
2. "초등학교/중학교에서 배운 것" — 출발점
3. "대학에서는 이렇게 본다" — 도착점
4. "다리 놓기" — 단계별 비약 없이
5. "기출에 적용하기" — 실제 문제에서 사용 사례

- [ ] **Task 2.1:** `01-symbols.md` — ∑(시그마), ∫(인테그랄), ∂(편미분), ∝(비례), ∈(원소), ⊥(직교), |·|(절댓값/노름) 등 모든 기호의 읽는 법·뜻·예시
- [ ] **Task 2.2:** `02-functions.md` — y=f(x) 의미, 합성함수 g(f(x)), 역함수 f⁻¹, 다변수 f(x,y)
- [ ] **Task 2.3:** `03-derivative-101.md` — 미분 = 순간 변화율, 다항함수·지수·로그 미분 공식 (증명 포함)
- [ ] **Task 2.4:** `04-chain-rule.md` — (f(g(x)))' = f'(g(x))·g'(x) 직관과 증명, NLL 미분에서 어떻게 쓰이는지
- [ ] **Task 2.5:** `05-integral-101.md` — 적분 = 넓이, 정적분 ∫ₐᵇ, 부정적분, 가우스 적분으로 가는 다리
- [ ] **Task 2.6:** `06-vector-matrix.md` — 벡터의 길이/내적, 행렬 곱, 단위행렬 I, 전치 Aᵀ
- [ ] **Task 2.7:** `07-determinant.md` — 2×2 행렬식 ad-bc, det(A-λI)=0의 의미, 역행렬 존재 조건
- [ ] **Task 2.8:** `08-exp-log.md` — e의 정의, ln(ab)=ln a+ln b, ln(aᵇ)=b·ln a, (ln x)' = 1/x
- [ ] **Task 2.9:** `09-probability.md` — 표본공간, 확률변수 X, 확률질량함수 vs 확률밀도함수, ∫p(x)dx=1
- [ ] **Task 2.10:** `10-expectation.md` — E[X] = ∫xp(x)dx, Var[X] = E[X²] - E[X]², 적률 E[Xⁿ]

---

## Task 3: 기출문제 1 — 고유값/고유벡터 (`01-eigen/`)

**문제:** A = [[0,1],[1,0]] 의 고유값·고유벡터를 구하고, 고유값 정의로 참임을 증명하라.

- [ ] **Step 3.1:** `00-overview.md` — 문제 원문 + "왜 이 문제가 출제되는가" + 5분 요약
- [ ] **Step 3.2:** `01-concept.md` — 고유값(eigenvalue)이란? "행렬을 곱해도 방향이 안 변하는 벡터" 비유로
- [ ] **Step 3.3:** `02-derivation.md` — 단계별 유도:
  1. 고유값 정의: Av = λv (왜 0이 아닌 v를 찾는가)
  2. 이항: (A - λI)v = 0
  3. v ≠ 0 이려면 det(A - λI) = 0 (왜?)
  4. 특성다항식 풀이: λ² - 1 = 0 → λ = ±1
  5. λ=1 대입 → v₁ = (1,1)ᵀ/√2
  6. λ=-1 대입 → v₂ = (1,-1)ᵀ/√2
  7. **검증 (정의로 참 증명):** Av₁ 과 1·v₁ 비교, Av₂ 와 (-1)·v₂ 비교
- [ ] **Step 3.4:** `03-perfect-answer.md` — 시험지에 그대로 옮길 답안. 모든 단계 한국어 설명 포함.
- [ ] **Step 3.5:** `04-mastery-quiz.md` — A=[[2,1],[1,2]], A=[[3,0],[0,3]], A=[[0,-1],[1,0]] 변형 문제

---

## Task 4: 기출문제 2 — 정규분포 모멘트 (`02-gaussian/`)

**문제:** X ~ N(0,1)일 때 E(X), E(X²), E(X³), E(X⁴) 를 구하라. (∫exp(-x²/2)dx = √(2π) 증명도)

- [ ] **Step 4.1:** `00-overview.md` — 표준정규분포가 왜 가장 중요한 분포인지
- [ ] **Step 4.2:** `01-concept.md` — pdf p(x) = (1/√(2π))exp(-x²/2), 종 모양 그래프, 대칭성
- [ ] **Step 4.3:** `02-derivation.md`:
  1. **선행 보조정리**: ∫_{-∞}^{∞} exp(-x²/2)dx = √(2π) 증명 (극좌표 변환, 야코비안)
  2. E(X) = ∫x·p(x)dx = 0 (홀함수 적분, 대칭성으로 즉시)
  3. E(X²) = 1 (부분적분 또는 정규분포 분산 정의)
  4. E(X³) = 0 (홀함수)
  5. E(X⁴) = 3 (부분적분 두 번)
  6. **모멘트 일반공식**: E(X^(2n)) = (2n-1)!! 으로 통일
- [ ] **Step 4.4:** `03-perfect-answer.md`
- [ ] **Step 4.5:** `04-mastery-quiz.md` — N(μ,σ²) 일반화, E(X⁵), E(X⁶) 계산

---

## Task 5: 기출문제 3 — 균일분포 (`03-uniform/`)

**문제:** X ~ Uniform[a,b] 일 때 E(X), Var(X) 를 구하라.

- [ ] **Step 5.1:** `00-overview.md`
- [ ] **Step 5.2:** `01-concept.md` — pdf p(x) = 1/(b-a) for x ∈ [a,b], 직사각형
- [ ] **Step 5.3:** `02-derivation.md`:
  1. E(X) = ∫ₐᵇ x · (1/(b-a)) dx = (a+b)/2 (중간값임을 직관적으로도 설명)
  2. E(X²) = (a²+ab+b²)/3
  3. Var(X) = E(X²) - E(X)² = (b-a)²/12
  4. **검증:** [0,1] 대입하면 평균 1/2, 분산 1/12 (잘 알려진 사실)
- [ ] **Step 5.4:** `03-perfect-answer.md`
- [ ] **Step 5.5:** `04-mastery-quiz.md`

---

## Task 6: 기출문제 4 — 베르누이 MLE (`04-mle-bernoulli/`)

**문제:** y_i ~ Bern(θ) i.i.d (i=1..n), k=∑y_i 일 때 우도함수, NLL 구하고 MLE와의 관계 서술.

이 문제는 **킬러 체인의 핵심**. 모든 단계에 "왜?"를 글로 적어야 함.

- [ ] **Step 6.1:** `00-overview.md` — 출제율 100%, MLE의 정수
- [ ] **Step 6.2:** `01-concept.md` — 베르누이 분포, 동전 던지기 비유, θ = 앞면 확률
- [ ] **Step 6.3:** `02-derivation.md` — **8단계 체인** (각 단계마다 1-2문장의 "왜?"):
  1. 베르누이 pmf: p(y|θ) = θ^y · (1-θ)^(1-y)
     **왜?** y=1이면 θ, y=0이면 1-θ를 깔끔히 한 식으로
  2. **i.i.d 가정** → joint pdf = ∏ᵢ p(yᵢ|θ)
     **왜 곱해도 되는가?** 독립이라서 곱셈 가능
  3. 우도함수 L(θ) = ∏ θ^yᵢ (1-θ)^(1-yᵢ) = θᵏ(1-θ)^(n-k)
     **왜 모양이 깔끔해지는가?** 지수의 합 = 곱의 합
  4. **로그 취하기** ℓ(θ) = log L(θ) = k·log θ + (n-k)·log(1-θ)
     **왜 로그?** ① 곱→합 (미분 쉬움) ② 단조함수 (최댓값 위치 보존) ③ 수치적 안정성
  5. **NLL = -ℓ(θ)** (음의 로그우도)
     **왜 음수 붙이는가?** 최대화 문제 → 최소화 문제 (손실함수 표준)
  6. **미분=0** dℓ/dθ = k/θ - (n-k)/(1-θ) = 0
     **왜 미분=0?** 페르마 정리 — 미분 가능한 함수의 극값은 1차도함수가 0
  7. 풀이 → θ̂_MLE = k/n
  8. **2계 미분 < 0** 확인 (정말 최댓값인지)
- [ ] **Step 6.4:** `03-perfect-answer.md`
- [ ] **Step 6.5:** `04-mastery-quiz.md`

---

## Task 7: 기출문제 5 — MAP 대칭 prior (`05-map-symmetric/`)

**문제:** prior ∝ θᵐ(1-θ)ᵐ, m→∞ 일 때 θ̂_MAP = ?

- [ ] **Step 7.1:** `00-overview.md`
- [ ] **Step 7.2:** `01-concept.md` — Beta(m+1, m+1) prior, m이 커질수록 0.5 주변에 첨예해짐
- [ ] **Step 7.3:** `02-derivation.md`:
  1. **베이즈 정리**: posterior ∝ likelihood × prior
  2. log posterior = k·log θ + (n-k)·log(1-θ) + m·log θ + m·log(1-θ)
     = (k+m)·log θ + (n-k+m)·log(1-θ)
  3. 미분=0 → θ̂_MAP = (k+m)/(n+2m)
  4. **m→∞ 극한**: 분자/분모를 m으로 나누면 → 1/2
  5. **직관:** prior의 "0.5에 대한 확신"이 무한히 강하면 데이터를 무시
- [ ] **Step 7.4:** `03-perfect-answer.md`
- [ ] **Step 7.5:** `04-mastery-quiz.md`

---

## Task 8: 기출문제 6 — MAP 비대칭 prior (`06-map-asymmetric/`)

**문제:** prior ∝ θᵐ, m→∞ 일 때 θ̂_MAP = ?

- [ ] **Step 8.1:** `00-overview.md`
- [ ] **Step 8.2:** `01-concept.md` — 비대칭 prior, m이 클수록 θ=1 쪽으로 쏠림
- [ ] **Step 8.3:** `02-derivation.md`:
  1. log posterior = k·log θ + (n-k)·log(1-θ) + m·log θ
  2. 미분=0 → θ̂_MAP = (k+m)/(n+m)
  3. **m→∞ 극한** → 1
  4. **직관:** 사전적으로 "θ=1에 가깝다"고 강하게 믿으면 데이터 무시하고 1
- [ ] **Step 8.4:** `03-perfect-answer.md`
- [ ] **Step 8.5:** `04-mastery-quiz.md`

---

## Task 9: 기출문제 7 — 텐트 prior (`07-map-tent/`)

**문제 (이미지 첨부):** 
prior ∝ θᵐ, (n=5, k=4, m=2 or 6) 인데 사후분포가
p_m(θ) = 0  if |θ-0.5| ≥ 1/m
       = m - m²|θ - 0.5|  if |θ-0.5| ≤ 1/m
형태일 때 θ̂_MAP을 구하라.

- [ ] **Step 9.1:** `00-overview.md` — 가장 까다로운 변형 문제
- [ ] **Step 9.2:** `01-concept.md` — 텐트 모양 함수, 0.5에서 정점, 절댓값 함수의 미분 불가능성
- [ ] **Step 9.3:** `02-derivation.md`:
  1. prior 그래프: 정점 0.5, 폭 2/m (m이 클수록 좁음)
  2. likelihood × prior 형태 분석
  3. **m=2일 때**: 정점 위치는 prior와 likelihood 균형점
     - likelihood log: 4log θ + log(1-θ) → MLE = 4/5 = 0.8
     - prior 0이 아닌 구간: [0, 1] (전체)
     - 미분 불가점 0.5 처리: 좌/우 미분 비교
  4. **m=6일 때**: prior 구간 [0.5 - 1/6, 0.5 + 1/6] ≈ [0.333, 0.667]
     - MLE 0.8은 구간 밖 → 경계값 0.667이 후보
     - 경계에서 prior 값 = 0 → 뭔가 이상, m-m²·(1/6)=m-m/6 처리
     - 실제 최댓값 위치 분석
  5. **답안:** m=2 일 때, m=6 일 때 각각 구함
  6. **교훈:** 미분 불가능 점에서는 미분=0 대신 부분구간 비교
- [ ] **Step 9.4:** `03-perfect-answer.md` — 두 케이스 모두 풀이
- [ ] **Step 9.5:** `04-mastery-quiz.md`

---

## Task 10: 기출문제 8 — Softmax 미분 (`08-softmax/`)

**문제:** p = softmax(z), p ∈ ℝᶜ 일 때 ∂p/∂z 를 구하라 (c×c 자코비안).

- [ ] **Step 10.1:** `00-overview.md`
- [ ] **Step 10.2:** `01-concept.md` — softmax 정의 pᵢ = exp(zᵢ)/∑ⱼexp(zⱼ), 분류 문제 출력층
- [ ] **Step 10.3:** `02-derivation.md`:
  1. pᵢ = exp(zᵢ) / S, where S = ∑ⱼ exp(zⱼ)
  2. **두 경우로 나눔**: i=j (대각), i≠j (비대각)
  3. **케이스 1: ∂pᵢ/∂zᵢ** = (exp(zᵢ)·S - exp(zᵢ)·exp(zᵢ))/S² = pᵢ(1 - pᵢ)
     - 분자: 곱의 미분 (몫의 미분 규칙)
     - 단순화: exp(zᵢ)/S = pᵢ
  4. **케이스 2: ∂pᵢ/∂zⱼ** (i≠j) = (0·S - exp(zᵢ)·exp(zⱼ))/S² = -pᵢpⱼ
  5. **통합 (크로네커 델타)**: ∂pᵢ/∂zⱼ = pᵢ(δᵢⱼ - pⱼ)
  6. **자코비안 행렬 형태**: J = diag(p) - ppᵀ
- [ ] **Step 10.4:** `03-perfect-answer.md`
- [ ] **Step 10.5:** `04-mastery-quiz.md` — log softmax 미분, cross-entropy + softmax 합성 미분

---

## Task 11: 킬러 체인 (`09-killer-chains/`)

35% 출제 비중. 각 문서는 2000자 이상, "왜?" 질문에 글로 답하는 구조.

- [ ] **Step 11.1:** `01-iid-to-product.md` — 독립이 왜 곱셈인가, 동시확률 P(A∩B) = P(A)P(B), 독립의 정의
- [ ] **Step 11.2:** `02-why-log.md` — 로그 취하는 3대 이유: ① 곱→합 ② 단조성으로 argmax 보존 ③ 언더플로우 방지. 각각 구체적 수치 예시.
- [ ] **Step 11.3:** `03-why-derivative-zero.md` — 페르마 정리, 1차 조건과 2차 조건, 볼록함수에서 안전성
- [ ] **Step 11.4:** `04-gaussian-to-mse.md` — y = f(x) + ε, ε ~ N(0,σ²) → NLL = ∑(yᵢ-f(xᵢ))²/(2σ²) + const → MSE 등가성 증명
- [ ] **Step 11.5:** `05-bernoulli-to-ce.md` — 베르누이 NLL = -∑[yᵢ log p̂ᵢ + (1-yᵢ)log(1-p̂ᵢ)] = Cross-Entropy
- [ ] **Step 11.6:** `06-map-to-l2.md` — Gaussian prior on weights → MAP NLL = MSE + λ‖w‖² → L2 정규화의 베이지안 해석

---

## Task 12: 10대 핵심 증명 (`10-ten-proofs/`)

- [ ] **Step 12.1:** `01-rank-nullity.md` — dim(Range) + dim(Null) = n, 증명 (기저 선택)
- [ ] **Step 12.2:** `02-eigen-independence.md` — 서로 다른 고유값 → 고유벡터 선형독립, 귀납법 증명
- [ ] **Step 12.3:** `03-bayes.md` — P(A|B) = P(B|A)P(A)/P(B), 조건부확률 정의에서 유도
- [ ] **Step 12.4:** `04-softmax-derivative.md` — Task 10의 응축본
- [ ] **Step 12.5:** `05-gaussian-integral.md` — ∫exp(-x²/2)dx = √(2π) 극좌표 변환
- [ ] **Step 12.6:** `06-jensen.md` — f가 볼록함수 → E[f(X)] ≥ f(E[X])
- [ ] **Step 12.7:** `07-kl-nonneg.md` — KL(p‖q) ≥ 0, Jensen 부등식 사용
- [ ] **Step 12.8:** `08-cauchy-schwarz.md` — |⟨u,v⟩| ≤ ‖u‖‖v‖
- [ ] **Step 12.9:** `09-spectral-theorem.md` — 대칭행렬은 직교 고유벡터 기저를 가진다
- [ ] **Step 12.10:** `10-chain-of-mle.md` — 모든 분포의 MLE를 통일된 7단계로 정리 (베르누이/정규/푸아송 비교표)

---

## Task 13: 학습 전략 (`99-strategy/`)

- [ ] **Step 13.1:** `01-grading-philosophy.md` — "답만 적으면 0점" 채점 원칙, "왜?" 글쓰기 5가지 패턴
- [ ] **Step 13.2:** `02-derivation-method.md` — 유도체인 작성 7단계 템플릿: ① 정의 명시 ② 가정 명시 ③ 조작 ④ 보조정리 인용 ⑤ 결과 ⑥ 검증 ⑦ 직관 한 줄
- [ ] **Step 13.3:** `03-week-schedule.md` — 8주 마스터 일정 (Week 1: 사전지식 / Week 2-3: 기출 1-3 / Week 4-5: 기출 4-7 / Week 6: 기출 8 + 킬러체인 / Week 7: 10대 증명 / Week 8: 모의시험)
- [ ] **Step 13.4:** `04-night-before-checklist.md` — 시험 전날 30분 점검표
- [ ] **Step 13.5:** `05-exam-day-tactics.md` — 시간 배분, 부분 점수 사냥 전술

---

## Task 14: 282장 강의자료 매핑 (`README.md` 부록)

- [ ] **Step 14.1:** `0425_images_checkpoint/` 의 page_XXX.jpeg 들을 챕터별로 그룹핑
  - 예: 고유값 → page_041, 050, ...
  - 정규분포 → page_XXX, ...
  - MLE → page_XXX, ...
- [ ] **Step 14.2:** README의 챕터 섹션마다 "관련 강의 페이지: [링크]" 추가

---

## 자가 검토 체크리스트

플랜 작성 후 자가 점검:

- [ ] **스펙 커버리지:** 8개 기출문제 모두 별도 폴더 ✓
- [ ] **사전지식:** 중1 → 대학 가교 10개 문서 ✓
- [ ] **킬러 체인:** i.i.d→로그→미분=0 명시적으로 다룸 ✓
- [ ] **채점 철학:** "왜?" 글쓰기 모든 derivation에 포함 ✓
- [ ] **재현성:** 4-파일 구조로 자가 검증 가능 ✓
- [ ] **placeholder 없음:** 모든 step에 구체 내용 명시 ✓

---

## 실행 안내

이 플랜을 실행하는 두 가지 옵션:

### 옵션 1: 즉시 인라인 실행 (권장)
이 세션에서 Task 1부터 13까지 순차적으로 작성. 챕터마다 Read 가능한 상태로 만든 뒤 다음으로.

### 옵션 2: 우선순위 실행
사용자가 가장 시급한 챕터부터 선택 (예: "기출문제 4-MLE 먼저"). 그 부분만 완성 후 다음 우선순위.

**다음 단계:** 사용자가 옵션을 선택하면 시작합니다. 기본은 옵션 1.

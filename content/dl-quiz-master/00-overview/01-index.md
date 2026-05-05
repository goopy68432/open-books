---
title: '통합 인덱스 — 딥러닝이론 기말고사 마스터'
description: '13개 퀴즈 통합 인덱스 + 5챕터 학기 흐름 요약'
draft: false
---

> **사용법**: 이 인덱스에서 학습할 Quiz 를 선택 → 해당 파일만으로 그 토픽 시험 준비 완료.
> **시험일**: 2026-05-08 (목) 18:30~22:30 / 508호
> **출제 비중**: 중간고사 이전 33% (Q1~Q7 변형) + 6~9주차 66% (Q8~Q13 + 개념)

---

## 1. 전체 13개 퀴즈 인덱스

| # | 파일 | 주제 | 분량 | Tier |
|---|------|------|------|------|
| 1 | [Quiz01_선형변환_공리_검증.md](Quiz01_선형변환_공리_검증.md) | 선형변환의 두 공리 (가법성·동차성) | 12KB | B |
| 2 | [Quiz02_Four_Fundamental_Subspaces.md](Quiz02_Four_Fundamental_Subspaces.md) | R(A), R(Aᵀ), N(A), N(Aᵀ) 4공간 | 14KB | B |
| 3 | [Quiz03_Softmax_Jacobian.md](Quiz03_Softmax_Jacobian.md) | $\text{diag}(p) - pp^\top$ | 12KB | **S** |
| 4 | [Quiz04_Bayes_Theorem_증명.md](Quiz04_Bayes_Theorem_증명.md) | 조건부확률 정의 → 4단계 증명 | 13KB | **S** |
| 5 | [Quiz05_Bernoulli_MLE.md](Quiz05_Bernoulli_MLE.md) | $\theta^* = k/n$ | 7KB | **S** |
| 6 | [Quiz06_Beta_Prior_MAP.md](Quiz06_Beta_Prior_MAP.md) | $\theta^* = (k+1)/(n+2)$ | 8KB | A |
| 7 | [Quiz07_Strong_Prior_극한.md](Quiz07_Strong_Prior_극한.md) | $M \to \infty$ → $1/2$ | 8KB | A |
| 8 | [Quiz08_Gaussian_KL_MSE.md](Quiz08_Gaussian_KL_MSE.md) | KL → MSE 동치성 증명 | 13KB | A |
| 9 | [Quiz09_Triangular_Prior_MAP.md](Quiz09_Triangular_Prior_MAP.md) | piecewise + 2차방정식 풀이 | 16KB | A |
| 10 | [Quiz10_Backpropagation_Chain_Rule.md](Quiz10_Backpropagation_Chain_Rule.md) | $\partial L/\partial z = p - e_y$ | 13KB | **S** |
| 11 | [Quiz11_1D_Convolution_계산.md](Quiz11_1D_Convolution_계산.md) | $w = [1,2]$, 4개 입력 | 7KB | A |
| 12 | [Quiz12_1D_Convolution_Matrix.md](Quiz12_1D_Convolution_Matrix.md) | $A \in \mathbb{R}^{6\times 7}$ Toeplitz | 8KB | **S** |
| 13 | [Quiz13_2D_Convolution_Matrix.md](Quiz13_2D_Convolution_Matrix.md) | $A \in \mathbb{R}^{4\times 9}$ BTTB | 10KB | **S** |

**총 분량**: 약 140KB / 3,421 줄 (각 파일 자족적, 한 파일만 봐도 해당 토픽 마스터)

---

## 2. 학기 흐름 5개 챕터로 보기

### 📐 Chapter A — 선형대수·미적분 도구 (Q1~Q3)

| # | 핵심 결과 | 시험 가치 |
|---|----------|----------|
| Q1 | 행렬 곱 = linear transformation | Q12·Q13 의 기초 |
| Q2 | rank + nullity = dim(domain) | conv matrix 차원 분석의 기초 |
| Q3 | $J_{\text{softmax}} = \text{diag}(p) - pp^\top$ | Q10 Backprop 의 도구 |

**Chapter A 한 줄 요약**: "행렬·미분의 도구 상자 — 후속 모든 퀴즈에 사용"

### 🎲 Chapter B — 베이지안 추론 사다리 (Q4~Q7)

| # | 핵심 결과 | 위치 |
|---|----------|------|
| Q4 | $P(H|E) = P(E|H) P(H) / P(E)$ | 모든 베이지안 풀이의 출발점 |
| Q5 | MLE: $\theta^* = k/n$ | uniform prior MAP 의 특수 케이스 |
| Q6 | MAP (Beta 2,2): $\theta^* = (k+1)/(n+2)$ | conjugate prior 입문 |
| Q7 | $M \to \infty$ MAP: $\theta^* \to 1/2$ | strong prior 극한 |

**Chapter B 한 줄 요약**: "Bayes → MLE → MAP → 극한 사다리 — 데이터와 prior 의 줄다리기"

### 📊 Chapter C — 정보이론과 손실함수 다리 (Q8)

| # | 핵심 결과 | 시험 가치 |
|---|----------|----------|
| Q8 | $D_{KL}(\mathcal{N}_1 \| \mathcal{N}_2) = (\mu_1-\mu_2)^2/(2\sigma^2)$ → MSE/2σ² | CE → NLL → MSE 동치성 |

**Chapter C 한 줄 요약**: "MSE 가 우연이 아닌 가우시안 가정 NLL"

### 🔍 Chapter D — Strong Prior 응용 + Optimization (Q9, Q10)

| # | 핵심 결과 | 시험 가치 |
|---|----------|----------|
| Q9 | Triangular prior MAP: $m=2 \to 2/3$, $m=6 \to 1/2$ (엄밀) | piecewise 함수 최적화 |
| Q10 | Backprop chain rule: $\partial L/\partial z = p - e_y$ | softmax + CE 미분 (분류의 핵심) |

**Chapter D 한 줄 요약**: "복잡한 prior 와 deep learning 의 그래디언트 핵심"

### 🖼️ Chapter E — CNN Inductive Bias (Q11~Q13)

| # | 핵심 결과 | 시험 가치 |
|---|----------|----------|
| Q11 | 1D conv 직접 계산 (Q11-(3) 출제표 오류 주의) | conv 정의 숙달 |
| Q12 | $A_{6\times7}$ Toeplitz | linear map 의 행렬표현 정리 |
| Q13 | $A_{4\times 9}$ BTTB (block Toeplitz) | 2D 일반화 + row-major 평탄화 |

**Chapter E 한 줄 요약**: "Convolution = sparse + weight-shared linear transformation = inductive bias"

---

## 3. 학습 우선순위 (Tier 별 진행)

### 🔴 Tier S — 절대 필수 (5개)
시험 첫 30분 안에 풀 수 있어야 함.

1. **Q4 (Bayes 증명)** — 모든 베이지안 풀이의 기초
2. **Q3 (Softmax Jacobian)** — Q10 의 도구
3. **Q5 (Bernoulli MLE)** — 가장 기본
4. **Q10 (Backprop)** — 분류 학습의 핵심
5. **Q12, Q13 (Conv Matrix)** — 9주차 강의 핵심 메시지

### 🟠 Tier A — 매우 중요 (5개)
Tier S 마스터 후 진행.

6. **Q9 (Triangular Prior MAP)** — 학생 다수가 틀림 → 출제 가능성 높음
7. **Q11 (1D Conv 계산)** — Q12 의 도구
8. **Q6, Q7 (MAP 극한)** — 중간고사 변형 가능성
9. **Q8 (KL → MSE)** — 4주차 NLL→MSE 사슬

### 🟡 Tier B — 중요 (3개)
시간 남으면.

10. **Q1 (선형변환 공리)** — Q12 의 개념적 배경
11. **Q2 (4 Fundamental Subspaces)** — 행렬 차원 분석

---

## 4. 각 마스터 파일 공통 구조

각 Quiz 파일은 **0~10절** 자족 구조:

| 절 | 내용 | 용도 |
|----|------|------|
| 0 | 한 줄 요약 | 30초 복습 |
| 1 | 문제 (출제 형태 그대로) | 시험 직전 확인 |
| 2 | 출제 의도와 시험 가치 | "왜 이게 시험에 나오나" |
| 3 | 사전 개념 (수학 도구 모음) | 수학 공포증 학생용 |
| 4 | 풀이 (모든 단계, 등호 근거) | 백지 재현 학습 |
| 5 | 검증 (수치/대입/일관성) | 답안 신뢰성 |
| 6 | 일반화·통찰 | 응용 문제 대비 |
| 7 | 시험 출제 변형 5가지 | 변형 출제 대비 |
| 8 | 백지 재현 체크리스트 | 자가 테스트 |
| 9 | 핵심 공식 카드 | 1분 복습용 |
| 10 | 다른 퀴즈와의 연결 | 통합 이해 |

→ **하나의 파일만 봐도 그 토픽 시험 준비 완료**.

---

## 5. 시험 직전 1주 학습 일정

| 일자 | 학습 파일 | 목표 |
|------|----------|------|
| Day 1 (5/1) | Q4, Q5 | 베이지안 기초 |
| Day 2 (5/2) | Q6, Q7, Q8 | MAP 사다리 + KL→MSE |
| Day 3 (5/3) | Q9 | Triangular Prior (틀린 학생 多) |
| Day 4 (5/4) | Q10 | Backprop chain rule |
| Day 5 (5/5) | Q11, Q12, Q13 | CNN 3종 |
| Day 6 (5/6) | Q1, Q2, Q3 | 도구 (선형대수·미적분) |
| Day 7 (5/7) | 모의시험 + 약점 보완 | 4시간 안에 모두 풀어보기 |
| Day 8 (5/8) | Tier S 5개만 빠르게 훑기 → 시험 | 18:30 시작 |

---

## 6. 핵심 공식 카드 (시험 직전 1분 복습)

### Tier S 5개 핵심
```
Q4: P(H|E) = P(E|H) · P(H) / P(E)        [Bayes]
Q3: J = diag(p) - p p^T                    [Softmax Jacobian]
Q5: θ* = k/n                               [Bernoulli MLE]
Q10: ∂L/∂z = p - e_y                       [Softmax + CE 미분]
Q12: A_ij Toeplitz, A의 j번째 열 = f(e_j)  [Conv Matrix]
```

### Tier A 핵심
```
Q9 (m=2): θ* = 2/3,  Q9 (m=6): θ* = 1/2 (엄밀)
Q11: w*x = (sum w_u · x_{i-u})
Q6: θ** = (k+1)/(n+2)
Q7: M→∞ ⇒ θ* → 1/2
Q8: D_KL(N(μ₁,1)||N(μ₂,1)) = (μ₁-μ₂)²/2
```

### Tier B 핵심
```
Q1: A(u+v) = Au + Av,  A(αu) = α Au
Q2: dim(domain) = rank + nullity
```

---

## 7. 연결망 — 어느 퀴즈가 어느 퀴즈를 사용하는가

```
Q1 (선형변환)
  └── Q12 (1D conv matrix) ── Q13 (2D conv matrix)
Q2 (4 subspaces)
  └── Q12 (conv matrix 차원)
Q3 (Softmax Jacobian)
  └── Q10 (Backprop, ∂p/∂z)
Q4 (Bayes)
  ├── Q5 (MLE = uniform prior MAP)
  ├── Q6 (Beta prior MAP)
  ├── Q7 (강한 prior 극한)
  └── Q9 (Triangular prior MAP)
Q5 (MLE) ── Q6 (MAP) ── Q7 (극한)
Q8 (KL → MSE)
  └── 4주차 NLL→MSE 사슬과 통합
Q11 (1D conv 계산)
  └── Q12 (1D conv matrix) ── Q13 (2D conv matrix)
```

---

## 8. 통합 슬로건

> **"이 13개 파일 = 33% 커버 (Q1~Q7 변형) + 66% 커버 (Q8~Q13 + 개념)"**
>
> **"각 파일은 자족적 — 모르는 토픽이 나오면 해당 Quiz 파일 하나만 펴라"**
>
> **"Inductive Bias = Strong Prior = Restriction = Sparse Matrix = Weight Sharing — 모두 같은 말"**

---

## 9. 보조 자료 (이 폴더 외부)

폴더 외부에 있는 관련 자료:

| 자료 | 위치 |
|------|------|
| 공식 퀴즈 종합본 (간단 버전) | `docs/QUIZ/공식퀴즈_종합.md` |
| 13문항 모범답안 | `docs/QUIZ/claude/모범답안.md` |
| 초보자용 학습 자료 | `docs/QUIZ/claude/퀴즈학습.md` |
| KL Divergence 학습 자료 | `docs/QUIZ/claude/KL_Divergence_Claude.md` |
| 6주차 Uniform Prior 복기 | `docs/QUIZ/claude/6주차_Uniform_Prior_퀴즈_복기.md` |
| 8주차 Newton's Method 복기 | `docs/QUIZ/claude/8주차_Newton_Method_퀴즈_복기.md` |
| 기말고사 전략 보고서 | `docs/QUIZ/claude/기말고사_전략_보고서.md` |

---

**최종 한 마디**:

> **"이 폴더 (master_files/claude/) 는 시험 직전 1주의 모든 것을 담고 있습니다. 각 Quiz 파일을 하나씩 마스터하면 13개 퀴즈 모두 백지 재현 가능합니다. 5월 8일 508호에서 화이팅! 🎯"**

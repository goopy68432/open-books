---
title: "딥러닝 이론 마스터 가이드 — README v2"
slug: master-readme-v2
order: 11
---

# 딥러닝 이론 마스터 가이드 — README v2

> **Version 2** — 7주차 강의 스크립트 직접 분석을 통해 **교수가 강조한 메시지**를 반영한 v2 마스터 자료 안내.
>
> 한양대 이성윤 교수 「딥러닝」 강의 (742장 슬라이드 + 강의 녹취록 7주차 분량) 전체를 마스터하기 위한 자료.

---

## ✅ v2 변경 사항 요약

| 항목 | v1 | v2 |
|------|-----|-----|
| 데이터 출처 | 슬라이드 742장만 | + 7주차 강의 스크립트 (수업_스크립트/) |
| 강조 표시 | (없음) | ★★★★★ ~ ★ 5단계 |
| 강의 인용 | (없음) | 직접 인용구 다수 |
| 메타 메시지 | (간략) | Deduction vs Induction · Tango · 채점철학 · 이론 vs 공학 4대 메시지 |
| Mermaid 다이어그램 | 6개 | 8개 (5대 핵심 체인 별도 섹션) |
| 시험 답안 가이드 | 일반 | 교수의 채점 철학 그대로 |

---

## 📁 v2 파일 구조

**위치:** `/Users/jeongseongchae/dev/university/deep_learning/docs/master/`

| 문서 | 크기 | 줄 수 | 핵심 |
|------|-----|------|-----|
| **master_readme.md** | 7.1KB | (v1) | v1 안내 |
| **master_readme_v2.md** | (이 문서) | - | v2 안내 |
| **MASTER-CONCEPTS.md** | 27KB | 726줄 | v1 (슬라이드 기반) |
| **MASTER-CONCEPTS_v2.md** | ~32KB | ~900줄 | **v2 (강의 강조 반영)** ★ |
| **LEARNING-MAP.md** | 19KB | 564줄 | v1 |
| **LEARNING-MAP_v2.md** | ~24KB | ~700줄 | **v2 (강의 흐름 반영)** ★ |

---

## 💡 v2의 핵심 통찰 — 교수가 직접 한 말 4가지

### 1. ★★★★★ "Deduction vs Induction"
> *"두 개의 차이를 이해하는 것이 중간고사 전까지 목표 중 굉장히 중요합니다. AI 역사에서는 인덕션이 이겼습니다."*
>
> *"문제를 인덕션으로 풀면 안 됩니다. 논리적으로 해야 됩니다."*

**의미:**
- AI의 역사적 흐름 = Deduction (Expert System) → Induction (Neural Network)
- 그러나 **시험 답안은 Deduction**으로 풀어야 함
- **수학적 귀납법 ≠ 일반적 인덕션** (사실은 deduction)

### 2. ★★★★★ "It Takes Two to Tango"
> *"이 두 가지 (Maximum Likelihood + Prior)를 이해하는 게 이 수업의 목표입니다. 이 둘을 이해하는 게 끝입니다."*

**의미:**
- 강의 전체의 목표는 **두 축**의 이해
- MLE: 데이터로부터 학습
- Prior: 사람의 사전 지식 (Inductive Bias)
- 모든 신경망 = 이 두 축의 균형

### 3. ★★★★★ "답만 적으면 0점"
> *"답이 아니라 왜 그 과정에 도달할 수 있었는지 ─ 인디펜던스도 어디서 쓰고, 미분에서 0이기 때문에 그렇다 ─ 이런 논리를 보는 겁니다."*
>
> *"수식어로만 쭉 쓰면 사실 의미가 없을 거고요. 그 과정을 잘 서술해주셔야 됩니다."*

**의미:**
- 모든 답안에 **"왜?" 설명** 필수
- 정리 인용 (페르마, 베이즈, Jensen 등)
- 가정 명시 (i.i.d, 미분 가능, etc.)

### 4. ★★★★★ "행렬은 마음의 고향"
> *"이인석 교수님 책에 이런 글이 있는데 ─ 행렬은 마음의 고향. 어떤 수학적 오브젝트를 만나더라도 행렬을 생각해야 한다."*
>
> *"우리가 아는 것은 행렬뿐이라고까지 강하게 얘기를 하십니다."*

**의미:**
- 모든 함수, 모든 미분, 모든 변환이 **행렬**
- Linear Algebra가 모든 것의 출발

---

## 📚 MASTER-CONCEPTS_v2.md 구조 (★ 표시 반영)

### 0. 강의의 메타 메시지 ★★★★★
- 0.1 Deduction vs Induction
- 0.2 It Takes Two to Tango (MLE + Prior)
- 0.3 이론 vs 공학적 산출물 (둘 다 중요)
- 0.4 답만 적으면 0점 (채점 철학)
- 0.5 행렬은 마음의 고향

### 1. 수학적 기초
- 1.1 ★★★★★ 선형대수 (Inner Product 면접 단골, Range/Null/Rank-Nullity, Eigenvalue/SVD)
- 1.2 미적분 (Linear Approx, Newton's Method, Jacobian, Softmax 자코비안)
- 1.3 ★★★★★ 확률 (Bernoulli vs Gaussian, Bayesian vs Frequentist, **Gauss 3-key**)

### 2. ★★★★★ 통계적 추정 (강의의 심장)
- 2.1 베이즈 정리 + Belief Update = Learning
- 2.2 MLE 7단계 체인
- 2.3 MAP — 데이터 + 지식 균형
- 2.4 NLL = 손실 함수의 정체 (Loss는 NLL의 결과)
- 2.5 MAP → 정규화
- 2.6 KL Divergence + 정보이론

### 3. 고전 ML
- 3.1 Hypothesis Space 제약 = Inductive Bias
- 3.2 Function Space → Parameter Space 전환
- 3.3 Perceptron (Rosenblatt 1958)
- 3.4 LDA vs QDA, SVM

### 4-11. 최적화, 신경망, 일반화, CNN, RNN, Transformer, 생성모델, 모던 응용
(상세는 v2 문서)

### 12. ★★★★★ 시험 답안 작성법 (교수의 채점 철학)

### 13. 강의의 8대 통합 메시지
1. 딥러닝 = 선형대수 + 미적분 + 확률
2. 모든 손실 함수는 분포 가정에서 (NLL이 통일 도구)
3. 모든 정규화는 Prior에서 (MAP 관점)
4. 모든 학습은 i.i.d → 로그 → 미분=0
5. Hypothesis Space 제약 = Inductive Bias
6. Linear Approx = 모든 최적화 근본
7. Function → Parameter Space 전환이 신경망
8. MLE ↔ Strong MAP, 균형이 best

---

## 🗺 LEARNING-MAP_v2.md 구조

### 0. 강의의 큰 그림 — 교수 직접 그린 지도
Mermaid: AI = Deduction or Induction → 인덕션 승 → MLE + Prior → 신경망

### 1. ★★★★★ 학습 순서 — 8주 코스 (강의 흐름 그대로)
- Stage 1: Why (1주차) — 메타 메시지
- Stage 2: 수학 기초 (2주차) — 행렬 = 선형변환
- Stage 3: Probability + Bayes (3주차) — **"가장 중요한 날"**
- Stage 4: MAP (4주차) — Prior 강도별 비교
- Stage 5: ERM = NLL (6-7주차) — Loss는 NLL의 결과
- Stage 6: Newton + Backprop (7주차) — Linear Approx
- Stage 7: 일반화 + 정규화
- Stage 8: 고급 (CNN/Transformer/생성모델)

### 2. ★★★★★ 강의의 핵심 흐름 (Mermaid)
8단계 stage를 모두 연결한 종합 다이어그램

### 3. 5대 핵심 체인 (Mermaid)
1. NLL 체인 (i.i.d → 로그 → 미분=0)
2. 분포 → 손실 매핑
3. MLE ↔ MAP 균형
4. Linear Approx (Newton's Method가 본질)
5. Inductive Bias 체인 (Hypothesis Space 제약)

### 4. "어디에 쓰이는가" 매트릭스 (강의 강조 응용)
8대 메시지 × 어디에 적용?

### 5. ★★★★★ 시험 답안 작성 체크리스트
- 답안 첫 줄 (모델 + 가정)
- 단계별 "왜?" 표준 문장
- 정리 인용 5개
- 절대 하지 말 것 4가지

### 6. 8주 학습 일정

### 7. 자가 진단 체크리스트 (Stage별)

### 9. ★★★★★ 마지막 한 줄
> **"이 수업은 '딥러닝을 사용하는 법'이 아니라 'AI 역사에서 인덕션이 이긴 이유'와 '그 인덕션을 뒷받침하는 수학(MLE + Prior)'을 가르친다."**

---

## 🎯 v2 사용 가이드

### 처음 시작하는 학생
1. **본 README v2 정독** (이 문서)
2. `LEARNING-MAP_v2.md` §1 학습 순서 5단계 파악
3. `MASTER-CONCEPTS_v2.md` §0 강의 메타 메시지 정독 (★★★★★ 표시 부분 집중)
4. Stage 1 (Why) → Stage 8 (생성모델) 순서대로

### 시험 임박 학생
1. `MASTER-CONCEPTS_v2.md` §12 (답안 작성법) 정독
2. `LEARNING-MAP_v2.md` §3 (5대 핵심 체인) Mermaid 외움
3. `LEARNING-MAP_v2.md` §5 (시험 답안 체크리스트) 외움
4. **베르누이 MLE 7단계 손유도** 매일 1번 (Cheat Card 역할)

### 강의 복습 학생
- 각 주차 학습 후 → `MASTER-CONCEPTS_v2.md`의 해당 섹션 정독
- 그 주의 인용구가 어떻게 강의 흐름에 위치하는지 확인

### 1주차: Stage 1 + Stage 2 시작 → §0, §1.1
### 2주차: Stage 2 완료 → §1.1
### 3주차: Stage 3 (가장 중요!) → §1.3, §2.1, §2.2
### 4주차: Stage 4 → §2.3
### 5주차: 중간고사
### 6주차: Stage 5 → §2.4, §2.5
### 7주차: Stage 6 → §3.1-§3.5, §5
### 8주차: Stage 7-8 + 모의시험 → §6-§11

---

## 🔗 v1 vs v2 비교

| 영역 | v1 | v2 |
|------|-----|-----|
| **메타 메시지** | 3개 (수학 조합, 왜?, NLL 보편성) | 8개 (Deduction/Induction 추가, 채점철학, 행렬 등) |
| **강의 인용** | 0개 | 30+개 직접 인용 |
| **★ 표시** | 없음 | 5단계 중요도 |
| **학습 흐름** | 5 stage | 8 stage (강의 주차 순서 그대로) |
| **Mermaid** | 6개 | 8개 (체인별 강조) |
| **시험 답안** | 일반 가이드 | 교수의 채점 철학 직접 인용 |

**v1과 v2 차이점:** v2는 "**교수가 직접 한 말**"이 곳곳에 박혀 있어서 학생이 강의실에서 받은 것과 같은 결의 자료입니다. v1은 슬라이드 기반의 객관적 정리.

→ **시험 답안 작성에는 v2를, 슬라이드 복습에는 v1을 권장.**

---

## 📂 final-fire/와의 관계 (변동 없음)

이 마스터 가이드는 시험 대비 자료(`final-fire/`)와 **상호 보완적**:
- **`docs/master/v2`** (이 폴더) = 과목 전체 큰 그림 + 교수 메타 메시지
- **`final-fire/`** = 시험 8문제 답안 작성 훈련

```
deep_learning/
├── docs/master/                    ← 이 폴더 (큰 그림 + 강의 인용)
│   ├── master_readme.md            (v1 안내)
│   ├── master_readme_v2.md         (이 문서)
│   ├── MASTER-CONCEPTS.md          (v1)
│   ├── MASTER-CONCEPTS_v2.md       (v2 — 강의 강조)
│   ├── LEARNING-MAP.md             (v1)
│   └── LEARNING-MAP_v2.md          (v2 — 강의 흐름)
│
├── docs/수업_스크립트/              ← v2의 출처 (강의 녹취 7주차)
│   ├── 딥러닝이론-1주차.md
│   ├── ...
│   └── 딥러닝이론-8주차.md
│
├── final-fire/                     ← 시험 대비
└── DL/0425_images/                 ← 원본 슬라이드 742장
```

---

## 🎓 마지막 한 줄

> **이 v2는 "강의 그 자체"입니다.**
>
> v1이 슬라이드의 객관적 요약이었다면, v2는 **교수의 목소리·강조·메시지가 그대로 살아있는** 자료입니다.
>
> v2의 ★★★★★ 표시된 항목들이 시험에 직접 출제될 가능성이 가장 높습니다.

---

**작성:** 2026-04-26 (v2)
**기반:** 7주차 강의 녹취록 (`수업_스크립트/`) + 742장 슬라이드
**문서 셋:** [`master_readme_v2.md`](./master_readme_v2.md) (이 문서) + [`MASTER-CONCEPTS_v2.md`](./MASTER-CONCEPTS_v2.md) + [`LEARNING-MAP_v2.md`](./LEARNING-MAP_v2.md)

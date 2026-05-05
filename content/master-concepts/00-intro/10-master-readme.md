---
title: "딥러닝 이론 마스터 가이드 — README"
slug: master-readme
order: 10
---

# 딥러닝 이론 마스터 가이드 — README

> 한양대 이성윤 교수 「딥러닝」 강의 (742장 슬라이드, 2025-11-28 update) 전체를 마스터하기 위한 단일 참조 문서 모음.

---

## ✅ 완성된 결과물

**위치:** `/Users/jeongseongchae/dev/university/deep_learning/docs/master/`

| 문서 | 크기 | 줄 수 | 핵심 |
|------|-----|------|-----|
| **MASTER-CONCEPTS.md** | 27KB | 726줄 | 14개 섹션, ~120개 LaTeX 수식 |
| **LEARNING-MAP.md** | 19KB | 564줄 | 5단계 학습순서 + 6개 Mermaid 관계도 |

---

## 📚 MASTER-CONCEPTS.md 구조 (14 섹션)

| § | 섹션 | 내용 |
|---|------|------|
| §0 | 과목 메시지 | 3대 메시지 (수학 조합, "왜?" 글로 설명, NLL 체인 보편성) |
| §1 | 수학적 기초 | 선형대수·미적분·확률 |
| §2 | 통계적 추정 | MLE·MAP·NLL·KL·CE |
| §3 | 고전 ML | ERM·LDA/QDA·SVM |
| §4 | 최적화 | GD·SGD·Adam·Lagrange |
| §5 | 신경망 | UAT·활성화·Backprop·Softmax |
| §6 | 일반화 | Bias-Variance·Double Descent·Sharpness |
| §7 | CNN | Convolution·Pooling·Receptive Field |
| §8 | RNN/LSTM | BPTT·게이트 식 |
| §9 | Attention/Transformer | Q,K,V·Multi-Head·Masked MHA |
| §10 | 현대 응용 | Transfer·Domain Adapt·CLIP·LLM |
| §11 | 생성모델 | GAN·VAE·Diffusion·ELBO |
| §12 | 10대 핵심 증명 색인 | (final-fire 참조) |
| §14-15 | 통합 메시지 + 자료 매핑 | final-fire/와의 연결 |

---

## 🗺 LEARNING-MAP.md 구조 (6개 Mermaid)

| § | 다이어그램 | 내용 |
|---|----------|------|
| §2 | 전체 개념 관계도 | 5단계 색상 분류, 노드 ~50개 |
| §3.1 | NLL 체인 | i.i.d → log → 미분=0 |
| §3.2 | 분포→손실 매핑 | Gauss→MSE, Bern→BCE 등 |
| §3.3 | 신경망 학습 체인 | Forward + Backprop |
| §3.4 | 일반화 체인 | Overfitting → Reg → Double Descent |
| §3.5 | Attention→Transformer 체인 | Q,K,V → Multi-Head → Decoder |
| §6 | 통합 학습 흐름도 | 8주 일정 |

---

## 🎯 검증 완료

- ✅ 8개 기출문제의 모든 개념 포함 (eigen, Gauss moment, Uniform, MLE Bern, MAP×3, Softmax)
- ✅ 모든 Mermaid 다이어그램 GitHub/NotebookLM 호환 (subgraph 라벨에 한글 지원)
- ✅ 모든 LaTeX 수식 `$...$` / `$$...$$` 형식 (NotebookLM 호환)
- ✅ 슬라이드 페이지 인용 (page_XXX 형식, 추적 가능)
- ✅ MASTER ↔ LEARNING 양방향 링크 작동

---

## 💡 핵심 통찰 (이 과목이 가르치고자 하는 것)

> **"딥러닝을 쓰는 법"이 아니라 "딥러닝이 왜 작동하는가"를 수학적으로 연역하는 사고법**
>
> 그 중심에 **i.i.d → 로그 → 미분=0 체인**과 **분포 가정 → 손실 함수 등가성**이 있음.

### 강의의 3대 메시지

1. **딥러닝 = 선형대수 + 미적분 + 확률의 조합**
   - 한 분야가 아니라 세 분야의 정수가 결합된 결과
2. **"왜?"를 글로 설명할 수 있어야 한다**
   - 교수님 채점 철학: "답만 적으면 0점"
3. **i.i.d → 로그 → 미분=0 체인의 보편성**
   - 모든 분포의 MLE/MAP은 동일한 7단계로 풀린다 (출제 35% 커버)

---

## 🔗 두 마스터 문서의 역할

| 문서 | 답하는 질문 | 사용 시점 |
|------|----------|---------|
| **MASTER-CONCEPTS.md** | "이 개념이 **무엇**인가? 어떻게 **유도**되는가?" | 개념 학습 시 |
| **LEARNING-MAP.md** | "이 개념을 **언제** 배워야 하나? **어디에** 쓰이나? **왜** 중요한가?" | 학습 계획·복습 시 |

두 문서는 상호 보완적으로 함께 사용해야 합니다.

---

## 📂 final-fire/ 폴더와의 관계

이 마스터 가이드는 시험 대비 자료(`final-fire/`)와 **상호 보완적**:

| 폴더 | 목적 | 특징 |
|------|-----|-----|
| `docs/master/` (이 폴더) | **과목 전체 큰 그림** | 시험 외 영역까지 포함, 상위 관점 |
| `final-fire/` | **시험 8문제 답안 훈련** | 5-파일 세트로 답안 작성 연습 |

### 시험 외 영역 (이 가이드만 다루는 내용)

- 미분방정식 기초
- Lagrange Multiplier
- Chebyshev/Hoeffding 부등식
- 중심극한정리
- LDA vs QDA
- SVM의 Margin
- 표본평균 수렴
- Sharpness 논쟁
- CNN 합성곱·풀링
- RNN/LSTM 게이트
- Self-Attention/Transformer
- Domain Adaptation
- CLIP, LLM 동향
- GAN/VAE/Diffusion 통합 관점

→ **시험을 넘어 과목 자체의 메시지**를 이해하는 자료.

---

## 🚀 학습 시작 가이드

### 처음 시작하는 경우
1. 본 README 정독
2. [`LEARNING-MAP.md`](./LEARNING-MAP.md) §1 학습 순서 5단계 파악
3. [`MASTER-CONCEPTS.md`](./MASTER-CONCEPTS.md) §0 과목 메시지 정독
4. Stage 1부터 [`LEARNING-MAP.md`](./LEARNING-MAP.md) §7 자가 진단 체크리스트로 진행

### 특정 주제 깊이 학습
- 개념 이해: `MASTER-CONCEPTS.md`의 해당 §
- 위치·관계 확인: `LEARNING-MAP.md`의 Mermaid 다이어그램
- 시험 답안 연습: `final-fire/`의 해당 챕터

### 시험 임박
- `final-fire/99-strategy/04-night-before-checklist.md` Cheat Card 1장
- `MASTER-CONCEPTS.md` §12 10대 증명 색인
- `LEARNING-MAP.md` §7 자가 진단 체크리스트

---

## 📊 전체 학습 자료 지도

```
deep_learning/
├── docs/master/                    ← 본 폴더 (과목 전체 큰 그림)
│   ├── master_readme.md            ← 이 문서
│   ├── MASTER-CONCEPTS.md          ← 14 섹션, 120 수식
│   └── LEARNING-MAP.md             ← 5단계, 6 Mermaid
│
├── final-fire/                     ← 시험 대비 (8 기출 + 사전지식)
│   ├── 00-prerequisites/
│   ├── 01-eigen ~ 08-softmax/      ← 기출 8문제
│   ├── 09-killer-chains/           ← 핵심 유도 체인
│   ├── 10-ten-proofs/              ← 10대 증명
│   ├── 11-extra-topics/            ← 추가 시험 범위
│   ├── 99-strategy/                ← 학습 전략
│   └── STUDY-MAP.excalidraw        ← 시각 학습 지도
│
├── docs/notebooklm/
│   ├── claude/                     ← final-fire 평탄화 (NotebookLM용)
│   ├── codex/                      ← codex 버전 평탄화
│   └── gemini/                     ← gemini 버전 평탄화
│
└── DL/0425_images/                 ← 원본 강의 슬라이드 742장
    └── 0425_images_checkpoint/     ← 필기 강의 슬라이드 282장
```

---

## 🎓 마지막 한 줄

> **이 과목은 "딥러닝을 사용하는 법"이 아니라 "딥러닝이 왜 작동하는가"를 수학적으로 연역해내는 사고법을 가르친다.**
>
> 그 핵심에는 **i.i.d → 로그 → 미분=0** 체인과 **MLE/MAP의 분포→손실 등가성**이 있다.
>
> 이 두 가지를 설명할 수 있으면 이 강의의 절반을 정복한 것이다.

---

**작성:** 2026-04-26
**기반:** 742장 강의 슬라이드 (한양대 이성윤 교수)
**문서 쌍:** [`MASTER-CONCEPTS.md`](./MASTER-CONCEPTS.md) + [`LEARNING-MAP.md`](./LEARNING-MAP.md)

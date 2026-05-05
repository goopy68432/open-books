---
title: "딥러닝 이론 마스터 가이드 — README v3"
slug: master-readme-v3
order: 12
---

# 딥러닝 이론 마스터 가이드 — README v3

> **🔥 Version 3 — 작년 시험 100% 출제 정보 반영 🔥**
>
> 한양대 이성윤 교수 「딥러닝」 강의 (742장 슬라이드 + 7주차 강의 스크립트 + **작년 시험 5장 이미지**) 종합 마스터 자료.
>
> **v3 핵심 정보:** 작년 시험 출제 9개 카테고리는 **100% 다시 출제**됨.

---

## ⚡ v3 신규 정보

### 🎯 작년 시험 100% 출제 9개 카테고리

작년 시험 이미지 5장 분석 결과:

| # | 카테고리 | 핵심 출제 |
|---|---------|---------|
| 1 | **Convex** | $x^2$, $-\log x$ convex 증명, 판별, set |
| 2 | **KL Divergence / Jensen** | KL ≥ 0, 정규분포 KL = ½(μ₁-μ₂)² |
| 3 | **Bias-Variance Decomposition** | 분해 증명 |
| 4 | **Gradient Descent 수렴 조건** | $\eta < 2/\lambda_{\max}(A)$ |
| 5 | **Convolution** | 1D/2D, $H_{out}$, Toeplitz matrix |
| 6 | **Markov Chain / DAG** | 조건부확률 분해 |
| 7 | **모델 비교** | Markov vs RNN vs Transformer |
| 8 | **PyTorch 코드** | forward → backward 5단계 |
| 9 | **Pooling** | Avg Pooling Matrix |

추가:
- **확률분포** (Uniform/Normal/Poisson) 평균·분산
- **MLE/MAP** 다양한 prior
- **손실함수 관계** (MSE = NLL = CE = H + KL)
- **행렬 미분** ($-\log\sigma(Ax+b)$)
- **생성 모델 NLL** (VAE/Diffusion)

---

## 📁 v3 파일 구조

**위치:** `/Users/jeongseongchae/dev/university/deep_learning/docs/master/`

| 문서 | 크기 | 용도 |
|------|-----|------|
| `master_readme.md` | 7KB | v1 안내 |
| `master_readme_v2.md` | 10KB | v2 안내 |
| **`master_readme_v3.md`** | (이 문서) | **v3 안내** |
| `MASTER-CONCEPTS.md` | 27KB | v1 본문 |
| `MASTER-CONCEPTS_v2.md` | 23KB | v2 본문 (강의 인용) |
| **`MASTER-CONCEPTS_v3.md`** | ~22KB | **v3 본문 (작년 영역 보강)** |
| `LEARNING-MAP.md` | 19KB | v1 지도 |
| `LEARNING-MAP_v2.md` | 18KB | v2 지도 |
| `PROFESSOR-PROOFS.md` | 25KB | 교수 직강 풀이 15개 |
| **`EXAM-BANK_v3.md`** | ~30KB | **★ 메인! 작년+올해 통합 풀이집** |
| **`STRATEGY_v3.md`** | ~12KB | **★ 시험 대비 전략 + 8주 일정 + Cheat Card** |

---

## 🎯 v3 학습 전략 (3-Tier)

### 🥇 Tier 1: 작년 시험 (100% 출제, 50% 시간)
- 10유형 풀이 패턴 그대로 외움
- `EXAM-BANK_v3.md`의 Tier 1 정독
- 손글씨 답안 5번씩 재현

### 🥈 Tier 2: 올해 기출 8문제 (매우 높음, 30%)
- 7단계 체인 마스터
- `final-fire/` 폴더 활용

### 🥉 Tier 3: 강의 추가 토픽 (변형 대비, 20%)
- 다양한 분포, 행렬 미분, VAE/Diffusion
- v2 본문 참조

---

## 🚀 사용 가이드

### 시험 임박 (1~2주 남음)
1. **`EXAM-BANK_v3.md`** Tier 1 정독 → 모든 풀이 손글씨
2. **`STRATEGY_v3.md`** Cheat Card 외움
3. 매일 1-2문제씩 답안 작성 연습 (시간 측정)

### 1개월 이상 (체계적)
1. **`STRATEGY_v3.md`** 8주 일정 따라가기
2. Week 1-4: Tier 1 (10유형 마스터)
3. Week 5-6: Tier 2 (올해 8문제)
4. Week 7: 모의시험
5. Week 8: 최종 점검

### 처음 시작
1. `master_readme_v3.md` (이 문서) 정독
2. `STRATEGY_v3.md` §3-Tier 전략 이해
3. `EXAM-BANK_v3.md` 통독으로 범위 파악
4. `MASTER-CONCEPTS_v3.md`로 개념 이해
5. 2번 반복

---

## 📚 v1 / v2 / v3 비교

| 측면 | v1 | v2 | v3 |
|------|-----|-----|-----|
| 출처 | 슬라이드 742장 | + 강의 스크립트 7주차 | **+ 작년 시험 이미지 5장** |
| 강조 | 8 기출 + 사전지식 | + 강의 인용 + ★ 표시 | **+ Tier 시스템 + 100% 보장** |
| 메인 산출물 | MASTER-CONCEPTS | + PROFESSOR-PROOFS | **+ EXAM-BANK** |
| 학습 전략 | 5 stage | 8 stage | **3-Tier × 8주 일정** |
| 시험 답안 | 일반 가이드 | 채점 철학 | **답안 7원칙 + Cheat Card** |

→ **v3을 메인으로**, v1·v2는 보조 (v3가 모든 v1·v2 핵심 포함).

---

## 🔥 시험 대비 핵심 5원칙

### 1. Tier 1은 무조건 100% 마스터
작년 9개 카테고리 → **그대로 출제**. 풀이 패턴 외워야 함.

### 2. 답안에 "왜?" 글로 적기
> *"답만 적으면 0점"* — 교수의 채점 철학.
모든 단계마다 i.i.d, 로그 이유, 페르마 등 명시.

### 3. 정리 인용 5개 외우기
- 페르마 정리 (1차 조건)
- 베이즈 정리 (MAP)
- Jensen 부등식 (KL ≥ 0)
- Spectral 정리 (대칭행렬)
- Hoeffding 부등식 (학습이론)

### 4. Cheat Card 1장 외우기
시험 전날 30분 → Cheat Card 1장 확인.
모든 Tier 1 핵심 식 + 7단계 체인.

### 5. 답안 시간 5분/문제
한 문제 8분 이상 X. 막히면 다음 + 부분점수 사냥.

---

## 💡 v3 핵심 통찰

### 작년 시험 패턴 분석

작년 시험에는 다음이 공존:
- **수학적 증명**: Convex, KL ≥ 0, Bias-Variance, GD 수렴
- **계산 문제**: 정규분포 KL, Convolution 출력, MAP 풀이
- **개념 비교**: 모델 비교 (Markov/RNN/Transformer)
- **코드 흐름**: PyTorch
- **매트릭스 표현**: Conv, Pooling

→ **순수 수학 (50%) + 응용 (30%) + 시스템 이해 (20%)** 구성.

### 시험 대비 이상적 상태

다음을 5분 안에 모두 할 수 있다면 만점 준비 완료:
1. $f(x) = x^2$ convex 증명 (정의 + 2차 미분)
2. KL ≥ 0 증명 (Jensen 사용)
3. 정규분포 N(μ₁,1) vs N(μ₂,1) KL 계산 = ½(μ₁-μ₂)²
4. Bias-Variance 분해 증명 (교차항 0)
5. GD η < 2/λ_max 증명 (오차 재귀 + 고유분해)
6. Convolution 출력 크기 + Toeplitz matrix
7. Markov 조건부확률 chain rule
8. Markov vs RNN vs Transformer 비교 표 (Inductive Bias)
9. PyTorch 5단계 학습 루프
10. Avg Pooling을 매트릭스로

---

## 📂 전체 학습 자료 지도

```
deep_learning/
├── docs/master/                    ← v1, v2, v3 마스터 자료
│   ├── master_readme.md            (v1 안내)
│   ├── master_readme_v2.md         (v2 안내)
│   ├── master_readme_v3.md         (★ 이 문서, v3 안내)
│   ├── MASTER-CONCEPTS.md          (v1 본문)
│   ├── MASTER-CONCEPTS_v2.md       (v2 본문, 강의 인용)
│   ├── MASTER-CONCEPTS_v3.md       (★ v3 본문, 작년 영역)
│   ├── LEARNING-MAP.md             (v1 지도)
│   ├── LEARNING-MAP_v2.md          (v2 지도)
│   ├── PROFESSOR-PROOFS.md         (v2 보조, 직강 풀이 15개)
│   ├── EXAM-BANK_v3.md             ★★★ v3 메인 풀이집
│   └── STRATEGY_v3.md              ★★★ v3 시험 전략
│
├── docs/수업_스크립트/              ← 7주차 강의 녹취 (v2/v3 출처)
│
├── final-fire/                     ← 시험 답안 작성 훈련 (Tier 2)
│
└── DL/0425_images/                 ← 원본 슬라이드 742장
```

---

## 🎯 v3로 시작하기 — 5분 빠른 시작

```
1️⃣  STRATEGY_v3.md §3-Tier 전략 (1분)
   → 어느 영역에 시간 투자할지 결정

2️⃣  EXAM-BANK_v3.md Tier 1 표 (1분)
   → 작년 출제 10유형 빠르게 훑기

3️⃣  본인 약점 1개 선택 (1분)
   → 가장 모르겠는 유형 1개

4️⃣  EXAM-BANK_v3.md 해당 풀이 정독 (1-2분)
   → 풀이 단계 + "왜?" 글로 이해

5️⃣  손글씨로 1번 재현 (5분)
   → 답안 작성 시뮬레이션
```

→ 매일 이 5분 사이클 1번씩 → **2주 안에 Tier 1 완성**.

---

## 🏆 마지막 메시지

> **"작년 문제는 100% 다시 나옵니다. Tier 1 10유형을 풀이 패턴 그대로 외우면, 시험 50% 이상이 자동으로 풀립니다."**
>
> **"답만 적으면 0점. 풀이 단계마다 '왜?'를 글로 적어야 합니다. 정리 5개 (페르마, 베이즈, Jensen, Spectral, Hoeffding)를 답안에 인용하세요."**
>
> **"이 v3 자료를 따라가면, 시험 만점에 가까운 답안을 작성할 수 있습니다. 화이팅!"**

---

**작성:** 2026-04-26 (v3)
**기반:** 작년 시험 5장 이미지 + 강의 스크립트 7주차 + 슬라이드 742장
**문서 셋:**
- [`EXAM-BANK_v3.md`](./EXAM-BANK_v3.md) ★ 메인
- [`STRATEGY_v3.md`](./STRATEGY_v3.md) ★ 전략
- [`MASTER-CONCEPTS_v3.md`](./MASTER-CONCEPTS_v3.md) 개념 보강
- [`master_readme_v3.md`](./master_readme_v3.md) (이 문서)

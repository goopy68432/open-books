---
title: "기출 1번 — 고유값/고유벡터 (Overview)"
slug: 01-eigen-00-overview
order: 11
---

# 기출 1번 — 고유값/고유벡터 (Overview)

## 문제 원문

$$A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

**문제:** A의 고유값(eigenvalue)과 고유벡터(eigenvector)를 구하고, **고유값 정의로 참임을 증명하라.**

---

## 출제 의도

이 문제는 두 가지를 동시에 평가합니다:
1. **계산 능력** — 특성방정식 풀이, 고유벡터 계산
2. **정의 이해** — "고유값 정의로 증명"이라는 표현은 단순 계산을 넘어 $A\mathbf{v} = \lambda \mathbf{v}$ 자체를 직접 검증하라는 의미.

A가 swap 행렬(좌표 바꾸기)이라는 직관도 함께 보면 만점.

---

## 5분 핵심

| 단계 | 내용 |
|-----|------|
| 1 | 고유값 정의: $A\mathbf{v} = \lambda \mathbf{v}$ ($\mathbf{v} \neq \mathbf{0}$) |
| 2 | 이항: $(A - \lambda I)\mathbf{v} = \mathbf{0}$ |
| 3 | 비자명 해 존재 → $\det(A - \lambda I) = 0$ |
| 4 | 특성방정식: $\lambda^2 - 1 = 0$ → $\lambda = \pm 1$ |
| 5 | $\lambda=1$: $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^T$ |
| 6 | $\lambda=-1$: $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, -1)^T$ |
| 7 | **검증:** $A\mathbf{v}_1 = 1 \cdot \mathbf{v}_1$, $A\mathbf{v}_2 = -1 \cdot \mathbf{v}_2$ |

---

## 학습 자료

| 파일 | 내용 |
|------|-----|
| [`01-concept.md`](./01-concept.md) | 고유값·고유벡터의 직관 (swap 행렬 예) |
| [`02-derivation.md`](./02-derivation.md) | 7단계 풀이 + 모든 단계의 "왜?" |
| [`03-perfect-answer.md`](./03-perfect-answer.md) | 시험장 그대로 답안 |
| [`04-mastery-quiz.md`](./04-mastery-quiz.md) | 변형 문제 5개 |

---

## 관련 사전지식

- [`../00-prerequisites/06-vector-matrix.md`](../00-prerequisites/06-vector-matrix.md) — 벡터·행렬
- [`../00-prerequisites/07-determinant.md`](../00-prerequisites/07-determinant.md) — 행렬식, 특성방정식

## 관련 강의 자료

`0425_images_checkpoint/` 의 다음 페이지가 고유값을 다룹니다 (시각 자료 참고):
- 선형대수 도입부 (page_034 ~ page_055 부근)
- Spectral Theorem 관련 (대칭행렬 직교 대각화)

(정확한 페이지 번호는 직접 폴더 확인 권장)

---

## 출제 변형 가능성

같은 패턴, 다른 행렬:
- $A = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$ (대각행렬 → 고유값=대각원소, 고유벡터=표준기저)
- $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$ (대칭, 비자명)
- $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ (회전 → 실수 고유값 없음, 복소수)

각 케이스의 풀이는 [`04-mastery-quiz.md`](./04-mastery-quiz.md)에 수록.

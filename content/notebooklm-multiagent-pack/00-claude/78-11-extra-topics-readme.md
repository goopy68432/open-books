---
title: "11. 추가 시험 범위 (Extra Topics)"
slug: 11-extra-topics-readme
order: 78
---

# 11. 추가 시험 범위 (Extra Topics)

> 강의에 등장한 추가 핵심 토픽 9개. 각 1파일에 (개념+유도+답안) 통합.

| # | 파일 | 우선순위 | 핵심 |
|---|------|--------|-----|
| 1 | [`01-activations.md`](./01-activations.md) | ★★★★★ | Sigmoid, ReLU, GELU 미분 |
| 2 | [`02-backpropagation.md`](./02-backpropagation.md) | ★★★★★ | 체인 룰의 신경망 적용 |
| 3 | [`03-bias-variance.md`](./03-bias-variance.md) | ★★★★★ | $E[(y-\hat{y})^2]$ 분해 증명 |
| 4 | [`04-information-theory.md`](./04-information-theory.md) | ★★★★ | H, H(p,q), KL, MI |
| 5 | [`05-universal-approx.md`](./05-universal-approx.md) | ★★★★ | UAT 인용 + 직관 |
| 6 | [`06-convexity.md`](./06-convexity.md) | ★★★★ | 볼록함수, Jensen 응용 |
| 7 | [`07-optimization.md`](./07-optimization.md) | ★★★ | SGD, Adam, 수렴 |
| 8 | [`08-initialization.md`](./08-initialization.md) | ★★★ | Xavier, He 분산 유도 |
| 9 | [`09-svd-pca.md`](./09-svd-pca.md) | ★★★ | SVD, PCA 유도 |

## 학습 동선
- 시간 부족: 1, 2, 3 우선
- 시간 여유: 1~6
- 깊은 이해: 1~9 모두

## 다른 폴더와의 관계
- 활성화·역전파 → 기출 8번(softmax)와 연결
- Bias-Variance → 정규화·MAP의 일반화 이론적 근거
- 정보이론 → CE 손실의 정체
- Convexity → 페르마 정리의 확장
- SVD → 고유값 분해의 일반화 (기출 1번 발전형)

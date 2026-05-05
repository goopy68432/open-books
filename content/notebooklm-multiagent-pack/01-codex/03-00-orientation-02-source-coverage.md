---
title: "원본 범위 커버리지"
slug: 00-orientation-02-source-coverage
order: 3
---

# 원본 범위 커버리지

이 문서는 `final-fire-codex`가 원본 `final-fire`의 어떤 내용을 어떤 새 문서에서 다루는지 보여줍니다.

| 원본 폴더/파일 | Codex 문서 | 커버 방식 |
|---|---|---|
| `00-prerequisites/01-symbols.md` | `01-novice/01-math-language.md`, `90-drills/flashcards.md` | 기호를 문장 번역 훈련으로 재구성 |
| `00-prerequisites/02-functions.md` | `01-novice/02-three-worlds.md` | 함수, 확률함수, 손실함수의 차이로 재배치 |
| `00-prerequisites/03-derivative-101.md` | `02-intermediate/02-estimation-engine.md` | 미분=0을 추정 엔진의 핵심 단계로 사용 |
| `00-prerequisites/04-chain-rule.md` | `02-intermediate/03-softmax-and-loss.md` | softmax+CE 합성 미분으로 연결 |
| `00-prerequisites/05-integral-101.md` | `02-intermediate/01-distribution-workbench.md` | 모멘트 계산 도구로 재배치 |
| `00-prerequisites/06-vector-matrix.md` | `03-advanced/02-geometry-and-regularization.md` | 선형변환과 정규화의 기하 관점 |
| `00-prerequisites/07-determinant.md` | `00-orientation/00-learning-map.md`, `03-advanced/02-geometry-and-regularization.md` | 고유값 엔진에 포함 |
| `00-prerequisites/08-exp-log.md` | `02-intermediate/02-estimation-engine.md` | 로그우도의 3가지 이유로 재구성 |
| `00-prerequisites/09-probability.md` | `01-novice/02-three-worlds.md`, `02-intermediate/02-estimation-engine.md` | pmf/pdf/i.i.d/베이즈 연결 |
| `00-prerequisites/10-expectation.md` | `02-intermediate/01-distribution-workbench.md` | $E[g(X)]$ 계산법 중심 |
| `01-eigen` | `03-advanced/02-geometry-and-regularization.md`, `04-master/02-exam-simulator.md` | 계산+정의 검증+변형 문제 |
| `02-gaussian` | `02-intermediate/01-distribution-workbench.md`, `03-advanced/01-proof-ladder.md` | 가우스 적분과 모멘트 계산 |
| `03-uniform` | `02-intermediate/01-distribution-workbench.md` | 평균/분산 계산 루틴 |
| `04-mle-bernoulli` | `02-intermediate/02-estimation-engine.md`, `04-master/01-master-synthesis.md` | 우도 엔진의 기준 예제 |
| `05-map-symmetric` | `02-intermediate/02-estimation-engine.md` | MAP 일반식과 극한 |
| `06-map-asymmetric` | `02-intermediate/02-estimation-engine.md` | prior의 방향성과 극한 |
| `07-map-tent` | `02-intermediate/02-estimation-engine.md`, `90-drills/error-clinic.md` | 구간 분할/경계/비미분점 전략 |
| `08-softmax` | `02-intermediate/03-softmax-and-loss.md` | 자코비안과 CE 그래디언트 |
| `09-killer-chains` | `04-master/01-master-synthesis.md`, `90-drills/flashcards.md` | 딥러닝 손실함수의 공통 체인 |
| `10-ten-proofs` | `03-advanced/01-proof-ladder.md` | 10개 증명을 난이도별 사다리로 재배열 |
| `11-extra-topics/01-activations.md` | `05-extended-theory/01-activation-functions.md` | 활성화 함수 식/미분/vanishing 원인 |
| `11-extra-topics/02-backpropagation.md` | `05-extended-theory/02-backprop-computational-graph.md` | 계산 그래프와 행렬형 역전파 |
| `11-extra-topics/03-bias-variance.md` | `05-extended-theory/06-bias-variance-double-descent.md` | Bias-Variance 증명과 Double Descent |
| `11-extra-topics/04-information-theory.md` | `05-extended-theory/07-cross-entropy-kl.md` | CE = H + KL 분해 |
| 강의 page_338 활성화 함수 | `05-extended-theory/01-activation-functions.md` | Sigmoid/Tanh/ReLU/LeakyReLU/GELU/Softplus |
| 강의 page_435 double descent | `05-extended-theory/06-bias-variance-double-descent.md` | Double Descent와 implicit regularization |
| 최적화/역전파 추가 범위 | `05-extended-theory/03-optimization-algorithms.md`, `05-extended-theory/04-vanishing-exploding-gradients.md` | GD/SGD/Momentum/RMSProp/Adam, gradient 문제 |
| 일반화/정규화 추가 범위 | `05-extended-theory/05-generalization-regularization.md` | Dropout, BatchNorm, regularized ERM |
| `99-strategy` | `04-master/02-exam-simulator.md`, `90-drills/error-clinic.md` | 시험 답안 훈련과 부분점수 전략 |

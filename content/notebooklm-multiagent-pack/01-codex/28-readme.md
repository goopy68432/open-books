---
title: "Final Fire Codex"
slug: readme
order: 28
---

# Final Fire Codex

이 폴더는 `final-fire`를 그대로 요약한 판본이 아니라, 같은 내용을 다른 관점으로 다시 배우도록 만든 새 학습 자료입니다.

원본은 기출문제별로 잘 정리되어 있습니다. 이 Codex 버전은 학습자의 머릿속에 다음 4개의 층을 쌓는 방식으로 재구성했습니다.

| 단계 | 목표 | 핵심 질문 | 사용 문서 |
|---|---|---|---|
| 입문 | 기호와 식을 읽는다 | "이 식이 한국어로 무슨 말인가?" | `01-novice/` |
| 중급 | 계산 루틴을 재현한다 | "어떤 문제든 어떤 엔진에 넣어 풀 것인가?" | `02-intermediate/` |
| 고급 | 증명과 모델링 이유를 말한다 | "왜 이 조작이 정당한가?" | `03-advanced/` |
| 마스터 | 변형 문제를 설계하고 방어한다 | "채점자가 꼬아서 물어도 버틸 수 있는가?" | `04-master/` |
| 확장 | 딥러닝 학습 이론으로 연결한다 | "손실을 얻은 뒤 어떻게 학습하고 일반화하는가?" | `05-extended-theory/` |

## 학습법

각 문서를 다음 순서로 다룹니다.

1. 읽기: 식을 눈으로 훑지 말고 한 줄씩 말로 번역합니다.
2. 가리기: 공식을 보고 외우는 대신 제목만 보고 처음부터 재구성합니다.
3. 손풀이: 계산은 반드시 종이에 씁니다. 특히 로그, 미분, 경계 검사는 생략하지 않습니다.
4. 방어: 마지막에는 "왜?" 질문 3개에 답합니다.

## 빠른 루트

시간이 적으면 아래 순서만 먼저 완료합니다.

1. `00-orientation/00-learning-map.md`
2. `01-novice/01-math-language.md`
3. `02-intermediate/02-estimation-engine.md`
4. `02-intermediate/03-softmax-and-loss.md`
5. `03-advanced/01-proof-ladder.md`
6. `04-master/02-exam-simulator.md`

추가 시험 범위까지 포함하면 아래를 이어서 봅니다.

7. `05-extended-theory/01-activation-functions.md`
8. `05-extended-theory/02-backprop-computational-graph.md`
9. `05-extended-theory/06-bias-variance-double-descent.md`
10. `05-extended-theory/07-cross-entropy-kl.md`

## 전체 구성

```text
final-fire-codex/
├── README.md
├── 00-orientation/
│   ├── 00-learning-map.md
│   ├── 01-diagnostic.md
│   └── 02-source-coverage.md
├── 01-novice/
│   ├── 01-math-language.md
│   └── 02-three-worlds.md
├── 02-intermediate/
│   ├── 01-distribution-workbench.md
│   ├── 02-estimation-engine.md
│   └── 03-softmax-and-loss.md
├── 03-advanced/
│   ├── 01-proof-ladder.md
│   └── 02-geometry-and-regularization.md
├── 04-master/
│   ├── 01-master-synthesis.md
│   └── 02-exam-simulator.md
├── 05-extended-theory/
│   ├── 00-overview.md
│   ├── 01-activation-functions.md
│   ├── 02-backprop-computational-graph.md
│   ├── 03-optimization-algorithms.md
│   ├── 04-vanishing-exploding-gradients.md
│   ├── 05-generalization-regularization.md
│   ├── 06-bias-variance-double-descent.md
│   ├── 07-cross-entropy-kl.md
│   └── 08-perfect-answer-templates.md
├── 90-drills/
│   ├── flashcards.md
│   ├── error-clinic.md
│   └── answer-key.md
└── 91-extended-drills/
    ├── quiz.md
    └── answer-key.md
```

## 최종 도달 기준

다음 항목을 노트 없이 설명하고 계산할 수 있으면 마스터 단계입니다.

- $A=\begin{pmatrix}0&1\\1&0\end{pmatrix}$의 고유값과 고유벡터를 정의로 검증한다.
- $X\sim N(0,1)$의 $E[X]$, $E[X^2]$, $E[X^3]$, $E[X^4]$를 대칭성과 부분적분으로 구한다.
- $X\sim Uniform[a,b]$의 평균과 분산을 적분 정의로 구한다.
- Bernoulli MLE에서 i.i.d, 로그, 페르마 정리를 빠짐없이 말한다.
- symmetric/asymmetric prior의 MAP 극한을 데이터와 prior의 힘겨루기로 설명한다.
- tent prior 문제를 영역 분할, 경계, 미분 불가능점으로 처리한다.
- softmax 자코비안 $J=\operatorname{diag}(p)-pp^T$를 유도한다.
- Gaussian prior가 L2 정규화로 바뀌는 이유를 음의 로그 posterior로 설명한다.
- 활성화 함수의 미분과 vanishing gradient 원인을 설명한다.
- 역전파 공식 $\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\sigma'(z^{(l)})$를 유도한다.
- SGD, Momentum, RMSProp, Adam의 차이를 설명한다.
- Dropout과 BatchNorm의 수식과 역할을 설명한다.
- Bias-Variance 분해와 Double Descent를 설명한다.
- Cross Entropy 분해 $H(p,q)=H(p)+KL(p\|q)$를 증명한다.

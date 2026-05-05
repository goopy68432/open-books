---
title: "학습 지도: 문제 8개를 5개 엔진으로 줄이기"
slug: 00-orientation-00-learning-map
order: 1
---

# 학습 지도: 문제 8개를 5개 엔진으로 줄이기

원본 자료는 8개 기출문제와 사전지식, 킬러 체인, 증명 묶음으로 되어 있습니다. Codex 버전에서는 이를 5개의 문제 해결 엔진으로 줄입니다.

## 엔진 1: 선형변환 엔진

다루는 원본: `01-eigen`, `00-prerequisites/06-vector-matrix.md`, `00-prerequisites/07-determinant.md`, `10-ten-proofs/02`, `10-ten-proofs/09`

핵심 문장:

> 행렬은 공간을 움직이는 규칙이고, 고유벡터는 그 움직임 속에서도 방향이 보존되는 축이다.

루틴:

1. $Av=\lambda v$를 쓴다.
2. $(A-\lambda I)v=0$로 옮긴다.
3. $v\ne0$인 해가 있으려면 $\det(A-\lambda I)=0$이어야 한다.
4. $\lambda$를 구한 뒤 각 $\lambda$에 대해 $v$를 구한다.
5. 마지막에 반드시 $Av=\lambda v$를 직접 계산해 검증한다.

## 엔진 2: 적분과 모멘트 엔진

다루는 원본: `02-gaussian`, `03-uniform`, `00-prerequisites/05`, `00-prerequisites/10`, `10-ten-proofs/05`

핵심 문장:

> 분포의 평균과 분산은 "확률밀도로 가중한 넓이"다.

루틴:

1. pdf를 먼저 적는다.
2. $E[g(X)] = \int g(x)p(x)\,dx$를 쓴다.
3. 대칭이면 홀함수 적분을 0으로 처리한다.
4. 다항식이면 직접 적분하거나 부분적분을 쓴다.
5. 분산은 $\operatorname{Var}(X)=E[X^2]-E[X]^2$로 정리한다.

## 엔진 3: 우도 엔진

다루는 원본: `04-mle-bernoulli`, `09-killer-chains/01~05`, `10-ten-proofs/10`

핵심 문장:

> 데이터를 고정하고 모수를 움직이면 확률식은 우도함수가 된다.

루틴:

1. 단일 관측치의 pmf/pdf를 쓴다.
2. i.i.d라서 결합확률을 곱으로 만든다.
3. 관측된 데이터는 고정하고 모수만 변수로 둔다.
4. 로그를 취한다: 곱을 합으로, 최대점은 그대로, 수치적으로 안정.
5. 미분해서 0으로 둔다.
6. 2계 미분이나 오목성으로 최댓값을 확인한다.

## 엔진 4: MAP 엔진

다루는 원본: `05-map-symmetric`, `06-map-asymmetric`, `07-map-tent`, `10-ten-proofs/03`, `09-killer-chains/06`

핵심 문장:

> MAP은 데이터의 목소리인 likelihood와 사전 믿음인 prior의 합의점이다.

루틴:

1. posterior $\propto likelihood\times prior$를 쓴다.
2. 로그 posterior를 만든다.
3. 미분 가능한 구간은 미분=0으로 후보를 찾는다.
4. prior가 0인 구간은 후보에서 제거한다.
5. 미분 불가능점과 경계도 후보로 비교한다.
6. prior가 강해지는 극한에서는 데이터가 밀릴 수 있음을 설명한다.

## 엔진 5: Softmax-CE 엔진

다루는 원본: `08-softmax`, `09-killer-chains/05`, `10-ten-proofs/04`

핵심 문장:

> softmax는 점수를 확률로 바꾸고, cross entropy는 정답 확률을 키우도록 압력을 준다.

루틴:

1. $p_i=e^{z_i}/S$, $S=\sum_k e^{z_k}$를 쓴다.
2. $\partial p_i/\partial z_j$는 $i=j$와 $i\ne j$를 나눈다.
3. 두 결과를 $p_i(\delta_{ij}-p_j)$로 합친다.
4. 행렬로 $J=\operatorname{diag}(p)-pp^T$라 쓴다.
5. CE와 합치면 $\partial L/\partial z_j=p_j-y_j$가 됨을 보인다.

## 4단계 진행 순서

| 주제 | 입문 | 중급 | 고급 | 마스터 |
|---|---|---|---|---|
| 선형대수 | 벡터/행렬 읽기 | 고유값 계산 | 독립성/스펙트럴 정리 | 변형 행렬 방어 |
| 분포 | pdf/pmf 읽기 | 평균/분산 계산 | 가우스 적분 증명 | 일반 모멘트 설명 |
| MLE | 우도 뜻 | Bernoulli 체인 | 모든 분포 MLE 템플릿 | 새로운 분포에 적용 |
| MAP | prior/posterior 뜻 | Beta류 prior | 경계/비미분 처리 | prior 설계 문제 |
| Softmax | 확률화 직관 | 자코비안 계산 | CE 합성 그래디언트 | 행렬/기하 해석 |

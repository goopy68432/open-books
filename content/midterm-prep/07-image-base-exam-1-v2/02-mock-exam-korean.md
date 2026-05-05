---
title: "딥러닝 수학 기초 — 중간고사 모의시험 (한국어 버전)"
slug: mock-exam-korean
order: 2
---

# 딥러닝 수학 기초 — 중간고사 모의시험 (한국어 버전)

> **한양대학교, 2026년 봄학기**
> **담당교수**: 이성윤 (LRNING Lab)
> **범위**: 딥러닝을 위한 선형대수 + 미분/최적화
> **총 문항**: 30문제
> **언어**: 영어 (시험은 영어, 본 파일은 한국어 번역)
> **중요**: 모든 유도 과정을 보여라. 채점은 논리적 과정에 초점을 맞추며, 최종 답만으로는 점수를 받기 어렵다.

---

## 교수님 출제 성향 분석

이성윤 교수님의 학문적 배경 (서울대 수리과학과 박사):
- **연역 중심**: 정의/공리에서 출발하여 단계별로 유도
- **답보다 과정**: 산술 오류가 있더라도 올바른 추론에 부분 점수 부여
- **"어떻게"보다 "왜"**: 매 단계마다 수학적 근거로 정당화
- **퀴즈형**: 수업 퀴즈와 유사한 난이도 및 형식
- **증명 지향**: 엄밀한 수학적 논증을 기대

---

# 파트 1: 선형대수 기초 (Q1–Q10)

---

## 1번. 내적과 직교성 [5점]

$\mathbf{u} = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}$이고 $\mathbf{v} = \begin{pmatrix} 1 \\ 4 \\ k \end{pmatrix}$일 때:

(a) $\mathbf{u}$와 $\mathbf{v}$가 직교하도록 하는 $k$의 값을 구하라.

(b) (a)에서 구한 $k$에 대해, $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$ (직교 벡터에 대한 피타고라스 정리)를 검증하라.

---

## 2번. 열의 선형결합으로서의 행렬-벡터 곱셈 [10점]

$A = \begin{pmatrix} 1 & 3 \\ 2 & -1 \\ 0 & 4 \end{pmatrix}$이고 $\mathbf{x} = \begin{pmatrix} 2 \\ 5 \end{pmatrix}$일 때:

(a) $A\mathbf{x}$를 $A$의 **열들의 선형결합**으로 표현하여 계산하라. 각 단계를 명시적으로 보여라.

(b) 행렬-벡터 곱셈을 열의 선형결합으로 보는 것이 행-내적 관점보다 왜 더 통찰력 있는지, 특히 $A$의 **열공간(column space)** 이해와 관련하여 설명하라.

---

## 3번. 선형 독립과 스팬 [10점]

벡터 $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$, $\mathbf{v}_2 = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$, $\mathbf{v}_3 = \begin{pmatrix} 7 \\ 8 \\ 9 \end{pmatrix}$에 대해:

(a) $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$이 선형 독립인지 판별하라. 완전한 증명을 제시하라.

(b) $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$의 차원은 무엇인가? 이 스팬의 기저를 구하라.

(c) 이 세 벡터가 $\mathbb{R}^3$ 전체를 생성(span)할 수 있는가? 이유를 설명하라.

---

## 4번. 기저와 차원 [5점]

(a) 벡터 공간 $V$에 대한 **기저(basis)**의 정의를 서술하라.

(b) $\{v_1, v_2, \ldots, v_n\}$이 $V$의 기저이면, 모든 벡터 $v \in V$가 기저 벡터들의 선형결합으로 **유일하게** 표현됨을 증명하라.

---

## 5번. 커널, 이미지, 랭크-퇴화차수 정리 [10점]

선형변환 $T : \mathbb{R}^4 \to \mathbb{R}^3$이 다음 행렬로 표현된다:

$$A = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 1 & 2 & 1 & 3 \end{pmatrix}$$

(a) $\ker(T)$ ($A$의 영공간)를 구하라. 벡터들의 스팬으로 답을 표현하라.

(b) $\text{Im}(T)$ ($A$의 열공간)를 구하라. 이미지의 기저를 제시하라.

(c) 랭크-퇴화차수 정리를 검증하라: $\dim(\ker(T)) + \dim(\text{Im}(T)) = 4$.

---

## 6번. 행렬 연산의 계산 복잡도 [5점]

(a) 두 $n \times n$ 행렬의 나이브 행렬 곱셈이 $O(n^3)$ 연산을 필요로 하는 이유를 설명하라.

(b) $A^{-1}$ 계산도 $O(n^3)$인데, 실제로는 $x = A^{-1}b$를 계산하는 대신 $Ax = b$를 직접 풀기(예: LU 분해)를 선호하는 이유를 설명하라.

---

## 7번. 선형변환의 성질 [10점]

$T : \mathbb{R}^n \to \mathbb{R}^m$이 선형변환일 때:

(a) $T(\mathbf{0}) = \mathbf{0}$임을 증명하라.

(b) $T$가 임의의 선형결합을 이미지의 같은 선형결합으로 사상함을 증명하라:
$$T\left(\sum_{i=1}^k \alpha_i \mathbf{v}_i\right) = \sum_{i=1}^k \alpha_i \, T(\mathbf{v}_i)$$

(c) (b)를 이용하여, 선형변환이 기저에 대한 작용만으로 **완전히 결정**되는 이유를 설명하라.

---

## 8번. 행렬 곱셈 순서와 결합법칙 [5점]

$A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $\mathbf{x} \in \mathbb{R}^p$일 때:

(a) $m = 1000$, $n = 1000$, $p = 1$인 경우 $(AB)\mathbf{x}$와 $A(B\mathbf{x})$의 계산 비용을 비교하라.

(b) 딥러닝에서 $(W_2 W_1)\mathbf{x}$가 아닌 $W_2(W_1 \mathbf{x})$로 레이어별로 계산하는 이유를, 계산 비용 이외의 관점에서 설명하라.

---

# 파트 2: 고유값 및 행렬 분해 (Q9–Q16)

---

## 9번. 고유값 계산 [10점]

$A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$일 때:

(a) 특성방정식 $\det(A - \lambda I) = 0$을 풀어 $A$의 고유값을 구하라.

(b) 각 고유값에 대응하는 고유벡터를 구하라.

(c) 각 고유값-고유벡터 쌍에 대해 $A\mathbf{v} = \lambda \mathbf{v}$를 확인하여 답을 검증하라.

---

## 10번. 고유값의 성질과 변환 [10점]

$A$가 고유값 $\lambda_1 = 4$, $\lambda_2 = 1$, $\lambda_3 = -2$인 $3 \times 3$ 행렬일 때:

(a) $A^3$의 고유값을 구하라.
(b) $A - 3I$의 고유값을 구하라.
(c) $A^{-1}$의 고유값을 구하라 (존재 이유를 먼저 정당화하라).
(d) 고유값만을 이용하여 $\det(A)$와 $\text{tr}(A)$를 계산하라.

---

## 11번. 대각화 [10점]

$A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$일 때:

(a) $A$의 고유값과 고유벡터를 구하라.

(b) $D$가 대각행렬인 대각화 $A = PDP^{-1}$을 쓰라. $P$와 $D$를 명시적으로 밝혀라.

(c) 대각화를 이용하여 $A^{100}$을 계산하라. 대각화가 이 계산을 가능하게 하는 이유를 설명하라.

---

## 12번. 대칭행렬과 스펙트럼 정리 [5점]

(a) 실수 대칭행렬에 대한 스펙트럼 정리를 서술하라.

(b) 실수 대칭행렬의 고유값이 항상 실수인 이유를 설명하라. (힌트: $\bar{\mathbf{v}}^T A \mathbf{v}$를 이용하라.)

(c) 이 성질이 PCA와 같은 딥러닝 응용에서 왜 중요한지 설명하라.

---

## 13번. 특이값 분해 (SVD) 개념 [10점]

$A \in \mathbb{R}^{m \times n}$의 SVD가 $A = U\Sigma V^T$일 때:

(a) $U$, $\Sigma$, $V$가 각각 무엇을 나타내는지 설명하라. 차원과 주요 성질을 서술하라.

(b) $A$의 특이값과 $A^T A$의 고유값 사이의 관계를 설명하라.

(c) $A \in \mathbb{R}^{3 \times 2}$의 특이값이 $\sigma_1 = 5$, $\sigma_2 = 2$일 때, $A^T A$의 고유값은? $\text{rank}(A)$는?

---

## 14번. SVD를 이용한 저랭크 근사 [10점]

$A \in \mathbb{R}^{m \times n}$의 랭크가 $r$일 때 SVD는 $A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$이다.

(a) Eckart-Young 정리를 서술하라: 프로베니우스 노름에서 $A$의 최적 랭크-$k$ 근사는 무엇인가?

(b) $A$의 특이값이 $\sigma_1 = 10, \sigma_2 = 5, \sigma_3 = 1, \sigma_4 = 0.1$일 때, 랭크-2 근사가 포착하는 "에너지" ($\sum \sigma_i^2$로 측정)의 비율은?

(c) 이 원리가 딥러닝에서 **차원 축소** (PCA)에 어떻게 사용되는지 설명하라.

---

## 15번. 고유값 문제로서의 PageRank [5점]

PageRank 알고리즘은 웹 페이지를 방향 그래프로 모델링한다. 전이 행렬 $M$은 랭크 벡터 $\mathbf{r}$에 대해 $\mathbf{r} = M\mathbf{r}$을 만족한다.

(a) PageRank 벡터를 찾는 것이 $M$의 고유값 $\lambda = 1$에 대응하는 고유벡터를 찾는 것과 동치인 이유를 설명하라.

(b) 확률행렬에서 고유값 $\lambda = 1$이 **최대** 고유값임이 보장되는 이유는?

---

## 16번. 양반정치 행렬 [5점]

(a) 행렬 $A$가 양반정치(PSD)라는 것의 정의를 서술하라.

(b) 임의의 행렬 $B \in \mathbb{R}^{m \times n}$에 대해, 행렬 $B^T B$가 항상 PSD임을 증명하라.

(c) 이 사실이 데이터 과학에서 공분산 행렬 $X^T X$ 계산에 왜 중요한지 설명하라.

---

# 파트 3: 미분과 최적화 (Q17–Q24)

---

## 17번. 선형 근사와 뉴턴 방법 [10점]

(a) 미분 가능한 함수 $f$의 점 $a$에서의 선형 근사 공식을 서술하라:
$$f(x) \approx \; ?$$

(b) 선형 근사를 이용하여 $\sqrt{7}$을 추정하라. (힌트: $f(x) = \sqrt{x}$를 $a = 9$ 또는 $a = 4$에서 사용하라.)

(c) $x_0 = 3$에서 출발하여 $g(x) = x^2 - 7$의 근을 찾기 위해 뉴턴 방법을 1회 반복 적용하라. 공식과 계산을 보여라.

(d) 뉴턴 방법의 기하학적 해석을 설명하라.

---

## 18번. 그래디언트 (스칼라-벡터 미분) [10점]

$f : \mathbb{R}^3 \to \mathbb{R}$이 $f(\mathbf{x}) = x_1^2 + 3x_1 x_2 - x_3^2 + 2x_3$로 정의될 때:

(a) 그래디언트 $\nabla f(\mathbf{x})$를 계산하라.

(b) $\mathbf{x}_0 = (1, -1, 2)^T$에서 $\nabla f$를 평가하라.

(c) $\mathbf{x}_0$에서 $f$가 가장 빠르게 증가하는 방향은? 증가율은?

(d) 경사하강법이 $f$를 **최소화**하기 위해 $-\nabla f$ 방향으로 업데이트하는 이유를 설명하라.

---

## 19번. 자코비안 행렬 (벡터-벡터 미분) [10점]

$\mathbf{f} : \mathbb{R}^2 \to \mathbb{R}^3$이 다음과 같이 정의될 때:
$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} x_1^2 + x_2 \\ x_1 x_2 \\ e^{x_1} \end{pmatrix}$$

(a) 자코비안 행렬 $J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$를 계산하라. 차원을 서술하라.

(b) $\mathbf{x}_0 = (0, 1)^T$에서 자코비안을 평가하라.

(c) 선형화를 이용하여 $\mathbf{f}(0.1, 1.05)$를 근사하라:
$$\mathbf{f}(\mathbf{x}_0 + \Delta \mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J(\mathbf{x}_0) \Delta \mathbf{x}$$

(d) 역전파에서 합성 함수를 통한 그래디언트 계산에 자코비안이 왜 핵심적인지 설명하라.

---

## 20번. 벡터 함수의 연쇄 법칙 [10점]

$\mathbf{g} : \mathbb{R}^2 \to \mathbb{R}^3$과 $h : \mathbb{R}^3 \to \mathbb{R}$이 다음과 같이 정의될 때:
$$\mathbf{g}(\mathbf{x}) = \begin{pmatrix} x_1 + x_2 \\ x_1 - x_2 \\ x_1 x_2 \end{pmatrix}, \quad h(\mathbf{z}) = z_1^2 + z_2^2 + z_3$$

$f(\mathbf{x}) = h(\mathbf{g}(\mathbf{x}))$로 정의한다.

(a) $\nabla_\mathbf{z} h$ ($h$의 $\mathbf{z}$에 대한 그래디언트)를 계산하라.

(b) $\mathbf{g}$의 자코비안 $J_\mathbf{g}(\mathbf{x})$를 계산하라.

(c) 연쇄 법칙 $\nabla_\mathbf{x} f = J_\mathbf{g}^T \nabla_\mathbf{z} h$를 사용하여 $\nabla_\mathbf{x} f$를 계산하라.

(d) $f(\mathbf{x})$를 직접 계산한 후 $\nabla_\mathbf{x} f$를 구하여 검증하라.

---

## 21번. 벡터-스칼라 미분 [5점]

$\mathbf{y} = A\mathbf{x}$ ($A \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$)일 때:

(a) $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$를 계산하라.

(b) $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \mathbf{x}^T\mathbf{x}$일 때, $\frac{\partial f}{\partial \mathbf{x}} = 2\mathbf{x}$를 유도하라.

(c) $A$가 대칭일 때 $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$이면, $\frac{\partial f}{\partial \mathbf{x}} = 2A\mathbf{x}$를 유도하라.

---

## 22번. 소프트맥스 함수와 그 도함수 [10점]

소프트맥스 함수 $\sigma : \mathbb{R}^K \to \mathbb{R}^K$가 다음과 같이 정의된다:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K$$

(a) $\sum_{i=1}^K \sigma(\mathbf{z})_i = 1$이고 모든 $i$에 대해 $\sigma(\mathbf{z})_i > 0$임을 확인하라.

(b) $i = j$인 경우와 $i \neq j$인 경우에 대해 $\frac{\partial \sigma_i}{\partial z_j}$를 계산하라.

(c) 전체 자코비안 $\frac{\partial \boldsymbol{\sigma}}{\partial \mathbf{z}}$를 $\text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$을 이용하여 행렬 형태로 표현하라.

(d) 몫의 미분법(quotient rule)을 사용하여 (b)의 결과를 단계별로 유도하라.

---

## 23번. 활성화 함수와 그 도함수 [5점]

각 활성화 함수에 대해 도함수를 계산하고 장점 하나를 논하라:

(a) 시그모이드: $\sigma(x) = \frac{1}{1 + e^{-x}}$. $\sigma'(x) = \sigma(x)(1 - \sigma(x))$임을 보여라.

(b) ReLU: $f(x) = \max(0, x)$. 도함수를 쓰고 "dying ReLU" 문제를 설명하라.

(c) Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. $\tanh'(x) = 1 - \tanh^2(x)$임을 보여라.

---

## 24번. 어텐션 메커니즘: 수학적 공식화 [10점]

스케일드 닷-프로덕트 어텐션은 다음과 같이 정의된다:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

$Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$일 때:

(a) $QK^T$, $\text{softmax}(\cdot)$, 최종 출력의 차원을 서술하라.

(b) 각 원소가 평균 0, 분산 1인 i.i.d.일 때 내적 $\mathbf{q}^T \mathbf{k}$의 분산을 분석하여 스케일링 팩터 $\frac{1}{\sqrt{d_k}}$가 필요한 이유를 수학적으로 설명하라.

(c) 어텐션 메커니즘에서 softmax의 행별 연산이 기하학적/확률적으로 무엇을 달성하는지 해석하라.

---

# 파트 4: 딥러닝 연결과 응용 (Q25–Q30)

---

## 25번. 경사하강법 유도 [10점]

미분 가능한 함수 $f : \mathbb{R}^n \to \mathbb{R}$의 최소화를 고려한다.

(a) 선형 근사를 이용하여, 최급경사 하강 방향이 $-\nabla f(\mathbf{x})$임을 보여라.
(힌트: $f(\mathbf{x} + \eta \mathbf{d}) \approx f(\mathbf{x}) + \eta \nabla f(\mathbf{x})^T \mathbf{d}$를 고려하고 단위벡터 $\mathbf{d}$에 대해 최소화하라.)

(b) 경사하강법의 업데이트 규칙을 쓰고 학습률 $\eta$의 역할을 설명하라.

(c) $\eta$가 너무 크면? 너무 작으면? $f$의 곡률(헤시안)과 관련지어 설명하라.

---

## 26번. 고차원에서의 뉴턴 방법 [5점]

다변수 뉴턴 방법의 업데이트는:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [H(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

(a) $H(\mathbf{x})$ (헤시안)가 무엇을 나타내며 그래디언트의 자코비안과 어떤 관계인지 설명하라.

(b) 뉴턴 방법이 최솟값 근처에서 경사하강법보다 빠르게 수렴하는 이유는? 비용은?

---

## 27번. 교차 엔트로피 손실의 그래디언트 [10점]

참 라벨 $y$ (원-핫 벡터)와 예측 확률 $\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$에 대한 단일 샘플의 교차 엔트로피 손실은:
$$\mathcal{L} = -\sum_{i=1}^K y_i \log \hat{y}_i$$

(a) $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$임을 보여라.
(힌트: 연쇄 법칙 $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \cdot \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}}$와 Q22의 소프트맥스 자코비안을 사용하라.)

(b) 이 간결한 결과 ($\hat{\mathbf{y}} - \mathbf{y}$)가 역전파에서 왜 계산적으로 편리한지 설명하라.

---

## 28번. 데이터 압축을 위한 SVD [10점]

흑백 이미지가 행렬 $A \in \mathbb{R}^{1000 \times 800}$으로 저장된다.

(a) 원본 이미지를 저장하는 데 필요한 값의 수는?

(b) 랭크-$k$ SVD 근사 $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$를 사용하면 저장해야 하는 값의 수는? $k$로 표현하라.

(c) SVD 근사가 원본보다 적은 값을 사용하는 $k$의 값을 구하라. 압축률은?

(d) 압축과 복원 품질 사이의 trade-off를 논하라.

---

## 29번. 학습 동역학에서의 고유값 [5점]

단순 선형 모델 $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$에 대해 데이터 $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$에서의 MSE 손실을 고려한다. 경사하강법 업데이트에 데이터 행렬 $X$에 대한 행렬 $X^T X$가 관여한다.

(a) $X^T X$의 고유값이 경사하강법의 수렴 속도를 결정하는 이유를 설명하라.

(b) 최대 고유값이 $\lambda_{\max}$이고 최소 고유값이 $\lambda_{\min}$일 때 조건수 $\kappa$는? 큰 $\kappa$가 최적화를 어렵게 하는 이유는?

---

## 30번. 종합 유도: 선형대수에서 역전파까지 [15점]

다음의 간단한 2층 신경망을 고려한다:
$$\mathbf{h} = \sigma(W_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{y} = \mathbf{w}_2^T \mathbf{h} + b_2$$

여기서 $\sigma$는 원소별 활성화 함수, $W_1 \in \mathbb{R}^{d \times n}$, $\mathbf{b}_1 \in \mathbb{R}^d$, $\mathbf{w}_2 \in \mathbb{R}^d$, $b_2 \in \mathbb{R}$이다.

손실은 $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$이다.

(a) $\frac{\partial \mathcal{L}}{\partial b_2}$와 $\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2}$를 계산하라.

(b) $\frac{\partial \mathcal{L}}{\partial \mathbf{h}}$를 계산하라.

(c) 연쇄 법칙을 사용하여 $\frac{\partial \mathcal{L}}{\partial W_1}$과 $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1}$을 계산하라.
(힌트: $\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1$로 놓고 $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}$을 이용하라.)

(d) 각 단계에서 어떤 선형대수 개념(행렬 곱셈, 자코비안, 연쇄 법칙)이 나타나는지 식별하라. 역전파 알고리즘이 이러한 개념을 어떻게 체계적으로 적용하는지 설명하라.

---

**시험 종료**

*기억하세요: 여러분의 성적은 유도 과정의 명확성과 엄밀성에 달려 있습니다. 정당화 없는 정답은 최소한의 점수만 받습니다.*

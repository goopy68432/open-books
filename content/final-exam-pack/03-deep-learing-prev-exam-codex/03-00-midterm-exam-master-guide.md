---
title: "딥러닝 중간고사 마스터 가이드"
slug: 00-midterm-exam-master-guide
order: 3
---

# 딥러닝 중간고사 마스터 가이드

> 목적: 교재 기반으로 중간고사 범위를 하나도 빼지 않고, 시험 대비용으로 재구성한 단일 기준 문서  
> 범위: Week 1 ~ Week 7, 교재 기준 핵심 페이지 `pp. 033-292`  
> 작성 원칙: `교재 기반`, `시험형 재구성`, `중1 산수부터 단계별 설명`, `수식 기호 Line-by-Line 해체`, `백엔드/시스템 설계 관점 포함`

---

## 0. 시험 전체 지도

### 한 줄 결론
중간고사는 `선형대수 → 미적분/최적화 → 확률 → 베이즈 → 손실함수 → 역전파`를 하나의 파이프라인으로 이해했는지를 묻는다.

### 시험을 시스템 설계로 보면
- 입력 계층: 선형대수
  - 벡터, 행렬, 고유값, quadratic form이 데이터 구조와 연산 엔진 역할을 한다.
- 미들웨어 계층: 미적분/최적화
  - 미분, 그래디언트, 야코비안, 헤시안이 업데이트 규칙을 만든다.
- 추론 계층: 확률/베이즈
  - 분포, likelihood, posterior가 "데이터를 어떻게 설명할 것인가"를 정의한다.
- 학습 계층: 손실함수/역전파
  - MSE, NLL, KL, softmax, backprop가 실제 학습 루프를 구성한다.

### 교수 스타일 요약
- 답만 쓰면 감점이다.
- 반드시 `정의 → 가정 → 전개 → 결론` 순서를 지켜야 한다.
- `왜 로그를 취하는가`, `왜 미분해서 0으로 두는가`, `왜 i.i.d.가 필요한가`를 글로 써야 한다.

### 답안 작성 공통 템플릿
1. 정의를 먼저 쓴다.
2. 필요한 가정을 적는다.
3. 수식을 한 줄씩 전개한다.
4. 각 줄 사이의 이유를 한 문장으로 붙인다.
5. 최종 결론을 문제의 언어로 다시 번역한다.

### 범위 커버리지 매트릭스

| 주차 | 교재 페이지 | 핵심 개념 | 시험에서 해야 할 것 |
|---|---:|---|---|
| W1 | 033-040 | 전체 로드맵, 표기법, 전치 | 수학 도구들이 왜 필요한지 설명 |
| W2 | 041-072 | 행렬, 고유값, 대각화, PSD/PD, 거듭제곱법 | 고유값/고유벡터 계산, quadratic 해석 |
| W3 | 073-112 | 미분, 체인룰, 야코비안, 헤시안, 뉴턴법 | 행렬미분, softmax/역전파 계산 |
| W4 | 153-196 | 확률, 분포, 조건부확률, 기대값, 분산, 정규분포 | 기대값/분산 계산, 분포 비교 |
| W5 | 230-255 | 베이즈, MLE, MAP, 엔트로피, KL | posterior, MAP, KL 증명/계산 |
| W6 | 273-285 | 분류/회귀, 손실함수, MSE와 NLL | Gaussian NLL → MSE 유도 |
| W7 | 293-301 | 선형회귀, 정규방정식, 릿지회귀 | 정규방정식, L2 정규화, MAP 연결 |

### 범위 누락 방지 체크리스트
- [x] Uniform / discrete uniform / equally likely
- [x] Normal / Gaussian
- [x] Poisson
- [x] 조건부 확률 / 베이즈 정리
- [x] 기대값 / 분산 / 모멘트
- [x] MLE / MAP / prior
- [x] MSE / NLL / KL / CE 관계
- [x] KL 비음수 증명
- [x] softmax / Jacobian / chain rule / backprop
- [x] 고유값 / quadratic / learning rate 조건
- [x] average pooling의 행렬 표현

---

## 1. 선형대수 기반: 고유값, 대각화, quadratic form

> 교재 근거: [page_055]~[page_061], [page_066]~[page_072]

### 1-1. 고유값과 고유벡터

#### 한 줄 결론
고유값은 행렬이 특정 방향을 몇 배로 늘리는지 나타내는 순수 스케일 값이고, 고유벡터는 그 방향 자체다.

#### Goal
복잡하게 얽힌 행렬 연산에서, `방향은 안 바뀌고 크기만 바뀌는 축`을 찾아 시스템의 본질을 보는 것.

#### Tool / Algorithm
\[
Av = \lambda v,\qquad \det(A-\lambda I)=0
\]

#### 중1 설명
- `2배`, `3배` 같은 배율 개념부터 생각하면 된다.
- 어떤 입력은 시스템을 통과해도 방향이 그대로고, 크기만 4배가 될 수 있다.
- 그때 그 입력 비율이 고유벡터, 4라는 배율이 고유값이다.

#### 중2 설명
- 연립방정식을 풀 때 "특별한 해"를 찾는 것과 비슷하다.
- 일반 벡터는 방향도 바뀌지만, 특정 벡터는 방향 유지 + 크기 변화만 생긴다.

#### 중3 설명
- 일차함수 \(y=ax\)에서 \(a\)가 배율인 것처럼, 행렬은 다차원 배율 기계다.
- 그런데 모든 방향에서 같은 배율이 아니라, 특정 축마다 다른 배율을 갖는다.

#### 고1 설명
- 행렬은 좌표 변환이다.
- 고유벡터는 변환 후에도 같은 직선 위에 남는 벡터다.
- 고유값은 그 직선 위 스케일링 계수다.

#### 고2 설명
- \(Av=\lambda v\)를 \(Av-\lambda v=0\)으로 바꾸면
\[
(A-\lambda I)v=0
\]
- 영벡터가 아닌 \(v\neq 0\)가 있으려면 \(A-\lambda I\)가 가역이 아니어야 한다.
- 따라서
\[
\det(A-\lambda I)=0
\]
가 고유값 조건이다.

#### 고3 설명
- 고유값은 특성다항식의 근이다.
- 대칭 행렬이면 고유값이 실수이고, 서로 다른 고유값의 고유벡터는 직교한다.
- 이 성질 때문에 대각화와 quadratic form 해석이 쉬워진다.

#### 대학수학 연결
- Hessian의 고유값은 손실 곡률을 나타낸다.
- 최대 고유값 \(\lambda_{\max}\)는 learning rate 상한과 연결된다.
- 조건수 \(\kappa=\lambda_{\max}/\lambda_{\min}\)는 최적화 난이도를 결정한다.

#### 수식 기호 일일이 해체 (Line-by-Line 주석)

| 기호 | 이름 | 뜻 | 백엔드/시스템 비유 |
|---|---|---|---|
| \(A\) | 행렬 | 시스템 본체 | 서비스 간 트래픽 간섭 로직 |
| \(v\) | 벡터 | 입력 방향 | 요청 패턴 / 리소스 배합비 |
| \(\lambda\) | 고유값 | 순수 배율 | 엔진 증폭률 |
| \(I\) | 항등행렬 | 변화 없는 기준 | 원본 유지 스위치 |
| \(\det\) | 행렬식 | 뒤집을 수 있는지 판정 | 복구 키 존재 여부 |

#### 알고리즘 순서로 다시 설명
1. 시스템 행렬 \(A\)를 본다.
2. \(\det(A-\lambda I)=0\)으로 가능한 배율 \(\lambda\)를 찾는다.
3. 각 \(\lambda\)마다 \((A-\lambda I)v=0\)을 풀어 황금 입력 비율 \(v\)를 찾는다.
4. 그 축들로 시스템을 재표현하면 복잡한 상호작용이 분리된다.

#### 백엔드/시스템 설계 비유
- 고유값: 시스템의 "진짜 체력".
- 고유벡터: 그 체력을 끌어내는 입력 배합비.
- 대각화: 꼬여 있던 시스템을 `입력 필터 + 독립 엔진 + 출력 복원` 구조로 리팩터링하는 것.

#### 저장 효율 관점
- 원래 \(n\times n\) 행렬을 통째로 해석하는 대신, 고유값 \(n\)개와 고유벡터 집합으로 구조를 분해해 볼 수 있다.
- 대칭 행렬은 \(U^{-1}=U^T\)라서 추가 저장 부담이 줄어든다.

#### 연산 부하 관점
- 전체 고유분해는 보통 \(O(n^3)\).
- 최대 고유값만 필요하면 거듭제곱법으로 더 싸게 구할 수 있다.

#### 예시: 2대 서버 트래픽 간섭 시스템
\[
A=\begin{bmatrix}3&1\\2&2\end{bmatrix}
\]

##### Line-by-Line
\[
\det(A-\lambda I)=
\det\begin{bmatrix}3-\lambda&1\\2&2-\lambda\end{bmatrix}=0
\]
- \(A-\lambda I\): 시스템 본체에서 "배율 후보 \(\lambda\)"를 뺀 것
- \(\det=0\): 완전히 뒤집히지 않는 순간, 즉 특별한 방향이 생기는 순간

\[
(3-\lambda)(2-\lambda)-2=0
\]
- 2x2 행렬식 공식 \(ad-bc\)

\[
\lambda^2-5\lambda+4=0
\]

\[
(\lambda-4)(\lambda-1)=0
\]

\[
\lambda_1=4,\quad \lambda_2=1
\]

이제 각각에 대해 고유벡터를 구한다.

\[
(A-4I)v=0
\Rightarrow
\begin{bmatrix}-1&1\\2&-2\end{bmatrix}
\begin{bmatrix}v_1\\v_2\end{bmatrix}=0
\]

\[
-v_1+v_2=0\Rightarrow v_1=v_2
\Rightarrow v^{(1)}=\begin{bmatrix}1\\1\end{bmatrix}
\]

\[
(A-I)v=0
\Rightarrow
\begin{bmatrix}2&1\\2&1\end{bmatrix}
\begin{bmatrix}v_1\\v_2\end{bmatrix}=0
\]

\[
2v_1+v_2=0\Rightarrow v_2=-2v_1
\Rightarrow v^{(2)}=\begin{bmatrix}1\\-2\end{bmatrix}
\]

조립하면
\[
P=\begin{bmatrix}1&1\\1&-2\end{bmatrix},\qquad
D=\begin{bmatrix}4&0\\0&1\end{bmatrix}
\]

\[
A=PDP^{-1}
\]

#### 핵심:
- \(\det(A-\lambda I)=0\)는 고유값 찾기 시작 버튼이다.
- 고유벡터는 시스템이 가장 자연스럽게 반응하는 방향이다.

#### 주의:
- \(\det(A+\lambda I)=0\)가 아니다.
- 고유벡터는 0벡터가 될 수 없다.

#### 연구 포인트:
- Hessian eigenvalue와 learning rate 안정성
- Spectral normalization

#### 쉽게 설명하면
행렬이라는 꼬인 시스템에서, 안 꼬이는 특별한 방향과 그 방향의 배율을 찾아내는 작업이다.

#### 남에게 설명하는 한 문장
"고유값은 시스템의 순수 증폭률이고, 고유벡터는 그 증폭률이 그대로 드러나는 입력 비율이다."

#### 핵심 킬러 요약
- 고유값 \(\lambda\): 시스템의 순수 마력
- 고유벡터 \(v\): 마력을 끌어내는 입력 비율
- 대각화 \(PDP^{-1}\): 꼬인 시스템을 독립 엔진으로 분해하는 아키텍처 리팩터링

---

### 1-2. Quadratic Form과 학습률 조건의 밑바탕

> 교재 근거: [page_056], [page_060], [page_061]

#### 한 줄 결론
\[
f(w)=\frac12 w^TAw
\]
는 "방향마다 다른 가파르기"를 가진 그릇이고, 그 가파르기를 고유값이 결정한다.

#### Goal
왜 gradient descent의 안정성이 \(\lambda_{\max}(A)\)와 연결되는지 이해하는 것.

#### Tool / Algorithm
\[
f(w)=\frac12 w^TAw,\qquad \nabla f(w)=Aw \quad (A=A^T)
\]

#### 중1 설명
- \(x^2\)는 위로 열린 그릇이다.
- 값이 커질수록 더 가파르게 올라간다.
- 여러 변수 버전이 quadratic form이다.

#### 중2 설명
- 한 변수에서는 \(ax^2\)의 \(a\)가 가파름을 정한다.
- 여러 변수에서는 행렬 \(A\)가 축마다 가파름을 정한다.

#### 중3 설명
- 원이 아니라 타원처럼 생긴 그릇이 된다.
- 어떤 방향은 급하고, 어떤 방향은 완만하다.

#### 고1 설명
- \(A\)를 대각화하면
\[
A=U\Lambda U^T
\]
- 좌표변환 \(y=U^Tw\) 후
\[
f(w)=\frac12 \sum_i \lambda_i y_i^2
\]
- 각 \(\lambda_i\)가 축별 곡률이다.

#### 고2 설명
- \(\lambda_i\)가 크면 그 방향은 매우 가파르다.
- \(\lambda_i\)가 작으면 그 방향은 평평하다.
- 따라서 가장 큰 고유값이 learning rate를 제한한다.

#### 고3 설명
- PSD면 \(\lambda_i\ge 0\), 따라서 볼록.
- PD면 \(\lambda_i>0\), 따라서 유일한 최소점.

#### 대학수학 연결
- Hessian의 최대 고유값은 local Lipschitz constant와 연결된다.
- Edge of Stability 현상도 여기서 나온다.

#### 수식 기호 일일이 해체

| 기호 | 이름 | 뜻 | 시스템 비유 |
|---|---|---|---|
| \(w\) | 파라미터 벡터 | 현재 모델 상태 | 서버 설정값 |
| \(A\) | PSD/PD 행렬 | 곡률 행렬 | 방향별 민감도 맵 |
| \(w^TAw\) | 이차형식 | 축별 가중합 | 방향별 비용 집계 |
| \(\lambda_{\max}\) | 최대 고유값 | 가장 가파른 축 | 가장 민감한 병목 축 |

#### 핵심:
- quadratic form은 행렬판 \(ax^2\)다.
- 고유값이 그릇의 가파름을 정한다.

#### 주의:
- \(A\)가 비대칭이면 gradient 식을 조심해야 한다.
- 시험에서는 대칭/PSD 조건을 먼저 적어야 한다.

---

## 2. 미적분과 최적화: 적분, 미분, 체인룰, GD

> 교재 근거: [page_089]~[page_097], [page_103]

### 2-1. 적분의 본질

#### 한 줄 결론
적분은 연속 데이터를 아주 잘게 쪼개서 전수조사 합산하는 reduce 연산이다.

#### Goal
기댓값, 확률밀도, 면적, 평균을 계산할 때 왜 적분이 필요한지 이해하는 것.

#### Tool / Algorithm
\[
\int_a^b f(x)\,dx
\]

#### 중1 설명
- 숫자 리스트의 합은 `for`문으로 더하면 된다.
- 적분은 리스트가 아니라 연속 구간 전체를 훑어 더하는 것이다.

#### 중2 설명
- 종이 아래 면적을 직사각형으로 잘게 잘라 더한다고 생각하면 된다.
- 잘게 자를수록 정확해진다.

#### 중3 설명
- 구간 \([a,b]\)를 매우 작은 간격으로 나누고,
- 각 칸의 높이 \(f(x)\)에 폭 \(dx\)를 곱해 더하는 것이다.

#### 고1 설명
- 정적분은 리만합의 극한이다.
- 확률에서는 전체 면적이 1이 되도록 정규화된 함수와 자주 연결된다.

#### 수식 기호 일일이 해체 (예시)
\[
E[X]=\int_a^b x\cdot \frac{1}{b-a}\,dx
\]

| 기호 | 이름 | 뜻 | 백엔드/시스템 비유 |
|---|---|---|---|
| \(\int\) | 적분 | 전체 구간 누적 합산 | `for` 루프 시작 |
| \(a\) | 아래끝 | 시작점 | 시작 인덱스 |
| \(b\) | 위끝 | 끝점 | 종료 인덱스 |
| \(x\) | 변수 | 현재 처리 중 값 | 현재 레코드 |
| \(\frac1{b-a}\) | 확률밀도 | 각 값의 가중치 | weight |
| \(dx\) | 미소 폭 | 매우 작은 칸 | micro-batch step |

#### 알고리즘 순서로 다시 설명
1. 구간 \([a,b]\)를 훑는다.
2. 각 위치 \(x\)마다 값 \(x\)를 읽는다.
3. 가중치 \(\frac1{b-a}\)를 곱한다.
4. 작은 폭 \(dx\)만큼 잘라 계속 더한다.

#### 핵심:
- 적분 = 연속 영역 reduce
- \(f(x)dx\) = "값 × 아주 작은 폭"

#### 주의:
- 적분 변수와 적분 구간을 빼먹지 말 것
- 연속분포에서 "합" 대신 "적분"을 써야 한다

---

### 2-2. 미분과 gradient

#### 한 줄 결론
미분은 입력을 아주 조금 바꿨을 때 출력이 얼마나 민감하게 변하는지 측정하는 로컬 변화율이다.

#### Goal
학습에서 "어느 방향으로 파라미터를 움직일지" 계산하는 것.

#### Tool / Algorithm
\[
f'(x),\qquad \nabla f(x)
\]

#### 중1 설명
- \(y=2x\)면 \(x\)를 1 늘릴 때 \(y\)는 2 늘어난다.
- 이 2가 변화율 감각이다.

#### 중2 설명
- 직선 기울기의 개념이 미분의 시작이다.

#### 중3 설명
- 곡선에서는 한 점에서의 순간 기울기를 본다.

#### 고1 설명
\[
\frac{f(x+h)-f(x)}{h}
\]
- 이것이 평균 변화율
- \(h\to 0\)으로 보내면 순간 변화율

#### 고2 설명
- 여러 변수에서는 각 방향의 편미분을 묶어 gradient를 만든다.
\[
\nabla f(x)=
\begin{bmatrix}
\partial f/\partial x_1\\
\partial f/\partial x_2\\
\vdots
\end{bmatrix}
\]

#### 고3 설명
- gradient는 함수가 가장 빨리 증가하는 방향이다.
- 따라서 경사하강법은 \(-\nabla f\) 방향으로 이동한다.

#### 백엔드/시스템 비유
- gradient는 "어느 설정값이 장애율을 가장 많이 악화시키는가"를 알려주는 민감도 벡터다.

#### 핵심:
- gradient = 최급상승 방향
- 학습은 그 반대 방향으로 이동

---

### 2-3. 체인룰과 역전파

#### 한 줄 결론
체인룰은 여러 레이어를 통과한 변화량을 곱셈 체인으로 추적하는 규칙이고, 역전파는 이 규칙을 뒤에서 앞으로 적용한 것이다.

#### Goal
복합 함수에서 최종 loss가 각 파라미터에 얼마나 책임이 있는지 계산하는 것.

#### Tool / Algorithm
\[
\frac{dL}{dx}=\frac{dL}{dz}\frac{dz}{dx}
\]

#### 중1 설명
- 수도꼭지 A가 물 양을 바꾸고, 그 물 양이 다시 탱크 수위를 바꾸면,
- 최종 수위 변화는 중간 단계들을 따라가야 계산된다.

#### 고1 설명
- \(L(z(x))\)에서 바깥 함수 미분 × 안쪽 함수 미분

#### 고2 설명
- 벡터 함수에서는 Jacobian이 등장한다.
\[
J_{f}(x)=\frac{\partial f}{\partial x}
\]

#### 고3 설명
- 역전파는 Jacobian transpose를 이용한 효율적 VJP(vector-Jacobian product) 계산이다.

#### 핵심:
- 체인룰은 역전파의 수학적 본체다.
- 순전파에서 저장한 중간값을 역전파에서 재사용한다.

#### 주의:
- 순서를 바꾸면 안 된다.
- 차원 일치를 항상 확인해야 한다.

---

## 3. 확률과 분포

> 교재 근거: [page_153], [page_156], [page_160], [page_173]~[page_180], [page_196]

### 3-1. Uniform Distribution

#### 한 줄 결론
균등분포는 가능한 값들이 모두 똑같은 확률/밀도를 가지는 가장 단순한 분포다.

#### Goal
기댓값과 분산 계산의 가장 기본 템플릿을 익히는 것.

#### Tool / Algorithm
- 이산 균등분포: 모든 결과가 같은 확률
- 연속 균등분포:
\[
f_X(x)=
\begin{cases}
\frac1{b-a}, & a\le x\le b\\
0, & \text{otherwise}
\end{cases}
\]

#### 중1 설명
- 시험 번호표가 1부터 10까지 있고 아무 번호나 똑같이 뽑히면 균등분포다.

#### 중2 설명
- 연속 버전에서는 선분 전체에 같은 "밀도"가 깔려 있다고 생각하면 된다.

#### 중3 설명
- 확률 그 자체가 아니라 "밀도"가 일정하다.
- 구간 길이가 길수록 확률이 커진다.

#### 고1 설명
- 전체 면적이 1이어야 하므로 높이는 \(\frac1{b-a}\)다.

#### 수식 기호 일일이 해체
\[
E[X]=\int_a^b x\cdot \frac1{b-a}\,dx
\]

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(E[X]\) | 기대값 | 평균적인 위치 |
| \(x\) | 현재 값 | 지금 읽는 데이터 |
| \(\frac1{b-a}\) | 밀도 | 모든 위치의 동일 가중치 |
| \(dx\) | 미소 폭 | 아주 잘게 나눈 구간 |

#### Line-by-Line
\[
E[X]=\int_a^b x\cdot \frac1{b-a}\,dx
\]
- 평균 정의: 값 × 그 값의 가중치 를 전체 범위에서 더함

\[
=\frac1{b-a}\int_a^b x\,dx
\]
- 상수 \(\frac1{b-a}\)를 적분 밖으로 뺌

\[
=\frac1{b-a}\left[\frac{x^2}{2}\right]_a^b
\]
- \(\int x\,dx=\frac{x^2}{2}\)

\[
=\frac1{b-a}\cdot \frac{b^2-a^2}{2}
\]

\[
=\frac1{b-a}\cdot \frac{(b-a)(a+b)}{2}
\]
- \(b^2-a^2=(b-a)(a+b)\)

\[
=\frac{a+b}{2}
\]

#### 알고리즘 순서로 다시 설명
1. 전체 구간을 훑는다.
2. 현재 위치 \(x\)를 읽는다.
3. 균등 가중치 \(\frac1{b-a}\)를 곱한다.
4. 다 더하면 중앙값 \((a+b)/2\)가 나온다.

#### 저장 효율 관점
- 균등분포는 사실상 \(a,b\) 두 숫자만 저장하면 된다.

#### 연산 부하 관점
- 평균/분산이 닫힌형으로 바로 나온다.
- 가장 계산이 싼 분포 중 하나다.

#### 핵심:
- 균등분포 평균은 중앙 \((a+b)/2\)
- 분산은 \((b-a)^2/12\)

#### 주의:
- 연속 균등분포에서 특정 한 점의 확률은 0이다.
- 밀도와 확률을 혼동하지 말 것.

#### 쉽게 설명하면
처음부터 끝까지 모든 위치가 똑같이 중요해서, 평균이 딱 가운데가 된다.

#### 남에게 설명하는 한 문장
"균등분포는 모든 값이 같은 무게를 가지는 분포라서 평균이 정확히 중앙으로 간다."

#### 핵심 킬러 요약
- \(\int_a^b\): \(a\)부터 \(b\)까지 반복문
- \(x\cdot f(x)\): 값 × 가중치
- \(dx\): 초미세 step

---

### 3-2. Normal Distribution

#### 한 줄 결론
정규분포는 평균 근처에 데이터가 가장 몰리고 멀어질수록 확률밀도가 빠르게 줄어드는 종 모양 분포다.

#### Goal
가우시안이 왜 회귀, 노이즈 모델링, MSE 유도의 핵심이 되는지 이해하는 것.

#### Tool / Algorithm
\[
\mathcal N(x;\mu,\sigma^2)=
\frac1{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac12\left(\frac{x-\mu}{\sigma}\right)^2
\right)
\]

#### 중1 설명
- 평균 점수 근처 학생이 많고, 극단 점수 학생은 적은 시험 분포를 생각하면 된다.

#### 중2 설명
- 가운데가 가장 높고, 양옆으로 갈수록 낮아지는 종 모양이다.

#### 중3 설명
- \(\mu\)는 중심
- \(\sigma\)는 퍼짐 정도

#### 고1 설명
- \((x-\mu)\)는 중심에서 얼마나 떨어졌는지
- \(\sigma\)로 나누면 "표준화된 거리"

#### 고2 설명
- 제곱이 들어가므로 멀리 떨어진 값은 패널티가 급격히 커진다.
- 이 구조 때문에 MSE와 연결된다.

#### 고3 설명
- \(\log p(x)\)를 취하면 이차식이 된다.
- 그래서 Gaussian NLL이 squared error로 바뀐다.

#### 수식 기호 일일이 해체

| 기호 | 이름 | 뜻 | 시스템 비유 |
|---|---|---|---|
| \(\mu\) | 평균 | 중심 | 정상 부하 기준치 |
| \(\sigma^2\) | 분산 | 퍼짐 | 부하 변동 폭 |
| \(\exp\) | 지수함수 | 큰 오차에 빠른 패널티 | 급격한 에러 비용 증가 |
| \(\sqrt{2\pi\sigma^2}\) | 정규화 상수 | 면적 1 맞춤 | 총 예산 100% 정규화 |

#### 핵심:
- 평균 \(\mu\): 중심
- 분산 \(\sigma^2\): 흩어짐
- 로그를 취하면 이차식

#### 주의:
- \(\sigma\)와 \(\sigma^2\)를 혼동하지 말 것
- 표준정규는 \(\mu=0,\sigma^2=1\)

---

### 3-3. Poisson Distribution

> 교재 보조 근거: 중간 정리 자료와 시험 예상 범위 반영

#### 한 줄 결론
Poisson 분포는 "고정 시간/공간 안에서 몇 번 사건이 일어나는가"를 모델링하는 카운트 분포다.

#### Goal
연속값이 아니라 이벤트 횟수 데이터를 확률적으로 다루는 것.

#### Tool / Algorithm
\[
P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!},\qquad k=0,1,2,\dots
\]

#### 중1 설명
- 1시간 동안 고객센터 전화가 몇 번 오느냐를 세는 분포다.

#### 중2 설명
- 결과는 \(0,1,2,3,\dots\) 같은 개수다.

#### 중3 설명
- \(\lambda\)는 평균 발생 횟수다.

#### 고1 설명
- 평균과 분산이 둘 다 \(\lambda\)다.

#### 백엔드 비유
- 초당 에러 로그 개수, 분당 주문 요청 수, 초당 패킷 도착 수를 모델링할 때 쓴다.

#### 핵심:
- count 데이터
- 평균 = 분산 = \(\lambda\)

---

### 3-4. 기대값, 모멘트, 분산

#### 한 줄 결론
기대값은 평균 위치, 분산은 퍼짐, 모멘트는 분포의 모양 정보를 담는 요약 통계다.

#### Goal
시험에서 \(E[X]\), \(E[X^2]\), \(E[X^{2n}]\), \(E[X^{2n-1}]\)를 계산하거나 성질을 설명하는 것.

#### Tool / Algorithm
\[
E[X]=\sum_x x p(x)\quad \text{또는}\quad E[X]=\int x f(x)\,dx
\]
\[
\operatorname{Var}(X)=E[X^2]-(E[X])^2
\]

#### 표준정규 핵심 결과
- \(E[Z]=0\)
- \(E[Z^{2n-1}]=0\)  (대칭성 때문에 홀수차 모멘트는 0)
- \(E[Z^2]=1\)
- \(E[Z^4]=3\)

#### 핵심:
- 홀수차 모멘트 0은 "좌우 대칭" 때문
- 분산은 \(E[X^2]\)와 평균만 알면 계산 가능

#### 주의:
- \(E[X^2]\neq (E[X])^2\)

---

## 4. 베이즈, MLE, MAP

> 교재 근거: [page_160], [page_230], [page_233]~[page_245]

### 4-1. 조건부 확률과 베이즈 정리

#### 한 줄 결론
베이즈 정리는 "새 데이터가 들어왔을 때 기존 믿음을 업데이트하는 공식"이다.

#### Tool / Algorithm
\[
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
\]

#### 중1 설명
- "B가 일어났다고 이미 알고 있을 때, A일 가능성"을 다시 계산하는 공식이다.

#### 기호 해체

| 기호 | 뜻 | 시스템 비유 |
|---|---|---|
| \(P(A)\) | prior | 기존 운영 정책 |
| \(P(B|A)\) | likelihood | A일 때 이런 로그가 뜰 확률 |
| \(P(B)\) | evidence | 전체 정규화 비용 |
| \(P(A|B)\) | posterior | 로그를 본 뒤 업데이트된 믿음 |

#### 핵심:
- posterior = likelihood × prior / evidence

---

### 4-2. MLE

#### 한 줄 결론
MLE는 데이터를 가장 그럴듯하게 만들어주는 파라미터를 찾는 방법이다.

#### Goal
데이터만 보고 파라미터를 추정하는 것.

#### Tool / Algorithm
\[
\hat\theta_{\text{MLE}}=\arg\max_\theta P(D|\theta)
\]

#### 중1 설명
- "이 데이터가 가장 잘 나오는 설정값이 뭐냐"를 찾는 문제다.

#### 중2 설명
- 여러 데이터가 있으면 전부 잘 설명하는 \(\theta\)를 찾는다.

#### 고1 설명
- i.i.d.면 결합확률을 곱으로 분해한다.
\[
P(D|\theta)=\prod_{i=1}^n p(x_i|\theta)
\]

#### 고2 설명
- 로그를 취하면 곱이 합으로 바뀐다.
\[
\ell(\theta)=\log P(D|\theta)=\sum_{i=1}^n\log p(x_i|\theta)
\]

#### 고3 설명
- 미분해서 0으로 두어 극값 후보를 찾는다.
- 이후 boundary도 체크한다.

#### Bernoulli MLE Line-by-Line
\[
p(x_i|\theta)=\theta^{x_i}(1-\theta)^{1-x_i}
\]

\[
P(D|\theta)=\prod_{i=1}^n \theta^{x_i}(1-\theta)^{1-x_i}
=\theta^S(1-\theta)^{n-S}
\]
- \(S=\sum x_i\): 성공 개수

\[
\ell(\theta)=S\log\theta+(n-S)\log(1-\theta)
\]

\[
\frac{d\ell}{d\theta}=\frac{S}{\theta}-\frac{n-S}{1-\theta}=0
\]

\[
S(1-\theta)=(n-S)\theta
\]

\[
S=n\theta
\Rightarrow
\hat\theta_{\text{MLE}}=\frac{S}{n}
\]

#### 왜 로그를 쓰는가
1. 곱을 합으로 바꾸기 위해
2. 언더플로우를 막기 위해
3. 미분을 쉽게 하기 위해

#### 핵심:
- MLE = likelihood 최대화
- i.i.d. → 곱 → 로그 → 미분

#### 주의:
- interior optimum만 보고 끝내면 안 된다.
- \(\theta\in[0,1]\) 경계 확인 필요

---

### 4-3. MAP

#### 한 줄 결론
MAP는 MLE에 prior를 추가해서 "데이터 + 사전지식"으로 추정하는 방법이다.

#### Goal
데이터가 적을 때 과하게 흔들리지 않게 사전지식을 반영하는 것.

#### Tool / Algorithm
\[
\hat\theta_{\text{MAP}}=\arg\max_\theta P(\theta|D)
=\arg\max_\theta P(D|\theta)P(\theta)
\]

#### 핵심 공식
\[
\log p(\theta|D)=\log p(D|\theta)+\log p(\theta)+C
\]

#### prior 예시 1
\[
p(\theta)\propto \theta^m
\]

#### prior 예시 2
\[
p(\theta)\propto \theta^m(1-\theta)^m
\]

#### Line-by-Line: 두 번째 prior 사용
\[
\log p(\theta|D)
=S\log\theta+(n-S)\log(1-\theta)
+m\log\theta+m\log(1-\theta)+C
\]

\[
=(S+m)\log\theta+(n-S+m)\log(1-\theta)+C
\]

\[
\frac{d}{d\theta}\log p(\theta|D)
=\frac{S+m}{\theta}-\frac{n-S+m}{1-\theta}=0
\]

\[
(S+m)(1-\theta)=(n-S+m)\theta
\]

\[
S+m=(n+2m)\theta
\]

\[
\hat\theta_{\text{MAP}}=\frac{S+m}{n+2m}
\]

#### 백엔드/시스템 비유
- MLE: 실시간 로그만 보고 운영 정책 결정
- MAP: 과거 운영 경험도 함께 반영

#### 저장 효율 관점
- prior는 파라미터 몇 개만 추가 저장하면 된다.
- 작은 저장 비용으로 추정 안정성을 얻는다.

#### 연산 부하 관점
- 보통 log-prior 항 하나 추가되는 정도라 계산 증가는 작다.

#### 핵심:
- MAP = MLE + 정규화
- 균등 prior면 MAP = MLE

#### 주의:
- MAP는 전체 posterior 분포를 구하는 베이지안 추론 전체와 다르다.

#### 쉽게 설명하면
데이터만 믿는 게 MLE, 데이터와 상식 둘 다 믿는 게 MAP다.

#### 남에게 설명하는 한 문장
"MAP는 MLE에 prior를 더해 데이터가 적어도 덜 극단적인 답을 주게 만든다."

---

## 5. 손실함수 대통일: MSE, NLL, CE, KL

> 교재 근거: [page_163], [page_254], [page_255], [page_285]

### 5-1. Gaussian NLL → MSE

#### 한 줄 결론
가우시안 노이즈를 가정하면 NLL 최소화는 MSE 최소화와 완전히 같은 문제다.

#### Goal
왜 회귀에서 MSE를 쓰는지 확률론으로 설명하는 것.

#### Tool / Algorithm
\[
y_i=h(x_i)+\varepsilon_i,\qquad \varepsilon_i\sim\mathcal N(0,\sigma^2)
\]

\[
p(y_i|x_i,h)=\mathcal N(y_i;h(x_i),\sigma^2)
\]

#### Line-by-Line
\[
\text{NLL}(h)=-\log P(E|H)
\]
- NLL: 데이터 \(E\)를 가설 \(H\)가 얼마나 못 설명하는지 보는 비용

\[
=\sum_{i=1}^n -\log p((x_i,y_i)|H)
\]
- i.i.d.라서 곱이 합으로 바뀜

\[
=\sum_{i=1}^n -\log p(y_i|x_i,H)+C
\]
- \(x_i\)는 주어진 입력으로 보고 상수 처리

\[
=\sum_{i=1}^n -\log \mathcal N(y_i;h(x_i),\sigma^2)+C
\]

\[
=\sum_{i=1}^n \frac1{2\sigma^2}(y_i-h(x_i))^2 + C
\]
- Gaussian 로그를 열면 제곱오차가 남음

\[
=\frac{n}{2\sigma^2}\cdot \text{MSE}(h)+C
\]

따라서
\[
\arg\min_h \text{NLL}(h)=\arg\min_h \text{MSE}(h)
\]

#### 중1 설명
- 예측이 실제값에서 멀수록 벌점을 더 크게 준다.

#### 중2 설명
- 오차를 제곱하는 이유는 음수/양수가 섞여 사라지지 않게 하고, 큰 실수를 더 강하게 벌주기 위해서다.

#### 고1 설명
- Gaussian 로그를 취하면 지수 안의 제곱항이 밖으로 나온다.

#### 핵심:
- 최소제곱법 = MSE = Gaussian NLL = MLE

#### 주의:
- \(\frac{n}{2\sigma^2}\)는 상수라서 argmin에 영향이 없다.

---

### 5-2. Bernoulli NLL → BCE

#### 한 줄 결론
이진 분류에서 Bernoulli를 가정하면 NLL이 binary cross-entropy가 된다.

#### Tool / Algorithm
\[
p(y_i|x_i;w)=\hat y_i^{y_i}(1-\hat y_i)^{1-y_i}
\]

#### Line-by-Line
\[
\ell(w)=\sum_{i=1}^n \left[
y_i\log \hat y_i + (1-y_i)\log(1-\hat y_i)
\right]
\]

\[
\text{NLL}(w)=
-\sum_{i=1}^n \left[
y_i\log \hat y_i + (1-y_i)\log(1-\hat y_i)
\right]
\]

이것이 BCE다.

#### 핵심:
- Gaussian → MSE
- Bernoulli → BCE

---

### 5-3. KL, CE, NLL 관계

#### 한 줄 결론
NLL 최소화는 결국 경험분포와 모델분포 사이의 KL divergence를 최소화하는 것이다.

#### Tool / Algorithm
\[
H(p)=-\sum_x p(x)\log p(x)
\]
\[
CE(p,q)=-\sum_x p(x)\log q(x)
\]
\[
KL(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
\]

#### Line-by-Line
\[
KL(p\|q)=\sum_x p(x)\log\frac{p(x)}{q(x)}
\]

\[
=\sum_x p(x)(\log p(x)-\log q(x))
\]

\[
=\sum_x p(x)\log p(x)-\sum_x p(x)\log q(x)
\]

\[
=-H(p)+CE(p,q)
\]

따라서
\[
CE(p,q)=KL(p\|q)+H(p)
\]

#### 왜 CE 최소화 = KL 최소화인가
- \(p\)가 고정되면 \(H(p)\)는 상수
- 따라서 CE를 줄이는 것은 KL을 줄이는 것과 같다

#### NLL와 연결
\[
\text{NLL}(\theta)=
-\frac1n\sum_{i=1}^n \log q_\theta(y_i|x_i)
\]
- 이것은 경험분포 \(p_E\)에 대한 \(CE(p_E,q_\theta)\)와 같다.

#### 핵심:
- \(CE = KL + H\)
- \(H(p)\)는 데이터가 고정되면 상수
- 그래서 NLL 최소화 = KL 최소화

#### 주의:
- KL은 비대칭이다.
\[
KL(p\|q)\neq KL(q\|p)
\]

---

## 6. KL Divergence

> 교재 근거: [page_254], [page_255]

### 6-1. KL 비음수 증명

#### 한 줄 결론
KL divergence는 항상 0 이상이며, 두 분포가 완전히 같을 때만 0이다.

#### Goal
KL이 왜 "거리 비슷한 척도"로 쓰이는지 정당화하는 것.

#### Tool / Algorithm
- Gibbs inequality
- Jensen inequality

#### 시험용 핵심 결론
\[
KL(p\|q)\ge 0
\]

#### 쉽게 설명하면
- 잘못된 모델 \(q\)로 데이터를 코딩하면, 참 분포 \(p\)로 코딩하는 것보다 절대 더 싸질 수는 없다.

#### 핵심:
- KL은 항상 비음수
- 0이면 \(p=q\)

---

### 6-2. Gaussian KL

#### 한 줄 결론
분산이 같으면 가우시안 KL은 평균 차이의 제곱, 즉 MSE 모양으로 단순해진다.

#### Tool / Algorithm
\[
P=\mathcal N(\mu_1,\sigma^2),\qquad Q=\mathcal N(\mu_2,\sigma^2)
\]

\[
KL(P\|Q)=\frac{(\mu_1-\mu_2)^2}{2\sigma^2}
\]

#### 핵심:
- 평균이 멀수록 KL이 커진다.
- 동일 분산이면 MSE와 같은 형태다.

#### 주의:
- 분산이 다르면 공식이 더 복잡해진다.

---

## 7. 행렬미분과 Backpropagation

> 교재 근거: [page_041], [page_094], [page_097], [page_103]

### 7-1. \(-\log \sigma(Ax+b)\) 미분

#### 한 줄 결론
\[
L=-\log\sigma(z),\quad z=Ax+b
\]
에서는 결국 \(1-\sigma(z)\)와 입력 \(x\)가 gradient를 결정한다.

#### Goal
단일 뉴런의 loss가 가중치, 입력, 편향에 어떻게 전파되는지 계산하는 것.

#### Tool / Algorithm
\[
\sigma(z)=\frac1{1+e^{-z}}
\]

#### Line-by-Line
\[
L=-\log \sigma(z)
\]

\[
\frac{dL}{dz}
=-\frac{1}{\sigma(z)}\cdot \sigma'(z)
\]

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

\[
\frac{dL}{dz}
=-\frac{1}{\sigma(z)}\cdot \sigma(z)(1-\sigma(z))
=-(1-\sigma(z))
=\sigma(z)-1
\]

이제 chain rule 적용:

\[
\frac{\partial L}{\partial A}
=\frac{dL}{dz}\cdot \frac{\partial z}{\partial A}
=(\sigma(z)-1)x^T
\]

\[
\frac{\partial L}{\partial x}
=A^T(\sigma(z)-1)
\]

\[
\frac{\partial L}{\partial b}
=\sigma(z)-1
\]

#### 차원 체크
- \(A\in \mathbb R^{1\times d}\)
- \(x\in \mathbb R^d\)
- \(z\in \mathbb R\)
- \(\partial L/\partial A \in \mathbb R^{1\times d}\)

#### 저장 효율 관점
- 역전파를 위해 순전파에서 \(z,\sigma(z),x\)를 저장해야 한다.

#### 연산 부하 관점
- 단일 뉴런은 \(O(d)\)
- 다층에서는 레이어별 행렬 곱이 지배적

#### 핵심:
- 바깥 미분 먼저, 안쪽 미분 나중
- 최종 gradient는 오차 신호 × 입력

#### 주의:
- \(\sigma'(z)\)를 \(1-\sigma(z)\)로 잘못 쓰면 오답

---

### 7-2. Softmax Jacobian

#### 한 줄 결론
softmax의 Jacobian은 "한 클래스 확률이 오르면 다른 클래스 확률은 내려가야 한다"는 경쟁 구조를 행렬로 적은 것이다.

#### Tool / Algorithm
\[
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
\]

#### Line-by-Line

\[
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)
\]

\[
\frac{\partial p_i}{\partial z_j}=-p_ip_j \qquad (i\neq j)
\]

따라서
\[
\frac{\partial p}{\partial z}
=\operatorname{diag}(p)-pp^T
\]

#### 기호 해체

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(p_i\) | i번째 확률 | 모델이 i 클래스를 믿는 정도 |
| \(z_i\) | i번째 logit | softmax 전 점수 |
| \(\operatorname{diag}(p)\) | 대각행렬 | 자기 자신 영향 |
| \(pp^T\) | 외적 | 클래스 간 경쟁 영향 |

#### 핵심:
- 대각 원소: 자기 증가 효과
- 비대각 원소: 다른 클래스 억제 효과

#### 주의:
- 모든 행의 합은 0이다.
- 확률 총합 1 제약 때문에 생기는 성질이다.

---

### 7-3. Backpropagation 전체 흐름

#### 한 줄 결론
backprop은 loss에서 시작해 오차를 역방향으로 전달하면서 각 가중치의 책임도를 계산하는 알고리즘이다.

#### Goal
다층 신경망에서 모든 파라미터의 gradient를 효율적으로 구하는 것.

#### 알고리즘 순서
1. 순전파로 중간값 저장
2. 최종 loss 계산
3. 출력층 gradient 계산
4. 체인룰로 이전 층에 오차 전파
5. 각 층의 gradient를 `오차 × 입력` 외적으로 계산

#### 백엔드 비유
- 순전파 로그 저장 = 요청 trace 저장
- 역전파 = 장애 원인 전파 경로 역추적

#### 핵심:
- forward cache가 필요하다
- Jacobian을 직접 만들지 않고 곱셈 형태로 전파한다

---

## 8. 최적화와 학습률 조건

> 교재 근거: [page_060], [page_061], [page_296], [page_301]

### 8-1. Gradient Descent 수렴 조건

#### 한 줄 결론
이차 함수에서 learning rate가 너무 크면 튕겨 나가고, 충분히 작으면 수렴한다.

#### Tool / Algorithm
\[
f(w)=\frac12 w^TAw,\qquad \nabla f(w)=Aw
\]

\[
w_{t+1}=w_t-\eta Aw_t=(I-\eta A)w_t
\]

#### 핵심 결과
\[
0<\eta<\frac{2}{\lambda_{\max}(A)}
\]
이면 수렴한다.

#### Line-by-Line
\[
w_{t+1}=(I-\eta A)w_t
\]

\[
A=U\Lambda U^T
\Rightarrow
w_{t+1}=U(I-\eta\Lambda)U^T w_t
\]

- 좌표를 고유벡터 축으로 바꾸면 각 축이 독립적으로 움직인다.

각 축에서
\[
y_{t+1}^{(i)}=(1-\eta\lambda_i)y_t^{(i)}
\]

수렴하려면 절댓값이 1보다 작아야 하므로
\[
|1-\eta\lambda_i|<1
\]

\[
-1<1-\eta\lambda_i<1
\]

\[
0<\eta\lambda_i<2
\]

모든 \(i\)에 대해 성립하려면 가장 큰 고유값 기준으로
\[
0<\eta<\frac2{\lambda_{\max}(A)}
\]

#### 중1 설명
- 너무 크게 움직이면 목표점을 지나쳐버린다.

#### 중2 설명
- 가장 가파른 축에 맞춰 보폭을 잡아야 한다.

#### 고1 설명
- 한 축에서는 사실상 \(x_{t+1}=(1-c)x_t\) 꼴이다.
- 이때 \(|1-c|<1\)이면 줄어든다.

#### 핵심:
- learning rate 상한은 최대 고유값이 결정
- 가장 가파른 방향이 전체 안정성을 지배

#### 주의:
- \(\lambda_{\min}\)이 아니라 \(\lambda_{\max}\)다.

---

## 9. Pooling과 선형대수 표현

> 교재 근거: CNN 보조자료 및 시험 예상 범위 반영

### 9-1. Average Pooling

#### 한 줄 결론
average pooling은 입력 여러 칸을 묶어 평균을 내는 다운샘플링 연산이고, 행렬 곱으로도 표현할 수 있다.

#### Goal
CNN 연산도 사실 선형대수 연산으로 옮겨 적을 수 있음을 이해하는 것.

#### Tool / Algorithm
\[
x=
\begin{bmatrix}
x_1\\x_2\\x_3\\x_4
\end{bmatrix},
\qquad
y=
\begin{bmatrix}
(x_1+x_2)/2\\
(x_3+x_4)/2
\end{bmatrix}
\]

그러면
\[
y=Px
\]
이고
\[
P=
\begin{bmatrix}
1/2 & 1/2 & 0 & 0\\
0 & 0 & 1/2 & 1/2
\end{bmatrix}
\]

#### 기호 해체

| 기호 | 뜻 | 설명 |
|---|---|---|
| \(x\) | 입력 벡터 | 원본 feature |
| \(P\) | pooling 행렬 | 평균 묶음 규칙 |
| \(y\) | 출력 벡터 | 다운샘플된 결과 |

#### 핵심:
- pooling도 행렬 곱으로 쓸 수 있다
- conv/pooling을 선형대수로 바꾸면 계산 구조가 명확해진다

---

## 10. 선형회귀, 정규방정식, 릿지

> 교재 근거: [page_293], [page_296], [page_301]

### 10-1. 선형회귀

#### 한 줄 결론
선형회귀는 입력 feature에 가중치를 곱해 더한 값으로 연속 출력을 예측하는 가장 기본 모델이다.

#### Tool / Algorithm
\[
f_w(x)=x^Tw+b
\]

편향 흡수 후
\[
f_w(x)=x^Tw
\]

설계 행렬
\[
X=
\begin{bmatrix}
x_1^T\\
\vdots\\
x_n^T
\end{bmatrix}
\]

예측
\[
f_w(X)=Xw
\]

#### 핵심:
- 중학교 1차함수의 다변수 확장
- 모든 회귀/분류 모델의 시작점

---

### 10-2. 정규방정식

#### 한 줄 결론
최소제곱 해는 미분을 0으로 두면 나오는 선형시스템을 풀어 구한다.

#### Tool / Algorithm
\[
L(w)=\frac12\|Xw-y\|^2
\]

\[
\nabla_w L = X^T(Xw-y)=0
\]

\[
X^TX\hat w = X^Ty
\]

이것이 정규방정식이다.

#### 핵심:
- 미분 → 0 → 정규방정식
- 해가 유일하려면 \(X^TX\) 가역 필요

---

### 10-3. 릿지 회귀

#### 한 줄 결론
릿지 회귀는 MSE에 L2 패널티를 더해 해를 안정화하고 과적합을 줄이는 방법이다.

#### Tool / Algorithm
\[
L_\lambda(w)=\frac12\|Xw-y\|^2+\frac12\lambda\|w\|^2
\]

\[
\hat w_\lambda=(X^TX+\lambda I)^{-1}X^Ty
\]

#### MAP 연결
- Gaussian prior를 쓰면 MAP가 릿지와 연결된다.

#### 핵심:
- \(\lambda>0\)이면 항상 유일한 해
- weight decay의 수학적 기원

---

## 11. 실전 예상문제

### 문제 1. Uniform 평균/분산
- 문제: \(X\sim U[a,b]\)일 때 \(E[X]\), \(\operatorname{Var}(X)\)를 정의부터 유도하라.
- 출제 의도: 적분 정의와 분산 공식 숙련도 확인
- 감점 포인트:
  - \(\frac1{b-a}\)를 빼먹음
  - \(E[X^2]\) 없이 분산 점프

### 문제 2. Bernoulli MLE
- 문제: 성공 \(S\), 시행 수 \(n\)일 때 MLE를 구하라.
- 풀이 시작 전에 적을 가정: i.i.d., \(\theta\in[0,1]\)
- 감점 포인트:
  - log-likelihood 생략
  - 경계 조건 언급 없음

### 문제 3. MAP with \(p(\theta)\propto \theta^m(1-\theta)^m\)
- 문제: MAP를 유도하고 MLE와 비교하라.
- 감점 포인트:
  - prior 로그항 빠뜨림
  - MAP와 posterior 전체를 혼동

### 문제 4. Gaussian NLL → MSE
- 문제: 왜 MSE가 Gaussian 가정하의 MLE와 동치인지 유도하라.
- 감점 포인트:
  - 로그를 왜 쓰는지 설명 없음
  - 상수항 처리 설명 없음

### 문제 5. KL = CE - H
- 문제: 정의에서 직접 유도하라.
- 감점 포인트:
  - 부호 오류
  - \(H(p)\) 상수 해석 누락

### 문제 6. Softmax Jacobian
- 문제: \(i=j\), \(i\neq j\)를 나누어 유도하고 행렬 형태로 쓰라.
- 감점 포인트:
  - off-diagonal 부호 실수
  - \(\operatorname{diag}(p)-pp^T\)를 못 씀

### 문제 7. \(-\log \sigma(Ax+b)\) 미분
- 문제: \(A,x,b\) 각각에 대한 gradient와 차원을 구하라.
- 감점 포인트:
  - \(\sigma'(z)\) 실수
  - row/column shape 틀림

### 문제 8. \(0<\eta<2/\lambda_{\max}(A)\) 증명
- 문제: quadratic loss에서 GD 수렴 조건을 고유값 관점으로 보이라.
- 감점 포인트:
  - \(I-\eta A\)를 안 씀
  - 각 축 독립 해석 빠짐

### 문제 9. Average Pooling의 행렬 표현
- 문제: stride 2 average pooling을 행렬 \(P\)로 쓰라.
- 감점 포인트:
  - 평균인데 \(1/2\) 계수 누락

### 문제 10. 고유값/고유벡터/대각화
- 문제: 2x2 행렬을 대각화하고 의미를 설명하라.
- 감점 포인트:
  - 특성방정식 실수
  - \(PDP^{-1}\) 순서 실수

---

## 12. 시험 직전 백지유도 체크리스트

- [ ] \(\det(A-\lambda I)=0\)에서 2x2 고유값을 1분 내 계산 가능
- [ ] \((A-\lambda I)v=0\)에서 고유벡터를 바로 구할 수 있음
- [ ] Uniform 평균/분산을 적분으로 유도 가능
- [ ] 표준정규 홀수차 모멘트가 왜 0인지 설명 가능
- [ ] Bernoulli MLE를 처음부터 끝까지 유도 가능
- [ ] MAP에 prior가 어떻게 추가되는지 설명 가능
- [ ] Gaussian NLL → MSE를 줄마다 설명 가능
- [ ] \(CE = KL + H\)를 정의에서 바로 유도 가능
- [ ] softmax Jacobian을 \(diag(p)-pp^T\)로 쓸 수 있음
- [ ] \(-\log\sigma(Ax+b)\) 미분을 shape까지 맞게 쓸 수 있음
- [ ] \(0<\eta<2/\lambda_{\max}\)를 고유값 축으로 설명 가능
- [ ] average pooling을 행렬로 바꿔 쓸 수 있음

---

## 13. 마지막 압축 요약

### 쉽게 설명하면
이 시험은 "데이터를 확률로 설명하고, 그걸 loss로 바꾸고, 미분해서 학습시키는 전 과정"을 묻는다.

### 남에게 설명하는 한 문장
"선형대수로 모델 구조를 만들고, 확률로 손실함수를 정당화하고, 미분과 고유값으로 학습 안정성을 분석하는 시험이다."

### 핵심 킬러 요약
- 확률 파트: Uniform, Gaussian, Poisson, 기대값, 분산
- 추정 파트: MLE, MAP, posterior, prior
- 손실 파트: MSE = Gaussian NLL, BCE = Bernoulli NLL
- 정보이론 파트: \(CE = KL + H\), \(KL\ge 0\)
- 미분 파트: chain rule, softmax Jacobian, backprop
- 최적화 파트: \(0<\eta<2/\lambda_{\max}(A)\)
- 선형대수 파트: 고유값, 대각화, quadratic form

### 핵심 정리
핵심:
- `확률 → loss → gradient` 연결을 끊기지 않게 이해해야 한다.
- 수식은 암기가 아니라 "왜 다음 줄이 나오는가"를 설명할 수 있어야 한다.
- 모든 계산은 shape, 가정, 경계조건까지 써야 한다.

주의:
- 로그를 왜 쓰는지 설명하지 않으면 감점
- boundary 체크 없이 interior optimum만 쓰면 감점
- softmax 미분에서 부호를 자주 틀린다
- \(\lambda_{\max}\)와 \(\lambda_{\min}\)를 바꾸면 치명적이다

연구 포인트:
- KL과 variational inference
- Hessian spectrum과 optimization stability
- MAP와 regularization의 베이지안 해석


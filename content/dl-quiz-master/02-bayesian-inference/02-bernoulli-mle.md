---
title: 'Quiz 5 — Bernoulli MLE'
description: '베르누이 분포 MLE: theta* = k/n'
draft: false
---

## 0. 한 줄 요약

베르누이 likelihood $L(\theta)=\theta^{k}(1-\theta)^{n-k}$ 에 로그를 씌우고 미분해 0으로 두면, MLE 는 단순 빈도 $\theta^{\ast}_{\text{MLE}}=k/n$. 2계 도함수가 음수이므로 strictly concave, 임계점은 유일한 전역 최대.

---

## 1. 문제

likelihood 가
$$
L(\theta)=\theta^{k}(1-\theta)^{n-k},\qquad \theta\in(0,1)
$$
일 때, $\ell(\theta)=\log L(\theta)$ 의 도함수 $\dfrac{d\ell}{d\theta}$ 를 구하고 $\theta^{\ast}_{\text{MLE}}$ 를 찾아라.

---

## 2. 출제 의도와 시험 가치

- **MLE 사다리의 출발점**: Quiz 5 → Quiz 6 (MAP) → Quiz 7 (강한 prior 극한) → Quiz 9 (Triangular MAP) 의 모든 풀이가 *동일한 미분 공식* 의 변주다. 여기서 손이 익어야 그 다음이 풀린다.
- **로그 변환 + 1계조건 + 2계조건** 이라는 최적화의 3단계 정석을 가장 단순한 모델에서 체험.
- 결과 $\theta^{\ast}=k/n$ 은 *표본 평균이 곧 추정값* 이라는 통계학의 가장 기본적인 사실.

---

## 3. 사전 개념 (모든 수학 도구)

### 3.1 베르누이 분포와 likelihood

- 베르누이 시행: 결과가 $\{0,1\}$ 이고 $P(X=1)=\theta$.
- $n$ 회 독립 시행에서 1이 $k$ 번 나왔다면, 결합확률(=likelihood, $\theta$ 의 함수로 본 것)은
$$
L(\theta)=\prod_{i=1}^{n}\theta^{x_i}(1-\theta)^{1-x_i}=\theta^{k}(1-\theta)^{n-k}.
$$

### 3.2 MLE 정의

$\theta^{\ast}_{\text{MLE}}=\arg\max_{\theta\in(0,1)} L(\theta).$

### 3.3 로그의 단조성

$\log$ 는 strictly increasing 이므로 임의의 양함수 $f>0$ 에 대해
$$
\arg\max_{\theta} f(\theta)=\arg\max_{\theta}\log f(\theta).
$$
→ 곱 형태 $L$ 을 합 형태 $\ell=\log L$ 로 바꿔 미분이 쉬워진다.

### 3.4 미분 공식

- $\dfrac{d}{d\theta}\log\theta=\dfrac{1}{\theta}$.
- 체인룰: $\dfrac{d}{d\theta}\log(1-\theta)=\dfrac{1}{1-\theta}\cdot(-1)=\dfrac{-1}{1-\theta}$.

### 3.5 1계 / 2계 조건

- 내부 임계점: $\dfrac{d\ell}{d\theta}=0$.
- 극대 판정: $\dfrac{d^{2}\ell}{d\theta^{2}}<0$ → strictly concave → 임계점이 *유일한* 전역 최대점.

---

## 4. 풀이 (모든 단계, 등호 근거)

**Step 1. 로그 변환.**
$$
\ell(\theta)=\log L(\theta)=\log\bigl[\theta^{k}(1-\theta)^{n-k}\bigr]=k\log\theta+(n-k)\log(1-\theta).
$$
*근거*: $\log(ab)=\log a+\log b$, $\log(a^{c})=c\log a$. 그리고 $\arg\max L=\arg\max\ell$ (3.3).

**Step 2. 미분.**
$$
\frac{d\ell}{d\theta}=k\cdot\frac{1}{\theta}+(n-k)\cdot\frac{-1}{1-\theta}=\boxed{\;\frac{k}{\theta}-\frac{n-k}{1-\theta}\;}.
$$
*근거*: 3.4 의 두 미분 공식, 합의 미분.

**Step 3. 1계 조건 $d\ell/d\theta=0$.**
$$
\frac{k}{\theta}=\frac{n-k}{1-\theta}.
$$
양변에 $\theta(1-\theta)$ 를 곱(분모 제거, $\theta\in(0,1)$ 이므로 양수, 부호 보존):
$$
k(1-\theta)=(n-k)\theta.
$$
좌변 전개: $k-k\theta=n\theta-k\theta$. 양변에 $k\theta$ 더하면(=좌·우 모두에서 $-k\theta$ 소거):
$$
k=n\theta\;\Longrightarrow\;\boxed{\theta^{\ast}_{\text{MLE}}=\frac{k}{n}}.
$$

**Step 4. 2계 조건 (극대 확인).**
$$
\frac{d^{2}\ell}{d\theta^{2}}=\frac{d}{d\theta}\!\left(\frac{k}{\theta}\right)-\frac{d}{d\theta}\!\left(\frac{n-k}{1-\theta}\right)=-\frac{k}{\theta^{2}}-\frac{n-k}{(1-\theta)^{2}}.
$$
$k\ge 0,\;n-k\ge 0,\;\theta^{2}>0,\;(1-\theta)^{2}>0$ 이므로
$$
\frac{d^{2}\ell}{d\theta^{2}}<0\quad(\forall\theta\in(0,1),\text{ 단 } 0<k<n).
$$
→ $\ell$ 은 strictly concave, 임계점은 *유일한 전역 최대*.

---

## 5. 검증

- **수치 예**: $n=10,\;k=3$ → $\theta^{\ast}=0.3$. likelihood 직접 그려도 같은 봉우리.
- **경계 케이스**:
  - $k=0$: $\ell=n\log(1-\theta)$, 미분 $=-n/(1-\theta)<0$ → 단조감소 → 최댓값은 $\theta\to 0^{+}$ 에서 → $\theta^{\ast}=0=k/n$. 공식 그대로 성립.
  - $k=n$: 대칭으로 $\theta^{\ast}=1=k/n$.
- **한계**: $k=3,n=3$ 처럼 작은 표본의 극단 케이스에서 $\theta^{\ast}=1$. "절대 100%" 라는 비현실적 추정 → MAP/prior 도입 동기 (Quiz 6, 7 로 이어짐).

---

## 6. 일반화·통찰

- **표본평균 = MLE**: $\bar X=\dfrac{1}{n}\sum x_i=\dfrac{k}{n}=\theta^{\ast}$.
- **지수족 정리의 최단 사례**: 지수족 분포에서 충분통계량(sufficient statistic) 의 *표본 평균* 이 MLE — 베르누이는 그 지수족의 가장 단순한 멤버.
- **3단계 알고리즘**: (1) 로그 → (2) 미분 → (3) 0 으로. 이 골격이 Quiz 6·7·9 모두 동일.
- **MLE 의 약점**: 작은 $n$ 에서 분산이 크고 경계로 튀어나감 → 정규화(prior) 가 필요해짐 → MAP.

---

## 7. 시험 출제 변형 5가지

1. **수치 대입형**: $n=20,k=7$ 의 MLE 와 그때 $\ell(\theta^{\ast})$ 값을 계산.
2. **2계 도함수 부호 증명**: $\ell$ 이 strictly concave 임을 *명시적으로* 보여라.
3. **다항(Categorical)으로 확장**: $K$ 개 범주, 빈도 $k_1,\dots,k_K$, 합 $n$ → MLE $\theta_j^{\ast}=k_j/n$ 유도 (Lagrange 승수 $\sum\theta_j=1$).
4. **표본평균과의 관계**: $\theta^{\ast}=\bar X$ 를 보이고, 이것이 *불편추정량(unbiased estimator)* 임을 검증.
5. **로그를 안 쓰면?**: $L$ 자체를 미분해도 같은 결과가 나옴을 보여라 (계산은 더 길지만).

---

## 8. 백지 재현 체크리스트

- [ ] $L(\theta)=\theta^{k}(1-\theta)^{n-k}$ 적기.
- [ ] 로그를 씌워 $\ell=k\log\theta+(n-k)\log(1-\theta)$.
- [ ] $\dfrac{d\ell}{d\theta}=\dfrac{k}{\theta}-\dfrac{n-k}{1-\theta}$ 미분.
- [ ] $=0$ 으로 두고 교차곱: $k(1-\theta)=(n-k)\theta$.
- [ ] $k=n\theta$ → $\theta^{\ast}=k/n$.
- [ ] 2계 도함수 $-k/\theta^{2}-(n-k)/(1-\theta)^{2}<0$ 적어 극대 확인.
- [ ] 경계 $k=0,n$ 도 $k/n$ 로 일치 코멘트.

---

## 9. 핵심 공식 카드

| 식 | 값 |
|---|---|
| $L(\theta)$ | $\theta^{k}(1-\theta)^{n-k}$ |
| $\ell(\theta)$ | $k\log\theta+(n-k)\log(1-\theta)$ |
| $\dfrac{d\ell}{d\theta}$ | $\dfrac{k}{\theta}-\dfrac{n-k}{1-\theta}$ |
| $\theta^{\ast}_{\text{MLE}}$ | $\dfrac{k}{n}$ |
| $\dfrac{d^{2}\ell}{d\theta^{2}}$ | $-\dfrac{k}{\theta^{2}}-\dfrac{n-k}{(1-\theta)^{2}}<0$ |

---

## 10. 다른 퀴즈와의 연결

- **Quiz 4 (Bayes)**: $P(\theta\mid D)\propto P(D\mid\theta)P(\theta)$ — Quiz 5 는 prior 를 *상수*(=uniform) 로 두었을 때의 MAP 과 동치. 즉 MLE = uniform-prior MAP.
- **Quiz 6 (Beta(2,2) prior MAP)**: 같은 미분 공식에서 지수만 $k\to k+1$, $n-k\to n-k+1$ 로 치환 → $\theta^{\ast\ast}=(k+1)/(n+2)$.
- **Quiz 7 (Strong prior 극한)**: 지수가 $k\to k+M$, $n-k\to n-k+M$ → $M\to\infty$ 면 $\theta^{\ast}\to 1/2$. 데이터를 prior 가 압도.
- **Quiz 8 (CE/NLL/MSE 동치)**: NLL 최소화 = likelihood 최대화 → MLE 와 같은 작업의 손실함수 표현.
- **Quiz 9 (Triangular prior MAP)**: 같은 미분 골격에 prior 미분항이 더해지고 support 경계 처리가 추가됨.

---
title: 'Quiz 6 — Beta Prior MAP'
description: 'Beta(2,2) 사전분포 MAP: theta* = (k+1)/(n+2)'
draft: false
---

## 0. 한 줄 요약

베르누이 likelihood 에 $p(\theta)\propto\theta(1-\theta)$ (Beta(2,2)) prior 를 곱하면 posterior 의 형태가 $\theta^{k+1}(1-\theta)^{n-k+1}$. Quiz 5 의 풀이를 *지수만 +1* 치환해 그대로 옮기면 $\theta^{\ast\ast}_{\text{MAP}}=(k+1)/(n+2)$. prior 가 *가상의 1성공 + 1실패* 를 더해주는 pseudo-counts 효과.

---

## 1. 문제

(Beta(2,2) prior 가 곱해진) likelihood 형태가
$$
L(\theta)=\theta^{k+1}(1-\theta)^{n-k+1},\qquad \theta\in(0,1)
$$
일 때, $\ell(\theta)=\log L(\theta)$ 의 도함수 $\dfrac{d\ell}{d\theta}$ 와 MAP 추정값 $\theta^{\ast\ast}$ 를 구하라.

> 의미: 베르누이 likelihood $\theta^{k}(1-\theta)^{n-k}$ 에 prior $p(\theta)\propto\theta(1-\theta)$ 를 곱한 *posterior* 형태이다.

---

## 2. 출제 의도와 시험 가치

- **MAP 의 작동 원리**: posterior $\propto$ likelihood $\times$ prior, 그리고 $\arg\max$ 에서는 evidence $P(E)$ 를 무시할 수 있다는 베이즈 핵심 사용.
- **Quiz 5 와의 *기호 치환* 대응**: $k\to k+1$, $n-k\to n-k+1$ 만 바꾸면 분모가 $n+2$ 로 변함. 같은 미분 골격을 두 번째로 사용해 *익숙해지기*.
- **Pseudo-counts 직관**: prior 의 영향이 *가상의 데이터* 를 더한 것과 같다는 통계학적 명관.
- **Conjugate prior**: Beta 가 베르누이의 conjugate prior — posterior 도 Beta 로 닫혀 MAP 식이 깔끔.

---

## 3. 사전 개념 (모든 수학 도구)

### 3.1 Bayes 정리와 MAP

$$
P(\theta\mid D)=\frac{P(D\mid\theta)\,P(\theta)}{P(D)}.
$$
$P(D)$ 는 $\theta$ 와 무관 (전확률 법칙: $P(D)=\int P(D\mid\theta)P(\theta)d\theta$). 따라서
$$
\theta^{\ast\ast}_{\text{MAP}}=\arg\max_{\theta} P(\theta\mid D)=\arg\max_{\theta} P(D\mid\theta)P(\theta).
$$

### 3.2 Beta 분포

$$
\mathrm{Beta}(\alpha,\beta):\quad p(\theta)=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)},\quad \theta\in(0,1).
$$
$\alpha=\beta=2$ 면 $p(\theta)\propto\theta(1-\theta)$ — 0.5 에 정점이 있는 *부드러운* 종 모양.

### 3.3 Conjugate prior

likelihood $\times$ prior 의 *함수 형태* 가 prior 와 같은 부류이면 conjugate. 베르누이의 likelihood $\theta^{k}(1-\theta)^{n-k}$ 와 Beta($\alpha,\beta$) 는 곱하면 다시 Beta($k+\alpha,\;n-k+\beta$). 그래서 posterior 가 닫힘.

### 3.4 Quiz 5 의 결과 재사용

$f(\theta)=\theta^{a}(1-\theta)^{b}$, $a,b>0$ 일 때
$$
\frac{d\log f}{d\theta}=\frac{a}{\theta}-\frac{b}{1-\theta},\qquad \arg\max=\frac{a}{a+b}.
$$
이 한 줄이 Quiz 5·6·7 모두의 엔진.

---

## 4. 풀이 (모든 단계, 등호 근거)

**Step 1. likelihood × prior 의 형태 확인.**
$$
P(D\mid\theta)P(\theta)\propto \underbrace{\theta^{k}(1-\theta)^{n-k}}_{\text{Bernoulli}}\cdot\underbrace{\theta(1-\theta)}_{\text{Beta(2,2)}}=\theta^{k+1}(1-\theta)^{n-k+1}.
$$
*근거*: 같은 밑수의 거듭제곱 곱은 지수의 합. $\theta^k\cdot\theta=\theta^{k+1}$.

**Step 2. 로그 변환.**
$$
\ell(\theta)=\log L(\theta)=(k+1)\log\theta+(n-k+1)\log(1-\theta).
$$
*근거*: $\log(ab)=\log a+\log b$, $\log(a^{c})=c\log a$, 그리고 $\log$ 단조증가 → $\arg\max L=\arg\max\ell$.

**Step 3. 미분.**
$$
\boxed{\;\frac{d\ell}{d\theta}=\frac{k+1}{\theta}-\frac{n-k+1}{1-\theta}\;}.
$$
*근거*: $\dfrac{d}{d\theta}\log\theta=1/\theta$, $\dfrac{d}{d\theta}\log(1-\theta)=-1/(1-\theta)$.

**Step 4. 1계 조건 $d\ell/d\theta=0$.**
$$
\frac{k+1}{\theta}=\frac{n-k+1}{1-\theta}.
$$
교차곱 ($\theta(1-\theta)>0$):
$$
(k+1)(1-\theta)=(n-k+1)\theta.
$$
좌변 전개: $k+1-(k+1)\theta$. 우변에 $(k+1)\theta$ 더해 정리:
$$
k+1=\theta\bigl[(k+1)+(n-k+1)\bigr]=\theta\,(n+2).
$$
따라서
$$
\boxed{\theta^{\ast\ast}_{\text{MAP}}=\frac{k+1}{n+2}}.
$$

**Step 5. 2계 조건 (극대 확인).**
$$
\frac{d^{2}\ell}{d\theta^{2}}=-\frac{k+1}{\theta^{2}}-\frac{n-k+1}{(1-\theta)^{2}}<0.
$$
$k+1>0,\;n-k+1>0$ (지수가 1 만큼 늘어나 *경계 케이스에서도 안전*하게 양수) → strictly concave → 임계점이 *유일한 전역 최대*.

---

## 5. 검증

- **수치 예** ($n=3,k=3$): MLE 는 $3/3=1.0$ ("100% 성공" 이라는 극단 추정). MAP 은 $\dfrac{3+1}{3+2}=\dfrac{4}{5}=0.8$. prior 가 1.0 의 극단을 0.5 쪽으로 끌어내려 *오버피팅 완화*.
- **수치 예** ($n=10,k=3$): MLE $=0.3$, MAP $=4/12\approx 0.333$. 데이터가 좀 있으면 prior 영향이 약해짐.
- **경계 안전**: $k=0$ 에서 MAP $=1/(n+2)>0$, MLE 는 0. MAP 이 *0/1 의 절대 추정* 을 피한다.
- **데이터 발산**: $n\to\infty$ 에서 $\dfrac{k+1}{n+2}\to\dfrac{k}{n}$ → MAP → MLE.

---

## 6. 일반화·통찰

- **Pseudo-counts 해석**: Beta(2,2) prior = "*가상의* 성공 1회 + 실패 1회 추가". 그래서 분자가 $k+1$, 분모가 $n+2$.
- **Beta($\alpha,\beta$) 일반식** ($\alpha,\beta>1$):
$$
\theta^{\ast}_{\text{MAP}}=\frac{k+\alpha-1}{n+\alpha+\beta-2}.
$$
$\alpha=\beta=2$ → $(k+1)/(n+2)$.
$\alpha=\beta=1$ (uniform) → $k/n$ = MLE.
- **Conjugate prior 의 깔끔함**: posterior 가 $\mathrm{Beta}(k+\alpha,n-k+\beta)$ 로 닫혀 MAP 식이 *분수 한 줄*.
- **Laplace smoothing**: $(k+1)/(n+2)$ 는 카운트에 1 을 더하는 *라플라스 보정* 의 베이지안적 정당화.
- **Regularization 의 통계적 원형**: prior = regularization. weak prior (작은 $\alpha,\beta$) → 약한 정규화. strong prior (큰 $\alpha,\beta$) → Quiz 7 의 강한 정규화.

---

## 7. 시험 출제 변형 5가지

1. **수치 대입**: $n=4,k=1$, Beta(2,2) → MAP $=2/6=1/3$ 계산.
2. **다른 $(\alpha,\beta)$**: Beta(3,1) prior → MAP 식과 수치 결과.
3. **MLE vs MAP 차이**: $n=2,k=2$ 에서 두 값을 비교하고 *왜* MAP 이 더 합리적인지 한 문단으로.
4. **Posterior 의 형태**: posterior 가 정확히 어떤 Beta 분포가 되는지(정규화 상수 포함) 적기.
5. **데이터 극한**: $n\to\infty$ 에서 MAP $\to$ MLE 임을 분수의 분자·분모를 $n$ 으로 나눠 보이기.

---

## 8. 백지 재현 체크리스트

- [ ] posterior $\propto \theta^{k+1}(1-\theta)^{n-k+1}$.
- [ ] $\ell=(k+1)\log\theta+(n-k+1)\log(1-\theta)$.
- [ ] $d\ell/d\theta=(k+1)/\theta-(n-k+1)/(1-\theta)$.
- [ ] $=0$ 교차곱 → $(k+1)(1-\theta)=(n-k+1)\theta$.
- [ ] 정리 → $\theta(n+2)=k+1$ → $\theta^{\ast\ast}=(k+1)/(n+2)$.
- [ ] 2계 도함수 $<0$ 적기.
- [ ] $n=3,k=3$ → $4/5$ 한 줄 코멘트.

---

## 9. 핵심 공식 카드

| 식 | 값 |
|---|---|
| posterior $\propto$ | $\theta^{k+1}(1-\theta)^{n-k+1}$ |
| $\ell$ | $(k+1)\log\theta+(n-k+1)\log(1-\theta)$ |
| $d\ell/d\theta$ | $\dfrac{k+1}{\theta}-\dfrac{n-k+1}{1-\theta}$ |
| $\theta^{\ast\ast}_{\text{MAP}}$ | $\dfrac{k+1}{n+2}$ |
| Beta($\alpha,\beta$) 일반식 | $\dfrac{k+\alpha-1}{n+\alpha+\beta-2}$ |
| 대표 수치 ($n{=}3,k{=}3$) | MLE $1.0$ → MAP $0.8$ |

---

## 10. 다른 퀴즈와의 연결

- **Quiz 4 (Bayes)**: $P(\theta\mid D)\propto P(D\mid\theta)P(\theta)$ — 본 문제의 *유도 정당화*.
- **Quiz 5 (MLE)**: 같은 미분 공식. 이 문제는 그저 지수에 +1 치환. uniform prior → MLE 와 동치.
- **Quiz 7 (Strong prior 극한)**: prior 를 $\theta^{M}(1-\theta)^{M}$ 로 강화 → MAP $=(k+M)/(n+2M)$, $M\to\infty$ 에서 $\to 1/2$. 본 문제는 $M=1$ 에 해당하는 *약한* 케이스.
- **Quiz 9 (Triangular MAP)**: 같은 미분 골격 + prior 의 piecewise 미분이 더해짐. 받침(Support) 경계 처리 추가.
- **Quiz 8 (CE/NLL/MSE)**: prior 를 손실의 *정규화 항* 으로 보면 MAP = penalized MLE. weight decay 등 딥러닝 정규화의 베이지안 해석.

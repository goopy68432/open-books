---
title: 'Quiz 7 — Strong Prior 극한'
description: 'M → ∞일 때 MAP가 1/2로 수렴'
draft: false
---

## 0. 한 줄 요약

prior $p(\theta)\propto\theta^{M}(1-\theta)^{M}$ 가 곱해진 posterior 를 미분해 0 으로 두면 $\theta^{\ast}_{\text{MAP}}=(k+M)/(n+2M)$. 분자·분모를 $M$ 으로 나눠 $M\to\infty$ 보내면 $\to 1/2$. *강한 prior 는 데이터를 무시하고 자기 봉우리(0.5) 로 끌어당긴다.*

---

## 1. 문제

베르누이 likelihood $L(\theta)=\theta^{k}(1-\theta)^{n-k}$ 에 prior
$$
p(\theta)\propto\theta^{M}(1-\theta)^{M},\qquad M\ge 0,\;\theta\in(0,1)
$$
가 주어진다. (이는 Beta($M+1,M+1$) prior 의 unnormalized 형태.)

(1) $\theta^{\ast}_{\text{MAP}}$ 을 $k,n,M$ 의 함수로 구하라.
(2) $M\to\infty$ 의 극한값을 구하라.

---

## 2. 출제 의도와 시험 가치

- **베이지안 사다리의 정점**: Quiz 5 (MLE) → Quiz 6 (약한 prior MAP) → Quiz 7 (강한 prior 극한). 같은 미분 골격이 *연속적으로 변하는 지수* 와 함께 어떻게 해를 이동시키는지 본다.
- **Pseudo-counts 의 극단**: prior 가 데이터를 *압도* 한다는 것의 정량적 의미. $M$ 만큼의 가짜 성공·실패가 더해지면 데이터 $k,n$ 은 무시될 수 있다.
- **극한 계산 테크닉**: 분자·분모 같은 차수로 나눠 비율의 극한을 구하는 표준 패턴.
- **Regularization 의 본질**: 딥러닝의 강한 weight decay = 강한 prior = 모델이 *데이터를 학습하지 않게 됨* — 이 문제는 그 베이지안 원형.

---

## 3. 사전 개념 (모든 수학 도구)

### 3.1 Posterior $\propto$ likelihood $\times$ prior

$P(D)$ 는 $\theta$ 와 무관한 정규화 상수이므로 $\arg\max$ 에서 무시 가능:
$$
P(\theta\mid D)\propto P(D\mid\theta)P(\theta).
$$

### 3.2 거듭제곱의 곱

$$
\theta^{k}(1-\theta)^{n-k}\cdot\theta^{M}(1-\theta)^{M}=\theta^{k+M}(1-\theta)^{n-k+M}.
$$

### 3.3 Quiz 5 의 결과 재사용

$f(\theta)=\theta^{a}(1-\theta)^{b}$, $a,b>0$ 일 때
$$
\frac{d\log f}{d\theta}=\frac{a}{\theta}-\frac{b}{1-\theta},\qquad \arg\max=\frac{a}{a+b}.
$$
본 문제는 $a=k+M$, $b=n-k+M$ 의 *치환* 사례.

### 3.4 비율의 극한

$$
\lim_{M\to\infty}\frac{k+M}{n+2M}=\lim_{M\to\infty}\frac{k/M+1}{n/M+2}=\frac{0+1}{0+2}=\frac{1}{2}.
$$
$k,n$ 이 *고정 상수*, $M$ 만 무한대로 가는 점이 핵심.

### 3.5 Strong prior = Dirac delta 직관

Beta($M+1,M+1$) 의 분산은 $\dfrac{(M+1)^{2}}{(2M+2)^{2}(2M+3)}\sim\dfrac{1}{8M}\to 0$. → prior 가 $\theta=1/2$ 한 점에 *무한히 집중* (델타에 가까워짐).

---

## 4. 풀이 (모든 단계, 등호 근거)

**Step 1. log-posterior 형태.**
$$
\log p(\theta\mid D)=\log L(\theta)+\log p(\theta)+\text{const}.
$$
$$
=k\log\theta+(n-k)\log(1-\theta)+M\log\theta+M\log(1-\theta)+\text{const}.
$$
같은 항끼리 묶기:
$$
=(k+M)\log\theta+(n-k+M)\log(1-\theta)+\text{const}.
$$
*근거*: $\log(ab)=\log a+\log b$, $\log(\theta^{M})=M\log\theta$. 정규화 상수 $-\log P(D)$ 와 prior 의 $-\log B(\cdot)$ 등은 $\theta$ 무관 → const.

**Step 2. 미분.**
$$
\frac{d}{d\theta}\log p(\theta\mid D)=\frac{k+M}{\theta}-\frac{n-k+M}{1-\theta}.
$$
*근거*: 3.3 의 공식 (Quiz 5 와 동일 골격).

**Step 3. 1계 조건.**
$$
\frac{k+M}{\theta}=\frac{n-k+M}{1-\theta}.
$$
교차곱 ($\theta(1-\theta)>0$):
$$
(k+M)(1-\theta)=(n-k+M)\theta.
$$
좌변 전개: $(k+M)-(k+M)\theta$. 양변에 $(k+M)\theta$ 더해 정리:
$$
k+M=\theta\bigl[(k+M)+(n-k+M)\bigr]=\theta\,(n+2M).
$$
따라서
$$
\boxed{\theta^{\ast}_{\text{MAP}}=\frac{k+M}{n+2M}}.
$$

**Step 4. 2계 조건 (극대 확인).**
$$
\frac{d^{2}}{d\theta^{2}}\log p=-\frac{k+M}{\theta^{2}}-\frac{n-k+M}{(1-\theta)^{2}}<0.
$$
$M\ge 0$ 이고 $k+M,n-k+M>0$ → strictly concave → 임계점이 *유일한 전역 최대*.

**Step 5. $M\to\infty$ 극한.**
$$
\theta^{\ast}_{\text{MAP}}=\frac{k+M}{n+2M}=\frac{k/M+1}{n/M+2}.
$$
$k,n$ 고정, $M\to\infty$ → $k/M\to 0$, $n/M\to 0$. 따라서
$$
\boxed{\lim_{M\to\infty}\theta^{\ast}_{\text{MAP}}=\frac{0+1}{0+2}=\frac{1}{2}.}
$$

---

## 5. 검증

- **수치 시연** ($n=10,k=8$, MLE $=0.8$):
  - $M=0$ → $8/10=0.8$ (= MLE).
  - $M=1$ → $9/12=0.75$.
  - $M=10$ → $18/30=0.6$.
  - $M=100$ → $108/210\approx 0.514$.
  - $M=10^{6}$ → $\approx 0.5000004$. → $1/2$ 로 수렴.
- **반대 극한 $M\to 0$**: $\theta^{\ast}\to k/n$ = MLE. uniform 에 가까운 prior 는 MLE 를 회복.
- **데이터 폭증 극한 $n\to\infty$** (M 고정): $\dfrac{k+M}{n+2M}\to\dfrac{k}{n}$ → MLE. 데이터가 prior 를 압도.
- **Pseudo-counts 무게 비교**: 데이터 무게 $n$ vs prior 무게 $2M$. $n\gg 2M$ → MLE 우세, $2M\gg n$ → prior 우세.

---

## 6. 일반화·통찰

- **데이터 vs prior 균형**: 분모 $n+2M$ 은 *총 effective sample size*. 데이터 $n$ 표본과 *가짜* $2M$ 표본의 합. 그 안에서 성공 비율 $(k+M)/(n+2M)$.
- **MAP 의 두 극단**:
  - $M\to 0$ 또는 $n\to\infty$: MAP = MLE = $k/n$.
  - $M\to\infty$ (with $n$ 고정): MAP $\to 1/2$ = prior 의 mode.
- **델타 prior 의 의미**: $\delta_{0.5}$ 는 "$\theta$ 는 무조건 1/2" 라는 *교조적* 신념. 그러면 데이터가 무엇이든 답은 1/2 — 베이지안 추론이 *학습 못 함*. 강한 prior 의 위험을 상징.
- **딥러닝 대응**: weight decay $\lambda$ 가 무한대로 가면 가중치 $\to 0$ — 데이터를 학습 안 함. $\lambda\to 0$ 이면 unregularized MLE — 오버피팅. 적절한 $\lambda$ 가 prior 의 적절한 강도와 같다.
- **Regularization 의 두 양식과의 관계** (Quiz 9 와 비교): Quiz 7 은 *soft* 압도 (prior 항이 분자에서 데이터 항을 누름), Quiz 9 의 $m=6$ 은 *hard* 절단 (prior support 가 데이터 mode 를 잘라냄).

---

## 7. 시험 출제 변형 5가지

1. **수치 극한**: $n=5,k=3$ 에서 $M=0,1,5,50$ 의 MAP 값을 모두 계산.
2. **반대 극한**: $M\to 0$ 의 극한이 MLE 임을 식으로 보이고 의미 서술.
3. **데이터 극한**: $n\to\infty$ ($M$ 고정) 에서 MAP $\to k/n$ 임을 보이기.
4. **비대칭 prior**: $p(\theta)\propto\theta^{M_1}(1-\theta)^{M_2}$ 에서 MAP 극한 ($M_1,M_2\to\infty$ 비율 $r=M_1/(M_1+M_2)$ 고정) → $r$.
5. **수렴 속도**: $\bigl|\theta^{\ast}_{\text{MAP}}-1/2\bigr|=\dfrac{|2k-n|}{2(n+2M)}=O(1/M)$ 임을 보이고 *M 이 두 배가 되면 편차가 절반* 임을 코멘트.

---

## 8. 백지 재현 체크리스트

- [ ] posterior $\propto\theta^{k+M}(1-\theta)^{n-k+M}$ 적기.
- [ ] log-posterior $=(k+M)\log\theta+(n-k+M)\log(1-\theta)+C$.
- [ ] 미분 $=(k+M)/\theta-(n-k+M)/(1-\theta)$.
- [ ] $=0$ → 교차곱 → $\theta(n+2M)=k+M$.
- [ ] $\theta^{\ast}=(k+M)/(n+2M)$ 박스.
- [ ] 분자·분모 $M$ 으로 나눠 $M\to\infty$ → $1/2$.
- [ ] 한 문장 해석: "강한 prior 가 데이터를 압도, 0.5 로 수렴".

---

## 9. 핵심 공식 카드

| 식 | 값 |
|---|---|
| posterior $\propto$ | $\theta^{k+M}(1-\theta)^{n-k+M}$ |
| log-posterior | $(k+M)\log\theta+(n-k+M)\log(1-\theta)+C$ |
| $d/d\theta$ | $\dfrac{k+M}{\theta}-\dfrac{n-k+M}{1-\theta}$ |
| $\theta^{\ast}_{\text{MAP}}$ | $\dfrac{k+M}{n+2M}$ |
| $M\to\infty$ | $\dfrac{1}{2}$ |
| $M\to 0$ | $\dfrac{k}{n}$ (= MLE) |
| $n\to\infty$ ($M$ 고정) | $\dfrac{k}{n}$ (= MLE) |

---

## 10. 다른 퀴즈와의 연결

- **Quiz 4 (Bayes)**: $P(\theta\mid D)\propto P(D\mid\theta)P(\theta)$ — 본 문제의 출발.
- **Quiz 5 (MLE)**: $M=0$ 의 특수 케이스 → $\theta^{\ast}=k/n$. 같은 미분 공식.
- **Quiz 6 (Beta(2,2) MAP)**: $M=1$ 의 특수 케이스 → $\theta^{\ast\ast}=(k+1)/(n+2)$. 본 문제는 그 *연속적 일반화*.
- **Quiz 9 (Triangular MAP)**: prior 의 강도 $m\to\infty$ 도 동일한 메시지: prior 가 정점 0.5 의 델타에 가까워지면 MAP $\to 0.5$. *방식만 다른 (soft 압도 vs hard 절단) 같은 결론*.
- **Quiz 8 (CE/NLL/MSE)**: regularized loss = MAP 의 손실 표현. 본 문제의 $M$ 은 weight decay $\lambda$ 의 베이지안 친구.
- **종합 메시지** (66% 마스터 노트): 베이지안 추론의 본질은 *데이터 무게 $n$ 과 prior 무게 $M$ 의 균형*. 어느 한쪽이 압도적이면 다른 쪽이 잊혀진다.

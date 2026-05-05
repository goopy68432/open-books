---
title: "딥러닝 중간고사 답안형 치트시트"
slug: 00-midterm-exam-answer-style-cheatsheet
order: 1
---

# 딥러닝 중간고사 답안형 치트시트

> 기준 문서: [00_midterm_exam_master_final_merged.md](/Users/jeongseongchae/dev/university/deep_learning/final/deep_learing_prev_exam/codex/00_midterm_exam_master_final_merged.md)  
> 목적: 시험장에서 바로 쓸 수 있는 문장, 유도 순서, 감점 방지 포인트만 압축한 백지복원용 버전

---

## 0. 답안 공통 템플릿

### 한 줄 결론
정의만 쓰면 부족하고, `가정 → 식 전개 → 이유 → 결론` 순서로 써야 점수가 나온다.

### 모든 문제에서 먼저 쓰는 문장
- "먼저 문제에서 최적화하려는 대상/계산하려는 대상을 정의하겠습니다."
- "이때 필요한 가정은 다음과 같습니다."
- "이 가정에 의해 식을 다음과 같이 전개할 수 있습니다."
- "각 단계의 이유를 함께 설명하겠습니다."
- "따라서 최종 결론은 다음과 같습니다."

### 감점 방지 공통 문장
- "i.i.d. 가정에 의해 결합확률을 각 샘플 확률의 곱으로 쓸 수 있습니다."
- "곱을 합으로 바꾸고 미분을 쉽게 하기 위해 log를 취합니다."
- "정규화 상수 또는 파라미터와 무관한 상수항은 argmax/argmin에 영향을 주지 않습니다."
- "내부 최적점뿐 아니라 경계도 함께 확인해야 합니다."

---

## 1. 시험 TIP 답안형 버전

- "이 과목은 답보다 과정을 평가하므로, 정의와 가정을 먼저 쓰고 수식을 한 줄씩 전개하겠습니다."
- "슬라이드의 증명과 공식이 그대로 출제될 수 있으므로 표준 전개를 그대로 재현하겠습니다."
- "Preliminary 파트와 알고리즘 비교 문제도 중요하므로 정의와 차이점을 함께 서술하겠습니다."

---

## 2. 확률분포 답안형

### 2-1. Uniform distribution \([a,b]\)

#### 평균
\[
f_X(x)=
\begin{cases}
\frac{1}{b-a}, & a\le x\le b\\
0, & \text{otherwise}
\end{cases}
\]

바로 쓰는 답안:

"연속확률변수의 기대값 정의에 의해"
\[
E[X]=\int_a^b x\cdot \frac{1}{b-a}\,dx
\]
"상수 \(\frac{1}{b-a}\)를 적분 밖으로 빼면"
\[
E[X]=\frac{1}{b-a}\int_a^b x\,dx
\]
"이를 적분하면"
\[
E[X]=\frac{1}{b-a}\left[\frac{x^2}{2}\right]_a^b
=\frac{1}{b-a}\cdot \frac{b^2-a^2}{2}
=\frac{a+b}{2}
\]
"따라서 균등분포의 평균은 구간의 중앙인 \((a+b)/2\)입니다."

#### 분산
"분산 정의에 의해"
\[
\operatorname{Var}(X)=E[X^2]-(E[X])^2
\]
"먼저 \(E[X^2]\)를 계산하면"
\[
E[X^2]=\int_a^b x^2\cdot \frac{1}{b-a}\,dx
=\frac{1}{b-a}\left[\frac{x^3}{3}\right]_a^b
=\frac{a^2+ab+b^2}{3}
\]
"따라서"
\[
\operatorname{Var}(X)=\frac{a^2+ab+b^2}{3}-\left(\frac{a+b}{2}\right)^2
=\frac{(b-a)^2}{12}
\]

### 2-2. Normal distribution

바로 쓰는 핵심:
- "정규분포 \(X\sim \mathcal N(\mu,\sigma^2)\)의 평균은 \(\mu\), 분산은 \(\sigma^2\)입니다."
- "표준정규분포 \(Z\sim\mathcal N(0,1)\)는 원점 대칭이므로 \(E[Z]=0\), 모든 홀수차 모멘트 \(E[Z^{2n-1}]=0\)입니다."
- "짝수차 모멘트는 \(E[Z^{2n}]=(2n-1)!!\)입니다."
- "특히 \(E[Z^2]=1\), \(E[Z^4]=3\)입니다."

### 2-3. Poisson distribution

바로 쓰는 답안:
\[
P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!}
\]
- "Poisson 분포는 단위 시간/공간당 사건 발생 횟수를 모델링합니다."
- "평균과 분산이 모두 \(\lambda\)입니다."

### 2-4. 여러 분포 평균 비교 답안 문장
- "Uniform \([a,b]\)의 평균은 구간의 중앙 \((a+b)/2\)입니다."
- "Gaussian의 평균은 중심 파라미터 \(\mu\)입니다."
- "Poisson의 평균은 발생률 파라미터 \(\lambda\)입니다."

---

## 3. MLE / MAP / Bayesian 답안형

### 3-1. Bayes theorem

바로 쓰는 답안:

"베이즈 정리는 조건부 확률 정의로부터 유도됩니다."
\[
p(h\mid e)=\frac{p(e\mid h)p(h)}{p(e)}
\]
"여기서 \(p(h)\)는 prior, \(p(e\mid h)\)는 likelihood, \(p(h\mid e)\)는 posterior, \(p(e)\)는 evidence입니다."
"따라서 posterior는 likelihood와 prior의 곱에 비례합니다."

### 3-2. Bernoulli MLE

바로 쓰는 답안:

"\(n\)번의 독립 시행에서 \(k\)번 성공했다고 하겠습니다. i.i.d. 가정에 의해 likelihood는"
\[
L(\theta)=\theta^k(1-\theta)^{n-k}
\]
"곱을 합으로 바꾸기 위해 로그를 취하면"
\[
\ell(\theta)=k\log\theta+(n-k)\log(1-\theta)
\]
"극값 후보를 찾기 위해 미분하여 0으로 두면"
\[
\frac{d\ell}{d\theta}=\frac{k}{\theta}-\frac{n-k}{1-\theta}=0
\]
"이를 정리하면"
\[
k(1-\theta)=(n-k)\theta
\Rightarrow k=n\theta
\Rightarrow \hat\theta_{\text{MLE}}=\frac{k}{n}
\]
"따라서 MLE는 관측된 성공 비율입니다."

### 3-3. MAP with uniform prior

바로 쓰는 답안:

"MAP는 posterior를 최대화하는 방법입니다."
\[
\hat\theta_{\text{MAP}}=\arg\max_\theta p(\theta\mid \text{data})
\]
"uniform prior이면 \(\log p(\theta)\)가 상수이므로"
\[
\log p(\theta\mid \text{data})=\log p(\text{data}\mid\theta)+C
\]
"따라서 MAP와 MLE가 일치합니다."

### 3-4. MAP with \(p(\theta)\propto \theta(1-\theta)\)

바로 쓰는 답안:

"이 경우 log-prior는"
\[
\log p(\theta)=\log\theta+\log(1-\theta)+C
\]
"따라서 log-posterior는"
\[
\log p(\theta\mid \text{data})
=(k+1)\log\theta+(n-k+1)\log(1-\theta)+C
\]
"미분하여 0으로 두면"
\[
\frac{k+1}{\theta}-\frac{n-k+1}{1-\theta}=0
\]
"따라서"
\[
\hat\theta_{\text{MAP}}=\frac{k+1}{n+2}
\]
"이는 MLE보다 덜 극단적인 추정입니다."

### 3-5. MAP with \(p(\theta)\propto \theta^m(1-\theta)^m\)

바로 쓰는 답안:

"이 경우 log-prior는"
\[
\log p(\theta)=m\log\theta+m\log(1-\theta)+C
\]
"따라서 log-posterior는"
\[
\log p(\theta\mid \text{data})
=(k+m)\log\theta+(n-k+m)\log(1-\theta)+C
\]
"미분하여 정리하면"
\[
\hat\theta_{\text{MAP}}=\frac{k+m}{n+2m}
\]
"\(m\)이 커질수록 prior가 강해져 \(\theta=1/2\) 근처를 더 선호합니다."

### 3-6. prior update 답안 문장
- "prior와 likelihood를 곱하여 posterior를 만들고, 이것이 belief update입니다."
- "MAP는 업데이트된 posterior의 최댓값을 선택하는 것입니다."

### 3-7. 경계점 / 미분 불가능 지점 답안 문장
- "내부 최적점뿐 아니라 정의역의 경계도 함께 확인해야 합니다."
- "특히 \(\log\theta\), \(\log(1-\theta)\)가 포함되므로 \(\theta=0,1\)은 별도 검토가 필요합니다."

---

## 4. 손실함수 관계 답안형

### 4-1. Gaussian NLL → MSE

바로 쓰는 답안:

"Regression에서 오차를 Gaussian noise로 가정하겠습니다."
\[
y_i=h(x_i)+\varepsilon_i,\qquad \varepsilon_i\sim \mathcal N(0,\sigma^2)
\]
"그러면 조건부 분포는"
\[
p(y_i\mid x_i,h)=\mathcal N(y_i;h(x_i),\sigma^2)
\]
"Negative log-likelihood는"
\[
\text{NLL}(h)=-\sum_{i=1}^n \log p(y_i\mid x_i,h)
\]
"Gaussian log-density를 전개하면"
\[
\text{NLL}(h)=\sum_{i=1}^n \frac{1}{2\sigma^2}(y_i-h(x_i))^2 + C
\]
"즉"
\[
\text{NLL}(h)=\frac{n}{2\sigma^2}\text{MSE}(h)+C
\]
"상수항과 양의 상수배는 argmin에 영향을 주지 않으므로 NLL 최소화와 MSE 최소화는 동치입니다."

### 4-2. NLL ↔ KL ↔ CE

바로 쓰는 답안:

\[
KL(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
\]
"이를 전개하면"
\[
KL(P\|Q)=\sum_x P(x)\log P(x)-\sum_x P(x)\log Q(x)
\]
\[
=-H(P)+CE(P,Q)
\]
"따라서"
\[
CE(P,Q)=KL(P\|Q)+H(P)
\]
"데이터 분포 \(P\)가 고정되면 \(H(P)\)는 상수이므로 cross entropy 최소화는 KL divergence 최소화와 동치입니다."

### 4-3. 전체 관계를 한 문장으로 쓰기
- "Gaussian 가정에서는 MLE가 NLL 최소화가 되고, NLL은 MSE와 동치이며, 더 큰 관점에서는 경험분포와 모델분포 사이의 KL divergence를 줄이는 문제로 해석됩니다."

---

## 5. KL Divergence 답안형

### 5-1. KL ≥ 0

바로 쓰는 답안:
- "KL divergence는 Gibbs inequality에 의해 항상 0 이상입니다."
- "또한 \(P=Q\)일 때에만 0이 됩니다."

### 5-2. Gaussian KL

바로 쓰는 답안:

"평균이 다르고 분산이 같은 두 Gaussian"
\[
P=\mathcal N(\mu_1,\sigma^2),\qquad Q=\mathcal N(\mu_2,\sigma^2)
\]
"에 대해"
\[
KL(P\|Q)=\frac{(\mu_1-\mu_2)^2}{2\sigma^2}
\]
"즉 동일 분산 Gaussian에서는 KL이 평균 차이의 제곱 형태가 됩니다."

### 5-3. 일반 KL 계산 템플릿
- "먼저 두 분포 \(P,Q\) 또는 밀도 \(p,q\)를 명시합니다."
- "그 다음 \(\log(P/Q)\) 또는 \(\log(p/q)\)를 만들고, \(P\) 또는 \(p\)로 가중하여 합 또는 적분합니다."
- "support 불일치, 특히 \(Q=0\)인데 \(P>0\)인 점이 있는지 확인합니다."

---

## 6. 행렬미분 / Backprop 답안형

### 6-1. \(-\log(\sigma(Ax+b))\) 미분

바로 쓰는 답안:

\[
L=-\log \sigma(z),\qquad z=Ax+b
\]
"먼저 chain rule에 의해"
\[
\frac{dL}{dz}=-\frac{1}{\sigma(z)}\sigma'(z)
\]
"그리고 sigmoid derivative는"
\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]
"따라서"
\[
\frac{dL}{dz}=-(1-\sigma(z))=\sigma(z)-1
\]
"이제 다시 chain rule을 적용하면"
\[
\frac{\partial L}{\partial A}=(\sigma(z)-1)x^T
\]
\[
\frac{\partial L}{\partial x}=A^T(\sigma(z)-1)
\]
\[
\frac{\partial L}{\partial b}=\sigma(z)-1
\]

### 6-2. softmax 미분

바로 쓰는 답안:

\[
p_i=\frac{e^{g_i}}{\sum_j e^{g_j}}
\]
"\(i=j\)인 경우"
\[
\frac{\partial p_i}{\partial g_i}=p_i(1-p_i)
\]
"\(i\neq j\)인 경우"
\[
\frac{\partial p_i}{\partial g_j}=-p_ip_j
\]
"따라서 행렬 형태는"
\[
\frac{\partial p}{\partial g}=\operatorname{diag}(p)-pp^T
\]

### 6-3. softmax의 \(a,W,b\) 미분

logit을
\[
g=Wa+b
\]
라고 두면

\[
\frac{\partial L}{\partial a}=W^T\frac{\partial L}{\partial g}
\]
\[
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial g}\,a^T
\]
\[
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial g}
\]

### 6-4. chain rule / backprop

바로 쓰는 답안:
- "복합 함수의 미분은 chain rule로 계산합니다."
- "역전파는 출력층의 gradient부터 이전 층으로 순차적으로 전달하는 계산 방식입니다."
- "각 층의 가중치 gradient는 일반적으로 `오차 × 입력` 외적 형태로 나타납니다."

---

## 7. 학습률 답안형

### 7-1. \( \eta < 2/\lambda_{\max}(A) \)

바로 쓰는 답안:

"Quadratic loss"
\[
f(w)=\frac12 w^TAw
\]
"에 대해 gradient descent update는"
\[
w_{t+1}=w_t-\eta Aw_t=(I-\eta A)w_t
\]
"이제 \(A\)를 고유벡터 축으로 보면 각 방향은 독립적으로"
\[
y_{t+1}^{(i)}=(1-\eta\lambda_i)y_t^{(i)}
\]
"로 움직입니다. 수렴하려면 각 방향에 대해"
\[
|1-\eta\lambda_i|<1
\]
"이어야 하므로"
\[
0<\eta\lambda_i<2
\]
"특히 가장 큰 고유값 방향이 가장 restrictive하므로"
\[
0<\eta<\frac{2}{\lambda_{\max}(A)}
\]
"가 됩니다."

---

## 8. Pooling 답안형

### 8-1. Average pooling matrix form

바로 쓰는 답안:

\[
x=
\begin{bmatrix}
x_1\\x_2\\x_3\\x_4
\end{bmatrix},\qquad
y=
\begin{bmatrix}
\frac{x_1+x_2}{2}\\
\frac{x_3+x_4}{2}
\end{bmatrix}
\]

"이를 행렬 형태로 쓰면"
\[
y=Px
\]
"이며"
\[
P=
\begin{bmatrix}
1/2 & 1/2 & 0 & 0\\
0 & 0 & 1/2 & 1/2
\end{bmatrix}
\]
"입니다."

---

## 9. Quadratic 문제 답안형

### 한 줄 결론
quadratic 문제는 결국 고유값과 고유벡터 관점으로 보면 가장 깔끔하게 풀린다.

### 바로 쓰는 문장
- "이 함수는 quadratic form이므로 고유벡터 기저로 좌표변환하면 각 방향이 분리됩니다."
- "고유값이 큰 방향은 함수가 더 가파르게 변하는 방향입니다."
- "따라서 최적화 안정성은 최대 고유값 \(\lambda_{\max}\)에 의해 결정됩니다."

---

## 10. 시험장에서 바로 쓸 비교 문장

### MLE vs MAP
- "MLE는 데이터에만 의존하고, MAP는 prior knowledge도 함께 반영합니다."
- "MLE는 넓은 hypothesis space, 높은 expressivity, 약한 inductive bias를 갖고, strong MAP는 좁은 hypothesis space, 낮은 expressivity, 강한 inductive bias를 가집니다."

### Classification vs Regression
- "Classification은 출력 \(y\)가 이산적일 때, regression은 출력 \(y\)가 연속적일 때의 문제입니다."
- "Bernoulli는 classification intuition의 기본 예제이고, Gaussian은 regression intuition의 기본 예제입니다."

---

## 11. 마지막 백지복원 체크리스트

- [ ] Uniform 평균/분산 유도 가능
- [ ] Gaussian 평균/분산 바로 씀
- [ ] \(E[Z], E[Z^{2n}], E[Z^{2n-1}]\) 설명 가능
- [ ] Poisson 평균/분산 바로 씀
- [ ] Bernoulli MLE 전체 유도 가능
- [ ] MAP with \(\theta(1-\theta)\) 유도 가능
- [ ] MAP with \(\theta^m(1-\theta)^m\) 유도 가능
- [ ] prior update를 말로 설명 가능
- [ ] 경계점/정의역 체크를 답안에 포함 가능
- [ ] Gaussian NLL → MSE 설명 가능
- [ ] CE = KL + H 유도 가능
- [ ] KL ≥ 0, Gaussian KL, 일반 KL 계산 가능
- [ ] \(-\log(\sigma(Ax+b))\) 미분 가능
- [ ] softmax 미분과 \(a,W,b\) 미분 가능
- [ ] chain rule / backprop 과정 설명 가능
- [ ] \(0<\eta<2/\lambda_{\max}(A)\) 증명 가능
- [ ] average pooling matrix form 가능
- [ ] quadratic 문제를 eigenvalue 관점으로 설명 가능


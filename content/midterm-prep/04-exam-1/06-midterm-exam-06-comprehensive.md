---
title: "딥러닝이론 중간고사 모의시험 — 최종 종합판"
slug: midterm-exam-06-comprehensive
order: 6
---

# 딥러닝이론 중간고사 모의시험 — 최종 종합판

> 한양대학교 AI융합대학원 | Deep Learning Theory | Midterm Examination
> **평가 핵심**: 최종 답보다 **논리적 도달 과정**을 평가. 수식만 나열하면 0점.
> **문제**: 영어 | **풀이**: 한국어 | **20문제** | **200점** | **120분**

---

# Part A: 기본 개념형 (1~5번, 50점)

## 문제 1 (10점)
**Q:** (a) State Bayes' theorem. Define prior, likelihood, posterior, evidence. Explain each term's role in learning. (5pts) (b) Compare Frequentist vs Bayesian using a coin-flip example. (5pts)

**출제 의도:** 베이즈 정리의 유도와 MAP=ML+Prior 프레임워크 이해를 평가.

**핵심 포인트:** - 조건부 확률에서 유도 - 4개 항 역할 (Prior→정규화, Likelihood→Loss) - Freq: θ=상수, Bayes: θ=확률변수

**모범 풀이:**
(a) 조건부 확률 정의 $P(A|B)=P(A\cap B)/P(B)$에서 $P(A\cap B)$를 소거하면: $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$. Prior $p(\theta)$=데이터 전 믿음→정규화. Likelihood $p(D|\theta)$=데이터 적합도→Loss. Posterior $p(\theta|D)$=갱신된 믿음→MAP 대상. Evidence $p(D)$=$\theta$ 무관 상수→제거 가능. 핵심: $\log p(H|E)=\log p(E|H)+\log p(H)+C$ → **MAP=ML+Prior**.

(b) 동전 3번, 3번 앞면: Frequentist(MLE)→$\hat\theta=1.0$ (극단). Bayesian(MAP, Beta(2,2) prior)→$\hat\theta=4/5=0.8$ (절충). Prior가 극단 완화.

**[채점]** 유도: 2 / 4항 역할: 3 / 비교+예시: 5

---

## 문제 2 (10점)
**Q:** (a) Why is ML fundamentally inductive? (4pts) (b) Define i.i.d. Separately explain what "independent" and "identically distributed" enable. Give failure example. (6pts)

**출제 의도:** 귀납적 추론과 i.i.d.의 수학적 역할 분리 이해.

**모범 풀이:**
(a) 귀납=관찰→규칙(불확실). ML=유한 데이터→일반 모델. 100%보장 불가→일반화 문제의 근원.

(b) **Independent**: $p(D|\theta)=\prod_i p(y_i|x_i,\theta)$ (곱 분해). 로그 취하면 합→SGD 가능. **Identically Distributed**: 같은 $\theta$ 공유→단일 모델 가능. 실패: 주식가격(시간 의존+환경 변화).

**[채점]** 귀납+ML: 4 / 독립 역할: 2 / 동일분포 역할: 2 / 실패 예: 2

---

## 문제 3 (10점)
**Q:** The slides show Classification(Categorical→CE) and Regression(Gaussian→MSE). (a) Explain the unifying NLL principle. (5pts) (b) Why is MSE bad for classification? (5pts)

**출제 의도:** 분포 가정→손실 함수 연결, CE vs MSE 그래디언트 차이.

**모범 풀이:**
(a) 통합: $\text{Loss}=-\frac{1}{n}\sum\log p(y_i|x_i,\theta)$. Categorical 대입→CE. Gaussian 대입→$-\log\exp(-z^2)=z^2$→MSE. **분포 가정이 Loss 결정.**

(b) MSE gradient: $(\hat{y}-y)\sigma'(z)x$. CE: $(\hat{y}-y)x$. MSE에 $\sigma'\leq 1/4$ 남음. $\sigma\approx 0$ or $1$이면 $\sigma'\approx 0$→교정 불가. CE는 canonical link 덕분에 σ' 소거→항상 강한 교정.

**[채점]** NLL 통합: 3 / 두 유도: 2 / σ' 문제: 3 / 직관: 2

---

## 문제 4 (10점)
**Q:** (a) Define inductive bias. Connect to prior $p(H)$. (3pts) (b) Rank bias: Linear, CNN, MLP, Transformer. (4pts) (c) 50 vs 100M images: architecture choice. (3pts)

**출제 의도:** 아키텍처=Prior, 데이터 양↔bias 강도.

**모범 풀이:**
(a) Inductive bias=아키텍처에 내장된 가정. MAP의 $p(H)$를 아키텍처로 구현.

(b) Linear(직선)>CNN(국소+이동불변)>MLP(무가정)>Transformer(최소가정).

(c) 50장→CNN(강한 bias, 소량 OK). 1억→Transformer(약한 bias, 자유 학습). Bias-Variance: 적은 데이터→강한 bias 유리, 많은 데이터→약한 bias 유리.

**[채점]** 정의+Prior: 3 / 순서+설명: 4 / 추천+근거: 3

---

## 문제 5 (10점)
**Q:** (a) Define Bernoulli and Gaussian (params, mean, var). (4pts) (b) CLT→Gaussian noise justification. (3pts) (c) Linear approximation→differentiation→Jacobian connection. (3pts)

**출제 의도:** 기본 분포, CLT, 미분=선형근사 이해.

**모범 풀이:**
(a) Bernoulli($\theta$): $\theta^x(1-\theta)^{1-x}$, mean=$\theta$, var=$\theta(1-\theta)$. Gaussian($\mu,\sigma^2$): mean=$\mu$, var=$\sigma^2$.

(b) CLT: 독립+동일분포+유한분산→$\sqrt{n}(\bar{X}-\mu)/\sigma\to\mathcal{N}(0,1)$. 노이즈=미지 요인 합→가우시안.

(c) $f(x+\delta)\approx f(x)+f'(x)\delta$. 벡터: $\mathbf{f}(x+\delta)\approx\mathbf{f}(x)+J\delta$. **미분의 결과=Jacobian 행렬.**

**[채점]** 분포: 4 / CLT: 3 / 선형근사→Jacobian: 3

---

# Part B: 핵심 수식 유도 (6~10번, 50점)

## 문제 6 (10점)
**Q:** Coin, $n$ flips, $k$ heads, i.i.d. Bernoulli($\theta$). (a) Write likelihood. **State where i.i.d. used.** (3pts) (b) Derive $\hat\theta_{\text{MLE}}=k/n$. **Explain each step.** (5pts) (c) Verify maximum. (2pts)

**출제 의도:** MLE 유도에서 i.i.d., 로그, 미분=0의 이유를 평가.

**모범 풀이:**
(a) **[독립]**→곱 분해: $L(\theta)=\prod_i\theta^{X_i}(1-\theta)^{1-X_i}$. **[동일분포]**→같은 $\theta$: $=\theta^k(1-\theta)^{n-k}$.

(b) **[로그]** 왜? 단조증가→argmax 보존, 곱→합, 수치안정. $\ell=k\log\theta+(n-k)\log(1-\theta)$. **[미분]** $d\ell/d\theta=k/\theta-(n-k)/(1-\theta)$. **[=0]** 왜? concave→극대=최대. $k(1-\theta)=(n-k)\theta$→$\boxed{\hat\theta=k/n}$.

(c) $d^2\ell/d\theta^2=-k/\theta^2-(n-k)/(1-\theta)^2<0$→최대 ✓

**[채점]** 독립 명시: 1 / 동일분포: 1 / 우도: 1 / 로그 이유: 1 / 미분+0 이유: 3 / 답: 1 / 2차미분: 2

---

## 문제 7 (10점)
**Q:** $x_1,...,x_n\sim\mathcal{N}(\mu,\sigma^2)$ i.i.d. (a) Log-likelihood. (2pts) (b) $\hat\mu=\bar{x}$. (3pts) (c) $\hat\sigma^2=\frac{1}{n}\sum(x_i-\bar{x})^2$. (2pts) (d) Prove bias: $\mathbb{E}[\hat\sigma^2]=\frac{n-1}{n}\sigma^2$. **State where independence and identical distribution are used separately.** (3pts)

**출제 의도:** 가우시안 MLE + 편향 증명에서 i.i.d. 분리.

**모범 풀이:**
(a) $\ell=-\frac{n}{2}\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum(x_i-\mu)^2$

(b) $\partial\ell/\partial\mu=\frac{1}{\sigma^2}\sum(x_i-\mu)=0$→$\boxed{\hat\mu=\bar{x}}$

(c) $\partial\ell/\partial\sigma^2=0$→$\boxed{\hat\sigma^2=\frac{1}{n}\sum(x_i-\bar{x})^2}$

(d) $\sum(x_i-\bar{x})^2=\sum(x_i-\mu)^2-n(\bar{x}-\mu)^2$. $\mathbb{E}[\sum(x_i-\mu)^2]=n\sigma^2$ **[동일분포: 같은 σ²]**. $\text{Var}(\bar{x})=\sigma^2/n$ **[독립: 합의 분산=분산의 합]**. $\mathbb{E}[\hat\sigma^2]=\frac{n-1}{n}\sigma^2$.

**[채점]** 로그우도: 2 / μ: 3 / σ²: 2 / 편향(독립/동일분포 분리): 3

---

## 문제 8 (10점)
**Q:** $n=5$, $k=4$. Compute MAP for: (a) Uniform. (3pts) (b) Beta(2,2). (4pts) (c) $\theta^m(1-\theta)^m$, $m\to\infty$. (3pts)

**출제 의도:** 3가지 Prior에서 MAP 직접 계산 + Prior 영향 해석.

**모범 풀이:**
공통: 로그 사후→미분→0 (concave→극대=최대).

(a) $4\log\theta+\log(1-\theta)$. $4/\theta-1/(1-\theta)=0$. $\boxed{\hat\theta=4/5=\hat\theta_{\text{MLE}}}$. Uniform=무정보→MAP=MLE.

(b) $5\log\theta+2\log(1-\theta)$. $\boxed{\hat\theta=5/7\approx 0.714}$. Prior가 0.5 방향으로 끌어당김.

(c) $(4+m)/(5+2m)\to\boxed{1/2}$. 극강 prior→데이터 무시→prior mode.

**[채점]** (a): 3 / (b): 4 / (c): 3

---

## 문제 9 (10점)
**Q:** Softmax $p_i=\exp(z_i)/\sum_k\exp(z_k)$. (a) $i=j$: quotient rule. (4pts) (b) $i\neq j$. (3pts) (c) Matrix form + backprop relevance. (3pts)

**출제 의도:** Softmax Jacobian 유도 (교수님 4순위).

**모범 풀이:**
(a) Quotient rule: $\frac{\exp(z_i)s-\exp(z_i)^2}{s^2}=\boxed{p_i(1-p_i)}$

(b) 분자 $\exp(z_i)$ 상수→$\frac{-\exp(z_i)\exp(z_j)}{s^2}=\boxed{-p_ip_j}$

(c) $\boxed{\text{diag}(p)-pp^\top}$. CE+softmax→$\partial L/\partial z=p-y$ (canonical link 단순화).

**[채점]** i=j: 4 / i≠j: 3 / 행렬+역전파: 3

---

## 문제 10 (10점)
**Q:** (a) Prove $\sigma'=\sigma(1-\sigma)$. (4pts) (b) Binary CE gradient: show σ' cancellation. (4pts) (c) max σ'=1/4, $(1/4)^L$ bound. (2pts)

**출제 의도:** 시그모이드 미분, CE 그래디언트 소거, vanishing gradient.

**모범 풀이:**
(a) 체인룰: $\sigma=(1+e^{-z})^{-1}$, 외부 $-u^{-2}$, 내부 $-e^{-z}$→$e^{-z}/(1+e^{-z})^2=\sigma(1-\sigma)$.

(b) $\frac{\hat{y}-y}{\hat{y}(1-\hat{y})}\cdot\hat{y}(1-\hat{y})\cdot x$ → 분모 소거 → $\boxed{(\hat{y}-y)x}$.

(c) $\sigma'=p(1-p)\leq 1/4$. $L$층: $\prod\sigma'\leq(1/4)^L$→vanishing gradient. ReLU 해결.

**[채점]** σ' 증명: 4 / 소거: 4 / 1/4+vanishing: 2

---

# Part C: 응용 (11~15번, 50점)

## 문제 11 (10점)
**Q:** $y_i=f_\theta(x_i)+\epsilon_i$, $\epsilon\sim\mathcal{N}(0,\sigma^2)$ i.i.d. Derive NLL=MSE. **Every line: one sentence why.**

**출제 의도:** 교수님 1순위. 가우시안→MSE 완전 유도 + 매 줄 이유.

**모범 풀이:**
[L1] 가우시안 우도 작성 (ε이 가우시안이므로). [L2] i.i.d. 독립→곱 분해. [L3] 로그 (단조증가+곱→합). [L4] NLL 전개: $\frac{n}{2}\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}\sum(y_i-f(x_i))^2$. [L5] θ 무관 상수 제거 (argmin 불변). [L6] 양의 상수 제거. → $\boxed{\arg\min\text{NLL}=\arg\min\text{MSE}}$

**[채점]** 우도: 1 / i.i.d.: 2 / 로그 이유: 2 / 상수 근거: 2 / 전개: 1 / 연결: 2

---

## 문제 12 (10점)
**Q:** (a) $-\log p(\theta)$ for $\theta\sim\mathcal{N}(0,\sigma_p^2 I)$. (2pts) (b) MAP=MSE+λ||θ||², derive λ. (4pts) (c) λ 해석. (2pts) (d) L2→일반화 이유. (2pts)

**출제 의도:** Prior→L2 연결, λ=σ²/(nσ_p²).

**모범 풀이:**
(a) $\|\theta\|^2/(2\sigma_p^2)+C$. (b) NLL+Prior→$\frac{1}{2\sigma^2}\sum(\cdot)^2+\frac{\|\theta\|^2}{2\sigma_p^2}$→$\boxed{\lambda=\sigma^2/(n\sigma_p^2)}$. (c) σ_p²→0: λ→∞. n→∞: λ→0. σ²→∞: λ→∞. (d) ||θ|| 크면 민감→과적합. L2로 작게→부드러운 함수→일반화.

**[채점]** (a): 2 / (b): 4 / (c): 2 / (d): 2

---

## 문제 13 (10점)
**Q:** (a) H(p), H(p,q), KL. (3pts) (b) Prove KL≥0. (3pts) (c) minCE=minKL. (2pts) (d) Compute for P=(1/4,...), Q=(1/2,1/4,1/8,1/8). (2pts)

**출제 의도:** KL/CE/Entropy 관계 + 증명 + 계산.

**모범 풀이:**
(a) $H=-\sum p\log p$, $H(p,q)=-\sum p\log q$, $D_{KL}=H(p,q)-H(p)$. (b) $\ln t\leq t-1$→$-D_{KL}\leq 0$→$\boxed{D_{KL}\geq 0}$. (c) $H(\hat{p})$ 상수→$\arg\min H(\hat{p},p_\theta)=\arg\min D_{KL}$. (d) H(P)=2, H(P,Q)=2.25, KL=0.25 bits.

**[채점]** 정의: 3 / Gibbs: 3 / CE=KL: 2 / 계산: 2

---

## 문제 14 (10점)
**Q:** (a) CLT with 3 conditions. (3pts) (b) CLT→Gaussian→NLL→MSE chain. (4pts) (c) Laplacian noise→? Derive. (3pts)

**출제 의도:** CLT 근거 + 노이즈 분포→손실 대응.

**모범 풀이:**
(a) 독립+동일분포+유한분산→정규수렴. (b) CLT→ε가우시안→$p(y|x)\sim\mathcal{N}$→i.i.d.합→$-\log\exp(-z^2)=z^2$→MSE. (c) Laplacian→$-\log\exp(-|z|/b)=|z|/b$→$\sum|y_i-f(x_i)|$→$\boxed{\text{MAE(L1)}}$. Gaussian→L2, Laplacian→L1.

**[채점]** CLT: 3 / 체인: 4 / 라플라시안: 3

---

## 문제 15 (10점)
**Q:** (a) $A=[[1,2],[2,4]]$: rank, null, R-N. (3pts) (b) SVD, rank-k, Eckart-Young. (4pts) (c) PCA=eigenvectors via Lagrange. (3pts)

**출제 의도:** 선형대수 핵심 3가지 (Rank-Nullity, SVD, PCA).

**모범 풀이:**
(a) rank=1, $\mathscr{N}=\text{span}\{(-2,1)^\top\}$, $2=1+1$ ✓. (b) $A=U\Sigma V^\top$, $A_k=\sum_{i=1}^k\sigma_iu_iv_i^\top$, Eckart-Young: 최적 랭크-k 근사. (c) $\mathcal{L}=w^\top Sw-\lambda(w^\top w-1)$→$Sw=\lambda w$→최대 고유값의 고유벡터=1st PC.

**[채점]** R-N: 3 / SVD: 4 / PCA: 3

---

# Part D: 통합형 (16~20번, 50점)

## 문제 16 (10점)
**Q:** Complete 9-step chain: Bayes→MAP→log→i.i.d.→Gaussian→NLL→MSE→Prior→MSE+λ||θ||². One sentence per step.

**출제 의도:** 교수님 최우선 주제. 전체 프레임워크 통합.

**모범 풀이:**
[1] Bayes 정리. [2] p(D) 제거 (θ무관). [3] 로그 (단조+곱→합). [4] 부호반전. [5] i.i.d.분해 (독립→곱, 동일분포→같은θ). [6] Gaussian가정 (CLT). [7] NLL→MSE ($-\log\exp=z^2$, 상수제거). [8] Gaussian prior ($\|\theta\|^2/(2\sigma_p^2)$). [9] $\boxed{\text{MSE}+\lambda\|\theta\|^2, \lambda=\sigma^2/(n\sigma_p^2)}$.

**[채점]** 9단계 각 1점 + 최종 1점

---

## 문제 17 (10점)
**Q:** Medical test: sensitivity 99%, specificity 99%, prevalence 0.1%. (a) P(D|+). (5pts) (b) 2nd test positive: update using (a) as new prior. (5pts)

**출제 의도:** Bayesian belief update 사이클 + 놀라운 결과.

**모범 풀이:**
(a) $P(D|+)=\frac{0.99\times 0.001}{0.01098}\approx\boxed{9\%}$. Prior 극저→대부분 거짓양성.

(b) 새 Prior=0.09→$P(D|+_2)=\frac{0.99\times 0.09}{0.0981}\approx\boxed{91\%}$. Bayesian cycle: Prior→Data→Posterior→새Prior→... 증거 누적 시 prior 영향 감소.

**[채점]** (a) 전개+해석: 5 / (b) 갱신+사이클: 5

---

## 문제 18 (10점)
**Q:** (a) MLE with small data: 3 flips, 3 heads. Why bad? (3pts) (b) MAP with Beta(2,2), n=3, k=3. Derive. (4pts) (c) Professor says "DL = likelihood + prior, solved as loss, learned by NN." Unpack. (3pts)

**출제 의도:** MLE 한계→MAP 보완 + 교수 메시지 해석.

**모범 풀이:**
(a) $\hat\theta_{\text{MLE}}=1.0$. 3번으로 "항상 앞면" 비합리. MLE=적은 데이터에서 극단.

(b) $(3+1)\log\theta+(0+1)\log(1-\theta)$→$4/\theta-1/(1-\theta)=0$→$\boxed{0.8}$. MLE(1.0)보다 합리적.

(c) Likelihood→Loss(CE/MSE). Prior→Regularization(L2). NN=$h_\theta(x)$. SGD=argmin 수치해법.

**[채점]** MLE한계: 3 / MAP유도: 4 / 교수메시지: 3

---

## 문제 19 (10점)
**Q:** Side-by-side: Categorical→CE and Gaussian→MSE from NLL. (a) CE derivation. (4pts) (b) MSE derivation. (4pts) (c) CE gradient σ' cancellation vs MSE. (2pts)

**출제 의도:** 두 Loss의 통합 유도 + 그래디언트 비교.

**모범 풀이:**
(a) $p(y_i|x_i)=h(x_i)_{y_i}$. i.i.d.→곱→$-\log$→합: $-\sum e_{y_i}^\top\log h(x_i)$=CE.

(b) CLT→Gaussian→$-\log\exp(-z^2)=z^2$→상수제거→MSE.

(c) CE: $\hat{y}-y$ (σ' 소거). MSE: $(\hat{y}-y)\sigma'$. 포화 시 MSE 학습 정체, CE는 항상 교정.

**[채점]** CE: 4 / MSE: 4 / σ'비교: 2

---

## 문제 20 (10점)
**Q:** Ridge: MSE+λ||θ||², n=1000, σ²=1, λ=0.01. (a) MAP trace. (3pts) (b) σ_p²=? Interpret. (2pts) (c) Train 0.05, Test 0.30: diagnose, adjust λ. (3pts) (d) Transformer+100 samples fails. Why? (2pts)

**출제 의도:** 전체 프레임워크 실전 적용 + inductive bias.

**모범 풀이:**
(a) CLT→Gaussian→NLL=MSE→Gaussian Prior→MAP=Ridge.

(b) $\sigma_p^2=1/(1000\times 0.01)=\boxed{0.1}$. "가중치 ±0.63 안"이라는 적당한 믿음.

(c) 과적합. λ 증가=prior 강화=가중치 작게=일반화 개선.

(d) Transformer=최약bias. 100장으로 거대 가설공간 못 채움→과적합. CNN 추천.

**[채점]** MAP체인: 3 / σ_p²: 2 / 진단+λ: 3 / inductive bias: 2

---

# Final Summary

## 1. 가장 중요한 5개 주제
1. **MAP→NLL→MSE 9단계 흐름** (Q11, Q16)
2. **MAP 3가지 Prior** (Q8)
3. **i.i.d. 두 역할 분리** (Q2, Q6, Q7)
4. **MLE 한계+MAP 보완** (Q18)
5. **Inductive Bias↔아키텍처↔데이터 양** (Q4, Q20)

## 2. 가장 많이 틀릴 5개 포인트
1. i.i.d.에서 독립과 동일분포 **분리** 못 함
2. 로그 이유를 "관례"만 쓰고 **단조증가+곱→합+수치안정** 안 씀
3. 미분=0 이유 **concave→극대=최대** 안 씀
4. $-\log(\exp(-z^2))=z^2$ 상쇄 빠뜨림
5. 상수 제거 시 **"θ무관→argmin 불변"** 안 씀

## 3. 반드시 써봐야 하는 유도 5개
1. Bernoulli MLE: $\theta^k(1-\theta)^{n-k}\to k/n$
2. MAP 3 Cases: Uniform / Beta(2,2) / Strong $m\to\infty$
3. Gaussian→NLL→MSE (6줄, 매 줄 이유)
4. MAP=MSE+λ||θ||², λ=σ²/(nσ_p²)
5. Softmax Jacobian: $i=j$/$i\neq j$→diag(p)-pp^T

## 4. 답안 작성 팁 7개
1. **수식 한 줄마다 "왜" 한 문장**
2. **i.i.d.=곱 분해 줄에서 "독립이므로"**
3. **로그**: "단조증가→argmax, 곱→합"
4. **미분=0**: "concave→극대=최대"
5. **상수 제거**: "θ무관→argmin 불변"
6. **결과에 박스** + 해석 1문장
7. **개념 문제에도 수식**: "inductive bias=MAP의 p(H)"

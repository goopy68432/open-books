---
title: "최종추천 모범답안 - Codex 기준"
slug: codex
order: 5
---

---
title: 최종추천 모범답안 - Codex 기준
created: 2026-05-04
tags:
  - deep-learning
  - quiz
  - recommended-answer
cssclasses:
  - quiz-note
  - math-note
---

# 최종추천 모범답안 - Codex 기준

## 선정 결과

세 답안 비교 결과, 최종 추천 답안은 다음 파일을 기준으로 한다.

`/Users/jeongseongchae/dev/university/deep_learning/docs/QUIZ/codex/공식퀴즈_모범답안.md`

선정 이유:

- 13문항 전체 풀이가 가장 완전하다.
- 수식 유도 체인이 가장 상세하다.
- Obsidian 호환 Markdown 포맷이 가장 안정적이다.
- Quiz 9와 Quiz 11의 원본 불일치 지점을 명시적으로 분리했다.

주의:

- 기존 `/Users/jeongseongchae/dev/university/deep_learning/docs/QUIZ/report/최종추천_모범답안.md`는 이전 산출물로 보이며, 현재 검산 결론과 다르게 Claude를 추천한다.
- 현재 검산 기준의 최신 추천은 이 파일과 `/Users/jeongseongchae/dev/university/deep_learning/docs/QUIZ/report/모범답안_비교분석_추천.md`이다.

---

# 시험 제출용 최종 답안 요약

## Quiz 1. 선형변환 성질

$$
Ae_1=
\begin{bmatrix}1\\1\end{bmatrix},
\qquad
Ae_2=
\begin{bmatrix}1\\2\end{bmatrix}
$$

$$
Ae_1+Ae_2=
\begin{bmatrix}2\\3\end{bmatrix}
=
A(e_1+e_2)
$$

$$
2Ae_1=
\begin{bmatrix}2\\2\end{bmatrix}
=
A(2e_1)
$$

## Quiz 2. Image / Kernel

$$
R(A)=\mathbb R
$$

$$
R(A^T)=
\operatorname{span}
\left\{
\begin{bmatrix}1\\2\end{bmatrix}
\right\}
$$

$$
N(A)=
\operatorname{span}
\left\{
\begin{bmatrix}-2\\1\end{bmatrix}
\right\}
=
\operatorname{span}
\left\{
\begin{bmatrix}2\\-1\end{bmatrix}
\right\}
$$

$$
N(A^T)=\{0\}
$$

## Quiz 3. Softmax Jacobian

$$
\frac{\partial p_i}{\partial z_j}
=
\begin{cases}
p_i(1-p_i),&i=j\\
-p_ip_j,&i\ne j
\end{cases}
$$

또는

$$
\frac{\partial p_i}{\partial z_j}
=
p_i(\delta_{ij}-p_j)
$$

행렬형:

$$
\frac{\partial p}{\partial z}
=
\operatorname{diag}(p)-pp^T
$$

## Quiz 4. Bayes' theorem

조건부확률 정의:

$$
P(H\mid E)=\frac{P(H\cap E)}{P(E)}
$$

$$
P(E\mid H)=\frac{P(H\cap E)}{P(H)}
$$

따라서

$$
P(H\cap E)=P(E\mid H)P(H)
$$

대입하면

$$
P(H\mid E)
=
\frac{P(E\mid H)P(H)}{P(E)}
$$

## Quiz 5. Bernoulli MLE

$$
L(\theta)=\theta^k(1-\theta)^{n-k}
$$

$$
\ell(\theta)=k\log\theta+(n-k)\log(1-\theta)
$$

$$
\frac{d\ell}{d\theta}
=
\frac{k}{\theta}
-
\frac{n-k}{1-\theta}
$$

$$
\frac{k}{\theta}
=
\frac{n-k}{1-\theta}
\Rightarrow
\theta_{\mathrm{MLE}}
=
\frac{k}{n}
$$

## Quiz 6. MAP with Beta prior

$$
L(\theta)=\theta^{k+1}(1-\theta)^{n-k+1}
$$

$$
\ell(\theta)
=(k+1)\log\theta+(n-k+1)\log(1-\theta)
$$

$$
\frac{d\ell}{d\theta}
=
\frac{k+1}{\theta}
-
\frac{n-k+1}{1-\theta}
$$

$$
\theta_{\mathrm{MAP}}
=
\frac{k+1}{n+2}
$$

## Quiz 7. MAP 극한

Posterior:

$$
p(\theta\mid D)
\propto
\theta^{k+M}(1-\theta)^{n-k+M}
$$

MAP:

$$
\theta_{\mathrm{MAP}}
=
\frac{k+M}{n+2M}
$$

극한:

$$
\lim_{M\to\infty}
\frac{k+M}{n+2M}
=
\frac12
$$

## Quiz 8. KL Divergence와 MSE

주어진 Gaussian KL에 $\sigma_1^2=\sigma_2^2=1$을 대입하면

$$
D_{\mathrm{KL}}
\left(
\mathcal N(\mu_1,1)
\Vert
\mathcal N(\mu_2,1)
\right)
=
\frac{(\mu_1-\mu_2)^2}{2}
$$

회귀에서

$$
y\mid x\sim \mathcal N(\mu_\theta(x),1)
$$

로 가정하면 NLL은 상수항을 제외하고

$$
\frac12(y-\mu_\theta(x))^2
$$

이므로 MSE 최소화와 동치이다.

## Quiz 9. Triangular Prior MAP

관측:

$$
n=5,\qquad k=4
$$

Likelihood:

$$
L(\theta)=\theta^4(1-\theta)
$$

### 엄밀한 triangular density 기준

$$
m=2:\quad
\theta_{\mathrm{MAP}}=\frac23
$$

$$
m=6:\quad
\theta_{\mathrm{MAP}}=\frac12
$$

이유:

- $m=6$일 때 support는 $[1/3,2/3]$이다.
- 경계 $\theta=2/3$에서 triangular prior 값은 $0$이다.
- 따라서 literal density 기준으로 $\theta=2/3$은 posterior 최대점이 될 수 없다.
- 실제 내부 최대는 $\theta=1/2$이다.

### 강의식 hard support 해석 기준

만약 prior의 삼각형 높이를 무시하고 support constraint만 사용하면

$$
\theta_{\mathrm{MLE}}=\frac45
$$

가 support 밖이므로 constrained solution은

$$
\theta=\frac23
$$

이다.

시험 답안에는 다음 문장을 함께 쓰는 것을 추천한다.

> 주어진 triangular density를 엄밀하게 쓰면 $m=6$의 MAP는 $1/2$이다. 다만 prior를 support constraint만으로 해석하면 constrained MLE는 $2/3$이다.

## Quiz 10. Backpropagation

$$
L=-\log p_y
\Rightarrow
\frac{\partial L}{\partial p_y}
=
-\frac1{p_y}
$$

$$
p_y=e_y^Tp
\Rightarrow
\frac{\partial p_y}{\partial p}
=
e_y^T
$$

Softmax Jacobian:

$$
\frac{\partial p}{\partial z}
=
\operatorname{diag}(p)-pp^T
$$

Chain rule:

$$
\frac{\partial L}{\partial z}
=
\left(-\frac1{p_y}\right)
e_y^T
\left(\operatorname{diag}(p)-pp^T\right)
$$

$$
=
p-e_y
$$

## Quiz 11. 1D Convolution 직접 계산

정의:

$$
(w*x)_i
=
1\cdot x_i+2\cdot x_{i+1},
\qquad i=1,\dots,6
$$

권장 답:

$$
(0,1,2,3,4,5,6)^T
\mapsto
(2,5,8,11,14,17)^T
$$

$$
(1,1,1,1,1,1,1)^T
\mapsto
(3,3,3,3,3,3)^T
$$

$$
(1,2,3,4,5,6,7)^T
\mapsto
(5,8,11,14,17,20)^T
$$

$$
(0,2,4,6,8,10,12)^T
\mapsto
(4,10,16,22,28,34)^T
$$

주의:

원본 표는 세 번째 입력에 대해

$$
(4,7,10,13,16,19)^T
$$

를 제시하지만, 이는 위 정의식과 불일치한다. 정의식 기준 첫 항은

$$
1\cdot1+2\cdot2=5
$$

이다.

## Quiz 12. 1D Convolution Matrix

$$
A=
\begin{bmatrix}
1&2&0&0&0&0&0\\
0&1&2&0&0&0&0\\
0&0&1&2&0&0&0\\
0&0&0&1&2&0&0\\
0&0&0&0&1&2&0\\
0&0&0&0&0&1&2
\end{bmatrix}
$$

검산:

$$
A(0,1,2,3,4,5,6)^T
=
(2,5,8,11,14,17)^T
$$

## Quiz 13. 2D Convolution Matrix

입력 평탄화:

$$
\begin{bmatrix}
x_1&x_2&x_3\\
x_4&x_5&x_6\\
x_7&x_8&x_9
\end{bmatrix}
$$

Kernel:

$$
w=
\begin{bmatrix}
0&1\\
2&3
\end{bmatrix}
$$

출력식:

$$
y_1=0x_1+1x_2+2x_4+3x_5
$$

$$
y_2=0x_2+1x_3+2x_5+3x_6
$$

$$
y_3=0x_4+1x_5+2x_7+3x_8
$$

$$
y_4=0x_5+1x_6+2x_8+3x_9
$$

행렬:

$$
A=
\begin{bmatrix}
0&1&0&2&3&0&0&0&0\\
0&0&1&0&2&3&0&0&0\\
0&0&0&0&1&0&2&3&0\\
0&0&0&0&0&1&0&2&3
\end{bmatrix}
$$

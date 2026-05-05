---
title: "고급 1: 증명 사다리"
slug: 03-advanced-01-proof-ladder
order: 9
---

# 고급 1: 증명 사다리

증명은 외운 문장을 길게 쓰는 일이 아닙니다. "조건 → 사용할 정리 → 결론"을 정확히 연결하는 일입니다.

이 문서는 원본의 10대 증명을 난이도와 쓰임새에 따라 다시 배열합니다.

## 0단계: 증명 답안의 기본 형식

```text
정리: 무엇을 보일 것인가
조건: 이 정리가 적용되는 조건은 무엇인가
핵심 조작: 어떤 식을 변형하는가
결론: 그래서 무엇이 따라오는가
검산/의미: 왜 중요한가
```

## 1. Bayes 정리

정리:

$$
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}
$$

유도:

$$
p(\theta,D)=p(\theta|D)p(D)=p(D|\theta)p(\theta)
$$

따라서:

$$
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}
$$

MAP에서는 $p(D)$가 $\theta$에 대해 상수이므로:

$$
p(\theta|D)\propto p(D|\theta)p(\theta)
$$

시험 포인트: posterior는 likelihood와 prior의 곱에 비례한다.

## 2. Fermat 정리

정리:

> 함수 $f$가 내부점 $x^\*$에서 미분 가능하고 그 점이 극값이면 $f'(x^\*)=0$이다.

직관:

극대점에서는 왼쪽에서 올라오고 오른쪽에서 내려갑니다. 순간 기울기가 존재한다면 그 값은 0이어야 합니다.

주의:

- 경계점에는 바로 적용할 수 없습니다.
- 미분 불가능점에도 바로 적용할 수 없습니다.
- $f'(x)=0$은 후보 조건이지 충분조건이 아닙니다.

시험 포인트: MLE/MAP에서 미분=0을 쓸 때 조건을 말해야 합니다.

## 3. Gaussian 적분

정리:

$$
\int_{-\infty}^{\infty}e^{-x^2/2}\,dx=\sqrt{2\pi}
$$

증명:

$$
I=\int_{-\infty}^{\infty}e^{-x^2/2}\,dx
$$

$$
I^2=\int_{\mathbb{R}}\int_{\mathbb{R}}e^{-(x^2+y^2)/2}\,dx\,dy
$$

극좌표 변환:

$$
x=r\cos\phi,\quad y=r\sin\phi,\quad dxdy=rdrd\phi
$$

$$
I^2=\int_0^{2\pi}\int_0^\infty e^{-r^2/2}r\,dr\,d\phi
$$

$u=r^2/2$이면 $du=rdr$:

$$
\int_0^\infty e^{-r^2/2}r\,dr=\int_0^\infty e^{-u}\,du=1
$$

따라서:

$$
I^2=2\pi,\quad I=\sqrt{2\pi}
$$

시험 포인트: 제곱, 이중적분, 극좌표, 야코비안 $r$.

## 4. Softmax 자코비안

정리:

$$
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
$$

증명 핵심:

$$
p_i=\frac{e^{z_i}}{\sum_k e^{z_k}}
$$

$i=j$:

$$
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)
$$

$i\ne j$:

$$
\frac{\partial p_i}{\partial z_j}=-p_ip_j
$$

두 경우를 크로네커 델타로 합치면 정리입니다.

시험 포인트: 케이스 분리 후 행렬형 $J=\operatorname{diag}(p)-pp^T$.

## 5. Rank-Nullity 정리

정리:

선형사상 $T:V\to W$에 대해:

$$
\dim V=\operatorname{rank}(T)+\operatorname{nullity}(T)
$$

증명 구조:

1. kernel의 기저를 잡는다: $v_1,\ldots,v_k$.
2. 이를 $V$ 전체의 기저로 확장한다: $v_1,\ldots,v_k,u_1,\ldots,u_r$.
3. $T(v_i)=0$이므로 실제 image를 만드는 것은 $T(u_1),\ldots,T(u_r)$이다.
4. 이들이 image의 기저가 됨을 보인다.
5. 따라서 $\dim V=k+r=\operatorname{nullity}(T)+\operatorname{rank}(T)$.

시험 포인트: kernel 기저를 전체 기저로 확장하는 아이디어.

## 6. 서로 다른 고유값의 고유벡터는 선형독립

정리:

서로 다른 고유값 $\lambda_1,\ldots,\lambda_m$에 대응하는 고유벡터 $v_1,\ldots,v_m$는 선형독립이다.

핵심 증명 아이디어:

가장 짧은 선형종속 관계가 있다고 가정합니다.

$$
c_1v_1+\cdots+c_mv_m=0
$$

양변에 $A$를 곱합니다.

$$
c_1\lambda_1v_1+\cdots+c_m\lambda_mv_m=0
$$

첫 식에 $\lambda_m$을 곱한 뒤 빼면 $v_m$ 항이 사라집니다.

$$
c_1(\lambda_1-\lambda_m)v_1+\cdots+c_{m-1}(\lambda_{m-1}-\lambda_m)v_{m-1}=0
$$

더 짧은 선형종속 관계가 생겨 모순입니다. 따라서 선형독립입니다.

시험 포인트: "서로 다른 고유값"이 $\lambda_i-\lambda_m\ne0$을 보장합니다.

## 7. Cauchy-Schwarz 부등식

정리:

$$
|\langle x,y\rangle|\le \|x\|\|y\|
$$

증명:

모든 $t$에 대해:

$$
0\le \|x-ty\|^2=\|x\|^2-2t\langle x,y\rangle+t^2\|y\|^2
$$

이 2차식이 모든 $t$에서 0 이상이려면 판별식이 0 이하입니다.

$$
4\langle x,y\rangle^2-4\|x\|^2\|y\|^2\le0
$$

따라서:

$$
\langle x,y\rangle^2\le \|x\|^2\|y\|^2
$$

시험 포인트: "항상 0 이상인 2차식 → 판별식 ≤ 0".

## 8. Jensen 부등식

정리:

함수 $\phi$가 convex이면:

$$
\phi(E[X])\le E[\phi(X)]
$$

concave이면 부등호가 반대입니다.

시험에서 쓰는 형태:

$\log$는 concave이므로:

$$
E[\log X]\le \log E[X]
$$

시험 포인트: KL divergence 비음수성 증명에 연결됩니다.

## 9. KL divergence는 0 이상

정리:

$$
D_{KL}(P\|Q)=\sum_x p(x)\log\frac{p(x)}{q(x)}\ge0
$$

증명 핵심:

$$
D_{KL}(P\|Q)
=E_P\left[\log\frac{p(X)}{q(X)}\right]
=-E_P\left[\log\frac{q(X)}{p(X)}\right]
$$

$\log$는 concave이므로 Jensen:

$$
E_P\left[\log\frac{q(X)}{p(X)}\right]
\le
\log E_P\left[\frac{q(X)}{p(X)}\right]
$$

오른쪽:

$$
\log\sum_x p(x)\frac{q(x)}{p(x)}
=\log\sum_x q(x)=\log 1=0
$$

따라서:

$$
D_{KL}(P\|Q)\ge0
$$

시험 포인트: $E_P[q/p]=1$이 결정타입니다.

## 10. Spectral theorem의 시험용 골격

정리:

실대칭행렬 $A=A^T$는 직교행렬 $Q$와 대각행렬 $\Lambda$에 대해:

$$
A=Q\Lambda Q^T
$$

핵심 사실 1: 고유값은 실수입니다.

복소 고유벡터 $v$에 대해 $Av=\lambda v$라 하면:

$$
v^*Av=\lambda v^*v
$$

대칭행렬의 이차형식은 실수이므로 $\lambda$가 실수입니다.

핵심 사실 2: 서로 다른 고유값의 고유벡터는 직교합니다.

$$
Av=\lambda v,\quad Aw=\mu w
$$

$$
\lambda v^Tw=(Av)^Tw=v^TA^Tw=v^TAw=\mu v^Tw
$$

$$
(\lambda-\mu)v^Tw=0
$$

$\lambda\ne\mu$이면 $v^Tw=0$.

시험 포인트: 대칭성 $A^T=A$가 직교성을 만듭니다.

## 11. MLE 통일 7단계

정리라기보다 모든 추정 문제의 증명 템플릿입니다.

| 단계 | 내용 | 이유 |
|---|---|---|
| 1 | 모델 선언 | 무엇을 가정하는지 명확히 |
| 2 | 단일 pmf/pdf | 출발점 |
| 3 | i.i.d 곱 | 독립성 사용 |
| 4 | 우도 단순화 | 충분통계량 등장 |
| 5 | 로그 | 곱→합, argmax 보존 |
| 6 | 미분=0 | 페르마 정리 |
| 7 | 2계 미분/경계 | 최댓값 검증 |

마스터 기준: Bernoulli가 아니라 Poisson, Exponential, Gaussian에도 이 템플릿을 적용할 수 있어야 합니다.

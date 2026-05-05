---
title: "플래시카드"
slug: 90-drills-flashcards
order: 24
---

# 플래시카드

앞면만 보고 답을 말한 뒤 펼쳐보세요.

## 기호와 정의

Q1. $\arg\max_\theta f(\theta)$는 무엇인가?

A. $f(\theta)$의 최댓값 자체가 아니라, 최댓값을 만들게 하는 $\theta$이다.

Q2. $\propto$는 무엇을 생략한다는 뜻인가?

A. 모수와 무관한 정규화 상수 또는 비례상수를 생략한다는 뜻이다.

Q3. $X\sim N(0,1)$의 뜻은?

A. 확률변수 $X$가 평균 0, 분산 1인 정규분포를 따른다는 뜻이다.

Q4. $\delta_{ij}$는?

A. $i=j$이면 1, $i\ne j$이면 0인 크로네커 델타이다.

## MLE/MAP

Q5. i.i.d에서 independent는 어디에 쓰이는가?

A. 결합확률을 곱으로 쓰는 데 쓰인다.

Q6. i.i.d에서 identical은 어디에 쓰이는가?

A. 모든 관측치가 같은 pmf/pdf 형태와 같은 모수 $\theta$를 공유한다는 데 쓰인다.

Q7. Bernoulli pmf를 한 식으로 쓰면?

A. $p(y|\theta)=\theta^y(1-\theta)^{1-y}$.

Q8. Bernoulli likelihood는?

A. $L(\theta)=\theta^k(1-\theta)^{n-k}$, $k=\sum_i y_i$.

Q9. Bernoulli MLE는?

A. $\hat{\theta}_{MLE}=k/n$.

Q10. 로그를 취하는 이유 3개는?

A. 곱을 합으로 바꿈, 단조증가라 최대점 보존, 수치 안정성.

Q11. NLL이란?

A. Negative Log Likelihood, 즉 $-\log L$이다. 최대화 문제를 최소화 문제로 바꾼다.

Q12. MAP의 출발식은?

A. $p(\theta|D)\propto p(D|\theta)p(\theta)$.

Q13. symmetric prior $\theta^m(1-\theta)^m$의 MAP은?

A. $(k+m)/(n+2m)$.

Q14. asymmetric prior $\theta^m$의 MAP은?

A. $(k+m)/(n+m)$.

Q15. tent prior에서 꼭 확인해야 할 후보는?

A. 각 구간의 내부 임계점, 경계점, 미분 불가능점.

## 분포와 모멘트

Q16. $Uniform[a,b]$의 평균은?

A. $(a+b)/2$.

Q17. $Uniform[a,b]$의 분산은?

A. $(b-a)^2/12$.

Q18. 표준정규의 $E[X]$, $E[X^2]$, $E[X^3]$, $E[X^4]$는?

A. $0,1,0,3$.

Q19. 표준정규의 홀수 모멘트가 0인 이유는?

A. 피적분함수가 기함수이고 적분 구간이 대칭이기 때문이다.

Q20. 가우스 적분 증명의 핵심 트릭은?

A. 적분을 제곱해 2차원 이중적분으로 만들고 극좌표로 바꾼다.

## Softmax

Q21. softmax 정의는?

A. $p_i=e^{z_i}/\sum_k e^{z_k}$.

Q22. $i=j$일 때 softmax 미분은?

A. $\partial p_i/\partial z_i=p_i(1-p_i)$.

Q23. $i\ne j$일 때 softmax 미분은?

A. $\partial p_i/\partial z_j=-p_ip_j$.

Q24. softmax 자코비안 행렬형은?

A. $J=\operatorname{diag}(p)-pp^T$.

Q25. softmax+CE의 그래디언트는?

A. $\partial L/\partial z_j=p_j-y_j$.

## 증명

Q26. KL divergence가 0 이상인 핵심 이유는?

A. Jensen 부등식과 $E_P[q(X)/p(X)]=1$ 때문이다.

Q27. Cauchy-Schwarz 증명의 대표 출발식은?

A. $0\le\|x-ty\|^2$.

Q28. 대칭행렬에서 다른 고유값의 고유벡터가 직교하는 이유는?

A. $A^T=A$라서 $\lambda v^Tw=(Av)^Tw=v^TAw=\mu v^Tw$가 되고, $\lambda\ne\mu$이면 $v^Tw=0$이다.

Q29. Rank-Nullity 정리의 식은?

A. $\dim V=\operatorname{rank}(T)+\operatorname{nullity}(T)$.

Q30. Gaussian prior는 어떤 정규화로 이어지는가?

A. L2 정규화.

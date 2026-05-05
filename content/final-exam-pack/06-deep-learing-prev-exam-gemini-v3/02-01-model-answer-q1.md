---
title: "📝 [V3] Model Answer - Q1. 확률분포와 기댓값 (Probability Distributions)"
slug: 01-model-answer-q1
order: 2
---

# 📝 [V3] Model Answer - Q1. 확률분포와 기댓값 (Probability Distributions)

**[평가 기준]**
* 적분 기호 $\int$ 와 $\sum$ 의 의미를 단순히 수식으로만 적지 않고, 백엔드의 '루프 누적 합산' 개념과 연관지었는가?
* 변수 치환이나 적분 과정을 건너뛰지 않고 단계별로 명시했는가?

---

## (a) Uniform $U(a, b)$ 의 기댓값과 분산 증명

**[정의]** 
Uniform 분포는 트래픽이 $a$부터 $b$까지 완벽히 1/N로 균등하게 들어오는 분산 서버 패턴과 같습니다.
확률밀도함수(pdf): $f(x) = \frac{1}{b-a}$ (단, $a \le x \le b$, 그 외는 $0$)

**[기댓값 $E[X]$ 증명]**
$E[X]$는 전체 구간을 스캔(`for` 루프)하며 가중치 $f(x)$를 곱해 더하는 연산입니다.
$$E[X] = \int_{a}^{b} x f(x) dx = \int_{a}^{b} x \frac{1}{b-a} dx$$ 상수 $\frac{1}{b-a}$를 캐싱(적분 밖으로 분리)하고, $x$를 적분하면 $\frac{1}{2}x^2$이 됩니다.
$$E[X] = \frac{1}{b-a} \left[ \frac{1}{2}x^2 \right]_a^b = \frac{b^2 - a^2}{2(b-a)}$$
합차공식 $b^2 - a^2 = (b-a)(b+a)$로 약분하면:
$$\therefore E[X] = \frac{a+b}{2}$$

**[분산 $V[X]$ 증명]**
분산 공식 $V[X] = E[X^2] - (E[X])^2$ 을 사용합니다.
1. $E[X^2] = \int_{a}^{b} x^2 \frac{1}{b-a} dx = \frac{1}{b-a} \left[ \frac{1}{3}x^3 \right]_a^b = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}$
2. $V[X] = \frac{a^2 + ab + b^2}{3} - \left(\frac{a+b}{2}\right)^2 = \frac{4a^2 + 4ab + 4b^2 - 3(a^2 + 2ab + b^2)}{12}$
$$\therefore V[X] = \frac{a^2 - 2ab + b^2}{12} = \frac{(b-a)^2}{12}$$ 

---

## (b) 표준 정규 분포 $\mathcal{N}(0, 1)$ 의 홀수/짝수 모멘트 증명

**[정의]** 
표준 정규 분포의 pdf: $f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$

**1. 홀수차 모멘트 ($E[Z^{2n-1}]$)**
$$E[Z^{2n-1}] = \int_{-\infty}^{\infty} z^{2n-1} \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} dz$$
*   **증명 로직:** $f(-z) = f(z)$ 인 우함수에 기함수 $z^{2n-1}$ 을 곱했으므로, 피적분 함수는 전체적으로 기함수(Odd function)가 됩니다.
*   **결론:** 원점을 기준으로 좌우 트래픽이 완벽히 대칭 상쇄되므로, 적분값은 **$0$** 이 됩니다.

**2. 짝수차 모멘트 ($E[Z^{2n}]$)**
*   **증명 로직 (부분적분):** 재귀 함수(Recursive Call) 패턴을 사용하여 꼬리물기 연산을 합니다.
$$E[Z^2] = \int z \cdot (z e^{-\frac{z^2}{2}}) dz \propto 1 \cdot E[Z^0] = 1$$
$$E[Z^4] = 3 \cdot E[Z^2] = 3$$
$$E[Z^6] = 5 \cdot E[Z^4] = 15$$
*   **결론:** $\therefore E[Z^{2n}] = (2n-1)!! = (2n-1) \times (2n-3) \times \dots \times 1$

---

## (c) Poisson 분포의 기댓값과 분산 증명

**[정의]** 
단위 시간당 특정 에러(이벤트)가 발생할 횟수를 모니터링하는 이산형 데이터 모델.
확률질량함수(pmf): $P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!}$

**[기댓값 $E[X]$ 증명]**
$x=0$일 때 값은 0이므로 루프 인덱스를 $x=1$부터 시작합니다.
$$E[X] = \sum_{x=0}^{\infty} x \frac{\lambda^x e^{-\lambda}}{x!} = \sum_{x=1}^{\infty} \frac{\lambda^x e^{-\lambda}}{(x-1)!}$$
$\lambda$ 하나를 앞으로 빼내어 메모리를 최적화합니다 ($\lambda^x = \lambda \cdot \lambda^{x-1}$). 
$$E[X] = \lambda \sum_{x=1}^{\infty} \frac{\lambda^{x-1} e^{-\lambda}}{(x-1)!}$$
`Let y = x-1` 로 치환하면, $\sum_{y=0}^{\infty} P(Y=y) = 1$ (확률의 총합은 1)이 됩니다.
$$\therefore E[X] = \lambda \cdot 1 = \lambda$$

**[분산 $V[X]$ 증명 ($E[X(X-1)]$ 트릭 사용)]**
$$E[X(X-1)] = \sum_{x=2}^{\infty} x(x-1) \frac{\lambda^x e^{-\lambda}}{x!} = \lambda^2 \sum_{x=2}^{\infty} \frac{\lambda^{x-2} e^{-\lambda}}{(x-2)!} = \lambda^2 \cdot 1 = \lambda^2$$
$$E[X^2] = E[X(X-1)] + E[X] = \lambda^2 + \lambda$$
$$V[X] = E[X^2] - (E[X])^2 = (\lambda^2 + \lambda) - \lambda^2$$
$$\therefore V[X] = \lambda$$ 

---

## (d) 세 분포의 '평균'이 가지는 시스템적 의미 비교

*   **Uniform 평균 ($rac{a+b}{2}$):** 트래픽이 평탄(Flat)하게 들어올 때, 부하 분산 서버가 갖는 중간값(Median)으로서의 대푯값입니다.
*   **Normal 평균 ($\mu$):** 트래픽이 특정 시간대에 종 모양으로 집중될 때, 가장 확률이 높은 모드(Mode)이자 전체 데이터의 무게중심입니다.
*   **Poisson 평균 ($\lambda$):** 이산형 데이터(Count)의 단위 시간당 평균 발생 빈도(Rate)입니다. 이 $\lambda$ 값이 커질수록 중심한계정리(CLT)에 의해 점차 Normal 분포의 평균 패턴과 유사해집니다.
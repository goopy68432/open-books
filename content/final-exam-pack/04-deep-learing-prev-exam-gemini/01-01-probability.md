---
title: "[파트 1] 확률분포 (Probability Distributions) - 백엔드 시스템 관점의 완전 정복"
slug: 01-probability
order: 1
---

# [파트 1] 확률분포 (Probability Distributions) - 백엔드 시스템 관점의 완전 정복

🔥 **한 줄 결론**
👉 "확률분포는 데이터가 발생할 패턴(트래픽 분포)을 미리 정의해둔 '서버 부하 예측 모델'이며, 기댓값과 분산은 평균적인 연산 비용과 그 변동성(Risk)을 계산하는 과정이다."

💡 **쉽게 설명하면**
- **Uniform (균등분포):** 트래픽이 항상 똑같이 들어옴 (분산 서버)
- **Normal (정규분포):** 특정 시간대에 몰리고 나머지는 적음 (일반적인 서비스 패턴)
- **Poisson (푸아송분포):** 간헐적으로 튀는 에러나 이벤트 발생 횟수 (에러 로그 모니터링)

🧠 **남에게 설명한다면**
"확률분포는 데이터가 시스템에 들어오는 패턴을 수식으로 압축 저장한 것이고, 기댓값(적분)은 이 패턴을 순회하며 평균 부하를, 분산은 부하의 널뛰기(변동폭)를 계산하는 로직이야."

---

## 1. 확률분포의 본질 (Goal vs. Tool)
*   **Goal:** 불확실한 데이터의 패턴을 파라미터 몇 개로 압축하여 예측하기 위함.
*   **Tool:** 확률밀도함수(pdf)라는 '함수형 데이터 구조'를 사용해 데이터를 가중치로 변환하고 적분(`for` 루프)으로 집계.

---

## 2. 핵심 확률분포 3대장 완벽 해부 (평균과 분산 유도)

분산(Variance) 공식은 모든 분포에서 동일하게 **$V[X] = E[X^2] - (E[X])^2$** 를 사용합니다. (시스템 비유: '제곱의 평균'에서 '평균의 제곱'을 뺀 잔차)

### 2.1 Uniform Distribution (균등 분포) $U(a, b)$
*   **백엔드 비유:** 로드 밸런서가 트래픽을 $a$부터 $b$까지 완벽히 균등 분배.

**[기댓값 $E[X]$ 유도]**
$$E[X] = \int_a^b x \cdot \frac{1}{b-a} dx = \frac{1}{b-a} \left[ \frac{1}{2}x^2 \right]_a^b = \frac{b^2 - a^2}{2(b-a)} = \frac{a+b}{2}$$ 

**[분산 $V[X]$ 유도]** (기출 빈출)
1. $E[X^2]$ 계산: $\int_a^b x^2 \frac{1}{b-a} dx = \frac{1}{b-a} \left[ \frac{1}{3}x^3 \right]_a^b = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}$
2. 분산 계산: $V[X] = \frac{a^2 + ab + b^2}{3} - \left(\frac{a+b}{2}\right)^2$
   $= \frac{4a^2 + 4ab + 4b^2 - 3(a^2 + 2ab + b^2)}{12} = \frac{a^2 - 2ab + b^2}{12} = \frac{(b-a)^2}{12}$

---

### 2.2 Normal Distribution (정규 분포) $\mathcal{N}(\mu, \sigma^2)$
표준 정규 분포 $Z \sim \mathcal{N}(0,1)$로 정규화(Normalization)하여 연산을 단순화합니다.

**[기댓값 $E[Z]$ 와 분산 $V[Z]$ 유도]**
*   $E[Z] = \int_{-\infty}^{\infty} z \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} dz = 0$ (기함수 대칭 상쇄)
*   $V[Z] = E[Z^2] - 0 = \int_{-\infty}^{\infty} z^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} dz = 1$ (부분적분 활용)

#### 🔥 킬러 파트: $E[X^{2n-1}]$ 와 $E[X^{2n}]$ 계산 (기출 완벽 대비)
1. **홀수차 모멘트 ($E[X^{2n-1}]$) = 0**
   *   $z$의 홀수 제곱은 기함수(Odd function)이므로 적분 시 좌우 대칭 상쇄로 0.
2. **짝수차 모멘트 ($E[X^{2n}]$) = $(2n-1)!!$**
   *   부분적분 꼬리물기: $E[Z^4] = 3 \cdot E[Z^2] = 3$, $E[Z^6] = 5 \cdot 3 \cdot 1 = 15$

---

### 2.3 Poisson Distribution (푸아송 분포) $Pois(\lambda)$
*   **백엔드 비유:** 단위 시간당 500에러 발생 횟수. 평균과 분산이 모두 $\lambda$로 동일한 극강의 압축 효율 모델.

**[기댓값 $E[X]$ 유도]**
$$E[X] = \sum_{x=0}^{\infty} x \frac{\lambda^x e^{-\lambda}}{x!} = \sum_{x=1}^{\infty} \frac{\lambda^x e^{-\lambda}}{(x-1)!} = \lambda \sum_{x=1}^{\infty} \frac{\lambda^{x-1} e^{-\lambda}}{(x-1)!} = \lambda \cdot 1 = \lambda$$ 

**[분산 $V[X]$ 유도]** (기출 빈출: $E[X(X-1)]$ 트릭 사용)
1. $E[X(X-1)] = \sum_{x=2}^{\infty} x(x-1) \frac{\lambda^x e^{-\lambda}}{x!} = \lambda^2 \sum_{x=2}^{\infty} \frac{\lambda^{x-2} e^{-\lambda}}{(x-2)!} = \lambda^2$
2. $E[X^2] = E[X(X-1)] + E[X] = \lambda^2 + \lambda$
3. $V[X] = E[X^2] - (E[X])^2 = (\lambda^2 + \lambda) - \lambda^2 = \lambda$

---

## 3. 여러 분포 평균의 시스템적 비교 (알고리즘 간 비교)

시험에서는 각 분포의 특징을 묻는 비교 문제가 나옵니다.
*   **Uniform 평균 ($\frac{a+b}{2}$):** 시스템 부하가 평탄할 때 중간값을 취함.
*   **Normal 평균 ($\mu$):** 가장 확률이 높은 모드(Mode)이자 데이터의 무게중심.
*   **Poisson 평균 ($\lambda$):** 이산형 데이터(Count)의 발생 빈도(Rate). $\lambda$가 커지면 중심한계정리에 의해 점차 Normal 분포의 평균과 유사한 모양(종 모양)으로 근사됨.

---

⚡ **핵심 킬러 요약**
1. **분산 유도 공식:** $V[X] = E[X^2] - (E[X])^2$ 무조건 암기.
2. **Normal:** $N(0,1)$에서 홀수 거듭제곱은 대칭 상쇄(0), 짝수는 재귀 곱셈($(2n-1)!!$).
3. **Poisson:** $E[X(X-1)]$ 트릭으로 계산하면 $V[X] = \lambda$ 가 됨.

📌 **교수님 강조 주의사항 (시험 TIP)**
*   **증명 과정 필수:** $\int$나 $\sum$ 전개 시, 변수 치환(Substitution) 과정을 건너뛰지 말고 "Let $y = x-1$"처럼 명시하세요. 짧게 답만 쓰면 부분 점수 없습니다.
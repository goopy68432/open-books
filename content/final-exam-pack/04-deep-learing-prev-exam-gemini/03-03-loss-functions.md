---
title: "[파트 3] 손실함수(Loss Function)의 연결 관계 - 백엔드 관점의 에러 측정 아키텍처"
slug: 03-loss-functions
order: 3
---

# [파트 3] 손실함수(Loss Function)의 연결 관계 - 백엔드 관점의 에러 측정 아키텍처

🔥 **한 줄 결론**
👉 "모든 Loss 함수는 하늘에서 떨어진 게 아니라 '데이터 분포 가정(Gaussian 등) $\rightarrow$ MLE $\rightarrow$ NLL $\rightarrow$ KL' 이라는 단일 아키텍처(하나의 파이프라인)에서 파생된 것이다."

---

## 1. 전체 관계 (MSE ↔ MLE ↔ NLL ↔ KL) 완벽 설명

시험에서 "전체 관계를 설명하고 증명하라"는 문제가 출제됩니다. 다음의 흐름도를 서술하세요.

**[논리 체인 아키텍처]**
1. **MLE (최대우도추정):** "데이터를 가장 잘 설명하는 모델 파라미터 $\theta$를 찾자!"
2. **NLL (음의 로그우도):** 곱셈을 덧셈으로 바꾸고($\log$), 최적화 함수의 규격(Minimize)에 맞게 부호를 뒤집자($-$). $\Rightarrow$ **[MLE = NLL 최소화]**
3. **KL Divergence:** 정답 분포 $P$와 예측 분포 $Q$의 차이는 $-H(P) + H(P,Q)$ 인데, $-H(P)$는 상수이므로 버리면 Cross-Entropy(NLL)만 남는다. $\Rightarrow$ **[KL 최소화 = NLL 최소화]**
4. **MSE (평균제곱오차):** 에러 분포가 가우시안(Normal)이라고 가정하고 NLL 수식을 전개한 뒤 상수를 제거하면 결국 제곱 오차만 남는다. $\Rightarrow$ **[Gaussian NLL = MSE]**

---

## 2. NLL과 MSE의 완벽한 상속 관계 (NLL ↔ MSE 증명)

**[가정: 에러가 정규분포 $\mathcal{N}(0, \sigma^2)$를 따른다]**
$$y \sim \mathcal{N}(\hat{y}, \sigma^2)$$

$$NLL(\theta) = - \sum_{i=1}^N \log P(y_i | x_i, \theta)$$
$$= - \sum_{i=1}^N \log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \hat{y}_i)^2}{2\sigma^2} \right) \right)$$
$$= - \sum_{i=1}^N \left[ \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) - \frac{(y_i - \hat{y}_i)^2}{2\sigma^2} \right]$$

*   **상수 제거 (Garbage Collection):** $\log(\dots)$ 부분은 $\theta$와 무관한 상수이므로 미분 시 0이 되어 삭제. 분모 $2\sigma^2$ 역수 상수 무시.
*   **결론:** $\propto \sum_{i=1}^N (y_i - \hat{y}_i)^2 = MSE$

---

## 3. NLL과 KL Divergence의 관계 (NLL ↔ KL 증명)

**[수식 해체: KL을 NLL로 분해]**
$$KL(P || Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$$
$$= \sum P(x) \log P(x) - \sum P(x) \log Q(x)$$
$$= - H(P) + H(P, Q)$$

| 기호 / 항 | 백엔드/시스템 비유 (설명) |
|---|---|
| $- H(P)$ | 정답의 엔트로피. 우리 시스템이 아무리 노력해도 줄일 수 없는 **고정 상수값(상수 에러)**. 미분 시 제거됨. |
| $H(P, Q)$ | Cross-Entropy (NLL). 정답 $P$를 예측 $Q$로 처리할 때 드는 연산 비용. 딥러닝이 실제로 최소화하는 **Loss 함수**. |

*   **결론:** $KL(P \|\| Q)$를 최소화하는 것은 $Q$ 파라미터에 대한 미분을 수행할 때 상수 $-H(P)$가 날아가므로, 결국 $H(P,Q)$ 즉, NLL을 최소화하는 것과 완벽히 동일한 동작입니다.

---

📌 **핵심 정리 (시험 서술 포인트)**
*   MSE, Cross-Entropy 등은 독립적인 공식이 아닙니다. **분포의 가정(Gaussian, Bernoulli 등)만 다를 뿐, 백엔드 로직은 동일한 NLL 엔진**을 사용한다는 점을 강조해야 만점을 받습니다.
---
title: "🎯 딥러닝 중간고사 '논리 체인' 최종 리허설 (교수님 스타일 완벽 반영)"
slug: 09-final-exam-checklist
order: 8
---

# 🎯 딥러닝 중간고사 '논리 체인' 최종 리허설 (교수님 스타일 완벽 반영)

🔥 **한 줄 결론**
👉 "시험의 핵심은 '계산 결과'가 아니라, **[독립성 가정(IID) → 로그 변환(Underflow 방지) → 목적 함수(Loss) 도출 → 미분(최적화)]**로 이어지는 논리적 근거를 영어로 서술하는 것이다."

---

## 1. 수식 전개 시 반드시 포함해야 할 '키워드 주석' (채점 포인트)

교수님은 "답만 쓰면 0점"이라고 여러 번 강조하셨습니다. 수식 사이사이에 다음 문구들을 적절히 섞어 쓰세요.

| 내 행동 | 영어 서술 예시 (시험용) | 교수님 의도 (백엔드 관점) |
|---|---|---|
| **독립성 사용** | "Assume IID (Independent and Identically Distributed) data." | 데이터 간 간섭이 없으므로 확률을 곱셈(`reduce-multiply`)할 수 있음. |
| **로그 변환** | "Take log to avoid numerical underflow and convert product to sum." | 컴퓨터 연산의 한계를 고려한 시스템 형변환. |
| **NLL 변환** | "Minimizing NLL is equivalent to maximizing Likelihood." | 최적화 프레임워크(Minimizer) 규격에 맞추기 위해 부호 반전. |
| **상수 제거** | "Remove constants that do not depend on $\theta$ during optimization." | 미분하면 0이 되거나 최적점에 영향을 주지 않는 불필요한 연산 제거. |

---

## 2. 킬러 문항 1순위: MAP vs MLE '서술형' 비교

**Q: "데이터가 매우 적은 상황($n=3, k=3$)에서 MLE보다 MAP가 유리한 이유를 설명하시오."**

*   **Step 1 (Belief Update):** 학습은 본질적으로 Prior(내 지식)를 Posterior(새로운 믿음)로 업데이트하는 과정입니다.
*   **Step 2 (Data Reliance):** MLE는 데이터에만 100% 의존하므로, 적은 데이터에서는 사기 동전($\theta=1$)이라는 극단적인 결론(Overfitting)을 냅니다.
*   **Step 3 (Knowledge Injection):** MAP는 Prior Knowledge(예: 동전은 보통 0.5다)라는 안전장치를 주입하여 데이터가 튀더라도 합리적인 예측($\theta=0.8$)을 유지하게 합니다 (Regularization).

---

## 3. 기호 해체: 딥러닝 수식의 '진짜 의미'

### [야코비안 행렬 (Jacobian Matrix)]
$$\mathbf{J} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$
- **해체:** 출력 벡터 $\mathbf{y}$의 모든 성분이 입력 벡터 $\mathbf{x}$의 모든 성분에 대해 어떻게 변하는지 기록한 **'변화율 로그 테이블'**.
- **백엔드 비유:** 마이크로서비스 간의 트래픽 의존도 맵과 같습니다. $x_j$ 서비스 부하가 1 늘 때 $y_i$ 서비스 부하가 얼마나 늘어나는지 전수 조사한 표입니다.

### [소프트맥스 (Softmax)]
$$p_i = \frac{e^{z_i}}{\sum e^{z_k}}$$
- **해체:** 어떤 값이든 들어와도(Any Float) 무조건 0~1 사이의 확률값으로 변환해주고 총합을 1로 맞춰주는 **'출력 인터페이스 규격화기'**.
- **킬러 포인트:** 미분 시 $i=j$ (자기 자신)일 때는 양수, $i \neq j$ (타인)일 때는 음수 그래디언트가 나옵니다. (한 명의 확률이 오르면 남의 확률은 반드시 깎이는 **Zero-sum 연산 로직**)

---

## 4. 학습률($\eta$)과 시스템 안정성

**Q: "왜 $\eta < 2 / \lambda_{max}$ 인가?"**

1.  **Hessian ($A$):** Loss 함수의 곡률(가파름) 정보를 담은 DB.
2.  **Eigenvalue ($\lambda$):** 시스템이 특정 방향으로 가속할 수 있는 '증폭 배율'.
3.  **Stability:** 한 번의 업데이트 강도($\eta$)가 시스템의 최대 가속도($\lambda_{max}$)를 곱했을 때 '브레이크' 범위($<2$)를 넘어서면, 에러($e_t$)는 무한히 커져서 서버가 터집니다( 발산).

---

## 5. 교수님 퀴즈 기반 마지막 체크리스트

- [ ] $\int_a^b x \frac{1}{b-a} dx$ 를 적분 공식 없이 '넓이'와 '가중치' 개념으로 설명할 수 있는가?
- [ ] Jensen 부등식을 사용하여 $\log \mathbb{E}[X] \ge \mathbb{E}[\log X]$ 임을 직관적으로(기울기가 완만해지는 로직으로) 설명할 수 있는가?
- [ ] 3번 던져서 3번 앞면이 나왔을 때, MAP 결과식 $\frac{k+1}{n+2}$ 유도 과정을 암기하지 않고 'Prior 항 더하기' 로직으로 유추 가능한가?
- [ ] Softmax Jacobian 미분을 몫의 미분법(Quotient Rule)을 사용하여 5분 안에 전개 가능한가?

**Task Executor 모드로 전환하여 준비한 자료들을 최종 검토하세요.**
**성채 님의 중간고사 만점을 기원합니다!**

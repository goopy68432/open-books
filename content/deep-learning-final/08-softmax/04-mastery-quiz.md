---
title: "04. 마스터리 퀴즈 — Softmax 미분"
slug: mastery-quiz
order: 4
---

# 04. 마스터리 퀴즈 — Softmax 미분

## 문제 1: log softmax 미분

$\partial \log p_i/\partial z_j = ?$

<details><summary>풀이</summary>

체인 룰:
$$\frac{\partial \log p_i}{\partial z_j} = \frac{1}{p_i} \cdot \frac{\partial p_i}{\partial z_j} = \frac{p_i(\delta_{ij} - p_j)}{p_i} = \delta_{ij} - p_j$$

**놀라운 단순성!** $p_i$가 약분.
</details>

## 문제 2: c=2 (이진 분류 → sigmoid)

c=2일 때 softmax는 sigmoid와 동치임을 보여라.

<details><summary>풀이</summary>

$p_1 = e^{z_1}/(e^{z_1} + e^{z_2})$. 분자/분모를 $e^{z_2}$로 나눔:
$$p_1 = \frac{e^{z_1 - z_2}}{e^{z_1 - z_2} + 1} = \sigma(z_1 - z_2)$$

여기서 $\sigma$는 sigmoid. $z = z_1 - z_2$로 두면 일변수 sigmoid.
</details>

## 문제 3: Cross Entropy + Softmax 그래디언트

$L = -\sum_i y_i \log p_i$ (y는 one-hot)일 때 $\partial L/\partial z_j$.

<details><summary>풀이</summary>

$$\frac{\partial L}{\partial z_j} = -\sum_i y_i \frac{\partial \log p_i}{\partial z_j} = -\sum_i y_i (\delta_{ij} - p_j)$$
$$= -y_j + p_j \sum_i y_i = p_j - y_j$$

(one-hot $\sum y_i = 1$)

이 단순한 형태가 신경망 학습의 핵심.
</details>

## 문제 4: 자코비안 J의 성질

$J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T$가 양반정치(positive semidefinite)임을 보여라.

<details><summary>풀이 스케치</summary>

임의의 $\mathbf{v}$에 대해:
$$\mathbf{v}^T J \mathbf{v} = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2 = E_p[V^2] - (E_p[V])^2 = \text{Var}_p[V] \geq 0$$

여기서 $V$는 $p_i$ 확률로 $v_i$ 값 갖는 확률변수. **분산 ≥ 0**으로 PSD.
</details>

## 문제 5: temperature 효과

$\text{softmax}(\mathbf{z}/T)$의 자코비안은?

<details><summary>풀이</summary>

체인 룰: $\partial p_i/\partial z_j = (1/T)\partial p_i/\partial(z_j/T)$.

따라서 $J' = (1/T)\, [\text{diag}(\mathbf{p}) - \mathbf{pp}^T]$.

T가 크면 자코비안이 작아짐 → 출력이 균일분포에 가까워짐.
</details>

---

## 시험 직전 체크
- [ ] softmax 정의 + S 분리 표기 즉답?
- [ ] 두 케이스 (i=j, i≠j) 분리 인식?
- [ ] 몫 규칙 적용 정확?
- [ ] 크로네커 델타 통합 표기?
- [ ] $J = \text{diag}(\mathbf{p}) - \mathbf{pp}^T$ 즉답?
- [ ] CE + softmax 그래디언트 = $\mathbf{p} - \mathbf{y}$ 외웠는가?

---

기출 8문제 챕터 모두 완료! [`../09-killer-chains/`](../09-killer-chains/)으로 진행.

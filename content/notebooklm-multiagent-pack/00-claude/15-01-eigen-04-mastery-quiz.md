---
title: "04. 마스터리 퀴즈 — 고유값·고유벡터"
slug: 01-eigen-04-mastery-quiz
order: 15
---

# 04. 마스터리 퀴즈 — 고유값·고유벡터

> 5문제 모두 손으로 풀고 답을 맞춰보세요. 4개 이상 정답이면 이 챕터 졸업.

---

## 문제 1 (기본 변형)

$$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

A의 고유값과 고유벡터를 구하라.

<details>
<summary>풀이 보기</summary>

특성방정식:
$$\det(A - \lambda I) = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda-1)(\lambda-3) = 0$$

→ $\lambda_1 = 3$, $\lambda_2 = 1$.

**λ = 3:** $(A - 3I) = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}$, $v_1 = v_2$ → $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1, 1)^T$.

**λ = 1:** $(A - I) = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, $v_2 = -v_1$ → $\mathbf{v}_2 = \frac{1}{\sqrt{2}}(1, -1)^T$.

**관찰:** 기출 1번 A에 단위행렬 2I를 더한 형태. 고유벡터는 동일, 고유값만 +2.

</details>

---

## 문제 2 (대각행렬)

$$A = \begin{pmatrix} 5 & 0 \\ 0 & 7 \end{pmatrix}$$

<details>
<summary>풀이 보기</summary>

대각행렬은 즉시:
- 고유값 = 대각원소: $\lambda_1 = 5, \lambda_2 = 7$
- 고유벡터 = 표준기저: $\mathbf{e}_1 = (1,0)^T$, $\mathbf{e}_2 = (0,1)^T$

**검증:**
- $A \mathbf{e}_1 = (5, 0)^T = 5 \mathbf{e}_1$ ✓
- $A \mathbf{e}_2 = (0, 7)^T = 7 \mathbf{e}_2$ ✓

**시험 팁:** 대각행렬이면 즉답.

</details>

---

## 문제 3 (중근)

$$A = \begin{pmatrix} 3 & 1 \\ 0 & 3 \end{pmatrix}$$

<details>
<summary>풀이 보기</summary>

특성방정식: $\det(A - \lambda I) = (3-\lambda)^2 = 0$

→ $\lambda = 3$ (중근, 대수적 중복도 2)

**고유벡터:** $(A - 3I)\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = 0$ → $v_2 = 0$.

→ $\mathbf{v} = (1, 0)^T$ 만 (기하적 중복도 1).

**관찰:** 대수적 중복도(2) ≠ 기하적 중복도(1). 이런 행렬은 **대각화 불가능**.

</details>

---

## 문제 4 (회전)

$$A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$ (90도 회전)

<details>
<summary>풀이 보기</summary>

특성방정식: $\det(A - \lambda I) = \lambda^2 + 1 = 0$

→ $\lambda = \pm i$ (복소수)

**관찰:** 실수 고유값 없음. 직관: 회전은 어떤 실수 벡터의 방향도 보존하지 않음.

**복소 고유벡터** (시험 범위 밖이지만):
- $\lambda = i$: $\mathbf{v} = (1, -i)^T$ 등 (복소수 성분)

</details>

---

## 문제 5 (정의로 증명 강조)

다음을 증명하라: $\mathbf{v}_1, \mathbf{v}_2$가 서로 다른 고유값 $\lambda_1, \lambda_2$의 고유벡터이면, $\mathbf{v}_1$과 $\mathbf{v}_2$는 **선형독립**이다.

<details>
<summary>풀이 보기</summary>

**증명 (모순법):**
$\mathbf{v}_1$과 $\mathbf{v}_2$가 선형종속이라 가정. 즉, $\mathbf{v}_2 = c \mathbf{v}_1$ 어떤 $c \neq 0$.

양변에 $A$ 곱:
$$A \mathbf{v}_2 = c A \mathbf{v}_1$$
$$\lambda_2 \mathbf{v}_2 = c \lambda_1 \mathbf{v}_1$$
$$\lambda_2 (c \mathbf{v}_1) = c \lambda_1 \mathbf{v}_1$$
$$c (\lambda_2 - \lambda_1) \mathbf{v}_1 = \mathbf{0}$$

$\mathbf{v}_1 \neq 0$이고 $c \neq 0$이므로 $\lambda_1 = \lambda_2$. 그러나 $\lambda_1 \neq \lambda_2$ 가정에 모순. ∎

**시험 출제 가능성:** 매우 높음. (`10-ten-proofs/02-eigen-independence.md`로 별도 정리됨)

</details>

---

## 자가 평가

| 점수 | 평가 |
|-----|-----|
| 5/5 | 마스터. 다음 챕터로. |
| 4/5 | 합격. 틀린 문제만 복습. |
| 3/5 | `02-derivation.md`로 돌아가서 다시. |
| 0~2 | `00-prerequisites/07-determinant.md`부터 다시. |

---

## 시험 직전 체크 (1분)

기출 1번 A=[[0,1],[1,0]]:
- [ ] 정의 첫 줄 적었나? ($A\mathbf{v} = \lambda \mathbf{v}$)
- [ ] $\det = 0$ 이유 한 문장 썼나?
- [ ] 특성방정식 $\lambda^2 - 1 = 0$
- [ ] 두 고유벡터 모두 구했나?
- [ ] 둘 다 정의로 검증했나?
- [ ] 직관 한 줄 추가했나? (보너스)

---

## 챕터 완료

이제 [`../02-gaussian/00-overview.md`](../02-gaussian/00-overview.md) — 기출 2번(정규분포 모멘트)으로 진행.

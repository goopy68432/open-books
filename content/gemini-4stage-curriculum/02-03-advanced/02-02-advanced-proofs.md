---
title: "Level 3: Advanced - 10대 핵심 증명 (Advanced Proofs)"
slug: 02-advanced-proofs
order: 2
---

# Level 3: Advanced - 10대 핵심 증명 (Advanced Proofs)

> **학습 목표:** 출제 교수가 유독 사랑하는, "결과가 아닌 과정을 재현할 수 있는가?"를 묻는 10대 수학적 증명들을 정복합니다.

---

## 1. 가우스 적분 (Gaussian Integral)
**증명 목표:** $I = \int_{-\infty}^{\infty} \exp(-\frac{x^2}{2}) dx = \sqrt{2\pi}$

* **왜 중요한가?** 정규분포(Gaussian)의 면적이 1이 되게 만드는 분모 $\sqrt{2\pi}$ 가 어디서 튀어나왔는지 증명하는 딥러닝 수학의 상징적인 문제입니다.
* **증명 아이디어 (극좌표 변환):**
  $I$를 그냥 구하기는 불가능합니다. 천재적인 트릭으로 $I^2$를 구합니다.
  1. $I^2 = \int \exp(-\frac{x^2}{2}) dx \int \exp(-\frac{y^2}{2}) dy = \iint \exp(-\frac{x^2+y^2}{2}) dxdy$
  2. $x, y$ 직교좌표계를 $r, \theta$ 원형(극)좌표계로 바꿉니다. $x^2+y^2 = r^2$ 이고, 면적 요소 $dxdy = r dr d\theta$ 가 됩니다. (야코비안 $r$이 튀어나옴!)
  3. $\int_0^{2\pi} \int_0^\infty \exp(-\frac{r^2}{2}) r dr d\theta$
  4. 안쪽 적분: $u = r^2/2$ 로 치환하면 $du = r dr$. $\int_0^\infty \exp(-u) du = 1$.
  5. 바깥쪽 적분: $\int_0^{2\pi} 1 d\theta = 2\pi$.
  6. 따라서 $I^2 = 2\pi$ 이고, $I = \sqrt{2\pi}$ 입니다.

---

## 2. 서로 다른 고유값 → 서로 독립인 고유벡터
**증명 목표:** $\lambda_1 \neq \lambda_2$ 이면 $v_1, v_2$ 는 선형독립이다.

* **왜 중요한가?** 데이터(행렬)를 여러 개의 독립적인 축(고유벡터)으로 깔끔하게 쪼갤 수 있다는 PCA(주성분 분석)의 핵심 토대입니다.
* **증명 아이디어 (귀류법):**
  1. 두 벡터가 종속이라고 가정해봅시다. 즉, $c_1 v_1 + c_2 v_2 = 0$ ($c_1, c_2 \neq 0$)
  2. 양변에 행렬 $A$를 곱합니다: $c_1 \lambda_1 v_1 + c_2 \lambda_2 v_2 = 0$ (식 A)
  3. 원래 식에 $\lambda_2$ 를 곱합니다: $c_1 \lambda_2 v_1 + c_2 \lambda_2 v_2 = 0$ (식 B)
  4. (식 A) - (식 B)를 합니다: $c_1 (\lambda_1 - \lambda_2) v_1 = 0$
  5. 고유벡터 $v_1 \neq 0$ 이고, 고유값이 다르므로 $(\lambda_1 - \lambda_2) \neq 0$ 입니다. 따라서 무조건 $c_1 = 0$ 이어야 합니다. 이는 가정을 위배하므로, 두 벡터는 독립입니다.

---

## 3. Rank-Nullity Theorem (차원 정리)
**증명 목표:** $\dim(\text{Range}) + \dim(\text{Null}) = n$ (입력 차원)

* **왜 중요한가?** 정보가 행렬을 통과할 때, 살아남은 정보(Range)와 0으로 찌그러져 사라진 정보(Null)의 합은 항상 처음 입력된 정보량(n)과 같다는 '정보 보존의 법칙'입니다.
* **증명 아이디어:** Null 공간의 기저(Basis)를 먼저 잡고, 이를 전체 공간으로 확장한 뒤 행렬을 통과시켜서 Range의 차원을 세어 증명합니다.

---

## 4. 코시-슈바르츠 부등식 (Cauchy-Schwarz Inequality)
**증명 목표:** $|\langle u, v \rangle| \le \|u\| \|v\|$

* **왜 중요한가?** 코사인 유사도(Cosine Similarity)가 무조건 -1에서 1 사이에 존재하게 만드는 마법의 부등식입니다. 딥러닝에서 두 벡터(단어, 이미지)가 얼마나 비슷한지 측정하는 핵심 근거입니다.
* **증명 아이디어:** 어떤 벡터의 길이의 제곱은 항상 0 이상임을 이용합니다. $\|u - tv\|^2 \ge 0$ 이 식을 전개하여 2차 부등식의 판별식 $D \le 0$ 을 적용하면 바로 유도됩니다.

---

## 5. KL Divergence의 비음수성 (KL $\ge 0$)
**증명 목표:** $KL(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx \ge 0$

* **왜 중요한가?** 두 확률분포 사이의 '거리(정확히는 정보량 차이)'를 재는 KL Divergence가 항상 0 이상임을 증명합니다. 최적화 과정의 손실 함수로 쓸 수 있는 자격을 증명하는 것입니다.
* **증명 아이디어 (Jensen's Inequality):**
  * $\log$ 함수는 위로 볼록(Concave)한 함수입니다. 
  * 젠센 부등식에 의해 $\log(E[X]) \ge E[\log(X)]$ 가 성립합니다.
  * $-KL = \int p \log(q/p) = E_p[\log(q/p)] \le \log(E_p[q/p]) = \log(\int p \cdot (q/p)) = \log(\int q) = \log(1) = 0$
  * 즉 $-KL \le 0$ 이므로, $KL \ge 0$ 입니다.

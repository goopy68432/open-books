---
title: "09. Likelihood, MLE, MAP — 마스터 자료 §1 백지 증명"
slug: 09-likelihood-mle
order: 10
---

# 09. Likelihood, MLE, MAP — 마스터 자료 §1 백지 증명

> 이 파일을 끝내면 `posterior ∝ likelihood × prior`, MAP=NLL+R(θ)=ERM 동치를 빈 종이에 5분 안에 쓸 수 있어야 한다.

## 1. 한 페이지 이론 정리

- **Likelihood**: `L(θ) = p(D|θ) = Π p(x_i|θ)` (i.i.d.).
- **MLE**: `θ̂_MLE = argmax_θ L(θ) = argmax_θ log L(θ) = argmin_θ −log L(θ)`.
- **Bayes 정리**: `p(θ|D) = p(D|θ)p(θ)/p(D)`. 분모는 θ에 무관.
- **MAP**: `θ̂_MAP = argmax_θ p(θ|D) = argmax_θ [log p(D|θ) + log p(θ)]`.

## 2. 백지 증명 — `MAP = NLL + R(θ)` 동치

### 단계 1. 사후분포 정의
$$\hat\theta_{\text{MAP}} = \arg\max_\theta p(\theta\mid D).$$

### 단계 2. 베이즈 정리 적용
$$p(\theta\mid D) = \frac{p(D\mid\theta)p(\theta)}{p(D)}.$$
`p(D)`는 θ에 무관하므로 arg max에 영향 없다 → 분자만 보면 충분.
$$\hat\theta_{\text{MAP}} = \arg\max_\theta\,p(D\mid\theta)\,p(\theta).$$

### 단계 3. 로그 (단조증가) 적용
$$= \arg\max_\theta\big[\log p(D\mid\theta) + \log p(\theta)\big].$$

### 단계 4. 부호 뒤집고 minimize로
$$= \arg\min_\theta\big[-\log p(D\mid\theta) - \log p(\theta)\big].$$

### 단계 5. 명명
- `−log p(D|θ)` = **NLL** (데이터 적합도).
- `−log p(θ)` = **R(θ)** (정규화/penalty).
$$\hat\theta_{\text{MAP}} = \arg\min_\theta\big[\text{NLL}(\theta) + R(\theta)\big].$$
∎

이게 **MAP = ERM(정규화 포함)** 동치다.

## 3. 사례 1. Gaussian Prior → L2 정규화

`p(θ) = N(0, 1/λ I)` 이면:
$$-\log p(\theta) = \tfrac{\lambda}{2}\|\theta\|_2^2 + \text{const}.$$

(증명: 정규분포 PDF에 `μ=0, σ²=1/λ` 대입, log 취하고 const 정리.)

따라서:
$$\hat\theta_{\text{MAP}} = \arg\min_\theta\Big[\text{NLL}(\theta) + \tfrac{\lambda}{2}\|\theta\|^2\Big].$$
이게 **L2 정규화 (Ridge)**.

## 4. 사례 2. Laplace Prior → L1 정규화

`p(θ) ∝ exp(−λ|θ|)` 이면 `−log p(θ) = λ|θ| + const`. 따라서:
$$\hat\theta_{\text{MAP}} = \arg\min_\theta\Big[\text{NLL}(\theta) + \lambda\|\theta\|_1\Big].$$
이게 **L1 정규화 (LASSO)**.

## 5. 사례 3. Gaussian noise + Linear Model = MLE = MSE

`y_i = wᵀx_i + ε_i`, `ε_i ~ N(0,σ²)`.

**NLL 유도** (08번 파일 복습):
$$\text{NLL}(w) = \frac{1}{2\sigma^2}\sum_i (y_i - w^\top x_i)^2 + \text{const}.$$

상수와 양의 배수 무시:
$$\hat w_{\text{MLE}} = \arg\min_w \sum_i (y_i - w^\top x_i)^2.$$
→ **MSE 최소화 = MLE = ERM**.

여기에 가우시안 prior 추가하면 → Ridge.

## 6. 사례 4. Bernoulli + 로지스틱 = MLE = BCE

`p(y_i=1|x_i) = σ(wᵀx_i+b)`. 베르누이로:
$$L(w,b) = \prod_i p_i^{y_i}(1-p_i)^{1-y_i}.$$
NLL:
$$-\sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)] = \text{BCE}.$$

## 7. MAP vs MLE 한 줄 비교

| 항목 | MLE | MAP |
|---|---|---|
| 사용 정보 | 데이터만 | 데이터 + 사전 |
| 출력 | argmax L | argmax (L · prior) |
| 동치 | ERM | ERM + R(θ) |
| 데이터 ↑ | 우세 | prior 영향 ↓ → MLE 수렴 |

## 8. 직접 손으로 — 백지에서 5분 안에

종이를 펴고 **다음을 보지 않고** 적어라:
1. 베이즈 정리.
2. MAP 정의 → NLL + R(θ) 동치 5단계.
3. Gaussian prior → L2.
4. 가우시안 noise 회귀 → MLE = MSE.
5. 베르누이 모델 → MLE = BCE.

5분 안에 막히지 않고 쓰면 합격. 막히면 다시 본다.

## 9. 자주 묻는 질문

**Q1. `argmax`에 log를 씌워도 되는 이유?**
→ log는 단조증가. `f > g ⇔ log f > log g` (양수 영역). arg가 보존된다.

**Q2. `−log` 인 이유?**
→ 컴퓨터/이론 모두 **최소화** 문제로 표준화하기 위함. likelihood는 0~1 사이라 log는 음수. `−log`로 양수 손실 만든다.

**Q3. prior가 없으면 어떻게 되나?**
→ MAP의 `R(θ)` = 0 → MAP = MLE.

**Q4. 왜 i.i.d. 가정이 필요?**
→ joint를 곱으로 쪼개기 위해. log 취하면 합으로 → 미분 쉬움.

## 10. 정리

> **베이즈 정리 → log → 부호 뒤집기 → NLL + R(θ).**
> **Gaussian prior = L2, Laplace prior = L1, Gaussian noise = MSE, Bernoulli = BCE.**
> 백지에 5분 안에 → 통과.

다음 → `10_softmax와_크로스엔트로피_증명.md`

---
title: "10. Softmax 미분과 Cross-Entropy 합성 미분 — 백지 증명"
slug: 10-softmax
order: 11
---

# 10. Softmax 미분과 Cross-Entropy 합성 미분 — 백지 증명

## 1. Softmax 정의 (반드시 외우기)
$$\hat y_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^C e^{z_k}}.$$
- 입력: 로짓 벡터 `z ∈ ℝᶜ`.
- 출력: 확률 벡터 `ŷ ∈ ℝᶜ` (`Σŷ_i = 1`, 모두 ≥0).

편의상 `Σ := Σ_k e^{z_k}` 라 두자.

## 2. 분수 미분 규칙 (복습)
$$\Big(\frac{f}{g}\Big)' = \frac{f'g - fg'}{g^2}.$$

## 3. Case 1. `i = j` 일 때 `∂ŷ_i/∂z_i`

`ŷ_i = e^{z_i}/Σ`. `z_i`에 대해 미분.
- 분자 `e^{z_i}` 의 `z_i` 미분 → `e^{z_i}`.
- 분모 `Σ` 의 `z_i` 미분 → `e^{z_i}` (Σ 안의 `e^{z_i}` 항만 남음, 나머지는 상수).

분수 미분:
$$\frac{\partial \hat y_i}{\partial z_i} = \frac{e^{z_i}\cdot\Sigma - e^{z_i}\cdot e^{z_i}}{\Sigma^2} = \frac{e^{z_i}}{\Sigma}\cdot\frac{\Sigma - e^{z_i}}{\Sigma} = \hat y_i (1 - \hat y_i). \;\square$$

## 4. Case 2. `i ≠ j` 일 때 `∂ŷ_i/∂z_j`

`ŷ_i = e^{z_i}/Σ`. `z_j`에 대해 미분 (`i≠j`).
- 분자 `e^{z_i}` 의 `z_j` 미분 → `0` (z_j에 무관).
- 분모 `Σ` 의 `z_j` 미분 → `e^{z_j}`.

분수 미분:
$$\frac{\partial \hat y_i}{\partial z_j} = \frac{0\cdot \Sigma - e^{z_i}\cdot e^{z_j}}{\Sigma^2} = -\frac{e^{z_i}}{\Sigma}\cdot\frac{e^{z_j}}{\Sigma} = -\hat y_i\,\hat y_j.\;\square$$

## 5. 두 case 합치기 (Kronecker delta)

$$\boxed{\frac{\partial \hat y_i}{\partial z_j} = \hat y_i(\delta_{ij} - \hat y_j).}$$

여기서 `δ_{ij} = 1 if i=j else 0`. 자코비안 행렬 형태:
$$J = \text{diag}(\hat y) - \hat y\hat y^\top.$$

## 6. Cross-Entropy + Softmax 합성 미분 (백프롭의 핵심)

손실:
$$L = -\sum_{i=1}^C y_i \log \hat y_i,\quad y\text{는 one-hot}.$$

목표: `∂L/∂z_j` 를 구한다.

### 단계 1. 체인룰
$$\frac{\partial L}{\partial z_j} = \sum_{i=1}^C \frac{\partial L}{\partial \hat y_i}\cdot \frac{\partial \hat y_i}{\partial z_j}.$$

### 단계 2. 각 항 계산
- `∂L/∂ŷ_i = -y_i/ŷ_i`.
- `∂ŷ_i/∂z_j = ŷ_i(δ_{ij} - ŷ_j)`.

### 단계 3. 대입
$$\frac{\partial L}{\partial z_j} = \sum_i \!\left(-\frac{y_i}{\hat y_i}\right)\cdot \hat y_i(\delta_{ij} - \hat y_j) = -\sum_i y_i(\delta_{ij} - \hat y_j).$$

### 단계 4. 분배
$$= -\sum_i y_i \delta_{ij} + \hat y_j \sum_i y_i.$$

### 단계 5. 단순화
- `Σ_i y_i δ_{ij} = y_j` (델타가 i=j에서만 1).
- `Σ_i y_i = 1` (one-hot).
$$\frac{\partial L}{\partial z_j} = -y_j + \hat y_j = \hat y_j - y_j.\;\square$$

## 7. 핵심 결론 (외울 것)

$$\boxed{\frac{\partial L}{\partial z} = \hat y - y.}$$

벡터로 한 줄. **예측-정답 잔차가 logit의 그래디언트**. BCE+시그모이드의 일반화.

## 8. 자주 헷갈리는 포인트

❓ **`δ_{ij}`가 왜 등장?**
→ Case 1과 Case 2를 한 식으로 묶기 위함. `i=j`면 `(1-ŷ_j)`, 아니면 `−ŷ_j` ←→ `δ_{ij}-ŷ_j`로 통일.

❓ **`Σ y_i = 1` 가정 어디서?**
→ y는 one-hot 라벨. 정확히 한 자리만 1.

❓ **시그모이드+BCE와 같은 결과?**
→ 그렇다. 이진 분류에서 `(σ(z)-y)` 가 똑같이 나온다. softmax는 그 일반화.

## 9. 백지 연습 (필수)

빈 종이에 다음을 **보지 않고** 작성:
1. softmax 정의.
2. Case 1 (분수미분 5줄).
3. Case 2 (분수미분 4줄).
4. 통합식 `ŷ_i(δ_{ij}-ŷ_j)`.
5. Cross-Entropy 합성 5단계.
6. 결론 `∂L/∂z = ŷ-y`.

**막힘 없이 적을 때까지 하루 1번씩, 3일 반복.**

## 10. 직접 해보기

1. `C=3`, `z=[1, 2, 3]`. `Σ = e+e²+e³ ≈ 30.19`. `ŷ`의 세 값을 계산.
2. `y = [0, 1, 0]`, `ŷ = [0.1, 0.7, 0.2]` 일 때 `∂L/∂z`?
   - 정답: `[0.1, -0.3, 0.2]`.
3. softmax의 자코비안 `J`가 대칭임을 보여라. (정답: `J_{ij} = ŷ_i(δ_{ij}-ŷ_j)`. `i,j` 바꾸면? `δ_{ij}=δ_{ji}`이고 `ŷ_iŷ_j = ŷ_jŷ_i`. ✓)

## 11. 정리

> **Softmax 자코비안 = `diag(ŷ) - ŷŷᵀ`.**
> **Softmax+CE 합성 미분 = `ŷ - y`.** (백프롭의 황금 결과)

다음 → `11_역전파_손계산.md`

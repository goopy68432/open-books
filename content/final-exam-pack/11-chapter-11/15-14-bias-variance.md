---
title: "14. Bias-Variance 분해 — 백지 5단계 증명"
slug: 14-bias-variance
order: 15
---

# 14. Bias-Variance 분해 — 백지 5단계 증명

## 1. 무엇을 증명하나?

데이터 `(x, y)` 가 `y = f(x) + ε`, `E[ε]=0`, `Var(ε)=σ²` 의 모델로 발생. 데이터셋 `D`로 학습된 추정기 `f̂_D`. 한 점 x에서 기대 일반화 오차는:
$$\mathbb E_{D,\varepsilon}\big[(y - \hat f_D(x))^2\big] = \underbrace{(f(x) - \bar f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb E_D[(\hat f_D(x) - \bar f(x))^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}.$$

여기서 `\bar f(x) := E_D[\hat f_D(x)]`.

## 2. 사전 도구 3가지 (꼭 챙겨라)

1. `E[(A+B)²] = E[A²] + 2E[AB] + E[B²]`.
2. 두 변수가 **독립**이고 `E[B] = 0` 이면 `E[AB] = E[A]·E[B] = 0`.
3. `Var(X) = E[X²] - (E[X])²` 또는 `E[(X-E[X])²]`.

## 3. 백지 증명 5단계

### 단계 1. 잔차를 두 조각으로
$$y - \hat f_D = (y - f) + (f - \hat f_D) = \varepsilon + (f - \hat f_D).$$
(`f`는 진짜 함수값, ε는 노이즈.)

### 단계 2. 제곱
$$(y-\hat f_D)^2 = \varepsilon^2 + 2\varepsilon(f-\hat f_D) + (f-\hat f_D)^2.$$

### 단계 3. 기대값 — 교차항 소거
ε는 D, x와 독립이고 `E[ε] = 0`이므로:
$$\mathbb E[\varepsilon(f-\hat f_D)] = E[\varepsilon]\cdot E[f-\hat f_D] = 0.$$
따라서:
$$\mathbb E[(y-\hat f_D)^2] = \underbrace{E[\varepsilon^2]}_{=\sigma^2} + \mathbb E_D[(f-\hat f_D)^2].$$

### 단계 4. `(f - f̂_D)` 를 또 두 조각으로
`\bar f := E_D[\hat f_D]` 를 끼워 넣고:
$$f - \hat f_D = (f - \bar f) + (\bar f - \hat f_D).$$

제곱:
$$(f-\hat f_D)^2 = (f-\bar f)^2 + 2(f-\bar f)(\bar f-\hat f_D) + (\bar f - \hat f_D)^2.$$

### 단계 5. `E_D` 적용 — 또 교차항 소거
`(f - \bar f)` 는 D에 무관한 상수. `E_D[\bar f - \hat f_D] = \bar f - \bar f = 0`.
따라서 교차항 = `2(f-\bar f)·E_D[\bar f - \hat f_D] = 0`.

남은 것:
$$\mathbb E_D[(f-\hat f_D)^2] = (f-\bar f)^2 + \mathbb E_D[(\bar f - \hat f_D)^2].$$

### 결론 (전부 합치기)
$$\mathbb E_{D,\varepsilon}[(y-\hat f_D)^2] = \underbrace{(f-\bar f)^2}_{\text{Bias}^2} + \underbrace{\mathbb E_D[(\hat f_D - \bar f)^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}. \;\square$$

## 4. 의미 직관

- **Bias²**: 모델이 평균적으로 진짜 f를 얼마나 못 맞추나. (모델이 너무 단순 → 큼.)
- **Variance**: 데이터셋이 바뀌면 추정값이 얼마나 흔들리나. (모델이 너무 복잡 → 큼.)
- **Noise σ²**: 데이터 자체의 본질적 흔들림. 모델로 줄일 수 없음.

복잡도 ↑ → Bias ↓, Variance ↑. 보통 **U자 곡선** → 어떤 sweet spot.

## 5. Bitter Lesson과의 관계

- 강한 inductive bias (도메인 지식 주입) = bias 추가.
- 데이터/계산이 적을 땐 유리.
- 데이터/계산이 충분해지면 **variance를 데이터로 흡수**해 약한 bias + 큰 모델이 더 나음 — 이것이 **Bitter Lesson** (Sutton 2019).

## 6. 두 번째 Descent (현대 딥러닝)

매우 큰 모델은 보간(interpolation regime) 후 다시 오차가 떨어지는 **double descent** 현상 관측. 고전 U자 그림 너머의 새 영역.

## 7. 직접 해보기

1. `f(x)=x`, 추정 `f̂(x)=0` (상수 추정기). `Bias²` 와 `Variance`?
   - `\bar f = 0`. Bias² = (x-0)² = x². Variance = 0. (완전 결정적이라 흔들리지 않음.)

2. `f̂_D(x)`가 `x`를 그대로 사용하는데 매번 잡음 `η_D` 가 더해진다고 해보자: `f̂_D(x) = x + η_D`, `E[η_D]=0, Var(η_D)=v`.
   - `\bar f = x`. Bias² = 0. Variance = v.

3. **Q**: 단계 3에서 ε와 (f-f̂_D)가 독립인 이유?
   - 정답: ε는 새 점의 노이즈, f̂_D는 학습데이터(다른 잔차)로 만든 것. 가정에 의해 독립.

## 8. 정리

> **5단계: 잔차 분해 → 제곱 → ε 교차항 0 → \bar f 끼우고 분해 → E_D 교차항 0.**
> **결과: Bias² + Variance + σ².**
> **모델 복잡도와의 U자 트레이드오프 직관 함께 외울 것.**

다음 → `15_백지증명_체크리스트.md`

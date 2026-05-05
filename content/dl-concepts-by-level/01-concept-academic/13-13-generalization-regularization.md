---
title: "13. 일반화와 정규화"
slug: 13-generalization-regularization
order: 13
---

# 13. 일반화와 정규화

## 13.1 동기부여 및 개요

딥러닝의 궁극적 목표는 훈련 데이터에서의 성능이 아니라, **한 번도 본 적 없는 데이터에서의 성능**(일반화)이다. 현대 딥러닝 모델은 파라미터 수가 데이터 수보다 훨씬 많은 과매개변수화(overparameterized) 상태임에도 불구하고 놀랍게 잘 일반화한다. 이 현상을 이해하는 것은 현대 학습 이론의 핵심 과제이다.

본 장에서는 일반화 오차의 수학적 분해, 편향-분산 트레이드오프, 정규화 기법(weight decay, dropout, 조기 종료), 이중 하강 현상, 그리고 SGD의 암묵적 정규화와 평평한 극소의 관계를 다룬다.

**연결**: 12장의 최적화 알고리즘이 **어떤 해**를 찾는지, 그리고 그 해가 왜 **좋은 일반화 성능**을 보이는지를 설명한다.

---

## 13.2 일반화 오차 분해

### 13.2.1 기본 정의

**정의 13.1.** 데이터 분포 $\mathcal{P}$에서 i.i.d.로 추출된 훈련 세트 $S = \{(x_i, y_i)\}_{i=1}^{n}$이 주어질 때:

- **경험적 위험(Empirical Risk)**: $L_S(h) = \frac{1}{n}\sum_{i=1}^{n}\ell(y_i, h(x_i))$
- **모집단 위험(Population Risk)**: $L_\mathcal{P}(h) = \mathbb{E}_{(x,y)\sim\mathcal{P}}[\ell(y, h(x))]$

**정의 13.2 (일반화 오차 분해).** 가설 공간 $\mathcal{H}$에서 $h^*_\mathcal{H} = \arg\min_{h \in \mathcal{H}} L_\mathcal{P}(h)$, $h_S = \arg\min_{h \in \mathcal{H}} L_S(h)$라 하면:

$$L_\mathcal{P}(h_S) = \underbrace{L_\mathcal{P}(h^*_\mathcal{H})}_{\text{근사 오차 (Approximation)}} + \underbrace{\left(L_\mathcal{P}(h_S) - L_\mathcal{P}(h^*_\mathcal{H})\right)}_{\text{추정 오차 (Estimation)}}$$

- **근사 오차**: 가설 공간 $\mathcal{H}$의 표현력이 부족하여 최적 함수를 근사하지 못하는 정도
- **추정 오차**: 유한한 데이터로 인해 $\mathcal{H}$ 내 최적 함수에서 벗어나는 정도

$|\mathcal{H}|$를 키우면 근사 오차는 감소하지만, 추정 오차가 증가한다 (고전적 관점).

### 13.2.2 일반화 바운드

**정리 13.1 (유한 가설 공간 바운드).** $|\mathcal{H}| < \infty$이고 $\ell \in [0,1]$이면, 확률 $1-\delta$ 이상으로:

$$L_\mathcal{P}(h) - L_S(h) \leq \sqrt{\frac{\log|\mathcal{H}| + \log(2/\delta)}{2n}}, \quad \forall h \in \mathcal{H}$$

*증명 스케치.* 각 고정된 $h$에 대해 Hoeffding 부등식을 적용하면 $P(|L_\mathcal{P}(h) - L_S(h)| > \epsilon) \leq 2e^{-2n\epsilon^2}$이다. Union bound로 $|\mathcal{H}|$개 가설에 확장하면 $2|\mathcal{H}|e^{-2n\epsilon^2} \leq \delta$에서 결과를 얻는다. $\square$

추정 오차는 $O(\sqrt{\log|\mathcal{H}|/n})$으로, 데이터가 많을수록 줄어든다. 무한 가설 공간으로의 확장에는 VC dimension이나 Rademacher complexity를 사용하나, 이 바운드들은 현대 DNN에는 너무 느슨(vacuous)하다.

---

## 13.3 편향-분산 트레이드오프

### 13.3.1 분해

**정리 13.2 (편향-분산 분해).** $h_S$를 ERM 해, $f$를 목표 함수, $\mu = \mathbb{E}_S[h_S]$라 하면:

$$\mathbb{E}_S\left[(h_S(x) - f(x))^2\right] = \underbrace{\text{Var}_S[h_S(x)]}_{\text{분산}} + \underbrace{(\mu(x) - f(x))^2}_{\text{편향}^2}$$

*증명.*

$$\mathbb{E}_S[(h_S - f)^2] = \mathbb{E}_S[(h_S - \mu + \mu - f)^2]$$
$$= \mathbb{E}_S[(h_S - \mu)^2] + 2\mathbb{E}_S[(h_S - \mu)](\mu - f) + (\mu - f)^2$$

$\mathbb{E}_S[h_S - \mu] = 0$이므로 교차항이 사라져 $\text{Var}_S[h_S] + \text{bias}^2(h_S)$를 얻는다. $\square$

### 13.3.2 모델 복잡도와의 관계

```
오차
 ↑
 │  \                      총 오차
 │   \  ___________________/
 │    \/       /
 │    /\      /
 │   /  \    /   분산
 │  /    \  /
 │ /  편향 \/
 │/________\__________________→ 모델 복잡도
          최적점
```

- **단순한 모델** (예: 직선): 높은 편향, 낮은 분산 (과소적합)
- **복잡한 모델** (예: 고차 다항식): 낮은 편향, 높은 분산 (과적합)
- **최적 복잡도**: 편향$^2$ + 분산의 합이 최소인 지점

---

## 13.4 정규화 기법

### 13.4.1 정규화된 경험적 위험 최소화

**정의 13.3 (Regularized ERM).**

$$\hat{\theta} = \arg\min_\theta \left[ L_S(\theta) + \lambda C(\theta) \right]$$

여기서 $C(\theta)$는 복잡도 벌점, $\lambda > 0$는 정규화 강도이다.

**확률적 해석** (MAP-ERM 동치):
- MLE $\leftrightarrow$ ERM: $\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(S|\theta) = \arg\min_\theta L_S(\theta)$
- MAP $\leftrightarrow$ Regularized ERM: $\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(S|\theta)p(\theta)$

| 정규화 | 벌점 $C(\theta)$ | 사전분포 $p(\theta)$ | 특성 |
|--------|-----------------|-------------------|------|
| L2 (Ridge) | $\|\theta\|_2^2$ | 가우시안 $\mathcal{N}(0, \lambda^{-1}I)$ | 부드러운 축소 |
| L1 (LASSO) | $\|\theta\|_1$ | 라플라스 $\text{Laplace}(0, \lambda^{-1})$ | 희소 해 유도 |

교차 검증(cross-validation)으로 $\lambda$를 선택: $\hat{\lambda} = \arg\min_\lambda \frac{1}{K}\sum_{k=1}^{K} L_{S_k}(\hat{\theta}_\lambda(S_{-k}))$

### 13.4.2 드롭아웃 (Dropout)

**정의 13.4 (Dropout, Srivastava et al. 2014).** 훈련 시 각 은닉 유닛 $h_i$에 마스크 $m_i \sim \text{Bernoulli}(1-p)$를 적용:

$$\tilde{h}_i = m_i \cdot h_i$$

추론 시에는 모든 유닛을 사용하되 출력에 $(1-p)$를 곱한다.

**다층적 효과**:
1. **공동 적응(co-adaptation) 방지**: 뉴런들이 서로 독립적으로 유용해야 한다
2. **암묵적 앙상블**: $2^n$개의 서브네트워크를 훈련하고 추론 시 평균을 근사
3. **Lipschitz 모델 선호**: 드롭아웃 노이즈는 함수의 Lipschitz 상수를 줄이는 방향으로 작용
4. **베이지안 근사**: Gal & Ghahramani (2016)에 따르면 드롭아웃은 가우시안 프로세스의 변분 근사

**MC Dropout**: 추론 시에도 드롭아웃을 켜고 여러 번 포워드 패스를 수행하여 예측 불확실성을 추정.

### 13.4.3 조기 종료 (Early Stopping)

검증 오차가 증가하기 시작하면 훈련을 중단한다. 이는 암묵적으로 가설 공간을 제한하는 정규화 효과를 가진다.

---

## 13.5 이중 하강과 과매개변수화

### 13.5.1 고전적 관점의 한계

고전적 통계학은 모델 복잡도를 올리면 U자 곡선(편향 감소 $\to$ 분산 증가)을 예측한다. 그러나 현대 딥러닝에서는 **이중 하강(double descent)** 현상이 관찰된다.

### 13.5.2 세 가지 레짐

**정의 13.5 (이중 하강).** 모델 복잡도(파라미터 수)를 증가시킬 때:

1. **Underfitting Regime**: 근사 오차가 지배적, 복잡도 증가가 도움
2. **Classical U-shaped Regime**: 보간 임계점(interpolation threshold, $p \approx n$) 부근에서 테스트 오차 최대
3. **Modern Interpolation Regime**: $p \gg n$에서 테스트 오차가 다시 감소

```
테스트 오차
    ↑
    │     ╱╲
    │    ╱  ╲
    │   ╱    ╲         (이중 하강)
    │  ╱      ╲       ╱
    │ ╱        ╲     ╱
    │╱          ╲   ╱
    │            ╲_╱
    │
    └──────────────────→ 모델 복잡도 (p/n)
              ↑
        보간 임계점 (p ≈ n)
```

### 13.5.3 선형 회귀에서의 수학적 분석

과매개변수화된 선형 회귀 ($p > n$)에서 최소 노름 해 $\hat{\beta} = X^+ y$의 분석:

비율 $\gamma = p/n$에 대해 (등방성 가정 하):

$$\text{Var}(\hat{\beta}) \propto \begin{cases} \frac{\gamma}{1-\gamma} & \text{if } \gamma < 1 \text{ (under-parameterized)} \\ \frac{1}{\gamma - 1} & \text{if } \gamma > 1 \text{ (over-parameterized)} \end{cases}$$

$\gamma \to 1$ (즉, $p \approx n$)에서 **분산이 발산**하며, 이것이 이중 하강의 peak이다. 편향은 bounded이므로, 이중 하강은 본질적으로 **분산의 발산과 감소**로 설명된다.

**핵심 통찰**: "훈련 데이터를 보간(interpolate)하는 추정기는 미래 데이터를 잘 예측하지 못한다"는 통계학 교과서의 통념은 과매개변수화 영역에서 성립하지 않는다 (Belkin et al., 2020).

---

## 13.6 암묵적 정규화

### 13.6.1 GD의 암묵적 편향

**정리 13.3 (비형식적).** 과매개변수화된 선형 모델에서 ($p > n$), $\theta(0) = 0$에서 시작하는 경사 하강법은 **최소 노름 해(minimum-norm solution)**로 수렴한다:

$$\lim_{t \to \infty} \theta(t) = \theta^*_{\text{MN}} = X^+ y = \arg\min_{\theta: X\theta = y} \|\theta\|_2$$

*직관적 해석*: GD는 명시적 정규화 없이도, 훈련 오차 0을 달성하는 무한히 많은 해 중에서 **가장 단순한(노름이 작은) 해**를 자동으로 선택한다.

### 13.6.2 SGD의 추가적 편향

SGD의 확률적 잡음은 추가적 암묵적 정규화를 제공한다:

| 조건 | 선호하는 해 |
|------|-----------|
| GD | 낮은 $\ell_2$ 노름, max-margin 분류기 |
| SGD + 큰 학습률 | 평평한 극소(flat minima), 낮은 sharpness |
| SGD + 작은 배치 | 더 평평한 극소 (추가 잡음 효과) |
| Dropout | Lipschitz 상수가 작은 모델 |

---

## 13.7 평평한 극소와 일반화

### 13.7.1 Sharpness와 일반화의 관계

**정의 13.6 (Sharpness).** 손실 함수의 헤시안 최대 고유값 $\lambda_{\max}(\nabla^2 L(\theta))$로 sharpness를 측정한다.

- **Flat minimum**: $\lambda_{\max}$ 작음 $\Rightarrow$ 파라미터 미세 변동에 강건 $\Rightarrow$ 좋은 일반화
- **Sharp minimum**: $\lambda_{\max}$ 큼 $\Rightarrow$ 훈련/테스트 손실 곡면 차이 큼 $\Rightarrow$ 나쁜 일반화

### 13.7.2 Edge of Stability (Cohen et al., 2021)

GD 훈련 시 두 단계가 관찰된다:

1. **Progressive Sharpening**: sharpness가 점진적으로 증가
2. **Edge of Stability**: sharpness가 $2/\eta$ (학습률의 역수)에 도달하면 그 근처에서 진동

$\Rightarrow$ 큰 학습률 $\eta$는 낮은 sharpness ($\approx 2/\eta$)를, 즉 **더 나은 일반화**를 유도한다.

### 13.7.3 SAM (Sharpness-Aware Minimization)

**알고리즘 13.1 (SAM, Foret et al. 2021).** Flat minima를 명시적으로 찾는 최적화:

$$\min_\theta \max_{\|\epsilon\| \leq \rho} L(\theta + \epsilon)$$

근사 알고리즘:
1. Adversarial perturbation: $\epsilon^* = \rho \cdot \nabla L(\theta) / \|\nabla L(\theta)\|$
2. 업데이트: $\theta \leftarrow \theta - \eta \nabla L(\theta + \epsilon^*)$

Worst-case 방향의 기울기로 업데이트하므로 자연스럽게 flat minima로 수렴한다.

---

## 13.8 베이지안 신경망

### 13.8.1 베이지안 주변화

단일 점 추정 $\hat{\theta}$ 대신, 사후분포 $p(\theta|S)$ 전체를 사용:

$$p(y|x, S) = \int p(y|x, \theta) p(\theta|S) d\theta$$

여기서 $p(\theta|S) \propto p(S|\theta)p(\theta)$이다.

이는 여러 모델의 **앙상블**과 동등하며, 자연스러운 정규화 효과를 가진다. MAP 추정은 $\arg\max_\theta p(\theta|S)$로 정규화된 ERM과 동등하지만, 완전 베이지안 추론은 적분을 수행하므로 더 강한 정규화 효과를 가진다.

### 13.8.2 근사 방법

정확한 사후 적분은 고차원에서 계산 불가능하므로 다음 근사를 사용:
- **변분 추론(Variational Inference)**: $q(\theta) \approx p(\theta|S)$
- **MC Dropout**: 드롭아웃을 통한 변분 근사
- **Laplace 근사**: 사후분포를 MAP 주변의 가우시안으로 근사

### 13.8.3 손실 경관의 구조

**Mode Connectivity** (Garipov et al., 2018): 두 독립적으로 학습된 최적해 사이에 훈련 손실이 거의 일정한 단순 경로가 존재한다. 이는 과매개변수화된 모델의 좋은 해들이 고차원 공간에서 연결된 매니폴드를 형성함을 시사한다.

---

## 13.9 흔한 오해와 주의점

| 오해 | 올바른 이해 |
|------|-----------|
| "모델이 크면 반드시 과적합한다" | 이중 하강: 충분히 큰 모델은 보간 임계점 이후 오히려 좋아진다 |
| "훈련 오차 0이면 과적합이다" | Benign overfitting: 과매개변수화 모델은 보간하면서도 일반화 가능 |
| "Weight decay는 가중치를 작게 만들 뿐이다" | MAP-가우시안 사전분포와 동등하며 효과적 복잡도를 제어한다 |
| "편향-분산 트레이드오프는 항상 성립한다" | 분해 자체는 성립하지만, "복잡도 $\uparrow$ $\Rightarrow$ 분산 $\uparrow$"는 보간 임계점 이후 깨진다 |
| "Sharp minima는 항상 나쁘다" | Reparameterization에 의존; invariant sharpness 측정이 필요하다 |
| "Dropout은 단순히 노이즈를 추가하는 것이다" | 앙상블 + 공동 적응 방지 + 베이지안 근사 + Lipschitz 제어 |

---

## 13.10 핵심 요약

| 개념 | 핵심 수식 | 왜 중요한가 |
|------|---------|-----------|
| 일반화 오차 분해 | $L_\mathcal{P} = \text{approx} + \text{estimation}$ | DL의 궁극적 성능 지표 |
| 편향-분산 분해 | $\text{MSE} = \text{Var} + \text{Bias}^2$ | 모델 선택의 이론적 기초 |
| 정규화 (Weight Decay) | $L_S(\theta;\lambda) = L_S(\theta) + \lambda\|\theta\|^2$ | 과적합 방지의 기본 도구 |
| 드롭아웃 | $\tilde{h}_i = m_i \cdot h_i$, $m_i \sim \text{Bernoulli}(1-p)$ | 앙상블 + 공동적응 방지 |
| 이중 하강 | $p \approx n$에서 peak, $p \gg n$에서 재하강 | 과매개변수화의 이론적 정당화 |
| 암묵적 정규화 | GD $\to$ min-norm, SGD $\to$ flat minima | 명시적 정규화 없이도 일반화 |
| SAM | $\min_\theta \max_{\|\epsilon\|\leq\rho} L(\theta+\epsilon)$ | 명시적 flat minima 탐색 |

> **핵심 메시지**: "왜 딥러닝이 일반화하는가"에 대한 완전한 이론은 아직 없다. 현재까지의 이해는 SGD의 암묵적 정규화(flat minima 선호, min-norm 해), 손실 경관의 구조적 특성(mode connectivity, benign overfitting), 그리고 과매개변수화에서의 이중 하강 현상을 부분적 설명으로 제시한다. 이것은 현대 이론 ML/DL의 가장 중요한 열린 문제이다.

**참고문헌**: Hoeffding (1963), Srivastava et al. (2014), Gal & Ghahramani (2016), Belkin et al. (2020), Cohen et al. (2021), Foret et al. (2021), Garipov et al. (2018)

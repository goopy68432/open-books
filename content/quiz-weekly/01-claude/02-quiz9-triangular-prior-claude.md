---
title: "Quiz 9. Triangular Prior MAP — 문제 해석 + 단계별 완전 풀이"
slug: quiz9-triangular-prior-claude
order: 2
---

# Quiz 9. Triangular Prior MAP — 문제 해석 + 단계별 완전 풀이

---

## 문제 (원문)

Prior:
$$
p_m(\theta) =
\begin{cases}
0, & |\theta - 0.5| \geq \dfrac{1}{m} \\[6pt]
m - m^2 |\theta - 0.5|, & |\theta - 0.5| \leq \dfrac{1}{m}
\end{cases}
$$

관측:
$$
n = 5, \qquad k = 4
$$

$m = 2$ 와 $m = 6$ 일 때 prior 의 형태와 MAP 추정값을 구하라.

---

# Part I. 문제 해석

## 0. 큰 그림: 이 문제가 묻는 것이 뭔가

한 문장으로:
> "동전을 5번 던졌더니 4번 앞면이 나왔다. 그런데 너에겐 '이 동전은 공정에 가까울 거야' 라는 **사전 믿음** 이 있다. 그 믿음의 강도가 $m=2$ 일 때와 $m=6$ 일 때, 데이터를 본 후의 **최선 추정값** $\theta$ 는 각각 얼마인가?"

여기서 $\theta$ 는 "이 동전이 앞면을 낼 확률" (0 과 1 사이 숫자). 진짜 정답은 모르고, 우리가 추정해야 합니다.

---

## 1. 첫 번째 수식: prior $p_m(\theta)$ 의 정의 해체

### 1.1 기호 한 글자씩 해체

- **$p_m$** : "prior 분포" 라는 함수 이름. 아래 첨자 $m$ 은 "$m$ 이라는 숫자가 모양을 결정한다" 는 표시.
- **$\theta$** ("세타") : 변수. 동전이 앞면 나올 확률. 0 이상 1 이하 어떤 실수.
- **좌변 $p_m(\theta)$** : "$\theta$ 라는 값이 $m$ 짜리 prior 에서 얼마나 그럴듯한가" 를 나타내는 밀도(높이). 큰 값 = 그럴듯함, 0 = 절대 아님.
- **$|\cdot|$** : 절댓값. 음수면 부호 뒤집어 양수로. 예: $|-3| = 3$, $|0.7 - 0.5| = 0.2$.
- **$|\theta - 0.5|$** : "$\theta$ 가 0.5 에서 얼마나 떨어져 있는가" 의 거리.
- **$\dfrac{1}{m}$** : 자연수 $m$ 의 역수. $m=2$ 면 $\frac{1}{2} = 0.5$, $m=6$ 면 $\frac{1}{6} \approx 0.167$.
- **$\geq$** / **$\leq$** : 이상 / 이하.

### 1.2 중괄호 표기 ("piecewise function")

이 모양은 **"경우 나누기"** 입니다.

- **윗줄** ($|\theta-0.5| \geq \frac{1}{m}$): 이 조건 만족하면 $p_m(\theta) = 0$.
- **아랫줄** ($|\theta-0.5| \leq \frac{1}{m}$): 이 조건 만족하면 $p_m(\theta) = m - m^2|\theta - 0.5|$.

말로 풀면:
> **"$\theta$ 가 0.5 근처에 있으면 (좁은 구간 안) 어떤 양수 값을 갖고, 멀리 떨어지면 0 이다"**

### 1.3 시각화 — "삼각형" 인 이유

아랫줄의 식 $m - m^2|\theta - 0.5|$ 를 그림으로 보면:

- $\theta = 0.5$ 일 때 값은 $m - m^2 \cdot 0 = m$ (가장 높음, 꼭짓점).
- $\theta$ 가 0.5 에서 $\frac{1}{m}$ 만큼 멀어지면 값은 $m - m^2 \cdot \frac{1}{m} = m - m = 0$ (양 끝).
- 그 사이에서는 **직선** ($|\theta - 0.5|$ 가 1차식이므로).

따라서 모양은 **"$\theta = 0.5$ 에서 꼭짓점이 $m$ 인 이등변 삼각형"**. 받침(0이 아닌 구간) 길이는 $\frac{2}{m}$, 높이는 $m$.

### 1.4 면적 = 1 검증 (정규화된 PDF인가?)

확률밀도함수는 면적이 1 이어야 합니다.
- 삼각형 면적 = $\dfrac{1}{2} \times \text{밑변} \times \text{높이} = \dfrac{1}{2} \times \dfrac{2}{m} \times m = 1$ ✓

좋습니다. **이건 진짜 확률분포**입니다.

### 1.5 $m$ 값에 따른 변화 — 시각적 직관

```
m = 2 (약한 prior)            m = 6 (강한 prior)

높이 2                         높이 6
       △                              ▲
      △△                              ▲   ← 매우 뾰족
     △ △                              ▲
    △  △                             ▲▲
   △   △                             ▲▲
  △    △                             ▲ ▲
 △     △                            ▲   ▲
└──────┴──────┐                     └─┴─┐
0    0.5    1                     1/3 0.5 2/3
받침: [0, 1]                       받침: [1/3, 2/3]
(전 구간)                          (좁은 구간)
```

- **$m$ 작으면**: 받침이 넓고 낮은 삼각형 → "0.5 근처가 더 그럴듯하지만 다른 값도 충분히 가능" 이라는 약한 의견.
- **$m$ 크면**: 받침이 좁고 뾰족한 삼각형 → "거의 무조건 0.5 근처! 다른 값은 절대 안 됨" 이라는 강한 의견.

---

## 2. 두 번째 수식: 관측 데이터

$$
n = 5, \qquad k = 4
$$

### 2.1 기호 풀이

- **$n$** : 동전을 던진 총 횟수.
- **$k$** : 그 중 앞면(성공) 이 나온 횟수.

### 2.2 의미

> "동전을 **5번** 던졌더니 **4번** 앞면이 나왔다."

- 앞면 비율 (관측 빈도) = $\dfrac{k}{n} = \dfrac{4}{5} = 0.8$.
- 데이터만 보면 "이 동전은 앞면 확률 0.8 인 편파 동전 같다" 는 인상.

### 2.3 Likelihood 작성

이 관측의 가능도 (likelihood) 는 베르누이 분포에서:
$$
L(\theta) = p(\text{데이터} \mid \theta) = \theta^k (1-\theta)^{n-k} = \theta^4 (1-\theta)^1
$$

기호 풀이:
- **$L(\theta)$** : "$\theta$ 가 진짜 값이라면 이 데이터가 관측될 확률".
- **$\theta^k$** : 앞면 $k$ 번 = 각 시도 확률 $\theta$ 가 $k$ 번 곱해짐.
- **$(1-\theta)^{n-k}$** : 뒷면 $(n-k)$ 번 = 각 시도 확률 $(1-\theta)$ 가 $(n-k)$ 번 곱해짐.

여기 대입: $L(\theta) = \theta^4 (1-\theta)$.

---

## 3. 세 번째 부분: $m=2$ 와 $m=6$ 일 때 구하라

### 3.1 두 가지를 구해야 함

문제는 **두 가지** 를 묻습니다:

1. **Prior 의 "형태"**: 그래프가 어떻게 생겼는지 — 받침, 정점, 모양.
2. **MAP 추정값**: 데이터 본 후 가장 그럴듯한 $\theta^*$ 값.

### 3.2 MAP 가 뭔가 (개념 복습)

**M**aximum **A** **P**osteriori (사후확률 최대화).

$$
\theta^*_{\text{MAP}} = \arg\max_\theta p(\theta \mid \text{데이터}) \;\propto\; \arg\max_\theta \underbrace{L(\theta)}_{\text{likelihood}} \cdot \underbrace{p_m(\theta)}_{\text{prior}}
$$

- **$\arg\max_\theta$** : "괄호 안을 가장 크게 만드는 $\theta$ 값" (값이 아니라 **위치**).
- **$\propto$** : "비례한다" (분모 $p(\text{데이터})$ 는 $\theta$ 무관 상수라 무시).
- 의미: "데이터와 사전 믿음 둘 다 고려해서 가장 그럴듯한 $\theta$" 를 찾는 작업.

### 3.3 Posterior 의 일반 형태

$$
\text{posterior}(\theta) \propto \theta^4 (1-\theta) \cdot p_m(\theta)
$$

- **$p_m(\theta) = 0$ 인 구간** 에서는 곱이 0 → 그 영역의 $\theta$ 는 **절대 답이 될 수 없음**. 받침 밖은 prior 가 0 이니까.
- **받침 안** 에서만 likelihood × prior 를 비교해 최댓값을 찾음.

---

# Part II. 단계별 완전 풀이

이제 본격적으로 $m=2$, $m=6$ 두 케이스를 끝까지 풉니다.

---

## 4. 풀이 전략 (왜 이 순서로 푸나)

문제의 어려운 점:
- prior 에 절댓값 $|\theta - 0.5|$ 가 있어 $\theta$ 가 0.5 보다 크냐 작냐에 따라 식이 달라짐.
- piecewise 함수라 받침 경계도 후보 (도함수 = 0 외에).

따라서 풀이 순서:

1. **STEP A**: prior 형태 명시 (받침 / 정점 / 좌·우 구간 식 분리).
2. **STEP B**: 좌·우 구간 각각 posterior 식 작성.
3. **STEP C**: log posterior 미분 → 1계 조건으로 임계점 찾기.
4. **STEP D**: 임계점이 받침 안에 있는지 확인.
5. **STEP E**: 후보 (임계점 + 양 구간 경계 + 정점) 비교 → 최댓값 선택.

---

## 5. 케이스 1: $m = 2$

### 5.1 STEP A — Prior 형태

- $\dfrac{1}{m} = \dfrac{1}{2} = 0.5$
- 받침: $|\theta - 0.5| \leq 0.5$ → $\theta \in [0, 1]$ ← **전 구간 다 받침!**
- 정점: $p_2(0.5) = 2$
- 받침 안 식: $p_2(\theta) = 2 - 4|\theta - 0.5|$

**좌·우 구간 분리** (절댓값 풀이):

- $\theta \geq 0.5$ (우측): $|\theta - 0.5| = \theta - 0.5$ →
$$
p_2(\theta) = 2 - 4(\theta - 0.5) = 2 - 4\theta + 2 = 4 - 4\theta = 4(1 - \theta)
$$

- $\theta \leq 0.5$ (좌측): $|\theta - 0.5| = 0.5 - \theta$ →
$$
p_2(\theta) = 2 - 4(0.5 - \theta) = 2 - 2 + 4\theta = 4\theta
$$

**검증**: $\theta = 0.5$ 에서 두 식 모두 $2$ 가 나오는지?
- 우측: $4(1 - 0.5) = 4 \cdot 0.5 = 2$ ✓
- 좌측: $4 \cdot 0.5 = 2$ ✓ (연속!)

**검증**: 양 끝 ($\theta = 0$, $\theta = 1$) 에서 0 이 나오는지?
- $\theta = 0$: $4 \cdot 0 = 0$ ✓
- $\theta = 1$: $4(1-1) = 0$ ✓

좋습니다. 두 식이 정확히 삼각형을 그립니다.

### 5.2 STEP B — Posterior 식 작성

상수배 (정규화 상수, $\theta$ 무관) 는 무시하고 $\propto$ 로 표기합니다.

**우측 구간** ($\theta \in [0.5, 1]$):
$$
\text{posterior}(\theta) \propto \theta^4 (1-\theta) \cdot 4(1-\theta) = 4 \theta^4 (1-\theta)^2 \;\propto\; \theta^4 (1-\theta)^2
$$

**좌측 구간** ($\theta \in [0, 0.5]$):
$$
\text{posterior}(\theta) \propto \theta^4 (1-\theta) \cdot 4\theta = 4 \theta^5 (1-\theta) \;\propto\; \theta^5 (1-\theta)
$$

### 5.3 STEP C — Log Posterior 미분

#### 5.3.1 우측 구간 임계점

$\ell(\theta) := \log[\theta^4 (1-\theta)^2] = 4\log\theta + 2\log(1-\theta)$

미분:
$$
\frac{d\ell}{d\theta} = \frac{4}{\theta} - \frac{2}{1-\theta}
$$

$\dfrac{d\ell}{d\theta} = 0$ 으로 두기:
$$
\frac{4}{\theta} = \frac{2}{1-\theta}
$$

양변에 $\theta(1-\theta)$ 곱:
$$
4(1 - \theta) = 2\theta \;\Longrightarrow\; 4 - 4\theta = 2\theta \;\Longrightarrow\; 4 = 6\theta \;\Longrightarrow\; \boxed{\theta = \frac{2}{3}}
$$

**받침 확인**: $\frac{2}{3} \approx 0.667$ 은 우측 구간 $[0.5, 1]$ 안에 있음 ✓.

#### 5.3.2 좌측 구간 임계점

$\ell(\theta) = \log[\theta^5(1-\theta)] = 5\log\theta + \log(1-\theta)$

미분:
$$
\frac{d\ell}{d\theta} = \frac{5}{\theta} - \frac{1}{1-\theta} = 0
\;\Longrightarrow\; 5(1-\theta) = \theta \;\Longrightarrow\; 5 = 6\theta \;\Longrightarrow\; \theta = \frac{5}{6}
$$

**받침 확인**: $\dfrac{5}{6} \approx 0.833$. 그러나 좌측 구간은 $[0, 0.5]$. **임계점이 구간 밖!** 따라서 좌측 구간엔 내부 임계점 없음.

즉 좌측 구간에서 $\dfrac{d\ell}{d\theta} = \dfrac{5}{\theta} - \dfrac{1}{1-\theta}$ 의 부호 확인:
- $\theta = 0.4$ 대입: $\dfrac{5}{0.4} - \dfrac{1}{0.6} = 12.5 - 1.67 = 10.83 > 0$.

→ **좌측 구간 전체에서 단조 증가** → 좌측에서의 최댓값은 **우측 끝 $\theta = 0.5$** (경계).

### 5.4 STEP D — 후보 비교

후보 3개:
1. 우측 구간 임계점 $\theta = \frac{2}{3}$
2. 양 구간 경계 (꼭짓점) $\theta = 0.5$
3. 받침 양 끝 $\theta = 0$ 과 $\theta = 1$ (prior $= 0$ 이므로 posterior $= 0$, 자동 탈락)

**Posterior 값 계산** (상수 $4$ 포함, 비교만 하면 되니 일관되게):

후보 1: $\theta = \frac{2}{3}$
$$
\text{posterior} = 4 \cdot \left(\frac{2}{3}\right)^4 \cdot \left(\frac{1}{3}\right)^2 = 4 \cdot \frac{16}{81} \cdot \frac{1}{9} = \frac{64}{729} \approx 0.0878
$$

후보 2: $\theta = 0.5$
$$
\text{posterior} = 4 \cdot \left(\frac{1}{2}\right)^5 \cdot \left(\frac{1}{2}\right) = 4 \cdot \frac{1}{32} \cdot \frac{1}{2} = \frac{4}{64} = \frac{1}{16} = 0.0625
$$

(검산: 우측 식 $4 \theta^4 (1-\theta)^2$ 에 $\theta = 0.5$ 대입 → $4 \cdot 1/16 \cdot 1/4 = 1/16 = 0.0625$ ✓ 동일)

**비교**: $0.0878 > 0.0625$.

### 5.5 STEP E — 결론 ($m=2$)

$$
\boxed{\;\theta^*_{\text{MAP}}(m=2) = \frac{2}{3} \approx 0.667\;}
$$

**해석**:
- MLE 만 보면 $\frac{k}{n} = 0.8$ 이지만, prior 가 0.5 쪽으로 끌어당겨서 0.667 로 줄어듦.
- 그래도 prior 정점 0.5 보다는 훨씬 데이터 쪽 (0.8) 으로 치우쳐 있음 → **약한 prior 라 데이터의 영향이 살아남음**.

---

## 6. 케이스 2: $m = 6$

### 6.1 STEP A — Prior 형태

- $\dfrac{1}{m} = \dfrac{1}{6} \approx 0.1667$
- 받침: $|\theta - 0.5| \leq \dfrac{1}{6}$ → $\theta \in \left[\dfrac{1}{3}, \dfrac{2}{3}\right] \approx [0.333, 0.667]$ ← **좁다!**
- 정점: $p_6(0.5) = 6$
- 받침 안 식: $p_6(\theta) = 6 - 36|\theta - 0.5|$

**좌·우 구간 분리**:

- $\theta \in [0.5, \frac{2}{3}]$ (우측):
$$
p_6(\theta) = 6 - 36(\theta - 0.5) = 6 - 36\theta + 18 = 24 - 36\theta
$$

- $\theta \in [\frac{1}{3}, 0.5]$ (좌측):
$$
p_6(\theta) = 6 - 36(0.5 - \theta) = 6 - 18 + 36\theta = 36\theta - 12
$$

**검증** (연속성 + 경계 0):
- $\theta = 0.5$: 우측 $24 - 18 = 6$ ✓, 좌측 $18 - 12 = 6$ ✓
- $\theta = \frac{2}{3}$: 우측 $24 - 36 \cdot \frac{2}{3} = 24 - 24 = 0$ ✓
- $\theta = \frac{1}{3}$: 좌측 $36 \cdot \frac{1}{3} - 12 = 12 - 12 = 0$ ✓

좋습니다.

### 6.2 STEP B — Posterior 식 작성

**우측 구간** ($\theta \in [0.5, \frac{2}{3}]$):
$$
\text{posterior}(\theta) \propto \theta^4 (1-\theta) \cdot (24 - 36\theta)
$$

**좌측 구간** ($\theta \in [\frac{1}{3}, 0.5]$):
$$
\text{posterior}(\theta) \propto \theta^4 (1-\theta) \cdot (36\theta - 12)
$$

### 6.3 STEP C — Log Posterior 미분

#### 6.3.1 우측 구간 임계점

$\ell(\theta) = 4\log\theta + \log(1-\theta) + \log(24 - 36\theta)$

미분 (각 항 미분):
- $\dfrac{d}{d\theta} \log\theta = \dfrac{1}{\theta}$
- $\dfrac{d}{d\theta} \log(1-\theta) = -\dfrac{1}{1-\theta}$ (체인룰: 안 미분 $-1$)
- $\dfrac{d}{d\theta} \log(24-36\theta) = \dfrac{-36}{24-36\theta}$

전체:
$$
\frac{d\ell}{d\theta} = \frac{4}{\theta} - \frac{1}{1-\theta} - \frac{36}{24 - 36\theta} = 0
$$

$24 - 36\theta = 12(2 - 3\theta)$ 로 정리하면 $\dfrac{36}{12(2-3\theta)} = \dfrac{3}{2-3\theta}$:
$$
\frac{4}{\theta} - \frac{1}{1-\theta} - \frac{3}{2 - 3\theta} = 0
$$

양변에 공통 분모 $\theta(1-\theta)(2-3\theta)$ 곱:
$$
4(1-\theta)(2-3\theta) - \theta(2-3\theta) - 3\theta(1-\theta) = 0
$$

각 항 전개:
- $4(1-\theta)(2-3\theta) = 4[2 - 3\theta - 2\theta + 3\theta^2] = 4[2 - 5\theta + 3\theta^2] = 8 - 20\theta + 12\theta^2$
- $-\theta(2-3\theta) = -2\theta + 3\theta^2$
- $-3\theta(1-\theta) = -3\theta + 3\theta^2$

합산:
$$
8 - 20\theta + 12\theta^2 - 2\theta + 3\theta^2 - 3\theta + 3\theta^2 = 0
$$

$\theta^2$ 항: $12 + 3 + 3 = 18\theta^2$
$\theta$ 항: $-20 - 2 - 3 = -25\theta$
상수: $8$

$$
18\theta^2 - 25\theta + 8 = 0
$$

근의 공식:
$$
\theta = \frac{25 \pm \sqrt{625 - 4 \cdot 18 \cdot 8}}{2 \cdot 18} = \frac{25 \pm \sqrt{625 - 576}}{36} = \frac{25 \pm \sqrt{49}}{36} = \frac{25 \pm 7}{36}
$$

두 해:
- $\theta = \dfrac{25 + 7}{36} = \dfrac{32}{36} = \dfrac{8}{9} \approx 0.889$
- $\theta = \dfrac{25 - 7}{36} = \dfrac{18}{36} = \dfrac{1}{2} = 0.5$

**받침 확인**: 우측 구간은 $[0.5, \frac{2}{3}]$.
- $\frac{8}{9} \approx 0.889$ → 받침 밖 ✗
- $\frac{1}{2} = 0.5$ → 우측 구간 좌측 끝 (경계 임계점) ✓

#### 6.3.2 좌측 구간 임계점

$\ell(\theta) = 4\log\theta + \log(1-\theta) + \log(36\theta - 12)$

미분:
- $\dfrac{d}{d\theta} \log(36\theta - 12) = \dfrac{36}{36\theta - 12}$

$36\theta - 12 = 12(3\theta - 1)$, $\dfrac{36}{12(3\theta-1)} = \dfrac{3}{3\theta - 1}$:
$$
\frac{d\ell}{d\theta} = \frac{4}{\theta} - \frac{1}{1-\theta} + \frac{3}{3\theta - 1} = 0
$$

공통 분모 $\theta(1-\theta)(3\theta - 1)$ 곱:
$$
4(1-\theta)(3\theta - 1) - \theta(3\theta - 1) + 3\theta(1-\theta) = 0
$$

전개:
- $4(1-\theta)(3\theta-1) = 4[3\theta - 1 - 3\theta^2 + \theta] = 4[4\theta - 1 - 3\theta^2] = 16\theta - 4 - 12\theta^2$
- $-\theta(3\theta-1) = -3\theta^2 + \theta$
- $3\theta(1-\theta) = 3\theta - 3\theta^2$

합산:
$$
16\theta - 4 - 12\theta^2 - 3\theta^2 + \theta + 3\theta - 3\theta^2 = 0
$$

$\theta^2$ 항: $-12 - 3 - 3 = -18\theta^2$
$\theta$ 항: $16 + 1 + 3 = 20\theta$
상수: $-4$

$$
-18\theta^2 + 20\theta - 4 = 0 \;\Longrightarrow\; 18\theta^2 - 20\theta + 4 = 0 \;\Longrightarrow\; 9\theta^2 - 10\theta + 2 = 0
$$

근의 공식:
$$
\theta = \frac{10 \pm \sqrt{100 - 72}}{18} = \frac{10 \pm \sqrt{28}}{18} = \frac{10 \pm 2\sqrt{7}}{18} = \frac{5 \pm \sqrt{7}}{9}
$$

$\sqrt{7} \approx 2.646$:
- $\theta \approx \dfrac{5 + 2.646}{9} \approx \dfrac{7.646}{9} \approx 0.849$
- $\theta \approx \dfrac{5 - 2.646}{9} \approx \dfrac{2.354}{9} \approx 0.262$

**받침 확인**: 좌측 구간은 $[\frac{1}{3}, 0.5] = [0.333, 0.5]$.
- $0.849$ → 받침 밖 ✗
- $0.262$ → 받침 밖 ($0.262 < 0.333$) ✗

→ **좌측 구간엔 내부 임계점 없음**. 단조성 확인:
- $\theta = 0.4$ 대입: $\dfrac{4}{0.4} - \dfrac{1}{0.6} + \dfrac{3}{1.2 - 1} = 10 - 1.667 + 15 = 23.33 > 0$.
- 좌측 구간 전체에서 $\dfrac{d\ell}{d\theta} > 0$ → **단조 증가** → 좌측 구간 최댓값은 **우측 끝 $\theta = 0.5$** (경계).

### 6.4 STEP D — 후보 비교

좌측·우측 모두에서 임계점이 결국 **경계 $\theta = 0.5$** (양 구간이 만나는 꼭짓점) 으로 모임.

후보:
1. $\theta = 0.5$ (양 구간 경계, 임계점)
2. 받침 양 끝 $\theta = \frac{1}{3}$, $\theta = \frac{2}{3}$ → prior $= 0$ → posterior $= 0$, 탈락

**Posterior 값**:

$\theta = 0.5$ 에서:
$$
\text{posterior} = (0.5)^4 \cdot 0.5 \cdot p_6(0.5) = (0.5)^5 \cdot 6 = \frac{1}{32} \cdot 6 = \frac{6}{32} = \frac{3}{16} = 0.1875
$$

다른 받침 안 점 검증 (예: $\theta = 0.6$):
- $p_6(0.6) = 24 - 36 \cdot 0.6 = 24 - 21.6 = 2.4$
- posterior = $(0.6)^4 \cdot 0.4 \cdot 2.4 = 0.1296 \cdot 0.4 \cdot 2.4 \approx 0.1244$

→ $0.5$ 의 $0.1875$ 가 더 큼. 일관됨.

### 6.5 STEP E — 결론 ($m=6$)

#### 엄밀한 수학적 답

$$
\boxed{\;\theta^*_{\text{MAP}}(m=6) = \frac{1}{2} = 0.5\;}
$$

**해석**:
- 강한 prior 가 데이터를 완전히 압도. MLE 0.8 은 받침 밖이라 불가능.
- 받침 안에서 likelihood × prior 의 최댓값은 prior 의 정점 (0.5) 에 위치.
- → **데이터가 무시되고 prior 의 정점으로 추정값이 끌려감**.

#### 강의 의도식 답 (참고)

만약 prior 를 단순한 "hard support 제약" 으로만 보고 likelihood 만 받침 안에서 최대화한다면:
- $\theta^4(1-\theta)$ 는 $\theta = \frac{4}{5} = 0.8$ 에서 최대.
- 받침 $[\frac{1}{3}, \frac{2}{3}]$ 안에서 likelihood 가 단조 증가 → 우측 경계 $\theta = \frac{2}{3}$ 에서 최대.
- → $\theta^* = \frac{2}{3}$.

이는 prior 가 받침 안에서 "균등" 이라고 단순화한 해석입니다. **하지만 본 문제의 prior 는 균등이 아니라 0.5 에서 정점인 삼각형** 이므로 엄밀히는 $\theta^* = 0.5$ 가 정답.

> **시험 답안 작성 권장**: 두 해석을 모두 적되, 엄밀 풀이 (1/2) 를 정답으로 두고 강의식 (2/3) 을 보조로 병기.

---

## 7. 두 케이스 비교 정리

| 항목 | $m = 2$ (약한 prior) | $m = 6$ (강한 prior) |
|------|----------------------|----------------------|
| 받침 | $[0, 1]$ (전 구간) | $[\frac{1}{3}, \frac{2}{3}]$ (좁음) |
| 정점 높이 | $2$ | $6$ (3배 뾰족) |
| 우측 구간 식 | $4(1-\theta)$ | $24 - 36\theta$ |
| 좌측 구간 식 | $4\theta$ | $36\theta - 12$ |
| 우측 임계점 | $\theta = \frac{2}{3}$ ✓ | $\theta = \frac{1}{2}$ (경계), $\theta = \frac{8}{9}$ (밖) |
| 좌측 임계점 | $\theta = \frac{5}{6}$ (밖) | $\theta = \frac{5 \pm \sqrt{7}}{9}$ (모두 밖) |
| **MAP $\theta^*$** | $\boxed{\frac{2}{3} \approx 0.667}$ | $\boxed{\frac{1}{2} = 0.5}$ (엄밀) / $\frac{2}{3}$ (강의식) |
| MLE 와의 관계 | MLE=0.8 → 약간 끌어내림 | MLE=0.8 → 받침 밖 → 무시 |

---

## 8. 직관적 그림으로 마무리

### $m = 2$ 의 경우
```
likelihood    prior          posterior (곱)
  /\           /\               /\
 /  \         /  \             /  \
/    \       /    \           /    \
       \    /                       \
        \  /                         \
0    0.5    1     0    0.5    1     0    2/3   1
                                          ↑
                                        MAP
```
약한 prior 가 likelihood 의 0.8 을 0.667 까지만 끌어내림.

### $m = 6$ 의 경우
```
likelihood        prior              posterior (곱)
       /\           ▲                       ▲
      /  \          ▲                       ▲
     /    \         ▲                       ▲
    /      \       ▲▲                      ▲▲
   /        \      ▲▲                      ▲▲
                  ▲  ▲                    ▲  ▲
0        0.8  1  1/3 0.5 2/3            1/3 0.5 2/3
                                              ↑
                                            MAP
```
좁고 뾰족한 prior 가 likelihood 를 prior 정점 (0.5) 으로 강제 이동.

---

## 9. 시험 출제 시 답안 작성 모범

> **Q. $m = 2$, $m = 6$ 일 때 prior 형태와 MAP 추정값을 구하라.**

**[해]**

(i) **$m = 2$ 인 경우**

Prior: 받침 $[0, 1]$, 정점 $p_2(0.5) = 2$, 모양은 $\theta = 0.5$ 에서 꼭짓점을 갖는 이등변 삼각형.

우측 구간 $\theta \in [0.5, 1]$ 에서 $p_2(\theta) = 4(1-\theta)$, posterior $\propto \theta^4(1-\theta)^2$.

$\dfrac{d}{d\theta}[4\log\theta + 2\log(1-\theta)] = \dfrac{4}{\theta} - \dfrac{2}{1-\theta} = 0$ 풀면 $\theta = \dfrac{2}{3}$.

좌측 구간엔 임계점이 받침 밖이라 단조 증가 → 경계 $0.5$ 가 최대. 두 후보 비교 시 $\theta = \frac{2}{3}$ 의 posterior 값 $\frac{64}{729} \approx 0.0878$ 이 $\theta = 0.5$ 의 $\frac{1}{16} = 0.0625$ 보다 큼.

$$
\therefore \theta^*_{\text{MAP}}(m=2) = \frac{2}{3}
$$

(ii) **$m = 6$ 인 경우**

Prior: 받침 $[\frac{1}{3}, \frac{2}{3}]$, 정점 $p_6(0.5) = 6$, 좁고 뾰족한 삼각형.

우측 구간 $\theta \in [0.5, \frac{2}{3}]$ 에서 $p_6(\theta) = 24 - 36\theta$, posterior $\propto \theta^4(1-\theta)(24-36\theta)$.

$\dfrac{d}{d\theta} \log(\text{posterior}) = \dfrac{4}{\theta} - \dfrac{1}{1-\theta} - \dfrac{36}{24-36\theta} = 0$.

정리하면 $18\theta^2 - 25\theta + 8 = 0$, $\theta = \dfrac{25 \pm 7}{36}$ → $\dfrac{8}{9}$ (받침 밖) 또는 $\dfrac{1}{2}$ (받침 안).

좌측 구간도 단조 증가하여 경계 $0.5$ 에서 최대. 양 구간 모두 $\theta = 0.5$ 가 최댓값.

$$
\therefore \theta^*_{\text{MAP}}(m=6) = \frac{1}{2}
$$

(MLE = 0.8 은 받침 밖이라 prior 가 데이터를 완전 압도, 추정값이 prior 정점에 고정됨.)

---

## 10. 한 줄 요약

> **"prior 의 강도 $m$ 이 클수록 받침이 좁아지고 정점이 높아져, 데이터(MLE)를 무시하고 추정값을 0.5 쪽으로 강제 이동시킨다. $m=2$ 면 약한 끌림 (MLE 0.8 → MAP 0.667), $m=6$ 면 완전 압도 (MAP = 0.5)."**

이것이 본 문제가 보여주려는 **prior strength vs data influence** 의 핵심입니다.

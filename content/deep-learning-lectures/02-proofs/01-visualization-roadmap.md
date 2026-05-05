---
title: "A+ 유도집 — 시각화 로드맵 (Mermaid 다이어그램 모음)"
slug: visualization-roadmap
order: 1
---

# A+ 유도집 — 시각화 로드맵 (Mermaid 다이어그램 모음)

> 17개 문서의 학습 로드맵, 개념 관계, 시험 시나리오, 유도 흐름을 한눈에.
> Obsidian Mermaid 플러그인에서 모두 렌더링됨.

---

## 1. 학습 로드맵 (Learning Roadmap) — 의존성 그래프

> **의존성 순서**: 화살표를 따라 학습. 부모 노드 없이 자식부터 보면 막힘.

```mermaid
flowchart TD
    A1[A1 수학 빌드업<br/>중1 → 대학원] --> B1[01 베이즈 정리]
    A1 --> M1[06 Softmax Jacobian]
    A1 --> O7[07 Linear Reg<br/>Closed Form]

    B1 --> B2[02 MLE Bernoulli]
    B2 --> B3[03 MAP 일반화]

    B2 --> C4[04 NLL → MSE<br/>Gaussian]
    C4 --> C5[05 Cross-Entropy<br/>Categorical]
    C5 --> M1

    C4 --> O7
    O7 --> N8[08 Newton<br/>2차 근사]
    N8 --> N9[09 Backpropagation]
    M1 --> N9
    N8 --> N12[12 GD/SGD/Adam]
    N9 --> N12

    O7 --> CV10[10 Conv = Linear]
    CV10 --> CV11[11 Output Size Formula]

    B3 --> H14[14 Hypothesis<br/>Restriction = MAP]
    O7 --> H14
    CV10 --> H14
    H14 --> H15[15 Inductive Bias<br/>강도 비교]

    C4 --> KL13[13 KL Divergence]
    C5 --> KL13

    classDef foundation fill:#e8f4f8,stroke:#2c7da0,stroke-width:2px,color:#000
    classDef bayes fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef loss fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    classDef opt fill:#fff8e1,stroke:#f57c00,stroke-width:2px,color:#000
    classDef cnn fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef integ fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000

    class A1 foundation
    class B1,B2,B3 bayes
    class C4,C5,KL13 loss
    class M1,O7,N8,N9,N12 opt
    class CV10,CV11 cnn
    class H14,H15 integ
```

**색상 범례**

| 색 | 영역 | 토픽 |
|----|------|------|
| 🔵 파랑 | 수학 기초 | A1 |
| 🟣 분홍 | 베이즈 / 확률모델 | 01-03 |
| 🟢 초록 | 손실함수 (likelihood → loss) | 04, 05, 13 |
| 🟡 주황 | 미분 / 최적화 / Backprop | 06, 07, 08, 09, 12 |
| 🟪 보라 | CNN / Convolution | 10, 11 |
| 🔴 빨강 | 통합 시각 (가장 중요) | 14, 15 |

---

## 2. 시험 출제 가능성 매트릭스 (Quadrant)

```mermaid
quadrantChart
    title "출제 가능성 vs 난이도"
    x-axis "낮은 난이도" --> "높은 난이도"
    y-axis "낮은 출제가능성" --> "높은 출제가능성"
    quadrant-1 "매우 출제 + 어려움 (집중!)"
    quadrant-2 "매우 출제 + 쉬움 (필승!)"
    quadrant-3 "적게 출제 + 쉬움"
    quadrant-4 "적게 출제 + 어려움"
    "01 베이즈정리": [0.25, 0.85]
    "02 MLE Bernoulli": [0.30, 0.95]
    "03 MAP 일반화": [0.45, 0.92]
    "04 NLL MSE": [0.55, 0.95]
    "05 Cross-Entropy": [0.50, 0.85]
    "06 Softmax Jacobian": [0.65, 0.95]
    "07 Linear Reg": [0.55, 0.80]
    "08 Newton": [0.50, 0.75]
    "09 Backprop": [0.80, 0.95]
    "10 Conv = Linear": [0.60, 0.95]
    "11 Output Size": [0.30, 0.85]
    "12 GD SGD Adam": [0.40, 0.65]
    "13 KL Divergence": [0.55, 0.55]
    "14 Hypothesis MAP": [0.45, 0.75]
    "15 Inductive Bias": [0.30, 0.70]
```

> 💡 **시험 직전 30분이 남는다면**: Q2 영역 (02 MLE, 11 Output Size) 부터 다시 보고 → Q1 영역 (06 Softmax, 09 Backprop, 10 Conv) 순으로.

---

## 3. 강의 메인 시각 — "두 축" 통합 그래프

```mermaid
flowchart LR
    subgraph Axis1[축 1 - Maximum Likelihood]
        L1[Likelihood<br/>P E given H]
        L2[NLL]
        L3[ERM<br/>Empirical Risk]
        L1 --> L2 --> L3
    end

    subgraph Axis2[축 2 - Prior]
        P1[Prior<br/>P H]
        P2[Hypothesis Space<br/>Restriction]
        P3[Inductive Bias]
        P1 --> P2 --> P3
    end

    L1 -.짝.-> P1
    Axis1 ==> Combined[Posterior P H given E<br/>= Likelihood × Prior / Evidence]
    Axis2 ==> Combined

    Combined --> M1[MLE<br/>uniform prior]
    Combined --> M2[MAP<br/>nonuniform prior]
    Combined --> M3[Strong-MAP<br/>delta prior]

    style Combined fill:#ffeb3b,stroke:#f57f17,stroke-width:3px,color:#000
    style L1 fill:#c8e6c9,color:#000
    style P1 fill:#bbdefb,color:#000
```

---

## 4. Likelihood → Loss 매핑 (회귀-분류 평행 구조)

```mermaid
flowchart TB
    subgraph Reg[회귀 Regression]
        RG1[데이터 y in R]
        RG2[Likelihood: Gaussian<br/>N mu sigma squared]
        RG3[NLL = -log Π N]
        RG4[exp / minus / square]
        RG5[MSE Loss<br/>1/n Σ y - h x squared]
        RG1 --> RG2 --> RG3 --> RG4 --> RG5
    end

    subgraph Cls[분류 Classification]
        CL1[데이터 y in 0,1 K one-hot]
        CL2[Likelihood: Categorical<br/>Π p_j to y_j]
        CL3[NLL = -log Π Π p]
        CL4[log of product of powers]
        CL5[Cross-Entropy Loss<br/>-Σ y_j log p_j]
        CL1 --> CL2 --> CL3 --> CL4 --> CL5
    end

    Reg -.같은 구조.- Cls
    RG5 --> ERM[ERM<br/>Empirical Risk Min]
    CL5 --> ERM

    style RG2 fill:#c8e6c9,color:#000
    style CL2 fill:#c8e6c9,color:#000
    style RG5 fill:#ffccbc,color:#000
    style CL5 fill:#ffccbc,color:#000
    style ERM fill:#ffeb3b,stroke:#f57f17,stroke-width:3px,color:#000
```

---

## 5. Optimization 알고리즘 진화 트리

```mermaid
flowchart TD
    Newton[Newton's Method<br/>θ - H inv g<br/>2nd order]
    Newton -->|H = 1/η · I 가정| GD[GD<br/>θ - η g]
    GD -->|mini-batch| SGD[SGD<br/>θ - η g_tilde]
    SGD -->|+ 관성| Mom[Momentum<br/>v = βv + g]
    SGD -->|+ per-param lr| Ada[AdaGrad<br/>η / sqrt G]
    Ada -->|EMA| RMS[RMSProp<br/>η / sqrt v]
    Mom -.결합.-> Adam
    RMS -.결합.-> Adam[Adam<br/>m + v + bias correction]

    Newton -.이론적 ideal.-> Adam

    style Newton fill:#ffcdd2,color:#000
    style Adam fill:#ffeb3b,stroke:#f57f17,stroke-width:3px,color:#000
    style GD fill:#bbdefb,color:#000
    style SGD fill:#bbdefb,color:#000
```

---

## 6. CNN 발전 — Linearity 의 점진적 제한

```mermaid
flowchart LR
    F1[모든 Linear Trans.<br/>A in R m by n<br/>mn 자유 파라미터] -->|sparse 제한| F2[Sparse LT<br/>지역 패턴]
    F2 -->|weight sharing 추가| F3[Convolution<br/>K개 자유도]
    F3 -->|+ pooling 비선형| F4[CNN Layer]
    F4 -->|stack| F5[Deep CNN<br/>AlexNet, VGG, ResNet]

    F1 -.- IB1[Inductive Bias 약]
    F3 -.- IB3[Locality + Translation Inv.]
    F5 -.- IB5[Hierarchical features]

    style F1 fill:#e1f5fe,color:#000
    style F3 fill:#fff9c4,color:#000
    style F5 fill:#ffccbc,color:#000
```

---

## 7. Inductive Bias 강도 스펙트럼

```mermaid
flowchart LR
    L1[Linear<br/>θᵀx] --> L2[Nonlinear basis<br/>fixed φ_j]
    L2 --> L3[Parametrized basis<br/>learned φ]
    L3 --> L4[MLP / Deep NN<br/>UAT]
    L4 --> L5[CNN<br/>+locality+TI]
    L5 --> L6[Transformer<br/>self-attention]

    L1 -.- T1[Prior 강함<br/>데이터 적게]
    L6 -.- T6[Prior 약함<br/>데이터 많이]

    Bitter[Bitter Lesson<br/>weak prior + scale = win] -.지지.-> L6

    style L1 fill:#ef5350,color:#fff
    style L4 fill:#ffa726,color:#000
    style L6 fill:#66bb6a,color:#000
    style Bitter fill:#ffeb3b,stroke:#f57f17,stroke-width:3px,color:#000
```

---

## 8. 시험 답안 작성 시나리오 (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    participant 문제 as 문제 읽기
    participant 학생 as 학생
    participant 답안 as 답안지

    문제->>학생: "Derive θ* = (k+M)/(n+2M)"
    학생->>답안: [Setup] 가정 명시 (IID, prior 형태)
    학생->>답안: [Step 1] Posterior ∝ Likelihood × Prior
    학생->>답안: [Step 2] Log → 곱이 합으로
    학생->>답안: [Step 3] d/dθ = 0 (필요조건)
    학생->>답안: [Step 4] 양변 정리 → 풀이
    학생->>답안: [Step 5] 2차 미분 < 0 검증 (충분조건)
    학생->>답안: [Conclusion] 박스 처리 + 의미 해석
    학생->>답안: [Limit] M→0, M→∞ 케이스
    답안-->>학생: 점수 산정<br/>각 단계마다 "왜" 점수
    Note over 학생,답안: 답만 적으면 0점!
```

---

## 9. Backpropagation 데이터 흐름 (Forward + Backward)

```mermaid
flowchart LR
    subgraph FW[Forward Pass]
        x[x 입력] --> L1[W₁x + b₁]
        L1 --> h[h]
        h --> AC[ReLU]
        AC --> a[a]
        a --> L2[W₂a + b₂]
        L2 --> z[z logits]
        z --> SM[softmax]
        SM --> p[p̂]
        p --> CE[CE loss with y]
        CE --> L_loss[L]
    end

    subgraph BW[Backward Pass - reverse]
        BL[∂L/∂L = 1] -.- δz[δ_z = p̂ - y]
        δz -.- gW2[∂L/∂W₂ = δ_z aᵀ]
        δz -.- gb2[∂L/∂b₂ = δ_z]
        δz -.- δa[δ_a = W₂ᵀ δ_z]
        δa -.- δh[δ_h = δ_a ⊙ σ′h]
        δh -.- gW1[∂L/∂W₁ = δ_h xᵀ]
        δh -.- gb1[∂L/∂b₁ = δ_h]
    end

    L_loss ==>|backprop start| BL

    style δz fill:#ffeb3b,stroke:#f57f17,stroke-width:3px,color:#000
    style L_loss fill:#ef5350,color:#fff
```

---

## 9.1. 학습의 전체 사이클 (The Big Picture: Training Loop)

```mermaid
flowchart TD
    subgraph PRE[1. 준비 단계 - Before FP]
        Init[<b>가중치 초기화</b><br/>W, b를 랜덤하게 설정<br/>Xavier/He 초기화]
        Data[<b>데이터 로드</b><br/>Mini-batch x, y 준비]
    end

    subgraph LOOP[2. 학습 루프 - The Core]
        direction TB
        FW[<b>Forward Pass</b><br/>예측값 및 Loss 계산]
        BW[<b>Backward Pass</b><br/>Chain Rule로 기울기 계산]
        Update[<b>Optimizer Update</b><br/>W = W - η·g<br/>GD/Adam 등으로 가중치 수정]
        
        FW --> BW --> Update
        Update -- "다음 배치(Batch)" --> FW
    end

    subgraph POST[3. 사후 단계 - After BP]
        Eval[<b>모델 평가</b><br/>Validation/Test Set 검증]
        Converge{<b>수렴 확인</b>}
    end

    Init --> Data --> FW
    Update --> Eval --> Converge
    Converge -- "No (Repeat)" --> Data
    Converge -- "Yes" --> Stop([<b>최적 모델 탄생</b>])

    style LOOP fill:#f9f9f9,stroke:#333,stroke-dasharray: 5 5
    style FW fill:#e1f5fe,color:#000
    style BW fill:#fff9c4,color:#000
    style Update fill:#ffeb3b,stroke:#f57f17,stroke-width:2px,color:#000
```

> 💡 **흐름 요약**: Forward에서 Loss를 구하고, Backward에서 각 가중치별 책임(기울기)을 묻고, Update에서 실제로 가중치를 고칩니다.

---

## 10. 개념 관계도 (Concept Graph) — 전체 17개 토픽

```mermaid
graph LR
    A1((A1<br/>수학)):::math
    T01((01<br/>Bayes)):::core
    T02((02<br/>MLE)):::core
    T03((03<br/>MAP)):::core
    T04((04<br/>NLL→MSE)):::loss
    T05((05<br/>CE)):::loss
    T06((06<br/>Softmax)):::deriv
    T07((07<br/>LinReg)):::opt
    T08((08<br/>Newton)):::opt
    T09((09<br/>Backprop)):::opt
    T10((10<br/>Conv=LT)):::cnn
    T11((11<br/>OutputSize)):::cnn
    T12((12<br/>Adam)):::opt
    T13((13<br/>KL)):::loss
    T14((14<br/>HypMAP)):::integ
    T15((15<br/>IB강도)):::integ

    A1 --- T01
    A1 --- T06
    A1 --- T07
    T01 --- T02
    T02 --- T03
    T02 --- T04
    T04 --- T05
    T04 --- T13
    T05 --- T13
    T05 --- T06
    T06 --- T09
    T07 --- T08
    T07 --- T14
    T08 --- T09
    T08 --- T12
    T09 --- T12
    T10 --- T11
    T10 --- T14
    T07 --- T10
    T03 --- T14
    T14 --- T15
    T10 --- T15

    classDef math fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef core fill:#fce4ec,stroke:#c2185b,color:#000
    classDef loss fill:#e8f5e9,stroke:#388e3c,color:#000
    classDef deriv fill:#fff3e0,stroke:#e65100,color:#000
    classDef opt fill:#fff8e1,stroke:#f57c00,color:#000
    classDef cnn fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef integ fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000
```

---

## 11. 시험 D-Day 학습 시나리오 (Gantt)

```mermaid
gantt
    title A+ 시험 대비 7일 플랜
    dateFormat  YYYY-MM-DD
    axisFormat  D-%d

    section Day 7 (기초)
    A1 수학 빌드업                 :a1, 2026-05-04, 1d
    01 Bayes + 02 MLE              :a2, after a1, 1d

    section Day 6 (확률모델)
    03 MAP 일반화                  :b1, 2026-05-05, 1d
    04 NLL→MSE + 05 CE             :b2, 2026-05-06, 1d

    section Day 5 (미분)
    06 Softmax Jacobian            :c1, 2026-05-07, 1d
    07 Linear Reg Closed Form      :c2, 2026-05-08, 1d

    section Day 3 (Optim)
    08 Newton + 09 Backprop        :d1, 2026-05-09, 1d
    12 GD/SGD/Adam                 :d2, 2026-05-10, 1d

    section Day 2 (CNN)
    10 Conv = LT + 11 Output Size  :e1, 2026-05-10, 1d

    section Day 1 (통합)
    13 KL + 14 HypMAP + 15 IB      :f1, 2026-05-11, 1d
    백지 재현 연습                 :f2, 2026-05-11, 1d
```

---

## 12. 답안 채점 기준 (Mind Map)

```mermaid
mindmap
  root((시험 답안<br/>채점 기준))
    Setup
      가정 명시
        IID
        독립
        분포 가정
      목표 명시
        Goal: argmin/argmax
    Steps
      각 단계 "왜"
      수식 변형 정당화
      log 사용 이유
      미분=0 정당화
    Math
      미분 정확도
      대수 변형
      극한 처리
    Validation
      2차 미분 / Hessian
      Boundary case
      극값 vs 안장점
    Conclusion
      박스 처리
      의미 해석
      직관 한 문장
    Style
      영어 가능
      깔끔한 표기
      LaTeX 수식
```

---

## 13. "답만 적으면 0점" — 점수 분배 시각화 (Pie)

```mermaid
pie title 한 문제의 점수 구성 (예시)
    "Setup + 가정 명시" : 15
    "단계별 변형 + 왜" : 40
    "미분 / 대수 정확도" : 20
    "극값 검증" : 10
    "의미 해석" : 10
    "최종 답" : 5
```

> 🎯 **"답"은 5점. "왜"가 95점.** — 강의 채점 철학.

---

## 14. 출제 시나리오 분기 (State Diagram)

```mermaid
stateDiagram-v2
    [*] --> 문제읽기
    문제읽기 --> 유도형: "Derive..." / "Prove..."
    문제읽기 --> 계산형: "Compute..." / "Find..."
    문제읽기 --> 비교형: "Compare..." / "Explain difference..."

    유도형 --> Setup명시
    Setup명시 --> 단계전개
    단계전개 --> 검증
    검증 --> 결론박스
    결론박스 --> [*]

    계산형 --> 식대입
    식대입 --> 차원검증
    차원검증 --> 결론
    결론 --> [*]

    비교형 --> 표작성
    표작성 --> 각항목해설
    각항목해설 --> 통합시각
    통합시각 --> [*]

    note right of 유도형: 01,02,03,04,05<br/>06,07,08,10,11,13
    note right of 계산형: 11(output size)<br/>09(backprop 미분)
    note right of 비교형: 12(GD vs SGD vs Adam)<br/>15(IB 강도)
```

---

## 15. 최종 통합 — "하나의 그림으로 본 강의 전체"

```mermaid
flowchart TB
    subgraph Foundation[수학 기초]
        Math[A1 미분 / 적분 / 벡터 / 행렬 / log / 확률]
    end

    subgraph Theory[확률 이론]
        Bayes[01 Bayes Theorem<br/>Posterior = Lik × Prior / Ev]
        MLE[02 MLE<br/>uniform prior]
        MAP[03 MAP<br/>general prior]
        Bayes --> MLE
        Bayes --> MAP
    end

    subgraph LossDerivation[손실 유도]
        NLL[NLL definition]
        Gauss[04 Gaussian → MSE]
        Cat[05 Categorical → CE]
        KL[13 KL divergence]
        NLL --> Gauss
        NLL --> Cat
        Gauss -.- KL
        Cat -.- KL
    end

    subgraph Calculus[미분 도구]
        Soft[06 Softmax Jacobian]
        BP[09 Backprop chain rule]
        Soft --> BP
    end

    subgraph Optim[최적화]
        Closed[07 Linear Reg Closed Form]
        Newton[08 Newton 2nd order]
        Adam[12 GD SGD Adam]
        Newton --> Adam
        BP --> Adam
    end

    subgraph CNN[CNN 구조]
        Conv[10 Conv = sparse + shared LT]
        Out[11 Output size formula]
        Conv --> Out
    end

    subgraph Integration[통합 시각 ★]
        HypMAP[14 Hyp Restriction = MAP]
        IB[15 Inductive Bias 강도]
        HypMAP --> IB
    end

    Math --> Theory
    MLE --> NLL
    Theory --> LossDerivation
    LossDerivation --> Calculus
    Calculus --> Optim
    Optim --> Closed
    Optim --> CNN
    Theory --> Integration
    LossDerivation --> Integration
    CNN --> Integration

    Integration ==> Final{{🎯 A+ 답안<br/>= 유도 + 연결 + 계산}}

    style Final fill:#ffeb3b,stroke:#f57f17,stroke-width:4px,color:#000
    style Integration fill:#ffebee,stroke:#c62828,stroke-width:3px,color:#000
```

---

## 16. 한 줄 요약 (Slogan)

> **"강의 전체 = Likelihood + Prior 두 축으로 모든 것이 설명된다."**
>
> - Likelihood (어떤 분포 가정?) → Loss 가 결정 (Gaussian→MSE, Categorical→CE)
> - Prior (어떤 hypothesis 만 허용?) → Inductive Bias 가 결정 (Linear, NN, CNN, Transformer)
> - 두 축의 곱 = Posterior → MAP/MLE 가 그 위에서 답을 찾음
> - Optimization (Newton, Adam) + Backprop = 그 답을 "실제로 계산"하는 방법

이 한 단락이 시험에서 어떤 문제가 나와도 첫 줄에 적을 수 있는 "전체 지도".

---

## 17. 사용법

1. Obsidian 에서 `00_시각화_로드맵.md` 를 열기.
2. Mermaid 플러그인이 활성화되어 있으면 모든 다이어그램이 즉시 렌더링됨.
3. 각 다이어그램은 해당 토픽 학습 전에 "내가 어디 있는지" 확인용.
4. 시험 직전에는 #15 (전체 통합) + #2 (출제 매트릭스) 만 빠르게 훑어보면 충분.

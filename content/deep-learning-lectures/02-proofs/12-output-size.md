---
title: '11. Output Size Formula 유도 — "외우지 말고 유도"'
slug: output-size
order: 12
---

# 11. Output Size Formula 유도 — \"외우지 말고 유도\"

> **출제 근거**: 9주차 ★9, 강의 \"외우지 말고 유도하라\" 명시
> **시험 출제 방식**: \"Derive the output size formula $O = \lfloor (W - K + 2P)/S\rfloor + 1$ for a 1D convolution with input width $W$, kernel size $K$, padding $P$, stride $S$.\"

---

## 1. 왜 시험에 나오는가

- AlexNet 등 architecture 분석에서 직접 사용.
- 강의에서 \"외우지 말고 유도\" 라고 명시한 거의 유일한 공식.
- 퀴즈 32 (AlexNet Layer Shape) 의 근간.

---

## 2. 사전 수학

### 2.1 [중1] 등차수열의 항 수

첫 항 $a$, 마지막 $b$, 공차 $d$ (간격) 인 수열의 항 수:

$$
\#\text{항} = \frac{b - a}{d} + 1
$$

> 예: 1, 3, 5, 7. 첫=1, 끝=7, 간격=2 → $(7-1)/2 + 1 = 4$ ✅

이게 output size formula 의 **모든 것**. 진짜.

### 2.2 [고1] Floor 함수

$\lfloor x \rfloor$ = $x$ 이하의 최대 정수.

> 예: $\lfloor 3.7 \rfloor = 3$, $\lfloor 5 \rfloor = 5$, $\lfloor -1.2 \rfloor = -2$.

Output size 가 정수여야 하니까 floor 가 필요.

---

## 3. 문제 설정 (1D)

- $W$: 입력 width (또는 length)
- $K$: 커널 크기
- $P$: padding (양쪽에 각각 $P$ 만큼 추가)
- $S$: stride (커널이 이동하는 간격)

목표: 출력 크기 $O$ 구하기.

---

## 4. 유도 (3단계)

### Step 1 — Padding 후 effective width

양쪽에 $P$ 만큼 padding → effective 입력 길이:

$$
W' = W + 2P
$$

### Step 2 — 커널이 놓일 수 있는 \"시작 위치\" 의 집합

커널 (크기 $K$) 의 **왼쪽 끝** 인덱스가 가능한 값:

- 가장 왼쪽: $1$ (padding 끝에서 시작)
- 가장 오른쪽: 커널 오른쪽 끝이 $W'$ 까지 — 즉 왼쪽 끝 $\leq W' - K + 1$
- 간격 (stride): $S$

따라서 시작 위치는 등차수열:

$$
1, \; 1+S, \; 1+2S, \; \ldots, \; \leq W' - K + 1
$$

### Step 3 — 항 수 계산 (= output size)

등차수열 항 수 공식 (2.1절):

$$
O \;=\; \left\lfloor \frac{(W' - K + 1) - 1}{S} \right\rfloor + 1 \;=\; \left\lfloor \frac{W' - K}{S} \right\rfloor + 1
$$

$W' = W + 2P$ 대입:

$$
\boxed{\; O \;=\; \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1 \;}
\tag{*}
$$

---

## 5. 각 항의 \"왜\" (Line-by-Line)

| 항 | 왜 |
|------|-----|
| $W$ | 입력 길이 |
| $-K$ | 커널이 안에 \"통째로 들어가야\" 하니까 마지막 시작 위치는 $W - K + 1$ — 즉 가능한 시작 위치의 폭은 $W - K$ |
| $+2P$ | 양쪽 padding 으로 효과적 폭 증가 |
| $/S$ | 간격 $S$ 로 나누면 \"몇 칸 이동했는지\" |
| $\lfloor \cdot \rfloor$ | 마지막 위치가 정확히 들어맞지 않으면 잘라냄 (안 맞는 끝부분은 못 씀) |
| $+1$ | 등차수열 첫 항도 포함 (off-by-one) |

---

## 6. 케이스별 단순화

### 6.1 No padding, stride 1: \"valid\"

$P=0$, $S=1$:

$$
O = W - K + 1
$$

> 1D conv 의 가장 기본 형태. 10 토픽 예시 ($W=4, K=3$)에서 $O=4-3+1=2$ ✅.

### 6.2 \"Same\" convolution (입력 = 출력 크기)

$O = W$ 가 되도록 $P$ 선택. $S=1$ 이면:

$$
W = W - K + 2P + 1 \;\Longrightarrow\; P = \frac{K - 1}{2}
$$

→ $K$ 가 홀수일 때만 정수 padding 가능 (그래서 conv 커널은 보통 홀수).

### 6.3 Stride 2 (downsampling)

$S=2$ → 출력 크기 약 절반.

> 예: $W=8, K=3, P=1, S=2$ → $O = \lfloor (8 - 3 + 2)/2 \rfloor + 1 = \lfloor 7/2 \rfloor + 1 = 3 + 1 = 4$.

---

## 7. 2D 확장

각 공간 차원마다 독립으로 적용:

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} - K_h + 2P_h}{S_h} \right\rfloor + 1
$$

$$
W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K_w + 2P_w}{S_w} \right\rfloor + 1
$$

채널은 영향 없음 (커널의 채널 = 입력 채널, 출력 채널 = 커널 개수).

---

## 8. AlexNet Layer 계산 예 (퀴즈 32 패턴)

| Layer | $W_{\text{in}}$ | $K$ | $P$ | $S$ | $W_{\text{out}}$ |
|-------|----------------|-----|-----|-----|------------------|
| Conv1 | 227 | 11 | 0 | 4 | $\lfloor (227-11)/4\rfloor + 1 = 54+1 = 55$ |
| MaxPool1 | 55 | 3 | 0 | 2 | $\lfloor 52/2\rfloor + 1 = 26 + 1 = 27$ |
| Conv2 | 27 | 5 | 2 | 1 | $\lfloor (27-5+4)/1\rfloor + 1 = 26 + 1 = 27$ |
| ... | ... | ... | ... | ... | ... |

> 시험에서 한 layer 주고 \"output shape 구하라\" 형식 가능.

---

## 9. 모범 답안 템플릿

```
[Setup]
1D convolution: input width W, kernel size K, padding P (each side),
stride S. Goal: derive O, the output width.

[Step 1 — Effective input after padding]
After padding: W' = W + 2P.

[Step 2 — Possible kernel start positions]
The kernel's leftmost index can be 1, 1+S, 1+2S, ...
The kernel must fully fit within W', so the leftmost index is at most
  W' - K + 1.
Thus the start positions form an arithmetic sequence with first term 1,
common difference S, last term ≤ W' - K + 1.

[Step 3 — Count terms]
Number of terms in {1, 1+S, ..., ≤ W'-K+1} is
  ⌊((W'-K+1) - 1) / S⌋ + 1 = ⌊(W'-K)/S⌋ + 1
                            = ⌊(W - K + 2P)/S⌋ + 1.

[Conclusion]
  O = ⌊(W - K + 2P)/S⌋ + 1.

[Sanity checks]
- valid mode (P=0, S=1): O = W - K + 1.   ✓
- same mode  (S=1, K odd, P=(K-1)/2): O = W. ✓
- 2D extension: apply formula independently to height and width.
```

---

## 10. 자주 틀리는 함정

1. **+1 누락** (off-by-one) → 가장 흔한 실수. \"등차수열 첫 항도 포함\" 한 줄로 정당화.
2. **Floor 생략**: $W - K + 2P$ 가 $S$ 의 배수가 아닐 때 정수 안 나옴.
3. **Padding을 한쪽만 적용으로 계산**: \"$+2P$\" 가 아닌 \"$+P$\" 로 적는 실수 — \"양쪽\" 명시.
4. **2D에서 같은 공식을 H, W에 따로 적용 안 함**: 채널은 무관, 공간만.
5. **Pooling 도 같은 공식 사용** 한 줄: pooling 도 \"커널 + stride\" 구조라 같음.

---

## 11. 연결 개념

- ← [10 Conv = Linear Transformation](10_Conv_Linear_증명.md): 출력 차원의 의미
- ↔ AlexNet 등 architecture 분석
- ↔ Pooling layer 의 출력 계산 (같은 공식 적용)

---
title: "Deep Learning Midterm Mock Exam (30 Questions)"
slug: midterm-mock-exam
order: 1
---

# Deep Learning Midterm Mock Exam (30 Questions)
## Professor Sungyoon Lee - Hanyang University
### Exam Characteristics: English, Process-oriented, Deduction → Logic → Solution

---

## Professor Analysis

**Prof. Sungyoon Lee's Exam Style:**
- PhD in Mathematics (SNU) - extremely rigorous mathematical reasoning
- Values **deduction and logical process** over final answers
- "A formula is worth a thousand pictures" philosophy
- Expects step-by-step derivations with clear justification
- Quiz-style problems requiring **complete solution process**
- Emphasizes "why" not just "how"

---

# PART 1: LINEAR ALGEBRA (Q1-Q8)

---

## Question 1. Linear Transformation Verification

**Let $L : \mathbb{R}^2 \to \mathbb{R}^2$ be defined by $L(v) = v + \begin{pmatrix} 1 \\ 0 \end{pmatrix}$. Is $L$ a linear transformation? Justify your answer with a complete proof or counterexample.**

### Solution

**[University Level]**

A linear transformation must satisfy two properties for all $u, v \in \mathbb{R}^n$ and $a \in \mathbb{R}$:
1. $L(v + u) = L(v) + L(u)$ (additivity)
2. $L(av) = aL(v)$ (homogeneity)

Check property (2): Let $a = 0$, $v = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.

$L(0 \cdot v) = L(\begin{pmatrix} 0 \\ 0 \end{pmatrix}) = \begin{pmatrix} 0 \\ 0 \end{pmatrix} + \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

$0 \cdot L(v) = 0 \cdot \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

Since $L(0 \cdot v) \neq 0 \cdot L(v)$, $L$ is **not** a linear transformation. $\square$

**[High School Level]**

A linear transformation must pass through the origin: $L(0) = 0$. But $L(0) = 0 + \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \neq 0$. So $L$ is not linear.

**[Middle School Level]**

Think of a linear transformation as a "straight-line rule" that keeps the origin fixed. If we input zero (nothing), we should get zero (nothing) out. But this function adds $(1, 0)$ to everything, so even zero becomes $(1, 0)$. That breaks the rule, so it's not a linear transformation.

---

## Question 2. Rank-Nullity Theorem Application

**Let $A \in \mathbb{R}^{3 \times 5}$ with $\text{rank}(A) = 2$. Find $\text{null}(A)$ (the dimension of the null space of $A$). Explain the relationship between the rank and nullity.**

### Solution

**[University Level]**

By the Rank-Nullity Theorem: For $A \in \mathbb{R}^{m \times n}$, $n = \text{rank}(A) + \text{null}(A)$.

Here $n = 5$, $\text{rank}(A) = 2$.

$\text{null}(A) = n - \text{rank}(A) = 5 - 2 = 3$

This means $\mathcal{N}(A) = \ker(A) = \{v \in \mathbb{R}^5 : Av = 0\}$ has dimension 3, i.e., there is a 3-dimensional subspace of $\mathbb{R}^5$ that maps to zero under $A$.

Also note: $\text{rank}(A) \leq \min(m, n) = \min(3, 5) = 3$, which is satisfied since $\text{rank}(A) = 2 \leq 3$. $\square$

**[High School Level]**

The matrix $A$ has 5 columns (5 inputs). The Rank-Nullity Theorem says: number of columns = rank + nullity. So nullity = 5 - 2 = 3. This means 3 independent directions in the input space are "crushed" to zero by the matrix.

**[Middle School Level]**

Imagine a matrix as a machine with 5 input slots. Rank = 2 means only 2 independent outputs are actually produced. The remaining 5 - 2 = 3 input directions are wasted (they produce nothing/zero). That "wasted" number is the nullity = 3.

---

## Question 3. Eigenvalue Properties

**Let $A$ be a symmetric matrix with eigenvalues $\lambda_1 = 3$ and $\lambda_2 = -1$. Determine the eigenvalues of the following: (a) $A^2$, (b) $A^{-1}$ (if it exists), (c) $A + 2I$, (d) $cA$ for a scalar $c$.**

### Solution

**[University Level]**

If $Av = \lambda v$, then:

(a) $A^2 v = A(Av) = A(\lambda v) = \lambda(Av) = \lambda^2 v$.
So eigenvalues of $A^2$: $\lambda_1^2 = 9$, $\lambda_2^2 = 1$.

(b) Since $\lambda_1 = 3 \neq 0$ and $\lambda_2 = -1 \neq 0$, $A^{-1}$ exists.
From $Av = \lambda v$: $v = \lambda A^{-1} v$, so $A^{-1} v = \frac{1}{\lambda} v$.
Eigenvalues of $A^{-1}$: $1/3$ and $-1$.

(c) $(A + 2I)v = Av + 2v = \lambda v + 2v = (\lambda + 2)v$.
Eigenvalues: $3 + 2 = 5$ and $-1 + 2 = 1$.

(d) $(cA)v = c(Av) = c\lambda v$.
Eigenvalues: $3c$ and $-c$.

The eigenvectors remain the same in all cases. $\square$

**[High School Level]**

Rule: If $\lambda$ is an eigenvalue of $A$:
- $A^2$ has eigenvalue $\lambda^2$ → $9, 1$
- $A^{-1}$ has eigenvalue $1/\lambda$ → $1/3, -1$
- $A + 2I$ has eigenvalue $\lambda + 2$ → $5, 1$
- $cA$ has eigenvalue $c\lambda$ → $3c, -c$

**[Middle School Level]**

An eigenvalue is like a "stretching factor." If the original stretch is 3, then squaring the machine stretches by $3 \times 3 = 9$. The reverse machine ($A^{-1}$) stretches by $1/3$. Adding 2I adds 2 to each stretch factor. Multiplying the machine by $c$ multiplies each stretch by $c$.

---

## Question 4. Matrix Multiplication as Outer Product

**Given $A = \begin{pmatrix} a_1^\top \\ a_2^\top \end{pmatrix}$ and $B = \begin{pmatrix} | & | \\ b_1 & b_2 \\ | & | \end{pmatrix}$, where $a_i, b_j \in \mathbb{R}^2$, write $AB$ in terms of inner products $a_i^\top b_j$. Then explain when $\text{rank}(AB) < \min(\text{rank}(A), \text{rank}(B))$ can occur.**

### Solution

**[University Level]**

$AB = \begin{pmatrix} a_1^\top b_1 & a_1^\top b_2 \\ a_2^\top b_1 & a_2^\top b_2 \end{pmatrix}$

Each entry $(AB)_{ij} = a_i^\top b_j$, the inner product of the $i$-th row of $A$ and $j$-th column of $B$.

We know $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$.

Strict inequality occurs when $\text{im}(B) \cap \ker(A) \neq \{0\}$, i.e., when some nonzero output of $B$ lies in the null space of $A$.

Example: $A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$. Both have rank 1, but $AB = 0$, rank 0. $\square$

**[High School Level]**

Matrix multiplication $AB$: each entry is a dot product of a row of $A$ with a column of $B$. The rank of $AB$ can be less than the rank of either factor when $B$'s output "falls into" $A$'s blind spot (null space).

**[Middle School Level]**

Think of two filters stacked. $B$ filters first, $A$ filters second. If $B$'s output happens to be exactly what $A$ ignores, then the combination produces nothing, even though each filter alone does something.

---

## Question 5. SVD Interpretation

**Explain in your own words what the Singular Value Decomposition (SVD) $A = U \Sigma V^\top$ means geometrically. What do $U$, $\Sigma$, and $V$ represent?**

### Solution

**[University Level]**

For $A \in \mathbb{R}^{m \times n}$, $A = U \Sigma V^\top$ where:
- $V \in \mathbb{R}^{n \times n}$ is orthogonal: its columns $v_i$ form an orthonormal basis for $\mathbb{R}^n$ (input space)
- $U \in \mathbb{R}^{m \times m}$ is orthogonal: its columns $u_i$ form an orthonormal basis for $\mathbb{R}^m$ (output space)
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$

Geometrically, any linear map $A$ can be decomposed into three steps:
1. **Rotate/reflect** the input space using $V^\top$ (align with principal axes)
2. **Scale** along each axis by $\sigma_i$ (stretch or compress)
3. **Rotate/reflect** the output space using $U$ (align with output axes)

The rank of $A$ equals the number of nonzero singular values. $\square$

**[High School Level]**

SVD says any matrix transformation = rotation → scaling → rotation. $V$ rotates the input, $\Sigma$ stretches each direction by different amounts, and $U$ rotates the result. It's like decomposing any complex motion into simple steps.

**[Middle School Level]**

Imagine reshaping a rubber sheet. SVD says any reshaping can be broken into: (1) turn the sheet, (2) stretch it wider/narrower in each direction, (3) turn it again. These three simple steps can reproduce ANY reshaping.

---

## Question 6. Pseudo-inverse and Solving $Ax = b$

**Consider the system $Ax = b$ where $A \in \mathbb{R}^{m \times n}$ with $m > n$ (overdetermined system). Explain why a solution may not exist and how the pseudo-inverse $A^+$ provides the "best" approximate solution. What is $A^+$ for full column rank $A$?**

### Solution

**[University Level]**

When $m > n$, there are more equations than unknowns, so $b$ may not lie in $\text{im}(A)$, meaning $Ax = b$ has no exact solution.

The pseudo-inverse provides the least-squares solution: $x^* = A^+ b = \arg\min_x \|Ax - b\|^2$.

For full column rank $A$ (rank $= n$), $A^\top A$ is invertible, and:
$$A^+ = (A^\top A)^{-1} A^\top$$

Derivation: Minimize $L(x) = \|Ax - b\|^2 = (Ax - b)^\top(Ax - b)$.
$\nabla_x L = 2A^\top(Ax - b) = 0 \Rightarrow A^\top A x = A^\top b \Rightarrow x = (A^\top A)^{-1} A^\top b$.

Properties: $AA^+ b = \text{Proj}(b; \text{im}(A))$, the orthogonal projection of $b$ onto the column space of $A$.

Also, $\ker(A^+A) = \ker(A)$, confirming the solution is exact when $b \in \text{im}(A)$. $\square$

**[High School Level]**

With more equations than unknowns, you can't satisfy all equations exactly. $A^+$ finds the $x$ that makes $Ax$ as close as possible to $b$ (minimizes the total error). For full column rank: $A^+ = (A^\top A)^{-1} A^\top$.

**[Middle School Level]**

Imagine trying to draw a straight line through 10 points that don't all lie on a line. You can't hit every point, but you can find the "best fit" line that is as close as possible to all points. The pseudo-inverse formula finds this best fit automatically.

---

## Question 7. Properties of Positive Definite Matrices

**Let $A$ be a symmetric positive definite (PD) matrix. Prove that all eigenvalues of $A$ are positive. Then explain why $A$ is invertible.**

### Solution

**[University Level]**

**Definition:** $A$ is PD iff $v^\top A v > 0$ for all $v \neq 0$.

**Proof:** Let $\lambda$ be an eigenvalue with eigenvector $v \neq 0$, so $Av = \lambda v$.
Then $v^\top A v = v^\top (\lambda v) = \lambda \|v\|^2$.
Since $A$ is PD, $v^\top A v > 0$, and $\|v\|^2 > 0$ (as $v \neq 0$), so $\lambda > 0$.

**Invertibility:** Since all eigenvalues are positive (hence nonzero), $\det(A) = \prod_i \lambda_i > 0 \neq 0$, so $A$ is invertible.

Alternatively, $A^{-1} = \sum_{i=1}^n \frac{1}{\lambda_i} u_i u_i^\top$ (spectral decomposition), and each $\frac{1}{\lambda_i}$ exists since $\lambda_i > 0$. $\square$

**[High School Level]**

PD means $v^\top A v > 0$ for any nonzero $v$. Plugging in an eigenvector: $\lambda \|v\|^2 > 0$, so $\lambda > 0$. All positive eigenvalues → determinant $\neq 0$ → matrix is invertible.

**[Middle School Level]**

A positive definite matrix is like a "bowl" shape — everything curves upward. The eigenvalues tell you how steep the bowl is in each direction. Since it's always curving up, all steepnesses (eigenvalues) must be positive, and no direction is flat (zero), meaning the matrix can be "undone" (inverted).

---

## Question 8. Frobenius Norm and Trace

**Show that $\|A\|_F^2 = \text{Tr}(A^\top A)$ for any matrix $A \in \mathbb{R}^{m \times n}$. Also show $\mathbb{E}[\|Ax\|^2] = \text{Tr}(A^\top A)$ when $x \sim \mathcal{N}(0, I)$.**

### Solution

**[University Level]**

**Part 1:**
$\|A\|_F^2 = \sum_{i,j} a_{ij}^2$

$(A^\top A)_{jj} = \sum_{i=1}^m a_{ij}^2$, so $\text{Tr}(A^\top A) = \sum_{j=1}^n \sum_{i=1}^m a_{ij}^2 = \sum_{i,j} a_{ij}^2 = \|A\|_F^2$. $\square$

**Part 2:**
$\mathbb{E}[\|Ax\|^2] = \mathbb{E}[x^\top A^\top A x] = \mathbb{E}[\text{Tr}(x^\top A^\top A x)]$

Since $x^\top A^\top A x$ is a scalar, $\text{Tr}(x^\top A^\top A x) = \text{Tr}(A^\top A x x^\top)$ (cyclic property of trace).

$= \text{Tr}(A^\top A \cdot \mathbb{E}[x x^\top]) = \text{Tr}(A^\top A \cdot I) = \text{Tr}(A^\top A) = \|A\|_F^2$. $\square$

**[High School Level]**

The Frobenius norm is "sum of all squared entries." The trace of $A^\top A$ is "sum of diagonal entries of $A^\top A$," and each diagonal entry sums the squares of a column of $A$. So they're the same thing.

For the expectation: since $x$ is random with identity covariance ($\mathbb{E}[xx^\top] = I$), we use the trace trick to swap the order, and the randomness averages out to give $\text{Tr}(A^\top A)$.

**[Middle School Level]**

The Frobenius norm measures the "total size" of a matrix by squaring every number in it and adding them up. The trace trick is a mathematical shortcut: instead of computing the complicated random quantity directly, we rearrange the formula so the randomness cancels out neatly.

---

# PART 2: MATRIX CALCULUS (Q9-Q13)

---

## Question 9. Chain Rule for Matrix Calculus

**Let $z = BAx$ where $B \in \mathbb{R}^{p \times m}$, $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$. Compute $\frac{\partial z}{\partial x}$ using the chain rule. Verify by direct computation.**

### Solution

**[University Level]**

**Method 1 (Chain Rule):**
Let $y = Ax \in \mathbb{R}^m$, then $z = By$.

$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x} = B \cdot A = BA$

(where $\frac{\partial z}{\partial y} = B \in \mathbb{R}^{p \times m}$ and $\frac{\partial y}{\partial x} = A \in \mathbb{R}^{m \times n}$)

**Method 2 (Direct):**
$z = BAx$, and $BA \in \mathbb{R}^{p \times n}$ is a constant matrix.

$\frac{\partial z}{\partial x} = BA \in \mathbb{R}^{p \times n}$

Both methods give the same result, confirming the chain rule. $\square$

The Jacobian $\frac{\partial z}{\partial x} \in \mathbb{R}^{p \times n}$ has dimension $d(z) \times d(x)$.

**[High School Level]**

Chain rule: "derivative of outer function × derivative of inner function." Here outer = multiply by $B$, inner = multiply by $A$. Result: $B \times A = BA$.

**[Middle School Level]**

If you do two operations in a row (first multiply by $A$, then by $B$), the combined effect is multiplying by $BA$. The derivative (rate of change) of a linear operation is just the operation itself.

---

## Question 10. Gradient of Quadratic Form

**Compute $\nabla_x (x^\top A x)$ where $A$ is a symmetric matrix and $x \in \mathbb{R}^n$.**

### Solution

**[University Level]**

$f(x) = x^\top A x = \sum_{i,j} a_{ij} x_i x_j$

**Method 1 (Product Rule):**
$\frac{\partial (x^\top A x)}{\partial x} = \frac{\partial (x^\top)}{\partial x} (Ax) + x^\top \frac{\partial (Ax)}{\partial x}$

Using the product rule for matrix calculus:
$= (Ax)^\top + x^\top A = x^\top A^\top + x^\top A$

Taking the transpose to get the gradient:
$\nabla_x (x^\top A x) = (A^\top + A)x$

For symmetric $A$ ($A^\top = A$):
$$\nabla_x (x^\top A x) = 2Ax$$

**Method 2 (Direct):**
$f(x + \epsilon) = (x + \epsilon)^\top A(x + \epsilon) = x^\top Ax + x^\top A\epsilon + \epsilon^\top Ax + \epsilon^\top A\epsilon$
$= f(x) + (x^\top A + x^\top A^\top)\epsilon + O(\|\epsilon\|^2)$
$= f(x) + 2x^\top A \epsilon + O(\|\epsilon\|^2)$

So $\nabla_x f = 2Ax$. $\square$

**[High School Level]**

For a symmetric matrix, $\nabla_x (x^\top A x) = 2Ax$. Think of it as the matrix version of $\frac{d}{dx}(ax^2) = 2ax$.

**[Middle School Level]**

In regular math, if $f(x) = ax^2$, the derivative is $2ax$. In matrix math, $x^\top A x$ is the "matrix version" of $ax^2$, so the derivative is $2Ax$. The matrix $A$ plays the role of the number $a$.

---

## Question 11. Jacobian of Softmax

**Let $p_i = \frac{\exp(z_i)}{\sum_{k=1}^C \exp(z_k)}$ for $z, p \in \mathbb{R}^C$. Compute $\frac{\partial p_i}{\partial z_j}$ for (a) $i = j$ and (b) $i \neq j$.**

### Solution

**[University Level]**

Let $Z = \sum_{k=1}^C \exp(z_k)$.

**(a) $i = j$:**
$\frac{\partial p_i}{\partial z_i} = \frac{\exp(z_i) \cdot Z - \exp(z_i) \cdot \exp(z_i)}{Z^2}$ (quotient rule)
$= \frac{\exp(z_i)}{Z} \cdot \frac{Z - \exp(z_i)}{Z}$
$= p_i(1 - p_i)$

**(b) $i \neq j$:**
$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot Z - \exp(z_i) \cdot \exp(z_j)}{Z^2}$
$= -\frac{\exp(z_i)}{Z} \cdot \frac{\exp(z_j)}{Z}$
$= -p_i p_j$

**Compact form:** $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$, or in matrix form:
$$\frac{\partial p}{\partial z} = \text{diag}(p) - p p^\top$$

$\square$

**[High School Level]**

Use the quotient rule on $p_i = e^{z_i} / \sum e^{z_k}$:
- Same index ($i = j$): $p_i(1 - p_i)$ — like the derivative of logistic function
- Different index ($i \neq j$): $-p_i p_j$ — increasing $z_j$ decreases $p_i$

**[Middle School Level]**

Softmax converts numbers into probabilities that sum to 1. If you increase one input $z_j$, its probability $p_j$ goes up (by $p_j(1-p_j)$), and all other probabilities go down (by $p_i \times p_j$). The total still sums to 1.

---

## Question 12. Derivative of $\|f(x)\|^2$

**Show that $\frac{\partial \|f(x)\|^2}{\partial x} = 2f(x)^\top \frac{\partial f(x)}{\partial x}$ where $f : \mathbb{R}^{d(x)} \to \mathbb{R}^{d(y)}$.**

### Solution

**[University Level]**

$\|f(x)\|^2 = f(x)^\top f(x)$

Using the product rule for matrix derivatives:
$\frac{\partial [f(x)^\top f(x)]}{\partial x} = \frac{\partial f(x)^\top}{\partial x} f(x) + f(x)^\top \frac{\partial f(x)}{\partial x}$

Since $\frac{\partial f^\top}{\partial x} f = (f^\top \frac{\partial f}{\partial x})^\top$, and both terms are transposes of each other summed, but more carefully:

$\frac{\partial}{\partial x}\|f\|^2 = 2 f(x)^\top \frac{\partial f(x)}{\partial x}$

This is a $1 \times d(x)$ row vector (the derivative of a scalar w.r.t. $x$).

Dimensions check: $f(x)^\top$ is $1 \times d(y)$, $\frac{\partial f}{\partial x}$ is $d(y) \times d(x)$, product is $1 \times d(x)$. $\checkmark$ $\square$

**[High School Level]**

This is the matrix version of $\frac{d}{dx}[g(x)]^2 = 2g(x) \cdot g'(x)$. Replace scalar $g$ with vector $f$, and scalar multiplication with dot product/matrix multiplication.

**[Middle School Level]**

In basic math, $(g^2)' = 2g \cdot g'$. The same pattern works for vectors: the derivative of "length squared" of $f$ is $2 \times f \times (\text{rate of change of } f)$.

---

## Question 13. Computing $\frac{\partial z}{\partial x}$ where $z = BAx$

**Let $z = BAx$ where $B \in \mathbb{R}^{p \times m}$, $A \in \mathbb{R}^{m \times n}$. What is the dimension of $\frac{\partial z}{\partial x}$? Write it explicitly.**

### Solution

**[University Level]**

$z \in \mathbb{R}^p$, $x \in \mathbb{R}^n$, so $\frac{\partial z}{\partial x} \in \mathbb{R}^{p \times n}$ (numerator layout convention).

$z = (BA)x$ where $BA \in \mathbb{R}^{p \times n}$.

Since $z$ is linear in $x$: $\frac{\partial z}{\partial x} = BA$.

Verification: $z_i = \sum_j (BA)_{ij} x_j$, so $\frac{\partial z_i}{\partial x_j} = (BA)_{ij}$, giving the matrix $BA$. $\square$

**[High School Level]**

Dimension = (output dimension) × (input dimension) = $p \times n$. The Jacobian of a linear function $Mx$ is just $M$.

**[Middle School Level]**

If a machine multiplies input by some number (say 3), the "rate of change" is just 3. Similarly, if a matrix multiplies $x$ by $BA$, the rate of change is $BA$.

---

# PART 3: CALCULUS & OPTIMIZATION (Q14-Q18)

---

## Question 14. Newton's Method

**Derive Newton's method for finding zeros of $f(x)$. Starting from $x_0 = 2$, apply one iteration to find an approximate root of $f(x) = x^2 - 3$.**

### Solution

**[University Level]**

**Derivation:** At point $x_n$, the tangent line to $f$ is:
$y = f'(x_n)(x - x_n) + f(x_n)$

Setting $y = 0$ to find where the tangent crosses the x-axis:
$0 = f'(x_n)(x_{n+1} - x_n) + f(x_n)$
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

**Application:** $f(x) = x^2 - 3$, $f'(x) = 2x$.

$x_0 = 2$:
$x_1 = 2 - \frac{2^2 - 3}{2 \cdot 2} = 2 - \frac{1}{4} = 1.75$

Check: $f(1.75) = 3.0625 - 3 = 0.0625$ (close to 0, actual $\sqrt{3} \approx 1.732$).

**Connection to optimization:** If $f(x) = \nabla_x L(x)$, Newton's method becomes:
$x_{n+1} = x_n - [H(x_n)]^{-1} \nabla L(x_n)$
where $H$ is the Hessian. $\square$

**[High School Level]**

Draw a tangent line at the current guess, see where it hits the x-axis, and use that as the next guess. Formula: $x_{n+1} = x_n - f(x_n)/f'(x_n)$. With $x_0 = 2$: $x_1 = 2 - 1/4 = 1.75$.

**[Middle School Level]**

To find $\sqrt{3}$, start with guess 2. The function $x^2 - 3$ tells us how far off we are ($2^2 - 3 = 1$, too high). Using the slope ($2 \times 2 = 4$), we adjust: $2 - 1/4 = 1.75$. Already very close to $\sqrt{3} \approx 1.732$!

---

## Question 15. Lagrange Multipliers

**Minimize $f(x, y) = x^2 + y^2$ subject to the constraint $x + y = 1$. Use the method of Lagrange multipliers. Provide the full derivation.**

### Solution

**[University Level]**

Define the Lagrangian: $\mathcal{L}(x, y, \lambda) = x^2 + y^2 + \lambda(x + y - 1)$

Set all partial derivatives to zero:
$$\frac{\partial \mathcal{L}}{\partial x} = 2x + \lambda = 0 \quad \Rightarrow \quad x = -\lambda/2$$
$$\frac{\partial \mathcal{L}}{\partial y} = 2y + \lambda = 0 \quad \Rightarrow \quad y = -\lambda/2$$
$$\frac{\partial \mathcal{L}}{\partial \lambda} = x + y - 1 = 0$$

From equations (1) and (2): $x = y = -\lambda/2$.
Substituting into (3): $-\lambda/2 + (-\lambda/2) = 1 \Rightarrow -\lambda = 1 \Rightarrow \lambda = -1$.

Therefore $x = y = 1/2$, and $f(1/2, 1/2) = 1/4 + 1/4 = 1/2$.

**Geometric interpretation:** $\nabla f = (2x, 2y)$ must be parallel to $\nabla g = (1, 1)$ at the optimum. At $(1/2, 1/2)$: $\nabla f = (1, 1) = -\lambda \cdot (1, 1)$, confirming $\lambda = -1$.

Note: $f$ has no maximum along the line (it grows without bound). $\square$

**[High School Level]**

Lagrange multipliers: at the optimal point, the gradient of $f$ must be parallel to the gradient of the constraint. Set up $\mathcal{L} = x^2 + y^2 + \lambda(x + y - 1)$, solve the system of 3 equations → $(x, y) = (1/2, 1/2)$, minimum value $= 1/2$.

**[Middle School Level]**

We want to find the point on the line $x + y = 1$ that is closest to the origin. By symmetry, it must be where $x = y$. So $x + y = 1$ gives $x = y = 1/2$. The distance squared is $1/4 + 1/4 = 1/2$.

---

## Question 16. Deriving Softmax from Constrained Optimization

**Show that the softmax function $p_i = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}$ arises from maximizing $f(p) = \sum_i p_i z_i + \tau H(p)$ subject to $\sum_i p_i = 1$, $p_i \geq 0$, where $H(p) = -\sum_i p_i \log p_i$ is the entropy.**

### Solution

**[University Level]**

Set up the Lagrangian:
$\mathcal{L}(p, \lambda) = \sum_i p_i z_i + \tau(-\sum_i p_i \log p_i) + \lambda(1 - \sum_i p_i)$

$= \sum_i [p_i z_i - \tau p_i \log p_i] + \lambda(1 - \sum_i p_i)$

Taking derivative w.r.t. $p_i$ and setting to zero:
$\frac{\partial \mathcal{L}}{\partial p_i} = z_i - \tau(\log p_i + 1) - \lambda = 0$

$\Rightarrow \log p_i = \frac{z_i - \lambda - \tau}{\tau}$

$\Rightarrow p_i = \exp\left(\frac{z_i - \lambda - \tau}{\tau}\right) = \frac{1}{Z}\exp\left(\frac{z_i}{\tau}\right)$

where $Z = \exp((\lambda + \tau)/\tau)$. Using constraint $\sum_i p_i = 1$:

$Z = \sum_j \exp(z_j / \tau)$

Therefore: $p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$

This is the softmax with temperature $\tau$. As $\tau \to 0$, $p$ approaches argmax; as $\tau \to \infty$, $p$ approaches uniform distribution. $\square$

**[High School Level]**

We want to find probabilities $p_i$ that balance "choosing high-scoring options" ($\sum p_i z_i$) and "being uncertain" (entropy $H$). Using Lagrange multipliers and solving the log equation gives the softmax formula. Temperature $\tau$ controls the balance.

**[Middle School Level]**

Softmax is a "soft" way to pick the best option. If we want to choose the option with the highest score but also keep some randomness, softmax does exactly that. The math shows this is the *optimal* balance between being decisive and being uncertain.

---

## Question 17. Gradient Descent Update

**Derive the gradient descent update rule from the first-order Taylor expansion of $L(\theta)$. Explain why the learning rate $\eta$ must be chosen carefully.**

### Solution

**[University Level]**

First-order Taylor expansion at $\theta_t$:
$L(\theta) \approx L(\theta_t) + \nabla L(\theta_t)^\top (\theta - \theta_t)$

We want to choose $\theta_{t+1}$ to decrease $L$. The direction of steepest descent is $-\nabla L(\theta_t)$ (negative gradient).

Taking a step of size $\eta$ in that direction:
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Then: $L(\theta_{t+1}) \approx L(\theta_t) - \eta \|\nabla L(\theta_t)\|^2 < L(\theta_t)$ (guaranteed decrease if $\eta$ is small enough).

**Why $\eta$ matters:**
- Too large: Taylor approximation breaks down, $L$ may increase (divergence)
- Too small: Very slow convergence
- For $L$-smooth functions ($\|\nabla^2 L\| \leq L$), convergence is guaranteed for $\eta \leq 1/L$
- Connection to Newton's method: $\eta = [H]^{-1}$ is the "optimal" learning rate (second-order method) $\square$

**[High School Level]**

Gradient descent: go downhill by moving opposite to the slope. Step size ($\eta$) matters: too big = overshoot, too small = too slow. The update is $\theta_{new} = \theta_{old} - \eta \times \text{gradient}$.

**[Middle School Level]**

Imagine you're on a hill and want to reach the bottom, but you're blindfolded. You feel the slope beneath your feet and take a step downhill. The "learning rate" is how big a step you take. Too big and you might jump over the valley; too small and it takes forever.

---

## Question 18. Differential Equations and Sigmoid

**The ODE $x' = a(x - \alpha)(x - \beta)$ with $\alpha \neq \beta$ has a sigmoidal solution. Derive the solution and explain why the sigmoid shape arises.**

### Solution

**[University Level]**

Separate variables: $\frac{1}{(x - \alpha)(x - \beta)} dx = a \, dt$

Partial fractions: $\frac{1}{(x-\alpha)(x-\beta)} = \frac{1}{\alpha - \beta}\left(\frac{1}{x-\alpha} - \frac{1}{x-\beta}\right)$

Integrate: $a t + C = \frac{1}{\alpha - \beta}[\log(x-\alpha) - \log(x-\beta)] = \frac{1}{\alpha - \beta}\log\frac{x-\alpha}{x-\beta}$

$(\alpha - \beta)at + C' = \log\frac{x - \alpha}{x - \beta}$

$\exp((\alpha-\beta)at + C') = \frac{x - \alpha}{x - \beta}$

Solving for $x$:
$$x(t) = \frac{\alpha - \beta}{1 - C\exp((\alpha - \beta)at)} + \beta$$

**Why sigmoid:** The solution transitions smoothly between the two fixed points $\alpha$ and $\beta$. Between them, the growth is initially exponential (like the logistic equation), then saturates as it approaches the stable equilibrium. This S-curve is the sigmoid shape.

When $\alpha = 1, \beta = 0, a = 1$: $x(t) = \frac{1}{1 + e^{-t}} = \sigma(t)$, the standard sigmoid. $\square$

**[High School Level]**

The equation has two equilibrium points ($\alpha$ and $\beta$). One is stable, one unstable. The solution smoothly transitions between them in an S-shape (sigmoid), because growth speeds up at first, then slows as it approaches the stable point.

**[Middle School Level]**

Think of a population that grows fast when small but levels off when it reaches a limit. The S-shaped curve (sigmoid) shows slow start → rapid growth → leveling off. The formula shows exactly how this happens mathematically.

---

# PART 4: PROBABILITY (Q19-Q25)

---

## Question 19. Bayes' Theorem Application

**A test for a disease has a 99% true positive rate and 2% false positive rate. If 1% of the population has the disease, what is the probability that someone who tests positive actually has the disease? Show your reasoning step by step.**

### Solution

**[University Level]**

Let $D$ = has disease, $T^+$ = tests positive.

Given: $P(T^+ | D) = 0.99$, $P(T^+ | D^c) = 0.02$, $P(D) = 0.01$.

By Bayes' Theorem:
$$P(D | T^+) = \frac{P(T^+ | D)P(D)}{P(T^+)}$$

$P(T^+) = P(T^+|D)P(D) + P(T^+|D^c)P(D^c)$
$= 0.99 \times 0.01 + 0.02 \times 0.99$
$= 0.0099 + 0.0198 = 0.0297$

$$P(D | T^+) = \frac{0.0099}{0.0297} = \frac{1}{3} \approx 0.333$$

**Key insight:** Despite the 99% accuracy, a positive test only means ~33% chance of disease because the disease is rare (low prior). The prior $P(D)$ dramatically affects the posterior. This illustrates why **prior matters** in Bayesian reasoning: $P(H) \to P(H|E)$. $\square$

**[High School Level]**

In 10,000 people: 100 have the disease (99 test positive), 9,900 don't (198 test positive by mistake). Total positive = 297. Of those, only 99 actually have the disease. So $99/297 = 1/3 \approx 33\%$.

**[Middle School Level]**

Imagine 10,000 people. Only 100 are sick. The test correctly catches 99 of them. But it also falsely flags 198 healthy people. So out of 297 positive results, only 99 are truly sick. That's about 1 in 3 — surprising but true because the disease is so rare.

---

## Question 20. MLE for Gaussian

**Given i.i.d. samples $x_1, \ldots, x_n$ from $\mathcal{N}(\mu, \sigma^2)$, derive the MLE estimators $\hat{\mu}_{ML}$ and $\hat{\sigma}^2_{ML}$. Is $\hat{\sigma}^2_{ML}$ an unbiased estimator?**

### Solution

**[University Level]**

Log-likelihood:
$\ell(\mu, \sigma^2) = \log \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)$
$= n \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$

**For $\mu$:** $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$
$\Rightarrow \hat{\mu}_{ML} = \frac{1}{n}\sum_{i=1}^n x_i$ (sample mean)

**For $\sigma^2$:** $\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0$
$\Rightarrow \hat{\sigma}^2_{ML} = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2$

**Bias check:**
$\mathbb{E}[\hat{\mu}_{ML}] = \mu$ → **unbiased** $\checkmark$
$\mathbb{E}[\hat{\sigma}^2_{ML}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2$ → **biased** $\times$

Unbiased estimator: $\hat{\sigma}^2_{unb} = \frac{n}{n-1}\hat{\sigma}^2_{ML} = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ $\square$

**[High School Level]**

MLE = find parameters that make the observed data most likely. For Gaussian: $\hat{\mu} = \text{sample mean}$, $\hat{\sigma}^2 = \text{sample variance with } n$. The variance estimate is biased (slightly too small on average); dividing by $n-1$ instead of $n$ fixes this.

**[Middle School Level]**

Given data points, the best guess for the average is just the actual average of the data. The best guess for spread uses the average of squared differences from the mean, but dividing by $n$ gives a slightly too-small answer. Dividing by $n-1$ corrects this.

---

## Question 21. MAP vs MLE: Coin Tossing

**You toss a coin $n = 3$ times and get $k = 3$ heads. Compare the MLE and MAP estimates of $\theta$ (probability of heads) when using the prior $p(\theta) \propto \theta(1 - \theta)$. Explain why MAP is preferable here.**

### Solution

**[University Level]**

**MLE:** $\hat{\theta}_{ML} = k/n = 3/3 = 1$

This predicts all future tosses will be heads — clearly unreasonable.

**MAP:** Posterior $\propto$ likelihood $\times$ prior:
$p(\theta | \text{data}) \propto \theta^k (1-\theta)^{n-k} \cdot \theta(1-\theta) = \theta^{k+1}(1-\theta)^{n-k+1}$

This is a Beta distribution $\text{Beta}(k+2, n-k+2)$.

MAP = mode of Beta:
$\hat{\theta}_{MAP} = \frac{k+1}{n+2} = \frac{4}{5} = 0.8$

**Why MAP is better:**
- MLE assigns probability 0 to tails, which is extreme given only 3 observations
- MAP incorporates the prior belief that extreme values (0 or 1) are unlikely
- The prior $\theta(1-\theta)$ is the Beta(2,2) prior, which is symmetric and peaked at 0.5
- MAP acts as regularization: it "shrinks" the estimate toward 0.5
- As $n \to \infty$, MAP $\to$ MLE (data dominates prior) $\square$

**[High School Level]**

MLE says $\theta = 1$ (always heads), which seems wrong for just 3 flips. MAP adds a "common sense prior" that extreme probabilities are unlikely, giving $\theta = 0.8$ — more reasonable. More data → MAP and MLE converge.

**[Middle School Level]**

If you flip a coin 3 times and get all heads, MLE says "this coin always lands heads!" MAP says "probably mostly heads, but let's not be too extreme" and guesses 80%. MAP is more sensible because 3 flips isn't enough to be sure.

---

## Question 22. Expected Value and Variance

**Let $X$ and $Y$ be independent random variables with $\mathbb{E}[X] = 2$, $\text{Var}[X] = 3$, $\mathbb{E}[Y] = -1$, $\text{Var}[Y] = 4$. Compute $\mathbb{E}[3X - 2Y + 5]$ and $\text{Var}[3X - 2Y + 5]$.**

### Solution

**[University Level]**

**Expected Value (linearity):**
$\mathbb{E}[3X - 2Y + 5] = 3\mathbb{E}[X] - 2\mathbb{E}[Y] + 5$
$= 3(2) - 2(-1) + 5 = 6 + 2 + 5 = 13$

**Variance:**
Since $X, Y$ are independent:
$\text{Var}[3X - 2Y + 5] = 3^2 \text{Var}[X] + (-2)^2 \text{Var}[Y] + 0$
$= 9(3) + 4(4) = 27 + 16 = 43$

Key properties used:
- $\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$
- $\text{Var}[aX + b] = a^2 \text{Var}[X]$ (constants don't add variance)
- For independent: $\text{Var}[X + Y] = \text{Var}[X] + \text{Var}[Y]$ $\square$

**[High School Level]**

Expectation is linear: multiply/add through. Variance: constants squared go in front, the constant +5 disappears (no randomness), and independent variances add. Result: $\mathbb{E} = 13$, $\text{Var} = 43$.

**[Middle School Level]**

Expected value: treat it like a regular equation with the averages: $3 \times 2 - 2 \times (-1) + 5 = 13$. Variance: measures "spread." Multiplying by 3 triples the spread (variance × 9), the minus sign doesn't matter for spread (still × 4), and adding 5 doesn't change spread. Total spread: $27 + 16 = 43$.

---

## Question 23. Gaussian Distribution and CLT

**Explain two reasons why the Normal (Gaussian) distribution is called "normal." Provide mathematical justification for each.**

### Solution

**[University Level]**

**Reason 1: Maximum Entropy**

Among all distributions with a given mean $\mu$ and variance $\sigma^2$, the Gaussian distribution maximizes the differential entropy:
$H(p) = -\int p(x) \log p(x) dx$

This can be shown via calculus of variations / Lagrange multipliers with constraints $\int p(x)dx = 1$, $\int x \cdot p(x)dx = \mu$, $\int (x-\mu)^2 p(x)dx = \sigma^2$. The Gaussian is the "most uncertain" (least biased) distribution given only mean and variance information.

**Reason 2: Central Limit Theorem (CLT)**

For i.i.d. random variables $X_1, \ldots, X_n$ with mean $\mu$ and finite variance $\sigma^2$:
$$Z_n = \sqrt{n} \frac{\hat{\mu} - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

where $\hat{\mu} = \frac{1}{n}\sum X_i$. This means sums/averages of many independent random variables tend toward a Gaussian, regardless of the original distribution. This ubiquity makes it "normal." $\square$

**[High School Level]**

1. **Most uncertain**: Given only mean and variance, the Gaussian is the distribution that makes the fewest additional assumptions (maximum entropy).
2. **CLT**: Average enough random numbers from *any* distribution, and the result looks Gaussian. That's why it appears everywhere in nature — it's "normal."

**[Middle School Level]**

1. The bell curve is the "fairest" shape: if you only know the average and spread, the Gaussian is the most balanced guess. 2. If you add up lots of random things (like heights of people, or measurement errors), the result always looks like a bell curve. That's why it's called "normal" — it's the most common shape in nature.

---

## Question 24. Chebyshev's Inequality

**State Chebyshev's inequality for the sample mean. Use it to determine how many i.i.d. samples from $[0, 1]$ are needed to ensure the sample mean is within 0.1 of the true mean with probability at least 0.95.**

### Solution

**[University Level]**

**Chebyshev's Inequality (Sample Mean):**
For i.i.d. $X_1, \ldots, X_n$ with $P(X_i \in [a,b]) = 1$:

$$P(|\hat{\mu}_n - \mathbb{E}[X_1]| \geq \epsilon) \leq \frac{(b-a)^2}{4n\epsilon^2}$$

**Application:** $[a,b] = [0,1]$, $\epsilon = 0.1$, want $P(\text{error} \geq \epsilon) \leq 0.05$ (i.e., $\delta = 0.05$).

$\frac{(1-0)^2}{4n(0.1)^2} \leq 0.05$

$\frac{1}{0.04n} \leq 0.05$

$n \geq \frac{1}{0.04 \times 0.05} = \frac{1}{0.002} = 500$

**Need at least 500 samples.**

Compare with Hoeffding (tighter bound):
$2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right) \leq 0.05$

$n \geq \frac{(b-a)^2 \log(2/\delta)}{2\epsilon^2} = \frac{1 \cdot \log(40)}{0.02} \approx \frac{3.69}{0.02} \approx 185$

Hoeffding only needs ~185 samples for the same guarantee. $\square$

**[High School Level]**

Chebyshev says: the probability the sample mean is off by more than $\epsilon$ is at most $(b-a)^2/(4n\epsilon^2)$. Set this $\leq 0.05$, solve for $n$: $n \geq 500$. The Hoeffding bound is tighter and gives $n \geq 185$.

**[Middle School Level]**

To estimate the average of some random number between 0 and 1, and be 95% sure your estimate is within 0.1 of the truth, you need at least 500 samples (by the basic formula). A fancier formula (Hoeffding) says 185 samples is actually enough.

---

## Question 25. KL Divergence and Cross-Entropy

**Define the KL divergence $KL(p \| q)$ and the cross-entropy $CE(p, q)$. Show that $KL(p \| q) = CE(p, q) - H(p)$ where $H(p)$ is the entropy. Why is KL divergence always non-negative?**

### Solution

**[University Level]**

**Definitions:**
- Entropy: $H(p) = -\sum_x p(x) \log p(x) = -\mathbb{E}_{x \sim p}[\log p(x)]$
- Cross-Entropy: $CE(p, q) = -\sum_x p(x) \log q(x) = -\mathbb{E}_{x \sim p}[\log q(x)]$
- KL Divergence: $KL(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$

**Relationship:**
$KL(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \sum_x p(x) [\log p(x) - \log q(x)]$
$= -H(p) + CE(p, q)$

Therefore: $KL(p \| q) = CE(p, q) - H(p)$ $\square$

**Non-negativity (Gibbs' Inequality):**
By Jensen's inequality (since $-\log$ is convex):
$KL(p \| q) = -\mathbb{E}_p\left[\log \frac{q(x)}{p(x)}\right] \geq -\log \mathbb{E}_p\left[\frac{q(x)}{p(x)}\right] = -\log \sum_x p(x) \frac{q(x)}{p(x)} = -\log \sum_x q(x) = -\log 1 = 0$

Equality holds iff $p = q$. $\square$

**In ML context:** Minimizing $CE(p_E, p_\theta)$ over $\theta$ is equivalent to minimizing $KL(p_E \| p_\theta)$ since $H(p_E)$ is constant. This is equivalent to MLE.

**[High School Level]**

KL divergence measures how different two distributions are. It equals cross-entropy minus entropy: $KL = CE - H$. Since $CE$ is always at least as large as $H$ (using extra bits from wrong distribution $q$), $KL \geq 0$.

**[Middle School Level]**

Entropy is the "ideal" cost to encode messages from distribution $p$. Cross-entropy is the cost using distribution $q$'s encoding instead. KL divergence = the *extra* cost of using the wrong encoding. Since using the wrong code can't be *better* than the ideal, $KL$ is always $\geq 0$.

---

# PART 5: BAYESIAN PROBABILITY & INFORMATION THEORY (Q26-Q30)

---

## Question 26. Prior, Likelihood, Posterior

**In the framework $\log p(H|E) = \log p(E|H) + \log p(H) - \log p(E)$, explain the role of each term. Why does the lecture say "It Takes Two to Tango" (ML and Prior)?**

### Solution

**[University Level]**

The terms in MAP estimation:
- $\log p(H|E)$: **Posterior** — our updated belief about hypothesis $H$ after observing evidence $E$
- $\log p(E|H)$: **Log-likelihood** — how well $H$ explains the data. Maximizing this alone = MLE
- $\log p(H)$: **Log-prior** — our belief about $H$ before seeing data. Acts as regularization
- $\log p(E)$: **Log-evidence** (marginal likelihood) — normalization constant, independent of $H$

**"It Takes Two to Tango":**

Learning requires both:
1. **Maximum Likelihood** (data-driven): How well does the hypothesis explain the observations? → Optimization (loss)
2. **Prior** (knowledge-driven): What do we believe a priori? → Regularization, generalization

In neural network training: loss function = negative log-likelihood, weight decay = Gaussian prior, dropout = implicit prior. The interplay of these two components (fitting data + incorporating prior knowledge) is essential for good generalization.

Without ML: no learning from data. Without prior: overfitting (as in the coin example where MLE gives $\theta = 1$ for 3/3 heads). $\square$

**[High School Level]**

MAP = ML + Prior. The likelihood fits the data, the prior prevents extreme answers. Like "Two to Tango" — both partners (data and knowledge) are needed for good learning.

**[Middle School Level]**

Learning is like forming an opinion: you look at evidence (data/likelihood) BUT you also use common sense (prior knowledge). If someone gets all A's on 3 tests, the data says "perfect student" but common sense says "probably very good, not necessarily perfect." You need both.

---

## Question 27. Entropy Maximization

**Prove that the discrete uniform distribution on a finite set $S$ with $|S| = K$ maximizes the entropy $H(p) = -\sum_{i=1}^K p_i \log p_i$ among all distributions on $S$.**

### Solution

**[University Level]**

Use Lagrange multipliers. Maximize $H(p) = -\sum_i p_i \log p_i$ subject to $g(p) = \sum_i p_i - 1 = 0$.

$\mathcal{L}(p, \lambda) = -\sum_i p_i \log p_i + \lambda(1 - \sum_i p_i)$

$\frac{\partial \mathcal{L}}{\partial p_i} = -\log p_i - 1 - \lambda = 0$

$\Rightarrow \log p_i = -(1 + \lambda)$ for all $i$

$\Rightarrow p_i = e^{-(1+\lambda)}$ = constant for all $i$

Using $\sum p_i = 1$: $K \cdot p_i = 1 \Rightarrow p_i = 1/K$ (uniform).

$H_{max} = -\sum_{i=1}^K \frac{1}{K} \log \frac{1}{K} = \log K$.

**Alternative proof using KL:** For any distribution $p$, let $u$ be uniform.
$KL(p \| u) = \sum p_i \log \frac{p_i}{1/K} = \sum p_i \log p_i + \log K = -H(p) + \log K \geq 0$

Therefore $H(p) \leq \log K$, with equality iff $p = u$. $\square$

**[High School Level]**

Setting derivatives to zero shows all $p_i$ must be equal. Since they sum to 1, $p_i = 1/K$. Maximum entropy = $\log K$. Alternatively: KL divergence from uniform $\geq 0$ implies entropy $\leq \log K$.

**[Middle School Level]**

Entropy measures "surprise" or "uncertainty." You're most uncertain when every outcome is equally likely (like a fair die). Any bias toward some outcomes makes things more predictable (less entropy). So the uniform distribution = maximum uncertainty = maximum entropy.

---

## Question 28. Mutual Information

**Define the mutual information $I(X; Y)$ and show that $I(X; Y) = KL(p(x, y) \| p(x)p(y))$. What does $I(X; Y) = 0$ mean?**

### Solution

**[University Level]**

**Definition:**
$I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$

$= \mathbb{E}_{(x,y) \sim p(x,y)}\left[\log \frac{p(x, y)}{p(x)p(y)}\right]$

This is exactly $KL(p(x,y) \| p(x)p(y))$ by definition of KL divergence. $\square$

**Alternative expressions:**
$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$

Also: $I(X; Y) = \mathbb{E}_{(x,y) \sim p(x,y)}\left[\log \frac{p(x|y)}{p(x)}\right]$

**$I(X; Y) = 0$ means:**
$KL(p(x,y) \| p(x)p(y)) = 0 \Leftrightarrow p(x,y) = p(x)p(y)$

i.e., $X$ and $Y$ are **independent**. Knowing $Y$ provides zero information about $X$ and vice versa.

Since $KL \geq 0$, we always have $I(X;Y) \geq 0$, with equality iff independence. $\square$

**[High School Level]**

Mutual information measures how much knowing one variable tells you about the other. It's the KL divergence between the joint distribution and the product of marginals. $I = 0$ means the variables are independent.

**[Middle School Level]**

Mutual information answers: "Does knowing X help predict Y?" If yes, $I > 0$. If knowing X tells you absolutely nothing about Y, $I = 0$ (they're independent). Example: temperature and ice cream sales have high $I$; temperature and your birthday have $I \approx 0$.

---

## Question 29. Law of Total Expectation

**State and prove the Law of Total Expectation $\mathbb{E}_X[X] = \mathbb{E}_Y[\mathbb{E}_{X|Y}[X|Y]]$. Apply it to compute $\mathbb{E}[X]$ where $X | N \sim \mathcal{N}(N, 1)$ and $N \sim \text{Poisson}(\lambda)$.**

### Solution

**[University Level]**

**Proof (continuous case):**
$\mathbb{E}_Y[\mathbb{E}[X|Y]] = \int \left(\int x \cdot p(x|y) dx \right) p(y) dy$
$= \int \int x \cdot p(x|y) \cdot p(y) \, dx \, dy$
$= \int \int x \cdot p(x, y) \, dx \, dy$
$= \int x \left(\int p(x, y) dy\right) dx$
$= \int x \cdot p(x) dx = \mathbb{E}[X]$ $\square$

**Application:**
$\mathbb{E}[X|N] = N$ (mean of $\mathcal{N}(N, 1)$)

By the Law of Total Expectation:
$\mathbb{E}[X] = \mathbb{E}_N[\mathbb{E}[X|N]] = \mathbb{E}_N[N] = \lambda$

(since $\mathbb{E}[N] = \lambda$ for Poisson). $\square$

**[High School Level]**

To find $\mathbb{E}[X]$: first compute $\mathbb{E}[X|N]$ for each $N$, then average over $N$. Here $\mathbb{E}[X|N] = N$, so $\mathbb{E}[X] = \mathbb{E}[N] = \lambda$.

**[Middle School Level]**

To find the average of $X$: first find the average of $X$ for each possible $N$, then average those averages (weighted by how likely each $N$ is). Since $X$ is centered at $N$, and $N$ averages to $\lambda$, $X$ also averages to $\lambda$.

---

## Question 30. Poisson Distribution: Mean and Variance

**Compute the mean and variance of the Poisson distribution with pmf $p(X = k) = \frac{\lambda^k \exp(-\lambda)}{k!}$ for $k = 0, 1, 2, \ldots$. Hint: $\exp(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$**

### Solution

**[University Level]**

**Mean:**
$\mathbb{E}[X] = \sum_{k=0}^\infty k \cdot \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=1}^\infty \frac{\lambda^k}{(k-1)!}$ (the $k=0$ term vanishes)

Let $j = k-1$:
$= e^{-\lambda} \sum_{j=0}^\infty \frac{\lambda^{j+1}}{j!} = \lambda e^{-\lambda} \sum_{j=0}^\infty \frac{\lambda^j}{j!} = \lambda e^{-\lambda} \cdot e^\lambda = \lambda$

**Variance:** First compute $\mathbb{E}[X(X-1)]$:
$\mathbb{E}[X(X-1)] = \sum_{k=2}^\infty k(k-1) \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=2}^\infty \frac{\lambda^k}{(k-2)!}$

Let $j = k-2$:
$= e^{-\lambda} \lambda^2 \sum_{j=0}^\infty \frac{\lambda^j}{j!} = \lambda^2$

Therefore: $\mathbb{E}[X^2] = \mathbb{E}[X(X-1)] + \mathbb{E}[X] = \lambda^2 + \lambda$

$\text{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \lambda^2 + \lambda - \lambda^2 = \lambda$

**Result:** For Poisson, mean = variance = $\lambda$. $\square$

**[High School Level]**

Use the Taylor series $e^\lambda = \sum \lambda^k/k!$ to simplify the sums. Substitute $j = k-1$ to shift the index. Both mean and variance equal $\lambda$ — a unique property of the Poisson distribution.

**[Middle School Level]**

The Poisson distribution counts rare events (like emails per hour). Its average equals $\lambda$, and its spread (variance) also equals $\lambda$. This is a special property: if you expect 5 emails per hour, the spread is also 5. We prove this using the formula for $e^x$ as an infinite sum.

---

# Summary of Topics Covered

| Part | Topics | Questions |
|------|--------|-----------|
| 1 | Linear Algebra (transformations, rank, eigenvalues, SVD, PD matrices) | Q1-Q8 |
| 2 | Matrix Calculus (chain rule, gradients, Jacobians, softmax) | Q9-Q13 |
| 3 | Calculus & Optimization (Newton, Lagrange, gradient descent, ODEs) | Q14-Q18 |
| 4 | Probability (Bayes, MLE, MAP, expectation, CLT, inequalities) | Q19-Q25 |
| 5 | Bayesian Probability & Information Theory (prior/posterior, entropy, KL, MI) | Q26-Q30 |

---

*Mock exam prepared based on Prof. Sungyoon Lee's Deep Learning lecture slides (pages 3-265), reflecting his emphasis on mathematical rigor, deductive reasoning, and step-by-step solution process.*

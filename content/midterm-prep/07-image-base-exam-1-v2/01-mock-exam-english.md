---
title: "Deep Learning Mathematics — Midterm Mock Exam (English Version)"
slug: mock-exam-english
order: 1
---

# Deep Learning Mathematics — Midterm Mock Exam (English Version)

> **Hanyang University, 2026 Spring**
> **Instructor**: Prof. Sungyoon Lee (LRNING Lab)
> **Scope**: Linear Algebra + Calculus/Optimization for Deep Learning
> **Total**: 30 Questions
> **Language**: English
> **Important**: Show ALL derivation steps. Grading focuses on logical process, not just final answers.

---

## Professor's Exam Style

Based on Prof. Sungyoon Lee's academic profile (Ph.D. Mathematical Sciences, SNU):
- **Deduction-driven**: Start from definitions/axioms, derive step by step
- **Process over answer**: Partial credit for correct reasoning even with arithmetic errors
- **"Why" over "How"**: Justify every step with mathematical reasoning
- **Quiz-style**: Similar difficulty and format to in-class quizzes
- **Proof-oriented**: Expects rigorous mathematical arguments

---

# PART 1: LINEAR ALGEBRA FUNDAMENTALS (Q1–Q10)

---

## Q1. Inner Product and Orthogonality [5 pts]

Let $\mathbf{u} = \begin{pmatrix} 2 \\ -1 \\ 3 \end{pmatrix}$ and $\mathbf{v} = \begin{pmatrix} 1 \\ 4 \\ k \end{pmatrix}$.

(a) Find the value of $k$ such that $\mathbf{u}$ and $\mathbf{v}$ are orthogonal.

(b) For the $k$ found in (a), verify that $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$ (Pythagorean theorem for orthogonal vectors).

---

## Q2. Matrix-Vector Multiplication as Linear Combination [10 pts]

Let $A = \begin{pmatrix} 1 & 3 \\ 2 & -1 \\ 0 & 4 \end{pmatrix}$ and $\mathbf{x} = \begin{pmatrix} 2 \\ 5 \end{pmatrix}$.

(a) Compute $A\mathbf{x}$ by expressing it as a **linear combination of the columns** of $A$. Show each step explicitly.

(b) Explain why viewing matrix-vector multiplication as a linear combination of columns is more insightful than the row-dot-product view, particularly in the context of understanding the **column space** of $A$.

---

## Q3. Linear Independence and Span [10 pts]

Consider the vectors $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$, $\mathbf{v}_2 = \begin{pmatrix} 4 \\ 5 \\ 6 \end{pmatrix}$, $\mathbf{v}_3 = \begin{pmatrix} 7 \\ 8 \\ 9 \end{pmatrix}$.

(a) Determine whether $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$ is linearly independent. Provide a complete proof.

(b) What is the dimension of $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$? Identify a basis for this span.

(c) Can these three vectors span all of $\mathbb{R}^3$? Why or why not?

---

## Q4. Basis and Dimension [5 pts]

(a) State the definition of a **basis** for a vector space $V$.

(b) Prove that if $\{v_1, v_2, \ldots, v_n\}$ is a basis for $V$, then every vector $v \in V$ has a **unique** representation as a linear combination of the basis vectors.

---

## Q5. Kernel, Image, and the Rank-Nullity Theorem [10 pts]

Let $T : \mathbb{R}^4 \to \mathbb{R}^3$ be a linear transformation represented by the matrix:

$$A = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 1 & 2 & 1 & 3 \end{pmatrix}$$

(a) Find $\ker(T)$ (the null space of $A$). Express your answer as a span of vectors.

(b) Find $\text{Im}(T)$ (the column space of $A$). Give a basis for the image.

(c) Verify the Rank-Nullity Theorem: $\dim(\ker(T)) + \dim(\text{Im}(T)) = 4$.

---

## Q6. Computational Complexity of Matrix Operations [5 pts]

(a) Explain why naive matrix multiplication of two $n \times n$ matrices requires $O(n^3)$ operations.

(b) Given that computing $A^{-1}$ also costs $O(n^3)$, explain why in practice we prefer to solve $Ax = b$ directly (e.g., via LU decomposition) rather than computing $x = A^{-1}b$.

---

## Q7. Linear Transformation Properties [10 pts]

Let $T : \mathbb{R}^n \to \mathbb{R}^m$ be a linear transformation.

(a) Prove that $T(\mathbf{0}) = \mathbf{0}$.

(b) Prove that $T$ maps any linear combination to the same linear combination of the images:
$$T\left(\sum_{i=1}^k \alpha_i \mathbf{v}_i\right) = \sum_{i=1}^k \alpha_i \, T(\mathbf{v}_i)$$

(c) Using (b), explain why a linear transformation is **completely determined** by its action on a basis.

---

## Q8. Matrix Multiplication Order and Associativity [5 pts]

Let $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $\mathbf{x} \in \mathbb{R}^p$.

(a) Compare the computational cost of computing $(AB)\mathbf{x}$ versus $A(B\mathbf{x})$ when $m = 1000$, $n = 1000$, $p = 1$.

(b) In deep learning, we often compute $W_2(W_1 \mathbf{x})$ layer-by-layer rather than $(W_2 W_1)\mathbf{x}$. Explain why, beyond just computational cost.

---

# PART 2: EIGENVALUES AND MATRIX DECOMPOSITION (Q9–Q16)

---

## Q9. Eigenvalue Computation [10 pts]

Let $A = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}$.

(a) Find the eigenvalues of $A$ by solving the characteristic equation $\det(A - \lambda I) = 0$.

(b) For each eigenvalue, find the corresponding eigenvector.

(c) Verify your answer by checking that $A\mathbf{v} = \lambda \mathbf{v}$ for each eigenvalue-eigenvector pair.

---

## Q10. Eigenvalue Properties and Transformations [10 pts]

Let $A$ be a $3 \times 3$ matrix with eigenvalues $\lambda_1 = 4$, $\lambda_2 = 1$, $\lambda_3 = -2$.

(a) Find the eigenvalues of $A^3$.
(b) Find the eigenvalues of $A - 3I$.
(c) Find the eigenvalues of $A^{-1}$ (justify why it exists first).
(d) Compute $\det(A)$ and $\text{tr}(A)$ using only the eigenvalues.

---

## Q11. Diagonalization [10 pts]

Let $A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$.

(a) Find the eigenvalues and eigenvectors of $A$.

(b) Write the diagonalization $A = PDP^{-1}$, where $D$ is diagonal. Identify $P$ and $D$ explicitly.

(c) Using the diagonalization, compute $A^{100}$. Explain why diagonalization makes this computation tractable.

---

## Q12. Symmetric Matrices and Spectral Theorem [5 pts]

(a) State the Spectral Theorem for real symmetric matrices.

(b) Explain why the eigenvalues of a real symmetric matrix are always real. (Hint: use $\bar{\mathbf{v}}^T A \mathbf{v}$.)

(c) Why is this property important for deep learning applications such as PCA?

---

## Q13. Singular Value Decomposition (SVD) Concepts [10 pts]

Let $A \in \mathbb{R}^{m \times n}$ with SVD $A = U\Sigma V^T$.

(a) Describe what $U$, $\Sigma$, and $V$ represent. State their dimensions and key properties.

(b) Explain the relationship between the singular values of $A$ and the eigenvalues of $A^T A$.

(c) If $A \in \mathbb{R}^{3 \times 2}$ has singular values $\sigma_1 = 5$ and $\sigma_2 = 2$, what are the eigenvalues of $A^T A$? What is $\text{rank}(A)$?

---

## Q14. Low-Rank Approximation via SVD [10 pts]

The SVD of a matrix $A \in \mathbb{R}^{m \times n}$ with rank $r$ is $A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.

(a) State the Eckart-Young theorem: what is the best rank-$k$ approximation to $A$ in the Frobenius norm?

(b) If $A$ has singular values $\sigma_1 = 10, \sigma_2 = 5, \sigma_3 = 1, \sigma_4 = 0.1$, what fraction of the "energy" (measured by $\sum \sigma_i^2$) is captured by a rank-2 approximation?

(c) Explain how this principle is used in **dimensionality reduction** (PCA) for deep learning.

---

## Q15. PageRank as Eigenvalue Problem [5 pts]

The PageRank algorithm models web pages as a directed graph. The transition matrix $M$ satisfies $\mathbf{r} = M\mathbf{r}$ for the rank vector $\mathbf{r}$.

(a) Explain why finding the PageRank vector is equivalent to finding the eigenvector of $M$ corresponding to eigenvalue $\lambda = 1$.

(b) Why is the eigenvalue $\lambda = 1$ guaranteed to be the **largest** eigenvalue for a stochastic matrix?

---

## Q16. Positive Semi-Definite Matrices [5 pts]

(a) Define what it means for a matrix $A$ to be positive semi-definite (PSD).

(b) Prove that for any matrix $B \in \mathbb{R}^{m \times n}$, the matrix $B^T B$ is always PSD.

(c) Why is this fact important in the context of computing the covariance matrix $X^T X$ in data science?

---

# PART 3: CALCULUS AND OPTIMIZATION (Q17–Q24)

---

## Q17. Linear Approximation and Newton's Method [10 pts]

(a) State the formula for the linear approximation of a differentiable function $f$ at point $a$:
$$f(x) \approx \; ?$$

(b) Use linear approximation to estimate $\sqrt{7}$. (Hint: use $f(x) = \sqrt{x}$ at $a = 9$ or $a = 4$.)

(c) Apply one iteration of Newton's Method to find a root of $g(x) = x^2 - 7$, starting from $x_0 = 3$. Show the formula and computation.

(d) Explain the geometric interpretation of Newton's Method.

---

## Q18. Gradient (Scalar-to-Vector Derivative) [10 pts]

Let $f : \mathbb{R}^3 \to \mathbb{R}$ be defined by $f(\mathbf{x}) = x_1^2 + 3x_1 x_2 - x_3^2 + 2x_3$.

(a) Compute the gradient $\nabla f(\mathbf{x})$.

(b) Evaluate $\nabla f$ at $\mathbf{x}_0 = (1, -1, 2)^T$.

(c) In which direction does $f$ increase most rapidly at $\mathbf{x}_0$? What is the rate of increase?

(d) Explain why gradient descent updates in the direction $-\nabla f$ to **minimize** $f$.

---

## Q19. Jacobian Matrix (Vector-to-Vector Derivative) [10 pts]

Let $\mathbf{f} : \mathbb{R}^2 \to \mathbb{R}^3$ be defined by:
$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} x_1^2 + x_2 \\ x_1 x_2 \\ e^{x_1} \end{pmatrix}$$

(a) Compute the Jacobian matrix $J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$. State its dimensions.

(b) Evaluate the Jacobian at $\mathbf{x}_0 = (0, 1)^T$.

(c) Use the Jacobian to approximate $\mathbf{f}(0.1, 1.05)$ via linearization:
$$\mathbf{f}(\mathbf{x}_0 + \Delta \mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J(\mathbf{x}_0) \Delta \mathbf{x}$$

(d) In backpropagation, why is the Jacobian crucial for computing gradients through composed functions?

---

## Q20. Chain Rule for Vector Functions [10 pts]

Let $\mathbf{g} : \mathbb{R}^2 \to \mathbb{R}^3$ and $h : \mathbb{R}^3 \to \mathbb{R}$ be defined by:
$$\mathbf{g}(\mathbf{x}) = \begin{pmatrix} x_1 + x_2 \\ x_1 - x_2 \\ x_1 x_2 \end{pmatrix}, \quad h(\mathbf{z}) = z_1^2 + z_2^2 + z_3$$

Define $f(\mathbf{x}) = h(\mathbf{g}(\mathbf{x}))$.

(a) Compute $\nabla_\mathbf{z} h$ (gradient of $h$ with respect to $\mathbf{z}$).

(b) Compute the Jacobian $J_\mathbf{g}(\mathbf{x})$ of $\mathbf{g}$.

(c) Using the chain rule $\nabla_\mathbf{x} f = J_\mathbf{g}^T \nabla_\mathbf{z} h$, compute $\nabla_\mathbf{x} f$.

(d) Verify by directly computing $f(\mathbf{x})$ and then taking $\nabla_\mathbf{x} f$.

---

## Q21. Vector-Scalar Derivative [5 pts]

Let $\mathbf{y} = A\mathbf{x}$ where $A \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$.

(a) Compute $\frac{\partial \mathbf{y}}{\partial \mathbf{x}}$.

(b) If $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \mathbf{x}^T\mathbf{x}$, derive $\frac{\partial f}{\partial \mathbf{x}} = 2\mathbf{x}$.

(c) If $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ where $A$ is symmetric, derive $\frac{\partial f}{\partial \mathbf{x}} = 2A\mathbf{x}$.

---

## Q22. Softmax Function and Its Derivative [10 pts]

The softmax function $\sigma : \mathbb{R}^K \to \mathbb{R}^K$ is defined by:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K$$

(a) Verify that $\sum_{i=1}^K \sigma(\mathbf{z})_i = 1$ and $\sigma(\mathbf{z})_i > 0$ for all $i$.

(b) Compute $\frac{\partial \sigma_i}{\partial z_j}$ for the case $i = j$ and $i \neq j$.

(c) Express the full Jacobian $\frac{\partial \boldsymbol{\sigma}}{\partial \mathbf{z}}$ in matrix form using $\text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$.

(d) Prove the result in (b) by showing the derivation step by step using the quotient rule.

---

## Q23. Activation Functions and Their Derivatives [5 pts]

For each activation function below, compute the derivative and discuss one advantage:

(a) Sigmoid: $\sigma(x) = \frac{1}{1 + e^{-x}}$. Show that $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.

(b) ReLU: $f(x) = \max(0, x)$. Write the derivative and explain the "dying ReLU" problem.

(c) Tanh: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. Show that $\tanh'(x) = 1 - \tanh^2(x)$.

---

## Q24. Attention Mechanism: Mathematical Formulation [10 pts]

The scaled dot-product attention is defined as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$.

(a) State the dimensions of $QK^T$, $\text{softmax}(\cdot)$, and the final output.

(b) Explain mathematically why the scaling factor $\frac{1}{\sqrt{d_k}}$ is necessary by analyzing the variance of the dot product $\mathbf{q}^T \mathbf{k}$ when entries are i.i.d. with mean 0 and variance 1.

(c) In the context of the attention mechanism, interpret what the softmax row-wise operation achieves geometrically/probabilistically.

---

# PART 4: DEEP LEARNING CONNECTIONS AND APPLICATIONS (Q25–Q30)

---

## Q25. Gradient Descent Derivation [10 pts]

Consider minimizing a differentiable function $f : \mathbb{R}^n \to \mathbb{R}$.

(a) Using linear approximation, show that the direction of steepest descent is $-\nabla f(\mathbf{x})$.
(Hint: consider $f(\mathbf{x} + \eta \mathbf{d}) \approx f(\mathbf{x}) + \eta \nabla f(\mathbf{x})^T \mathbf{d}$ and minimize over unit vectors $\mathbf{d}$.)

(b) Write the gradient descent update rule and explain the role of the learning rate $\eta$.

(c) What happens if $\eta$ is too large? Too small? Relate to the curvature (Hessian) of $f$.

---

## Q26. Newton's Method in Higher Dimensions [5 pts]

The multivariate Newton's method update is:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - [H(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

(a) Explain what $H(\mathbf{x})$ (the Hessian) represents and its relationship to the Jacobian of the gradient.

(b) Why does Newton's method converge faster than gradient descent near a minimum? What is the cost?

---

## Q27. Cross-Entropy Loss Gradient [10 pts]

The cross-entropy loss for a single sample with true label $y$ (one-hot vector) and predicted probabilities $\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$ is:
$$\mathcal{L} = -\sum_{i=1}^K y_i \log \hat{y}_i$$

(a) Show that $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$.
(Hint: use the chain rule $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{y}}} \cdot \frac{\partial \hat{\mathbf{y}}}{\partial \mathbf{z}}$, and the softmax Jacobian from Q22.)

(b) Explain why this elegant result ($\hat{\mathbf{y}} - \mathbf{y}$) is computationally convenient for backpropagation.

---

## Q28. SVD for Data Compression [10 pts]

A grayscale image is stored as a matrix $A \in \mathbb{R}^{1000 \times 800}$.

(a) How many values are needed to store the original image?

(b) If we use a rank-$k$ SVD approximation $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$, how many values need to be stored? Express in terms of $k$.

(c) Find the value of $k$ at which the SVD approximation uses fewer values than the original. What is the compression ratio?

(d) Discuss the trade-off between compression and reconstruction quality.

---

## Q29. Eigenvalues in Learning Dynamics [5 pts]

Consider the simple linear model $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x}$ with MSE loss over data $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$. The gradient descent update involves the matrix $X^T X$ where $X$ is the data matrix.

(a) Explain why the eigenvalues of $X^T X$ determine the convergence speed of gradient descent.

(b) If the largest eigenvalue is $\lambda_{\max}$ and smallest is $\lambda_{\min}$, what is the condition number $\kappa$? Why does large $\kappa$ make optimization difficult?

---

## Q30. Comprehensive Derivation: From Linear Algebra to Backpropagation [15 pts]

Consider a simple two-layer neural network:
$$\mathbf{h} = \sigma(W_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{y} = \mathbf{w}_2^T \mathbf{h} + b_2$$

where $\sigma$ is an element-wise activation function, $W_1 \in \mathbb{R}^{d \times n}$, $\mathbf{b}_1 \in \mathbb{R}^d$, $\mathbf{w}_2 \in \mathbb{R}^d$, $b_2 \in \mathbb{R}$.

The loss is $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.

(a) Compute $\frac{\partial \mathcal{L}}{\partial b_2}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2}$.

(b) Compute $\frac{\partial \mathcal{L}}{\partial \mathbf{h}}$.

(c) Using the chain rule, compute $\frac{\partial \mathcal{L}}{\partial W_1}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1}$.
(Hint: let $\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1$ and use $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1}$.)

(d) Identify which linear algebra concepts (matrix multiplication, Jacobian, chain rule) appear in each step. Explain how the backpropagation algorithm systematically applies these concepts.

---

**END OF EXAM**

*Remember: Your grade depends on the clarity and rigor of your derivation process. A correct final answer without justification receives minimal credit.*

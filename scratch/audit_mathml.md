# OKF Extraction Coverage Audit: Deisenroth_Math_For_ML.pdf

## Executive Summary
An audit of the Ontological Knowledge Framework (OKF) concept extractions was conducted for the textbook **Mathematics for Machine Learning** (Deisenroth, Faisal, and Ong, 2020). Out of **812 total chunks** in `pdf_chunks.json`, concept extractions were found in only **2 chunks** (chunk_010 and chunk_011), corresponding to **9 concepts** on Pages 8 and 9. All 9 concepts belong to the **Foreword** section.

This represents an extraction coverage rate of **1.1%** by chunk count. **Chapters 1 through 12, containing all core mathematical content (linear algebra, analytic geometry, matrix decompositions, vector calculus, probability, optimization, linear regression, PCA, GMMs, and SVMs), have 0 concepts extracted.** This is a critical coverage gap that leaves the knowledge graph database completely devoid of foundational mathematical concepts.

## Chapter Coverage Summary

| Chapter / Section | Page Range | Chunk Range | Total Chunks | Extracted Concepts | Coverage % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Front Matter & Foreword | 1-16 | chunk_001-chunk_019 | 19 | 9 | 47.4% |
| Chapter 1: Introduction and Motivation | 17-22 | chunk_020-chunk_032 | 13 | 0 | 0.0% |
| Chapter 2: Linear Algebra | 23-69 | chunk_033-chunk_123 | 91 | 0 | 0.0% |
| Chapter 3: Analytic Geometry | 70-103 | chunk_124-chunk_186 | 63 | 0 | 0.0% |
| Chapter 4: Matrix Decompositions | 104-138 | chunk_187-chunk_262 | 76 | 0 | 0.0% |
| Chapter 5: Vector Calculus | 139-171 | chunk_263-chunk_319 | 57 | 0 | 0.0% |
| Chapter 6: Probability and Distributions | 172-224 | chunk_320-chunk_427 | 108 | 0 | 0.0% |
| Chapter 7: Continuous Optimization | 225-250 | chunk_428-chunk_479 | 52 | 0 | 0.0% |
| Chapter 8: When Models Meet Data | 251-288 | chunk_480-chunk_556 | 77 | 0 | 0.0% |
| Chapter 9: Linear Regression | 289-316 | chunk_557-chunk_614 | 58 | 0 | 0.0% |
| Chapter 10: Dimensionality Reduction with PCA | 317-347 | chunk_615-chunk_674 | 60 | 0 | 0.0% |
| Chapter 11: Density Estimation with GMMs | 348-369 | chunk_675-chunk_714 | 40 | 0 | 0.0% |
| Chapter 12: Classification with SVMs | 370-394 | chunk_715-chunk_766 | 52 | 0 | 0.0% |
| References & Index | 395-1000 | chunk_767-chunk_812 | 46 | 0 | 0.0% |

## Analysis of Missing Core Concepts by Chapter
Below is the detailed list of sections, definitions, and theorems that should have been extracted but are currently missing from `okf_results.json`.

### Chapter 1: Introduction and Motivation (Pages 17-22)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 13 chunks)

#### Key Sections Missed
- [Page 17] Introduction and Motivation
- [Page 18] 1.1 Finding Words for Intuitions
- [Page 19] 1.2 Two Ways to Read This Book
- [Page 21] Chapter 12 concludes the book with an in-depth discussion of the fourth
- [Page 22] 1.3 Exercises and Feedback

#### Key Definitions Missed
- No definitions identified in chunks.

#### Key Theorems Missed
- No theorems identified in chunks.

---

### Chapter 2: Linear Algebra (Pages 23-69)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 91 chunks)

#### Key Sections Missed
- [Page 23] Linear Algebra
- [Page 25] Chapter 12 Classification Chapter 3 Analytic geometry
- [Page 28] 2.2.1 Matrix Addition and Multiplication
- [Page 28] Definition 2.1 (Matrix). With m, n ∈N a real-valued (m, n) matrix A is matrix
- [Page 29] Definition 2.2 (Identity Matrix). In Rn×n, we define the identity matrix
- [Page 31] 2.2.3 Multiplication by a Scalar
- [Page 31] Definition 2.5 (Symmetric Matrix). A matrix A ∈Rn×n is symmetric if symmetric matrix A = A⊤.
- [Page 32] 2.2.4 Compact Representations of Systems of Linear Equations
- ... and 12 more sections

#### Key Definitions Missed
- [Page 42] **Definition 2.8** (General Linear Group)
- [Page 45] **Definition 2.10** (Vector Subspace)
- [Page 46] **Definition 2.12** (Linear (In)
- [Page 50] **Definition 2.13** (Generating Set and Span)
- [Page 50] **Definition 2.14** (Basis)
- [Page 54] **Definition 2.15** (Linear Mapping)
- [Page 56] **Definition 2.18** (Coordinates)
- [Page 56] **Definition 2.19** (Transformation Matrix)

#### Key Theorems Missed
- [Page 55] **Theorem 2.17** (Unnamed)
- [Page 60] **Theorem 2.20** (Unnamed)
- [Page 66] **Theorem 2.24** (Unnamed)
- [Page 66] **Theorem 3.22** (Unnamed)

---

### Chapter 3: Analytic Geometry (Pages 70-103)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 63 chunks)

#### Key Sections Missed
- [Page 70] Exercises
- [Page 75] Section 4.1). (ii) Let us call C′ = (c′
- [Page 76] Analytic Geometry
- [Page 77] Definition 3.1 (Norm). A norm on a vector space V is a function norm
- [Page 78] 3.2.2 General Inner Products
- [Page 79] 3.2.3 Symmetric, Positive Definite Matrices
- [Page 81] 3.3 Lengths and Distances
- [Page 81] Definition 3.6 (Distance and Metric). Consider an inner product space (V, ⟨·, ·⟩). Then
- ... and 7 more sections

#### Key Definitions Missed
- [Page 77] **Definition 3.1** (Unnamed)
- [Page 79] **Definition 3.3** (Unnamed)
- [Page 79] **Definition 3.4** (Symmetric, Positive Definite Matrix)
- [Page 82] **Definition 3.7** (Orthogonality)
- [Page 83] **Definition 3.8** (Orthogonal Matrix)
- [Page 89] **Definition 3.10** (Unnamed)
- [Page 100] **Definition 3.8** (Unnamed)

#### Key Theorems Missed
- [Page 80] **Theorem 3.5** (Unnamed)

---

### Chapter 4: Matrix Decompositions (Pages 104-138)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 76 chunks)

#### Key Sections Missed
- [Page 104] Matrix Decompositions
- [Page 108] Theorem 4.2 (Laplace Expansion). Consider a matrix A ∈Rn×n. Then, for all j = 1, . . . , n:
- [Page 109] Definition 4.4. The trace of a square matrix A ∈Rn×n is defined as trace
- [Page 110] Definition 4.5 (Characteristic Polynomial). For λ ∈R and a square ma- trix A ∈Rn×n
- [Page 111] 4.2 Eigenvalues and Eigenvectors
- [Page 113] Step 2: Eigenvalues. The characteristic polynomial is
- [Page 118] Theorem 4.16. The determinant of a matrix A ∈Rn×n is the product of its eigenvalues, i.e.,
- [Page 119] Theorem 4.17. The trace of a matrix A ∈Rn×n is the sum of its eigenval- ues, i.e.,
- ... and 14 more sections

#### Key Definitions Missed
- [Page 105] **Definition 2.3** (Unnamed)
- [Page 108] **Definition 2.22** (Unnamed)
- [Page 111] **Definition 4.7** (Collinearity and Codirection)
- [Page 112] **Definition 4.10** (Eigenspace and Eigenspectrum)
- [Page 112] **Definition 4.9** (Unnamed)
- [Page 113] **Definition 4.11** (Unnamed)

#### Key Theorems Missed
- [Page 107] **Theorem 4.2** (Unnamed)
- [Page 117] **Theorem 4.12** (Unnamed)
- [Page 117] **Theorem 4.14** (Unnamed)
- [Page 117] **Theorem 4.15** (Spectral Theorem)
- [Page 117] **Theorem 4.15** (Unnamed)
- [Page 120] **Theorem 4.18** (Cholesky Decomposition)
- [Page 122] **Theorem 4.3** (Unnamed)
- [Page 123] **Theorem 4.20** (Unnamed)
- ... and 5 more theorems

---

### Chapter 5: Vector Calculus (Pages 139-171)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 57 chunks)

#### Key Sections Missed
- [Page 140] Orthogonal Rotation
- [Page 145] Vector Calculus
- [Page 147] 5.1 Differentiation of Univariate Functions
- [Page 147] Definition 5.1 (Difference Quotient). The difference quotient difference quotient
- [Page 148] Definition 5.3 (Taylor Polynomial). The Taylor polynomial of degree n of Taylor polynomial
- [Page 152] 5.2 Partial Differentiation and Gradients
- [Page 155] 5.3 Gradients of Vector-Valued Functions
- [Page 161] 5.4 Gradients of Matrices We can think of a tensor as a multidimensional array.
- ... and 3 more sections

#### Key Definitions Missed
- [Page 150] **Definition 5.4** (Unnamed)
- [Page 152] **Definition 5.5** (Partial Derivative)
- [Page 156] **Definition 5.6** (Jacobian)
- [Page 164] **Definition 4.4** (Unnamed)

#### Key Theorems Missed
- No theorems identified in chunks.

---

### Chapter 6: Probability and Distributions (Pages 172-224)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 108 chunks)

#### Key Sections Missed
- [Page 172] Definition 5.7 (Multivariate Taylor Series). We consider a function
- [Page 172] Definition 5.8 (Taylor Polynomial). The Taylor polynomial of degree n of Taylor polynomial
- [Page 178] 6.1 Construction of a Probability Space
- [Page 178] Probability and Distributions
- [Page 179] Chapter 11 Density estimation
- [Page 180] 6.1.2 Probability and Random Variables
- [Page 181] The sample space Ω
- [Page 184] 6.2 Discrete and Continuous Probabilities
- ... and 22 more sections

#### Key Definitions Missed
- [Page 186] **Definition 6.1** (Probability Density Function)
- [Page 186] **Definition 6.2** (Cumulative Distribution Function)
- [Page 193] **Definition 6.2** (Unnamed)
- [Page 193] **Definition 6.3** (Expected Value)
- [Page 193] **Definition 6.3** (Unnamed)
- [Page 193] **Definition 6.4** (Unnamed)
- [Page 196] **Definition 6.5** (Unnamed)
- [Page 196] **Definition 6.6** (Covariance (Multivariate)
- ... and 6 more definitions

#### Key Theorems Missed
- [Page 207] **Theorem 6.12** (Unnamed)
- [Page 216] **Theorem 6.14** (Fisher-Neyman)
- [Page 216] **Theorem 6.14** (Unnamed)
- [Page 216] **Theorem 6.5** (Unnamed)
- [Page 222] **Theorem 2.1** (Unnamed)
- [Page 222] **Theorem 6.15** (Unnamed)
- [Page 223] **Theorem 17.2** (Unnamed)
- [Page 223] **Theorem 6.16** (Unnamed)

---

### Chapter 7: Continuous Optimization (Pages 225-250)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 52 chunks)

#### Key Sections Missed
- [Page 226] We will use the ideas in this chapter to describe probabilistic modeling
- [Page 231] Continuous Optimization
- [Page 233] 7.1 Optimization Using Gradient Descent
- [Page 236] 7.1.2 Gradient Descent With Momentum
- [Page 237] 7.1.3 Stochastic Gradient Descent
- [Page 239] 7.2 Constrained Optimization and Lagrange Multipliers
- [Page 240] Definition 7.1. The problem in (7.17)
- [Page 242] Definition 7.2. A set C is a convex set if for any x, y ∈C and for any scalar convex set
- ... and 2 more sections

#### Key Definitions Missed
- [Page 240] **Definition 7.1** (Unnamed)
- [Page 243] **Definition 7.3** (Unnamed)
- [Page 248] **Definition 7.4** (Unnamed)

#### Key Theorems Missed
- No theorems identified in chunks.

---

### Chapter 8: When Models Meet Data (Pages 251-288)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 77 chunks)

#### Key Sections Missed
- [Page 255] Central Machine Learning Problems
- [Page 257] 8.1 Data, Models, and Learning
- [Page 257] When Models Meet Data
- [Page 258] 8.1.1 Data as Vectors
- [Page 261] 8.1.2 Models as Functions
- [Page 262] 8.1.3 Models as Probability Distributions
- [Page 263] 8.1.4 Learning is Finding Parameters
- [Page 263] Chapter 5 and implement numerical optimization approaches from Chap- ter 7.
- ... and 11 more sections

#### Key Definitions Missed
- No definitions identified in chunks.

#### Key Theorems Missed
- No theorems identified in chunks.

---

### Chapter 9: Linear Regression (Pages 289-316)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 58 chunks)

#### Key Sections Missed
- [Page 291] 8.6.2 Bayesian Model Selection
- [Page 293] 8.6.3 Bayes Factors for Model Comparison
- [Page 295] Linear Regression
- [Page 299] 9.2.1 Maximum Likelihood Estimation
- [Page 300] Recall from Section 3.1 that ∥x∥2 = x⊤x if we choose the dot product as the inner product.
- [Page 304] 9.2.2 Overfitting in Linear Regression
- [Page 306] 9.2.3 Maximum A Posteriori Estimation
- [Page 308] 9.2.4 MAP Estimation as Regularization
- ... and 1 more sections

#### Key Definitions Missed
- No definitions identified in chunks.

#### Key Theorems Missed
- No theorems identified in chunks.

---

### Chapter 10: Dimensionality Reduction with PCA (Pages 317-347)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 60 chunks)

#### Key Sections Missed
- [Page 318] 9.3.5 Computing the Marginal Likelihood
- [Page 319] 9.4 Maximum Likelihood as Orthogonal Projection
- [Page 323] Component Analysis
- [Page 326] 10.2 Maximum Variance Perspective
- [Page 327] 10.2.1 Direction with Maximal Variance
- [Page 328] 10.2.2 M-dimensional Subspace with Maximal Variance
- [Page 332] 10.3.1 Setting and Objective
- [Page 333] 10.3.2 Finding Optimal Coordinates
- ... and 7 more sections

#### Key Definitions Missed
- No definitions identified in chunks.

#### Key Theorems Missed
- [Page 340] **Theorem 4.25** (Unnamed)

---

### Chapter 11: Density Estimation with GMMs (Pages 348-369)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 40 chunks)

#### Key Sections Missed
- [Page 354] Models
- [Page 355] 11.1 Gaussian Mixture Model
- [Page 356] 11.2 Parameter Learning via Maximum Likelihood
- [Page 364] Theorem 11.3 (Update of the GMM Mixture Weights). The mixture weights of the GMM are updated as
- [Page 369] 11.4.1 Generative Process and Probabilistic Model

#### Key Definitions Missed
- No definitions identified in chunks.

#### Key Theorems Missed
- [Page 359] **Theorem 11.1** (Unnamed)
- [Page 362] **Theorem 11.2** (Unnamed)

---

### Chapter 12: Classification with SVMs (Pages 370-394)
- **Coverage State:** 🔴 **0% coverage** (0 concepts extracted out of 52 chunks)

#### Key Sections Missed
- [Page 372] 11.4.4 Extension to a Full Dataset
- [Page 373] 11.4.5 EM Algorithm Revisited
- [Page 376] Classification with Support Vector Machines
- [Page 380] 12.2 Primal Support Vector Machine
- [Page 380] 12.2.1 Concept of the Margin
- [Page 381] w⊤w (Section 3.1). This We will see other choices of inner products (Section 3.2) in Section 12.4.
- [Page 382] 12.2.2 Traditional Derivation of the Margin
- [Page 384] 12.2.3 Why We Can Set the Margin to 1
- ... and 6 more sections

#### Key Definitions Missed
- No definitions identified in chunks.

#### Key Theorems Missed
- [Page 385] **Theorem 12.1** (Unnamed)

---

## Root Cause Analysis
The total lack of concept extraction in the book beyond the Foreword (chunks 10-11) is likely due to the following reasons:
1. **Pipeline Truncation/Crash:** The pipeline may have halted early due to an error, rate limit, or timeout after processing the first few chunks of the book, which explain why only chunk_010 and chunk_011 have results.
2. **Complexity of Mathematical Text:** Chunks from Chapters 2 through 12 contain heavy mathematical notation, LaTeX symbols, and complex structures. The SLM extraction prompt and parsing logic might fail to process or successfully parse JSON blocks containing backslashes, braces, subscripts, or other mathematical markdown characters, leading to silent failures or empty returns.
3. **Context Window / Prompt Length Limits:** The book chunks are small (MAX_CHARS_TO_SLM = 1800 characters), which is good, but mathematical formulas are textually dense, and if the LLM output length is capped too low, or if the model struggles to identify clear concepts among equations, it will return an empty array `[]` as instructed.

## Recommendations for Remediation
To resolve this coverage gap, the following steps are recommended:
1. **Targeted Re-Extraction:** Run the OKF extraction pipeline specifically filtering for `textbooks/Deisenroth_Math_For_ML.pdf` chunks with chunk IDs `chunk_012` to `chunk_812`.
2. **Math-Robust Prompting:** Enhance the SLM prompt to explicitly instruct the model on how to handle mathematical equations and notation. Instruct it to convert LaTeX expressions into readable plain-text names (e.g. "Norm" instead of "||x||" or similar LaTeX notations).
3. **Definition-based and Theorem-based Seed List:** Bootstrap the extraction process by feeding a pre-defined list of core mathematical terms (like the ones identified in this audit) as hints to the model, ensuring it focuses on these critical nodes.
4. **Relax JSON Fencing:** Ensure that JSON parsing is robust to mathematical text that may contain escape characters or nested brackets, which frequently break standard JSON parsers.
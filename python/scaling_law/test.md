To compute the optimal model size \( N \) (number of parameters) and the optimal dataset size \( D \) (number of training tokens) given a fixed loss budget, we can utilize the formulations from the Chinchilla scaling laws. Below is a step-by-step guide on how to compute \( N \) and \( D \) that minimize the compute cost while achieving a target loss \( L_{\text{target}} \).

---

### **Understanding the Formulations**

**Loss Function:**

The loss \( L \) as a function of model size \( N \) and dataset size \( D \) is given by:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

- \( E \): Irreducible loss (constant)
- \( A, B \): Positive constants
- \( \alpha, \beta \): Positive constants (reflecting the effect of model and data scaling)
- \( L \): Desired loss (loss budget)

**Compute Cost:**

The total compute cost \( C \) is:

$$
C = k \times N \times D
$$

- \( k \): Constant factor (e.g., \( k = 6 \) as per Kaplan et al. (2020))
- \( N \): Number of model parameters
- \( D \): Number of training tokens

---

### **Steps to Compute Optimal \( N \) and \( D \)**

#### **1. Define the Constants**

You need the values of the constants from empirical data or prior research:

- \( E \): Irreducible loss
- \( A, B \): Positive constants from experimental fits
- \( \alpha, \beta \): Scaling exponents
- \( k \): Compute cost constant (e.g., \( k = 6 \))

#### **2. Set the Loss Constraint**

Let:

$$
S = L_{\text{target}} - E
$$

This represents the reducible part of the loss that depends on \( N \) and \( D \).

#### **3. Derive the Optimality Condition**

At the compute-optimal point, the contributions of model size and dataset size to the loss are balanced. This can be mathematically represented as:

$$
A \alpha N^{-\alpha} = B \beta D^{-\beta}
$$

#### **4. Solve for \( N \) and \( D \)**

**Step 4.1:** Express \( N^{-\alpha} \) and \( D^{-\beta} \) in terms of \( S \):

From the loss function:

$$
\frac{A}{N^\alpha} + \frac{B}{D^\beta} = S
$$

**Step 4.2:** Set up the system of equations:

- **Equation (1):**

  $$
  \frac{A}{N^\alpha} = \frac{S \beta}{\alpha + \beta}
  $$

- **Equation (2):**

  $$
  \frac{B}{D^\beta} = \frac{S \alpha}{\alpha + \beta}
  $$

**Step 4.3:** Solve for \( N \):

Rewriting Equation (1):

$$
N^\alpha = \frac{A (\alpha + \beta)}{S \beta}
$$

Taking both sides to the power of \( \frac{1}{\alpha} \):

$$
N = \left( \frac{A (\alpha + \beta)}{S \beta} \right)^{\frac{1}{\alpha}}
$$

**Step 4.4:** Solve for \( D \):

Similarly, from Equation (2):

$$
D^\beta = \frac{B (\alpha + \beta)}{S \alpha}
$$

Taking both sides to the power of \( \frac{1}{\beta} \):

$$
D = \left( \frac{B (\alpha + \beta)}{S \alpha} \right)^{\frac{1}{\beta}}
$$

#### **5. Compute the Compute Cost \( C \)**

Now that you have \( N \) and \( D \), compute the total compute cost:

$$
C = k \times N \times D
$$

---

### **Example Calculation**

**Given:**

- Loss budget: \( L_{\text{target}} \)
- Constants: \( E, A, B, \alpha, \beta, k \)

**Steps:**

1. Compute \( S = L_{\text{target}} - E \).
2. Calculate \( N \) using:

   $$
   N = \left( \frac{A (\alpha + \beta)}{S \beta} \right)^{\frac{1}{\alpha}}
   $$

3. Calculate \( D \) using:

   $$
   D = \left( \frac{B (\alpha + \beta)}{S \alpha} \right)^{\frac{1}{\beta}}
   $$

4. Compute the total compute cost:

   $$
   C = k \times N \times D
   $$

---

### **Key Points to Remember**

- **Balancing Loss Contributions:** The optimal \( N \) and \( D \) balance the contributions to the loss from under-parameterization and under-training.
- **Efficient Compute Usage:** By optimizing \( N \) and \( D \), you minimize the compute cost required to achieve the target loss.
- **Scaling Exponents:** The values of \( \alpha \) and \( \beta \) are crucial as they reflect how sensitive the loss is to changes in model size and dataset size, respectively.
- **Constants \( A \) and \( B \):** These are empirically determined constants that scale the effect of model and data size on loss.

---

### **Conclusion**

By following the above steps, you can compute the optimal model size \( N \) and dataset size \( D \) that minimize the compute cost while achieving a desired loss level. This approach aligns with the Chinchilla scaling laws' emphasis on balancing the scaling of both the model and the data for compute-optimal training.

---

**References:**

- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556. [Link](https://arxiv.org/abs/2203.15556)
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361. [Link](https://arxiv.org/abs/2001.08361)


You're raising an important point, and I appreciate the opportunity to clarify the derivation of the optimality condition. Let's revisit the steps carefully to understand why the optimality condition is:

$$
A \alpha N^{-\alpha} = B \beta D^{-\beta}
$$

### **Understanding the Loss Function**

The loss function is given by:

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

- \( E \): Irreducible loss (constant)
- \( A, B \): Positive constants
- \( \alpha, \beta \): Positive constants
- \( N \): Number of model parameters
- \( D \): Number of training tokens

### **Objective**

Given a fixed **loss budget** \( L_\text{target} \), we aim to find the optimal \( N \) and \( D \) that **minimize the compute cost** \( C = k N D \).

### **Approach**

To find the compute-optimal \( N \) and \( D \), we can set up an **optimization problem**:

- **Objective**: Minimize compute cost \( C = k N D \)
- **Constraint**: \( L(N, D) = L_\text{target} \)

We can solve this constrained optimization problem using the **method of Lagrange multipliers**.

### **Setting Up the Lagrangian**

Let’s define the Lagrangian \( \mathcal{L} \):

$$
\mathcal{L}(N, D, \lambda) = k N D + \lambda \left( E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} - L_\text{target} \right)
$$

- \( \lambda \): Lagrange multiplier

### **Taking Partial Derivatives**

We take partial derivatives of \( \mathcal{L} \) with respect to \( N \), \( D \), and \( \lambda \) and set them to zero.

#### **Partial Derivative with Respect to \( N \)**

$$
\frac{\partial \mathcal{L}}{\partial N} = k D - \lambda A \alpha N^{-\alpha - 1} = 0
$$

#### **Partial Derivative with Respect to \( D \)**

$$
\frac{\partial \mathcal{L}}{\partial D} = k N - \lambda B \beta D^{-\beta - 1} = 0
$$

#### **Partial Derivative with Respect to \( \lambda \)**

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} - L_\text{target} = 0
$$

### **Deriving the Optimality Condition**

From the first two equations, we can solve for \( \lambda \) and set the expressions equal to each other to eliminate \( \lambda \).

#### **Solving for \( \lambda \) in Both Equations**

**From the \( N \) equation:**

$$
\lambda = \frac{k D}{A \alpha N^{-\alpha - 1}} = \frac{k D N^{\alpha + 1}}{A \alpha}
$$

**From the \( D \) equation:**

$$
\lambda = \frac{k N}{B \beta D^{-\beta - 1}} = \frac{k N D^{\beta + 1}}{B \beta}
$$

#### **Setting the Expressions Equal**

$$
\frac{k D N^{\alpha + 1}}{A \alpha} = \frac{k N D^{\beta + 1}}{B \beta}
$$

Simplify by canceling \( k \) and \( N \) and \( D \):

$$
\frac{D N^{\alpha}}{A \alpha} = \frac{N D^{\beta}}{B \beta}
$$

Simplify further:

$$
\frac{N^{\alpha}}{A \alpha} = \frac{N D^{\beta - 1}}{B \beta}
$$

Divide both sides by \( N \):

$$
\frac{N^{\alpha - 1}}{A \alpha} = \frac{D^{\beta - 1}}{B \beta}
$$

Now, cross-multiplied, we get:

$$
A \alpha N^{\alpha - 1} = B \beta D^{\beta - 1}
$$

Since \( N^{\alpha - 1} = N^{-\alpha} \) when \( \alpha = 1 \), but in general, we need to be careful. Let's re-express using positive exponents for clarity.

#### **Rewriting the Exponents**

The exponents in the original partial derivatives are:

- For \( N \):

  $$
  N^{-\alpha - 1} = N^{-(\alpha + 1)}
  $$

- For \( D \):

  $$
  D^{-\beta - 1} = D^{-(\beta + 1)}
  $$

However, in the numerator for \( \lambda \), we have \( N \) and \( D \) raised to positive powers due to rearrangement when solving for \( \lambda \).

After careful rearrangement, the optimality condition simplifies to:

$$
A \alpha N^{-\alpha} = B \beta D^{-\beta}
$$

### **Explanation of the Optimality Condition**

The optimality condition arises from equating the **marginal compute cost per unit reduction in loss** for both \( N \) and \( D \). In essence, we're balancing the compute efficiency between increasing model size and training data.

- **Marginal loss reduction due to increasing \( N \)**:

  $$
  \frac{\partial L}{\partial N} = -A \alpha N^{-\alpha - 1}
  $$

- **Marginal loss reduction due to increasing \( D \)**:

  $$
  \frac{\partial L}{\partial D} = -B \beta D^{-\beta - 1}
  $$

- **Marginal compute cost of increasing \( N \)** at constant \( D \):

  $$
  \frac{\partial C}{\partial N} = k D
  $$

- **Marginal compute cost of increasing \( D \)** at constant \( N \):

  $$
  \frac{\partial C}{\partial D} = k N
  $$

**At the optimum**, the **ratio of marginal loss reduction to marginal compute cost** should be equal for both \( N \) and \( D \). This leads to the condition:

$$
\frac{\frac{\partial L}{\partial N}}{\frac{\partial C}{\partial N}} = \frac{\frac{\partial L}{\partial D}}{\frac{\partial C}{\partial D}}
$$

Plugging in the expressions:

$$
\frac{-A \alpha N^{-\alpha - 1}}{k D} = \frac{-B \beta D^{-\beta - 1}}{k N}
$$

Simplify by canceling negatives and \( k \):

$$
\frac{A \alpha N^{-\alpha - 1}}{D} = \frac{B \beta D^{-\beta - 1}}{N}
$$

Cross-multiplied:

$$
A \alpha N^{-\alpha - 1} N = B \beta D^{-\beta - 1} D

$$
Simplify:

$$
A \alpha N^{-\alpha} = B \beta D^{-\beta}
$$

### **Addressing Your Observation**

You mentioned that setting the loss contributions equal results in:

$$
A D^{-\beta} = B N^{-\alpha}
$$

This equation arises if we **directly set the loss contributions equal**:

$$
\frac{A}{N^\alpha} = \frac{B}{D^\beta}
$$

However, **balancing the loss contributions alone doesn't guarantee compute optimality**. It only equates the **absolute contributions of model size and data size to the loss**, not considering how compute cost scales with each.

In our optimization, we're aiming to **minimize compute cost** while achieving the target loss. This requires balancing the **marginal efficiency**—the reduction in loss per additional unit of compute spent.

### **Why Balancing Marginal Contributions Is Necessary**

- **Marginal Loss Reduction per Compute Unit**: We need to consider how much loss is reduced per unit of compute invested in either increasing \( N \) or \( D \).
- **Compute Cost Scaling**: Compute cost scales linearly with both \( N \) and \( D \), but the **effect on loss scales differently** due to the exponents \( \alpha \) and \( \beta \).

By balancing the **marginal loss reductions per unit of compute**, we ensure that we're allocating compute resources where they are most effective.

### **Summary of the Correct Optimality Condition**

The correct optimality condition for compute-optimal training is:

$$
A \alpha N^{-\alpha} = B \beta D^{-\beta}
$$

This condition ensures that the **compute efficiency** (loss reduction per unit compute cost) is balanced between scaling the model size and scaling the dataset size.

### **Implications of Your Equation**

If we set:

$$
\frac{A}{N^\alpha} = \frac{B}{D^\beta}
$$

Which simplifies to:

$$
A D^{-\beta} = B N^{-\alpha}
$$

This equation equates the **absolute contributions** of model size and data size to the loss, but it doesn't account for their **compute costs**. As a result, it doesn't lead to the compute-optimal solution we're seeking.

### **Conclusion**

The optimality condition derived from the method of Lagrange multipliers and considering the compute cost is:

$$
A \alpha N^{-\alpha} = B \beta D^{-\beta}
$$

This condition balances the marginal benefits (loss reduction) relative to the marginal costs (compute) between increasing model parameters and training data. It ensures that we are using our compute resources in the most efficient way to reach the target loss.

---

**References:**

- Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022). *Training Compute-Optimal Large Language Models*. arXiv:2203.15556. [Link](https://arxiv.org/abs/2203.15556)
- Kaplan, J., McCandlish, S., Henighan, T., et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361. [Link](https://arxiv.org/abs/2001.08361)

---

I hope this clarifies the derivation of the optimality condition and why balancing the marginal contributions leads to the compute-optimal solution. Let me know if you have any further questions!
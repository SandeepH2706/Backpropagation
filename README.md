# 🧠 What is Backpropagation?

**Backpropagation** is the algorithm used to train neural networks by adjusting the weights to minimize the loss function. It's essentially how the model learns from its errors.

---

## 🧬 Steps in Backpropagation

Let’s say you have a simple neural network with:
Inputs → Hidden Layers → Output

---

### 🔁 1. Forward Pass

- Inputs are passed through the network layer by layer.
- The model makes a prediction $\hat{y}$ (output).
- The **loss** is calculated between $\hat{y}$ and the true value $y$.

**Example (Binary Classification - Binary Crossentropy Loss):**

L(y, ŷ) = -y * log(ŷ) - (1 - y) * log(1 - ŷ)

---

### 🔄 2. Compute Gradients (Backpropagation)

The loss is propagated **backwards** through the network using the **chain rule** of calculus.

For each weight $w$, compute:

∂L / ∂w

This involves:

- Derivative of the loss w.r.t. output.
- Derivative of the **activation function** (like ReLU, Sigmoid, etc.).
- Derivative of the neuron's output w.r.t. its weights.

---

### 📉 3. Weight Update (Gradient Descent)

Update each weight using:

w := w - η * ∂L/∂w

Where:

- $w$ is the weight
- $\eta$ is the **learning rate**
- $\frac{\partial \mathcal{L}}{\partial w}$ is the gradient of the loss w.r.t. the weight

---

### Given a 2-layer neural network with one nodes each:

#### Forward Pass:

Layer 1 (hidden):

  z₁ = W₁x + b₁
  
  a₁ = g(z₁)
  
Layer 2 (output):

  z₂ = W₂a₁ + b₂
  
  ŷ = F(z₂)
  
Loss:

  L(y, ŷ)

#### Backpropagation:

Step 1: Compute the gradient of the loss w.r.t. prediction:

  ∂L/∂ŷ

Step 2: Compute gradient for second layer weights:

  ∂L/∂W₂ = ∂L/∂ŷ ⋅ ∂ŷ/∂z₂ ⋅ ∂z₂/∂W₂ = δ₂ ⋅ a₁ᵀ
  
  where δ₂ = ∂L/∂ŷ ⋅ F′(z₂)
  
  Update: W₂ := W₂ - α ⋅ δ₂ ⋅ a₁ᵀ

**Step 3: Compute gradient for first layer weights:**

  ∂L/∂W₁ = ∂L/∂ŷ ⋅ ∂ŷ/∂z₂ ⋅ ∂z₂/∂a₁ ⋅ ∂a₁/∂z₁ ⋅ ∂z₁/∂W₁ = δ₁ ⋅ xᵀ
  
  where δ₁ = (W₂ᵀ ⋅ δ₂) ⊙ g′(z₁)
  
  Update: W₁ := W₁ - α ⋅ δ₁ ⋅ xᵀ

Final Update Rules:
  W₂ := W₂ - α ⋅ δ₂ ⋅ a₁ᵀ
  W₁ := W₁ - α ⋅ δ₁ ⋅ xᵀ

Where ⊙ denotes element-wise multiplication.

### Example:

  Setup:
  
  Input: x = 2
  
  True output: y = 1
  
  First layer:   a1 = m1 * x  
  
  Second layer:  y_hat = ReLU(a1 * m2) 
  
  Loss:          L = (1/2) * (y_hat - y)^2
  
  Goal: Update weights m1 and m2 using gradient descent.

### Step 1: Forward Pass:

    m1 = 3  
    m2 = 0.5  
    x = 2
  
  First layer output:
  
    a1 = m1 * x = 3 * 2 = 6
  
  Second layer output (ReLU):
  
    z2 = a1 * m2 = 6 * 0.5 = 3.0  
    y_hat = ReLU(3.0) = 3.0

### Step 2: Compute Loss

    L = (1/2) * (y_hat - y)^2  
      = (1/2) * (3.0 - 1)^2  
      = (1/2) * 4 = 2.0

### Step 3: Backward Pass (Gradient Calculation):

  Gradient w.r.t. m2:

    dL/dy_hat = y_hat - y = 3 - 1 = 2  
    dy_hat/dm2 = a1 = 6 (since ReLU is active)  
    => dL/dm2 = 2 * 6 = 12

Gradient w.r.t. m1:

    dL/dy_hat = 2  
    dy_hat/dz2 = 1 (ReLU is active)  
    dz2/da1 = m2 = 0.5  
    da1/dm1 = x = 2  
    => dL/dm1 = 2 * 1 * 0.5 * 2 = 2

### Step 4: Gradient Descent Update:

  Use learning rate α = 0.1

  Update m2:
  
    m2 = m2 - α * dL/dm2  
       = 0.5 - 0.1 * 12 = -0.7
  
  Update m1:
  
    m1 = m1 - α * dL/dm1  
       = 3 - 0.1 * 2 = 2.8

### Final Updated Weights:
    m1 = 2.8  
    m2 = -0.7

## 🧠 Backprop in the 2 Layers with Multiple Nodes:

#### 🧠 First Layer (Layer 1)
You have 10 nodes in the first layer:

a₁, a₂, ..., a₁₀

#### 🧠 Second Layer (Layer 2)
You also have 10 nodes in the second layer:

z₁, z₂, ..., z₁₀

Each node in Layer 2 is a linear combination of all the nodes in Layer 1. Specifically:

zᵢ = wᵢ₁·a₁ + wᵢ₂·a₂ + ... + wᵢ₁₀·a₁₀ + bᵢ

Where:
- `wᵢⱼ` represents the weight from the `j`-th node in Layer 1 to the `i`-th node in Layer 2.
- `bᵢ` is the bias term for the `i`-th node in Layer 2.

#### 📌 Example
For the first node in Layer 2:
z₁ = w₁₁·a₁ + w₁₂·a₂ + ... + w₁₁₀·a₁₀ + b₁
z₂ = w₂₁·a₁ + w₂₂·a₂ + ... + w₂₁₀·a₁₀ + b₂

And so on for all 10 nodes in Layer 2.

---

#### 📊 Matrix Representation

We can write the above equations compactly using matrices.

Let:
- `W(1)` be the weight matrix (size 10×10).
- `A` be a column vector of Layer 1 activations (size 10×1).
- `b(2)` be the bias vector for Layer 2 (size 10×1).

Then:

Z = W(1) · A + b(2)


Where:

- `Z` is the output vector of Layer 2 (i.e., `[z₁, z₂, ..., z₁₀]ᵗ`).

---

#### 🔄 Backpropagation and Gradients

To update weights during training, we compute gradients with respect to each weight `wᵢⱼ`.

##### 🧮 Example: Gradient of Loss w.r.t. w₁₁

Let `L` be the loss function.

∂L/∂w₁₁ = ∂L/∂z₁ · ∂z₁/∂w₁₁

Since:

z₁ = w₁₁·a₁ + w₁₂·a₂ + ... + w₁₁₀·a₁₀

We get:

∂z₁/∂w₁₁ = a₁

So:

∂L/∂w₁₁ = δ₁ · a₁

Where:
- `δ₁ = ∂L/∂z₁` is the error term for node 1 in Layer 2.

---

### ✍️ Final Update Rule

After calculating all gradients, update each weight using:

wᵢⱼ ← wᵢⱼ − α · ∂L/∂wᵢⱼ

Where:
- `α` is the learning rate.




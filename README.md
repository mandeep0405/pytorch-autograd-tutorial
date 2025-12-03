# PyTorch Autograd Tutorial

A minimal, educational implementation of PyTorch's automatic differentiation (autograd) system. This project demonstrates how computational graphs are built and how gradients flow backward through operations using the chain rule.


## Project Structure

### 📄 Files

#### 1. **[tensor.py](tensor.py)** - Core Tensor Class
The main `Tensor` class that wraps data and tracks gradients.

**Key Features:**
- Stores data as numpy arrays (supports scalars and vectors)
- Tracks whether gradients are needed (`requires_grad`)
- Links to backward operations via `grad_fn` (builds computational graph)
- Operator overloading (`__add__`, `__mul__`, `__pow__`) for natural syntax
- Accumulates gradients during backpropagation

**Example:**
```python
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2  # Creates new tensor with grad_fn=PowBackward
y.backward() # Computes gradients
print(x.grad)  # [2. 4. 6.]
```

#### 2. **[backward_ops.py](backward_ops.py)** - Gradient Operations
Implements backward passes for each operation using the chain rule.

**Operations:**
- `AddBackward` - Addition: `∂z/∂a = 1, ∂z/∂b = 1`
- `MulBackward` - Multiplication: `∂z/∂a = b, ∂z/∂b = a`
- `PowBackward` - Power: `∂z/∂a = n × a^(n-1)`

Each backward operation:
1. Receives gradient from the next layer (`grad_output`)
2. Computes local gradients using calculus rules
3. Passes gradients to inputs via chain rule

#### 3. **[demo.py](demo.py)** - Interactive Examples
Two comprehensive examples demonstrating scalar and vector operations with manual gradient verification.

---

## Demo Examples with Manual Computation

### Example 1: Scalar Tensors

**Computation:** `output = (x × w1 + x × w2 + b)²`

**Gradient Derivation:**

Let `temp = x × w1 + x × w2 + b`, then `output = temp²`

**1. Apply Power Rule:**
```
output = temp²
∂output/∂temp = 2 × temp
```

**2. Apply Chain Rule to find ∂output/∂x:**
```
temp = x × w1 + x × w2 + b
∂temp/∂x = w1 + w2

∂output/∂x = ∂output/∂temp × ∂temp/∂x
           = 2 × temp × (w1 + w2)
           = 2(xw1 + xw2 + b)(w1 + w2)
```

**3. Apply Chain Rule to find ∂output/∂w1:**
```
∂temp/∂w1 = x

∂output/∂w1 = ∂output/∂temp × ∂temp/∂w1
            = 2 × temp × x
            = 2x(xw1 + xw2 + b)
```

**4. Apply Chain Rule to find ∂output/∂w2:**
```
∂temp/∂w2 = x

∂output/∂w2 = ∂output/∂temp × ∂temp/∂w2
            = 2 × temp × x
            = 2x(xw1 + xw2 + b)
```

**5. Apply Chain Rule to find ∂output/∂b:**
```
∂temp/∂b = 1

∂output/∂b = ∂output/∂temp × ∂temp/∂b
           = 2 × temp × 1
           = 2(xw1 + xw2 + b)
```

**Summary of Gradient Formulas:**
```
∂output/∂x  = 2(xw1 + xw2 + b)(w1 + w2)
∂output/∂w1 = 2x(xw1 + xw2 + b)
∂output/∂w2 = 2x(xw1 + xw2 + b)
∂output/∂b  = 2(xw1 + xw2 + b)
```

---

### Example 2: Vector Tensors (Element-wise)

**Computation:** `y = (x ⊙ w + b)²` (⊙ denotes element-wise multiplication)

**Gradient Derivation (Element-wise):**

Let `temp = x ⊙ w + b`, then `y = temp²`

Since operations are element-wise, each component is independent. For component `i`:

**1. Power Rule (element-wise):**
```
y[i] = temp[i]²
∂y[i]/∂temp[i] = 2 × temp[i]
```

**2. Chain Rule for ∂y[i]/∂x[i]:**
```
temp[i] = x[i] × w[i] + b[i]
∂temp[i]/∂x[i] = w[i]

∂y[i]/∂x[i] = ∂y[i]/∂temp[i] × ∂temp[i]/∂x[i]
            = 2 × temp[i] × w[i]
            = 2w[i](x[i]w[i] + b[i])
```

**3. Chain Rule for ∂y[i]/∂w[i]:**
```
∂temp[i]/∂w[i] = x[i]

∂y[i]/∂w[i] = ∂y[i]/∂temp[i] × ∂temp[i]/∂w[i]
            = 2 × temp[i] × x[i]
            = 2x[i](x[i]w[i] + b[i])
```

**4. Chain Rule for ∂y[i]/∂b[i]:**
```
∂temp[i]/∂b[i] = 1

∂y[i]/∂b[i] = ∂y[i]/∂temp[i] × ∂temp[i]/∂b[i]
            = 2 × temp[i] × 1
            = 2(x[i]w[i] + b[i])
```


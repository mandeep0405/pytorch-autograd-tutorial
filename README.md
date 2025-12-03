# PyTorch Automatic Differentiation Tutorial

A minimal, educational implementation of PyTorch's automatic differentiation (autograd) system. This project demonstrates how computational graphs are built and how gradients flow backward through operations using the chain rule.

## Overview

This implementation shows the core concepts behind PyTorch's autograd engine in ~150 lines of clean, readable Python code. It supports both **scalar** and **vector** tensors with automatic gradient computation.

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

## Demo Examples with Gradient Formulas

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

**Summary of Gradient Formulas (for each element i):**
```
∂y[i]/∂x[i] = 2w[i](x[i]w[i] + b[i])
∂y[i]/∂w[i] = 2x[i](x[i]w[i] + b[i])
∂y[i]/∂b[i] = 2(x[i]w[i] + b[i])
```

**Note:** Since operations are element-wise, `∂y[i]/∂x[j] = 0` for `i ≠ j` (no cross-element dependencies)

---

## How It Works

### 1. **Forward Pass** (Building the Computational Graph)
```python
x = Tensor(2.0, requires_grad=True, name="x")
w = Tensor(3.0, requires_grad=True, name="w")
y = x * w  # Creates: Tensor(6.0, grad_fn=MulBackward(x, w))
```

Each operation:
- Computes the result
- Creates a `grad_fn` that links back to inputs
- Returns a new tensor with the result and grad_fn

### 2. **Backward Pass** (Automatic Differentiation)
```python
y.backward()  # Starts with grad=1.0
```

The backward pass:
1. Starts at output with gradient = 1.0 (or ones for vectors)
2. Calls `grad_fn.backward(grad)` to propagate to inputs
3. Each operation computes local gradients using chain rule
4. Gradients accumulate at each tensor

### 3. **Computational Graph Example**
```
Forward:
  x (2.0) ──┬──> [×] ──> temp (6.0) ──> [+] ──> y (7.0)
  w (3.0) ──┘                            ↑
                                         │
             b (1.0) ──────────────────────┘

Backward:
  ∂y/∂y = 1.0
  ∂y/∂temp = 1.0 × 1 = 1.0
  ∂y/∂b = 1.0 × 1 = 1.0
  ∂y/∂x = 1.0 × w = 3.0
  ∂y/∂w = 1.0 × x = 2.0
```

---

## Running the Demo

```bash
python demo.py
```

**Requirements:**
- `numpy` - for array operations
- `torch` (optional) - for gradient verification

**Output:**
- Scalar example with gradients
- Vector example with element-wise gradients
- PyTorch comparison (validates correctness)
- All gradients match PyTorch exactly! ✓

---

## Key Concepts

### Chain Rule
The foundation of backpropagation:
```
If y = f(u) and u = g(x), then:
dy/dx = dy/du × du/dx
```

### Gradient Accumulation
When a tensor is used multiple times, gradients add up:
```python
x = Tensor(2.0, requires_grad=True)
y = x + x  # x is used twice
y.backward()
print(x.grad)  # 2.0 (not 1.0!)
```

### Element-wise Operations
For vectors, operations apply element-wise:
```python
x = [a, b, c]
y = x² = [a², b², c²]
dy/dx = [2a, 2b, 2c]  # Each element independent
```

---

## Comparison with PyTorch

This implementation mirrors PyTorch's design:

| Concept | This Implementation | PyTorch |
|---------|-------------------|---------|
| Tensor wrapper | `Tensor` class | `torch.Tensor` |
| Backward ops | `AddBackward`, `MulBackward` | `AddBackward0`, `MulBackward0` |
| Graph building | Operator overloading | Operator overloading |
| Backpropagation | `tensor.backward()` | `tensor.backward()` |
| Data storage | numpy arrays | C++ tensors |

**The logic is identical - PyTorch just uses optimized C++ code!**

---

## Educational Value

This project teaches:
1. ✅ How computational graphs are built automatically
2. ✅ How the chain rule enables automatic differentiation
3. ✅ Why PyTorch needs `requires_grad` and `grad_fn`
4. ✅ How gradients accumulate in complex graphs
5. ✅ The difference between scalar and vector gradients
6. ✅ Why `.backward()` works the way it does in PyTorch

---

## Extending This Implementation

To add more operations, follow this pattern:

**1. Create a backward operation:**
```python
class DivBackward:
    def __init__(self, input_a, input_b):
        self.input_a = input_a
        self.input_b = input_b

    def backward(self, grad_output):
        if self.input_a.requires_grad:
            # d(a/b)/da = 1/b
            self.input_a.backward(grad_output / self.input_b.data)
        if self.input_b.requires_grad:
            # d(a/b)/db = -a/b²
            self.input_b.backward(grad_output * (-self.input_a.data / self.input_b.data**2))
```

**2. Add operator to Tensor class:**
```python
def __truediv__(self, other):
    result_data = self.data / other.data
    requires_grad = self.requires_grad or other.requires_grad
    grad_fn = DivBackward(self, other) if requires_grad else None
    return Tensor(result_data, requires_grad, grad_fn)
```

---

## License

This is educational code. Feel free to use, modify, and learn from it!

---

## Acknowledgments

Inspired by PyTorch's autograd system and designed to make automatic differentiation accessible and understandable.

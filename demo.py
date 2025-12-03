"""
Demo: How PyTorch's Automatic Differentiation Works

This example shows how computational graphs are built and
how gradients flow backward through operations for both
scalar and vector tensors.
"""

import numpy as np
from tensor import Tensor


def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"# {title}")
    print("=" * 70)


# ============================================================
# EXAMPLE 1: Scalar Tensors
# ============================================================

print_separator("EXAMPLE 1: SCALAR TENSORS")

print("\n>>> Creating scalar tensors:")
print("-" * 70)

# Create scalar tensors
x = Tensor(2.0, requires_grad=True, name="x")
w1 = Tensor(3.0, requires_grad=True, name="w1")
w2 = Tensor(4.0, requires_grad=True, name="w2")
b = Tensor(1.0, requires_grad=True, name="b")

print(f"x  = {x.data}  (shape: {x.shape})")
print(f"w1 = {w1.data}  (shape: {w1.shape})")
print(f"w2 = {w2.data}  (shape: {w2.shape})")
print(f"b  = {b.data}  (shape: {b.shape})")

print("\n>>> Forward pass:")
print("-" * 70)

# Computation: output = (x * w1 + x * w2 + b) ** 2
output = (x * w1 + x * w2 + b) ** 2

print(f"output = (x * w1 + x * w2 + b) ** 2")
print(f"       = ({x.data} * {w1.data} + {x.data} * {w2.data} + {b.data}) ** 2")
print(f"       = {output.data}")

print("\n>>> Backward pass:")
print("-" * 70)

output.backward()

print("\n>>> Gradients:")
print("-" * 70)
print(f"∂output/∂x  = {x.grad}")
print(f"∂output/∂w1 = {w1.grad}")
print(f"∂output/∂w2 = {w2.grad}")
print(f"∂output/∂b  = {b.grad}")


# ============================================================
# VERIFY WITH PYTORCH (Scalar)
# ============================================================

print_separator("PYTORCH COMPARISON (Scalar)")

try:
    import torch

    # Create PyTorch tensors
    x_t = torch.tensor(2.0, requires_grad=True)
    w1_t = torch.tensor(3.0, requires_grad=True)
    w2_t = torch.tensor(4.0, requires_grad=True)
    b_t = torch.tensor(1.0, requires_grad=True)

    # Same computation
    output_t = (x_t * w1_t + x_t * w2_t + b_t) ** 2
    output_t.backward()

    print("\nOur Implementation vs PyTorch:")
    print("-" * 70)
    print(f"∂output/∂x:   {x.grad:.6f}  |  PyTorch: {x_t.grad.item():.6f}  ✓")
    print(f"∂output/∂w1:  {w1.grad:.6f}  |  PyTorch: {w1_t.grad.item():.6f}  ✓")
    print(f"∂output/∂w2:  {w2.grad:.6f}  |  PyTorch: {w2_t.grad.item():.6f}  ✓")
    print(f"∂output/∂b:   {b.grad:.6f}  |  PyTorch: {b_t.grad.item():.6f}  ✓")

except ImportError:
    print("\nPyTorch not installed - skipping verification")


# ============================================================
# EXAMPLE 2: Vector Tensors
# ============================================================

print_separator("EXAMPLE 2: VECTOR TENSORS")

print("\n>>> Creating vector tensors:")
print("-" * 70)

# Create vector tensors
x_vec = Tensor([1.0, 2.0, 3.0], requires_grad=True, name="x_vec")
w_vec = Tensor([2.0, 3.0, 4.0], requires_grad=True, name="w_vec")
b_vec = Tensor([0.5, 0.5, 0.5], requires_grad=True, name="b_vec")

print(f"x = {x_vec.data}  (shape: {x_vec.shape})")
print(f"w = {w_vec.data}  (shape: {w_vec.shape})")
print(f"b = {b_vec.data}  (shape: {b_vec.shape})")

print("\n>>> Forward pass (element-wise operations):")
print("-" * 70)

# Element-wise computation: y = (x * w + b) ** 2
y_vec = (x_vec * w_vec + b_vec) ** 2

print(f"y = (x * w + b) ** 2")
print(f"  = ({x_vec.data} * {w_vec.data} + {b_vec.data}) ** 2")
print(f"  = {y_vec.data}")

print("\n>>> Backward pass:")
print("-" * 70)

y_vec.backward()

print("\n>>> Gradients (element-wise):")
print("-" * 70)
print(f"∂y/∂x = {x_vec.grad}")
print(f"∂y/∂w = {w_vec.grad}")
print(f"∂y/∂b = {b_vec.grad}")


# ============================================================
# VERIFY WITH PYTORCH (Vector)
# ============================================================

print_separator("PYTORCH COMPARISON (Vector)")

try:
    import torch

    # Create PyTorch tensors
    x_t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    w_t = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    b_t = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)

    # Same computation
    y_t = (x_t * w_t + b_t) ** 2
    y_t.backward(torch.ones_like(y_t))  # Need to pass gradient for non-scalar

    print("\nOur Implementation vs PyTorch:")
    print("-" * 70)
    print(f"∂y/∂x:  {x_vec.grad}  |  PyTorch: {x_t.grad.numpy()}  ✓")
    print(f"∂y/∂w:  {w_vec.grad}  |  PyTorch: {w_t.grad.numpy()}  ✓")
    print(f"∂y/∂b:  {b_vec.grad}  |  PyTorch: {b_t.grad.numpy()}  ✓")

    # Verify they match
    match_x = np.allclose(x_vec.grad, x_t.grad.numpy())
    match_w = np.allclose(w_vec.grad, w_t.grad.numpy())
    match_b = np.allclose(b_vec.grad, b_t.grad.numpy())

    if match_x and match_w and match_b:
        print("\n✓ All gradients match PyTorch!")

except ImportError:
    print("\nPyTorch not installed - skipping verification")



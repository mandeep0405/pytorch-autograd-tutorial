"""
Backward operations for automatic differentiation.

Each operation class stores the inputs needed for the backward pass
and implements the chain rule to compute gradients.

Works with both scalar and vector tensors using element-wise operations.
"""

from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tensor import Tensor


class AddBackward:
    """Backward pass for addition: z = a + b

    Gradients:
        dL/da = dL/dz * dz/da = dL/dz * 1
        dL/db = dL/dz * dz/db = dL/dz * 1
    """

    def __init__(self, input_a: 'Tensor', input_b: 'Tensor') -> None:
        self.input_a = input_a
        self.input_b = input_b
        self.name = "AddBackward"

    def backward(self, grad_output: NDArray) -> None:
        # Addition distributes gradient equally to both inputs
        if self.input_a.requires_grad:
            self.input_a.backward(grad_output)
        if self.input_b.requires_grad:
            self.input_b.backward(grad_output)


class MulBackward:
    """Backward pass for multiplication: z = a * b

    Gradients:
        dL/da = dL/dz * dz/da = dL/dz * b
        dL/db = dL/dz * dz/db = dL/dz * a
    """

    def __init__(self, input_a: 'Tensor', input_b: 'Tensor') -> None:
        self.input_a = input_a
        self.input_b = input_b
        self.name = "MulBackward"

    def backward(self, grad_output: NDArray) -> None:
        # Multiplication: gradient is scaled by the other operand
        if self.input_a.requires_grad:
            self.input_a.backward(grad_output * self.input_b.data)
        if self.input_b.requires_grad:
            self.input_b.backward(grad_output * self.input_a.data)


class PowBackward:
    """Backward pass for power: z = a ** n

    Gradient:
        dL/da = dL/dz * dz/da = dL/dz * n * a^(n-1)
    """

    def __init__(self, input_tensor: 'Tensor', exponent: float) -> None:
        self.input_tensor = input_tensor
        self.exponent = exponent
        self.name = "PowBackward"

    def backward(self, grad_output: NDArray) -> None:
        # Power rule: multiply by exponent and reduce power by 1
        if self.input_tensor.requires_grad:
            grad = grad_output * self.exponent * (self.input_tensor.data ** (self.exponent - 1))
            self.input_tensor.backward(grad)
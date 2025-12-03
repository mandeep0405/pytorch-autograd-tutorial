"""
Custom Tensor class with automatic differentiation.

This demonstrates how PyTorch builds a computational graph and
computes gradients automatically using the chain rule.
"""

from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from backward_ops import AddBackward, PowBackward, MulBackward


class Tensor:
    """Tensor with operators doing ALL the autograd work"""

    def __init__(
        self,
        data: Union[int, float, list, NDArray],
        requires_grad: bool = False,
        grad_fn: Optional[object] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Args:
            data: The actual numeric value (scalar, list, or numpy array)
            requires_grad: Whether to track gradients for this tensor
            grad_fn: The backward operation that created this tensor
            name: Optional name for debugging
        """
        # Convert to numpy array for consistent vector operations
        if isinstance(data, (int, float)):
            self.data = np.array(data)
        elif isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = np.array(data) if not isinstance(data, np.ndarray) else data

        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn
        self.name = name or f"Tensor({data})"
    
    def backward(self, grad: Optional[NDArray] = None) -> None:
        """Compute gradients via backpropagation.

        Args:
            grad: Gradient from the next operation (defaults to ones with same shape)
        """
        # Initialize gradient for the output (dL/dL = 1)
        if grad is None:
            grad = np.ones_like(self.data)

        # Accumulate gradients if this tensor requires them
        if self.requires_grad:
            if self.grad is None:
                self.grad = grad
            else:
                # Accumulate gradients (important for nodes used multiple times)
                self.grad += grad

        # Continue backpropagation through the computational graph
        if self.grad_fn is not None:
            self.grad_fn.backward(grad)
    
    def zero_grad(self) -> None:
        """Reset gradients to None (like PyTorch's optimizer.zero_grad())"""
        self.grad = None
    
    # ============================================================
    # ALL LOGIC IN THE OPERATOR METHODS!
    # No need for separate add(), mul(), pow()
    # ============================================================
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Addition - ALL logic here"""
        # 1. Compute result
        result_data = self.data + other.data
        
        # 2. Check if we need gradients
        requires_grad = self.requires_grad or other.requires_grad
        
        # 3. Create grad_fn if needed
        grad_fn = AddBackward(self, other) if requires_grad else None
        
        # 4. Return new tensor
        return Tensor(
            result_data,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            name=f"({self.name}+{other.name})"
        )
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Multiplication - ALL logic here"""
        result_data = self.data * other.data
        requires_grad = self.requires_grad or other.requires_grad
        grad_fn = MulBackward(self, other) if requires_grad else None
        
        return Tensor(
            result_data,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            name=f"({self.name}*{other.name})"
        )
    
    def __pow__(self, exponent: Union[int, float]) -> 'Tensor':
        """Power - ALL logic here"""
        result_data = self.data ** exponent
        requires_grad = self.requires_grad
        grad_fn = PowBackward(self, exponent) if requires_grad else None
        
        return Tensor(
            result_data,
            requires_grad=requires_grad,
            grad_fn=grad_fn,
            name=f"({self.name}**{exponent})"
        )
    
    def __repr__(self) -> str:
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}{grad_str})"

    @property
    def shape(self) -> tuple:
        """Return the shape of the tensor data"""
        return self.data.shape

    def item(self) -> Union[int, float]:
        """Return scalar value (only works for single-element tensors)"""
        return self.data.item()
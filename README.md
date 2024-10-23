# Micrograd

Micrograd is a simple implementation of an automatic differentiation engine, following the principles of backpropagation. This project is inspired by the lectures of Andrej Karpathy and demonstrates how the key components of gradient-based optimization work from scratch.

## Features
- Implements a computational graph where each operation creates a node.
- Supports basic mathematical operations such as addition, multiplication, and power.
- Enables backpropagation through the computational graph, accumulating gradients to optimize parameters.

### Usage

Here's a basic example of how to use Micrograd:

```python
from micrograd.engine import Value

# Create input variables
x = Value(2.0)
y = Value(-3.0)
z = Value(5.0)

# Define the computational graph
out = x * y + z**2

# Forward pass
print(f"Output: {out.data}")  # Result of the operation

# Backpropagate
out.backward()

# Access gradients
print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")

# Divergence measures between persistence diagrams
Python implementation of Wasserstein and bottleneck divergences for persistence diagrams used in the paper New divergence measures between persistence diagrams and stability of vectorizations. Configurable penalties for unmatched target points are provided.

The order of the diagrams matters: the divergence from A to B can differ from the divergence from B to A.

## Requirements and installation

Dependencies:

- Python 3
- NumPy
- SciPy

Install the Python dependencies with:

```bash
python -m pip install numpy scipy
```

The examples below were checked with Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. These are tested versions, not minimum-version requirements.

Download divergence_measures.py and place it in the same directory as your Python script, or in the working directory of your notebook. Import the functions as shown below.

Running the module by itself does not print results; it provides functions to import.

## Input format

Each diagram must be a NumPy array of shape (n, 2). Each row contains a birth and a death:

```python
import numpy as np

diagram = np.array([
    [0.0, 2.0],
    [1.0, 3.5],
    [0.0, np.inf],
], dtype=float)

empty_diagram = np.empty((0, 2), dtype=float)
```

Births must be finite, deaths may be finite or positive infinity, and every birth must be less than or equal to its corresponding death. Do not use NaN values.


## Public functions

| Function | Usage |
| `bottleneck_divergence(dgm1, dgm2, func=arctan_penalty)` | Compute the directed bottleneck divergence. |
| `wasserstein_divergence(dgm1, dgm2, func, p=1)` | Compute the directed Wasserstein divergence for finite `p >= 1`. Use `p=np.inf` to call the bottleneck routine. |

Both functions return a float, which may be np.inf. The 'func' argument is required for wasserstein_divergence; the bottleneck function defaults to 'arctan_penalty'.

Functions whose names begin with an underscore are internal helpers.

## Penalty functions

| Function | Default parameters | Admissible parameters |
| `arctan_penalty(x, a=1.0, b=1.0)` | `a=1.0, b=1.0` | Finite `a, b >= 0` and `a*b <= 1`. |
| `tanh_penalty(x, a=1.0, b=1.0)` | `a=1.0, b=1.0` | Finite `a, b >= 0` and `a*b <= 2`. |
| `linear_penalty(x, slope=0.5)` | `slope=0.5` | Finite `0 <= slope <= 1`. |

Use 'functools.partial' to configure a penalty's parameters, as in the second example.

Custom penalties must accept a NumPy array and return an array of the same shape or a broadcastable scalar. For the intended divergences, they must be nonnegative, subadditive, and no greater than identity function. These mathematical properties are the caller's responsibility. Penalties must also evaluate correctly at np.inf when the target contains essential points.

## Example 1: finite diagrams and asymmetry

```python
import numpy as np
from persistence_divergences import (
    arctan_penalty,
    bottleneck_divergence,
    wasserstein_divergence,
)

A = np.array([[0.0, 2.0]])
B = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=float)

forward = wasserstein_divergence(A, B, func=arctan_penalty, p=2)
reverse = wasserstein_divergence(B, A, func=arctan_penalty, p=2)
bottleneck = bottleneck_divergence(A, B)

print(f"W2(A || B): {forward:.6f}")
print(f"W2(B || A): {reverse:.6f}")
print(f"Bottleneck(A || B): {bottleneck:.6f}")
```

Expected output:

```text
W2(A || B): 0.785398
W2(B || A): 1.000000
Bottleneck(A || B): 0.785398
```

## Example 2: custom parameters and essential points

```python
from functools import partial
import numpy as np
from persistence_divergences import arctan_penalty, wasserstein_divergence

A = np.array([[0.0, np.inf]], dtype=float)
B = np.array([[0.0, np.inf], [4.0, np.inf]], dtype=float)

# Admissible parameters: a*b = 0.75 <= 1.
penalty = partial(arctan_penalty, a=0.5, b=1.5)

forward = wasserstein_divergence(A, B, func=penalty, p=2)
bottleneck = wasserstein_divergence(A, B, func=penalty, p=np.inf)
reverse = wasserstein_divergence(B, A, func=penalty, p=2)

print(f"W2(A || B): {forward:.6f}")
print(f"Bottleneck(A || B): {bottleneck:.6f}")
print(f"W2(B || A): {reverse}")
```

Expected output:

'''text
W2(A || B): 2.356194
Bottleneck(A || B): 2.356194
W2(B || A): inf
'''

The extra essential target point receives a finite penalty in the forward direction. In the reverse direction, an essential source point cannot be matched at finite cost, so the result is np.inf.

## Current limitations
- **Large finite exponents:** directly raising costs to 'p' can cause overflow or underflow and produce an error or an incorrect value. Use p=np.inf when the intended quantity is the bottleneck divergence.
- **Memory:** the implementation constructs a dense augmented cost matrix, so memory use grows quadratically with the combined number of input points.

## Author and citation

Authors: Martín Mijangos, Alessandro Bravetti

When using this code in research, please cite this repository and record the release or commit used in your experiments.



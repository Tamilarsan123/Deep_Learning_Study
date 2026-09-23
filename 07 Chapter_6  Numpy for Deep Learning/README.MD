# NumPy Practice

Personal practice notebooks covering NumPy fundamentals — from "why NumPy exists" through array properties, reshaping, and a from-scratch linear regression model built using only NumPy.

## Project Files

| File | Description |
|---|---|
| `Numpy_Chapter_1_The_Why.ipynb` | Why NumPy exists: speed vs. Python lists, multi-dimensional data, vectorization, math/stats ops, and a mini neural network (linear regression) built from scratch with NumPy |
| `Chapter_02_Numpy_Properties_and_Operations.ipynb` | List vs. array properties, 1D/2D/3D arrays, dtype conversion, list↔array conversion, reshaping/flattening, sequencing (`arange`, `linspace`), repetition, and random number generation |
| `Practice_Numpy.ipynb` | Practice version of Chapter 2 — same concepts (array properties, reshaping, random numbers) worked through independently |
| `*-checkpoint.ipynb` files | Jupyter auto-save checkpoints of the above notebooks — safe to ignore/delete, not needed to run the material |

## Requirements

```bash
pip install numpy pandas matplotlib
```

> `Chapter_02_Numpy_Properties_and_Operations.ipynb` also reads a `drinks.csv` file in the "Repetition" section — place that file in the same folder as the notebook, or update/remove the path if you don't have it.

## Topics Covered

### 1. Numpy_Chapter_1_The_Why.ipynb
- **Why NumPy?** — Speed, structure, vectorization, math, AI foundation
- Python list vs. NumPy array creation and performance comparison (`%%time`, large-array timing)
- Multi-dimensional arrays (2D matrices, 3D+ tensors) and what they represent (images)
- Vectorized operations vs. manual loops
- Statistical operations: `np.mean`, `np.std`
- **Mini project:** builds a single-neuron linear regression model (`y = 2x + 3`) from scratch — forward pass, MSE loss, manual gradient computation, and a gradient descent training loop — using nothing but NumPy

### 2. Chapter_02_Numpy_Properties_and_Operations.ipynb / Practice_Numpy.ipynb
- **List vs. Array:** element-wise operations, `.append()` vs `.extend()`, why `list + int` fails but `array + int` works
- **Array creation:** 1D, 2D, 3D arrays; `.ndim`, `.size`, `.argmax()`, `.fill()`, `.repeat()`
- **Type conversion:** int → float (`.astype()`), list ↔ array conversion (`.tolist()`, `np.array()`)
- **Reshaping:** `.reshape()`, `.flatten()` (n-dim → 1D)
- **Sequencing & generation:** `np.arange()`, `np.linspace()`, `np.zeros()`, `np.ones()`
- **Random numbers:** `np.random.rand()` (uniform), `np.random.randn()` (normal), `np.random.seed()` for reproducibility, visualized with `matplotlib.pyplot.hist()`
- **Normal distribution** empirical rule (68-95-99.7%)

## How to Run

1. Install the requirements above.
2. Open any notebook in Jupyter Notebook / JupyterLab.
3. Run cells top to bottom — each notebook is organized into numbered chapters/sections with markdown headers explaining the concept before the code that demonstrates it.

**Suggested order** if working through this as a learning path:
1. `Numpy_Chapter_1_The_Why.ipynb` — motivation and core concepts
2. `Chapter_02_Numpy_Properties_and_Operations.ipynb` — properties and operations in depth
3. `Practice_Numpy.ipynb` — reinforce Chapter 2 concepts independently

## Author

Tamilarasan K

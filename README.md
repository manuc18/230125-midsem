# partB/data — Dataset Documentation

## Dataset: `make_moons` (sklearn.datasets)

**Type:** Synthetic binary classification dataset  
**Samples:** 500 (350 train / 150 test, 70/30 split)  
**Features:** 2 continuous numeric features  
**Classes:** 2 (binary, balanced)  
**Noise level:** 0.25  
**Random seed:** 42

---

## How to Obtain

This dataset requires **no download**. It is generated programmatically using:

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
```

It is included in `scikit-learn >= 0.18` and requires no external files.

---

## How It Is Used

The dataset is used across all Part B notebooks:

| Notebook | Usage |
|----------|-------|
| `task_2_1.ipynb` | Dataset selection, visualisation, preprocessing |
| `task_2_2.ipynb` | Core reproduction — Nyström vs RFF accuracy comparison |
| `task_2_3.ipynb` | Result comparison, reproducibility checklist |
| `task_3_1.ipynb` | Ablation study — budget sweep and landmark ablation |
| `task_3_2.ipynb` | Failure mode — compared against uniform random dataset |

---

## Preprocessing

All notebooks apply `StandardScaler` (zero mean, unit variance) before any kernel computation:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

This is standard for RBF kernel methods which are sensitive to feature scale.

---

## Why This Dataset

`make_moons` is non-linearly separable in the original 2D space, meaning a nonlinear kernel is genuinely required. The RBF kernel on this dataset produces a kernel matrix with fast-decaying eigenvalues (λ₁/λ₁₀ ≈ 36.7), which is the structural property that makes Nyström outperform RFF according to the paper's Theorem 1.

---

## Failure Mode Dataset (Task 3.2 Only)

Task 3.2 additionally uses a **uniform random dataset** generated as:

```python
Xb = np.random.uniform(-2, 2, size=(500, 2))
yb = (Xb[:, 0] + Xb[:, 1] > 0).astype(int)
```

This dataset is also generated in-memory and requires no file download.

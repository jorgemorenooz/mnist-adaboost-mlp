# AdaBoost on MNIST — from scratch, with benchmarks vs scikit‑learn and a Keras MLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![NumPy](https://img.shields.io/badge/NumPy-✓-orange)]()
[![scikit--learn](https://img.shields.io/badge/scikit--learn-✓-ff9f1c)]()
[![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow%2FKeras-✓-ff6f00)]()

**TL;DR**  
Single‑file project that implements **AdaBoost from scratch** (NumPy, decision stumps) on **MNIST**, then benchmarks it against **`sklearn.AdaBoostClassifier`** and a small **Keras MLP**. Includes binary (one‑vs‑rest) and multi‑class experiments plus plots of **accuracy vs. training time**.

---

## 🔎 What this shows (specs)

- **Algorithm insight**: a readable AdaBoost implementation with weighted errors and margins.
- **Engineering rigor**: measured runs with reproducibility, clear plots, and comparisons.
- **Practical baseline**: tiny MLP in Keras for contrast with classic boosting.

> Repo structure: a single Python file (e.g., `main.py`). Toggle experiments from the `__main__` block.

---

## 🧰 Environment & Installation

- Python **3.10+**
- CPU is enough; GPU is optional for the Keras MLP.

```bash
pip install -r requirements.txt
```

**`requirements.txt` (minimal)**
```txt
numpy
matplotlib
scikit-learn
tensorflow>=2.13
```

> If you only plan to run the classical ML parts, TensorFlow is only needed for the MLP baseline.

---

## ▶️ How to run

By default, the script runs the **binary AdaBoost** experiment (digit **9** vs rest).  
To change experiments, **comment/uncomment** the corresponding lines in the `if __name__ == "__main__":` section.

**Run the default experiment**
```bash
python main.py
```

**Enable other experiments** (edit `__main__` section and uncomment the desired call):

- Binary performance curves (accuracy/time vs `A` and `T`)
  ```python
  # tarea_1C_graficas_rendimiento()
  ```

- Multi‑class AdaBoost (One‑vs‑Rest)
  ```python
  # tarea_1D_adaboost_multiclase(T=25, A=300, verbose=True)
  ```

- scikit‑learn AdaBoost comparison
  ```python
  # tarea_2A_AdaBoostClassifier_default(100)
  ```

- Compare plots: scratch AdaBoost vs scikit‑learn
  ```python
  # tarea_2B_graficas_rendimiento()
  ```

- Keras MLP baseline
  ```python
  # tarea_2D_clasificador_MLP_para_MNIST_con_Keras()
  ```

- Combined plots (AdaBoost scratch + scikit‑learn + MLP)
  ```python
  # tarea_2F_graficas_rendimiento()
  ```

> Plots show **accuracy (train/test)** and **training time (s)** against the estimator/search parameters.  
> Figures are displayed interactively; you can save them from the window, or extend the code to save into `./plots/`.

---

## 📊 Example results layout

Fill this table after your first run (numbers are dataset/seed dependent).

| Model                               | Train Acc | Test Acc | Train Time (s) |
|-------------------------------------|-----------|----------|----------------|
| AdaBoost (scratch, T=25, A=300)     |     –     |    –     |       –        |
| `sklearn.AdaBoostClassifier` (T=40) |     –     |    –     |       –        |
| Keras MLP (2 dense layers)          |     –     |    –     |       –        |

---

## 🧠 How the scratch AdaBoost works

- Weak learner: **decision stump** (`x_j < θ` or `x_j > θ` with a polarity).
- Instance weights start uniform and are **updated every round** based on errors.
- The strong classifier sums **α-weighted** predictions from weak learners.
- Multi‑class = **One‑vs‑Rest**: train 10 binary models and pick the highest margin.

---

## ♻️ Reproducibility

- Seed **NumPy**, **Python random**, and **TensorFlow**.
- Keep preprocessing deterministic (MNIST normalized to `[0,1]`).

> See “Implementation fixes” below for a `set_seed(42)` helper and TF log/OneDNN flags ordering.

---

## ⚙️ Implementation fixes (before → after, with reasons)

Below are small but important tweaks that make the project **cleaner, faster, and more stable**.  
Each shows the **original** snippet and the **fixed** one, plus a short **why**.

### 1) Set TensorFlow env flags **before** importing TensorFlow
**Why**: TF reads these at import; setting later won’t take effect.
```python
# BEFORE
import tensorflow as tf
import logging, os
logging.disable(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
```
```python
# AFTER
import os, logging
logging.disable(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # flags now applied
```

### 2) Avoid mixing standalone Keras with TF‑Keras
**Why**: Mixing `from tensorflow import keras` and `from keras...` can cause version drift.
```python
# BEFORE
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
```
```python
# AFTER (use TF‑Keras consistently)
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
```

### 3) Global seeding for reproducibility
**Why**: Benchmarks/plots should be repeatable.
```python
# BEFORE: no seeding
```
```python
# AFTER
def set_seed(seed: int = 42):
    import random, numpy as np, tensorflow as tf
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

# call once in __main__
if __name__ == "__main__":
    set_seed(42)
```

### 4) Stable α computation (clip error)
**Why**: Prevents log(0) when error→0 or →1 and avoids exploding weights.
```python
# BEFORE
alfa = (0.5 * np.log((1-best_error)/(best_error+1e-15)))
```
```python
# AFTER
eps = 1e-12
err = np.clip(best_error, eps, 1.0 - eps)
alfa = 0.5 * np.log((1.0 - err) / err)
```

### 5) Signed predictions with deterministic tie handling
**Why**: np.sign(0)==0; keep labels in {−1,+1}.
```python
# BEFORE
predicciones_finales = np.sign(predicciones_finales)
```
```python
# AFTER
predicciones_finales = np.sign(predicciones_finales)
predicciones_finales[predicciones_finales == 0] = 1
```

### 6) Deterministic, safer class balancing
**Why**: Use a fixed seed and guard when class sizes differ.
```python
# BEFORE
negativos_random = np.random.choice(negativos, size=len(positivos), replace=False)
```
```python
# AFTER
def balancear_clases(X, Y, clase, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.where(Y == clase)[0]
    neg = np.where(Y != clase)[0]
    k = min(len(pos), len(neg))
    pos = rng.choice(pos, size=k, replace=False)
    neg = rng.choice(neg, size=k, replace=False)
    idx = np.concatenate([pos, neg]); rng.shuffle(idx)
    Xb = X[idx]
    Yb = np.where(Y[idx] == clase, 1, -1).astype(np.int8)
    return Xb, Yb
```

### 7) Cleaner weighted error
**Why**: Slightly clearer and avoids an intermediate array.
```python
# BEFORE
error = np.sum(Dpesos * (predictions != Y))
```
```python
# AFTER
mismatch = (predictions != Y)
error = Dpesos[mismatch].sum()
```

### 8) (Optional) Better stumps via weighted fitting over quantile thresholds
**Why**: Replacing random feature/threshold guesses with **data‑driven** thresholds improves weak learner quality and speed.
```python
# BEFORE (random DecisionStump)
self.caracteristica = np.random.randint(0, n_features)
self.umbral = np.random.rand()
self.polaridad = np.random.choice([1, -1])
```
```python
# AFTER (sketch)
class WeightedDecisionStump:
    def fit(self, X, y, w, max_features=None, n_thresholds=16, rng=None):
        # sample features, try n quantile thresholds each, pick lowest weighted error
        ...
    def predict(self, X):
        ...
# then in Adaboost.fit(): stump.fit(X, Y, Dpesos, n_thresholds=min(self.intentosClasif, 32))
```

### 9) Minor MLP cleanup (Flatten not needed if input is already (n, 784))
```python
# BEFORE
model = Sequential()
model.add(Flatten(input_shape=(28*28,)))
model.add(Dense(128, activation='relu'))
...
```
```python
# AFTER
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])
```

---

## 🗺️ Roadmap

- Cache sorted feature indices for stump search (O(n log n) once, then O(n) per threshold).
- Early stopping if training margin stops improving.
- Try multi‑class loss variants (e.g., SAMME.R).
- Add `--save-plots` CLI and write to `./plots/` by default.
- Add `requirements-lock.txt` and a `Makefile` target for common runs.

---

## 🙏 Acknowledgements

- MNIST by Yann LeCun et al. (via `keras.datasets.mnist`).
- scikit‑learn and TensorFlow teams.

---

## 📬 Contact

If you’re interested in the implementation or want to discuss improvements, feel free to reach out on LinkedIn.

# AdaBoost on MNIST — Step-by-Step (EN/ES)

> A very clear walkthrough of this project for high-school students. Each paragraph is explained in **English** 🇺🇸 and **Spanish** 🇪🇸.

---

## 1) What this project does

- 🇺🇸 **English:** This project trains a simple **AdaBoost** classifier to recognize handwritten digits from the **MNIST** dataset. It also includes a small **MLP (neural network)** as a reference. You can run binary “one digit vs the rest,” full **multiclass** (0–9 with one-vs-all), and draw **performance charts**.  
- 🇪🇸 **Español:** Este proyecto entrena un clasificador **AdaBoost** sencillo para reconocer dígitos escritos a mano del conjunto **MNIST**. También incluye una pequeña **MLP (red neuronal)** como referencia. Puedes ejecutar el caso binario “un dígito contra el resto”, el caso **multiclase** (0–9 con uno-contra-todos) y dibujar **gráficas de rendimiento**.

---

## 2) Files & structure

- 🇺🇸 **English:**  
  - `Jorge_Moreno_Ozores.py` — all the code (AdaBoost, MLP, plots, helpers).  
  - No extra data files; MNIST downloads automatically via `tf.keras`.  
- 🇪🇸 **Español:**  
  - `Jorge_Moreno_Ozores.py` — todo el código (AdaBoost, MLP, gráficas, utilidades).  
  - Sin archivos de datos extra; MNIST se descarga automáticamente con `tf.keras`.

---

## 3) How to run

- 🇺🇸 **English:**  
  - Install Python 3.10+ and run:  
    ```bash
    pip install tensorflow matplotlib scikit-learn numpy
    python Jorge_Moreno_Ozores.py
    ```  
  - By default it runs a **binary AdaBoost** for digit **9** and prints accuracy and time.  
  - Uncomment other lines in `__main__` to run the multiclass version, sklearn baseline, MLP, and plots.  
- 🇪🇸 **Español:**  
  - Instala Python 3.10+ y ejecuta:  
    ```bash
    pip install tensorflow matplotlib scikit-learn numpy
    python Jorge_Moreno_Ozores.py
    ```  
  - Por defecto ejecuta **AdaBoost binario** para el dígito **9** e imprime precisión y tiempo.  
  - Descomenta otras líneas en `__main__` para lanzar la versión multiclase, la línea base de sklearn, la MLP y las gráficas.

---

## 4) Reproducibility (same results every run)

- 🇺🇸 **English:** We **fix random seeds** for Python, NumPy, and TensorFlow. We also set **TensorFlow flags before importing it** so settings take effect. That means: same inputs ⇒ same outputs.  
- 🇪🇸 **Español:** **Fijamos semillas** para Python, NumPy y TensorFlow. También ponemos **las flags de TensorFlow antes de importarlo** para que surtan efecto. Así: mismas entradas ⇒ mismos resultados.

---

## 5) Key concepts (mini-glossary)

- 🇺🇸 **English:**  
  - **Classifier:** a program that assigns a label (e.g., “this is a 9”).  
  - **Feature:** a measurable property of data (here, a pixel value among 784).  
  - **Threshold:** a cut-off value used to split decisions (e.g., pixel < 0.3).  
  - **Polarity (±1):** which side of the threshold predicts +1 vs −1.  
  - **Weighted error:** mistakes counted with sample weights (some examples matter more).  
  - **Alpha (α):** the weight given to each weak classifier in AdaBoost.  
  - **Margin:** the sum of (alpha × prediction) across weak classifiers; sign decides the class.  
  - **Seed:** a fixed number to make randomness repeatable.  
- 🇪🇸 **Español:**  
  - **Clasificador:** programa que asigna una etiqueta (p. ej., “esto es un 9”).  
  - **Característica (feature):** propiedad medible del dato (aquí, un píxel entre 784).  
  - **Umbral (threshold):** valor de corte para decidir (p. ej., píxel < 0.3).  
  - **Polaridad (±1):** qué lado del umbral predice +1 frente a −1.  
  - **Error ponderado:** fallos contados con pesos (algunos ejemplos importan más).  
  - **Alfa (α):** peso que se da a cada clasificador débil en AdaBoost.  
  - **Margen:** suma de (alfa × predicción) de los clasificadores débiles; el signo decide la clase.  
  - **Semilla:** número fijo para que el azar sea repetible.

---

## 6) What is AdaBoost (intuitive)

- 🇺🇸 **English:** AdaBoost builds a **team** of very simple “weak” classifiers (here: **decision stumps** that look at just **one feature** and a **threshold**). Each weak classifier is not great alone, but AdaBoost **weights** them (using **alpha**) and **adds** their opinions. Together, they become strong.  
- 🇪🇸 **Español:** AdaBoost forma un **equipo** de clasificadores “débiles” muy simples (aquí: **stumps** que miran **una sola característica** y un **umbral**). Cada débil no es muy bueno, pero AdaBoost los **pondera** (con **alfa**) y **suma** sus opiniones. Juntos se vuelven fuertes.

---

## 7) The Decision Stump (our weak classifier)

- 🇺🇸 **English:** A stump picks **one pixel** (feature), a **threshold** (like 0.27), and a **polarity** (which side is +1). Then it predicts +1 or −1 for each image based on that single rule.  
- 🇪🇸 **Español:** Un stump elige **un píxel** (característica), un **umbral** (como 0.27) y una **polaridad** (qué lado es +1). Luego predice +1 o −1 para cada imagen con esa única regla.

---

## 8) AdaBoost training — the `fit` loop (clear steps)

**Round by round (T rounds). In each round we try A random stumps and keep the best one.**

1) **Initialize sample weights (D):**  
   - 🇺🇸 All training images start with equal weight (each is equally important).  
   - 🇪🇸 Todas las imágenes de entrenamiento empiezan con el mismo peso (todas importan igual).

2) **Try A random stumps; pick the best:**  
   - 🇺🇸 We create A random stumps (random feature, threshold, polarity).  
     For each stump, we predict on all images and compute the **weighted error** (sum weights where it’s wrong). We **keep the stump with the smallest weighted error**.  
   - 🇪🇸 Creamos A stumps aleatorios (característica, umbral y polaridad aleatorios).  
     Para cada stump, predecimos en todas las imágenes y calculamos el **error ponderado** (sumamos pesos en los fallos). **Guardamos el stump con menor error ponderado**.

3) **Compute alpha (α) for the best stump:**  
   - 🇺🇸 We convert the error into a strength with  
     \\( \\alpha = \\tfrac{1}{2}\\log\\frac{1 - e}{e} \\)  
     We **clip** `e` to avoid 0 or 1 (numerical safety).  
   - 🇪🇸 Convertimos el error en “fuerza” con  
     \\( \\alpha = \\tfrac{1}{2}\\log\\frac{1 - e}{e} \\)  
     Hacemos **clip** de `e` para evitar 0 o 1 (seguridad numérica).

4) **Update sample weights (D):**  
   - 🇺🇸 Increase weights of misclassified images and decrease weights of correct ones using  
     \\( D \\leftarrow D \\cdot \\exp(-\\alpha \\cdot y \\cdot \\hat{y}) \\)  
     Then **normalize** D so it sums to 1.  
   - 🇪🇸 Aumentamos el peso de las imágenes mal clasificadas y reducimos el de las bien clasificadas con  
     \\( D \\leftarrow D \\cdot \\exp(-\\alpha \\cdot y \\cdot \\hat{y}) \\)  
     Después **normalizamos** D para que sume 1.

5) **Store the best stump:**  
   - 🇺🇸 Save the stump and its α; we’ll use it for prediction later.  
   - 🇪🇸 Guardamos el stump y su α; lo usaremos luego para predecir.

6) **Repeat for T rounds:**  
   - 🇺🇸 Each round focuses more on the examples that were hard in previous rounds.  
   - 🇪🇸 Cada ronda se centra más en los ejemplos que fueron difíciles en rondas anteriores.

---

## 9) How prediction works

- 🇺🇸 **English:** For a new image, each saved stump makes a prediction in {−1, +1}. We compute a **margin** = sum over stumps of (α × prediction). The final class is the **sign** of the margin. We use a **deterministic tie-break**: if margin = 0, we return **+1** (no zeros).  
- 🇪🇸 **Español:** Para una nueva imagen, cada stump guardado da una predicción en {−1, +1}. Calculamos el **margen** = suma de (α × predicción). La clase final es el **signo** del margen. Usamos un **desempate determinista**: si el margen = 0, devolvemos **+1** (sin ceros).

---

## 10) Binary vs Multiclass

- 🇺🇸 **English:**  
  - **Binary:** Detect one digit (e.g., “is it a 9?”) ⇒ labels are {−1, +1}.  
  - **Multiclass (0–9):** We train 10 binary classifiers (one-vs-all). For a test image, each classifier outputs a **margin**, and we pick the class with the **largest margin**.  
- 🇪🇸 **Español:**  
  - **Binario:** Detectar un dígito (p. ej., “¿es un 9?”) ⇒ etiquetas {−1, +1}.  
  - **Multiclase (0–9):** Entrenamos 10 clasificadores binarios (uno-contra-todos). Para una imagen, cada clasificador devuelve un **margen** y elegimos la clase con el **margen más grande**.

---

## 11) Safe class balancing (for one-vs-all training)

- 🇺🇸 **English:** To avoid huge class imbalance (far more “not-9” than “9”), we **balance** the training subset: take all positives and the **same number** of negatives (or as many as exist), then **shuffle** with a fixed seed so it’s **deterministic**.  
- 🇪🇸 **Español:** Para evitar un gran desbalance (muchos más “no-9” que “9”), **balanceamos** el subconjunto: tomamos todos los positivos y el **mismo número** de negativos (o los que haya), y **barajamos** con semilla fija para que sea **determinista**.

---

## 12) Reference MLP (neural network)

- 🇺🇸 **English:** We include a small **MLP** (two Dense layers and a softmax) to compare with AdaBoost. Since inputs are already flat (784), we **don’t need a Flatten layer**.  
- 🇪🇸 **Español:** Incluimos una **MLP** pequeña (dos capas Dense y softmax) para comparar con AdaBoost. Como las entradas ya son planas (784), **no necesitamos capa Flatten**.

---

## 13) Plots & experiments

- 🇺🇸 **English:** Functions `tarea_1C_graficas_rendimiento`, `tarea_2B_graficas_rendimiento`, and `tarea_2F_graficas_rendimiento` scan hyper-parameters and draw charts of **accuracy** and **training time** so you can see trade-offs.  
- 🇪🇸 **Español:** Las funciones `tarea_1C_graficas_rendimiento`, `tarea_2B_graficas_rendimiento` y `tarea_2F_graficas_rendimiento` exploran hiperparámetros y dibujan gráficas de **precisión** y **tiempo de entrenamiento** para ver los compromisos.

---

## 14) Troubleshooting (common issues)

- 🇺🇸 **English:**  
  - If results change between runs, ensure you **didn’t edit seeds** and that you run a **single process**.  
  - If TensorFlow prints many messages, confirm env flags are **set before importing TensorFlow**.  
- 🇪🇸 **Español:**  
  - Si los resultados cambian entre ejecuciones, comprueba que **no cambiaste las semillas** y que ejecutas en **un solo proceso**.  
  - Si TensorFlow imprime muchos mensajes, confirma que las flags se **ponen antes de importar TensorFlow**.

---

## 15) Quick commands

- 🇺🇸 **English:**  
  - Binary (default in `__main__`):  
    ```bash
    python Jorge_Moreno_Ozores.py
    ```  
  - Multiclass one-vs-all (edit `__main__` to uncomment):  
    ```python
    # tarea_1D_adaboost_multiclase(T=25, A=300, verbose=True)
    ```  
  - MLP reference (edit `__main__` to uncomment):  
    ```python
    # tarea_2D_clasificador_MLP_para_MNIST_con_Keras()
    ```  
- 🇪🇸 **Español:**  
  - Binario (por defecto en `__main__`):  
    ```bash
    python Jorge_Moreno_Ozores.py
    ```  
  - Multiclase uno-contra-todos (edita `__main__` y descomenta):  
    ```python
    # tarea_1D_adaboost_multiclase(T=25, A=300, verbose=True)
    ```  
  - MLP de referencia (edita `__main__` y descomenta):  
    ```python
    # tarea_2D_clasificador_MLP_para_MNIST_con_Keras()
    ```

---

## 16) Recent Changes (what we improved & why)

- 🇺🇸 **English:**  
  1. **Set TF env flags before import** — TF reads them at import time; doing it later has no effect (robustness, cleaner logs).  
  2. **Global seeding** (`random`, `numpy`, `tensorflow`, `PYTHONHASHSEED`) — ensures **reproducibility** (same results every run).  
  3. **Deterministic TF ops when available** — reduces run-to-run noise in GPU/CPU kernels.  
  4. **Stable alpha with error clipping** — avoid `log(0)` or `log(∞)` when a stump is perfect or terrible (numerical stability).  
  5. **Deterministic sign rule** (`sign_with_tie`, where 0 ⇒ +1) — avoids invalid class “0” in binary classification (consistency).  
  6. **Safe, deterministic class balancing** — never sample more negatives than exist; shuffle with a fixed seed (robustness + reproducibility).  
  7. **Clear weighted-error mask** — use a boolean mask `pred != Y` then sum weights (readability, same speed).  
  8. **Remove redundant Flatten in MLP** — inputs are already `(n, 784)` (simplicity, tiny efficiency gain).  
  9. **Set `random_state` in sklearn AdaBoost** — makes the sklearn baseline reproducible.  
  10. **Avoid shadowing names** (don’t use `tf` as a timer variable) — prevents confusion with the TensorFlow module.  
  11. **Unify Keras imports under `tf.keras`** — fewer inconsistencies across versions.  
- 🇪🇸 **Español:**  
  1. **Flags de TF antes de importar** — TF las lee al importarse; ponerlas después no sirve (robustez, logs limpios).  
  2. **Semillas globales** (`random`, `numpy`, `tensorflow`, `PYTHONHASHSEED`) — garantiza **reproducibilidad** (mismos resultados).  
  3. **Operaciones deterministas de TF si existen** — reduce variaciones entre ejecuciones en GPU/CPU.  
  4. **Alfa estable con clip del error** — evita `log(0)` o `log(∞)` cuando un stump es perfecto o muy malo (estabilidad numérica).  
  5. **Regla de signo determinista** (`sign_with_tie`, donde 0 ⇒ +1) — evita clase inválida “0” en binario (consistencia).  
  6. **Balanceo seguro y determinista** — nunca pedimos más negativos de los que hay; barajado con semilla (robustez + reproducibilidad).  
  7. **Máscara clara para error ponderado** — usamos `pred != Y` y sumamos pesos (legibilidad, misma velocidad).  
  8. **Quitar Flatten redundante en MLP** — la entrada ya es `(n, 784)` (simpleza, ligera eficiencia).  
  9. **`random_state` en AdaBoost de sklearn** — hace reproducible la línea base.  
  10. **Evitar nombres que se solapan** (no usar `tf` como tiempo) — evita confusiones con el módulo TensorFlow.  
  11. **Unificar imports de Keras en `tf.keras`** — menos inconsistencias entre versiones.

---

**Enjoy experimenting! / ¡Disfruta experimentando!**

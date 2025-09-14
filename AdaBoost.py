import os, logging, random, time

# ====== TensorFlow env flags and logging (must be set BEFORE importing TF/Keras) ======
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # avoid CPU math changes across runs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # silence TF logs
os.environ['TF_DETERMINISTIC_OPS'] = '1'        # request deterministic ops when available

logging.disable(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# ====== Global seeding for reproducibility ======
SEED = 42

def set_global_seed(seed: int = SEED):
    """Set seeds for Python, NumPy, and TensorFlow to get repeatable results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        tf.random.set_seed(seed)
        try:
            # Some TF versions expose deterministic toggle under experimental
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass

set_global_seed(SEED)

# ====== Utilities ======
def sign_with_tie(x: np.ndarray) -> np.ndarray:
    """
    Like np.sign but defines ties deterministically:
    sign_with_tie(0) = +1. This avoids '0' as a class in binary tasks.
    """
    return np.where(x >= 0, 1, -1)

def load_MNIST_for_adaboost():
    """Load MNIST, flatten to (n, 784), normalize to [0,1], keep integer labels."""
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test  = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    Y_train = Y_train.astype("int8")
    Y_test  = Y_test.astype("int8")
    return X_train, Y_train, X_test, Y_test

class DecisionStump:
    """Weak classifier: single threshold on one feature with polarity ±1."""
    def __init__(self, n_features: int):
        # Random but reproducible thanks to global seed
        self.feature = np.random.randint(0, n_features)
        self.threshold = np.random.uniform(0.0, 1.0)
        self.polarity = np.random.choice([1, -1])
        self.alpha = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        preds = np.ones(n)
        col = X[:, self.feature]
        if self.polarity == 1:
            preds[col < self.threshold] = -1
        else:
            preds[col > self.threshold] = -1
        return preds

class Adaboost:
    """
    Binary AdaBoost with random decision stumps.
    - T: number of selected weak classifiers
    - A: number of random stump attempts per boosting round (choose the best by weighted error)
    """
    def __init__(self, T=5, A=20):
        self.numClasifs = T
        self.numAttempts = A
        self.classifiers = []
    
    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool=False):
        n_obs, n_feat = X.shape
        D = np.full(n_obs, 1.0 / n_obs, dtype=np.float64)  # sample weights
        self.classifiers = []
        
        for t in range(1, self.numClasifs + 1):
            best_clf = None
            best_pred = None
            best_error = np.inf
            
            for _ in range(1, self.numAttempts + 1):
                clf = DecisionStump(n_feat)
                pred = clf.predict(X)
                mask = (pred != Y)                     # boolean mask of errors
                error = float(D[mask].sum())           # weighted error
                if error < best_error:
                    best_error = error
                    best_clf = clf
                    best_pred = pred
            
            # Numerically stable alpha: clip error away from {0,1}
            e = np.clip(best_error, 1e-12, 1.0 - 1e-12)
            alpha = 0.5 * np.log((1.0 - e) / e)
            best_clf.alpha = alpha
            
            # Update weights and renormalize
            D *= np.exp(-alpha * Y * best_pred)
            S = D.sum()
            if S == 0.0:
                D = np.full_like(D, 1.0 / len(D))     # safety fallback
            else:
                D /= S
            
            self.classifiers.append(best_clf)
            
            if verbose:
                pol = '+1' if best_clf.polarity == 1 else '-1'
                print(f"[t={t:02d}] feature={best_clf.feature}, thr={best_clf.threshold:.4f}, "
                      f"pol={pol}, weighted_err={best_error:.6f}, alpha={alpha:.5f}")
        return self.classifiers
                     
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the real-valued margin (sum of alpha * weak_pred)."""
        margin = np.zeros(X.shape[0], dtype=np.float64)
        for clf in self.classifiers:
            margin += clf.predict(X) * clf.alpha
        return margin
    
    def predictBin(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions in {-1, +1} with deterministic tie-breaking."""
        return sign_with_tie(self.predict(X))

def balancear_clases(X: np.ndarray, Y: np.ndarray, clase: int):
    """
    Balance positive (== clase) and negative (!= clase) samples in a safe, deterministic way.
    Returns X_balanced, Y_balanced with labels in {-1, +1}.
    """
    positives = np.where(Y == clase)[0]
    negatives = np.where(Y != clase)[0]

    n_pos = len(positives)
    n_neg = len(negatives)
    take_neg = min(n_neg, n_pos)  # never request more than available
    
    negatives_sampled = np.random.choice(negatives, size=take_neg, replace=False)
    positives_taken = positives[:take_neg]
    idx = np.concatenate((positives_taken, negatives_sampled))
    np.random.shuffle(idx)  # reproducible due to global seed
    
    Xb = X[idx]
    Yb = np.where(Y[idx] == clase, 1, -1)
    return Xb, Yb

# ====== Tasks ======
def binary_adaboost(clase: int, T: int, A: int, verbose: bool):
    """
    Train a binary AdaBoost to detect one digit vs the rest and report accuracy/time.
    Returns [train_acc(%), test_acc(%), time(s)].
    """
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    strong_classifier = Adaboost(T, A)
    
    if verbose:
        print(f"Training binary AdaBoost for digit {clase}, T={T}, A={A}")
    
    Y_train_bin = np.where(Y_train == clase, 1, -1)
    Y_test_bin  = np.where(Y_test  == clase, 1, -1)
    
    start = time.time()
    strong_classifier.fit(X_train, Y_train_bin, verbose)
    elapsed = time.time() - start
    
    Y_pred_train = strong_classifier.predictBin(X_train)
    Y_pred_test  = strong_classifier.predictBin(X_test)
    
    train_hits = np.sum(Y_train_bin == Y_pred_train)
    train_acc = (train_hits / len(Y_train_bin)) * 100.0
    test_hits = np.sum(Y_test_bin == Y_pred_test)
    test_acc = (test_hits / len(Y_test_bin)) * 100.0
    
    if verbose:
        print(f"Accuracies (train, test) and time: {train_acc:.2f}%, {test_acc:.2f}%, {elapsed:.3f} s")
    return [train_acc, test_acc, elapsed]  

def binary_adaboost_graph():
    """Grid over T and A; plot accuracy (train/test) and training time."""
    T_range = range(10, 30, 5)
    A_range = range(100, 500, 50)
    results = {t: [] for t in T_range}
    
    for t in T_range:
        for a in A_range:
            train_acc, test_acc, elapsed = binary_adaboost(clase=9, T=t, A=a, verbose=False)
            results[t].append((a, train_acc, test_acc, elapsed))
    
    for t in T_range:
        As, acc_train, acc_test, times = [], [], [], []
        for a, tr, te, tm in results[t]:
            As.append(a); acc_train.append(tr); acc_test.append(te); times.append(tm)
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('A (stump attempts)')
        ax1.set_ylabel('Accuracy (%)', color='tab:red')
        ax1.plot(As, acc_train, label='Train', marker='x', color='tab:red')
        ax1.plot(As, acc_test,  label='Test',  marker='o', color='tab:green', linestyle='dashed')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Training Time (s)', color='tab:blue')
        ax2.plot(As, times, label='Training Time', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(loc='upper right')

        plt.title(f'Accuracy & Training Time for T={t}')
        fig.tight_layout()
        plt.show()

def predict_multiclass(X_test: np.ndarray, clasificadores: dict) -> np.ndarray:
    """
    One-vs-all: for each class, collect its real-valued margins and pick argmax.
    """
    margins = np.zeros((X_test.shape[0], 10), dtype=np.float64)
    for clase, clf in clasificadores.items():
        margins[:, clase] = clf.predict(X_test)
    return np.argmax(margins, axis=1)

def adaboost_multiclass(T: int, A: int, verbose: bool=False):
    """
    Train 10 one-vs-all AdaBoost classifiers and evaluate multiclass accuracy.
    Returns (accuracy, total_time_seconds).
    """
    classes = range(10)
    classifiers = {}
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    
    t0 = time.time()
    for clase in classes:
        if verbose:
            print(f"Training one-vs-all AdaBoost for digit {clase}, T={T}, A={A}")
        Xb, Yb = balancear_clases(X_train, Y_train, clase)
        clf = Adaboost(T, A)
        clf.fit(Xb, Yb, verbose=False)
        classifiers[clase] = clf
    
    total_time = time.time() - t0
    pred = predict_multiclass(X_test, classifiers)
    correct = np.sum(pred == Y_test)
    total = len(Y_test)
    accuracy = correct / total
    
    if verbose:
        print(f"Multiclass accuracy: {accuracy*100:.2f}%, total time: {total_time:.3f} s")
    return accuracy, total_time

def adaBoostClassifier_default(n_estimators: int, verbose: bool=False):
    """Reference: use sklearn's AdaBoostClassifier with a fixed random_state."""
    if verbose:
        print(f"Training sklearn AdaBoostClassifier with T={n_estimators}")
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    adaboost_sklearn = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=0.1, random_state=SEED)
    t0 = time.time()
    adaboost_sklearn.fit(X_train, Y_train)
    total_time = time.time() - t0
    
    Y_pred_train = adaboost_sklearn.predict(X_train)
    Y_pred_test  = adaboost_sklearn.predict(X_test)
    
    accuracy_test  = accuracy_score(Y_test, Y_pred_test)
    accuracy_train = accuracy_score(Y_train, Y_pred_train)
    
    if verbose:
        print(f"Accuracies (train, test) and time: {(accuracy_train * 100):.2f}%, {(accuracy_test * 100):.2f}%, {total_time:.3f} s")
    return accuracy_test, total_time

def comparison_graph_myAdaBoost_vs_sklearn():
    """Compare my AdaBoost vs sklearn in accuracy/time across several T values."""
    T_values = [10, 20, 40]

    results_mine = {T: [] for T in T_values}
    results_sklearn = {T: [] for T in T_values}

    for T in T_values:
        for A in T_values:
            acc_mine, time_mine = adaboost_multiclass(T, A)
            results_mine[T].append((A, acc_mine, time_mine))
        for A in T_values:
            acc_skl, time_skl = adaBoostClassifier_default(T)
            results_sklearn[T].append((A, acc_skl, time_skl))

    for T in T_values:
        As_1D, acc_1D, time_1D = [], [], []
        acc_skl, time_skl = [], []
        
        for A, accm, timem in results_mine[T]:
            As_1D.append(A); acc_1D.append(accm); time_1D.append(timem)
        for _, accs, times in results_sklearn[T]:
            acc_skl.append(accs); time_skl.append(times)
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('A (stump attempts)')
        ax1.set_ylabel('Accuracy', color='tab:red')
        ax1.plot(As_1D, acc_1D, label=f'My AdaBoost T={T}', color='tab:red', marker='o')
        ax1.plot(As_1D, acc_skl, label=f'sklearn T={T}', color='magenta')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Training Time (s)', color='tab:blue')
        ax2.plot(As_1D, time_1D, label=f'My AdaBoost Time T={T}', color='forestgreen')
        ax2.plot(As_1D, time_skl, label=f'sklearn Time T={T}', color='lime')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.legend(loc='upper right')

        plt.title(f'Accuracy & Time for T={T}')
        fig.tight_layout()
        plt.show()
  
def MLP_MNIST_keras():
    """
    Simple MLP for MNIST with tf.keras.
    Note: input is already flat (784), so no Flatten layer is needed.
    """
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    t0 = time.time()
    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    elapsed = time.time() - t0
    print(f"Test Accuracy: {test_acc*100:.2f}%, {elapsed:.2f} seconds")
    return test_acc, elapsed

def comparison_graph_myAdaBoost_vs_sklearn_vs_Keras():
    """
    Compare: my AdaBoost, sklearn AdaBoost, and Keras MLP.
    Plots accuracy and training time versus A (reused as an index on x-axis).
    """
    T_values = [10, 20, 40]

    results_mine = {T: [] for T in T_values}
    results_sklearn = {T: [] for T in T_values}
    results_mlp = {T: [] for T in T_values}

    for T in T_values:
        for A in T_values:
            acc_mine, time_mine = adaboost_multiclass(T, A)
            results_mine[T].append((A, acc_mine, time_mine))
        for A in T_values:
            acc_skl, time_skl = adaBoostClassifier_default(T)
            results_sklearn[T].append((A, acc_skl, time_skl))
        for A in T_values:
            acc_mlp, time_mlp = MLP_MNIST_keras()
            results_mlp[T].append((A, acc_mlp, time_mlp)) 

    for T in T_values:
        As_1D, acc_1D, time_1D = [], [], []
        acc_skl, time_skl = [], []
        acc_mlp, time_mlp = [], []

        for A, accm, timem in results_mine[T]:
            As_1D.append(A); acc_1D.append(accm); time_1D.append(timem)
        for _, accs, times in results_sklearn[T]:
            acc_skl.append(accs); time_skl.append(times)
        for _, acck, timek in results_mlp[T]:
            acc_mlp.append(acck); time_mlp.append(timek)

        fig, ax1 = plt.subplots()
        ax1.set_xlabel('A (stump attempts / iteration index)')
        ax1.set_ylabel('Accuracy', color='tab:red')
        ax1.plot(As_1D, acc_1D, label=f'My AdaBoost T={T}', color='tab:red')
        ax1.plot(As_1D, acc_skl, label=f'sklearn T={T}', color='magenta')
        ax1.plot(As_1D, acc_mlp, label=f'Keras MLP', color='orange')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Training Time (s)', color='tab:blue')
        ax2.plot(As_1D, time_1D, label=f'My AdaBoost Time T={T}', color='forestgreen')
        ax2.plot(As_1D, time_skl, label=f'sklearn Time T={T}', color='lime')
        ax2.plot(As_1D, time_mlp, label='Keras MLP Time', color='yellow')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1 + h2, l1 + l2, loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.title(f'Accuracy & Training Time for T={T} including MLP')
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Reproducible example run
    res_1A = binary_adaboost(clase=9, T=25, A=1000, verbose=True)
    # Other tasks (uncomment as needed)
    # binary_adaboost_graph()
    # adaboost_multiclass(T=25, A=300, verbose=True)
    # adaBoostClassifier_default(100, verbose=True)
    # comparison_graph_myAdaBoost_vs_sklearn()
    # MLP_MNIST_keras()
    # comparison_graph_myAdaBoost_vs_sklearn_vs_Keras()

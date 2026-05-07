# import scipy.io as sio
# import sys
# import pacmap
# import numpy as np
#
# def pyumap(fn_in, fn_out):
#   random_seed = 1986
#   mat_contents = sio.loadmat(fn_in)
#   X = mat_contents["X"]
#
#   embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
#   # fit the data (The index of transformed data corresponds to the index of the original data)
#   X_pacmap = embedding.fit_transform(X, init="random") #init="pca"
#   sio.savemat(fn_out, {"X_pacmap":X_pacmap})
#
# if __name__ == '__main__':
#     x = str(sys.argv[1])
#     y = str(sys.argv[2])
#     pyumap(x, y)


import os
import sys
import warnings
import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat

warnings.filterwarnings("ignore")

def safe_make_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def load_X(fn_in: Path, key_candidates=("X", "x", "data")):
    mat = loadmat(fn_in)
    for k in key_candidates:
        if k in mat:
            X = mat[k]
            break
    else:
        raise KeyError(f"No array key found in {fn_in}. Tried {key_candidates}")
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def compute_embedding(X, target_k=60, init="random", random_state=42):
    # remove NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    n_samples = X.shape[0]

    # if sample is 0: return empty
    if n_samples == 0:
        return np.empty((0, 2))

    # If there are only 1 or 2 samples: fall back to a simpler mode / degrade gracefully.
    if n_samples < 3:
        try:
            from sklearn.decomposition import PCA
            n_comp = 2 if n_samples >= 2 else 1
            Y = PCA(n_components=n_comp).fit_transform(X)
            if n_comp == 1:
                Y = np.hstack([Y, np.zeros((n_samples, 1))])
            return Y
        except Exception:
            return np.zeros((n_samples, 2))

    # run PaCMAP
    k = min(target_k, max(1, n_samples - 1))
    try:
        import pacmap
        emb = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=k,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=random_state
        )
        Y = emb.fit_transform(X, init=init)  # "random" or "pca"
        return Y
    except Exception as e:
        # PaCMAP failed, then PCA
        try:
            from sklearn.decomposition import PCA
            Y = PCA(n_components=2).fit_transform(X)
            return Y
        except Exception:
            return np.zeros((n_samples, 2))

def pyumap(fn_in: str, fn_out: str, init="random"):
    fn_in = Path(fn_in).resolve()
    fn_out = Path(fn_out).resolve()
    safe_make_parent(fn_out)

    X = load_X(fn_in)
    Y = compute_embedding(X, target_k=60, init=init, random_state=1986)

    savemat(fn_out, {"X_pacmap": Y})

if __name__ == "__main__":
    x = sys.argv[1]
    y = sys.argv[2]
    init = sys.argv[3] if len(sys.argv) > 3 else "random"  # or "pca"

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_WARNINGS", "0")

    pyumap(x, y, init=init)
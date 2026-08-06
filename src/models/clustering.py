from scipy import sparse
from sklearn.cluster import KMeans


def fit_kmeans(X, k: int, random_state: int = 42) -> KMeans:
    if sparse.issparse(X):
        X_dense = X  # KMeans in sklearn now supports CSR for some versions; else convert
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            km.fit(X_dense)
            return km
        except Exception:
            X_dense = X.toarray()
    else:
        X_dense = X
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    km.fit(X_dense)
    return km

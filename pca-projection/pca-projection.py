import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.array(X)
    n = X.shape[0]
    X_c = X - X.mean(axis=0) 
    C = X_c.T @ X_c * 1 / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]  

    W = eigenvectors
    X_proj = X_c @ W
    return X_proj[:, :k]
    
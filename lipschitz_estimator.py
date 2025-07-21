import numpy as np

def estimate_lipschitz(X, loss_name, norm_type):
    n = X.shape[0]
    
    # Compute norms for rows of X
    row_norm_2 = np.sqrt(np.sum(X ** 2, axis=1))  # L2 norm of each row
    row_norm_1 = np.sum(np.abs(X), axis=1)         # L1 norm of each row
    row_norm_inf = np.max(np.abs(X), axis=1)       # L_inf norm of each row
    
    # Compute operator norms for X^T X
    X_norm_2 = np.linalg.norm(X, ord=2) ** 2       # Spectral norm squared
    X_norm_1 = np.max(np.abs(X.T @ X))  # max |a_ij| over all entries 
    X_norm_inf = np.max(np.sum(np.abs(X.T @ X), axis=1)) # Max row sum for L_inf operator norm
    
    # Cross norms for row-wise products in L1 and L_inf cases
    cross_norm_1_inf = row_norm_1 * row_norm_inf
    cross_norm_inf_1 = row_norm_inf * row_norm_1
    
    # Convert norm_type to string for consistent checking
    norm_str = str(norm_type) if norm_type is not np.inf else "inf"
    
    if loss_name == "squared":
        if norm_str == "1":
            L = X_norm_inf / n  # Corrected: max row sum for L_1 norm
        elif norm_str == "2":
            L = X_norm_2 / n    # Spectral norm squared / n for L_2 norm
        elif norm_str == "inf":
            L = X_norm_1 / n    # Corrected: max column sum for L_inf norm
    elif loss_name == "logsumexp":
        if norm_str == "1":
            L = np.sum(cross_norm_1_inf) / (4 * n)
        elif norm_str == "2":
            L = np.sum(row_norm_2 ** 2) / (4 * n)
        elif norm_str == "inf":
            L = np.sum(cross_norm_inf_1) / (4 * n)
    elif loss_name == "log1px2":
        if norm_str == "1":
            L = 2 * np.sum(cross_norm_1_inf) / n
        elif norm_str == "2":
            L = 2 * np.sum(row_norm_2 ** 2) / n
        elif norm_str == "inf":
            L = 2 * np.sum(cross_norm_inf_1) / n
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    # Validate L
    if L <= 0 or np.isinf(L) or np.isnan(L):
        raise ValueError(f"Invalid Lipschitz constant {L} for loss {loss_name}, norm {norm_type}")
    
    return {"L": L}

# Example usage (optional for testing)
if __name__ == "__main__":
    X_test = np.array([[1, 2], [3, 4], [5, 6]])
    for loss in ["squared", "logsumexp", "log1px2"]:
        for norm in [1, 2, np.inf]:
            L_dict = estimate_lipschitz(X_test, loss, norm)
            print(f"{loss}, Norm {norm}: L = {L_dict['L']}")

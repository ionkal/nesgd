import numpy as np

# 1. Squared Loss
def squared_loss(x, X, y):
    preds = X @ x
    return 0.5 * np.mean((preds - y)**2)

def squared_grad(x, X, y):
    preds = X @ x
    return (X.T @ (preds - y)) / len(y)


# 2. Log-Sum-Exp Loss (Regression variant)
#     f_i(x) = log(1 + exp(X_i x - y_i))
# This is a smooth convex surrogate for absolute error.
def logsumexp_loss(x, X, y):
    z = X @ x - y
    # Numerically stable computation
    return np.mean(np.log1p(np.exp(z)))  # log(1 + exp(z))

def logsumexp_grad(x, X, y):
    z = X @ x - y
    sigmoid = 1 / (1 + np.exp(-z))  # Derivative of log(1 + exp(z))
    return (X.T @ sigmoid) / len(y)


# 3. Log(1 + x^2) Loss
#     f_i(x) = log(1 + (X_i x - y_i)^2)
def log1px2_loss(x, X, y):
    z = X @ x - y
    return np.mean(np.log(1 + z**2))

def log1px2_grad(x, X, y):
    z = X @ x - y
    return (X.T @ (2 * z / (1 + z**2))) / len(y)


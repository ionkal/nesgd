import numpy as np
import matplotlib.pyplot as plt
# generate random linear function with some normal noise
def generate_data(n_samples=1000, d=10, noise=0.01, seed=0):
    np.random.seed(seed)
    X = np.random.uniform(-1, 1, size = (n_samples, d))  # normalised for simplicity
    true_w = np.random.uniform(-1,1, size = d) # normalised for simplicity 
    y = X @ true_w + noise * np.random.randn(n_samples)
    return X, y, true_w


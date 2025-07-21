import numpy as np

def estimate_parameters(X, y, grad_fn, lmo_fn, dual_norm_fn, norm_type, L, batch_size=32, num_samples=100):
    """Estimate sigma_3^2 and theta by sampling x uniformly within ranges derived from y/X_i."""
    n, d = X.shape

    # Determine range for each x_i using all rows
    x_samplers = []
    for i in range(d):
        X_i = X[:, i]
        zero_mask = X_i == 0
        if np.all(zero_mask):
            x_range = [-1, 1]  # Default range if all zeros
        else:
            valid_mask = ~zero_mask
            y_over_X_i = y[valid_mask] / X_i[valid_mask]  # Avoid division by zero
            triples = np.stack([y[valid_mask], X_i[valid_mask],y_over_X_i ], axis = 1)
            sorted_triples = triples[np.argsort(np.abs(triples[:, 2]))][-20:]

            print(f"{'y':>10} {'X_i':>10} {'y/X_i':>10}")
            for row in sorted_triples:
                print(f"{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}")
            loc = np.median(y_over_X_i)
            scale = np.mean(np.abs(y_over_X_i-loc))
            sampler = lambda loc=loc, scale=scale: np.random.laplace(loc, scale)
            x_samplers.append(sampler)

    # Sample all x values upfront
    if len(x_samplers) == 0:
        raise ValueError("x_samplers is empty, check input data")
    x_samples = [np.array([sampler() for sampler in x_samplers]) for _ in range(num_samples)]
    print(f"x_samples: {x_samples}")
    # Initialize lists to store expectations
    deviations = []
    ratios = []

    for x in x_samples:
        full_grad = grad_fn(x, X, y)
        full_grad_dual_norm = dual_norm_fn(full_grad, norm_type)

        if full_grad_dual_norm == 0:
            continue  # Skip if gradient is zero

        d = lmo_fn(full_grad, norm_type)

        # Estimate E[||g(x)||*^2] for sigma_3^2
        g_norm_sq_sum = 0
        for _ in range(30):  # Inner samples for expectation
            idx = np.random.choice(n, size=batch_size, replace=False)
            X_batch, y_batch = X[idx], y[idx]
            g = grad_fn(x, X_batch, y_batch)
            g_norm_sq = dual_norm_fn(g, norm_type) ** 2
            g_norm_sq_sum += g_norm_sq
        expected_g_norm_sq = g_norm_sq_sum / 30
        deviation = expected_g_norm_sq - full_grad_dual_norm **2
        deviations.append(deviation if deviation > 0 else 0)

        # Estimate E[<delta f(x), delta phi*(x)>] for theta
        dot_product_sum = 0
        for _ in range(30):  # Inner samples for expectation
            idx = np.random.choice(n, size=batch_size, replace=False)
            X_batch, y_batch = X[idx], y[idx]
            g = grad_fn(x, X_batch, y_batch)
            dphi_g = - dual_norm_fn(g, norm_type) * lmo_fn(g, norm_type)
            dot_product = np.dot(full_grad, dphi_g)
            dot_product_sum += dot_product
        expected_dot_product = dot_product_sum / 30
        ratio = abs(expected_dot_product) / (full_grad_dual_norm **2)
        ratios.append(ratio)

    print(full_grad)
    print(full_grad_dual_norm**2)
    print(sorted(deviations))
    # Take supremum and infimum over all samples
    sigma3_sq = np.quantile(deviations, 0.95) * 1.5  # 95th percentile for upper bound
    sigma3_sq = min(sigma3_sq, 1e9)                  # still clamp maximum

    theta = np.quantile(ratios, 0.05) * 0.5          # 5th percentile for lower bound
    theta = max(theta, 1e-6)                          # clamp minimum

    print(f"Estimated sigma_3^2 supremum (adjusted): {sigma3_sq}")
    print(f"Computed fixed theta (infimum ratio, adjusted): {theta}")
    return sigma3_sq, theta



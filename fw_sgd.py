import numpy as np
from norm_calculator import dual_norm, lmo

def fw_sgd_until_convergence(X, y, norm_type, initial_point, L, loss_fn, grad_fn, lmo_fn, dual_norm_fn, sigma3_sq, theta, batch_size=32, max_iters=2000, tol=1e-8, patience=50, neighborhood_tol=0.1):
    x = initial_point.copy()
    history = [x.copy()]
    n = X.shape[0]
    best_loss = float('inf')
    no_improve_count = 0
    
    c = 1.0 
    step_size = c * theta  # First step size
    
    # Track average E[delta f(x_k)] for neighborhood check
    delta_f_sum = 0
    num_visited = 0
    
    for i in range(max_iters):
        idx = np.random.choice(n, size=batch_size, replace=False)
        X_batch, y_batch = X[idx], y[idx]
        g = grad_fn(x, X_batch, y_batch)
        g_norm_dual = dual_norm_fn(g, norm_type)
        d = lmo_fn(g, norm_type)
        
        # Approximate E[delta f(x_k)] using current gradient and direction
        delta_f = grad_fn(x, X, y) 
        delta_f_sum += dual_norm_fn(delta_f, norm_type) ** 2 
        num_visited += 1
        avg_delta_f = delta_f_sum / num_visited
        
        print(f"dual*lmo: {dual_norm_fn(delta_f, norm_type)*(lmo_fn(delta_f, norm_type))}, g_norm_dual*d: {g_norm_dual*d}")

        # Update with current step size
        step = np.clip((g_norm_dual / L) * d, -1.0, 1.0) * step_size
        x_new = x + step
        history.append(x_new.copy())
        

        current_loss = loss_fn(x_new, X, y)
        if np.isnan(current_loss) or np.isinf(current_loss):
            print(f"Invalid loss at iteration {i}: {current_loss}")
            break
        
        # Check neighborhood based on avg_delta_f vs sigma_3^2
        if avg_delta_f <= sigma3_sq * (1 + neighborhood_tol) and i > 10:
            K = i + 1
            step_size = theta / (L * np.sqrt(K))  # Second step size
            print(f"Switched step size to {step_size} at iteration {i} (K={K}, avg_delta_f={avg_delta_f}, sigma3_sq={sigma3_sq})")
        if best_loss - current_loss > tol:
            best_loss = current_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
        
        x = x_new
        print(f"Iter: {i}, Loss: {current_loss}, Step Size: {step_size}, avg_delta_f: {avg_delta_f}")
        print(x_new)
    return x, history

import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_data
from norm_calculator import dual_norm, lmo
from loss_functions import (
    squared_loss, squared_grad,
    logsumexp_loss, logsumexp_grad,
    log1px2_loss, log1px2_grad
)
from lipschitz_estimator import estimate_lipschitz
from fw_sgd import fw_sgd_until_convergence
from stepsize_estimator import estimate_parameters

def compute_histories(X, y, initial_point, loss_functions, norm_types):
    histories = {}
    for loss_fn, grad_fn, loss_name in loss_functions:
        print(f"\nRunning with {loss_name} loss...")
        norm_histories = {}
        for norm_type in norm_types:
            L = estimate_lipschitz(X, loss_name, norm_type)["L"]
            print(f"Norm {norm_type}, Lipschitz constant: {L}")
            sigma3_sq, theta = estimate_parameters(X, y, grad_fn, lmo, dual_norm, norm_type, L)
            f_x0 = loss_fn(initial_point, X, y)
            final_x, history = fw_sgd_until_convergence(
                X, y, norm_type, initial_point, L,
                loss_fn, grad_fn, lmo, dual_norm,
                sigma3_sq, theta
            )
            loss_vals = [loss_fn(x, X, y) for x in history]
            approx_errors = [np.linalg.norm(x - true_w) for x in history]
            norm_histories[norm_type] = {
                "loss": loss_vals,
                "approx_error": approx_errors,
                "history": history,
                "sigma3_sq": sigma3_sq,
                "theta": theta,
                "f_x0": f_x0,
                "L": L
            }
        histories[loss_name] = norm_histories
    return histories
def plot_combined(histories, X, y, loss_functions, norm_types):
    colors = {1: 'blue', 2: 'orange', np.inf: 'green'}

    for loss_fn, grad_fn, loss_name in loss_functions:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), num=f"{loss_name} Full Results")

        for norm_type in norm_types:
            data = histories[loss_name][norm_type]
            history = data["history"]
            k_vals = np.arange(len(history))
            grad_norm_sq_list = [dual_norm(grad_fn(x, X, y), norm_type) ** 2 for x in history]

            axs[0].plot(k_vals, data["loss"], label=f"Norm {norm_type}", color=colors[norm_type])
            axs[1].plot(k_vals, data["approx_error"], label=f"Norm {norm_type}", color=colors[norm_type])

            sigma3_sq = data["sigma3_sq"]
            theta = data["theta"]
            L = data["L"]
            f_x0 = data["f_x0"]
            inf_f = 0

            print(f"[{loss_name} | Norm {norm_type}] L={L:.4g}, f(x₀)={f_x0:.4g}, θ={theta:.4g}, σ₃²={sigma3_sq:.4g}")

            larger_bound = [(2 * L * (f_x0 - inf_f)) / (theta**2 * (k + 1)) + sigma3_sq for k in k_vals]
            smaller_bound = [(2 * L * (f_x0 - inf_f)) / (theta**2 * np.sqrt(k + 1)) + (sigma3_sq / np.sqrt(k + 1)) for k in k_vals]

            switch_idx = next((i for i, val in enumerate(grad_norm_sq_list) if val < sigma3_sq), len(k_vals))
            bound_curve = np.where(k_vals < switch_idx, larger_bound, smaller_bound)

            # Define a visibility threshold to avoid flattening the graph
            threshold = max(np.percentile(grad_norm_sq_list, 95), 10 * np.median(grad_norm_sq_list))
            visible_bound = [b if b < threshold else None for b in bound_curve]

            # Plot main gradient norm squared
            axs[2].plot(k_vals, grad_norm_sq_list, label=f"Norm {norm_type}", color=colors[norm_type], linewidth=2)

            # Plot the visible bound (only if below threshold)
            if any(v is not None for v in visible_bound):
                visible_k = [k for k, v in zip(k_vals, visible_bound) if v is not None]
                visible_vals = [v for v in visible_bound if v is not None]
                axs[2].plot(visible_k, visible_vals, color=colors[norm_type], linestyle='--', alpha=0.4)

        axs[0].set_title(f"{loss_name} Loss over Iterations")
        axs[0].set_ylabel("Loss")
        axs[1].set_title(f"||x_k - true_w|| over Iterations")
        axs[1].set_ylabel("Approximation Error")
        axs[2].set_title(f"Gradient Norm Squared and Bounds for {loss_name} Loss")
        axs[2].set_ylabel(r"$\|\nabla f(x_k)\|_*^2$")
        axs[2].set_xlabel("Iteration k")

        for ax in axs:
            ax.set_xlabel("Iteration")
            ax.legend()

        plt.tight_layout()
        plt.show()

# === Run Experiment ===
loss_functions = [
    (squared_loss, squared_grad, "squared"),
    (logsumexp_loss, logsumexp_grad, "logsumexp"),
    (log1px2_loss, log1px2_grad, "log1px2")
]
norm_types = [1, 2, np.inf]
X, y, true_w = generate_data()
initial_point = np.zeros(X.shape[1])
histories = compute_histories(X, y, initial_point, loss_functions, norm_types)
plot_combined(histories, X, y, loss_functions, norm_types)


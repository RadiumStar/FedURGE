"""
:file: optimizer.py
:date: 2026-08-04 (create date) / 2026-08-04 (last modified date)
:description: Loss functions, gradients, and gradient descent update
:src: [Federated Unlearning with Contractive Unlearning Perturbation]()
"""

import numpy as np


def gradient_descent(w: np.ndarray, grad: np.ndarray, lr: float = 0.01) -> np.ndarray:
    """Performs a single gradient descent step.

    :param w: Current model weights
    :param grad: Aggregated gradient to apply
    :param lr: Learning rate, defaults to 0.01
    :return: Updated model weights
    """
    return w - lr * grad


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = 0.0) -> float:
    """Logistic loss: L(w) = (1/n) Σ log(1 + exp(-y_i w^T x_i)) + (λ/2) ||w||²

    :param w: Model weights
    :param X: Feature matrix
    :param y: Label vector in {+1, -1}
    :param lam: Regularization parameter, 0.0 for convex case, >0 for strong convex case
    :return: Logistic loss value
    """
    z = y * (X @ w)
    log_exp = np.log(1 + np.exp(-z))
    loss_unreg = np.mean(log_exp)
    reg = 0.5 * lam * np.sum(w ** 2)
    return loss_unreg + reg


def grad_logistic(w: np.ndarray, X: np.ndarray, y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Gradient of logistic loss: ∇L(w) = -(1/n) Σ [y_i \sigma(-y_i w^T x_i) x_i] + λ w

    :param w: Model weights
    :param X: Feature matrix
    :param y: Label vector in {+1, -1}
    :param lam: Regularization parameter, 0.0 for convex case, >0 for strong convex case
    :return: Gradient vector
    """
    z = y * (X @ w)
    sigma_negz = 1.0 / (1.0 + np.exp(z))  # \sigma(-z)
    grad_unreg = -(X.T @ (y * sigma_negz)) / X.shape[0]
    grad_reg = lam * w
    return grad_unreg + grad_reg


def evaluate_accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate classification accuracy.

    :param w: Model weights
    :param X: Feature matrix
    :param y: Label vector in {+1, -1}
    :return: Accuracy in [0, 1]
    """
    logits = X @ w
    preds = (logits >= 0).astype(int)  # predict +1 if logit >= 0 else 0

    y_binary = (y + 1) // 2  # convert from {-1, +1} to {0, 1}
    accuracy = np.mean(preds == y_binary)
    return accuracy


def dynamic_lambda(g_est: np.ndarray, grad_u: np.ndarray, delta: float = 0.9, cap: float = 1.0, eps: float = 1e-12) -> float:
    """Compute a data-dependent lambda_i satisfying the contraction condition.

    We want the (error-injected) bias to be a contraction of the target update:
        lambda_i^2 || g_est - grad_u ||^2 <= delta^2 || g_est ||^2,  delta < 1,
    where g_est = grad_r + e_i is the retained gradient augmented with the error.
    Setting delta as a constant and solving for lambda_i gives:
        lambda_i = delta * || g_est || / || g_est - grad_u ||,
    clipped to cap (typically 1) so that lambda stays a valid mixing weight.

    :param g_est: Estimated target gradient grad_r + e_i
    :param grad_u: Gradient on the unlearned data
    :param delta: Contraction constant in (0, 1), defaults to 0.9
    :param cap: Upper bound on lambda (defaults to 1)
    :param eps: Small constant to avoid division by zero, defaults to 1e-12
    :return: The dynamic (clipped) lambda value for this client at the current step
    """
    norm_est = np.linalg.norm(g_est)
    norm_diff = np.linalg.norm(g_est - grad_u)
    if norm_diff < eps:
        # If the two gradients coincide there is no bias to compensate.
        return 0.0
    lam = delta * norm_est / norm_diff
    return float(max(min(lam, cap), 0.0))


def dynamic_lambda_gdiff(d: np.ndarray, grad_u: np.ndarray, delta: float = 0.9, cap: float = 1.0, eps: float = 1e-12) -> float:
    """Variant-1 lambda for the gradient-difference (proposed2) framework.

    In the gradient-difference framework we maintain gr_i and send it. To satisfy
        lambda_i^2 || d - grad_u ||^2 <= delta^2 || d ||^2,
    with d = grad_r - gr_i, we set delta as a constant and solve for lambda_i:
        lambda_i = delta * || d || / || d - grad_u ||,
    clipped to cap (defaults to 1) so that lambda stays a valid mixing weight.

    :param d: Gradient difference d = grad_r - gr_i
    :param grad_u: Gradient on the unlearned data
    :param delta: Contraction constant in (0, 1), defaults to 0.9
    :param cap: Upper bound on lambda (defaults to 1)
    :param eps: Small constant to avoid division by zero, defaults to 1e-12
    :return: The dynamic (clipped) lambda value for this client at the current step
    """
    norm_d = np.linalg.norm(d)
    norm_diff = np.linalg.norm(d - grad_u)
    if norm_diff < eps:
        # If the two terms coincide there is no bias to compensate.
        return 0.0
    lam = delta * norm_d / norm_diff
    return float(max(min(lam, cap), 0.0))


def control_gd_u(d: np.ndarray, grad_u: np.ndarray, lam: float, delta: float = 0.9, eps: float = 1e-12) -> np.ndarray:
    """Variant-2: project grad_u so that the gradient-difference contraction holds.

    With a fixed lambda_i and a fixed delta, we solve
        min || g_u - grad_u ||^2  s.t.  || d - g_u || <= R,
    where d = grad_r - gr_i and R = (delta / lambda_i) || d ||.
    The closed-form solution is
        g_u = d + clip(grad_u - d, R)
            = d + min(1, R / || grad_u - d ||) * (grad_u - d).

    :param d: Gradient difference d = grad_r - gr_i
    :param grad_u: Gradient on the unlearned data
    :param lam: Fixed lambda_i = |D^u| / (|D^r| + |D^u|)
    :param delta: Contraction constant in (0, 1), defaults to 0.9
    :param eps: Small constant to avoid division by zero, defaults to 1e-12
    :return: The projected (clipped) grad_u
    """
    norm_d = np.linalg.norm(d)
    if lam <= eps:
        # if lambda ~ 0 the unlearning term should vanish
        return np.zeros_like(grad_u)
    R = (delta / lam) * norm_d
    diff = grad_u - d
    norm_diff = np.linalg.norm(diff)
    if norm_diff <= eps:
        return d
    scale = min(1.0, R / norm_diff)
    return d + scale * diff


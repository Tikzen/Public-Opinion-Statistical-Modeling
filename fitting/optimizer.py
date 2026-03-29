from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

from .metrics import normalize_series

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except Exception:
    gp_minimize = None
    Real = None
    SKOPT_AVAILABLE = False


LossEvalFunc = Callable[[float, float], np.ndarray]



def _prepare_fit_series(real_series: np.ndarray, pred_series: np.ndarray, use_normalized_fit: bool):
    if use_normalized_fit:
        return normalize_series(real_series), normalize_series(pred_series)
    return np.array(real_series, dtype=float), np.array(pred_series, dtype=float)



def _calc_loss(real_series: np.ndarray, pred_series: np.ndarray, use_normalized_fit: bool) -> float:
    real_for_fit, pred_for_fit = _prepare_fit_series(real_series, pred_series, use_normalized_fit)
    return float(np.sum((real_for_fit - pred_for_fit) ** 2))



def fit_parameters_grid(
    real_series: np.ndarray,
    simulate_func: LossEvalFunc,
    beta_min: float,
    beta_max: float,
    beta_steps: int,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
    use_normalized_fit: bool,
):
    """单阶段网格搜索。"""
    best_beta = None
    best_gamma = None
    best_loss = float("inf")
    best_curve = None

    beta_grid = np.linspace(beta_min, beta_max, beta_steps)
    gamma_grid = np.linspace(gamma_min, gamma_max, gamma_steps)

    total_count = len(beta_grid) * len(gamma_grid)
    progress = st.progress(0)
    status_text = st.empty()
    current_count = 0

    for beta in beta_grid:
        for gamma in gamma_grid:
            current_count += 1
            status_text.text(f"正在拟合：{current_count}/{total_count}")

            pred = np.array(simulate_func(float(beta), float(gamma)), dtype=float)
            loss = _calc_loss(real_series, pred, use_normalized_fit)

            if loss < best_loss:
                best_loss = loss
                best_beta = float(beta)
                best_gamma = float(gamma)
                best_curve = pred.copy()

            progress.progress(current_count / total_count)

    status_text.text("参数拟合完成")
    return best_beta, best_gamma, best_loss, best_curve



def fit_parameters_two_stage(
    real_series: np.ndarray,
    simulate_func: LossEvalFunc,
    beta_min: float,
    beta_max: float,
    beta_steps: int,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
    use_normalized_fit: bool,
):
    """粗搜索 + 局部细搜索。"""
    st.info("当前使用：粗搜索 + 局部细搜索")

    coarse_beta, coarse_gamma, coarse_loss, coarse_curve = fit_parameters_grid(
        real_series=real_series,
        simulate_func=simulate_func,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_steps=beta_steps,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_steps=gamma_steps,
        use_normalized_fit=use_normalized_fit,
    )

    beta_span = (beta_max - beta_min) / max(beta_steps - 1, 1)
    gamma_span = (gamma_max - gamma_min) / max(gamma_steps - 1, 1)

    fine_beta_min = max(0.001, coarse_beta - beta_span)
    fine_beta_max = coarse_beta + beta_span
    fine_gamma_min = max(0.001, coarse_gamma - gamma_span)
    fine_gamma_max = coarse_gamma + gamma_span

    st.info(
        f"局部细搜索范围：β ∈ [{fine_beta_min:.4f}, {fine_beta_max:.4f}]，"
        f"γ ∈ [{fine_gamma_min:.4f}, {fine_gamma_max:.4f}]"
    )

    fine_beta, fine_gamma, fine_loss, fine_curve = fit_parameters_grid(
        real_series=real_series,
        simulate_func=simulate_func,
        beta_min=fine_beta_min,
        beta_max=fine_beta_max,
        beta_steps=max(30, beta_steps),
        gamma_min=fine_gamma_min,
        gamma_max=fine_gamma_max,
        gamma_steps=max(30, gamma_steps),
        use_normalized_fit=use_normalized_fit,
    )

    if fine_loss < coarse_loss:
        return fine_beta, fine_gamma, fine_loss, fine_curve
    return coarse_beta, coarse_gamma, coarse_loss, coarse_curve



def fit_parameters_bayesian(
    real_series: np.ndarray,
    simulate_func: LossEvalFunc,
    beta_min: float,
    beta_max: float,
    gamma_min: float,
    gamma_max: float,
    use_normalized_fit: bool,
    n_calls: int = 30,
    n_initial_points: int = 8,
    random_state: int = 42,
):
    """贝叶斯优化。依赖 scikit-optimize。"""
    if not SKOPT_AVAILABLE:
        raise ImportError(
            "未检测到 scikit-optimize。请先安装：pip install scikit-optimize"
        )

    progress = st.progress(0)
    status_text = st.empty()
    best_holder = {
        "loss": float("inf"),
        "beta": None,
        "gamma": None,
        "curve": None,
        "count": 0,
    }

    def objective(params: Sequence[float]) -> float:
        beta, gamma = float(params[0]), float(params[1])
        pred = np.array(simulate_func(beta, gamma), dtype=float)
        loss = _calc_loss(real_series, pred, use_normalized_fit)

        best_holder["count"] += 1
        status_text.text(f"正在进行贝叶斯优化：{best_holder['count']}/{n_calls}")
        progress.progress(min(best_holder["count"] / max(n_calls, 1), 1.0))

        if loss < best_holder["loss"]:
            best_holder["loss"] = loss
            best_holder["beta"] = beta
            best_holder["gamma"] = gamma
            best_holder["curve"] = pred.copy()

        return loss

    result = gp_minimize(
        func=objective,
        dimensions=[
            Real(beta_min, beta_max, name="beta"),
            Real(gamma_min, gamma_max, name="gamma"),
        ],
        n_calls=max(n_calls, 10),
        n_initial_points=max(min(n_initial_points, n_calls), 1),
        random_state=random_state,
    )

    status_text.text("贝叶斯优化完成")

    best_beta = best_holder["beta"] if best_holder["beta"] is not None else float(result.x[0])
    best_gamma = best_holder["gamma"] if best_holder["gamma"] is not None else float(result.x[1])
    best_loss = best_holder["loss"] if np.isfinite(best_holder["loss"]) else float(result.fun)
    best_curve = best_holder["curve"]

    if best_curve is None:
        best_curve = np.array(simulate_func(best_beta, best_gamma), dtype=float)

    return best_beta, best_gamma, best_loss, best_curve



def optimize_parameters(
    method: str,
    real_series: np.ndarray,
    simulate_func: LossEvalFunc,
    beta_min: float,
    beta_max: float,
    beta_steps: int,
    gamma_min: float,
    gamma_max: float,
    gamma_steps: int,
    use_normalized_fit: bool,
    bayes_n_calls: int = 30,
    bayes_n_initial_points: int = 8,
    random_state: int = 42,
):
    """统一调度参数优化策略。"""
    method = method.lower().strip()

    if method == "grid":
        return fit_parameters_grid(
            real_series=real_series,
            simulate_func=simulate_func,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_steps=beta_steps,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_steps=gamma_steps,
            use_normalized_fit=use_normalized_fit,
        )

    if method == "two_stage":
        return fit_parameters_two_stage(
            real_series=real_series,
            simulate_func=simulate_func,
            beta_min=beta_min,
            beta_max=beta_max,
            beta_steps=beta_steps,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_steps=gamma_steps,
            use_normalized_fit=use_normalized_fit,
        )

    if method == "bayesian":
        return fit_parameters_bayesian(
            real_series=real_series,
            simulate_func=simulate_func,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            use_normalized_fit=use_normalized_fit,
            n_calls=bayes_n_calls,
            n_initial_points=bayes_n_initial_points,
            random_state=random_state,
        )

    raise ValueError(f"不支持的优化方法：{method}")

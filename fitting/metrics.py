import numpy as np
import pandas as pd


def normalize_series(arr: np.ndarray) -> np.ndarray:
    """将序列按最大值归一化到 [0, 1]。"""
    arr = np.array(arr, dtype=float)
    max_val = np.max(arr) if len(arr) > 0 else 0.0
    if max_val <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / max_val



def calc_metrics(real_series: np.ndarray, pred_series: np.ndarray):
    """计算拟合评价指标。"""
    real_series = np.array(real_series, dtype=float)
    pred_series = np.array(pred_series, dtype=float)

    error = real_series - pred_series
    abs_error = np.abs(error)

    sse = float(np.sum(error ** 2))
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_error))

    safe_real = np.where(real_series == 0, np.nan, real_series)
    mape = float(np.nanmean(np.abs(error / safe_real)) * 100) if np.any(~np.isnan(safe_real)) else 0.0
    if np.isnan(mape):
        mape = 0.0

    mean_real = float(np.mean(real_series))
    sst = float(np.sum((real_series - mean_real) ** 2))
    if sst == 0:
        r2 = 1.0 if sse == 0 else 0.0
    else:
        r2 = 1 - sse / sst

    real_peak = float(np.max(real_series)) if len(real_series) > 0 else 0.0
    pred_peak = float(np.max(pred_series)) if len(pred_series) > 0 else 0.0

    real_peak_round = int(np.argmax(real_series) + 1) if len(real_series) > 0 else 0
    pred_peak_round = int(np.argmax(pred_series) + 1) if len(pred_series) > 0 else 0

    peak_error = float(abs(real_peak - pred_peak))
    peak_round_error = int(abs(real_peak_round - pred_peak_round))

    return {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2,
        "RealPeak": real_peak,
        "PredPeak": pred_peak,
        "RealPeakRound": real_peak_round,
        "PredPeakRound": pred_peak_round,
        "PeakError": peak_error,
        "PeakRoundError": peak_round_error,
    }



def make_result_dataframe(real_series: np.ndarray, pred_series: np.ndarray) -> pd.DataFrame:
    """构造拟合结果表。"""
    real_series = np.array(real_series, dtype=float)
    pred_series = np.array(pred_series, dtype=float)
    rounds = np.arange(1, len(real_series) + 1)
    return pd.DataFrame({
        "轮次": rounds,
        "真实值": real_series,
        "拟合值": pred_series,
        "误差": real_series - pred_series,
        "绝对误差": np.abs(real_series - pred_series),
    })

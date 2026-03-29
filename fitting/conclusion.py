def build_fit_conclusion(
    metrics,
    best_beta,
    best_gamma,
    beta_min,
    beta_max,
    gamma_min,
    gamma_max,
    use_normalized_fit,
):
    """根据拟合指标生成结论文本。"""
    texts = []

    r2 = metrics["R2"]
    rmse = metrics["RMSE"]
    peak_error = metrics["PeakError"]
    peak_round_error = metrics["PeakRoundError"]

    if use_normalized_fit:
        texts.append("当前采用归一化拟合，模型重点刻画的是传播趋势形状，而非绝对规模。")
    else:
        texts.append("当前采用原始值拟合，模型同时关注传播趋势与绝对规模。")

    if r2 >= 0.90:
        texts.append("当前拟合优度较高，模型对整体传播趋势的解释能力较强。")
    elif r2 >= 0.70:
        texts.append("当前拟合优度中等，模型能够刻画主要传播趋势，但仍存在一定偏差。")
    else:
        texts.append("当前拟合优度较低，说明模型与输入数据之间仍存在较明显差异。")

    if peak_round_error == 0:
        texts.append("模型对传播峰值出现轮次的刻画较准确。")
    elif peak_round_error <= 2:
        texts.append("模型对传播峰值轮次的判断基本合理，但仍存在少量时序偏差。")
    else:
        texts.append("模型对峰值出现时机的刻画偏差较大。")

    if peak_error <= max(5, metrics["RealPeak"] * 0.1):
        texts.append("模型对峰值规模的拟合较好。")
    elif peak_error <= max(15, metrics["RealPeak"] * 0.25):
        texts.append("模型能够大致拟合峰值规模，但峰值高度仍有一定偏差。")
    else:
        texts.append("模型对峰值规模的拟合偏差较大，峰值高低与真实数据差距明显。")

    if abs(best_beta - beta_min) < 1e-9 or abs(best_beta - beta_max) < 1e-9:
        texts.append("传播率 β 落在搜索边界上，建议扩大或平移 β 搜索区间后重新拟合。")

    if abs(best_gamma - gamma_min) < 1e-9 or abs(best_gamma - gamma_max) < 1e-9:
        texts.append("恢复率 γ 落在搜索边界上，建议扩大或平移 γ 搜索区间后重新拟合。")

    if r2 < 0.7 and peak_round_error <= 1:
        texts.append("当前模型可能已捕捉到传播节奏，但对传播规模的刻画仍不足，建议检查节点规模、网络结构或数据尺度是否匹配。")

    if metrics["PredPeak"] < metrics["RealPeak"] * 0.7:
        texts.append("拟合曲线峰值明显偏低，可能说明传播强度不足，或网络结构限制了扩散速度。")

    if rmse > metrics["RealPeak"] * 0.2:
        texts.append("整体误差相对较大，建议优先使用模型自生成测试数据进行自检，以确认参数估计模块逻辑是否稳定。")

    return "\n".join([f"{i + 1}. {t}" for i, t in enumerate(texts)])

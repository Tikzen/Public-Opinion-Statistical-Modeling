def _clamp(value, low, high):
    return max(low, min(high, value))


def generate_param_suggestion(summary: dict) -> dict:
    """
    根据时间序列摘要，生成参数估计页面可用的建议参数（中文版增强版）

    返回字段包括：
    - beta / gamma 搜索建议
    - 是否建议归一化拟合
    - 节点数、连接边数、传播源类型、随机种子数建议
    - 中文解释文本
    """
    length = int(summary.get("length", 0) or 0)
    peak_index = int(summary.get("peak_index", 0) or 0)
    peak_ratio = float(summary.get("peak_ratio", 1.0) or 1.0)
    trend = summary.get("trend", "") or "整体平稳"
    volatility = summary.get("volatility", "中") or "中"
    start_value = float(summary.get("start_value", 0) or 0)
    end_value = float(summary.get("end_value", 0) or 0)
    max_value = float(summary.get("max_value", 0) or 0)
    mean_value = float(summary.get("mean_value", 0) or 0)

    peak_position = peak_index / length if length > 0 else 0.5

    # =========================
    # 1. beta 搜索建议
    # =========================
    if peak_ratio >= 3.5:
        beta_init = 0.40
        beta_min = 0.15
        beta_max = 0.70
        beta_feature = "传播峰值非常突出，说明扩散爆发性较强"
    elif peak_ratio >= 2.5:
        beta_init = 0.30
        beta_min = 0.10
        beta_max = 0.55
        beta_feature = "传播峰值较明显，说明扩散速度中等偏快"
    elif peak_ratio >= 1.8:
        beta_init = 0.22
        beta_min = 0.08
        beta_max = 0.42
        beta_feature = "传播峰值存在，但整体爆发性不算很强"
    else:
        beta_init = 0.15
        beta_min = 0.05
        beta_max = 0.30
        beta_feature = "传播曲线较平缓，建议从相对保守的传播率区间开始搜索"

    # =========================
    # 2. gamma 搜索建议
    # =========================
    if trend == "先上升后下降":
        if peak_position < 0.30:
            gamma_init = 0.20
            gamma_min = 0.08
            gamma_max = 0.35
            gamma_feature = "峰值出现较早，后续回落较长，恢复率可适当设高"
        elif peak_position < 0.70:
            gamma_init = 0.12
            gamma_min = 0.05
            gamma_max = 0.25
            gamma_feature = "峰值位置居中，建议采用中等恢复率搜索范围"
        else:
            gamma_init = 0.08
            gamma_min = 0.03
            gamma_max = 0.20
            gamma_feature = "峰值出现偏晚，说明传播持续时间较长，恢复率不宜设得过高"
    elif trend == "整体上升":
        gamma_init = 0.08
        gamma_min = 0.02
        gamma_max = 0.18
        gamma_feature = "序列仍偏上升，建议恢复率从较低区间开始搜索"
    elif trend == "整体下降":
        gamma_init = 0.16
        gamma_min = 0.06
        gamma_max = 0.30
        gamma_feature = "序列整体回落，说明抑制或恢复作用较强，恢复率可适当提高"
    else:
        gamma_init = 0.10
        gamma_min = 0.03
        gamma_max = 0.20
        gamma_feature = "整体波动较平缓，恢复率可采用常规中等区间"

    # =========================
    # 3. 是否建议归一化拟合
    # =========================
    use_normalized_fit = bool(peak_ratio >= 2.5 or volatility == "高" or max_value > 5 * max(1.0, mean_value))

    # =========================
    # 4. 网络规模建议
    # =========================
    if length >= 300:
        suggest_num_nodes = 300
    elif length >= 180:
        suggest_num_nodes = 220
    elif length >= 90:
        suggest_num_nodes = 160
    else:
        suggest_num_nodes = 100

    if peak_ratio >= 3.0:
        suggest_num_nodes += 40
    if volatility == "高":
        suggest_num_nodes += 30

    suggest_num_nodes = int(_clamp(suggest_num_nodes, 50, 1000))

    # =========================
    # 5. 节点连接边数建议
    # =========================
    if peak_ratio >= 3.2:
        suggest_attach_edges = 4
    elif peak_ratio >= 2.2:
        suggest_attach_edges = 3
    else:
        suggest_attach_edges = 2

    if volatility == "高":
        suggest_attach_edges += 1

    suggest_attach_edges = int(_clamp(suggest_attach_edges, 1, 10))

    # =========================
    # 6. 初始传播源与随机种子建议
    # =========================
    if peak_ratio >= 3.0 and peak_position < 0.45:
        suggest_source_type = "key"
        source_feature = "峰值较高且出现偏早，可优先尝试关键节点作为初始传播源"
    elif trend == "整体平稳":
        suggest_source_type = "normal"
        source_feature = "曲线较平缓，建议优先使用普通节点作为初始传播源"
    else:
        suggest_source_type = "random"
        source_feature = "当前曲线更适合从随机节点开始建立基准拟合"

    if volatility == "高":
        suggest_seed_count = 7
    elif volatility == "中":
        suggest_seed_count = 5
    else:
        suggest_seed_count = 3

    # =========================
    # 7. 中文结构化说明
    # =========================
    if trend == "先上升后下降":
        trend_feature = "该序列呈现较完整的单峰传播过程，适合进行传播—恢复型参数估计。"
    elif trend == "整体上升":
        trend_feature = "该序列当前仍以上升趋势为主，可能尚未完全进入衰退阶段。"
    elif trend == "整体下降":
        trend_feature = "该序列已表现出明显回落特征，说明传播热度正在衰减。"
    else:
        trend_feature = "该序列整体波动相对平稳，传播特征不算特别尖锐。"

    if volatility == "高":
        volatility_feature = "序列波动性较高，建议增加随机种子平均次数，并优先关注趋势拟合。"
    elif volatility == "中":
        volatility_feature = "序列波动性中等，拟合时既要考虑趋势，也要兼顾峰值位置。"
    else:
        volatility_feature = "序列波动性较低，可采用相对收敛的参数搜索范围。"

    fit_mode_feature = (
        "建议优先使用归一化拟合，以增强对传播趋势形状的匹配能力。"
        if use_normalized_fit
        else
        "建议先尝试原始值拟合，若峰值误差较大，再切换到归一化拟合。"
    )

    cn_advice_text = (
        f"{trend_feature}"
        f"{beta_feature}，因此建议 β 初值设为 {beta_init:.2f}，搜索范围可先放在 [{beta_min:.2f}, {beta_max:.2f}]。"
        f"{gamma_feature}，因此建议 γ 初值设为 {gamma_init:.2f}，搜索范围可先放在 [{gamma_min:.2f}, {gamma_max:.2f}]。"
        f"{volatility_feature}"
        f"在网络结构上，建议代理网络节点数优先尝试 {suggest_num_nodes}，每个新节点连接边数优先尝试 {suggest_attach_edges}。"
        f"初始传播源类型建议先用“{suggest_source_type}”模式，随机种子平均次数建议设为 {suggest_seed_count}。"
        f"{fit_mode_feature}"
    )

    return {
        "beta_init": round(beta_init, 4),
        "beta_min": round(beta_min, 4),
        "beta_max": round(beta_max, 4),
        "gamma_init": round(gamma_init, 4),
        "gamma_min": round(gamma_min, 4),
        "gamma_max": round(gamma_max, 4),
        "use_normalized_fit": use_normalized_fit,
        "suggest_num_nodes": suggest_num_nodes,
        "suggest_attach_edges": suggest_attach_edges,
        "suggest_source_type": suggest_source_type,
        "suggest_seed_count": suggest_seed_count,
        "trend_feature": trend_feature,
        "volatility_feature": volatility_feature,
        "beta_feature": beta_feature,
        "gamma_feature": gamma_feature,
        "source_feature": source_feature,
        "cn_advice_text": cn_advice_text,
        "advice_text": cn_advice_text,
    }

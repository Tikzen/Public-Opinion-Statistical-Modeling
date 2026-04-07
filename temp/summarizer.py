import math


def summarize_time_series(dates, values):
    """
    对时间序列做基础摘要，供大模型分析使用

    参数:
        dates: 日期列表
        values: 数值列表

    返回:
        dict，包含基础统计特征与趋势判断
    """
    if not values:
        raise ValueError("values 不能为空")

    n = len(values)
    max_value = max(values)
    min_value = min(values)
    mean_value = sum(values) / n

    peak_index = values.index(max_value)
    peak_date = dates[peak_index] if dates and peak_index < len(dates) else None

    start_value = values[0]
    end_value = values[-1]

    # 简单方差 / 波动性
    variance = sum((x - mean_value) ** 2 for x in values) / n
    std_value = math.sqrt(variance)

    # 趋势判断（简化版）
    if peak_index != 0 and peak_index != n - 1 and start_value < max_value and end_value < max_value:
        trend = "先上升后下降"
    elif end_value > start_value:
        trend = "整体上升"
    elif end_value < start_value:
        trend = "整体下降"
    else:
        trend = "整体平稳"

    # 峰值强度
    if mean_value != 0:
        peak_ratio = max_value / mean_value
    else:
        peak_ratio = None

    # 粗略波动等级
    if std_value < 0.1 * mean_value:
        volatility = "低"
    elif std_value < 0.3 * mean_value:
        volatility = "中"
    else:
        volatility = "高"

    return {
        "length": n,
        "start_value": start_value,
        "end_value": end_value,
        "max_value": max_value,
        "min_value": min_value,
        "mean_value": round(mean_value, 4),
        "std_value": round(std_value, 4),
        "peak_index": peak_index,
        "peak_date": peak_date,
        "peak_ratio": round(peak_ratio, 4) if peak_ratio is not None else None,
        "trend": trend,
        "volatility": volatility,
    }

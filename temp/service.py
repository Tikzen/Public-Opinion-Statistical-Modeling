from ai_analysis.data_sources.baidu_index_source import BaiduIndexSource
from ai_analysis.summarizer import summarize_time_series
from ai_analysis.llm_client import analyze_with_llm
from ai_analysis.param_advisor import generate_param_suggestion


def run_data_source(file):
    """
    统一数据源入口

    参数:
        file: 文件路径 或 Streamlit上传文件对象

    返回:
        标准数据格式 dict
    """
    source = BaiduIndexSource()

    if hasattr(file, "read"):
        import os
        import tempfile

        suffix = ".xlsx"
        file_name = getattr(file, "name", "")
        if isinstance(file_name, str):
            lower_name = file_name.lower()
            if lower_name.endswith(".csv"):
                suffix = ".csv"
            elif lower_name.endswith(".xls"):
                suffix = ".xls"
            elif lower_name.endswith(".xlsx"):
                suffix = ".xlsx"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            data = source.load_data(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        data = source.load_data(file)

    return data


def run_ai_analysis(file, api_key=None, config_path=None):
    """
    AI分析总入口

    流程：
    1. 读取百度指数文件
    2. 生成时间序列摘要
    3. 基于规则生成参数建议
    4. 调用大模型输出综合分析结果

    参数:
        file: 文件路径 或 Streamlit上传文件对象
        api_key: 前端输入的API Key（当json没配时作为备用）
        config_path: 可选，自定义agent_config.json路径

    返回:
        {
            "raw_data": 原始标准化数据,
            "summary": 时间序列摘要,
            "param_suggestion": 参数建议,
            "llm_result": 大模型分析文本
        }
    """
    raw_data = run_data_source(file)

    dates = raw_data.get("dates", [])
    values = raw_data.get("values", [])

    summary = summarize_time_series(dates, values)
    param_suggestion = generate_param_suggestion(summary)

    llm_input = {
        "keyword": raw_data.get("keyword"),
        "source": raw_data.get("source"),
        "summary": summary,
        "param_suggestion": param_suggestion,
    }

    llm_result = analyze_with_llm(
        llm_input,
        api_key=api_key,
        config_path=config_path
    )

    return {
        "raw_data": raw_data,
        "summary": summary,
        "param_suggestion": param_suggestion,
        "llm_result": llm_result,
    }

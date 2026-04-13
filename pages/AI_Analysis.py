import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from ai_analysis.service import run_data_source, run_ai_analysis
from ai_analysis.summarizer import summarize_time_series

st.set_page_config(page_title="AI Analysis", page_icon="🤖", layout="wide")
st.title("🤖 AI Analysis")

uploaded_file = st.file_uploader("上传百度指数文件", type=["xlsx", "csv"])
api_key = st.text_input("输入 API Key（如未配置json需要填写 支持DeepSeek与硅基流动（千问））", type="password")

if "ai_result" not in st.session_state:
    st.session_state["ai_result"] = None

if "ai_data_preview" not in st.session_state:
    st.session_state["ai_data_preview"] = None

if "ai_summary_preview" not in st.session_state:
    st.session_state["ai_summary_preview"] = None

if uploaded_file:
    st.success("文件已上传")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("① 测试数据读取", use_container_width=True):
            data = run_data_source(uploaded_file)
            st.session_state["ai_data_preview"] = data

    with col2:
        if st.button("② 测试摘要", use_container_width=True):
            data = run_data_source(uploaded_file)
            summary = summarize_time_series(data["dates"], data["values"])
            st.session_state["ai_summary_preview"] = summary

    with col3:
        if st.button("③ 生成 AI 分析", use_container_width=True, type="primary"):
            st.session_state["ai_result"] = run_ai_analysis(uploaded_file, api_key)
            st.success("AI 分析已生成并保存。")

if st.session_state.get("ai_data_preview") is not None:
    data = st.session_state["ai_data_preview"]
    with st.expander("查看数据读取结果", expanded=False):
        st.write("数据长度：", len(data["values"]))
        st.write("关键词：", data["keyword"])
        st.write("前10个日期：", data["dates"][:10])
        st.write("前10个数值：", data["values"][:10])

if st.session_state.get("ai_summary_preview") is not None:
    with st.expander("查看摘要结果", expanded=False):
        st.json(st.session_state["ai_summary_preview"])

if st.session_state.get("ai_result") is not None:
    result = st.session_state["ai_result"]
    suggestion = result["param_suggestion"]

    st.subheader("中文参数建议")
    st.info(suggestion.get("cn_advice_text", "暂无中文建议。"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("建议 β 初值", f"{suggestion.get('beta_init', 0):.4f}")
    c2.metric("建议 γ 初值", f"{suggestion.get('gamma_init', 0):.4f}")
    c3.metric("建议节点数", f"{suggestion.get('suggest_num_nodes', 0)}")
    c4.metric("建议连接边数", f"{suggestion.get('suggest_attach_edges', 0)}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("β 搜索下界", f"{suggestion.get('beta_min', 0):.4f}")
    c6.metric("β 搜索上界", f"{suggestion.get('beta_max', 0):.4f}")
    c7.metric("γ 搜索下界", f"{suggestion.get('gamma_min', 0):.4f}")
    c8.metric("γ 搜索上界", f"{suggestion.get('gamma_max', 0):.4f}")

    st.markdown("### 建议配置")
    st.write(
        f"- 建议初始传播源类型：**{suggestion.get('suggest_source_type', 'random')}**\n"
        f"- 建议随机种子平均次数：**{suggestion.get('suggest_seed_count', 5)}**\n"
        f"- 建议拟合方式：**{'归一化拟合' if suggestion.get('use_normalized_fit', False) else '原始值拟合'}**"
    )

    # 新增：可直接复制到“数据输入”的空格分隔格式
    raw_data = result.get("raw_data", {})
    values = raw_data.get("values", [])
    if values:
        copy_text = " ".join(str(int(v)) if float(v).is_integer() else str(round(float(v), 4)) for v in values)

        st.markdown("### 可复制数据（空格分隔）")
        st.caption("下面这串数据可以直接复制到 Parameter Estimation 页的“手动输入”数据框中。")
        st.text_area(
            "复制下面的数据",
            value=copy_text,
            height=140,
            key="copy_series_text",
        )

    with st.expander("查看结构化参数建议", expanded=False):
        st.json(suggestion)

    st.subheader("AI 分析结果")
    st.write(result["llm_result"])

    with st.expander("查看摘要结果", expanded=False):
        st.json(result["summary"])

    col_sync, col_clear = st.columns(2)

    with col_sync:
        if st.button("④ 同步参数到拟合页面", use_container_width=True):
            st.session_state["fit_beta_init"] = suggestion.get("beta_init")
            st.session_state["fit_beta_min"] = suggestion.get("beta_min")
            st.session_state["fit_beta_max"] = suggestion.get("beta_max")

            st.session_state["fit_gamma_init"] = suggestion.get("gamma_init")
            st.session_state["fit_gamma_min"] = suggestion.get("gamma_min")
            st.session_state["fit_gamma_max"] = suggestion.get("gamma_max")

            st.session_state["fit_use_normalized_fit"] = suggestion.get("use_normalized_fit")

            st.session_state["fit_num_nodes"] = suggestion.get("suggest_num_nodes")
            st.session_state["fit_attach_edges"] = suggestion.get("suggest_attach_edges")
            st.session_state["fit_source_type"] = suggestion.get("suggest_source_type")
            st.session_state["fit_seed_count"] = suggestion.get("suggest_seed_count")

            st.success("参数建议已同步到 Parameter Estimation 页面。")

    with col_clear:
        if st.button("⑤ 清空当前 AI 分析结果", use_container_width=True):
            st.session_state["ai_result"] = None
            st.session_state["ai_data_preview"] = None
            st.session_state["ai_summary_preview"] = None
            st.success("已清空当前分析结果，请重新生成。")

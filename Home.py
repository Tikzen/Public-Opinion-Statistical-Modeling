import streamlit as st

st.set_page_config(
    page_title="Public Opinion Statistical Modeling",
    page_icon="📊",
    layout="wide"
)

lang = st.sidebar.selectbox(
    "Language / 语言",
    ["中文", "English"]
)

is_cn = lang == "中文"

TEXT = {
    "title": "📊 舆情传播统计建模与智能分析平台" if is_cn else "📊 Public Opinion Statistical Modeling",
    "subtitle": "数据驱动 + 传播建模 + AI分析的一体化系统" if is_cn else "Data-driven Modeling + Simulation + AI Analysis Platform",

    "overview": "项目简介" if is_cn else "Project Overview",
    "overview_text": (
        "本系统融合真实数据分析、传播模型、参数估计与AI分析，实现完整建模闭环：数据→AI→拟合→仿真。"
        if is_cn else
        "Integrated platform for data-driven modeling, simulation and AI analysis."
    ),

    "modules": "核心模块" if is_cn else "Core Modules",

    "ai_title": "🤖 AI Analysis",
    "ai_text": "时间序列分析 + 参数建议 + LLM解释",

    "fit_title": "⚙️ Parameter Estimation",
    "fit_text": "β/γ反推 + 网格/两阶段/贝叶斯优化 + 稳定性增强",

    "sim_title": "🌐 Simulation",
    "sim_text": "传播模型 + 网络仿真 + 策略对比",

    "flow": "推荐流程" if is_cn else "Workflow",
    "flow_text": "AI → 参数估计 → 仿真 → 导出",

    "start": "开始使用" if is_cn else "Start",
}

st.title(TEXT["title"])
st.markdown(f"### {TEXT['subtitle']}")

st.divider()

st.subheader(TEXT["overview"])
st.write(TEXT["overview_text"])

st.divider()

st.subheader(TEXT["modules"])

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(TEXT["ai_title"])
    st.write(TEXT["ai_text"])

with c2:
    st.markdown(TEXT["fit_title"])
    st.write(TEXT["fit_text"])

with c3:
    st.markdown(TEXT["sim_title"])
    st.write(TEXT["sim_text"])

st.divider()

st.subheader(TEXT["flow"])
st.info(TEXT["flow_text"])

st.divider()

st.subheader(TEXT["start"])

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/AI_Analysis.py", label="AI Analysis")

with col2:
    st.page_link("pages/Parameter_Estimation.py", label="Parameter Estimation")

with col3:
    st.page_link("pages/Simulator.py", label="Simulation")

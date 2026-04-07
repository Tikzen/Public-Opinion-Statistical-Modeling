import os
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(PROJECT_ROOT, "pages")


def find_page(candidates):
    for name in candidates:
        full_path = os.path.join(PAGES_DIR, name)
        if os.path.exists(full_path):
            return f"pages/{name}"
    return None


AI_PAGE = find_page(["AI_Analysis.py", "4_AI_Analysis.py"])
FIT_PAGE = find_page(["3_Parameter_Estimation.py", "Parameter_Estimation.py", "3_参数估计.py"])
SIM_PAGE = find_page(["Simulator.py", "simulator.py", "2_Simulator.py"])


def page_button(target, label, icon="🚀", use_container_width=True):
    if target:
        st.page_link(target, label=label, icon=icon, use_container_width=use_container_width)
    else:
        st.button(f"{icon} {label}", use_container_width=use_container_width, disabled=True)


TEXT = {
    "title": "📊 Public Opinion Statistical Modeling",
    "subtitle": "舆情传播统计建模与智能分析平台" if is_cn else "Public Opinion Propagation Modeling & Intelligent Analysis Platform",
    "tagline": (
        "真实数据驱动 + 参数估计 + 复杂网络传播仿真 + AI辅助分析"
        if is_cn else
        "Data-driven analysis + Parameter estimation + Complex-network simulation + AI-assisted interpretation"
    ),
    "abstract_title": "项目概述" if is_cn else "Project Overview",
    "abstract_text": (
        "本系统围绕舆情传播问题，构建了一个从真实数据分析到参数反推、再到传播仿真与策略验证的完整闭环平台。"
        "项目以百度指数等时间序列数据为输入，通过 AI Analysis 模块进行趋势识别与参数建议，"
        "再在 Parameter Estimation 模块中估计传播率 β 与恢复率 γ，最后在 Simulation 模块中将参数回填到复杂网络传播模型中，"
        "观察不同治理策略下的扩散过程与峰值变化。"
        if is_cn else
        "This platform builds a complete closed-loop workflow from real-world time series data to parameter fitting, network simulation, and strategy evaluation."
    ),
    "workflow_title": "推荐使用流程" if is_cn else "Recommended Workflow",
    "workflow_steps": [
        ("① 获取数据", "导出百度指数或准备舆情时间序列", "Prepare Baidu Index or time-series data"),
        ("② AI Analysis", "分析趋势、峰值、波动性并生成中文参数建议", "Analyze trends, peaks, volatility, and get AI suggestions"),
        ("③ Parameter Estimation", "估计 β / γ 并验证拟合质量", "Estimate β / γ and evaluate fitting quality"),
        ("④ Simulation", "在复杂网络中进行传播仿真与策略对比", "Run diffusion simulation and compare interventions"),
    ] if is_cn else [
        ("① Data Input", "Prepare Baidu Index or time-series data", ""),
        ("② AI Analysis", "Analyze trends, peaks, volatility, and get AI suggestions", ""),
        ("③ Parameter Estimation", "Estimate β / γ and evaluate fitting quality", ""),
        ("④ Simulation", "Run diffusion simulation and compare interventions", ""),
    ],
    "module_title": "三大核心模块" if is_cn else "Three Core Modules",
    "modules": [
        {
            "name": "AI Analysis",
            "icon": "🤖",
            "desc": "面向真实数据进行时间序列摘要、中文参数建议与大模型解释，并可一键同步参数到拟合页面。"
            if is_cn else
            "Time-series analysis, Chinese parameter suggestions, and LLM interpretation.",
            "highlights": [
                "支持百度指数文件导入",
                "自动提取峰值、趋势、波动性",
                "支持 DeepSeek / 硅基流动回退"
            ] if is_cn else [
                "Baidu Index import",
                "Peak/trend/volatility extraction",
                "DeepSeek with fallback"
            ],
            "target": AI_PAGE,
            "button": "进入 AI Analysis" if is_cn else "Open AI Analysis"
        },
        {
            "name": "Parameter Estimation",
            "icon": "⚙️",
            "desc": "根据真实数据反推传播参数 β / γ，支持原始值拟合、归一化拟合以及多种优化方法。"
            if is_cn else
            "Estimate β / γ from real data with multiple fitting strategies.",
            "highlights": [
                "网格搜索 / 两阶段搜索 / 贝叶斯优化",
                "多随机种子平均提升稳定性",
                "支持 AI 参数建议同步"
            ] if is_cn else [
                "Grid / two-stage / Bayesian optimization",
                "Multi-seed averaging",
                "AI parameter sync"
            ],
            "target": FIT_PAGE,
            "button": "进入 Parameter Estimation" if is_cn else "Open Parameter Estimation"
        },
        {
            "name": "Simulation",
            "icon": "🌐",
            "desc": "在无标度网络中模拟舆情传播过程，比较辟谣机制与关键节点治理等策略效果。"
            if is_cn else
            "Simulate rumor diffusion on scale-free networks and compare interventions.",
            "highlights": [
                "支持 SIR / SI / SIS 模型",
                "支持辟谣机制与关键节点限制",
                "支持网络动态演化可视化"
            ] if is_cn else [
                "SIR / SI / SIS models",
                "Refutation & key-node interventions",
                "Dynamic network visualization"
            ],
            "target": SIM_PAGE,
            "button": "进入 Simulation" if is_cn else "Open Simulation"
        },
    ],
    "highlight_title": "技术亮点" if is_cn else "Highlights",
    "highlights": [
        ("🧠 AI参数推荐", "AI Analysis 模块不是简单文本生成，而是基于时间序列摘要、规则系统与大模型解释共同生成建模建议。"),
        ("📈 参数反推闭环", "系统支持从真实数据出发，先分析、再拟合、再仿真，实现完整建模闭环。"),
        ("⚡ 贝叶斯优化", "基于 scikit-optimize 的高斯过程代理建模，显著减少搜索次数，提高参数估计效率。"),
        ("🕸️ 复杂网络仿真", "采用无标度网络结构模拟社交网络，更适合刻画关键节点在传播中的放大作用。"),
    ] if is_cn else [
        ("🧠 AI Suggestions", "Structured parameter recommendations with LLM interpretation."),
        ("📈 Closed-loop Modeling", "Data → AI → Estimation → Simulation."),
        ("⚡ Bayesian Optimization", "Efficient skopt-based Gaussian Process search."),
        ("🕸️ Complex Networks", "Scale-free network simulation for social diffusion."),
    ],
    "stats_title": "系统能力概览" if is_cn else "Capability Overview",
    "footer_tip": "建议答辩演示顺序：AI Analysis → Parameter Estimation → Simulation"
    if is_cn else
    "Suggested demo order: AI Analysis → Parameter Estimation → Simulation",
}


def render_html_card(title, icon, desc, highlights):
    items = "".join([f"<li style='margin-bottom:6px;color:#1f2937;'>{x}</li>" for x in highlights])
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #eef4ff, #f6f0ff);
            border: 1px solid #d8def0;
            border-radius: 20px;
            padding: 20px 18px 16px 18px;
            min-height: 320px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        ">
            <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
            <div style="font-size: 22px; font-weight: 700; margin-bottom: 10px; color:#111827;">{title}</div>
            <div style="font-size: 15px; line-height: 1.75; margin-bottom: 12px; color:#1f2937;">{desc}</div>
            <ul style="padding-left: 18px; margin-top: 8px;">{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div style="
        padding: 30px 32px 22px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, #dbeafe, #ede9fe);
        border: 1px solid #d8def0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 14px;
    ">
        <div style="font-size: 38px; font-weight: 800; margin-bottom: 10px; color:#111827;">{TEXT["title"]}</div>
        <div style="font-size: 22px; font-weight: 600; margin-bottom: 12px; color:#1f2937;">{TEXT["subtitle"]}</div>
        <div style="font-size: 17px; line-height: 1.8; color:#374151;">{TEXT["tagline"]}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_col1, hero_col2 = st.columns([2.2, 1.2])

with hero_col1:
    st.subheader(TEXT["abstract_title"])
    st.write(TEXT["abstract_text"])

with hero_col2:
    st.subheader(TEXT["stats_title"])
    a, b = st.columns(2)
    a.metric("Pages", "3")
    b.metric("Models", "3+")
    c, d = st.columns(2)
    c.metric("Optimizers", "3")
    d.metric("AI Mode", "LLM")
    st.info(TEXT["footer_tip"])

st.divider()

st.subheader(TEXT["workflow_title"])
w1, w2, w3, w4 = st.columns(4)

for col, step in zip([w1, w2, w3, w4], TEXT["workflow_steps"]):
    title, desc_cn, desc_en = step
    desc = desc_cn if is_cn else (desc_en or desc_cn)
    col.markdown(
        f"""
        <div style="
            background: #f8fafc;
            border: 1px solid #dbe3ef;
            border-radius: 18px;
            padding: 18px 16px;
            min-height: 165px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 18px; font-weight: 700; margin-bottom: 12px; color:#111827;">{title}</div>
            <div style="font-size: 14px; line-height: 1.8; color:#374151;">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.subheader(TEXT["module_title"])
m1, m2, m3 = st.columns(3)

for col, module in zip([m1, m2, m3], TEXT["modules"]):
    with col:
        render_html_card(
            module["name"],
            module["icon"],
            module["desc"],
            module["highlights"],
        )
        page_button(module["target"], module["button"], icon="🚀")

st.divider()

st.subheader(TEXT["highlight_title"])
h1, h2 = st.columns(2)

for idx, (title, desc) in enumerate(TEXT["highlights"]):
    col = h1 if idx % 2 == 0 else h2
    with col:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f8fbff, #faf5ff);
                border-left: 6px solid #6366f1;
                border-radius: 14px;
                padding: 16px 18px;
                margin-bottom: 14px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.04);
            ">
                <div style="font-size: 18px; font-weight: 700; margin-bottom: 8px; color:#111827;">{title}</div>
                <div style="font-size: 14px; line-height: 1.8; color:#374151;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        padding: 10px 8px 0 8px;
        color:#6b7280;
        font-size: 14px;
    ">
        GitHub README、答辩演示与系统页面已形成统一结构，可直接用于比赛展示与项目说明。
    </div>
    """,
    unsafe_allow_html=True,
)

import os
import sys
import random
import importlib
import pkgutil

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# 把项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import model
from network.generator import create_network
from fitting.metrics import calc_metrics, make_result_dataframe, normalize_series
from fitting.conclusion import build_fit_conclusion
from fitting.optimizer import optimize_parameters


# =========================
# Matplotlib 中文显示修复（与 simulator 同风格）
# =========================
def setup_matplotlib_font():
    """
    修复 matplotlib 中文显示问题：
    1. 优先加载项目内置字体文件
    2. 若项目字体不存在，再回退系统中文字体
    3. 最后才使用 DejaVu Sans
    """
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)

    font_candidates = [
        os.path.join(project_root, "assets", "fonts", "NotoSansSC-Regular.ttf"),
        os.path.join(project_root, "assets", "fonts", "NotoSansCJKsc-Regular.otf"),
        os.path.join(project_root, "assets", "fonts", "SourceHanSansCN-Regular.otf"),
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                font_prop = font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()

                mpl.rcParams["font.family"] = "sans-serif"
                mpl.rcParams["font.sans-serif"] = [font_name]
                mpl.rcParams["axes.unicode_minus"] = False
                return font_prop, f"{font_name} (from file)"
            except Exception:
                pass

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]

    available_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [font_name]
            mpl.rcParams["axes.unicode_minus"] = False
            return font_manager.FontProperties(family=font_name), f"{font_name} (system)"

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    return font_manager.FontProperties(family="DejaVu Sans"), "DejaVu Sans (fallback)"


CN_FONT, SELECTED_FONT = setup_matplotlib_font()


# =========================
# 模型加载（与 simulator 同逻辑）
# =========================
def load_models():
    models = {}

    for _, module_name, _ in pkgutil.iter_modules(model.__path__):
        if module_name.startswith("__"):
            continue

        module = importlib.import_module(f"model.{module_name}")

        # 新规范模型
        if hasattr(module, "MODEL_NAME") and hasattr(module, "STATES") and hasattr(module, "step"):
            models[module.MODEL_NAME] = {
                "step": module.step,
                "states": module.STATES,
                "module_name": module_name,
            }
            continue

        # 兼容旧规范
        if module_name == "si_model" and hasattr(module, "si_step"):
            models["SI"] = {
                "step": module.si_step,
                "states": ["S", "I"],
                "module_name": module_name,
            }
        elif module_name == "sis_model" and hasattr(module, "sis_step"):
            models["SIS"] = {
                "step": module.sis_step,
                "states": ["S", "I"],
                "module_name": module_name,
            }
        elif module_name == "sir_model" and hasattr(module, "sir_step"):
            models["SIR"] = {
                "step": module.sir_step,
                "states": ["S", "I", "R"],
                "module_name": module_name,
            }

    return models


MODEL_REGISTRY = load_models()


# =========================
# 辅助函数
# =========================
def parse_series_from_text(text: str) -> np.ndarray:
    if not text.strip():
        return np.array([])

    text = text.replace("\n", ",").replace(" ", ",").replace("，", ",")
    parts = [x.strip() for x in text.split(",") if x.strip()]
    values = [float(x) for x in parts]
    return np.array(values, dtype=float)



def get_blocked_nodes(G, block_ratio):
    import networkx as nx
    centrality = nx.degree_centrality(G)
    num_blocked = max(1, int(len(G.nodes()) * block_ratio))
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    return set(sorted_nodes[:num_blocked])



def choose_initial_node(G, blocked_nodes, source_type):
    import networkx as nx

    all_nodes = list(G.nodes())
    centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)

    source_key_nodes = sorted_nodes[:max(1, int(len(all_nodes) * 0.05))]

    normal_nodes = [
        node for node in all_nodes
        if node not in source_key_nodes and node not in blocked_nodes
    ]

    available_key_nodes = [node for node in source_key_nodes if node not in blocked_nodes]
    available_random_nodes = [node for node in all_nodes if node not in blocked_nodes]

    if source_type == "key":
        candidate_nodes = available_key_nodes or source_key_nodes or available_random_nodes or all_nodes
    elif source_type == "normal":
        candidate_nodes = normal_nodes or available_random_nodes or all_nodes
    else:
        candidate_nodes = available_random_nodes or all_nodes

    return random.choice(candidate_nodes)



def run_simulation_for_fit(
    model_type,
    num_nodes,
    attach_edges,
    infection_prob,
    recovery_prob,
    rounds,
    seed,
    source_type,
    enable_refutation=False,
    refutation_round=10,
    refutation_factor=0.5,
    enable_key_control=False,
    key_control_ratio=0.05,
):
    random.seed(seed)

    # 保险保护，避免 attach_edges >= num_nodes
    attach_edges = min(attach_edges, max(1, num_nodes - 1))

    G = create_network(num_nodes, attach_edges)

    blocked_nodes = set()
    if enable_key_control:
        blocked_nodes = get_blocked_nodes(G, key_control_ratio)

    if source_type == "key" and enable_key_control:
        enable_key_control = False
        blocked_nodes = set()

    model_info = MODEL_REGISTRY[model_type]
    step_func = model_info["step"]
    states_list = model_info["states"]

    state = {node: "S" for node in G.nodes()}
    initial_node = choose_initial_node(G, blocked_nodes, source_type)
    blocked_nodes.discard(initial_node)
    state[initial_node] = "I"

    history = [state.copy()]
    count_history = {s: [] for s in states_list}

    for step_idx in range(rounds):
        current_infection_prob = (
            infection_prob * refutation_factor
            if enable_refutation and step_idx + 1 >= refutation_round
            else infection_prob
        )

        state = step_func(
            G,
            state,
            current_infection_prob,
            recovery_prob,
            blocked_nodes=blocked_nodes,
        )

        history.append(state.copy())

        counts = {s: 0 for s in states_list}
        for node_state in state.values():
            if node_state in counts:
                counts[node_state] += 1

        for s in states_list:
            count_history[s].append(counts[s])

        s_count = counts.get("S", 0)
        i_count = counts.get("I", 0)

        if model_type == "SI" and s_count == 0:
            break

        if model_type in ["SIS", "SIR"] and i_count == 0:
            break

    return {
        "G": G,
        "history": history,
        "count_history": count_history,
        "states_list": states_list,
        "initial_node": initial_node,
        "blocked_nodes": blocked_nodes,
    }



def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr.astype(float)

    if len(arr) == 0:
        return np.zeros(target_len, dtype=float)

    if len(arr) > target_len:
        return arr[:target_len].astype(float)

    pad_value = arr[-1]
    pad_array = np.full(target_len - len(arr), pad_value, dtype=float)
    return np.concatenate([arr.astype(float), pad_array])



def simulate_avg_curve(
    model_type,
    target_state,
    beta,
    gamma,
    rounds,
    num_nodes,
    attach_edges,
    source_type,
    seeds,
):
    curves = []

    for seed in seeds:
        result = run_simulation_for_fit(
            model_type=model_type,
            num_nodes=num_nodes,
            attach_edges=attach_edges,
            infection_prob=beta,
            recovery_prob=gamma,
            rounds=rounds,
            seed=seed,
            source_type=source_type,
            enable_refutation=False,
            enable_key_control=False,
        )

        state_series = np.array(result["count_history"].get(target_state, []), dtype=float)
        state_series = pad_or_truncate(state_series, rounds)
        curves.append(state_series)

    if not curves:
        return np.zeros(rounds, dtype=float)

    return np.mean(np.vstack(curves), axis=0)



def generate_self_test_data(
    model_type,
    target_state,
    beta,
    gamma,
    rounds,
    num_nodes,
    attach_edges,
    source_type,
    seed,
):
    result = run_simulation_for_fit(
        model_type=model_type,
        num_nodes=num_nodes,
        attach_edges=attach_edges,
        infection_prob=beta,
        recovery_prob=gamma,
        rounds=rounds,
        seed=seed,
        source_type=source_type,
        enable_refutation=False,
        enable_key_control=False,
    )

    series = np.array(result["count_history"].get(target_state, []), dtype=float)
    return pad_or_truncate(series, rounds)


# =========================
# 页面配置
# =========================
st.set_page_config(page_title="参数估计", page_icon="📈", layout="wide")

st.title("📈 参数估计（接入模型系统）")
st.markdown(
    """
本模块根据真实传播时间序列，反向估计传播模型参数：

- **传播率 β**
- **恢复率 γ**

特点：
- 与模拟页面共用模型注册系统
- 支持 **SIR / SI / SIS / SEIR / 自定义模型**
- 支持选择拟合目标状态（默认优先 I）
- 采用 **多随机种子平均** 降低仿真波动
- 支持 **自检模式**
- 支持 **原始值 / 归一化** 双重对比
- 默认采用 **归一化拟合**
- 支持 **网格搜索 / 两阶段搜索 / 贝叶斯优化**
"""
)

st.caption(f"当前图表字体：{SELECTED_FONT}")


# =========================
# 侧边栏参数
# =========================
st.sidebar.header("模型与网络设置")

available_models = list(MODEL_REGISTRY.keys())
default_model_index = available_models.index("SIR") if "SIR" in available_models else 0

model_type = st.sidebar.selectbox(
    "传播模型",
    available_models,
    index=default_model_index,
)

states_list = MODEL_REGISTRY[model_type]["states"]
default_target_state = "I" if "I" in states_list else states_list[0]

target_state = st.sidebar.selectbox(
    "拟合目标状态",
    states_list,
    index=states_list.index(default_target_state),
    help="通常建议拟合 I；若模型没有 I，可改选其他状态。"
)

source_labels = {
    "random": "随机节点",
    "normal": "普通节点",
    "key": "关键节点",
}
source_keys = ["random", "normal", "key"]

source_type = st.sidebar.selectbox(
    "初始传播源类型",
    source_keys,
    format_func=lambda x: source_labels[x],
)

num_nodes = st.sidebar.slider("代理网络节点数", 50, 1000, 100, 10)
if num_nodes >= 500:
    st.sidebar.warning("当前节点规模较大，拟合时间会明显增加")

attach_edges = st.sidebar.slider("每个新节点连接边数", 1, 10, 3, 1)

st.sidebar.header("搜索范围设置")
beta_min = st.sidebar.slider("β 最小值", 0.01, 1.50, 0.05, 0.01)
beta_max = st.sidebar.slider("β 最大值", 0.05, 2.00, 1.00, 0.01)
beta_steps = st.sidebar.slider("β 搜索步数", 5, 60, 20, 1)

gamma_min = st.sidebar.slider("γ 最小值", 0.00, 1.50, 0.01, 0.01)
gamma_max = st.sidebar.slider("γ 最大值", 0.01, 2.00, 0.80, 0.01)
gamma_steps = st.sidebar.slider("γ 搜索步数", 5, 60, 20, 1)

st.sidebar.header("优化方法设置")
method_label_to_key = {
    "网格搜索": "grid",
    "两阶段搜索（推荐）": "two_stage",
    "贝叶斯优化": "bayesian",
}
optimization_method_label = st.sidebar.selectbox(
    "参数优化方法",
    list(method_label_to_key.keys()),
    index=1,
)
optimization_method = method_label_to_key[optimization_method_label]

bayes_n_calls = 30
bayes_n_initial_points = 8
if optimization_method == "bayesian":
    bayes_n_calls = st.sidebar.slider("贝叶斯优化总迭代次数", 10, 100, 30, 1)
    bayes_n_initial_points = st.sidebar.slider("贝叶斯优化初始探索次数", 3, 30, 8, 1)
    st.sidebar.caption("如未安装 scikit-optimize，请先执行：pip install scikit-optimize")

st.sidebar.header("稳定性设置")
seed_count = st.sidebar.slider("平均随机种子数量", 1, 10, 5, 1)
base_seed = st.sidebar.number_input("基础随机种子", value=42, step=1)

use_normalized_fit = st.sidebar.checkbox("使用归一化拟合（推荐）", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "建议：\n"
    "- 先用自检模式确认拟合逻辑是否正常\n"
    "- 再导入外部数据进行正式拟合\n"
    "- 若参数落在边界，需调整搜索范围\n"
    "- 归一化拟合更适合真实大规模数据的趋势匹配"
)


# =========================
# 自检模式
# =========================
st.subheader("🧪 自检模式（推荐先运行）")
st.markdown("先用当前模型自动生成测试数据，再反向拟合，看能否找回接近的参数。")

with st.expander("展开自检设置", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    self_beta = col_a.number_input("自检 β", min_value=0.01, max_value=2.0, value=0.30, step=0.01)
    self_gamma = col_b.number_input("自检 γ", min_value=0.00, max_value=2.0, value=0.10, step=0.01)
    self_rounds = col_c.number_input("自检轮数", min_value=5, max_value=100, value=16, step=1)
    self_seed = col_d.number_input("自检随机种子", min_value=0, max_value=999999, value=42, step=1)

    if st.button("生成自检测试数据"):
        self_series = generate_self_test_data(
            model_type=model_type,
            target_state=target_state,
            beta=self_beta,
            gamma=self_gamma,
            rounds=int(self_rounds),
            num_nodes=num_nodes,
            attach_edges=attach_edges,
            source_type=source_type,
            seed=int(self_seed),
        )

        st.session_state["pe_self_test_series"] = self_series
        st.session_state["pe_self_test_beta"] = self_beta
        st.session_state["pe_self_test_gamma"] = self_gamma
        st.session_state["pe_self_test_rounds"] = int(self_rounds)

        st.success("已生成自检测试数据，并写入当前页面。")

if "pe_self_test_series" in st.session_state:
    st.info(
        f"当前已载入自检数据：β={st.session_state['pe_self_test_beta']:.2f}，"
        f"γ={st.session_state['pe_self_test_gamma']:.2f}，"
        f"长度={st.session_state['pe_self_test_rounds']}"
    )


# =========================
# 数据输入
# =========================
st.subheader("📥 数据输入")

tab1, tab2, tab3 = st.tabs(["手动输入", "上传 CSV", "使用自检数据"])

real_series = np.array([])

with tab1:
    default_text = "1,3,8,20,45,80,120,150,130,95,60,35,18,8,3,1"
    text_data = st.text_area(
        "输入真实传播时间序列（支持逗号、空格、换行）",
        value=default_text,
        height=140,
    )

    try:
        manual_series = parse_series_from_text(text_data)
        if len(manual_series) > 0:
            real_series = manual_series
            st.success(f"已读取 {len(real_series)} 个数据点。")
    except Exception as e:
        st.error(f"数据解析失败：{e}")

with tab2:
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write("数据预览：")
            st.dataframe(df_upload.head(), use_container_width=True)

            column_name = st.selectbox("选择拟合列", df_upload.columns)
            csv_series = df_upload[column_name].dropna().astype(float).to_numpy()
            real_series = csv_series
            st.success(f"已从列 {column_name} 读取 {len(real_series)} 个数据点。")
        except Exception as e:
            st.error(f"CSV 读取失败：{e}")

with tab3:
    if "pe_self_test_series" in st.session_state:
        self_series = st.session_state["pe_self_test_series"]
        st.write("当前自检数据：")
        st.dataframe(
            pd.DataFrame({
                "轮次": np.arange(1, len(self_series) + 1),
                "自检数据": self_series
            }),
            use_container_width=True
        )
        if st.button("使用这组自检数据进行拟合"):
            real_series = np.array(self_series, dtype=float)
            st.session_state["pe_force_use_self_series"] = real_series
            st.success("已切换为自检数据。")
    else:
        st.warning("还没有生成自检数据，请先在上方生成。")

if "pe_force_use_self_series" in st.session_state:
    real_series = st.session_state["pe_force_use_self_series"]


# =========================
# 原始数据展示
# =========================
if len(real_series) > 0:
    st.subheader("📊 原始数据")
    df_real = pd.DataFrame({
        "轮次": np.arange(1, len(real_series) + 1),
        f"{target_state}状态真实值": real_series
    })
    st.dataframe(df_real, use_container_width=True)

    fig_real, ax_real = plt.subplots(figsize=(10, 4))
    ax_real.plot(df_real["轮次"], df_real[f"{target_state}状态真实值"], marker="o")
    ax_real.set_title(f"{target_state} 状态真实曲线", fontproperties=CN_FONT)
    ax_real.set_xlabel("轮次", fontproperties=CN_FONT)
    ax_real.set_ylabel("节点数量", fontproperties=CN_FONT)
    ax_real.grid(True, alpha=0.3)
    st.pyplot(fig_real)
    plt.close(fig_real)


# =========================
# 参数拟合
# =========================
if len(real_series) > 0:
    if beta_min >= beta_max:
        st.error("β 最小值必须小于 β 最大值。")
    elif gamma_min > gamma_max:
        st.error("γ 最小值不能大于 γ 最大值。")
    else:
        if st.button("开始参数估计", type="primary"):
            seeds = list(range(int(base_seed), int(base_seed) + int(seed_count)))

            def simulate_func(beta, gamma):
                return simulate_avg_curve(
                    model_type=model_type,
                    target_state=target_state,
                    beta=beta,
                    gamma=gamma,
                    rounds=len(real_series),
                    num_nodes=num_nodes,
                    attach_edges=attach_edges,
                    source_type=source_type,
                    seeds=seeds,
                )

            with st.spinner("正在进行参数估计，请稍候..."):
                try:
                    best_beta, best_gamma, best_loss, pred_series = optimize_parameters(
                        method=optimization_method,
                        real_series=real_series,
                        simulate_func=simulate_func,
                        beta_min=beta_min,
                        beta_max=beta_max,
                        beta_steps=beta_steps,
                        gamma_min=gamma_min,
                        gamma_max=gamma_max,
                        gamma_steps=gamma_steps,
                        use_normalized_fit=use_normalized_fit,
                        bayes_n_calls=bayes_n_calls,
                        bayes_n_initial_points=bayes_n_initial_points,
                        random_state=int(base_seed),
                    )
                except ImportError as e:
                    st.error(str(e))
                    st.stop()

                metrics = calc_metrics(real_series, pred_series)
                result_df = make_result_dataframe(real_series, pred_series)
                conclusion_text = build_fit_conclusion(
                    metrics=metrics,
                    best_beta=best_beta,
                    best_gamma=best_gamma,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    gamma_min=gamma_min,
                    gamma_max=gamma_max,
                    use_normalized_fit=use_normalized_fit,
                )

                st.session_state["pe_best_beta"] = best_beta
                st.session_state["pe_best_gamma"] = best_gamma
                st.session_state["pe_best_loss"] = best_loss
                st.session_state["pe_pred_series"] = pred_series
                st.session_state["pe_result_df"] = result_df
                st.session_state["pe_metrics"] = metrics
                st.session_state["pe_model_type"] = model_type
                st.session_state["pe_target_state"] = target_state
                st.session_state["pe_num_nodes"] = num_nodes
                st.session_state["pe_attach_edges"] = attach_edges
                st.session_state["pe_source_type"] = source_type
                st.session_state["pe_seed_list"] = seeds
                st.session_state["pe_conclusion_text"] = conclusion_text
                st.session_state["pe_use_normalized_fit"] = use_normalized_fit
                st.session_state["pe_optimization_method_label"] = optimization_method_label
                st.session_state["pe_optimization_method_key"] = optimization_method


# =========================
# 结果展示
# =========================
if "pe_best_beta" in st.session_state:
    st.subheader("📌 参数估计结果")

    metrics = st.session_state["pe_metrics"]
    best_beta = st.session_state["pe_best_beta"]
    best_gamma = st.session_state["pe_best_gamma"]
    result_df = st.session_state["pe_result_df"]
    fit_mode_text = "归一化拟合" if st.session_state.get("pe_use_normalized_fit", True) else "原始值拟合"
    optimization_method_text = st.session_state.get("pe_optimization_method_label", "两阶段搜索（推荐）")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("传播率 β", f"{best_beta:.4f}")
    c2.metric("恢复率 γ", f"{best_gamma:.4f}")
    c3.metric("SSE", f"{metrics['SSE']:.2f}")
    c4.metric("RMSE", f"{metrics['RMSE']:.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("MSE", f"{metrics['MSE']:.4f}")
    c6.metric("MAE", f"{metrics['MAE']:.4f}")
    c7.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    c8.metric("R²", f"{metrics['R2']:.4f}")

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("真实峰值", f"{metrics['RealPeak']:.2f}")
    c10.metric("拟合峰值", f"{metrics['PredPeak']:.2f}")
    c11.metric("峰值误差", f"{metrics['PeakError']:.2f}")
    c12.metric("峰值轮次误差", f"{metrics['PeakRoundError']}")

    st.markdown("### 当前拟合配置")
    st.write(
        f"- 模型：**{st.session_state['pe_model_type']}**\n"
        f"- 拟合状态：**{st.session_state['pe_target_state']}**\n"
        f"- 代理网络节点数：**{st.session_state['pe_num_nodes']}**\n"
        f"- 每个新节点连接边数：**{st.session_state['pe_attach_edges']}**\n"
        f"- 初始传播源类型：**{source_labels[st.session_state['pe_source_type']]}**\n"
        f"- 随机种子：**{st.session_state['pe_seed_list']}**\n"
        f"- 当前拟合模式：**{fit_mode_text}**\n"
        f"- 当前优化方法：**{optimization_method_text}**"
    )

    if abs(best_beta - beta_min) < 1e-9 or abs(best_beta - beta_max) < 1e-9:
        st.warning("传播率 β 落在搜索边界上，建议适当调整 β 搜索范围后重新拟合。")

    if abs(best_gamma - gamma_min) < 1e-9 or abs(best_gamma - gamma_max) < 1e-9:
        st.warning("恢复率 γ 落在搜索边界上，建议适当调整 γ 搜索范围后重新拟合。")

    st.subheader("📈 原始值对比")
    fig_fit, ax_fit = plt.subplots(figsize=(10, 5))
    ax_fit.plot(result_df["轮次"], result_df["真实值"], marker="o", label="真实数据")
    ax_fit.plot(result_df["轮次"], result_df["拟合值"], marker="s", label="模型拟合")
    ax_fit.set_title("真实曲线与拟合曲线对比", fontproperties=CN_FONT)
    ax_fit.set_xlabel("轮次", fontproperties=CN_FONT)
    ax_fit.set_ylabel("节点数量", fontproperties=CN_FONT)
    ax_fit.legend(prop=CN_FONT)
    ax_fit.grid(True, alpha=0.3)
    st.pyplot(fig_fit)
    plt.close(fig_fit)

    st.subheader("📉 归一化对比")
    real_norm = normalize_series(result_df["真实值"].to_numpy())
    pred_norm = normalize_series(result_df["拟合值"].to_numpy())

    fig_norm, ax_norm = plt.subplots(figsize=(10, 5))
    ax_norm.plot(result_df["轮次"], real_norm, marker="o", label="真实数据（归一化）")
    ax_norm.plot(result_df["轮次"], pred_norm, marker="s", label="模型拟合（归一化）")
    ax_norm.set_title("归一化后的趋势对比", fontproperties=CN_FONT)
    ax_norm.set_xlabel("轮次", fontproperties=CN_FONT)
    ax_norm.set_ylabel("归一化值", fontproperties=CN_FONT)
    ax_norm.legend(prop=CN_FONT)
    ax_norm.grid(True, alpha=0.3)
    st.pyplot(fig_norm)
    plt.close(fig_norm)

    st.subheader("📋 误差表")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("📝 自动拟合结论")
    st.text_area(
        "结论文本",
        st.session_state["pe_conclusion_text"],
        height=220
    )

    csv_data = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="下载拟合结果 CSV",
        data=csv_data,
        file_name="parameter_estimation_result.csv",
        mime="text/csv"
    )

    if "pe_self_test_beta" in st.session_state and "pe_self_test_gamma" in st.session_state:
        st.subheader("🧪 自检对照结果")
        gt_beta = st.session_state["pe_self_test_beta"]
        gt_gamma = st.session_state["pe_self_test_gamma"]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("自检真实 β", f"{gt_beta:.4f}")
        s2.metric("拟合 β", f"{best_beta:.4f}", delta=f"{best_beta - gt_beta:.4f}")
        s3.metric("自检真实 γ", f"{gt_gamma:.4f}")
        s4.metric("拟合 γ", f"{best_gamma:.4f}", delta=f"{best_gamma - gt_gamma:.4f}")

    st.success(
        f"已估计得到参数：β={best_beta:.4f}，γ={best_gamma:.4f}。"
        "下一步可以把这两个参数回填到模拟页面进行验证。"
    )

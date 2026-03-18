"""BP 反向传播可视化（西瓜书 5.3）

页面用途：
- 用一个 1-1-1 神经网络演示一次完整的前向传播、反向传播与参数更新
- 强调“公式 - 代入 - 数值结果”的对应关系，帮助初学者理解 BP

运行方式：
- streamlit run character5/5.3/app.py

适合学习内容：
- 《机器学习（西瓜书）》第 5 章 5.3 节：误差逆传播算法（BP）
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager


DEFAULTS = {
    "x": 1.0,
    "y": 1.0,
    "w1": 0.5,
    "b1": 0.0,
    "w2": 0.8,
    "b2": 0.0,
    "eta": 0.5,
    "auto_compute": True,
}

CLIP_LIMIT = 60.0


def fmt(value: float, digits: int = 6) -> str:
    """统一格式化浮点数，避免页面展示过于杂乱。"""
    return f"{value:.{digits}f}"


def configure_matplotlib_fonts() -> str | None:
    """为图表选择可用的中文字体，避免 matplotlib 标题和标签乱码。"""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimSun",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name] + preferred_fonts
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def sigmoid(z: float) -> tuple[float, bool]:
    """安全计算 sigmoid，并返回是否触发了截断提示。"""
    clipped = False
    safe_z = z
    if z > CLIP_LIMIT:
        safe_z = CLIP_LIMIT
        clipped = True
    elif z < -CLIP_LIMIT:
        safe_z = -CLIP_LIMIT
        clipped = True
    return 1.0 / (1.0 + math.exp(-safe_z)), clipped


def sigmoid_grad_from_output(sigmoid_output: float) -> float:
    """利用 sigmoid(z) * (1 - sigmoid(z)) 计算导数。"""
    return sigmoid_output * (1.0 - sigmoid_output)


def forward_pass(params: dict[str, float]) -> dict[str, float | bool]:
    """执行一次前向传播，返回教学展示所需的全部中间量。"""
    x = params["x"]
    y = params["y"]
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    z1 = w1 * x + b1
    h, clip_h = sigmoid(z1)
    z2 = w2 * h + b2
    y_hat, clip_y_hat = sigmoid(z2)
    loss = 0.5 * (y - y_hat) ** 2

    return {
        "x": x,
        "y": y,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "z1": z1,
        "h": h,
        "z2": z2,
        "y_hat": y_hat,
        "loss": loss,
        "clip_warning": clip_h or clip_y_hat,
    }


def backward_pass(forward: dict[str, float | bool]) -> dict[str, float]:
    """基于前向结果计算反向传播中的误差项与梯度。"""
    y = float(forward["y"])
    y_hat = float(forward["y_hat"])
    z1 = float(forward["z1"])
    h = float(forward["h"])
    w2 = float(forward["w2"])
    x = float(forward["x"])

    dE_dy_hat = y_hat - y
    dy_hat_dz2 = sigmoid_grad_from_output(y_hat)
    delta_out = dE_dy_hat * dy_hat_dz2

    dE_dw2 = delta_out * h
    dE_db2 = delta_out

    _ = z1
    dh_dz1 = sigmoid_grad_from_output(h)
    delta_hidden = delta_out * w2 * dh_dz1

    dE_dw1 = delta_hidden * x
    dE_db1 = delta_hidden

    return {
        "dE_dy_hat": dE_dy_hat,
        "dy_hat_dz2": dy_hat_dz2,
        "delta_out": delta_out,
        "dE_dw2": dE_dw2,
        "dE_db2": dE_db2,
        "dh_dz1": dh_dz1,
        "delta_hidden": delta_hidden,
        "dE_dw1": dE_dw1,
        "dE_db1": dE_db1,
    }


def update_params(params: dict[str, float], grads: dict[str, float]) -> tuple[dict[str, float], pd.DataFrame]:
    """按梯度下降规则更新参数，并生成教学表格。"""
    eta = params["eta"]
    updated = {
        "x": params["x"],
        "y": params["y"],
        "eta": eta,
        "w1": params["w1"] - eta * grads["dE_dw1"],
        "b1": params["b1"] - eta * grads["dE_db1"],
        "w2": params["w2"] - eta * grads["dE_dw2"],
        "b2": params["b2"] - eta * grads["dE_db2"],
    }

    table = pd.DataFrame(
        [
            {
                "参数名": "w1",
                "更新前": params["w1"],
                "梯度": grads["dE_dw1"],
                "学习率 eta": eta,
                "更新后": updated["w1"],
            },
            {
                "参数名": "b1",
                "更新前": params["b1"],
                "梯度": grads["dE_db1"],
                "学习率 eta": eta,
                "更新后": updated["b1"],
            },
            {
                "参数名": "w2",
                "更新前": params["w2"],
                "梯度": grads["dE_dw2"],
                "学习率 eta": eta,
                "更新后": updated["w2"],
            },
            {
                "参数名": "b2",
                "更新前": params["b2"],
                "梯度": grads["dE_db2"],
                "学习率 eta": eta,
                "更新后": updated["b2"],
            },
        ]
    )
    return updated, table


def run_updated_forward(updated_params: dict[str, float]) -> dict[str, float | bool]:
    """用更新后的参数再做一次前向传播，用于比较 loss 是否下降。"""
    return forward_pass(updated_params)


def format_formula(title: str, formula: str, substituted: str, result: float) -> str:
    """把教学步骤统一格式化成“公式-代入-结果”的展示文本。"""
    return (
        f"**{title}**\n\n"
        f"- 公式：`{formula}`\n"
        f"- 代入：`{substituted}`\n"
        f"- 结果：`{fmt(result)}`"
    )


def draw_network_diagram() -> plt.Figure:
    """绘制 1-1-1 网络结构图，让学习者先看到连接关系。"""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(-0.7, 0.7)
    ax.axis("off")

    positions = {"x": (0, 0), "h": (1, 0), "y_hat": (2, 0)}
    labels = {"x": "输入 x", "h": "隐藏层 h", "y_hat": "输出 y_hat"}

    for node, (px, py) in positions.items():
        circle = plt.Circle((px, py), 0.16, color="#E8F1FB", ec="#2F5D8C", lw=2)
        ax.add_patch(circle)
        ax.text(px, py, labels[node], ha="center", va="center", fontsize=11)

    ax.annotate("", xy=(0.84, 0), xytext=(0.16, 0), arrowprops=dict(arrowstyle="->", lw=2, color="#2F5D8C"))
    ax.annotate("", xy=(1.84, 0), xytext=(1.16, 0), arrowprops=dict(arrowstyle="->", lw=2, color="#2F5D8C"))
    ax.text(0.5, 0.16, "w1", ha="center", fontsize=11, color="#1B4965")
    ax.text(1.5, 0.16, "w2", ha="center", fontsize=11, color="#1B4965")
    ax.text(0.5, -0.25, "z1 = w1*x + b1", ha="center", fontsize=10, color="#4F6D7A")
    ax.text(1.5, -0.25, "z2 = w2*h + b2", ha="center", fontsize=10, color="#4F6D7A")
    return fig


def draw_param_compare_chart(before: dict[str, float], after: dict[str, float]) -> plt.Figure:
    """绘制参数更新前后对比条形图。"""
    names = ["w1", "b1", "w2", "b2"]
    before_vals = [before[name] for name in names]
    after_vals = [after[name] for name in names]
    x_pos = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x_pos - width / 2, before_vals, width=width, label="更新前", color="#9CC5A1")
    ax.bar(x_pos + width / 2, after_vals, width=width, label="更新后", color="#2D6A4F")
    ax.set_xticks(x_pos, names)
    ax.set_ylabel("参数值")
    ax.set_title("参数更新前后对比")
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    return fig


def draw_loss_compare_chart(loss_before: float, loss_after: float) -> plt.Figure:
    """绘制更新前后 loss 对比图。"""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(["更新前", "更新后"], [loss_before, loss_after], color=["#F4A261", "#2A9D8F"])
    ax.set_ylabel("Loss")
    ax.set_title("一次更新前后的损失对比")
    for idx, value in enumerate([loss_before, loss_after]):
        ax.text(idx, value, fmt(value, 4), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    return fig


def build_summary_text(forward: dict[str, float | bool], grads: dict[str, float], after: dict[str, float | bool]) -> list[str]:
    """根据当前计算结果自动生成几句中文教学总结。"""
    y = float(forward["y"])
    y_hat = float(forward["y_hat"])
    loss_before = float(forward["loss"])
    loss_after = float(after["loss"])
    delta_out = grads["delta_out"]

    if y_hat < y:
        output_judgement = "当前 y_hat 小于 y，说明模型输出偏低，需要把输出往更大的方向推。"
    elif y_hat > y:
        output_judgement = "当前 y_hat 大于 y，说明模型输出偏高，需要把输出往更小的方向拉。"
    else:
        output_judgement = "当前 y_hat 与 y 几乎一致，模型已经非常接近目标。"

    if grads["dE_dw2"] < 0:
        w2_judgement = "因为 dE/dw2 为负，所以做梯度下降后 w2 会增大。"
    elif grads["dE_dw2"] > 0:
        w2_judgement = "因为 dE/dw2 为正，所以做梯度下降后 w2 会减小。"
    else:
        w2_judgement = "因为 dE/dw2 接近 0，所以 w2 这一步几乎不会变化。"

    hidden_effect = (
        "隐藏层虽然没有直接标签，但它会通过输出层误差项继续反传，形成 delta_hidden。"
        if abs(delta_out) > 1e-10
        else "输出层误差项已经很小，因此传回隐藏层的调整也会很弱。"
    )

    if loss_after < loss_before:
        loss_judgement = "这次更新后 loss 下降了，说明当前梯度方向是有效的。"
    elif loss_after > loss_before:
        loss_judgement = "这次更新后 loss 反而上升，通常意味着学习率偏大或当前位置较敏感。"
    else:
        loss_judgement = "这次更新前后 loss 几乎不变，说明当前步长或梯度都比较小。"

    return [output_judgement, w2_judgement, hidden_effect, loss_judgement]


def init_state() -> None:
    """初始化页面状态，支持重置与自动计算。"""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "result_ready" not in st.session_state:
        st.session_state.result_ready = True


def reset_defaults() -> None:
    """把侧边栏参数恢复为默认值。"""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.result_ready = True


def sidebar_controls() -> tuple[dict[str, float], bool, bool]:
    """渲染侧边栏控件，收集用户输入。"""
    with st.sidebar:
        st.header("参数控制")
        st.caption("建议先用默认值观察一遍，再逐个修改参数看链式求导如何变化。")

        x = st.slider("x（输入值）", min_value=-2.0, max_value=2.0, value=float(st.session_state.x), step=0.1, key="x")
        y = st.slider("y（真实标签）", min_value=0.0, max_value=1.0, value=float(st.session_state.y), step=0.1, key="y")
        w1 = st.number_input("w1", min_value=-5.0, max_value=5.0, value=float(st.session_state.w1), step=0.1, key="w1")
        b1 = st.number_input("b1", min_value=-5.0, max_value=5.0, value=float(st.session_state.b1), step=0.1, key="b1")
        w2 = st.number_input("w2", min_value=-5.0, max_value=5.0, value=float(st.session_state.w2), step=0.1, key="w2")
        b2 = st.number_input("b2", min_value=-5.0, max_value=5.0, value=float(st.session_state.b2), step=0.1, key="b2")
        eta = st.slider("eta（学习率）", min_value=0.01, max_value=1.0, value=float(st.session_state.eta), step=0.01, key="eta")
        auto_compute = st.checkbox("自动实时计算", value=bool(st.session_state.auto_compute), key="auto_compute")

        run_clicked = st.button("执行一次前向 + 反向传播", use_container_width=True)
        st.button("重置为默认值", use_container_width=True, on_click=reset_defaults)

    params = {"x": x, "y": y, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "eta": eta}
    return params, auto_compute, run_clicked or auto_compute


def render_step_block(index: int, title: str, formula: str, substituted: str, result: float) -> None:
    """渲染单个步骤块，突出公式与代入关系。"""
    with st.container(border=True):
        st.markdown(format_formula(f"Step {index}: {title}", formula, substituted, result))


def render_formula_explainer() -> None:
    """渲染公式说明折叠区，补充链式法则背后的直觉。"""
    with st.expander("公式说明：为什么 BP 可以这样算？", expanded=False):
        st.markdown(
            """
1. 输出层误差项为什么是 `delta_out = (y_hat - y) * sigmoid'(z2)`  
因为损失函数是 `E = 1/2 * (y - y_hat)^2`，先对 `y_hat` 求导得到 `dE/dy_hat = y_hat - y`，再乘上 `dy_hat/dz2 = sigmoid'(z2)`，这就是链式法则。

2. 隐藏层误差项为什么是 `delta_hidden = delta_out * w2 * sigmoid'(z1)`  
隐藏层不会直接和标签比较，所以它的“责任”来自后面的输出层。误差先经过权重 `w2` 传回，再乘上本层激活函数导数 `sigmoid'(z1)`。

3. 为什么梯度等于“误差项 × 输入”  
以 `w2` 为例，`z2 = w2 * h + b2`，所以 `dz2/dw2 = h`。因此 `dE/dw2 = dE/dz2 * dz2/dw2 = delta_out * h`。  
同理，`dE/dw1 = delta_hidden * x`。可以把它理解成：边终点的误差有多大，乘以前一节点给了多少输入。
"""
        )


def render_flow_caption() -> None:
    """用文字流程强调前向与反向的方向差异。"""
    st.markdown(
        """
`前向：x -> z1 -> h -> z2 -> y_hat -> E`  
`反向：E -> delta_out -> (w2) -> delta_hidden -> grads`
"""
    )


def main() -> None:
    """构建完整的 Streamlit 教学页面。"""
    st.set_page_config(page_title="BP 反向传播可视化（西瓜书 5.3）", layout="wide")
    init_state()
    font_name = configure_matplotlib_fonts()

    params, auto_compute, should_run = sidebar_controls()

    if should_run:
        st.session_state.result_ready = True

    st.title("BP 反向传播可视化（西瓜书 5.3）")
    st.markdown(
        """
这是一个 **1-1-1 神经网络** 的 BP 教学演示器。  
前向传播负责算预测值，反向传播负责算梯度，梯度下降负责更新参数。
"""
    )

    st.info("BP 一句话总结：前向传播算预测，反向传播算责任，梯度下降改参数。")

    col_net, col_flow = st.columns([1.2, 1])
    with col_net:
        st.subheader("网络结构图")
        st.pyplot(draw_network_diagram(), use_container_width=True)
    with col_flow:
        st.subheader("流程图")
        render_flow_caption()
        st.markdown(
            """
- 输入层到隐藏层：`z1 = w1*x + b1`，`h = sigmoid(z1)`
- 隐藏层到输出层：`z2 = w2*h + b2`，`y_hat = sigmoid(z2)`
- 损失函数：`E = 1/2 * (y - y_hat)^2`
"""
        )
        if font_name is None:
            st.caption("提示：当前环境未检测到常见中文字体，图表中的中文可能显示异常。")

    render_formula_explainer()

    if not st.session_state.result_ready:
        st.warning("请点击“执行一次前向 + 反向传播”，或开启“自动实时计算”。")
        return

    forward = forward_pass(params)
    grads = backward_pass(forward)
    updated_params, update_table = update_params(params, grads)
    after_forward = run_updated_forward(updated_params)

    if forward["clip_warning"] or after_forward["clip_warning"]:
        st.warning("检测到 sigmoid 输入过大，内部已做温和截断以避免 exp 溢出。你仍然可以继续观察 BP 过程。")

    st.subheader("前向传播区域")
    render_step_block(
        1,
        "z1 = w1*x + b1",
        "z1 = w1*x + b1",
        f"z1 = {fmt(params['w1'])}*{fmt(params['x'])} + {fmt(params['b1'])}",
        float(forward["z1"]),
    )
    render_step_block(
        2,
        "h = sigmoid(z1)",
        "h = sigmoid(z1)",
        f"h = sigmoid({fmt(float(forward['z1']))})",
        float(forward["h"]),
    )
    render_step_block(
        3,
        "z2 = w2*h + b2",
        "z2 = w2*h + b2",
        f"z2 = {fmt(params['w2'])}*{fmt(float(forward['h']))} + {fmt(params['b2'])}",
        float(forward["z2"]),
    )
    render_step_block(
        4,
        "y_hat = sigmoid(z2)",
        "y_hat = sigmoid(z2)",
        f"y_hat = sigmoid({fmt(float(forward['z2']))})",
        float(forward["y_hat"]),
    )
    render_step_block(
        5,
        "E = 1/2*(y-y_hat)^2",
        "E = 1/2*(y-y_hat)^2",
        f"E = 1/2*({fmt(params['y'])} - {fmt(float(forward['y_hat']))})^2",
        float(forward["loss"]),
    )

    st.subheader("反向传播区域")
    render_step_block(
        1,
        "dE/dy_hat",
        "dE/dy_hat = y_hat - y",
        f"dE/dy_hat = {fmt(float(forward['y_hat']))} - {fmt(params['y'])}",
        grads["dE_dy_hat"],
    )
    render_step_block(
        2,
        "dy_hat/dz2",
        "dy_hat/dz2 = sigmoid(z2)*(1-sigmoid(z2))",
        f"dy_hat/dz2 = {fmt(float(forward['y_hat']))}*(1-{fmt(float(forward['y_hat']))})",
        grads["dy_hat_dz2"],
    )
    render_step_block(
        3,
        "delta_out = dE/dz2",
        "delta_out = dE/dy_hat * dy_hat/dz2",
        f"delta_out = {fmt(grads['dE_dy_hat'])}*{fmt(grads['dy_hat_dz2'])}",
        grads["delta_out"],
    )
    render_step_block(
        4,
        "dE/dw2",
        "dE/dw2 = delta_out * h",
        f"dE/dw2 = {fmt(grads['delta_out'])}*{fmt(float(forward['h']))}",
        grads["dE_dw2"],
    )
    render_step_block(
        5,
        "dE/db2",
        "dE/db2 = delta_out",
        f"dE/db2 = {fmt(grads['delta_out'])}",
        grads["dE_db2"],
    )
    render_step_block(
        6,
        "dh/dz1",
        "dh/dz1 = sigmoid(z1)*(1-sigmoid(z1))",
        f"dh/dz1 = {fmt(float(forward['h']))}*(1-{fmt(float(forward['h']))})",
        grads["dh_dz1"],
    )
    render_step_block(
        7,
        "delta_hidden = dE/dz1",
        "delta_hidden = delta_out * w2 * dh/dz1",
        f"delta_hidden = {fmt(grads['delta_out'])}*{fmt(params['w2'])}*{fmt(grads['dh_dz1'])}",
        grads["delta_hidden"],
    )
    render_step_block(
        8,
        "dE/dw1",
        "dE/dw1 = delta_hidden * x",
        f"dE/dw1 = {fmt(grads['delta_hidden'])}*{fmt(params['x'])}",
        grads["dE_dw1"],
    )
    render_step_block(
        9,
        "dE/db1",
        "dE/db1 = delta_hidden",
        f"dE/db1 = {fmt(grads['delta_hidden'])}",
        grads["dE_db1"],
    )

    st.subheader("参数更新区域")
    st.markdown("统一更新规则：`new_param = old_param - eta * grad`")
    styled_table = update_table.copy()
    for column in ["更新前", "梯度", "学习率 eta", "更新后"]:
        styled_table[column] = styled_table[column].map(lambda value: float(fmt(value, 6)))
    st.dataframe(styled_table, use_container_width=True, hide_index=True)

    st.subheader("更新前后对比区域")
    loss_before = float(forward["loss"])
    loss_after = float(after_forward["loss"])
    y_hat_before = float(forward["y_hat"])
    y_hat_after = float(after_forward["y_hat"])
    loss_change = loss_after - loss_before

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("y_hat_before", fmt(y_hat_before, 6))
    m2.metric("E_before", fmt(loss_before, 6))
    m3.metric("y_hat_after", fmt(y_hat_after, 6), delta=fmt(y_hat_after - y_hat_before, 6))
    m4.metric("E_after", fmt(loss_after, 6), delta=fmt(loss_change, 6), delta_color="inverse")
    m5.metric("loss_change", fmt(loss_change, 6))

    if loss_after < loss_before:
        st.success("loss 下降：这次梯度更新让模型更接近真实标签。")
    elif loss_after > loss_before:
        st.warning("loss 上升：这次更新没有变好，可以尝试减小学习率 eta。")
    else:
        st.info("loss 基本不变：说明这一步更新很小，或者模型已接近局部稳定点。")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(draw_param_compare_chart(params, updated_params), use_container_width=True)
    with chart_col2:
        st.pyplot(draw_loss_compare_chart(loss_before, loss_after), use_container_width=True)

    st.subheader("人话解释")
    for sentence in build_summary_text(forward, grads, after_forward):
        st.markdown(f"- {sentence}")

    st.subheader("结论总结")
    st.markdown(
        f"""
- 当前前向传播得到 `y_hat = {fmt(y_hat_before)}`，对应损失 `E = {fmt(loss_before)}`。
- 输出层误差项 `delta_out = {fmt(grads['delta_out'])}`，它决定了输出层参数 `w2、b2` 应该朝哪个方向调整。
- 隐藏层误差项 `delta_hidden = {fmt(grads['delta_hidden'])}`，说明隐藏层虽然没有直接监督，但会接收到由输出误差反传回来的“责任”。
- 每条边的梯度都可以理解为：`终点误差项 × 起点输出`。例如 `dE/dw2 = delta_out * h`，`dE/dw1 = delta_hidden * x`。
- 这一步更新后，loss 变化为 `E_after - E_before = {fmt(loss_change)}`。
"""
    )


if __name__ == "__main__":
    main()

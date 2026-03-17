from __future__ import annotations

import streamlit as st

from utils import (
    QuadraticLossConfig,
    describe_feasibility,
    format_point,
    l1_norm,
    l2_norm,
    plot_regularization_geometry,
    solve_l1_constrained,
    solve_l2_constrained,
)


st.set_page_config(page_title="L1 / L2 正则化几何可视化", layout="wide")


DEFAULTS = {
    "regularization_type": "L1",
    "w1_star": 2.0,
    "w2_star": 1.0,
    "a": 1.0,
    "b": 1.0,
    "t": 1.5,
    "plot_range": 4.0,
}


EXAMPLES = {
    "案例 1：L1 基础压缩": {
        "regularization_type": "L1",
        "w1_star": 2.0,
        "w2_star": 1.0,
        "a": 1.0,
        "b": 1.0,
        "t": 1.5,
        "plot_range": 4.0,
    },
    "案例 2：L2 整体收缩": {
        "regularization_type": "L2",
        "w1_star": 2.0,
        "w2_star": 1.0,
        "a": 1.0,
        "b": 1.0,
        "t": 1.5,
        "plot_range": 4.0,
    },
    "案例 3：L1 更明显稀疏": {
        "regularization_type": "L1",
        "w1_star": 3.0,
        "w2_star": 0.2,
        "a": 1.0,
        "b": 1.0,
        "t": 1.5,
        "plot_range": 4.0,
    },
}


def initialize_state() -> None:
    """初始化页面默认参数。"""

    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_preset(preset_name: str) -> None:
    """应用示例参数或恢复默认参数。"""

    preset = DEFAULTS if preset_name == "默认参数" else EXAMPLES[preset_name]
    for key, value in preset.items():
        st.session_state[key] = value


def render_sidebar() -> dict[str, float | str]:
    """渲染侧边栏控件并返回当前参数。"""

    st.sidebar.header("控制区")
    st.sidebar.radio("正则类型", ["L1", "L2"], key="regularization_type")
    st.sidebar.slider("w1_star", -4.0, 4.0, key="w1_star", step=0.1)
    st.sidebar.slider("w2_star", -4.0, 4.0, key="w2_star", step=0.1)
    st.sidebar.slider("约束半径 t", 0.2, 5.0, key="t", step=0.1)
    st.sidebar.slider("a（控制 w1 方向曲率）", 0.2, 3.0, key="a", step=0.1)
    st.sidebar.slider("b（控制 w2 方向曲率）", 0.2, 3.0, key="b", step=0.1)
    st.sidebar.slider("坐标范围 plot_range", 2.0, 6.0, key="plot_range", step=0.5)

    st.sidebar.markdown("---")
    st.sidebar.button(
        "重置默认参数",
        use_container_width=True,
        on_click=apply_preset,
        args=("默认参数",),
    )

    st.sidebar.caption("示例按钮")
    for preset_name in EXAMPLES:
        st.sidebar.button(
            preset_name,
            use_container_width=True,
            on_click=apply_preset,
            args=(preset_name,),
        )

    return {
        "regularization_type": st.session_state["regularization_type"],
        "w1_star": float(st.session_state["w1_star"]),
        "w2_star": float(st.session_state["w2_star"]),
        "a": float(st.session_state["a"]),
        "b": float(st.session_state["b"]),
        "t": float(st.session_state["t"]),
        "plot_range": float(st.session_state["plot_range"]),
    }


def render_result_summary(parameters: dict[str, float | str], constrained_point: tuple[float, float], sparse: bool) -> None:
    """展示关键数值结果。"""

    regularization_type = str(parameters["regularization_type"])
    w1_star = float(parameters["w1_star"])
    w2_star = float(parameters["w2_star"])

    if regularization_type == "L1":
        original_norm = l1_norm((w1_star, w2_star))
        constrained_norm = l1_norm(constrained_point)
        norm_label = "L1 范数"
    else:
        original_norm = l2_norm((w1_star, w2_star))
        constrained_norm = l2_norm(constrained_point)
        norm_label = "L2 范数"

    col1, col2, col3 = st.columns(3)
    col1.metric("原始最优点", format_point((w1_star, w2_star)))
    col2.metric("约束最优点", format_point(constrained_point))
    col3.metric(norm_label, f"原始 {original_norm:.3f} / 约束后 {constrained_norm:.3f}")

    if sparse:
        st.success("该解可视为稀疏解：至少有一个参数接近 0。")
    else:
        st.info("该解不是典型稀疏解：两个参数目前都明显非零。")


def render_conclusion_module() -> None:
    """展示教学总结。"""

    st.subheader("对比结论")
    st.markdown(
        """
        - L1 约束边界是菱形，四个尖角正好落在坐标轴上。
        - 当损失等高线与菱形尖角相切时，往往意味着某个参数被压到了 0，因此更容易得到稀疏解。
        - L2 约束边界是圆，边界处处光滑，没有明显偏向坐标轴的尖角。
        - 因此 L2 更常见的效果是“整体收缩”：两个参数一起变小，但通常不会直接删掉某个参数。
        """
    )


def render_math_expander() -> None:
    """补充惩罚形式与约束形式的关系。"""

    with st.expander("数学说明", expanded=False):
        st.markdown(
            r"""
            常见的正则化写法是：

            - 惩罚形式：$\min_w J(w) + \lambda \|w\|_1$ 或 $\min_w J(w) + \lambda \|w\|_2^2$
            - 约束形式：$\min_w J(w),\ \text{s.t.}\ \|w\|_1 \le t$ 或 $\|w\|_2 \le t$

            这两种写法在思想上是等价的：
            `lambda` 控制“惩罚强度”，`t` 控制“可用预算”。
            在几何图像里，约束形式更直观，因为可以直接看到等高线与约束边界第一次接触的位置。
            """
        )


def main() -> None:
    """主函数。"""

    initialize_state()

    st.title("L1 / L2 正则化几何可视化")
    st.caption("通过二维参数空间中的等高线、L1 菱形约束、L2 圆形约束，理解为什么 L1 更容易产生稀疏解。")

    parameters = render_sidebar()
    config = QuadraticLossConfig(
        w1_star=float(parameters["w1_star"]),
        w2_star=float(parameters["w2_star"]),
        a=float(parameters["a"]),
        b=float(parameters["b"]),
    )

    if parameters["regularization_type"] == "L1":
        result = solve_l1_constrained(config, float(parameters["t"]))
    else:
        result = solve_l2_constrained(config, float(parameters["t"]))

    figure = plot_regularization_geometry(
        config=config,
        radius=float(parameters["t"]),
        regularization_type=str(parameters["regularization_type"]),
        plot_range=float(parameters["plot_range"]),
        result=result,
    )
    st.pyplot(figure, use_container_width=False)

    render_result_summary(parameters, result.point, result.sparse)

    st.subheader("当前情形解释")
    for text_line in describe_feasibility(result, str(parameters["regularization_type"])):
        st.write(f"- {text_line}")

    st.write(
        f"当前原始损失函数为：J(w1, w2) = {parameters['a']:.1f} * (w1 - {parameters['w1_star']:.1f})^2 + "
        f"{parameters['b']:.1f} * (w2 - {parameters['w2_star']:.1f})^2"
    )
    st.write(f"当前约束最优点数值：`constrained optimum = {format_point(result.point)}`")

    if result.active_constraint:
        st.write(
            f"由于原始最优点超出了 {parameters['regularization_type']} 可行域，最终解落在边界上，这就是正则化实际产生影响的时刻。"
        )

    render_conclusion_module()
    render_math_expander()


if __name__ == "__main__":
    main()

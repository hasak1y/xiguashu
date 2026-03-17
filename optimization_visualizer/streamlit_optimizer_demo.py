from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager


DEFAULT_X_RANGE = (-3.0, 3.0)
DEFAULT_POINT_COUNT = 800
SECOND_DERIVATIVE_EPS = 1e-8
CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]


@dataclass
class IterationRecord:
    step: int
    x: float
    fx: float
    grad: float
    hess: float
    note: str = ""


class Function1D:
    def __init__(
        self,
        name: str,
        formula: str,
        func: Callable[[np.ndarray | float], np.ndarray | float],
        grad: Callable[[np.ndarray | float], np.ndarray | float],
        hess: Callable[[np.ndarray | float], np.ndarray | float],
    ) -> None:
        self.name = name
        self.formula = formula
        self._func = func
        self._grad = grad
        self._hess = hess

    def f(self, x: np.ndarray | float) -> np.ndarray | float:
        return self._func(x)

    def df(self, x: np.ndarray | float) -> np.ndarray | float:
        return self._grad(x)

    def d2f(self, x: np.ndarray | float) -> np.ndarray | float:
        return self._hess(x)


def configure_matplotlib_for_chinese() -> str:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    selected_font = "DejaVu Sans"

    for candidate in CHINESE_FONT_CANDIDATES:
        if candidate in available_fonts:
            selected_font = candidate
            break

    matplotlib.rcParams["font.sans-serif"] = [selected_font] + CHINESE_FONT_CANDIDATES + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return selected_font


def create_quadratic(a: float, b: float, c: float) -> Function1D:
    return Function1D(
        name="二次函数",
        formula=f"f(x) = {a:.3f}x^2 + {b:.3f}x + {c:.3f}",
        func=lambda x: a * np.asarray(x) ** 2 + b * np.asarray(x) + c,
        grad=lambda x: 2 * a * np.asarray(x) + b,
        hess=lambda x: np.zeros_like(np.asarray(x), dtype=float) + 2 * a,
    )


def create_quartic(a: float, b: float, c: float, d: float, e: float) -> Function1D:
    return Function1D(
        name="四次函数",
        formula=f"f(x) = {a:.3f}x^4 + {b:.3f}x^3 + {c:.3f}x^2 + {d:.3f}x + {e:.3f}",
        func=lambda x: a * np.asarray(x) ** 4
        + b * np.asarray(x) ** 3
        + c * np.asarray(x) ** 2
        + d * np.asarray(x)
        + e,
        grad=lambda x: 4 * a * np.asarray(x) ** 3
        + 3 * b * np.asarray(x) ** 2
        + 2 * c * np.asarray(x)
        + d,
        hess=lambda x: 12 * a * np.asarray(x) ** 2 + 6 * b * np.asarray(x) + 2 * c,
    )


def create_fixed_wave() -> Function1D:
    return Function1D(
        name="固定教学函数",
        formula="f(x) = x^4 - 3x^2 + 0.5x + 1",
        func=lambda x: np.asarray(x) ** 4 - 3 * np.asarray(x) ** 2 + 0.5 * np.asarray(x) + 1,
        grad=lambda x: 4 * np.asarray(x) ** 3 - 6 * np.asarray(x) + 0.5,
        hess=lambda x: 12 * np.asarray(x) ** 2 - 6,
    )


def create_local_wave() -> Function1D:
    return Function1D(
        name="局部波动函数",
        formula="f(x) = x^4 - 3x^2 + 0.5x + sin(3x)",
        func=lambda x: np.asarray(x) ** 4
        - 3 * np.asarray(x) ** 2
        + 0.5 * np.asarray(x)
        + np.sin(3 * np.asarray(x)),
        grad=lambda x: 4 * np.asarray(x) ** 3
        - 6 * np.asarray(x)
        + 0.5
        + 3 * np.cos(3 * np.asarray(x)),
        hess=lambda x: 12 * np.asarray(x) ** 2 - 6 - 9 * np.sin(3 * np.asarray(x)),
    )


def create_random_function(kind: str, seed: int) -> Function1D:
    rng = random.Random(seed)

    if kind == "随机二次函数":
        a = rng.choice([-1, 1]) * rng.uniform(0.8, 2.5)
        b = rng.uniform(-4.0, 4.0)
        c = rng.uniform(-2.0, 2.0)
        return create_quadratic(a, b, c)

    if kind == "随机四次函数":
        a = rng.uniform(0.3, 1.2)
        b = rng.uniform(-2.5, 2.5)
        c = rng.uniform(-4.0, 4.0)
        d = rng.uniform(-3.0, 3.0)
        e = rng.uniform(-2.0, 2.0)
        return create_quartic(a, b, c, d, e)

    amplitude = rng.uniform(0.5, 1.8)
    bias = rng.uniform(-1.0, 1.0)
    return Function1D(
        name="随机局部波动函数",
        formula=f"f(x) = x^4 - 3x^2 + {bias:.3f}x + {amplitude:.3f}sin(3x)",
        func=lambda x: np.asarray(x) ** 4
        - 3 * np.asarray(x) ** 2
        + bias * np.asarray(x)
        + amplitude * np.sin(3 * np.asarray(x)),
        grad=lambda x: 4 * np.asarray(x) ** 3
        - 6 * np.asarray(x)
        + bias
        + 3 * amplitude * np.cos(3 * np.asarray(x)),
        hess=lambda x: 12 * np.asarray(x) ** 2 - 6 - 9 * amplitude * np.sin(3 * np.asarray(x)),
    )


def gradient_descent(function: Function1D, x0: float, lr: float, max_iter: int, tol: float) -> List[IterationRecord]:
    records: List[IterationRecord] = []
    x = float(x0)

    for step in range(max_iter + 1):
        fx = float(function.f(x))
        grad = float(function.df(x))
        hess = float(function.d2f(x))
        records.append(IterationRecord(step=step, x=x, fx=fx, grad=grad, hess=hess))

        if abs(grad) < tol:
            records[-1].note = "梯度足够小，停止。"
            break

        next_x = x - lr * grad
        if not math.isfinite(next_x):
            records[-1].note = "迭代出现非有限值，停止。"
            break

        if abs(next_x - x) < tol:
            records[-1].note = "步长足够小，停止。"
            break

        x = next_x

    return records


def newton_method(function: Function1D, x0: float, max_iter: int, tol: float) -> List[IterationRecord]:
    records: List[IterationRecord] = []
    x = float(x0)

    for step in range(max_iter + 1):
        fx = float(function.f(x))
        grad = float(function.df(x))
        hess = float(function.d2f(x))
        records.append(IterationRecord(step=step, x=x, fx=fx, grad=grad, hess=hess))

        if abs(grad) < tol:
            records[-1].note = "梯度足够小，停止。"
            break

        if abs(hess) < SECOND_DERIVATIVE_EPS:
            records[-1].note = "二阶导数过小，为避免除零已停止。"
            break

        next_x = x - grad / hess
        if not math.isfinite(next_x):
            records[-1].note = "迭代出现非有限值，停止。"
            break

        if abs(next_x - x) < tol:
            records[-1].note = "步长足够小，停止。"
            break

        x = next_x

    return records


def choose_function(kind: str, seed: int, a: float, b: float, c: float, d: float, e: float) -> Function1D:
    if kind == "固定教学函数":
        return create_fixed_wave()
    if kind == "二次函数":
        return create_quadratic(a, b, c)
    if kind == "四次函数":
        return create_quartic(a, b, c, d, e)
    if kind == "局部波动函数":
        return create_local_wave()
    return create_random_function(kind, seed)


def compute_tangent_line(function: Function1D, x0: float, x_min: float, x_max: float) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, 120)
    y0 = float(function.f(x0))
    slope = float(function.df(x0))
    ys = y0 + slope * (xs - x0)
    return xs, ys


def prepare_plot_bounds(function: Function1D, gd_records: List[IterationRecord], newton_records: List[IterationRecord]) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    iter_xs = [item.x for item in gd_records] + [item.x for item in newton_records]
    x_min = min(DEFAULT_X_RANGE[0], min(iter_xs) - 1.0)
    x_max = max(DEFAULT_X_RANGE[1], max(iter_xs) + 1.0)
    xs = np.linspace(x_min, x_max, DEFAULT_POINT_COUNT)
    ys = np.asarray(function.f(xs), dtype=float)
    finite_ys = ys[np.isfinite(ys)]

    if finite_ys.size == 0:
        return xs, (x_min, x_max), (-5.0, 5.0)

    y_min = float(np.min(finite_ys))
    y_max = float(np.max(finite_ys))
    margin = max(1.0, 0.15 * (y_max - y_min + 1e-9))
    return xs, (x_min, x_max), (y_min - margin, y_max + margin)


def build_records_table(records: List[IterationRecord], method_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "方法": method_name,
                "步数": item.step,
                "x": item.x,
                "f(x)": item.fx,
                "f'(x)": item.grad,
                "f''(x)": item.hess,
                "备注": item.note,
            }
            for item in records
        ]
    )


def plot_frame(
    function: Function1D,
    gd_records: List[IterationRecord],
    newton_records: List[IterationRecord],
    frame: int,
    draw_tangents: bool,
):
    xs, x_limits, y_limits = prepare_plot_bounds(function, gd_records, newton_records)
    ys = function.f(xs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"梯度下降法与牛顿法对比\n{function.formula}", fontsize=15)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    ax.plot(xs, ys, color="black", linewidth=2.2, label="函数曲线")

    gd_index = min(frame, len(gd_records) - 1)
    nt_index = min(frame, len(newton_records) - 1)
    gd_slice = gd_records[: gd_index + 1]
    nt_slice = newton_records[: nt_index + 1]

    gd_x = [item.x for item in gd_slice]
    gd_y = [item.fx for item in gd_slice]
    nt_x = [item.x for item in nt_slice]
    nt_y = [item.fx for item in nt_slice]

    ax.plot(gd_x, gd_y, color="#1f77b4", linewidth=2.0, marker="o", label="梯度下降路径")
    ax.plot(nt_x, nt_y, color="#d62728", linewidth=2.0, marker="o", label="牛顿法路径")
    ax.plot(gd_x[-1], gd_y[-1], "o", color="#1f77b4", markersize=9)
    ax.plot(nt_x[-1], nt_y[-1], "o", color="#d62728", markersize=9)

    if draw_tangents:
        gd_tx, gd_ty = compute_tangent_line(function, gd_slice[-1].x, x_limits[0], x_limits[1])
        nt_tx, nt_ty = compute_tangent_line(function, nt_slice[-1].x, x_limits[0], x_limits[1])
        ax.plot(gd_tx, gd_ty, linestyle="--", linewidth=1.4, color="#1f77b4", alpha=0.75, label="梯度下降当前切线")
        ax.plot(nt_tx, nt_ty, linestyle="--", linewidth=1.4, color="#d62728", alpha=0.75, label="牛顿法当前切线")

    gd_item = gd_slice[-1]
    nt_item = nt_slice[-1]
    annotation = (
        f"当前显示步数: {frame}\n"
        f"梯度下降: step={gd_item.step}, x={gd_item.x:.4f}, f(x)={gd_item.fx:.4f}\n"
        f"牛顿法: step={nt_item.step}, x={nt_item.x:.4f}, f(x)={nt_item.fx:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def render_summary(function: Function1D, gd_records: List[IterationRecord], newton_records: List[IterationRecord]) -> None:
    gd_last = gd_records[-1]
    nt_last = newton_records[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("函数类型", function.name)
    col2.metric("梯度下降迭代步数", gd_last.step)
    col3.metric("牛顿法迭代步数", nt_last.step)

    st.markdown(
        f"""
**当前函数**: `{function.formula}`  
**梯度下降最终点**: `x={gd_last.x:.6f}`, `f(x)={gd_last.fx:.6f}`  
**牛顿法最终点**: `x={nt_last.x:.6f}`, `f(x)={nt_last.fx:.6f}`
"""
    )


def main() -> None:
    selected_font = configure_matplotlib_for_chinese()
    st.set_page_config(page_title="函数优化可视化演示", layout="wide")
    st.title("函数优化可视化演示工具")
    st.caption("使用 Streamlit 调节函数、初始点和优化参数，对比梯度下降法与牛顿法的迭代过程。")

    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42

    with st.sidebar:
        st.header("参数面板")
        function_kind = st.selectbox(
            "函数类型",
            ["固定教学函数", "二次函数", "四次函数", "局部波动函数", "随机二次函数", "随机四次函数", "随机局部波动函数"],
        )
        if st.button("重新随机生成函数"):
            st.session_state.random_seed = random.randint(0, 10**6)

        st.markdown("**自定义参数**")
        a = st.slider("a", -3.0, 3.0, 1.5, 0.1)
        b = st.slider("b", -5.0, 5.0, -2.0, 0.1)
        c = st.slider("c", -5.0, 5.0, 0.5, 0.1)
        d = st.slider("d", -5.0, 5.0, 0.8, 0.1)
        e = st.slider("e", -5.0, 5.0, 1.0, 0.1)

        st.markdown("**优化参数**")
        x0 = st.slider("初始点 x0", -4.0, 4.0, 1.8, 0.1)
        lr = st.slider("学习率 lr", 0.001, 0.5, 0.08, 0.001)
        max_iter = st.slider("最大迭代次数", 3, 80, 25, 1)
        tol = st.slider("收敛阈值", 1e-8, 1e-2, 1e-6, format="%.8f")
        draw_tangents = st.checkbox("显示当前切线", value=True)
        auto_play = st.checkbox("自动播放步数", value=False)
        play_speed = st.slider("播放速度（秒）", 0.1, 1.5, 0.4, 0.1)

    function = choose_function(function_kind, st.session_state.random_seed, a, b, c, d, e)
    gd_records = gradient_descent(function, x0=x0, lr=lr, max_iter=max_iter, tol=tol)
    newton_records = newton_method(function, x0=x0, max_iter=max_iter, tol=tol)
    total_steps = max(len(gd_records), len(newton_records)) - 1

    st.info(f"Matplotlib 当前中文字体回退链首选为: `{selected_font}`。如果仍乱码，请安装 `Microsoft YaHei` 或 `SimHei`。")
    render_summary(function, gd_records, newton_records)

    frame = st.slider("手动查看到第几步", 0, total_steps, min(3, total_steps), 1)
    plot_placeholder = st.empty()

    if auto_play:
        for current_frame in range(total_steps + 1):
            fig = plot_frame(function, gd_records, newton_records, current_frame, draw_tangents)
            plot_placeholder.pyplot(fig, clear_figure=True)
            plt.close(fig)
            time.sleep(play_speed)
    else:
        fig = plot_frame(function, gd_records, newton_records, frame, draw_tangents)
        plot_placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.subheader("迭代日志")
    gd_table = build_records_table(gd_records, "梯度下降法")
    nt_table = build_records_table(newton_records, "牛顿法")
    st.dataframe(pd.concat([gd_table, nt_table], ignore_index=True), use_container_width=True)

    st.subheader("关键逻辑解释")
    st.markdown(
        """
1. **梯度下降为什么这样更新**  
   梯度 `f'(x)` 表示当前位置最陡的上升方向，所以沿着负梯度方向更新 `x <- x - lr * f'(x)`，就能朝更小的函数值移动。

2. **牛顿法为什么这样更新**  
   牛顿法使用当前位置的一阶导数和二阶导数构造二阶近似，再直接估计这个近似模型的极值点，因此更新式为 `x <- x - f'(x)/f''(x)`。

3. **为什么牛顿法通常更快**  
   因为它不仅利用斜率，还利用曲率信息。靠近最优点时，二阶近似通常很准确，所以会比固定步长的梯度下降更快逼近极小值。

4. **为什么牛顿法有时不稳定**  
   当 `f''(x)` 很小、符号变化大，或者当前位置离目标太远时，牛顿步长可能非常大，甚至跳到不理想的位置，所以需要检查二阶导数是否接近 0。
"""
    )


if __name__ == "__main__":
    main()

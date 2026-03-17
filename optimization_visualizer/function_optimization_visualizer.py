"""
函数优化可视化演示工具

教学目标：
1. 展示一元函数曲线
2. 直观观察梯度下降法与牛顿法的迭代过程
3. 比较两种方法在同一函数上的路径差异
4. 支持固定函数、参数化函数和随机函数

运行方式：
    python function_optimization_visualizer.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# =========================
# 全局配置：默认最小可运行版本
# =========================
USE_INTERACTIVE_PROMPT = True
DEFAULT_FUNCTION_KIND = "fixed_wave"
DEFAULT_X0 = 1.8
DEFAULT_LR = 0.08
DEFAULT_MAX_ITER = 25
DEFAULT_TOL = 1e-6
DEFAULT_X_RANGE = (-3.0, 3.0)
DEFAULT_POINT_COUNT = 800
DEFAULT_DRAW_TANGENTS = True
DEFAULT_ANIMATION_INTERVAL_MS = 900
SECOND_DERIVATIVE_EPS = 1e-8


@dataclass
class IterationRecord:
    """保存一次迭代的信息，便于打印和绘图。"""

    step: int
    x: float
    fx: float
    grad: float
    hess: float
    note: str = ""


class Function1D:
    """封装一元函数、导数和二阶导数。"""

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


def create_quadratic(a: float, b: float, c: float) -> Function1D:
    return Function1D(
        name="Quadratic",
        formula=f"f(x) = {a:.3f}x^2 + {b:.3f}x + {c:.3f}",
        func=lambda x: a * np.asarray(x) ** 2 + b * np.asarray(x) + c,
        grad=lambda x: 2 * a * np.asarray(x) + b,
        hess=lambda x: np.zeros_like(np.asarray(x), dtype=float) + 2 * a,
    )


def create_quartic(a: float, b: float, c: float, d: float, e: float) -> Function1D:
    return Function1D(
        name="Quartic",
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


def create_wave_function() -> Function1D:
    return Function1D(
        name="Fixed Teaching Function",
        formula="f(x) = x^4 - 3x^2 + 0.5x + 1",
        func=lambda x: np.asarray(x) ** 4 - 3 * np.asarray(x) ** 2 + 0.5 * np.asarray(x) + 1,
        grad=lambda x: 4 * np.asarray(x) ** 3 - 6 * np.asarray(x) + 0.5,
        hess=lambda x: 12 * np.asarray(x) ** 2 - 6,
    )


def create_local_wave_function() -> Function1D:
    return Function1D(
        name="Local Wavy Function",
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


def create_random_function(kind: str) -> Function1D:
    """随机生成函数参数。"""
    if kind == "quadratic":
        a = random.choice([-1, 1]) * random.uniform(0.8, 2.5)
        if abs(a) < 0.4:
            a = 1.0
        b = random.uniform(-4.0, 4.0)
        c = random.uniform(-2.0, 2.0)
        return create_quadratic(a, b, c)

    if kind == "quartic":
        a = random.uniform(0.3, 1.2)
        b = random.uniform(-2.5, 2.5)
        c = random.uniform(-4.0, 4.0)
        d = random.uniform(-3.0, 3.0)
        e = random.uniform(-2.0, 2.0)
        return create_quartic(a, b, c, d, e)

    amplitude = random.uniform(0.5, 1.8)
    bias = random.uniform(-1.0, 1.0)
    return Function1D(
        name="Random Wavy Function",
        formula=(
            f"f(x) = x^4 - 3x^2 + {bias:.3f}x + "
            f"{amplitude:.3f}sin(3x)"
        ),
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


def gradient_descent(
    function: Function1D,
    x0: float,
    lr: float,
    max_iter: int,
    tol: float,
) -> List[IterationRecord]:
    """一元梯度下降：x_{t+1} = x_t - lr * f'(x_t)。"""
    records: List[IterationRecord] = []
    x = float(x0)

    for step in range(max_iter + 1):
        fx = float(function.f(x))
        grad = float(function.df(x))
        hess = float(function.d2f(x))
        records.append(IterationRecord(step=step, x=x, fx=fx, grad=grad, hess=hess))

        if abs(grad) < tol:
            records[-1].note = "Gradient small enough, stop."
            break

        next_x = x - lr * grad
        if abs(next_x - x) < tol:
            records[-1].note = "Step size small enough, stop."
            break

        x = next_x

    return records


def newton_method(
    function: Function1D,
    x0: float,
    max_iter: int,
    tol: float,
    second_derivative_eps: float = SECOND_DERIVATIVE_EPS,
) -> List[IterationRecord]:
    """一元牛顿法：x_{t+1} = x_t - f'(x_t) / f''(x_t)。"""
    records: List[IterationRecord] = []
    x = float(x0)

    for step in range(max_iter + 1):
        fx = float(function.f(x))
        grad = float(function.df(x))
        hess = float(function.d2f(x))
        note = ""
        records.append(IterationRecord(step=step, x=x, fx=fx, grad=grad, hess=hess, note=note))

        if abs(grad) < tol:
            records[-1].note = "Gradient small enough, stop."
            break

        if abs(hess) < second_derivative_eps:
            records[-1].note = "Second derivative too small, stop to avoid division by zero."
            break

        next_x = x - grad / hess
        if not math.isfinite(next_x):
            records[-1].note = "Encountered non-finite iterate, stop."
            break

        if abs(next_x - x) < tol:
            records[-1].note = "Step size small enough, stop."
            break

        x = next_x

    return records


def print_iteration_table(name: str, records: List[IterationRecord]) -> None:
    """在终端中打印每一步迭代细节。"""
    print(f"\n{'=' * 72}")
    print(f"{name} iteration log")
    print(f"{'=' * 72}")
    print("{:>4} | {:>12} | {:>12} | {:>12} | {:>12} | note".format("step", "x", "f(x)", "f'(x)", "f''(x)"))
    print("-" * 72)
    for item in records:
        print(
            f"{item.step:>4} | "
            f"{item.x:>12.6f} | "
            f"{item.fx:>12.6f} | "
            f"{item.grad:>12.6f} | "
            f"{item.hess:>12.6f} | "
            f"{item.note}"
        )


def compute_tangent_line(
    function: Function1D,
    x0: float,
    x_min: float,
    x_max: float,
    point_count: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算当前点的切线，用于教学演示。"""
    xs = np.linspace(x_min, x_max, point_count)
    y0 = float(function.f(x0))
    slope = float(function.df(x0))
    ys = y0 + slope * (xs - x0)
    return xs, ys


def choose_function() -> Function1D:
    """支持简单交互选择函数。"""
    if not USE_INTERACTIVE_PROMPT:
        if DEFAULT_FUNCTION_KIND == "fixed_wave":
            return create_wave_function()
        if DEFAULT_FUNCTION_KIND == "quadratic":
            return create_quadratic(1.5, -2.0, 0.5)
        if DEFAULT_FUNCTION_KIND == "quartic":
            return create_quartic(0.5, -1.2, -1.0, 0.8, 1.0)
        if DEFAULT_FUNCTION_KIND == "local_wave":
            return create_local_wave_function()
        return create_random_function("local_wave")

    print("\n请选择函数类型：")
    print("1. 固定教学函数: f(x) = x^4 - 3x^2 + 0.5x + 1")
    print("2. 二次函数: f(x) = ax^2 + bx + c")
    print("3. 四次函数: f(x) = ax^4 + bx^3 + cx^2 + dx + e")
    print("4. 带局部波动函数: f(x) = x^4 - 3x^2 + 0.5x + sin(3x)")
    print("5. 随机生成二次函数")
    print("6. 随机生成四次函数")
    print("7. 随机生成局部波动函数")

    choice = input("输入编号，直接回车使用默认教学函数 [1]: ").strip() or "1"

    if choice == "1":
        return create_wave_function()
    if choice == "2":
        a = read_float("请输入 a", 1.5)
        b = read_float("请输入 b", -2.0)
        c = read_float("请输入 c", 0.5)
        return create_quadratic(a, b, c)
    if choice == "3":
        a = read_float("请输入 a", 0.5)
        b = read_float("请输入 b", -1.2)
        c = read_float("请输入 c", -1.0)
        d = read_float("请输入 d", 0.8)
        e = read_float("请输入 e", 1.0)
        return create_quartic(a, b, c, d, e)
    if choice == "4":
        return create_local_wave_function()
    if choice == "5":
        return create_random_function("quadratic")
    if choice == "6":
        return create_random_function("quartic")
    if choice == "7":
        return create_random_function("local_wave")

    print("输入无效，已回退到默认教学函数。")
    return create_wave_function()


def read_float(prompt: str, default: float) -> float:
    text = input(f"{prompt} [{default}]: ").strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        print("输入无效，使用默认值。")
        return default


def read_int(prompt: str, default: int) -> int:
    text = input(f"{prompt} [{default}]: ").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        print("输入无效，使用默认值。")
        return default


def read_bool(prompt: str, default: bool) -> bool:
    default_hint = "y" if default else "n"
    text = input(f"{prompt} [y/n, default={default_hint}]: ").strip().lower()
    if not text:
        return default
    return text in {"y", "yes", "1", "true"}


def get_runtime_config() -> Tuple[float, float, int, float, bool]:
    """收集运行参数。"""
    if not USE_INTERACTIVE_PROMPT:
        return (
            DEFAULT_X0,
            DEFAULT_LR,
            DEFAULT_MAX_ITER,
            DEFAULT_TOL,
            DEFAULT_DRAW_TANGENTS,
        )

    print("\n请输入优化参数：")
    x0 = read_float("初始点 x0", DEFAULT_X0)
    lr = read_float("梯度下降学习率 lr", DEFAULT_LR)
    max_iter = read_int("最大迭代次数", DEFAULT_MAX_ITER)
    tol = read_float("收敛阈值", DEFAULT_TOL)
    draw_tangents = read_bool("是否绘制切线", DEFAULT_DRAW_TANGENTS)
    return x0, lr, max_iter, tol, draw_tangents


def prepare_plot_bounds(
    function: Function1D,
    gd_records: List[IterationRecord],
    newton_records: List[IterationRecord],
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """根据函数和迭代路径生成合理的绘图范围。"""
    iter_xs = [item.x for item in gd_records] + [item.x for item in newton_records]
    x_min = min(DEFAULT_X_RANGE[0], min(iter_xs) - 1.0)
    x_max = max(DEFAULT_X_RANGE[1], max(iter_xs) + 1.0)
    xs = np.linspace(x_min, x_max, DEFAULT_POINT_COUNT)
    ys = np.asarray(function.f(xs), dtype=float)

    finite_ys = ys[np.isfinite(ys)]
    if finite_ys.size == 0:
        y_min, y_max = -5.0, 5.0
    else:
        y_min = float(np.min(finite_ys))
        y_max = float(np.max(finite_ys))
        margin = max(1.0, 0.15 * (y_max - y_min + 1e-9))
        y_min -= margin
        y_max += margin

    return xs, (x_min, x_max), (y_min, y_max)


def create_visualization(
    function: Function1D,
    gd_records: List[IterationRecord],
    newton_records: List[IterationRecord],
    draw_tangents: bool,
) -> None:
    """使用动画展示两种优化方法的迭代路径。"""
    xs, x_limits, y_limits = prepare_plot_bounds(function, gd_records, newton_records)
    ys = function.f(xs)
    total_frames = max(len(gd_records), len(newton_records))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.canvas.manager.set_window_title("函数优化可视化演示工具")
    ax.set_title(
        "梯度下降法 vs 牛顿法 一维优化演示\n"
        f"{function.formula}",
        fontsize=14,
    )
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)

    ax.plot(xs, ys, color="black", linewidth=2.2, label="函数曲线 f(x)")

    gd_line, = ax.plot([], [], color="#1f77b4", linewidth=2.0, marker="o", label="梯度下降路径")
    newton_line, = ax.plot([], [], color="#d62728", linewidth=2.0, marker="o", label="牛顿法路径")

    gd_point, = ax.plot([], [], "o", color="#1f77b4", markersize=9)
    newton_point, = ax.plot([], [], "o", color="#d62728", markersize=9)

    gd_tangent, = ax.plot([], [], linestyle="--", linewidth=1.4, color="#1f77b4", alpha=0.75, label="梯度下降当前切线")
    newton_tangent, = ax.plot([], [], linestyle="--", linewidth=1.4, color="#d62728", alpha=0.75, label="牛顿法当前切线")

    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    explanation_text = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#f8f8f8", "alpha": 0.85},
    )

    ax.legend(loc="upper right")

    def update(frame: int):
        gd_index = min(frame, len(gd_records) - 1)
        newton_index = min(frame, len(newton_records) - 1)
        gd_slice = gd_records[: gd_index + 1]
        newton_slice = newton_records[: newton_index + 1]

        gd_x = [item.x for item in gd_slice]
        gd_y = [item.fx for item in gd_slice]
        nt_x = [item.x for item in newton_slice]
        nt_y = [item.fx for item in newton_slice]

        gd_line.set_data(gd_x, gd_y)
        newton_line.set_data(nt_x, nt_y)
        gd_point.set_data([gd_x[-1]], [gd_y[-1]])
        newton_point.set_data([nt_x[-1]], [nt_y[-1]])

        if draw_tangents:
            gd_tx, gd_ty = compute_tangent_line(function, gd_slice[-1].x, x_limits[0], x_limits[1])
            nt_tx, nt_ty = compute_tangent_line(function, newton_slice[-1].x, x_limits[0], x_limits[1])
            gd_tangent.set_data(gd_tx, gd_ty)
            newton_tangent.set_data(nt_tx, nt_ty)
        else:
            gd_tangent.set_data([], [])
            newton_tangent.set_data([], [])

        gd_item = gd_slice[-1]
        nt_item = newton_slice[-1]
        step_text.set_text(
            "当前动画步数\n"
            f"Gradient Descent: {gd_item.step}\n"
            f"Newton Method: {nt_item.step}"
        )
        explanation_text.set_text(
            "蓝色: 梯度下降按负梯度方向前进\n"
            f"x <- x - lr * f'(x)\n"
            "红色: 牛顿法利用一阶与二阶导数修正\n"
            f"x <- x - f'(x)/f''(x)"
        )

        return (
            gd_line,
            newton_line,
            gd_point,
            newton_point,
            gd_tangent,
            newton_tangent,
            step_text,
            explanation_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=DEFAULT_ANIMATION_INTERVAL_MS,
        blit=False,
        repeat=False,
    )
    fig._animation = animation

    plt.tight_layout()
    plt.show()


def main() -> None:
    """程序入口。"""
    function = choose_function()
    x0, lr, max_iter, tol, draw_tangents = get_runtime_config()

    print("\n已选择函数：")
    print(function.formula)
    print(f"初始点 x0 = {x0}")
    print(f"梯度下降学习率 lr = {lr}")
    print(f"最大迭代次数 = {max_iter}")
    print(f"收敛阈值 = {tol}")
    print(f"绘制切线 = {draw_tangents}")

    gd_records = gradient_descent(function=function, x0=x0, lr=lr, max_iter=max_iter, tol=tol)
    newton_records = newton_method(function=function, x0=x0, max_iter=max_iter, tol=tol)

    print_iteration_table("Gradient Descent", gd_records)
    print_iteration_table("Newton Method", newton_records)

    create_visualization(
        function=function,
        gd_records=gd_records,
        newton_records=newton_records,
        draw_tangents=draw_tangents,
    )


if __name__ == "__main__":
    main()

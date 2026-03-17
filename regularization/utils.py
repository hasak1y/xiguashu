from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


matplotlib.use("Agg")


EPSILON = 1e-9
SPARSE_THRESHOLD = 1e-3


@dataclass(frozen=True)
class QuadraticLossConfig:
    """二次损失函数参数。"""

    w1_star: float
    w2_star: float
    a: float
    b: float


@dataclass(frozen=True)
class OptimizationResult:
    """记录约束优化结果及教学解释所需信息。"""

    point: tuple[float, float]
    unconstrained_point: tuple[float, float]
    unconstrained_loss: float
    constrained_loss: float
    inside_feasible_region: bool
    active_constraint: bool
    sparse: bool
    zero_like_axes: list[str]


def configure_matplotlib_for_chinese() -> None:
    """为 matplotlib 设置常见中文字体候选，并修复负号显示。"""

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def quadratic_loss(w1: np.ndarray | float, w2: np.ndarray | float, config: QuadraticLossConfig) -> np.ndarray | float:
    """计算 J(w1, w2) = a * (w1 - w1_star)^2 + b * (w2 - w2_star)^2。"""

    return config.a * (w1 - config.w1_star) ** 2 + config.b * (w2 - config.w2_star) ** 2


def l1_norm(point: tuple[float, float]) -> float:
    """返回点的 L1 范数。"""

    return abs(point[0]) + abs(point[1])


def l2_norm(point: tuple[float, float]) -> float:
    """返回点的 L2 范数。"""

    return float(np.hypot(point[0], point[1]))


def solve_l2_constrained(config: QuadraticLossConfig, radius: float) -> OptimizationResult:
    """
    求解 L2 约束下的最优点。

    无约束最优点若已在圆内，直接返回该点。
    否则使用 KKT 条件将二维问题化为一维 multiplier 搜索。
    """

    unconstrained = (config.w1_star, config.w2_star)
    unconstrained_loss = float(quadratic_loss(unconstrained[0], unconstrained[1], config))
    if l2_norm(unconstrained) <= radius + EPSILON:
        return build_result(unconstrained, unconstrained, unconstrained_loss, unconstrained_loss, True, False)

    def radius_gap(multiplier: float) -> float:
        w1 = config.a * config.w1_star / (config.a + multiplier)
        w2 = config.b * config.w2_star / (config.b + multiplier)
        return w1**2 + w2**2 - radius**2

    lower = 0.0
    upper = 1.0
    while radius_gap(upper) > 0:
        upper *= 2.0

    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if radius_gap(middle) > 0:
            lower = middle
        else:
            upper = middle

    multiplier = 0.5 * (lower + upper)
    w1 = config.a * config.w1_star / (config.a + multiplier)
    w2 = config.b * config.w2_star / (config.b + multiplier)
    point = (float(w1), float(w2))
    constrained_loss = float(quadratic_loss(point[0], point[1], config))
    return build_result(point, unconstrained, unconstrained_loss, constrained_loss, False, True)


def solve_l1_constrained(config: QuadraticLossConfig, radius: float) -> OptimizationResult:
    """
    求解 L1 约束下的最优点。

    若原始最优点在菱形外，则最优点一定落在边界 |w1| + |w2| = radius 上。
    对四条边分别做一元二次最小化，取其中损失最小的点。
    """

    unconstrained = (config.w1_star, config.w2_star)
    unconstrained_loss = float(quadratic_loss(unconstrained[0], unconstrained[1], config))
    if l1_norm(unconstrained) <= radius + EPSILON:
        return build_result(unconstrained, unconstrained, unconstrained_loss, unconstrained_loss, True, False)

    best_point: tuple[float, float] | None = None
    best_loss = float("inf")
    for sign_w1 in (-1.0, 1.0):
        for sign_w2 in (-1.0, 1.0):
            numerator = config.a * sign_w1 * config.w1_star + config.b * (radius - sign_w2 * config.w2_star)
            denominator = config.a + config.b
            optimal_u = np.clip(numerator / denominator, 0.0, radius)
            candidate = (sign_w1 * optimal_u, sign_w2 * (radius - optimal_u))
            loss_value = float(quadratic_loss(candidate[0], candidate[1], config))
            if loss_value < best_loss:
                best_loss = loss_value
                best_point = (float(candidate[0]), float(candidate[1]))

    assert best_point is not None
    return build_result(best_point, unconstrained, unconstrained_loss, best_loss, False, True)


def build_result(
    point: tuple[float, float],
    unconstrained_point: tuple[float, float],
    unconstrained_loss: float,
    constrained_loss: float,
    inside_feasible_region: bool,
    active_constraint: bool,
) -> OptimizationResult:
    """统一构造优化结果，并给出是否稀疏的判断。"""

    zero_like_axes: list[str] = []
    if abs(point[0]) < SPARSE_THRESHOLD:
        zero_like_axes.append("w1")
    if abs(point[1]) < SPARSE_THRESHOLD:
        zero_like_axes.append("w2")

    return OptimizationResult(
        point=point,
        unconstrained_point=unconstrained_point,
        unconstrained_loss=unconstrained_loss,
        constrained_loss=constrained_loss,
        inside_feasible_region=inside_feasible_region,
        active_constraint=active_constraint,
        sparse=bool(zero_like_axes),
        zero_like_axes=zero_like_axes,
    )


def build_grid(plot_range: float, points_per_axis: int = 301) -> tuple[np.ndarray, np.ndarray]:
    """构造绘图网格。"""

    axis_values = np.linspace(-plot_range, plot_range, points_per_axis)
    return np.meshgrid(axis_values, axis_values)


def add_constraint_region(ax: plt.Axes, regularization_type: str, radius: float) -> None:
    """在图上绘制约束边界和浅色可行域。"""

    if regularization_type == "L1":
        diamond_points = np.array(
            [
                [radius, 0.0],
                [0.0, radius],
                [-radius, 0.0],
                [0.0, -radius],
            ]
        )
        polygon = patches.Polygon(
            diamond_points,
            closed=True,
            facecolor="#f8d7da",
            edgecolor="#c0392b",
            linewidth=2.0,
            alpha=0.35,
            label=r"L1 约束边界: $|w_1| + |w_2| = t$",
        )
        ax.add_patch(polygon)
    else:
        circle = patches.Circle(
            (0.0, 0.0),
            radius=radius,
            facecolor="#d6eaf8",
            edgecolor="#2874a6",
            linewidth=2.0,
            alpha=0.35,
            label=r"L2 约束边界: $w_1^2 + w_2^2 = t^2$",
        )
        ax.add_patch(circle)


def tick_step_for_range(plot_range: float) -> float:
    """根据图范围选择更细的坐标轴刻度。"""

    if plot_range <= 3.0:
        return 0.5
    if plot_range <= 5.0:
        return 0.5
    return 1.0


def plot_regularization_geometry(
    config: QuadraticLossConfig,
    radius: float,
    regularization_type: str,
    plot_range: float,
    result: OptimizationResult,
) -> plt.Figure:
    """绘制损失等高线、约束区域和两个最优点。"""

    configure_matplotlib_for_chinese()
    grid_w1, grid_w2 = build_grid(plot_range)
    loss_values = quadratic_loss(grid_w1, grid_w2, config)

    figure, axis = plt.subplots(figsize=(8, 8))
    contour_min = float(np.min(loss_values))
    contour_max = float(np.percentile(loss_values, 35))
    if contour_max <= contour_min + EPSILON:
        contour_max = contour_min + 1.0
    contour_levels = np.linspace(contour_min, contour_max, 10)
    axis.contour(grid_w1, grid_w2, loss_values, levels=contour_levels, cmap="viridis", linewidths=1.3)

    add_constraint_region(axis, regularization_type, radius)

    axis.scatter(
        [config.w1_star],
        [config.w2_star],
        color="#d35400",
        s=140,
        marker="*",
        label="原始无正则最优点",
        zorder=5,
    )
    axis.scatter(
        [result.point[0]],
        [result.point[1]],
        color="#1e8449",
        s=95,
        marker="o",
        edgecolor="white",
        linewidth=1.2,
        label="约束最优点",
        zorder=6,
    )

    axis.axhline(0.0, color="#7f8c8d", linewidth=1.0, alpha=0.8)
    axis.axvline(0.0, color="#7f8c8d", linewidth=1.0, alpha=0.8)
    axis.set_xlim(-plot_range, plot_range)
    axis.set_ylim(-plot_range, plot_range)
    axis.set_xlabel("w1")
    axis.set_ylabel("w2")
    axis.set_title("二维参数空间中的损失等高线与正则约束")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.set_aspect("equal", adjustable="box")
    tick_step = tick_step_for_range(plot_range)
    axis.xaxis.set_major_locator(MultipleLocator(tick_step))
    axis.yaxis.set_major_locator(MultipleLocator(tick_step))
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def format_point(point: tuple[float, float]) -> str:
    """用更适合教学展示的格式输出点坐标。"""

    return f"({point[0]:.3f}, {point[1]:.3f})"


def describe_feasibility(result: OptimizationResult, regularization_type: str) -> list[str]:
    """根据当前几何关系生成中文解释。"""

    lines: list[str] = []
    if result.inside_feasible_region:
        lines.append("原始最优点已经落在当前可行域内部，所以加入约束后，最优点并不需要移动。")
        lines.append("这说明当前的正则预算还比较宽松，约束没有真正把解推向边界。")
        return lines

    lines.append("原始最优点位于可行域外部，因此最优点被约束推回到了边界上。")
    lines.append("几何上可以理解为：不断扩张损失等高线，直到它第一次与约束边界相切。")

    if regularization_type == "L1":
        if result.sparse:
            zero_axis_text = "、".join(result.zero_like_axes)
            lines.append(f"当前约束最优点已经非常接近菱形角点，{zero_axis_text} 接近 0，因此表现出明显的稀疏性。")
            lines.append("这正是 L1 更容易产生稀疏解的几何原因：菱形边界在坐标轴方向存在尖角。")
        else:
            lines.append("当前点虽然落在菱形边界上，但还没有压到角点，因此稀疏性还不算最明显。")
            lines.append("继续减小 t，或者把原始最优点放得更靠近某条坐标轴，通常更容易看到某个参数被压到 0。")
    else:
        if not result.sparse:
            lines.append("当前约束最优点更像是两个参数一起缩小，而不是删除某一个参数。")
            lines.append("这对应了 L2 的典型效果：整体收缩，而不是直接变成稀疏解。")
        else:
            lines.append("当前点恰好很接近坐标轴，但这更多是原始最优点位置造成的，并不是 L2 本身偏好尖角稀疏。")

    return lines

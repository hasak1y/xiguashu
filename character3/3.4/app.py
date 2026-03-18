"""Streamlit UI for visualizing binary LDA in a teaching-oriented way."""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
except Exception:  # pragma: no cover - optional dependency for verification only
    LinearDiscriminantAnalysis = None


EPS = 1e-6


@dataclass
class CriterionResult:
    """Container for the Fisher criterion under one projection direction."""

    w: np.ndarray
    projected_class1: np.ndarray
    projected_class2: np.ndarray
    between_term: float
    within_term: float
    projected_mean_gap: float
    projected_within_spread: float
    score: float


def generate_data(
    seed: int,
    n_samples: int,
    mean1: np.ndarray,
    mean2: np.ndarray,
    std1: float,
    std2: float,
    allow_correlation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two 2D Gaussian classes.

    The goal is not to mimic a production dataset, but to give learners a
    controllable geometric playground for observing how LDA reacts.
    """
    rng = np.random.default_rng(seed)

    if allow_correlation:
        corr = 0.65
        cov1 = np.array(
            [[std1**2, corr * std1 * std1], [corr * std1 * std1, (std1 * 0.8) ** 2]]
        )
        cov2 = np.array(
            [[(std2 * 0.9) ** 2, -corr * std2 * std2], [-corr * std2 * std2, std2**2]]
        )
    else:
        cov1 = np.diag([std1**2, std1**2])
        cov2 = np.diag([std2**2, std2**2])

    x1 = rng.multivariate_normal(mean1, cov1, size=n_samples)
    x2 = rng.multivariate_normal(mean2, cov2, size=n_samples)
    return x1, x2


def compute_class_stats(class1: np.ndarray, class2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the mean vector for each class."""
    return class1.mean(axis=0), class2.mean(axis=0)


def compute_sw(class1: np.ndarray, class2: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compute within-class scatter matrix Sw.

    Sw measures how much samples spread around their own class center.
    Larger Sw means samples from the same class are more scattered.
    """
    centered1 = class1 - m1
    centered2 = class2 - m2
    return centered1.T @ centered1 + centered2.T @ centered2


def compute_sb(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compute between-class scatter matrix Sb.

    For binary LDA, Sb only depends on the distance between the two class means.
    """
    mean_diff = (m1 - m2).reshape(-1, 1)
    return mean_diff @ mean_diff.T


def direction_from_theta(theta_deg: float) -> np.ndarray:
    """Construct a unit direction vector from an angle in degrees."""
    theta_rad = np.deg2rad(theta_deg)
    w = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
    return w / (np.linalg.norm(w) + EPS)


def project_points(points: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Project 2D points onto a 1D axis defined by direction w."""
    unit_w = w / (np.linalg.norm(w) + EPS)
    return points @ unit_w


def compute_criterion(
    class1: np.ndarray,
    class2: np.ndarray,
    sb: np.ndarray,
    sw: np.ndarray,
    w: np.ndarray,
) -> CriterionResult:
    """Evaluate Fisher criterion J(w) for an arbitrary projection direction.

    J(w) = (w^T Sb w) / (w^T Sw w)
    The numerator rewards separation between class means.
    The denominator penalizes spread within each class.
    """
    unit_w = w / (np.linalg.norm(w) + EPS)
    y1 = project_points(class1, unit_w)
    y2 = project_points(class2, unit_w)

    between_term = float(unit_w.T @ sb @ unit_w)
    within_term = float(unit_w.T @ sw @ unit_w)
    within_term = max(within_term, EPS)
    projected_mean_gap = float(abs(y1.mean() - y2.mean()))
    projected_within_spread = float(y1.var(ddof=1) + y2.var(ddof=1))
    score = between_term / within_term

    return CriterionResult(
        w=unit_w,
        projected_class1=y1,
        projected_class2=y2,
        between_term=between_term,
        within_term=within_term,
        projected_mean_gap=projected_mean_gap,
        projected_within_spread=projected_within_spread,
        score=score,
    )


def compute_lda_direction(sw: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compute the binary LDA optimal direction w* ∝ Sw^{-1}(m1 - m2).

    A tiny diagonal regularizer is added for numerical stability when Sw is
    nearly singular, which often happens in interactive demos.
    """
    sw_reg = sw + EPS * np.eye(sw.shape[0])
    direction = np.linalg.solve(sw_reg, m1 - m2)
    norm = np.linalg.norm(direction)
    if norm < EPS:
        return np.array([1.0, 0.0])
    return direction / norm


def line_points_for_direction(w: np.ndarray, center: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Build endpoints for drawing a line with direction w through a chosen center."""
    unit_w = w / (np.linalg.norm(w) + EPS)
    p1 = center - scale * unit_w
    p2 = center + scale * unit_w
    return p1, p2


def projection_to_line(points: np.ndarray, w: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Return 2D coordinates of orthogonal projections onto a line.

    By default the line passes through the origin, matching the vector form y = Xw.
    """
    unit_w = w / (np.linalg.norm(w) + EPS)
    if center is None:
        center = np.zeros(points.shape[1], dtype=float)
    shifted = points - center
    scalar_proj = shifted @ unit_w
    return center + np.outer(scalar_proj, unit_w)


def plot_2d_data(
    class1: np.ndarray,
    class2: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    manual_w: np.ndarray,
    lda_w: np.ndarray,
    show_manual: bool,
    show_lda: bool,
    show_guides: bool,
    show_means: bool,
) -> plt.Figure:
    """Plot 2D samples, mean points, and projection directions."""
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.scatter(class1[:, 0], class1[:, 1], color="#1f77b4", alpha=0.75, label="类别 1 样本")
    ax.scatter(class2[:, 0], class2[:, 1], color="#d62728", alpha=0.75, label="类别 2 样本")

    combined = np.vstack([class1, class2, m1, m2])
    span = np.max(np.ptp(combined, axis=0)) * 0.8 + 1.0
    axis_center = np.zeros(2, dtype=float)

    if show_means:
        ax.scatter(m1[0], m1[1], color="#0b4f8a", marker="X", s=180, label="类别 1 均值")
        ax.scatter(m2[0], m2[1], color="#8a1111", marker="X", s=180, label="类别 2 均值")

    if show_manual:
        p1, p2 = line_points_for_direction(manual_w, axis_center, span)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#2ca02c", linewidth=2.5, label="当前手动方向")
        if show_guides:
            all_points = np.vstack([class1, class2])
            projected_points = projection_to_line(all_points, manual_w, axis_center)
            for point, projected in zip(all_points, projected_points):
                ax.plot(
                    [point[0], projected[0]],
                    [point[1], projected[1]],
                    color="#7f7f7f",
                    linewidth=0.7,
                    alpha=0.25,
                )

    if show_lda:
        p1, p2 = line_points_for_direction(lda_w, axis_center, span)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#ff7f0e", linewidth=2.5, linestyle="--", label="LDA 最优方向")

    ax.axhline(0, color="#dddddd", linewidth=1)
    ax.axvline(0, color="#dddddd", linewidth=1)
    ax.set_title("二维样本分布与投影方向")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    return fig


def plot_projection_1d(
    manual_result: CriterionResult,
    lda_result: CriterionResult,
    show_lda: bool,
) -> plt.Figure:
    """Plot projected 1D distributions for manual and optional LDA directions."""
    fig, axes = plt.subplots(2 if show_lda else 1, 1, figsize=(8.0, 5.5 if show_lda else 3.0), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    def draw_axis(ax: plt.Axes, result: CriterionResult, title: str) -> None:
        rng = np.random.default_rng(0)
        jitter1 = rng.uniform(-0.08, 0.08, size=result.projected_class1.shape[0])
        jitter2 = rng.uniform(-0.08, 0.08, size=result.projected_class2.shape[0])
        ax.scatter(result.projected_class1, jitter1, color="#1f77b4", alpha=0.75, label="类别 1 投影")
        ax.scatter(result.projected_class2, jitter2 + 0.22, color="#d62728", alpha=0.75, label="类别 2 投影")
        ax.axvline(result.projected_class1.mean(), color="#1f77b4", linestyle="--", linewidth=2, label="类别 1 投影均值")
        ax.axvline(result.projected_class2.mean(), color="#d62728", linestyle="--", linewidth=2, label="类别 2 投影均值")
        ax.set_yticks([])
        ax.set_title(title)
        ax.set_xlabel("投影到一维后的坐标")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right", fontsize=9)

    draw_axis(axes[0], manual_result, "当前手动方向下的一维投影")
    if show_lda:
        draw_axis(axes[1], lda_result, "LDA 最优方向下的一维投影")

    fig.tight_layout()
    return fig


def plot_fisher_curve(
    class1: np.ndarray,
    class2: np.ndarray,
    sb: np.ndarray,
    sw: np.ndarray,
    current_theta: float,
    lda_w: np.ndarray,
) -> tuple[plt.Figure, float]:
    """Plot J(w) over theta to show why direction choice matters."""
    theta_grid = np.linspace(0.0, 180.0, 361)
    scores = []
    for theta in theta_grid:
        w = direction_from_theta(theta)
        scores.append(compute_criterion(class1, class2, sb, sw, w).score)
    scores = np.array(scores)

    lda_theta = math.degrees(math.atan2(lda_w[1], lda_w[0]))
    if lda_theta < 0:
        lda_theta += 180.0
    if lda_theta > 180.0:
        lda_theta -= 180.0

    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(theta_grid, scores, color="#6a3d9a", linewidth=2, label="Fisher 准则 J(w)")
    current_score = compute_criterion(class1, class2, sb, sw, direction_from_theta(current_theta)).score
    lda_score = compute_criterion(class1, class2, sb, sw, lda_w).score
    ax.scatter([current_theta], [current_score], color="#2ca02c", s=60, label="当前 theta")
    ax.scatter([lda_theta], [lda_score], color="#ff7f0e", s=60, label="LDA 最优 theta")
    ax.set_xlabel("theta（度）")
    ax.set_ylabel("J(w)")
    ax.set_title("Fisher 准则随方向变化")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, lda_theta


def format_vector(vec: np.ndarray) -> str:
    """Pretty-print a vector for markdown display."""
    return np.array2string(vec, precision=3, suppress_small=True)


def format_matrix(matrix: np.ndarray) -> str:
    """Pretty-print a matrix for markdown display."""
    return np.array2string(matrix, precision=3, suppress_small=True)


def teaching_message(manual_result: CriterionResult, lda_result: CriterionResult) -> str:
    """Generate a short natural-language explanation for the current state."""
    gap_ratio = manual_result.projected_mean_gap / (lda_result.projected_mean_gap + EPS)
    spread_ratio = manual_result.projected_within_spread / (lda_result.projected_within_spread + EPS)

    if manual_result.score >= lda_result.score * 0.95:
        return (
            "当前手动方向已经接近 LDA 最优方向。你看到的现象是：两类中心在这条线上已经被拉开，"
            "同时类内投影没有明显变散，所以 Fisher 准则也接近最优。"
        )
    if gap_ratio < 0.8:
        return (
            "当前方向的问题主要是没有把两类中心充分拉开。虽然样本被压到了一条线上，"
            "但两类投影均值靠得太近，所以分类信息被浪费了。"
        )
    if spread_ratio > 1.15:
        return (
            "当前方向虽然能看到一定的类间距离，但同类样本在一维上仍然拉得比较散。"
            "这说明类内散度偏大，J(w) 会被分母拖低。"
        )
    return (
        "当前方向和最优方向之间仍有差距。LDA 更优，是因为它同时兼顾了“拉开类中心”和“压缩类内分散”这两件事。"
    )


def sklearn_reference_direction(class1: np.ndarray, class2: np.ndarray) -> np.ndarray | None:
    """Use sklearn LDA as an optional verification reference."""
    if LinearDiscriminantAnalysis is None:
        return None
    x = np.vstack([class1, class2])
    y = np.array([0] * len(class1) + [1] * len(class2))
    model = LinearDiscriminantAnalysis(n_components=1)
    model.fit(x, y)
    coef = model.coef_.ravel()
    norm = np.linalg.norm(coef)
    if norm < EPS:
        return None
    return coef / norm


def render_numeric_block(
    m1: np.ndarray,
    m2: np.ndarray,
    sw: np.ndarray,
    sb: np.ndarray,
    manual_result: CriterionResult,
    lda_result: CriterionResult,
) -> None:
    """Render the core numeric explanations with formulas and plain-language notes."""
    st.subheader("4）数值解释区")
    st.latex(r"J(w) = \frac{w^T S_b w}{w^T S_w w}")
    st.markdown("这里的核心思想是：**分子希望大，分母希望小**。也就是让两类中心投影后尽量远，同时让同类样本尽量紧。")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**两类均值向量**  \n`m1 = {format_vector(m1)}`  \n通俗理解：类别 1 的“中心位置”。")
        st.markdown(f"**两类均值向量**  \n`m2 = {format_vector(m2)}`  \n通俗理解：类别 2 的“中心位置”。")
        st.markdown(f"**类内散度矩阵 Sw**  \n```text\n{format_matrix(sw)}\n```  \nSw 越大，说明同类样本整体越松散。")
        st.markdown(f"**类间散度矩阵 Sb**  \n```text\n{format_matrix(sb)}\n```  \nSb 越大，说明两类中心之间的差异越明显。")

    with col2:
        st.markdown(f"**当前方向向量 w**  \n`w = {format_vector(manual_result.w)}`  \n它决定你把二维样本往哪条线压缩。")
        st.markdown(f"**当前方向下的类间距离**  \n`{manual_result.projected_mean_gap:.4f}`  \n值越大，表示两类投影均值越远。")
        st.markdown(f"**当前方向下的类内离散程度**  \n`{manual_result.projected_within_spread:.4f}`  \n值越大，表示同类样本在一维上越散。")
        st.markdown(f"**当前方向下的 J(w)**  \n`{manual_result.score:.6f}`  \n当前方向的综合判别效果。")
        st.markdown(f"**LDA 最优方向的 J(w)**  \n`{lda_result.score:.6f}`  \n二分类里它对应 `S_w^{{-1}}(m_1-m_2)` 方向。")


def render_principle_explanation(show_extra_text: bool) -> None:
    """Render the learning-oriented explanation section."""
    st.subheader("5）原理说明区")
    st.markdown(
        """
LDA 不是随便找一条线把样本压扁，而是在找一条**最有利于区分类别**的方向。

- **类内散度**：看同一类样本是不是挤在一起。如果一类点本来就很散，那么投影后也容易互相混在一起。
- **类间散度**：看两类中心是不是离得远。如果两类中心投影后很接近，就算你做了降维，也不利于分类。
- **LDA 的核心目标**：让“类间距离尽量大”，同时让“类内分散尽量小”。
- **二分类最优方向**：可以证明最优方向和 `S_w^{-1}(m_1-m_2)` 有关。直觉上，先看两类中心差在哪，再结合类内噪声的方向做修正。
- **为什么是广义瑞利商**：因为我们想最大化“分子=拉开程度”和“分母=松散程度”的比值，这个比值正好就是广义瑞利商。
"""
    )
    if show_extra_text:
        st.info(
            "观察技巧：拖动 theta 时，重点看两个变化。第一，两个类别在一维上是否更分开；第二，每一类内部是否更紧。"
            " 这两者同时改善时，J(w) 才会显著增大。"
        )


def render_comparison_experiment(manual_result: CriterionResult, lda_result: CriterionResult) -> None:
    """Render side-by-side comparison between manual and optimal direction."""
    st.subheader("6）对比实验区")
    left, right = st.columns(2)

    with left:
        st.markdown("**当前手动方向**")
        st.metric("J(w)", f"{manual_result.score:.6f}")
        st.write(f"投影后类间距离：`{manual_result.projected_mean_gap:.4f}`")
        st.write(f"投影后类内离散：`{manual_result.projected_within_spread:.4f}`")

    with right:
        st.markdown("**LDA 最优方向**")
        st.metric("J(w)", f"{lda_result.score:.6f}")
        st.write(f"投影后类间距离：`{lda_result.projected_mean_gap:.4f}`")
        st.write(f"投影后类内离散：`{lda_result.projected_within_spread:.4f}`")

    better = "LDA 最优方向" if lda_result.score >= manual_result.score else "当前手动方向"
    st.success(f"当前更优的方向：**{better}**")
    st.write(teaching_message(manual_result, lda_result))


def main() -> None:
    """Build the full Streamlit teaching app."""
    st.set_page_config(page_title="LDA 线性判别分析可视化实验", layout="wide")
    st.title("LDA 线性判别分析可视化实验")
    st.caption("目标不是只给出答案，而是让你看清：LDA 为什么要找一个方向，以及这个方向如何影响投影后的可分性。")

    st.sidebar.header("1）参数控制区")
    seed = st.sidebar.number_input("随机种子", min_value=0, max_value=9999, value=42, step=1)
    n_samples = st.sidebar.slider("每类样本数", min_value=20, max_value=300, value=80, step=10)

    st.sidebar.markdown("**类别 1 均值**")
    mu1_x = st.sidebar.slider("mu1_x", -5.0, 5.0, -1.5, 0.1)
    mu1_y = st.sidebar.slider("mu1_y", -5.0, 5.0, 0.5, 0.1)

    st.sidebar.markdown("**类别 2 均值**")
    mu2_x = st.sidebar.slider("mu2_x", -5.0, 5.0, 1.8, 0.1)
    mu2_y = st.sidebar.slider("mu2_y", -5.0, 5.0, -0.3, 0.1)

    std1 = st.sidebar.slider("类别 1 标准差", 0.2, 3.0, 1.0, 0.1)
    std2 = st.sidebar.slider("类别 2 标准差", 0.2, 3.0, 1.2, 0.1)
    allow_correlation = st.sidebar.checkbox("两类样本允许相关性", value=False)
    theta = st.sidebar.slider("手动投影方向角度 theta（度）", 0, 180, 35, 1)
    show_lda = st.sidebar.checkbox("显示 LDA 自动求得的最优方向", value=True)
    show_manual = st.sidebar.checkbox("显示当前手动方向", value=True)
    show_guides = st.sidebar.checkbox("显示投影辅助线", value=True)
    show_means = st.sidebar.checkbox("显示类均值点", value=True)
    show_explanations = st.sidebar.checkbox("显示类内/类间解释文字", value=True)
    show_fisher_curve = st.sidebar.checkbox("显示 Fisher 准则随 theta 变化曲线", value=True)
    show_sklearn_reference = st.sidebar.checkbox("显示 sklearn LDA 对照结果", value=False)

    if st.sidebar.button("重新生成数据"):
        # Streamlit 每次交互本来就会重跑，这里手动刷新是为了符合教学操作直觉。
        st.rerun()

    mean1 = np.array([mu1_x, mu1_y], dtype=float)
    mean2 = np.array([mu2_x, mu2_y], dtype=float)
    class1, class2 = generate_data(seed, n_samples, mean1, mean2, std1, std2, allow_correlation)
    m1, m2 = compute_class_stats(class1, class2)
    sw = compute_sw(class1, class2, m1, m2)
    sb = compute_sb(m1, m2)
    manual_w = direction_from_theta(theta)
    lda_w = compute_lda_direction(sw, m1, m2)

    manual_result = compute_criterion(class1, class2, sb, sw, manual_w)
    lda_result = compute_criterion(class1, class2, sb, sw, lda_w)

    st.subheader("2）二维样本分布图")
    st.markdown(
        "这里重点看两件事：第一，两类样本本身在平面里如何分布；第二，同一批点沿不同方向投影时，"
        "是不是更容易在一条线上被区分开。灰色辅助线表示“把二维点压到一维轴上”的过程。"
    )
    fig_2d = plot_2d_data(class1, class2, m1, m2, manual_w, lda_w, show_manual, show_lda, show_guides, show_means)
    st.pyplot(fig_2d, clear_figure=True)

    st.subheader("3）投影后一维分布图")
    st.markdown(
        "这部分直接回答一个关键问题：**同样的数据，换一条投影方向，一维上的分离程度会不会明显不同？**"
    )
    fig_proj = plot_projection_1d(manual_result, lda_result, show_lda)
    st.pyplot(fig_proj, clear_figure=True)

    render_numeric_block(m1, m2, sw, sb, manual_result, lda_result)
    render_principle_explanation(show_explanations)

    if show_fisher_curve:
        st.subheader("额外增强：Fisher 准则随 theta 变化曲线")
        st.markdown(
            "如果你还不确定“为什么 LDA 要找方向”，看这张图最直接：不同角度对应不同的 `J(w)`，说明并不是任何方向都一样好。"
        )
        fisher_fig, lda_theta = plot_fisher_curve(class1, class2, sb, sw, float(theta), lda_w)
        st.pyplot(fisher_fig, clear_figure=True)
        st.write(f"LDA 最优方向对应的近似角度为：`{lda_theta:.2f}°`。")

    if show_sklearn_reference:
        st.subheader("额外增强：sklearn LDA 对照")
        ref_w = sklearn_reference_direction(class1, class2)
        if ref_w is None:
            st.warning("当前环境没有可用的 sklearn，对照结果无法显示。")
        else:
            ref_result = compute_criterion(class1, class2, sb, sw, ref_w)
            st.write(f"手写 LDA 方向：`{format_vector(lda_w)}`")
            st.write(f"sklearn LDA 方向：`{format_vector(ref_w)}`")
            st.write(
                f"手写 LDA 的 J(w)：`{lda_result.score:.6f}`，sklearn 方向的 J(w)：`{ref_result.score:.6f}`。"
                " 如果只是方向正负相反，也属于同一条判别轴。"
            )

    st.subheader("教学模式提示")
    st.info(teaching_message(manual_result, lda_result))

    render_comparison_experiment(manual_result, lda_result)

    if show_explanations and lda_result.score < 0.1:
        st.warning(
            "当前两类重叠比较严重。即使 LDA 找到了最优方向，这个方向也只能在“已有信息”里尽量做得更好，"
            "但无法凭空创造出明显可分性。"
        )


if __name__ == "__main__":
    main()

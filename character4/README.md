# 决策树教学可视化工具

这是一个基于 Streamlit 的教学演示项目，用来帮助理解《机器学习（周志华）》第四章中的决策树。

项目重点支持：

- ID3：信息增益
- C4.5：信息增益率
- CART：基尼指数与二叉划分
- 逐步建树
- 预剪枝演示
- 后剪枝演示
- 西瓜数据集与 Play Tennis 示例数据集

## 项目结构

```text
character4/
├─ app.py
├─ builder.py
├─ datasets.py
├─ metrics.py
├─ pruning.py
├─ requirements.txt
├─ README.md
├─ splitters.py
├─ tree_node.py
├─ utils.py
└─ visualization.py
```

## 功能说明

- 左侧控制栏可切换算法、数据集、目标列、特征列、剪枝模式与停止条件。
- 支持 `Next Step` 单步推进建树过程。
- 支持 `Build Full Tree` 一次生成完整树。
- 会展示当前节点样本、类别分布、候选特征评分表、分裂结果和日志时间线。
- 若启用预剪枝，会显示“不分裂”和“尝试分裂”的验证集准确率对比。
- 若启用后剪枝，可在完整树生成后逐步检查剪枝点，并展示剪枝前后对比。
- 支持导出当前树结构为 JSON。

## 安装依赖

建议先创建虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

## 启动方式

在 `character4` 目录下运行：

```bash
streamlit run app.py
```

## 使用建议

- 首次演示建议选择“西瓜数据集 2.0（离散版）”。
- 若重点讲信息熵，优先使用 ID3。
- 若重点讲“为什么不能只看信息增益”，切换到 C4.5。
- 若重点讲二叉划分和剪枝，可选择 CART 或后剪枝模式。

## 说明

- 本项目核心划分逻辑由代码自行实现，没有直接调用 sklearn 的现成决策树训练器。
- 项目更偏向“教学可视化”而不是工业级建模。

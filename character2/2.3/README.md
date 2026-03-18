# 西瓜书 2.3 性能度量可视化

这是一个基于 Streamlit 的教学型小应用，用来直观演示《机器学习（西瓜书）》第 2 章 2.3 节中常见二分类性能指标的意义与变化关系。

应用重点展示：

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- AUC
- 混淆矩阵（TP / FP / TN / FN）

## 功能说明

应用支持多种人工生成的数据场景：

- 场景 A：类别平衡
- 场景 B：类别不平衡
- 场景 C：模型效果较好
- 场景 D：模型效果一般
- 场景 E：模型效果很差
- 场景 F：自定义参数生成

用户可以在侧边栏实时调节：

- 总样本数
- 正类比例
- 正类分数均值
- 负类分数均值
- 分数分布标准差
- 分类阈值 threshold
- 随机种子

主界面会同步更新：

- 当前样本概览
- 混淆矩阵和 TP / FP / TN / FN
- Accuracy / Precision / Recall / F1 / AUC
- 分数分布图
- ROC 曲线
- 指标-阈值变化曲线
- 动态中文解读
- 部分样本预览

## 运行方式

先安装依赖：

```bash
pip install -r character2/2.3/requirements.txt
```

然后运行：

```bash
streamlit run character2/2.3/app.py
```

## 教学建议

推荐重点演示以下几个问题：

1. 在类别不平衡时，为什么 Accuracy 可能有误导性。
2. 为什么阈值降低时 Recall 常常上升，而 Precision 可能下降。
3. 为什么 F1 可以作为 Precision 和 Recall 的平衡指标。
4. ROC 曲线反映的是什么，为什么 AUC 不依赖单一阈值。
5. 为什么同一个模型在不同 threshold 下会表现出不同的分类结果。

## 文件说明

- `app.py`：Streamlit 页面入口与布局
- `utils.py`：数据生成、指标计算、绘图和动态解读
- `requirements.txt`：依赖列表

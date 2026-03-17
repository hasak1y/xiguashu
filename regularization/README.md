# L1 / L2 正则化几何可视化

这是一个基于 Streamlit 的教学演示小项目，用二维参数空间中的图形帮助理解 L1 正则化和 L2 正则化的几何含义。

页面会同时展示：

- 原始损失函数 `J(w1, w2)` 的等高线
- 无正则时的最优点
- L1 约束对应的菱形可行域，或 L2 约束对应的圆形可行域
- 加入约束之后的最优点
- 当前情形下的中文解释与对比结论

## 项目作用

这个项目适合用来理解《西瓜书》中与正则化相关的直观图像：

- 为什么 L1 约束是菱形
- 为什么 L2 约束是圆
- 为什么 L1 更容易产生稀疏解
- 为什么 L2 更倾向于整体收缩，而不是直接把参数压成 0

## 安装依赖

在项目根目录执行：

```bash
pip install -r xiguashu/regularization/requirements.txt
```

如果你已经位于 `xiguashu` 目录，也可以执行：

```bash
pip install -r regularization/requirements.txt
```

## 启动方式

如果你当前就在 `regularization` 目录下，直接运行：

```bash
streamlit run app.py
```

如果你在上一级 `xiguashu` 目录下，运行：

```bash
streamlit run regularization/app.py
```

在项目根目录执行：

```bash
streamlit run xiguashu/regularization/app.py
```

如果你已经位于 `xiguashu` 目录，也可以执行：

```bash
streamlit run regularization/app.py
```

## 页面主要功能

- 侧边栏切换 `L1` / `L2` 正则类型
- 调整无正则最优点 `(w1_star, w2_star)`
- 调整损失函数中 `a`、`b`，观察等高线如何变成不同方向的椭圆
- 调整约束半径 `t`
- 展示约束最优点的数值，并判断是否可视为稀疏解
- 提供默认参数重置按钮与 3 个典型示例按钮
- 提供“数学说明”折叠区，简要说明惩罚形式与约束形式的联系

## 几何解释简述

项目默认使用如下原始损失函数：

```text
J(w1, w2) = a * (w1 - w1_star)^2 + b * (w2 - w2_star)^2
```

它的等高线是椭圆。

- L1 约束：`|w1| + |w2| <= t`
  对应的边界是菱形，四个尖角落在坐标轴上，所以更容易在角点处出现某个参数等于 0 的情况。

- L2 约束：`w1^2 + w2^2 <= t^2`
  对应的边界是圆，边界平滑，所以更常见的是两个参数一起变小，而不是某一个被直接压成 0。

## 文件说明

```text
regularization/
├── app.py
├── README.md
├── requirements.txt
├── utils.py
└── assets/
```

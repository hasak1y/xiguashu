# 多项式拟合与正则化可视化实验

这是一个适合初学者的 Streamlit 小项目，用来可视化演示机器学习中的：

- 经验误差（训练误差）
- 测试误差
- 欠拟合与过拟合
- L1 正则化与 L2 正则化

你可以一边调参数，一边观察拟合曲线、误差指标和模型参数变化。

## 项目结构

```text
2.1/
├─ app.py
├─ utils/
│  ├─ data.py
│  ├─ features.py
│  ├─ model.py
│  ├─ metrics.py
│  └─ __init__.py
├─ requirements.txt
└─ README.md
```

## 功能说明

- 支持数据集：
  - `sin(x)`
  - `cos(x)`
  - `sin(x) + cos(x)`
  - `x^2`
  - `x^3 - x`
  - `custom mix`
- 可调参数：
  - 训练集样本数
  - 测试集样本数
  - 噪声强度，范围 `0 ~ 3`
  - 随机种子
  - 多项式次数 `degree`
  - 是否标准化 `x`
  - 正则化类型 `None / L1 / L2`
  - 正则化系数 `lambda`
- 页面展示：
  - 真实函数曲线
  - 训练样本散点
  - 测试集真实点
  - 当前模型拟合曲线
  - 训练集 MSE 和测试集 MSE
  - 权重向量
  - 自动解释区
  - 一键生成误差-复杂度曲线

## 核心概念

### degree 是什么

`degree` 表示多项式最高次数。

例如：

- `degree = 1`：`w0 + w1*x`
- `degree = 2`：`w0 + w1*x + w2*x^2`
- `degree = 5`：继续加入 `x^3, x^4, x^5`

次数越高，模型越复杂。

### 为什么 degree 太低会欠拟合

当真实规律比较复杂，而模型次数太低时，模型表达能力不够。

常见现象：

- 训练误差高
- 测试误差也高
- 曲线过于简单

### 为什么 degree 太高会过拟合

次数太高时，模型可能不只学习真实规律，还把训练数据中的噪声一起学进去。

常见现象：

- 训练误差很低
- 测试误差反而变高
- 曲线出现不自然摆动

### 训练误差和测试误差的区别

- 训练误差：模型在训练集上的误差，也叫经验误差
- 测试误差：模型在测试集上的误差，用来近似衡量泛化能力

如果训练误差很低但测试误差明显更高，通常表示泛化能力不足。

### L1 和 L2 正则化有什么区别

L1 正则化目标：

```text
min ||Xw - y||^2 + lambda * sum(|w|)
```

L2 正则化目标：

```text
min ||Xw - y||^2 + lambda * ||w||^2
```

区别可以简单理解为：

- L1 倾向于把一部分参数直接压成 0，让模型更稀疏
- L2 倾向于把参数整体压小，让曲线更平滑

本项目默认不对截距项进行正则化。

## 安装与运行

进入目录：

```powershell
cd D:\CODE\PythonProject\xiguashu\character2\2.1
```

安装依赖：

```powershell
C:/Users/24044/.conda/envs/xiguashu/python.exe -m pip install -r requirements.txt
```

启动应用：

```powershell
C:/Users/24044/.conda/envs/xiguashu/python.exe -m streamlit run app.py
```

如果你已经激活 `xiguashu` conda 环境，也可以直接运行：

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## 代码说明

- `utils/data.py`：生成不同真实函数数据
- `utils/features.py`：构造多项式特征和标准化
- `utils/model.py`：实现普通最小二乘、L1 和 L2 正则化
- `utils/metrics.py`：实现 MSE 和自动解释
- `app.py`：Streamlit 页面主程序

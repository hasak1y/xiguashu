"""
经验误差：模型在训练集上的误差
泛化误差：模型在新样本上的误差
过拟合：训练误差很小，但测试误差变大
"""
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 生成一份“真实世界”数据
# =========================
np.random.seed(42)

# 真实函数：y = sin(x)
def true_function(x):
    return np.sin(x)

# 训练集：样本少，而且带噪声
x_train = np.linspace(0, 2 * np.pi, 12)
y_train = true_function(x_train) + np.random.normal(0, 0.2, size=x_train.shape)

# 测试集：更密、更接近“未来新样本”
x_test = np.linspace(0, 2 * np.pi, 200)
y_test = true_function(x_test)

# =========================
# 2. 构造多项式特征
# =========================
def make_poly_features(x, degree):
    """
    输入:
        x: shape (n,)
        degree: 多项式阶数
    输出:
        X: shape (n, degree+1)
    例如 degree=3 时:
        [1, x, x^2, x^3]
    """
    x = np.asarray(x)
    X = np.vstack([x ** i for i in range(degree + 1)]).T
    return X

# =========================
# 3. 用最小二乘法训练
# =========================
def fit_polynomial(x, y, degree):
    """
    求解 w = (X^T X)^(-1) X^T y
    更稳定的写法用 np.linalg.lstsq
    """
    X = make_poly_features(x, degree)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

def predict(x, w):
    degree = len(w) - 1
    X = make_poly_features(x, degree)
    return X @ w

# =========================
# 4. 定义均方误差 MSE
# =========================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# =========================
# 5. 比较不同复杂度模型
# =========================
degrees = [1, 3, 9]
results = []

plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees, 1):
    w = fit_polynomial(x_train, y_train, degree)

    y_train_pred = predict(x_train, w)
    y_test_pred = predict(x_test, w)

    train_error = mse(y_train, y_train_pred)
    test_error = mse(y_test, y_test_pred)

    results.append((degree, train_error, test_error))

    plt.subplot(2, 2, i)
    plt.scatter(x_train, y_train, label="train data")
    plt.plot(x_test, y_test, label="true sin(x)")
    plt.plot(x_test, y_test_pred, label=f"degree={degree}")
    plt.title(
        f"degree={degree}\ntrain_mse={train_error:.4f}, test_mse={test_error:.4f}"
    )
    plt.legend()

# 第4个子图显示误差对比
plt.subplot(2, 2, 4)
train_errors = [r[1] for r in results]
test_errors = [r[2] for r in results]
labels = [str(r[0]) for r in results]

x_pos = np.arange(len(labels))
width = 0.35

plt.bar(x_pos - width/2, train_errors, width, label="train error")
plt.bar(x_pos + width/2, test_errors, width, label="test error")
plt.xticks(x_pos, labels)
plt.xlabel("polynomial degree")
plt.ylabel("MSE")
plt.title("Train Error vs Test Error")
plt.legend()

plt.tight_layout()
plt.show()

# =========================
# 6. 打印结果
# =========================
print("不同模型复杂度下的误差：")
for degree, train_error, test_error in results:
    print(
        f"degree={degree:2d} | 训练误差(经验误差)={train_error:.6f} | 测试误差(近似泛化误差)={test_error:.6f}"
    )
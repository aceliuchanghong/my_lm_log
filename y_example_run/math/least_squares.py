import numpy as np

# 生成模拟数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加截距项 x0 = 1
X_b = np.c_[np.ones((100, 1)), X]

# 最小二乘法求解回归系数
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("回归系数:", theta_best)

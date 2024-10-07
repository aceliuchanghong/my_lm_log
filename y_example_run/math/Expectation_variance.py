import numpy as np

# 生成一组随机变量
data = np.array([1, 2, 3, 4, 5])

# 计算期望值
expectation = np.mean(data)

# 计算方差
variance = np.var(data)

# 生成两个变量的数据
np.random.seed(42)
x = np.random.normal(0, 1, 100)  # 均值为0，方差为1
y = 2 * x + np.random.normal(0, 1, 100)  # 另一个与x相关的变量

# 计算样本协方差
cov_matrix = np.cov(x, y)
cov_xy = cov_matrix[0, 1]  # 取协方差矩阵中的协方差值

print("期望值: ", expectation)
print("方差: ", variance)
print(f"X和Y之间的协方差: {cov_xy}")

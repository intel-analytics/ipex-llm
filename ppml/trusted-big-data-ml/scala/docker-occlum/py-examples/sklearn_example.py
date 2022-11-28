import numpy as np
from sklearn_example.linear_model import LinearRegression
from sklearn_example.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

# 生成测试数据:
nSample = 100
x = np.linspace(0, 10, nSample)  # 起点为 0，终点为 10，均分为 nSample个点
e = np.random.normal(size=len(x))  # 正态分布随机数
y = 2.36 + 1.58 * x + e  # y = b0 + b1*x1

# 按照模型要求进行数据转换：输入是 array类型的 n*m 矩阵，输出是 array类型的 n*1 数组
x = x.reshape(-1, 1)  # 输入转换为 n行 1列（多元回归则为多列）的二维数组
y = y.reshape(-1, 1)  # 输出转换为 n行1列的二维数组
# print(x.shape,y.shape)

# 一元线性回归：最小二乘法(OLS)
modelRegL = LinearRegression()  # 创建线性回归模型
modelRegL.fit(x, y)  # 模型训练：数据拟合
yFit = modelRegL.predict(x)  # 用回归模型来预测输出

# 输出回归结果 XUPT
print('回归截距: w0={}'.format(modelRegL.intercept_))  # w0: 截距
print('回归系数: w1={}'.format(modelRegL.coef_))  # w1,..wm: 回归系数

# 回归模型的评价指标 YouCans
print('R2 确定系数：{:.4f}'.format(modelRegL.score(x, y)))  # R2 判定系数
print('均方误差：{:.4f}'.format(mean_squared_error(y, yFit)))  # MSE 均方误差
print('平均绝对值误差：{:.4f}'.format(mean_absolute_error(y, yFit)))  # MAE 平均绝对误差
print('中位绝对值误差：{:.4f}'.format(median_absolute_error(y, yFit)))  # 中值绝对误差
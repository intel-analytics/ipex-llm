import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error

# Generate test data:
nSample = 100
x = np.linspace(0, 10, nSample)
e = np.random.normal(size=len(x))
y = 2.36 + 1.58 * x + e  # y = b0 + b1*x1

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
# print(x.shape,y.shape)

# OLS
modelRegL = LinearRegression()
modelRegL.fit(x, y)
yFit = modelRegL.predict(x)

print('intercept: w0={}'.format(modelRegL.intercept_))
print('coef: w1={}'.format(modelRegL.coef_))

print('R2_score ：{:.4f}'.format(modelRegL.score(x, y)))
print('mean_squared_error：{:.4f}'.format(mean_squared_error(y, yFit)))
print('mean_absolute_error：{:.4f}'.format(mean_absolute_error(y, yFit)))
print('median_absolute_error：{:.4f}'.format(median_absolute_error(y, yFit)))

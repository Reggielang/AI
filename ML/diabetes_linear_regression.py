import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#加载糖尿病数据集
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

#将数据集拆分为训练集合测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

#创建一个多元线性回归算法对象
lr = LinearRegression()

#使用训练集训练模型
lr.fit(train_X, train_y)

#使用测试集进行预测
y_pred_train = lr.predict(train_X)
y_pred_test = lr.predict(test_X)

#计算均方误差
mse_train = mean_squared_error(train_y, y_pred_train)
mse_pred = mean_squared_error(test_y, y_pred_test)

print("Mean Squared Error: %.3f" % mse_train)
print("Mean Squared Error: %.3f" % mse_pred)

print(X)
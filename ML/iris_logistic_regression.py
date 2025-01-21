from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

#将数据集拆分为训练集合测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(y_train)

#创建一个逻辑回归对象, 逻辑回归回根据我们的数据决定是二分类还是多分类 -- 逻辑回归到底是多分类转成了多个二分类，还是使用了Softmax回归呢
# lr = LogisticRegression(multi_class='auto') #多分类转成了多个二分类
# lr = LogisticRegression(multi_class='multinomial') #Softmax回归做多分类

lr = LogisticRegression()

#使用训练集训练模型
lr.fit(X_train, y_train)

#使用测试集进行预测
y_pred_test = lr.predict(X_test)

#计算准确率
accuracy = accuracy_score(y_test, y_pred_test)
print("准确率: %.2f" % accuracy)

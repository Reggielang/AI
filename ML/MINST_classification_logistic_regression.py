import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



#载入 MINST 手写图片数据集
mnist = fetch_openml("mnist_784")


img0 = np.array(mnist.data)[0]
print(np.array(mnist.target)[0])

img0 = img0.reshape(28,28)
plt.imshow(img0,cmap='gray')
plt.show()

#数据预处理 ：标准归一化

scaler = StandardScaler()
X = scaler.fit_transform(mnist.data)

#划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, mnist.target, test_size=0.2, random_state=42)

#创建逻辑回归模型
model = LogisticRegression(max_iter=1000)

#在训练集上训练模型
model.fit(X_train, y_train)

#在测试集上进行预测
y_pred = model.predict(X_test)

#计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


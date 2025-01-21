import  numpy as np

from sklearn.cluster import KMeans

#定义数据集
X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])

#定义kmeans算法
kmeans = KMeans(n_clusters=2, random_state=0)

#训练模型
kmeans.fit(X)

#输出聚类结果
print(kmeans.labels_)
print(kmeans.cluster_centers_)


import numpy as np
import matplotlib.pyplot as plt

#实现kmeans算法,不依赖机器学习库
def kmeans(X,k,max_iter=100):
    #随机初始化K个聚类中心
    center = X[np.random.choice(X.shape[0],k,replace=False)]

    #初始化聚类结果 -- 存放kmeans给样本大的标签
    labels = np.zeros(X.shape[0])

    for i in range(max_iter):
        #计算每个样本到K个聚类中心的距离 -- 欧式距离
        distances = np.sqrt(((X - center[:,np.newaxis])**2).sum(axis=2))
        #找到每个样本到哪个聚类中心更近
        new_labels = np.argmin(distances,axis=0)

        #更新中心
        for j in range(k):
            center[j] = X[new_labels == j].mean(axis=0)

        #如果聚类结果没有变化，则结束循环
        if (labels == new_labels).all():
            break
        else:
            labels = new_labels

    return labels,center


#生成数据集
X = np.vstack((np.random.randn(100,2) *0.75+np.array([1,0]),
              np.random.randn(100,2) *0.25+np.array([-0.5,0.5]),
              np.random.randn(100,2) *0.5+np.array([-0.5,-0.5])))
k = 3
#调用kmeans算法
labels,center = kmeans(X,k)
print(labels)
print(center)

#可视化结果
plt.scatter(X[:,0],X[:,1],c=labels)
plt.scatter(center[:,0],center[:,1],marker='*',c='r',linewidths=3,s=200)
plt.show()
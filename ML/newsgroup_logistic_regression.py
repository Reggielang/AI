from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

#加载20newsgroups数据集
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# X = newsgroups.data
# y = newsgroups.target

#

# 创建一个pipeline, 用于文件特征提取，接着使用逻辑回归
pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=3000))

# 使用训练集训练模型
pipeline.fit(newsgroups_train.data, newsgroups_train.target)

# 使用测试集进行预测
y_pred_test = pipeline.predict(newsgroups_test.data)

# 输出准确率
print("准确率: %.2f" % accuracy_score(newsgroups_test.target, y_pred_test))

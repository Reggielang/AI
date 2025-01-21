
from sklearn.feature_extraction.text import CountVectorizer

data = [
    "I love machine learning.",
    "I love natural language processing love.",
    "I love programming.",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)
print(vectorizer.get_feature_names_out())
print(X.toarray())




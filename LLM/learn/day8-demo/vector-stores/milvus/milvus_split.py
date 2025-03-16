# pip install --upgrade --quiet  pymilvus
import os
from langchain_community.vectorstores import Zilliz
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 检索zilliz云数据库collection_1
vector_db = Zilliz(
    embeddings,
    collection_name="collection_1",
    connection_args={"uri": os.getenv("MILVUS_API_URL"), "token": os.getenv("MILVUS_API_KEY")},

)

query = "Pixar公司是做什么的?"
docs = vector_db.similarity_search(query)
print(docs[0].page_content)

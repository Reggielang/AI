# pip install langchain-chroma
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
# pip install -U langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# 加载文档并将其分割成片段
loader = TextLoader("../../resource/knowledge.txt", encoding="UTF-8")
documents = loader.load()
# 将其分割成片段
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# 创建开源嵌入函数
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 将其加载到 Chroma 内存中
db = Chroma.from_documents(docs, embedding_function)
# 进行查询
query = "Pixar公司是做什么的?"
#返回的距离分数是余弦距离。因此，分数越低越好。
# docs = db.similarity_search_with_score(query)
# # 打印结果
# print(docs[0])
#使用mmr进行相似性搜索。
retriever = db.as_retriever(search_type="mmr")
result = retriever.invoke(query)
print(result[0])

# 筛选已更新来源的集合
# docs = db.get(where={"source": "../../resource/knowledge.txt"})
# print(docs)

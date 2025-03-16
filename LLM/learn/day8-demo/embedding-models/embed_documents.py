#pip install langchain-openai

from langchain_openai import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        "嗨！",
        "哦，你好！",
        "你叫什么名字？",
        "我的朋友们叫我World",
        "Hello World！"
    ]
)
print(len(embeddings), len(embeddings[0]))

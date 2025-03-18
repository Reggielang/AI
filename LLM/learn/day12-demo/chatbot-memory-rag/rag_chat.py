# 安装所需的库
# pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入BeautifulSoup库，用于解析HTML内容
import bs4
# 从langchain库中导入创建检索链的方法
from langchain.chains import create_retrieval_chain
# 从langchain库中导入创建文档组合链的方法
from langchain.chains.combine_documents import create_stuff_documents_chain
# 从langchain_chroma库中导入Chroma类，用于向量存储
from langchain_chroma import Chroma
# 从langchain_community库中导入WebBaseLoader类，用于加载网页内容
from langchain_community.document_loaders import WebBaseLoader

# 使用WebBaseLoader加载网页内容，指定URL和解析参数
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # 使用 BeautifulSoup 解析 HTML 内容时，用来指定要解析的 HTML 元素的class
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
# 加载文档内容
docs = loader.load()
# 创建文本分割器，设定每个块的大小和重叠部分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 将文档分割成多个块
splits = text_splitter.split_documents(docs)
# 创建向量存储，使用分割后的文档和OpenAI的嵌入
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()
# 定义系统提示，用于问答任务
system_prompt = (
    "您是一个用于问答任务的助手。"
    "使用以下检索到的上下文片段来回答问题。"
    "如果您不知道答案，请说您不知道。"
    "最多使用三句话，保持回答简洁。"
    "\n\n"
    "{context}"
)
# 创建聊天提示模板，包含系统提示和用户输入
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# 创建OpenAI聊天模型
chat = ChatOpenAI(
    openai_api_base="http://10.94.28.24:7070/v1",
    openai_api_key="",
    model="/root/data/models/DeepSeek-R1-Distill-Qwen-14B")
# 创建问答链，使用聊天模型和提示模板
question_answer_chain = create_stuff_documents_chain(chat, prompt)
# 创建检索链，将检索器和问答链结合
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# 调用检索链，输入问题并获取回答
response = rag_chain.invoke({"input": "什么是任务分解？"})
# 打印回答
print(response["answer"])
response = rag_chain.invoke({"input": "我刚刚问了什么?"})
# 打印出来的结果是错误的，没有上下文记忆
print(response["answer"])

# 安装需要的库
# pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4 langgraph
# 导入 BeautifulSoup 库，用于解析 HTML 文档
import bs4
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 从 langgraph.checkpoint.memory 模块中导入 MemorySaver 类，用于保存内存检查点
from langgraph.checkpoint.memory import MemorySaver
# 从 langgraph.prebuilt 模块中导入 create_react_agent 函数，用于创建反应代理
from langgraph.prebuilt import create_react_agent

# 创建一个 MemorySaver 对象，用于保存代理的内存状态
memory = MemorySaver()

# 创建一个 ChatOpenAI 对象，指定使用 "gpt-4" 模型，温度设为 0
llm = ChatOpenAI(model="gpt-4", temperature=0)

### 构建检索器 ###
# 创建一个 WebBaseLoader 对象，用于加载指定网页的内容
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),  # 指定网页路径
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")  # 只解析指定的 HTML 类
        )
    ),
)

# 加载网页文档内容
docs = loader.load()
# 创建一个 RecursiveCharacterTextSplitter 对象，用于将文档分割成小块文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# 将文档分割成小块文本
splits = text_splitter.split_documents(docs)
# 创建一个 Chroma 对象，用于将分割后的文本转换为向量存储
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
# 将向量存储转换为检索器
retriever = vectorstore.as_retriever()
### 构建检索工具 ###
# 创建一个检索工具，指定检索器、工具名称和说明
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "搜索并返回《自主代理》博客文章摘录。",
)
# 将检索工具放入工具列表中
tools = [tool]
print(tool.invoke("任务分解"))
# 创建一个反应代理，指定使用的语言模型、工具和内存检查点
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
# 配置代理的参数
config = {"configurable": {"thread_id": "abc123"}}
# 定义查询内容
query = "什么是任务分解？"
# 向代理发送查询，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

# 定义另一个查询内容
query = "常见的做法有哪些？"
# 向代理发送查询，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

# 向代理发送消息，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content="你好，我叫Cyber")]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

# 定义第三个查询内容
query = "我叫什么名字？"

# 向代理发送查询，启动对话流
for s in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config=config
):
    # 打印代理的响应
    print(s)
    print("----")

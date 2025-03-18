# 安装所需的库
# pip install --upgrade langchain langchain-community langchainhub langchain-chroma bs4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_chroma import Chroma
# 从langchain库中导入创建历史感知检索器的方法
from langchain.chains import create_history_aware_retriever
# 从langchain库中导入创建检索链的方法
from langchain.chains import create_retrieval_chain
# 从langchain库中导入创建文档组合链的方法
from langchain.chains.combine_documents import create_stuff_documents_chain
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
llm = ChatOpenAI()

# 创建问答链，使用聊天模型和提示模板
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链，将检索器和问答链结合
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 定义系统提示，用于重新构造问题
contextualize_q_system_prompt = (
    "给定聊天历史和最新的用户问题，"
    "该问题可能引用聊天历史中的上下文，"
    "重新构造一个可以在没有聊天历史的情况下理解的独立问题。"
    "如果需要，不要回答问题，只需重新构造问题并返回。"
)
# 创建聊天提示模板，包含系统提示、聊天历史占位符和用户输入
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# 创建历史感知检索器，使用聊天模型、检索器和重新构造问题的提示模板
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
# 创建新的聊天提示模板，包含系统提示、聊天历史占位符和用户输入
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# 创建新的问答链，使用聊天模型和新的提示模板
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# 创建新的检索链，将历史感知检索器和新的问答链结合
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
# 初始化聊天历史为空列表
chat_history = []
# 定义第一个问题
question = "什么是任务分解？"
# 调用检索链，输入第一个问题和聊天历史，并获取回答
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

# 更新聊天历史，添加用户消息和AI消息
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)
# 打印第一个问题的回答
print(ai_msg_1["answer"])
# 定义第二个问题
second_question = "我刚刚问了什么?"
# 调用检索链，输入第二个问题和更新后的聊天历史，并获取回答
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
# 打印出来的结果是正确的，具备上下文记忆
print(ai_msg_2["answer"])

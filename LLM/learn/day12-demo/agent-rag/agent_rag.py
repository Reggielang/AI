# 从指定模块导入所需的类和函数
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client

# 定义一组URL，这些URL指向我们要加载的博客文章
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 使用WebBaseLoader加载每个URL的内容，并将其存储在docs列表中
docs = [WebBaseLoader(url).load() for url in urls]
# 处理嵌套列表，生成一个包含所有文档的单一列表
docs_list = [item for sublist in docs for item in sublist]

# 创建一个递归字符文本分割器，用于将文档分割成小块
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
# 将文档分割成小块
doc_splits = text_splitter.split_documents(docs_list)

# 将分割后的文档添加到向量数据库
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
# 创建一个检索器，用于从向量数据库中检索相关文档
retriever = vectorstore.as_retriever()

# 从指定模块导入函数
from langchain.tools.retriever import create_retriever_tool

# 创建一个检索工具
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

# 将检索工具添加到工具列表中
tools = [retriever_tool]

# 导入所需的类型和函数
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# 定义一个代理状态类，包含消息序列
from typing import Literal


# 再次导入所需的类型和函数
class AgentState(TypedDict):
    # add_messages函数定义了更新应该如何处理，默认是替换，add_messages表示“追加”
    messages: Annotated[Sequence[BaseMessage], add_messages]
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# 从预构建模块中导入工具条件
from langgraph.prebuilt import tools_condition


# 定义一个函数，用于判断检索到的文档是否与问题相关
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    确定检索到的文档是否与问题相关。

    参数:
        state (messages): 当前状态

    返回:
        str: 决定文档是否相关
    """

    print("---检查相似性---")

    # 数据模型
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="是否相似 'yes' 或 'no'")

    # LLM模型
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

    # 带有工具和验证的LLM
    llm_with_tool = model.with_structured_output(grade)

    # 提示模板
    prompt = PromptTemplate(
        template="""你是一名评估员，负责评估检索到的文档是否与用户问题相关。\n
        这里是检索到的文档：\n\n {context} \n\n
        这里是用户的问题：{question} \n
        如果文档包含与用户问题相关的关键词或语义含义，请将其评定为相关。\n
        给出一个是否相似判定 'yes' 或 'no'，以表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    # 链接提示模板和LLM
    chain = prompt | llm_with_tool

    # 获取当前状态的消息
    messages = state["messages"]
    last_message = messages[-1]

    # 获取问题和文档内容
    question = messages[0].content
    docs = last_message.content

    # 执行链条，获取评分结果
    scored_result = chain.invoke({"question": question, "context": docs})

    # 获取评分
    score = scored_result.binary_score

    if score == "yes":
        print("---决策: 文档相似---")
        return "generate"

    else:
        print("---决策: 文档不相似---")
        print(score)
        return "rewrite"


# 定义代理函数，用于根据当前状态生成响应
def agent(state):
    """
    调用代理模型，根据当前状态生成响应。给定问题，它将决定是否使用检索工具，或者直接结束。

    参数:
        state (messages): 当前状态

    返回:
        dict: 包含代理响应的更新状态
    """
    print("---调用 Agent---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # 返回一个列表，因为这将被添加到现有列表中
    return {"messages": [response]}


# 定义重写函数，用于改写问题
def rewrite(state):
    """
    转换查询以生成更好的问题。

    参数:
        state (messages): 当前状态

    返回:
        dict: 包含重新措辞问题的更新状态
    """

    print("---rewrite(改写问题)---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
        查看输入并尝试推理其背后的语义意图/含义。\n 
        这是初始问题：
        \n ------- \n
        {question} 
        \n ------- \n
        形成一个改进的问题：使用中文输出""",
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


# 定义生成函数，用于生成答案
def generate(state):
    """
    生成答案

    参数:
        state (messages): 当前状态

    返回:
         dict: 包含答案的更新状态
    """
    print("---generate(生成答案)---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    # LLM模型
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

    # 后处理函数，用于格式化文档
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 构提示词模板和LLM Chain
    rag_chain = prompt | llm | StrOutputParser()

    # 执行Chains，生成答案
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
# 参考官方提示词模型：https://smith.langchain.com/hub/rlm/rag-prompt
"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
# 打印提示模板
prompt.pretty_print()  # 显示提示模板的样子

# 从指定模块导入函数和类
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

# 定义一个新的状态图
workflow = StateGraph(AgentState)

# 定义我们将在其中循环的节点
workflow.add_node("agent", agent)  # 代理
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)  # 检索
workflow.add_node("rewrite", rewrite)  # 重写问题
workflow.add_node(
    "generate", generate
)  # 在我们知道文档相关后生成响应
# 调用代理节点以决定是否检索
workflow.add_edge(START, "agent")

# 决定是否检索
workflow.add_conditional_edges(
    "agent",
    # 评估代理决策
    tools_condition,
    {
        # 将条件输出转换为图中的节点
        "tools": "retrieve",
        END: END,
    },
)

# 在调用`action`节点后采取的边。
workflow.add_conditional_edges(
    "retrieve",
    # 评估
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# 编译状态图
graph = workflow.compile()

# 将生成的图片保存到文件
graph_png = graph.get_graph().draw_mermaid_png()
with open("agent_rag.png", "wb") as f:
    f.write(graph_png)

# 导入pprint模块用于打印输出
import pprint

# 定义输入消息
inputs = {
    "messages": [
        #("user", "Lilian Weng 对代理记忆的类型有什么看法？请用中文输出"),
        ("user", "Lilian Weng 对深度记忆的类型有什么看法？请用中文输出"),
    ]
}
# 逐步执行状态图，并打印每个节点的输出
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
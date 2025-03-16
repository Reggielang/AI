# pip install -qU langchain-community wikipedia
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "您是一个乐于助人的助手"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = load_tools(["wikipedia"])
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, stream_runnable=False
)
with get_openai_callback() as cb:
    response = agent_executor.invoke(
        {
            "input": "蜂鸟的学名是什么，哪种鸟是最快的？"
        }
    )
    print(f"总令牌数：{cb.total_tokens}")
    print(f"提示令牌：{cb.prompt_tokens}")
    print(f"完成令牌：{cb.completion_tokens}")
    print(f"总成本（美元）：${cb.total_cost}")

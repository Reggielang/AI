# !pip install -qU langchain-community wikipedia
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
with get_openai_callback() as cb:
    result = llm.invoke("告诉我一个笑话")
    print(cb)

# pip install -qU langchain langchain-openai
from langchain.output_parsers import YamlOutputParser
from langchain_core.prompts import PromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# 定义您期望的数据结构。
class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解答笑话的答案")

model = ChatOpenAI(temperature=0)
# 创建一个查询，旨在提示语言模型填充数据结构。
joke_query = "告诉我一个笑话。"
# 设置一个解析器 + 将指令注入到提示模板中。
parser = YamlOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model
print(parser.get_format_instructions())
response = chain.invoke({"query": joke_query})
print(response.content)
#print(parser.parse(response))
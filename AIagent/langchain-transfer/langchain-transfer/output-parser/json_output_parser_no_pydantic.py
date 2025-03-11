# pip install -qU langchain langchain-openai
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o", temperature=0)

joke_query = "告诉我一个笑话。"
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="A回答用户的查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | model | parser
response = chain.invoke({"query": joke_query})
print(response)

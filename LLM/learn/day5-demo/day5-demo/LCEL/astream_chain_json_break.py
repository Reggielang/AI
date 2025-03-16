import asyncio
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()
chain = (
        model | JsonOutputParser()
)

# 由于Langchain旧版本中的一个错误，JsonOutputParser未能从某些模型中流式传输结果


async def async_stream():
    # 一个在最终输入上操作的函数而不是在输入流上操作
    def _extract_country_names(inputs):
        # 如果输入不是字典，则返回空字符串
        if not isinstance(inputs, dict):
            return ""
        # 如果输入字典中不包含 "countries" 键，则返回空字符串
        if "countries" not in inputs:
            return ""
        # 获取输入字典中的 "countries" 键对应的值
        countries = inputs["countries"]
        # 如果 "countries" 的值不是列表，则返回空字符串
        if not isinstance(countries, list):
            return ""
        # 提取每个国家的名字，并返回名字列表
        country_names = [
            country.get("name") for country in countries if isinstance(country, dict)
        ]
        return country_names

    # 将模型、JSON 输出解析器和提取国家名字的函数链式连接
    chain = model | JsonOutputParser() | _extract_country_names
    async for text in chain.astream(
            "以JSON 格式输出法国、西班牙和日本的国家及其人口列表。"
            '使用一个带有“countries”外部键的字典，其中包含国家列表。'
            "每个国家都应该有键`name`和`population`"
    ):
        # 打印输出结果，以 "|" 分隔，并立即刷新输出缓冲区
        print(text, end="|", flush=True)


# 运行异步流处理
asyncio.run(async_stream())

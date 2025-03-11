import base64
import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片中的天气"},
        #如果图片地址是大模型访问不到，因为网络问题
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)
response = model.invoke([message])
print(response.content)

#response = model.invoke([message])
#print(response.content)


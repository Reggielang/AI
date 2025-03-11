from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {"type": "text", "text": "用中文描述这张图片中的天气"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)
response = model.invoke([message])
print(response.content)
#这张图片展示了一个晴朗的天气。天空中有一些淡淡的云，阳光明媚，照亮了图中的草地和木板路。天空呈现出明亮的蓝色，与绿色的草地形成了鲜明的对比。整体感觉是非常清新和舒适的，适合户外活动和散步。

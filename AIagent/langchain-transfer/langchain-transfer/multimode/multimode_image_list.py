from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

image_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Morning_in_China_Snow_Town.jpg/1280px-Morning_in_China_Snow_Town.jpg"
model = ChatOpenAI(model="gpt-4o")
message = HumanMessage(
    content=[
        {"type": "text", "text": "这两张图片是一样的吗？"},
        {"type": "image_url", "image_url": {"url": image_url_1}},
        {"type": "image_url", "image_url": {"url": image_url_2}},
    ],
)
response = model.invoke([message])
print(response.content)
# 这两张图片不一样。第一张是一个晴天的草地景色，有一条木板小路通向远方；第二张是一个覆盖着雪的村庄，有多栋房屋和一些红色灯笼。两张图片显示的是完全不同的场景。

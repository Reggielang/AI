import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io

# client = OpenAI(
#     base_url='https://api.openai-proxy.org/v1',
#     api_key='sk-Z42jgDd31SK7UY6cv0l2sHL1CtaV7EqeoVlEyOSCjNbBMAok',
# )

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-Z42jgDd31SK7UY6cv0l2sHL1CtaV7EqeoVlEyOSCjNbBMAok',
)

#定义函数来处理文本输入
def process_text(prompt):
    #调用OpenAI API获取回复
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100#回复的最大长度
    )
    #获取回复内容
    return response.choices[0].message.content.strip()

#定义函数来将图片编码为base64字符串
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        #将图片编码为base64字符串
        return base64.b64encode(image_file.read()).decode('utf-8')

#定义函数处理图片输入
def process_image(image):
    #将图片转换为PIL Image对象
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    #将图片编码为base64字符串
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    #调用OpenAI API获取回复
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "这张图片里面有什么？"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]}
        ]
    )
    return response.choices[0].message.content.strip()

#Streamlit应用程序布局
st.title("Chatbot 多模态应用")
#文本输入部分
#设置文本输入部分的标题
st.header("文本输入")
#创建一个文本输入框
text_input = st.text_input("输入你的问题：")
#创建一个按钮，当按钮被点击时执行以下操作
if st.button("发送"):
    #如果文本输入框不为空
    if text_input:
        #调用process_text函数处理文本输入
        response = process_text(text_input)
        #在Streamlit应用程序中显示回复
        st.write(f"回答:{response}")
    else:
        #如果文本输入框为空，显示提示信息
        st.write("请输入问题。")

#图像输入部分
#设置图像输入部分的标题
st.header("图像输入")
#创建一个文件上传器，允许用户上传图片
uploaded_file = st.file_uploader("上传图片：", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #将上传的图片显示在Streamlit应用程序中
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_column_width=True)
    #创建一个按钮，当按钮被点击时执行以下操作
    if st.button("发送"):
        #调用process_image函数处理图片输入
        response = process_image(uploaded_file)
        st.write(f"回答:{response}")
    else:
        st.write("请上传图片。")

# 导入streamlit库并简写为st，streamlit是一个用于创建数据应用的Python库
import streamlit as st
from webui import rag_chat_page, knowledge_base_page  # , platforms_page
from utils import get_img_base64

# 检查当前模块是否是主程序入口
if __name__ == "__main__":
    # 使用streamlit的sidebar上下文管理器，在侧边栏中添加内容
    with st.sidebar:
        # 在侧边栏中显示一个logo，使用get_img_base64函数获取图片的base64编码
        st.logo(
            get_img_base64("chatchat_lite_logo.png"),  # 获取大图标的base64编码
            size="large",  # 设置logo的大小为large
            icon_image=get_img_base64("chatchat_lite_small_logo.png"),  # 获取小图标的base64编码
        )

    # 创建一个导航对象pg，定义应用的页面结构
    pg = st.navigation({
        "对话": [  # 定义一个名为“对话”的页面组
            st.Page(rag_chat_page, title="智能客服", icon=":material/chat:"),
        ],
        "设置": [  # 定义一个名为“设置”的页面组
            st.Page(knowledge_base_page, title="行业知识库", icon=":material/library_books:"),
            # 添加一个页面，使用knowledge_base_page函数，标题为“知识库管理”，图标为图书馆
        ]
    })
    # 运行导航对象pg，显示定义的页面
    pg.run()

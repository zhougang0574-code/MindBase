import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="MindBase",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 MindBase 个人知识库")
st.caption("上传文档，随时提问")

with st.sidebar:
    st.header("📁 上传文档")

    # 知识库名称输入
    collection_name = st.text_input(
        "知识库名称",
        value="default",
        help="同一个知识库的文档放在一起，可按项目分类"
    )

    # 文件上传控件
    uploaded_file = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "txt"],
        help="支持 PDF、Word、TXT 格式"
    )

    # 上传按钮
    if st.button("上传到知识库", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.warning("请先选择文件")
        else:
            with st.spinner("处理中..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                        data={"collection_name": collection_name},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ 上传成功，共处理 {data['chunks']} 个文本块")
                    else:
                        st.error(f"上传失败：{response.json().get('detail')}")
                except Exception as e:
                    st.error(f"连接失败，请确认后端已启动：{e}")

    # 显示已有知识库列表
    st.divider()
    st.subheader("📂 已有知识库")
    try:
        res = requests.get(f"{API_BASE}/collections")
        if res.status_code == 200:
            collections = res.json().get("collections", [])
            if collections:
                for col in collections:
                    st.badge(col)
            else:
                st.caption("暂无知识库，请上传文档")
    except:
        st.caption("后端未连接")

# 初始化对话历史，存在 session_state 里
# session_state 相当于前端的状态管理，页面刷新前数据一直在
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # 如果是助手消息，显示来源
        if msg.get("sources"):
            with st.expander("📄 参考来源"):
                for src in msg["sources"]:
                    st.caption(f"**{src['file']}**")
                    st.caption(src["content_preview"])

# 底部输入框
if question := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 调用后端获取答案
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": question,
                        "collection_name": collection_name,
                        "top_k": 5,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.write(answer)

                    if sources:
                        with st.expander("📄 参考来源"):
                            for src in sources:
                                st.caption(f"**{src['file']}**")
                                st.caption(src["content_preview"])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    err = response.json().get("detail", "未知错误")
                    st.error(f"查询失败：{err}")

            except Exception as e:
                st.error(f"连接失败，请确认后端已启动：{e}")
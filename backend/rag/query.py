import os

import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from backend.rag.ingest import get_embeddings

dotenv.load_dotenv()


def get_llm():
    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.3
    )


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一个专业的知识库助手，请根据以下文档内容回答用户问题。

规则：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关内容，明确告知用户"文档中未找到相关信息"
3. 回答要简洁清晰，重点突出
4. 如果信息来自特定文件，可以提及来源

文档内容：
{context}

用户问题：{question}

回答："""
)

# TODO
def format_docs(docs):
    """把检索到的文档列表拼接成字符串，塞进 Prompt 的 context"""
    return "\n\n".join(doc.page_content for doc in docs)

def query(question: str, collection_name: str = "default", top_k: int = 5) -> dict:
    # 第一步：连接向量库
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_function=get_embeddings(),
    )

    # 第二步：构建检索器
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # 第三步：LCEL 链
    chain = (
        {
            "context": retriever | format_docs,  # 检索 → 格式化成字符串
            "question": RunnablePassthrough(),    # 问题直接透传
        }
        | QA_PROMPT   # 填入 Prompt 模板
        | get_llm()   # 发给 LLM
        | StrOutputParser()  # 把 LLM 输出解析成字符串
    )

    # 第四步：执行查询
    answer = chain.invoke(question)

    # 第五步：单独检索一次拿来源（LCEL 链不直接返回原始文档）
    source_docs = retriever.invoke(question)
    sources = []
    seen = set()
    for doc in source_docs:
        source_file = doc.metadata.get("source_file", "未知文件")
        if source_file not in seen:
            seen.add(source_file)
            sources.append({
                "file": source_file,
                "content_preview": doc.page_content[:100] + "...",
            })

    return {
        "answer": answer,
        "sources": sources,
    }
import os

import dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from backend.rag.ingest import get_embeddings
from langchain_community.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.documents import Document
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

    # 第二步：从向量库取出所有文档，用于构建 BM25
    print("📚 获取知识库文档...")
    all_docs = vectorstore.get()

    documents = [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(all_docs['documents'], all_docs['metadatas'])
    ]

    # 第三步：构建向量检索器
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # 第四步：构建 BM25 检索器
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = top_k  # 返回数量和向量检索一致

    # 第五步：用 EnsembleRetriever 融合两路结果（RRF算法）
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # 两路各占50%权重，可以调整
    )

    # 第六步：LCEL 链（把检索器换成混合检索器）
    chain = (
        {
            "context": ensemble_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | get_llm()
        | StrOutputParser()
    )

    # 第七步：执行查询
    answer = chain.invoke(question)

    # 第八步：单独获取来源
    source_docs = ensemble_retriever.invoke(question)
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
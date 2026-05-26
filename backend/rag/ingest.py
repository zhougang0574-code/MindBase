from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
import os
import dotenv
from langchain_chroma import Chroma

dotenv.load_dotenv()

# 1、加载文件
def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()

    if ext == '.pdf':
        load = PyPDFLoader(file_path)
    elif ext == '.txt':
        load = TextLoader(file_path=file_path, encoding="utf-8")
    elif ext in ('.docx', '.doc'):
        load = UnstructuredWordDocumentLoader(file_path=file_path, mode="single")
    else:
        raise ValueError(f"不支持的文件类型：{ext}，支持 PDF、DOCX、TXT")

    return load.load()

# 2、切分文件内容

def split_documents(documents):
    chunk = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '。', '！', '？', '……', '，', ''],
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    ).split_documents(documents)
    return chunk

#3、Embedding向量工具


# 全局缓存，避免重复加载模型
_embeddings_instance = None

def get_embeddings():
    # 声明使用外部全局变量，而不是创建新的局部变量
    global _embeddings_instance

    # 如果还没加载过，才初始化（单例模式）
    if _embeddings_instance is None:
        print("⏳ 首次加载 Embedding 模型...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",  # 本地已下载的模型
            model_kwargs={"device": "cpu"},  # 没有 GPU，指定用 CPU 运行
            encode_kwargs={"normalize_embeddings": True},  # 向量归一化，提升检索精度
        )
        print("✅ Embedding 模型加载完成")

    # 直接返回已加载的实例
    return _embeddings_instance

# 4保存至向量库


def ingest_file(file_path:str,collection_name:str="default") ->int:
    # 加载文档
    documents = load_document(file_path)
    # 第二步：给每个文档加元数据，记录来源文件名
    file_name = Path(file_path).name  # 只取文件名，不要完整路径
    for doc in documents:
        # doc.metadata["source_file"] = file_name：给每个 chunk 打上来源标记，问答时可以告诉用户答案来自哪个文件。
        doc.metadata["source_file"] = file_name

    # 第三步：切分
    chunks = split_documents(documents)

    # 第四步：存入向量库
    vectorstore = Chroma(
        collection_name=collection_name,
        # os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")：读取.env里配置的存储路径，如果没配置就用. / chroma_db作为默认值。
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        embedding_function=get_embeddings(),
    )
    vectorstore.add_documents(chunks)

    # 返回存入的 chunk 数量，方便调用方知道处理了多少块
    return len(chunks)
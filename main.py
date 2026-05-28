import os
import tempfile

import dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import ingest_file, query

dotenv.load_dotenv()

app = FastAPI(title="MindBase API", version="1.0.0")

# 允许跨域，Streamlit 前端访问后端需要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class QueryRequest(BaseModel):
    question: str
    collection_name: str = "default"
    top_k: int = 5

# 响应模型
class QueryResponse(BaseModel):
    answer: str
    sources: list



@app.get("/")
def health_check():
    """健康检查，确认服务是否正常运行"""
    return {"status": "ok", "message": "MindBase 服务运行中"}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Form(default="default"),
):
    """上传文档到知识库"""
    # 检查文件类型
    allowed_types = [".pdf", ".docx", ".txt"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {file_ext}，支持：{allowed_types}"
        )

    # 保存为临时文件，处理完自动删除
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunk_count = ingest_file(tmp_path, collection_name, original_filename=file.filename)
        return {
            "success": True,
            "filename": file.filename,
            "collection": collection_name,
            "chunks": chunk_count,
            "message": f"成功处理 {chunk_count} 个文本块"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # 无论成功失败都删除临时文件


@app.post("/query", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    """向知识库提问"""
    try:
        result = query(
            question=req.question,
            collection_name=req.collection_name,
            top_k=req.top_k,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def get_collections():
    """获取所有知识库列表"""
    import chromadb
    client = chromadb.PersistentClient(
        path=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    collections = [col.name for col in client.list_collections()]
    return {"collections": collections}
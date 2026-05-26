from sentence_transformers import SentenceTransformer
print("开始下载模型...")
model = SentenceTransformer("BAAI/bge-m3")
print("✅ 下载完成")
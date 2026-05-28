MindBase
1、下载Embedding向量模型
执行文件 download_model 下载向量模型  
2、安装依赖
requirements.txt
pip install -r requirements.txt

3、配置环境变量
.env 配置APIKEY，向量数据库位置


4、ingest.py
主要功能，加载文档，文档切分，加载向量库，将文档存储至向量库
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

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

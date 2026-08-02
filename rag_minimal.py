"""
金融合规 RAG 最小可运行 Demo（项目① · Week 1）
技术栈：LangChain + Chroma(本地向量库) + bge-small-zh(本地 embedding) + DeepSeek API(LLM)

设计原则（务必记住）：
  - embedding 模型与向量库都在本机运行，监管文档不出本机；
  - 只有「用户问题 + 检索到的相关片段」会发往 DeepSeek 生成答案，数据不出境；
  - 答案用 [1][2] 标注引用编号，这是合规场景防幻觉的关键，面试能讲。
"""

import os

from langchain_community.document_loaders import TextLoader
# 后续读真实监管 PDF 时，把上面这行换成：
# from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ===== 0. 配置 LLM（DeepSeek 是 OpenAI 兼容接口，用 ChatOpenAI 即可）=====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 必须先设置这个环境变量
if not DEEPSEEK_API_KEY:
    raise SystemExit("❌ 未检测到 DEEPSEEK_API_KEY，请先 export 该环境变量后再运行。")

llm = ChatOpenAI(
    model="deepseek-chat",            # 也可用 deepseek-reasoner 做效果对比
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",  # DeepSeek 的兼容网关（如遇路径报错试加 /v1）
    temperature=0,                    # 合规问答要稳定可复现，temperature 设 0
)


# ===== 1. 加载文档 =====
# Week 1 先用一个示例 txt 跑通流程；Week 2 换成央行/金融监管总局公开 PDF：
#   loader = PyPDFLoader("data/xxx.pdf")
loader = TextLoader("data/sample.txt", encoding="utf-8")
documents = loader.load()


# ===== 2. 切分：长文档切成小块，便于检索与上下文拼接 =====
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)


# ===== 3. 本地 Embedding：把文字变成向量（本机运行，数据不出本机）=====
# bge-small-zh-v1.5 体积小、中文效果好，首次运行会自动下载约 130MB
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ===== 4. 建向量库（Chroma，本地持久化到 ./chroma_db）=====
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 每次取最相关的 3 段


# ===== 5. 检索 + 生成（LCEL 管道，带引用编号，避免已废弃的 RetrievalQA）=====
prompt = ChatPromptTemplate.from_template(
    "你是一个金融合规问答助手。请仅依据下面的「上下文」作答，"
    "并在答案中用 [1]、[2] 标注所引用的上下文编号；"
    "若上下文无法回答该问题，请如实说明「依据现有材料无法回答」。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)


def format_docs(docs):
    """把检索到的片段拼成带编号的上下文。"""
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


question = "请总结本文档中的合规要求要点。"
retrieved = retriever.invoke(question)
context = format_docs(retrieved)

chain = prompt | llm | StrOutputParser()   # 提示词 → 大模型 → 纯文本输出
answer = chain.invoke({"context": context, "question": question})


# ===== 6. 展示结果 =====
print("【问题】", question)
print("\n【回答】\n", answer)
print("\n【引用来源】")
for i, doc in enumerate(retrieved, 1):
    print(f"[{i}] {doc.page_content[:120]}...")

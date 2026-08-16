"""
金融合规 RAG · W2 升级版（项目①）
在 W1 基础上：TextLoader → PyPDFLoader，支持批量读取 data/ 下所有公开监管 PDF
技术栈：LangChain + Chroma(本地) + bge-small-zh(本地) + DeepSeek API(LLM)

合规红线（务必遵守）：
  - 只使用央行/金融监管总局/证监会等官网公开发布的规章、规范性文件、政策解读 PDF
  - 绝对不使用：公司内部合规制度、客户数据、未公开材料
  - 个人项目与公司职务严格隔离，规避竞业/泄密风险
  - README 须注明"仅用公开数据"
"""

import os
import glob
import sys

# 修复 Windows GBK 控制台中文乱码：强制 stdout/stderr 用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ===== 0. 配置 =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise SystemExit("❌ 未检测到 DEEPSEEK_API_KEY，请先 export 该环境变量后再运行。")

# TEST_MODE：验证检索链路用，正式使用请设为 False
TEST_MODE = False
PROBE_FACT = "本机构规定：所有员工须于每周五17点前提交手写合规自查报告，报告统一编号为 CX-2026-001。"

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,   # 合规问答要稳定可复现
)


# ===== 1. 批量加载 PDF（data/ 目录下所有 .pdf）=====
pdf_paths = glob.glob("data/*.pdf")
if not pdf_paths:
    raise SystemExit("❌ data/ 目录下未找到任何 PDF，请先放入公开监管文件后再运行。")

print(f"📄 发现 {len(pdf_paths)} 个 PDF：")
for p in pdf_paths:
    print("   -", p)

documents = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())   # 每个 PDF 的每页成为一个 Document
print(f"   共加载 {len(documents)} 页\n")


# ===== 2. 切分（PDF 页数多，调大 chunk 更贴合段落）=====
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
chunks = splitter.split_documents(documents)
print(f"   切分为 {len(chunks)} 个片段\n")


# ===== 3. 本地 Embedding（本机运行，数据不出本机）=====
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# ===== 4. 建向量库（本地持久化）=====
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# ===== 5. 检索 + 生成（带引用编号）=====
prompt = ChatPromptTemplate.from_template(
    "你是一个金融合规问答助手。请仅依据下面的「上下文」作答，"
    "并在答案中用 [1]、[2] 标注所引用的上下文编号；"
    "若上下文无法回答该问题，请如实说明「依据现有材料无法回答」。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)

def format_docs(docs):
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


if TEST_MODE:
    question = "员工自查报告的编号规则是什么？"
    print("[TEST_MODE] 探针问题：", question)
else:
    question = "银行保险机构消费者权益保护管理办法，什么时候公布？什么时候生效？"   

retrieved = retriever.invoke(question)
context = format_docs(retrieved)

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})


# ===== 6. 展示 =====
print("【问题】", question)
print("\n【回答】\n", answer)
print("\n【引用来源】")
for i, doc in enumerate(retrieved, 1):
    src = doc.metadata.get("source", "未知文件")
    print(f"[{i}] ({src}) {doc.page_content[:200]}...")


# ===== 7. 测试自动判定 =====
if TEST_MODE:
    if PROBE_FACT.split("编号为")[-1].strip("。") in answer:
        print("\n[TEST PASS] 检索链路已验证。")
    else:
        print("\n[TEST FAIL] 答案未出现 ，请检查 原文。")

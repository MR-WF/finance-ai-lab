"""
金融合规 RAG 最小可运行 Demo（项目① · Week 1）
技术栈：LangChain + Chroma(本地向量库) + bge-small-zh(本地 embedding) + DeepSeek API(LLM)

设计原则（务必记住）：
  - embedding 模型与向量库都在本机运行，监管文档不出本机；
  - 只有「用户问题 + 检索到的相关片段」会发往 DeepSeek 生成答案，数据不出境；
  - 答案用 [1][2] 标注引用编号，这是合规场景防幻觉的关键，面试能讲。
"""

import os
import sys

# 修复 Windows GBK 控制台中文乱码：强制 stdout/stderr 用 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_community.document_loaders import TextLoader
# 后续读真实监管 PDF 时，把上面这行换成：
# from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ===== 0. 配置 =====
# 测试模式开关：
#   True  → 运行「埋假事实」探针，验证检索链路是否真从文档取数（验证 W1 用）
#   False → 跑正常业务问题
# ⚠️ 开启测试模式前，请先把下面 PROBE_FACT 那行加进 data/sample.txt
TEST_MODE = True
PROBE_FACT = "本机构规定：所有员工须于每周五17点前提交手写合规自查报告，报告统一编号为 CX-2026-001。"

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


# 测试模式：用探针问题验证「检索是否真发生了」（而非 LLM 凭记忆作答）
# 预期：若检索正常，答案应出现 CX-2026-001 并标注 [1] 引用
# 若答案胡扯或无此编号 → 检索没接上，检查 VectorStore / retriever 配置
if TEST_MODE:
    question = "员工自查报告的编号规则是什么？"
    print("[TEST_MODE] 探针问题：", question)
    print("   提示：请确认 data/sample.txt 已包含这一行 ->")
    print("   ", PROBE_FACT, "\n")
else:
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

# ===== 7. 测试模式自动判定 =====
if TEST_MODE:
    if PROBE_FACT.split("编号为")[-1].strip("。") in answer:
        print("\n[TEST PASS] 答案包含探针编号 CX-2026-001，检索链路已验证（答案确实来自你喂的文档）。")
    else:
        print("\n[TEST FAIL] 答案未出现 CX-2026-001。可能：①sample.txt 未加探针行；②chroma_db 是旧库，请删除 chroma_db 文件夹后重跑；③检索未接上。")

"""
金融合规 RAG · W3 升级版（项目①）
在 W2 基础上引入：
  1) 混合检索  = 向量检索(Chroma) + 关键词检索(BM25) → RRF 融合
  2) 重排 rerank = CrossEncoder(BAAI/bge-reranker-v2-m3) 对融合候选精排
  3) 引用溯源增强 = 引用编号映射到最终精排片段，并带 文件名 + 页码
技术栈：LangChain + Chroma + bge-small-zh(本地) + rank_bm25 + sentence-transformers(CrossEncoder) + DeepSeek
模型加载策略：在线（HF 镜像）优先；镜像不可达或连续失败后自动切换 ./models 本地离线模型。
合规红线：只用公开监管材料，文档明文绝不出境。
"""

import os
import glob
import sys
import time
import socket
import urllib.request

# 修复 Windows GBK 控制台中文乱码
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
from rank_bm25 import BM25Okapi


# ===== 0. 配置 =====
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

USE_RERANK = True      # reranker 模型下载失败可临时改 False，退化为纯混合检索
FORCE_LOCAL = True     # 默认本地离线加载：国内 HF 镜像普遍不稳，直接跳过在线尝试、零超时；需要时改 False 走在线兜底
MAX_RETRY   = 3        # 在线加载失败的重试次数，耗尽后自动切换本地
RETRY_DELAY = 3        # 重试间隔（秒）
MODEL_DIR   = "./models"   # 本地模型目录（由 download_models_modelscope.py 拉取）
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "8")  # 在线下载单次超时（秒），超时即失败、避免久等
VECTOR_K = 15          # 向量检索候选数

# ===== 0.1 模型加载兜底：在线优先；镜像不可达则秒切本地 =====
_OFFLINE = False
HF_PROBE_TIMEOUT = 3  # 网络探针超时（秒），仅用于在线上尝试前快速判断镜像是否活着

def _hf_host():
    """取出当前 HF 镜像主机名（用于连通性探针）。"""
    ep = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    return ep.split("//")[-1].split("/")[0] or "huggingface.co"

def hf_reachable(timeout=HF_PROBE_TIMEOUT):
    """HTTP 层探活：TCP 能握手但 HTTP 死掉也会被判定不可达，避免漫长重试。"""
    host = _hf_host()
    req = urllib.request.Request(
        f"https://{host}",
        method="HEAD",
        headers={"User-Agent": "hf-probe"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False

def _local_model_dir(repo_id):
    """BAAI/bge-small-zh-v1.5 -> ./models/bge-small-zh-v1.5"""
    return os.path.join(MODEL_DIR, repo_id.split("/")[-1])

def load_model_with_fallback(repo_id, loader):
    """
    优先在线（HF 镜像）加载；若 FORCE_LOCAL 已开、或镜像不可达、或连续
    MAX_RETRY 次失败，则自动切换到本地离线模型（需提前用
    download_models_modelscope.py 拉取）。
    loader: 接收 model_name 字符串，返回加载好的对象。
    """
    global _OFFLINE
    # 镜像不可达 → 跳过在线尝试，直接本地兜底（秒切，不再漫长重试）
    if not _OFFLINE and not FORCE_LOCAL and not hf_reachable():
        print(f"[网络探针] {_hf_host()} 不可达，跳过在线尝试，直接本地加载。")
        _OFFLINE = True
    if not _OFFLINE and not FORCE_LOCAL:
        last_err = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                print(f"[在线加载] {repo_id}（第 {attempt}/{MAX_RETRY} 次尝试）")
                return loader(repo_id)
            except Exception as e:
                last_err = e
                print(f"[在线失败] 第 {attempt}/{MAX_RETRY} 次：{type(e).__name__}: {e}")
                time.sleep(RETRY_DELAY)
        print(f"[兜底触发] {repo_id} 在线加载连续 {MAX_RETRY} 次失败，切换本地离线模式。")
        _OFFLINE = True
    local = _local_model_dir(repo_id)
    if not os.path.isdir(local):
        raise SystemExit(
            f"[兜底失败] 本地模型目录不存在：{local}\n"
            f"请先运行 `python download_models_modelscope.py` 把模型拉到本地，\n"
            f"或排查网络后重试（取消 FORCE_LOCAL）。"
        )
    print(f"[本地加载] {repo_id}  <-  {local}（离线，不再访问外网）")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return loader(local)
BM25_K = 15            # 关键词检索候选数
CANDIDATE_N = 10       # RRF 融合后保留的候选数
FINAL_K = 4            # 精排后送给 LLM 的片段数
RRF_K = 60             # RRF 常数（经验值 60）

TEST_MODE = False
PROBE_FACT = "本机构规定：所有员工须于每周五17点前提交手写合规自查报告，报告统一编号为 CX-2026-001。"

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,      # 合规问答要稳定可复现
)


# ===== 1. 加载 + 切分 =====
pdf_paths = glob.glob("data/*.pdf")
if not pdf_paths:
    raise SystemExit("data/ 目录下未找到任何 PDF，请先放入公开监管文件。")

documents = []
for path in pdf_paths:
    documents.extend(PyPDFLoader(path).load())
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
chunks = splitter.split_documents(documents)
print(f"加载 {len(documents)} 页，切分为 {len(chunks)} 个片段")


# ===== 2. 向量库（本地）=====
embeddings = load_model_with_fallback(
    "BAAI/bge-small-zh-v1.5",
    loader=lambda name: HuggingFaceEmbeddings(model_name=name),
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})


# ===== 3. BM25 关键词索引（本机，零外部依赖）=====
def tokenize(text):
    # 中文按字切，简单够用；英文可按空格进一步分词
    return [c for c in text if not c.isspace()]

bm25 = BM25Okapi([tokenize(c.page_content) for c in chunks])


# ===== 4. 重排模型（CrossEncoder）=====
if USE_RERANK:
    from sentence_transformers import CrossEncoder
    reranker = load_model_with_fallback(
        "BAAI/bge-reranker-v2-m3",
        loader=lambda name: CrossEncoder(name),
    )


# ===== 5. 混合检索 + 重排 =====
def hybrid_retrieve(question):
    # 向量检索（Chroma 返回的是新 Document 对象，不能用 id() 比对，改用内容作稳定 key）
    v_docs = vector_retriever.invoke(question)
    # 关键词检索（直接用同一个 chunks 列表，身份一致）
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_order = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_K]
    bm25_docs = [chunks[i] for i in bm25_order]

    # RRF 融合：以 page_content 为稳定 key，把"语义相近"与"字面命中"统一打分
    pool = {}  # key=内容 -> {"doc": Document, "score": 累计分}
    def _add(docs):
        for rank, d in enumerate(docs):
            k = d.page_content
            if k not in pool:
                pool[k] = {"doc": d, "score": 0}
            pool[k]["score"] += 1 / (RRF_K + rank + 1)
    _add(v_docs)
    _add(bm25_docs)

    top = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:CANDIDATE_N]
    candidates = [t["doc"] for t in top]
    # 重排：CrossEncoder 对 (问题, 片段) 打相关分，高分排前
    if USE_RERANK:
        pairs = [(question, c.page_content) for c in candidates]
        scores = reranker.predict(pairs)
        candidates = [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])]
    return candidates[:FINAL_K]


# ===== 6. 引用溯源增强：编号 + 文件名 + 页码 =====
def format_docs(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "未知文件")
        page = d.metadata.get("page", 0) + 1
        lines.append(f"[{i}] (来源：{src} 第{page}页)\n{d.page_content}")
    return "\n\n".join(lines)


# ===== 7. 提示优化 =====
prompt = ChatPromptTemplate.from_template(
    "你是一个严谨的金融合规问答助手。请严格依据下面的「上下文」作答：\n"
    "1) 答案中必须用 [1]、[2] 标注所引用的上下文编号，编号对应下方上下文顺序；\n"
    "2) 优先采信与问题最直接相关的片段；\n"
    "3) 若上下文无法回答该问题，如实说明「依据现有材料无法回答」，不得编造。\n\n"
    "上下文：\n{context}\n\n问题：{question}"
)

# 推理链（模块级构建一次，CLI 与 Streamlit 共用）
chain = prompt | llm | StrOutputParser()


def ask(question: str):
    """给定问题，返回 (回答, 检索到的片段列表)。CLI 与 Streamlit 共用。"""
    retrieved = hybrid_retrieve(question)
    context = format_docs(retrieved)
    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved


def _run_cli():
    if TEST_MODE:
        question = "员工自查报告的编号规则是什么？"
        print("[TEST_MODE] 探针问题：", question)
    else:
        question = "个人金融信息在存储和传输环节有哪些保护要求？"
    answer, retrieved = ask(question)
    print("【问题】", question)
    print("\n【回答】\n", answer)
    print("\n【引用来源】")
    for i, doc in enumerate(retrieved, 1):
        src = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", 0) + 1
        print(f"[{i}] ({src} 第{page}页) {doc.page_content[:100]}...")
    if TEST_MODE:
        if PROBE_FACT.split("编号为")[-1].strip("。") in answer:
            print("\n[TEST PASS] 检索链路已验证。")
        else:
            print("\n[TEST FAIL] 答案未出现 CX-2026-001，请检查语料/chroma_db。")


if __name__ == "__main__":
    _run_cli()

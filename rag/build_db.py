# 配置日志记录
import functools
import logging
import os

import torch
from langchain.vectorstores import faiss
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, \
    TokenTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = "C:\\coding\\project\\ds-test\\rag\\testfile"
VECTOR_INDEX_FILE = 'C:\\coding\\project\\ds-test\\rag\\data'
FILE_LIST_FILE = 'C:\\coding\\project\\ds-test\\rag\\data\\index.pkl'
OLLAMA_API_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL_NAME = "deepseek-r1:14b"
METADATA_SOURCE_ID = "source_id"
CHUNK_ORDER = "chunk_order"

EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"

@functools.lru_cache(maxsize=1)
def build_vector_index(directory):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    with open("C:\\coding\\project\\ds-test\\rag\\testfile\\2025面经.md", "r", encoding="utf-8") as f:
        markdown_text = f.read()

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ('#', 'header1'),
            ('##', 'header2'),
            ('###', 'header3')
        ],
        strip_headers=True
    )
    token_splitter = TokenTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    md_docs = md_splitter.split_text(markdown_text)
    source_id = 0
    for doc in md_docs:
        header2 = doc.metadata.get("header2", "")
        header3 = doc.metadata.get("header3", "")
        print("get-header:", header2, ", ", header3)
        if header3:
            # 把标题加到内容开头，强化语义
            doc.page_content = f"{header3}\n{doc.page_content}"
        if header2:
            doc.page_content = f"{header2}\n{doc.page_content}"
        source_id += 1
        doc.metadata[METADATA_SOURCE_ID] = source_id
    # 二次切分，一个段落里面可能放不下
    final_docs = token_splitter.split_documents(md_docs)
    # 切分后的chunk要排序
    from collections import defaultdict
    source_counter = defaultdict(int)
    for doc in final_docs:
        source_id = doc.metadata[METADATA_SOURCE_ID]
        # 当前是第几个 chunk（从0开始）
        order = source_counter[source_id]
        doc.metadata[CHUNK_ORDER] = order
        # 更新计数
        source_counter[source_id] += 1
    docsearch = FAISS.from_documents(final_docs, embeddings)

    os.makedirs(os.path.dirname(VECTOR_INDEX_FILE), exist_ok=True)
    docsearch.save_local(VECTOR_INDEX_FILE)
    print(f"✅ 向量数据库已保存至: {VECTOR_INDEX_FILE}")

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, docsearch.index)
        docsearch.index = index_gpu
    return docsearch


# build_vector_index(KNOWLEDGE_BASE_DIR)


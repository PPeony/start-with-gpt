
import json
from collections import OrderedDict

import aiohttp
import faiss
from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from starlette.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from build_db import logger, OLLAMA_API_URL, OLLAMA_MODEL_NAME, VECTOR_INDEX_FILE, EMBEDDING_MODEL_NAME, \
    KNOWLEDGE_BASE_DIR, METADATA_SOURCE_ID, CHUNK_ORDER
from collections import defaultdict

app = FastAPI()
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
docsearch = None
async def load_or_build_vector_index():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    docsearch = FAISS.load_local(
        VECTOR_INDEX_FILE,
        embeddings,
        allow_dangerous_deserialization=True  # 允许加载 pickle 文件（安全前提下开启）
    )
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        docsearch.index = faiss.index_cpu_to_gpu(res, 0, docsearch.index)

    isinstance(docsearch.index, faiss.GpuIndex)
    logger.info("load_or_build_vector_index-done")
    return docsearch

# 结果去重
def dedup_by_source(candidates, key="METADATA_SOURCE_ID"):
    # 用字典收集每个 source_id 的所有 chunk
    source_to_docs = defaultdict(list)
    for doc in candidates:
        source = doc.metadata.get(key)
        if source is not None:
            source_to_docs[source].append(doc)
    unique_docs = []
    for source, docs in source_to_docs.items():
        # 按 chunk_order 排序，取最小的
        min_doc = min(docs, key=lambda x: x.metadata.get(CHUNK_ORDER, 0))
        unique_docs.append(min_doc)
    return unique_docs

# 生成答案的流式输出函数
async def generate_answer(question):
    global docsearch
    if docsearch is None:
        docsearch = await load_or_build_vector_index()
    try:
        logger.info(f"Received question: {question}")
        # company_name = extract_company_name(question)
        # logger.info(f"你要查询的公司是： {company_name}")

        # 从知识库中检索相关文档
        candidates = docsearch.similarity_search(question, k=10)

        # 文档同一个段落拆分成多份之后要去重
        candidates = dedup_by_source(candidates, key=METADATA_SOURCE_ID)

        if not candidates:
            logger.info("No relevant documents found. Using LLM directly.")
            context = ""
        else:
            logger.info("Found relevant documents in the knowledge base.")
            # 重排序并选取 Top-3
            ranked_candidates = rerank(question, candidates)
            final_results = ranked_candidates[:3]

            # 把去重之后的拼接出来完整的。
            final_results = search_and_reconstruct(final_results, docsearch)

            if not final_results:
                logger.info("No relevant documents after reranking. Using LLM directly.")
                context = ""
            else:
                logger.info(f"Final results: {final_results}")
                # context = "\n".join([doc.page_content for doc in final_results])
                context = f"{final_results}"

        # 构建完整的提示
        # full_prompt = f"上下文信息：{context}\n问题：{question}"
        # 添加明确的指令和结构化格式
        full_prompt = (f"你必须根据知识库搜索结果回答问题。\n"
                       f"知识库搜索结果,前三位：\n{context}\n---\n"
                       f"问题：{question}\n---\n回答要求：你必须根据知识库搜索结果回答问题。请简洁回答，如有引用请注明出处。")

        # 流式请求 Ollama
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": full_prompt,
            "parameters": {
                "max_tokens": 100,
                "temperature": 0.2
            },
            "stream": True
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_API_URL, headers=headers, json=data) as res:
                answer = ""
                async for line in res.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            result = await process_ollama_stream(chunk)
                            if result:
                                answer_chunk = result.decode('utf-8')
                                answer += answer_chunk
                                # 流式返回数据
                                yield answer_chunk.encode('utf-8')
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decoding error: {e}, line: {line}")

    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        logger.error(error_message)
        yield error_message.encode('utf-8')


# 异步处理 llm.stream 结果的函数
async def process_ollama_stream(chunk):
    if 'response' in chunk:
        return chunk['response'].encode('utf-8')
    else:
        logger.error(f"Unexpected chunk format: {chunk}")
        return None


def rerank(query, candidates):
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

    max_length = 1024
    ranked_candidates = []
    for candidate in candidates:
        query_tokens = tokenizer.tokenize(query)
        doc_tokens = tokenizer.tokenize(candidate.page_content)

        total_length = len(query_tokens) + len(doc_tokens) + 2
        if total_length > max_length:
            remaining_length = max_length - len(query_tokens) - 2
            doc_tokens = doc_tokens[:remaining_length]

        input_text = tokenizer.convert_tokens_to_string(query_tokens + doc_tokens)
        inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            score = logits[0][1].item() if logits.shape[1] >= 2 else logits[0][0].item()
            ranked_candidates.append((candidate, score))

    ranked_candidates.sort(key=lambda x: x[1], reverse=True)
    return [candidate for candidate, _ in ranked_candidates]


def search_and_reconstruct(candidates, docsearch, top_k=1):

    # 存放所有需要还原的 source_id
    source_ids = []
    for doc in candidates:
        source_ids.append(doc.metadata[METADATA_SOURCE_ID])

    # 从 FAISS 中取出所有属于这些 source_id 的 chunk
    all_docs = docsearch.docstore._dict  # 所有文档的字典 {doc_id: Document}

    reconstructed = []

    for source_id in source_ids:
        # 找出所有属于该 source_id 的 chunk，这是全表扫描会慢。可以在构建数据库的时候，存一个hash表，保存source_id到chunk的映射
        # llm把这个索引设计成了一个json文件，查询时候直接读取json
        # 当前用FAISS数据库，不支持用metadata查询，所以需要用别的数据库，比如chroma
        chunks = []
        for doc_id, doc in all_docs.items():
            if doc.metadata.get(METADATA_SOURCE_ID) == source_id:
                chunks.append(doc)

        # 按 chunk_order 排序
        chunks.sort(key=lambda x: x.metadata["chunk_order"])

        # 拼接内容
        # full_text = "\n".join([chunk.page_content.strip() for chunk in chunks])
        reconstructed.append({
            # "page_content": full_text,
            "chunks": chunks,
            "metadata": chunks[0].metadata  # 保留原始元数据（如 header2 等）
        })

    return reconstructed
@app.get("/ask")
async def ask_question(question: str):
    return StreamingResponse(generate_answer(question), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # https://zhuanlan.zhihu.com/p/27947782392
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from DeepSeek应用开发实战.class27.llm import QdrantVecStoreFromDocs


def load_code(ext: str, dir_path: str):
    if not os.path.exists(dir_path):
        print(f"文件夹{dir_path}不存在")
        return

    files = []

    for file in os.listdir(dir_path):
        if file.endswith(ext):
            print(f"加载文件{file}")
            files.append(os.path.join(dir_path, file))

    all_docs = []
    code_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)

    for file in files:
        loader = TextLoader(file, encoding='utf-8').load()
        docs = code_text_splitter.split_documents(loader)
        for doc in docs:
            doc.metadata["source"] = file
            all_docs.append(doc)

    QdrantVecStoreFromDocs(all_docs, "code")


if __name__ == '__main__':
    load_code(".go", "C:\\coding\\project\\ds-test\\DeepSeek应用开发实战\\class27\\middleware")
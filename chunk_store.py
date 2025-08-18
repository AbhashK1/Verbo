import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


'''def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([text])
    return chunks


def save_chunks(chunks, path='data/chunks.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([chunk.page_content for chunk in chunks], f)


def load_chunks(path='data/chunks.json'):
    with open(path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return texts'''


# Updating chunking+storing as jsonl
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([text])
    # return list of dicts for each chunk
    return [{"id": f"chunk_{i}", "text": c.page_content} for i, c in enumerate(chunks)]


def save_chunks_jsonl(chunks, path="data/chunks.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path="data/chunks.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

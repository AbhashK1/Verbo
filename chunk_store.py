import json
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([text])
    return chunks


def save_chunks(chunks, path='data/chunks.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([chunk.page_content for chunk in chunks], f)


def load_chunks(path='data/chunks.json'):
    with open(path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    return texts

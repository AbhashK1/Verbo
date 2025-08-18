from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

'''model = SentenceTransformer('all-MiniLM-L6-v2')
index_path = 'data/faiss.index'
meta_path = 'data/docs.pkl'


def create_faiss_index(docs):
    embeddings = model.encode(docs)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    with open(meta_path, 'wb') as f:
        pickle.dump(docs, f)
    faiss.write_index(index, index_path)

def load_faiss_index():
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or document metadata not found.")
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        docs = pickle.load(f)
    return index, docs'''


# Update append text to existing text
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL_NAME)

INDEX_PATH = "data/indexes/faiss.index"
META_PATH = "data/indexes/docs_meta.pkl"


def create_faiss_index(docs_texts):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    embeddings = model.encode(docs_texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(docs_texts, f)
    return index


def load_faiss_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Index or metadata not found. Build index first.")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs


def add_to_index(new_texts):
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index, docs = load_faiss_index()
        docs.extend(new_texts)
        return create_faiss_index(docs)
    else:
        return create_faiss_index(new_texts)

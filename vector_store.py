from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
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
    return index, docs

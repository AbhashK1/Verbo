import numpy as np
import torch
from transformers import pipeline
from sentence_transformers import CrossEncoder
from vector_store import model as embed_model

# device = 0 if torch.cuda.is_available() else -1
'''if torch.cuda.is_available():
    device = 0
    print("Using CUDA for generation.")
else:
    device = -1
    print("Using CPU for generation.")

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=device)  #  flan-t5-large


def retrieve_top_k(query, index, docs, k=3):
    query_vector = model.encode([query])[0]
    D, I = index.search(np.array([query_vector]), k)
    return [docs[i] for i in I[0]]


def format_answer(raw_text):
    # Fix bullet styles
    formatted = raw_text.replace("•", "-").replace("o ", "- ").replace("Excerpt", "\n### Excerpt")

    # Optional: add newlines after colons for readability
    formatted = formatted.replace(": ", ":\n")

    return formatted.strip()


def generate_answer(query, context_docs):
    # context = "\n".join(context_docs)
    context = "\n\n".join([f"Excerpt {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])  # for better LLM attention
    # input_text = f"Answer the question based on the context:\n{context}\n\nQuestion: {query}"
    input_text = (
        f"You are a helpful assistant. Read the following content and provide an in-depth, complete, and well-structured answer. Present the answer in well-formatted bullet points if applicable.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )  # prompt engineering
    # result = qa_pipeline(input_text, max_length=256)[0]['generated_text']
    result = qa_pipeline(input_text, max_length=512, do_sample=True, temperature=0.7)[0]['generated_text']  # altering for longer answers
    return format_answer(result)'''


# Update added cross encoder
device = 0 if torch.cuda.is_available() else -1

GEN_MODEL = "google/flan-t5-base"
qa_pipeline = pipeline("text2text-generation", model=GEN_MODEL, device=device)

# optional
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=0 if torch.cuda.is_available() else "cpu")
except Exception:
    reranker = None


def retrieve_top_k(query, index, docs, k=5):
    q_vec = embed_model.encode([query])[0]
    D, I = index.search(np.array([q_vec]), k)
    candidate_texts = [docs[i] for i in I[0]]
    # re ranking here
    if reranker:
        pairs = [[query, t] for t in candidate_texts]
        scores = reranker.predict(pairs)
        ranked = [t for _, t in sorted(zip(scores, candidate_texts), key=lambda x: x[0], reverse=True)]
        return ranked
    return candidate_texts


def format_answer_md(raw_text):
    formatted = raw_text.replace("•", "-").replace("o ", "- ").replace("Excerpt", "\n### Excerpt")
    formatted = formatted.replace(": ", ":\n")
    return formatted.strip()


def generate_answer(query, context_docs, depth="detailed"):
    # adding context here
    context = "\n\n".join([f"Excerpt {i+1}:\n{d}" for i, d in enumerate(context_docs)])
    prompt = (
        "You are a helpful assistant. Use the context to answer the question. "
        "Provide: (1) one-line short answer, (2) medium summary, (3) detailed answer with bullet points, and (4) sources.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    max_len = 384
    if depth == "short":
        max_len = 128
    elif depth == "medium":
        max_len = 256
    else:
        max_len = 512

    out = qa_pipeline(prompt, max_length=max_len, do_sample=True, temperature=0.6)[0]["generated_text"]
    return format_answer_md(out)

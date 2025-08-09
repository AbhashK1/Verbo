import numpy as np
from transformers import pipeline
from vector_store import model
import torch

# device = 0 if torch.cuda.is_available() else -1
if torch.cuda.is_available():
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
    return format_answer(result)

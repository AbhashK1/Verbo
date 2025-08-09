from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = "D:/Tools/Tesseract/tesseract.exe"

image = Image.open('page1.jpg')
text = pytesseract.image_to_string(image)


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([text])

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = [chunk.page_content for chunk in chunks]
embeddings = model.encode(docs)

# FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

query = input("Ask a question:")

query_vector = model.encode([query])[0]
D, I = index.search(np.array([query_vector]), k=3)

retrieved_docs = [docs[i] for i in I[0]]
context = "\n".join(retrieved_docs)
response = qa_pipeline(f"Answer based on context:\n{context}\n\nQuestion: {query}")


print(response)

import streamlit as st
import os
from ocr_utils import extract_text_from_image, convert_pdf_to_images
from chunk_store import chunk_text, save_chunks, load_chunks
from vector_store import create_faiss_index, load_faiss_index
from rag_chain import retrieve_top_k, generate_answer

st.title("📚 Ask Questions from Scanned Documents")

if 'index_loaded' not in st.session_state:
    st.session_state['index_loaded'] = False

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:
    file_path = os.path.join(uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith("pdf"):
        image_paths = convert_pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    full_text = ""
    for img_path in image_paths:
        full_text += extract_text_from_image(img_path) + "\n"

    chunks = chunk_text(full_text)
    save_chunks(chunks)
    docs = [chunk.page_content for chunk in chunks]
    create_faiss_index(docs)
    st.session_state['index_loaded'] = True
    st.success("Document processed and indexed!")

if st.session_state['index_loaded']:
    query = st.text_input("Ask a question")
    if st.button("Get Answer") and query:
        index, docs = load_faiss_index()
        top_docs = retrieve_top_k(query, index, docs)
        answer = generate_answer(query, top_docs)
        print(answer)
        st.markdown(f"**Answer:** {answer}")

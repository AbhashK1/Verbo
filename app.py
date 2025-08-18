import streamlit as st
import os
from ocr_utils import extract_text_from_file
from doc_analysis import analyze_document
from chunk_store import chunk_text, save_chunks_jsonl
from vector_store import create_faiss_index, load_faiss_index
from rag_chain import retrieve_top_k, generate_answer

st.set_page_config(layout="wide", page_title="Ask Questions from Scanned Documents")
st.title("📚 Ask Questions from Scanned Documents")

os.makedirs("data/images", exist_ok=True)
os.makedirs("data/indexes", exist_ok=True)


if 'index_ready' not in st.session_state:
    st.session_state['index_ready'] = False
if 'ocr_text' not in st.session_state:
    st.session_state['ocr_text'] = None
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = None
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None


col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])
    run_analysis = st.checkbox("Run Document Analysis after ingestion", value=True)
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=500)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=300, value=50)

    # Clear/reset button
    if st.button("🗑️ Clear uploaded document"):
        st.session_state['uploaded_file'] = None
        st.session_state['ocr_text'] = None
        st.session_state['analysis'] = None
        st.session_state['index_ready'] = False
        st.success("Session cleared. You can upload a new document now.")

    if uploaded and uploaded.name != st.session_state['uploaded_file']:
        # Reset state for new file
        st.session_state['uploaded_file'] = uploaded.name
        st.session_state['ocr_text'] = None
        st.session_state['analysis'] = None
        st.session_state['index_ready'] = False

        save_path = os.path.join("data", uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name} to disk.")

        with st.spinner("Running OCR..."):
            st.session_state['ocr_text'] = extract_text_from_file(save_path)
        st.success("OCR complete.")

        if run_analysis:
            with st.spinner("Running document analysis..."):
                st.session_state['analysis'] = analyze_document(
                    st.session_state['ocr_text'],
                    file_path=save_path,
                    out_path=f"data/{uploaded.name}.analysis.json"
                )
            st.success("Analysis complete.")

        with st.spinner("Chunking and indexing..."):
            chunks = chunk_text(st.session_state['ocr_text'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            save_chunks_jsonl(chunks, path=f"data/{uploaded.name}.chunks.jsonl")
            docs = [c["text"] for c in chunks]
            create_faiss_index(docs)
        st.session_state['index_ready'] = True

    if st.session_state['analysis']:
        with st.expander("📊 Document Analysis (Preview)", expanded=False):
            st.json(st.session_state['analysis'])

with col2:
    st.markdown("### Query & Answer")
    depth = st.selectbox("Answer depth", ["short", "medium", "detailed"], index=2)
    query = st.text_input("Ask a question about the uploaded document(s):")
    get_answer = st.button("Get Answer")

    if st.session_state['index_ready'] or os.path.exists("data/indexes/faiss.index"):
        try:
            index, docs = load_faiss_index()
            st.success("✅ Document processed and indexed. Index ready.")
            st.session_state['index_ready'] = True
        except Exception:
            st.error("⚠️ No index available. Ingest a document first.")
            index, docs = None, []

    if get_answer and query:
        if not st.session_state.get('index_ready', False):
            st.error("Index not ready. Upload and index a document first.")
        else:
            with st.spinner("Retrieving relevant passages..."):
                top_docs = retrieve_top_k(query, index, docs, k=6)
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, top_docs, depth=depth)

            st.markdown("### Answer")
            st.markdown(answer)

            with st.expander("Show retrieved passages"):
                for i, d in enumerate(top_docs):
                    st.write(f"**Passage {i+1}:**")
                    st.write(d[:800] + ("..." if len(d) > 800 else ""))

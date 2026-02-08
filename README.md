# Verbo: RAG + OCR + Document Analysis

Verbo is an advanced AI-powered document intelligence system designed to bridge the gap between static documents and actionable insights. By combining Optical Character Recognition (OCR), Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG), Verbo allows users to transform scanned PDFs and images into searchable, interactive data.

## Key Features
 - Intelligent OCR Integration: Seamlessly extracts text from scanned PDFs and images using Tesseract OCR.
 - Deep Document Analysis: Automatically generates summaries, extracts keywords, identifies named entities (NER), and performs sentiment analysis using Stanza and Hugging Face Transformers.
 - Semantic Search & Indexing: Utilizes Sentence-Transformers for high-dimensional embeddings and FAISS for lightning-fast vector similarity search.
 - Context-Aware RAG Pipeline: Uses a Retrieval-Augmented Generation workflow to answer user queries based specifically on the uploaded document's content.
 - Intuitive Two-Column UI: A sleek Streamlit interface featuring a side-by-side view:
     - Left Panel: Upload, OCR processing, and analytical insights.
     - Right Panel: Interactive Q&A chat interface.
 - Smart State Management: Optimized session handling to prevent redundant OCR/Analysis processing, ensuring a smooth user experience.

---

## Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | Streamlit |
| OCR | Tesseract OCR |
| NLP Engine | Hugging Face Transformers, Stanza, BERT |
| Embeddings | Sentence-Transformers |
| Summarization | distilbart-cnn-12-6 |
| Vector Store | FAISS |
| Language Models | Flan-t5 |

---

## Installation

### Prerequisites
 - Python 3.9 or higher
 - Tesseract OCR engine installed on your system.
     - Ubuntu: ```sudo apt install tesseract-ocr```
     - macOS: ```brew install tesseract```
     - Windows: [Download the installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup

  1. Clone the repository:

  ```bash
  git clone https://github.com/AbhashK1/Verbo.git
  cd Verbo
  ```

  2. Create a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

  3. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

  4. Run the application:

  ```bash
  streamlit run app.py
  ```

---

## How It Works
 - Ingestion: Upload a PDF or Image.
 - Extraction: The system detects if the file is text-based or an image and applies OCR if necessary.
 - Analysis: The NLP pipeline processes the text to extract metadata, sentiment, and key entities.
 - Vectorization: Text is split into semantic chunks and converted into embeddings stored in a local FAISS index.
 - Querying: When you ask a question, Verbo retrieves the most relevant chunks and passes them to the LLM to generate a precise answer.


Author:
Abhash Kumar

# AskPDF

A local, privacy-first **Retrieval-Augmented Generation (RAG)** system that lets you upload PDF documents and ask natural language questions about their contents, powered by **Google Gemini**, **ChromaDB**, and a **Gradio** web interface.

### ⚝ Tech Stack
- **LLM Backend**: Google Gemini (via `langchain-google-genai`), configurable model & thinking level
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (HuggingFace Sentence Transformers, runs fully locally)
- **Vector Store**: ChromaDB (persistent, cosine-similarity search)
- **PDF Parsing**: pdfplumber (native text extraction + optional table-to-markdown conversion)
- **OCR Fallback**: pytesseract + Tesseract binary (for scanned/image-only PDFs)
- **RAG Orchestration**: LangChain Core (`langchain-core`, `langchain-text-splitters`)
- **Web UI**: Gradio 5.x (dark-mode, multi-document sidebar, streaming chat)
- **Configuration**: `python-dotenv`, dataclass-based settings

### ⚝ System Architecture <br>
The pipeline is structured around an automated ingest, embed, retrieve, and generate loop:
- **PDF Loader**: Opens documents with `pdfplumber`, parses per-page text, extracts markdown tables, and falls back to OCR via `pytesseract` for scanned files.
- **Vector Store & Ingestion**: Splits pages into overlapping chunks with `RecursiveCharacterTextSplitter`, generates dense vector embeddings via `BAAI/bge-small-en-v1.5`, and stores them in persistent per-file Chroma collections keyed by SHA-256 hash.
- **QA & Retrieval Engine**: Retrieves top-k semantically relevant chunks across all active session documents, synthesizes context, and streams responses with inline page citations.
- **Gradio Application**: Interactive dual-pane interface with real-time file upload tracking, per-document removal, dark theme styling, and responsive streaming chat.

```
                      +-----------------------------+
                      |         User Browser        |
                      |   (Gradio Web Interface)    |
                      +--------------+--------------+
                                     |
                         PDF Upload  |  Question Prompt
                                     v
                      +-----------------------------+
                      |          app.py             | :7860
                      |   - Document Session State  |
                      |   - Multi-Doc Progress Sync |
                      +-------+-------------+-------+
                              |             |
                 Extract Text |             | Vector Query
                              v             v
             +--------------------+     +---------------------+
             |  src/pdf_loader.py |     |      src/qa.py      |
             |  - pdfplumber      |     |  - Multi-doc Top-K  |
             |  - pytesseract OCR |     |  - Prompt Synthesis |
             +---------+----------+     +----------+----------+
                       |                           |
                Chunks |                           | Stream Context
                       v                           v
             +--------------------+     +---------------------+
             | src/vectorstore.py |     |  Google Gemini API  |
             |  - bge-small-en    |     |  - gemini-2.0-flash |
             |  - ChromaDB Store  |     |  - Token Streaming  |
             +--------------------+     +---------------------+
```

### ⚝ Installation & Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Susheyyy/Ask-PDF.git
   cd Ask-PDF
   ```
2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Optional)* For scanned document OCR support, ensure the `tesseract` binary is installed on your operating system.
4. **Configure Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here

   # Optional overrides (defaults shown)
   LLM_MODEL=gemini-2.0-flash
   LLM_TEMPERATURE=1.0
   LLM_THINKING=low
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K=6
   MAX_PDF_MB=25
   MAX_PAGES=300
   MAX_DOCS=10
   OCR_ENABLED=true
   EXTRACT_TABLES=false
   GRADIO_SHARE=false
   ```
5. **Start the Application**
   ```bash
   python app.py
   ```

### ⚝ How to Use
- **Access Web Interface**: [http://127.0.0.1:7860](http://127.0.0.1:7860)
- **Upload Documents**: Drag & drop or browse PDF files in the sidebar upload zone. Each document is parsed and indexed with live progress updates.
- **Query Your PDFs**: Type natural language questions in the chat console. The assistant provides streaming responses with inline citations (e.g., `(p. 3)` or `(guide.pdf, p. 5)`).
- **Multi-Document Synthesis**: Add multiple PDFs; retrieval aggregates context across all indexed documents simultaneously.
- **Document Management**: Remove individual documents on the fly via the `×` button or reset your workspace using `Clear all`.

---

*Version 1 of this project, built using only jupyter notebook:*  
https://github.com/Susheyyy/Ask-PDF_v1

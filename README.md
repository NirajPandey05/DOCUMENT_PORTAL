# Document Portal

A robust, multimodal Retrieval-Augmented Generation (RAG) platform with Gemini LLM/Vision, FAISS, and a user-friendly web UI. Supports ingestion, chat, and analysis for all major document types and website URLs.

---

## Features

- **Multimodal Ingestion:**
  - Supports PDF, DOCX, TXT, PPT, PPTX, XLSX, XLS, CSV, MD, and website URLs.
  - Extracts text, tables, and images (with OCR for images).
- **RAG Chat & Analysis:**
  - Chat with your documents using Gemini, Groq, OpenAI, Claude, Hugging Face, or Ollama LLMs.
  - FAISS-based vector search for fast retrieval.
- **Evaluation:**
  - Integrated DeepEval for LLM response quality and RAG-specific metrics.
- **User Authentication:**
  - Simple login/registration with file-based user store.
  - Passwords are securely hashed.
- **Frontend:**
  - Modern HTML/JS UI for chat, upload, and analysis.
- **Backend:**
  - FastAPI, LangChain, Gemini LLM/Vision, FAISS, Unstructured, python-pptx, openpyxl, tabula-py, pdfplumber, pytesseract, Pillow.

---

## Quick Start

```bash
# Clone the repository
https://github.com/sunnysavita10/document_portal.git
cd document_portal

# (Recommended) Create and activate a Python 3.10+ environment
conda create -p ./env python=3.10 -y
conda activate ./env

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api.main:app --reload

# Visit http://localhost:8000 in your browser
```

---

## Usage

1. **Register a new profile** or login with your credentials.
2. **Upload documents** (PDF, DOCX, PPTX, XLSX, CSV, etc.) or enter a website URL.
3. **Chat** with your documents, analyze, or compare them.
4. **Evaluate** LLM responses with built-in metrics.

---

## API Keys

- Set your API keys in the `.env` file or as environment variables:
  - `GROQ_API_KEY` ([Get key](https://console.groq.com/keys))
  - `GOOGLE_API_KEY` ([Get key](https://aistudio.google.com/apikey))
  - `OPENAI_API_KEY`, `CLAUDE_API_KEY`, etc. as needed

---

## Supported Document Types

- `.pdf`, `.docx`, `.txt`, `.ppt`, `.pptx`, `.xlsx`, `.xls`, `.csv`, `.md`, and website URLs

---

## Project Structure

- `api/` - FastAPI backend
- `templates/` - HTML templates (login, register, chat, etc.)
- `static/` - CSS/JS/static assets
- `utils/` - Document loaders, file I/O, logging
- `src/` - Core ingestion, analysis, and chat logic
- `requirements.txt` - Python dependencies

---

## Contributing

Pull requests and issues are welcome!

---

## License

MIT License



import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
import json
import hashlib
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter,read_pdf_via_handler,extract_text_and_tables,load_documents
from utils.eval_metrics import ResponseEvaluator

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home page: login or create new profile
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    resp = templates.TemplateResponse("home.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp


# --- User Auth Helpers ---
USER_DB_PATH = BASE_DIR / "api" / "users.json"
def load_users():
    try:
        with open(USER_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
def save_users(users):
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
def hash_pw(pw):
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    users = load_users()
    if not username or not password:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Please enter username and password."})
    if username not in users:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Profile not found. Please create a new profile."})
    if users[username] != hash_pw(password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Incorrect password."})
    # Success
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/index", status_code=303)


# Registration page (GET/POST)
@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})

@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request):
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    users = load_users()
    if not username or not password:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Please provide username and password."})
    if username in users:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Profile already exists. Please login."})
    users[username] = hash_pw(password)
    save_users(users)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/login", status_code=303)

# Main app page (after login)
@app.get("/index", response_class=HTMLResponse)
async def serve_ui(request: Request):
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        # Use load_documents to get all text/tables/images, then extract text/tables for LLM context
        docs = load_documents([Path(saved_path)])
        from logger import GLOBAL_LOGGER as log
        log.info("Loaded document page_contents", page_contents=[d.page_content for d in docs])
        context = extract_text_and_tables(docs, ocr_images=True)
        # Truncate context to avoid exceeding LLM token limits (e.g., 6000 tokens)
        context = context[:5000] if context else None
        log.info("LLM context for analysis (FULL)", context=context)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(context)

        # Evaluate the analysis result using DeepEval
        evaluator = ResponseEvaluator()
        # Use the first 5000 chars of the document as context for evaluation (more context for LLM)
        context_eval = context[:5000] if context else None
        # If result is a dict with 'summary' or 'result', use that as response
        response = result.get('summary') if isinstance(result, dict) and 'summary' in result else (
            result.get('result') if isinstance(result, dict) and 'result' in result else str(result)
        )
        evaluation = evaluator.evaluate_response(
            question="Analyze document", response=response, context=context_eval
        )
        return JSONResponse(content={"result": result, "evaluation": evaluation})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        # Use load_documents to get all text/tables/images for both reference and actual
        ref_docs = load_documents([Path(ref_path)])
        act_docs = load_documents([Path(act_path)])
        # Extract all text and tables for LLM context
        ref_context = extract_text_and_tables(ref_docs, ocr_images=True)
        act_context = extract_text_and_tables(act_docs, ocr_images=True)
        from logger import GLOBAL_LOGGER as log
        log.info("LLM context for comparison (FULL)", ref_context=ref_context, act_context=act_context)
        # Combine for comparison context
        combined_context = f"<<REFERENCE_DOCUMENTS>>\n{ref_context}\n\n<<ACTUAL_DOCUMENTS>>\n{act_context}"
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_context)

        # Evaluate the comparison result using DeepEval
        evaluator = ResponseEvaluator()
        # Use the first 5000 chars of the combined context for evaluation
        context_eval = combined_context[:5000] if combined_context else None
        # Use the first row of the DataFrame as the response (stringified)
        response = str(df.iloc[0].to_dict()) if not df.empty else "No comparison result."
        evaluation = evaluator.evaluate_response(
            question="Compare documents", response=response, context=context_eval
        )
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id, "evaluation": evaluation}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------

from fastapi import Form
from typing import List, Optional

@app.post("/chat/index")
async def chat_build_index(
    files: Optional[List[UploadFile]] = File(None),
    website_urls: Optional[List[str]] = Form(None),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files] if files else []
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        ci.built_retriver(
            wrapped,
            website_urls=website_urls,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k,
        )
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])

        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")






# command for executing the fast api
# uvicorn api.main:app --reload    
# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
# uvicorn api.main:app --port 8083 --reload
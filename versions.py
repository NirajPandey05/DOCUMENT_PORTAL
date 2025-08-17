import importlib.metadata
# packages = [
#     "langchain",
#     "python-dotenv",
#     "ipykernel",
#     "langchain_groq",
#     "langchain_google_genai",
#     "langchain-community",
#     "faiss-cpu",
#     "structlog",
#     "PyMuPDF",
#     "pylint",
#     "langchain-core",
#     "pytest",
#     "streamlit",
#     "fastapi",
#     "uvicorn",
#     "python-multipart",
#     "docx2txt"
# ]
packages = [
"langchain",
"python-dotenv",
"ipykernel",
"langchain_groq",
"langchain_google_genai",
"langchain_community",
"langchain_text_splitters",
"pypdf",
"faiss-cpu",
"tqdm",
"structlog",
"PyMuPDF",
"pandas",
"pydantic",
"langchain_core",
"docx",
"streamlit",
"pytest",
"langchain-core[tracing]",
"python-multipart",
"doc2txt",
"docx2txt",
"uvicorn",
"fastapi"
]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")

# # serve static & templates
# app.mount("/static", StaticFiles(directory="../static"), name="static")
# templates = Jinja2Templates(directory="../templates")
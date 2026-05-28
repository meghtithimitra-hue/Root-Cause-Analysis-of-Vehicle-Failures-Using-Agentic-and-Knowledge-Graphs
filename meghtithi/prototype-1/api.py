from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, os
from query import answer_query

app = FastAPI(title="Automotive KG Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 8

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    save_path = f"./data/{file.filename}"
    os.makedirs("./data", exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import ingest
    pages  = ingest.parse_pdf(save_path)
    chunks = ingest.build_graph(pages, doc_title=file.filename)
    ingest.extract_entities(chunks)
    ingest.store_embeddings(chunks)

    return {
        "message": f"✅ '{file.filename}' ingested successfully",
        "chunks": len(chunks)
    }

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    result = answer_query(req.question, req.top_k)
    return result

@app.get("/health")
def health():
    return {"status": "running"}

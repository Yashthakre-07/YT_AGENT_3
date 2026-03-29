# =========================
# YouTube AI Agent - FastAPI
# Stack: Groq LLM + HuggingFace Embeddings + FAISS + LangChain
# =========================

import re
import os
import subprocess
import sys

# Auto-install dependencies on first run
def install_deps():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

install_deps()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# =========================
# API KEY
# =========================
os.environ["GROQ_API_KEY"] = "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"  # 🔑 Replace with your Groq key
# Get free key at: https://console.groq.com

# =========================
# FASTAPI SETUP
# =========================
app = FastAPI(title="YouTube AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST MODEL
# =========================
class QueryRequest(BaseModel):
    url: str
    question: str
    language: str = "English"

# =========================
# HELPER: GET VIDEO ID
# =========================
def get_video_id(url: str) -> str | None:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

# =========================
# HELPER: FETCH TRANSCRIPT
# =========================
def get_transcript(video_id: str) -> dict:
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        available = []
        for t in transcript_list:
            label = f"{t.language} ({t.language_code})" + (" [auto]" if t.is_generated else " [manual]")
            available.append(label)

        # Try English first
        try:
            fetched = ytt_api.fetch(video_id, languages=['en'])
            transcript = " ".join(chunk.text for chunk in fetched)
            return {"success": True, "transcript": transcript, "available": available, "language_used": "English"}
        except Exception:
            pass

        # Fallback: any available language
        for t in ytt_api.list(video_id):
            try:
                fetched = t.fetch()
                transcript = " ".join(chunk.text for chunk in fetched)
                return {"success": True, "transcript": transcript, "available": available, "language_used": t.language}
            except Exception:
                continue

        return {"success": False, "error": "No transcript could be fetched."}

    except TranscriptsDisabled:
        return {"success": False, "error": "Transcripts are disabled for this video."}
    except NoTranscriptFound:
        return {"success": False, "error": "No transcript found for this video."}
    except Exception as e:
        return {"success": False, "error": str(e)}

# =========================
# HELPER: SPLIT TEXT
# =========================
def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# =========================
# HELPER: VECTOR STORE
# =========================
def create_vector_store(chunks: list[str]) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

# =========================
# HELPER: ASK QUESTION
# =========================
def ask_question(db: FAISS, query: str, language: str) -> str:
    docs = db.similarity_search(query, k=3)
    context = " ".join([d.page_content for d in docs])
    lang_instruction = "Answer in Hindi." if language.lower() == "hindi" else "Answer in English."

    prompt = f"""
You are an intelligent YouTube Video Assistant. Your job is to answer questions strictly based on the transcript context provided.

═══════════════════════════════════════
CONTEXT (from video transcript):
───────────────────────────────────────
{context}
═══════════════════════════════════════

QUESTION:
{query}

═══════════════════════════════════════
INSTRUCTIONS:
───────────────────────────────────────
1. Answer ONLY using the context above. Do NOT use outside knowledge.
2. If the answer is not found in the context, respond exactly with:
   "This information was not covered in the video."
3. {lang_instruction}
4. Always follow the output format below.

═══════════════════════════════════════
OUTPUT FORMAT:
───────────────────────────────────────
📌 Direct Answer:
<Give a concise 1-3 sentence direct answer here>

📝 Detailed Explanation:
<Break down the answer in bullet points with key details from the transcript>

💡 Key Takeaway:
<One final sentence summarizing the most important point>
═══════════════════════════════════════
"""

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    response = llm.invoke(prompt)
    return response.content

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask(req: QueryRequest):
    video_id = get_video_id(req.url)
    if not video_id:
        return JSONResponse({"success": False, "error": "Invalid YouTube URL."}, status_code=400)

    result = get_transcript(video_id)
    if not result["success"]:
        return JSONResponse({"success": False, "error": result["error"]}, status_code=400)

    chunks = split_text(result["transcript"])
    db = create_vector_store(chunks)
    answer = ask_question(db, req.question, req.language)

    return JSONResponse({
        "success": True,
        "answer": answer,
        "video_id": video_id,
        "transcript_length": len(result["transcript"]),
        "chunks": len(chunks),
        "language_used": result.get("language_used", "Unknown"),
        "available_transcripts": result.get("available", [])
    })
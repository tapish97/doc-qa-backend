from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from pypdf import PdfReader
from uuid import uuid4
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from uuid import uuid4
import os
import shutil
import time
import gc

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:5173"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store user sessions
sessions = {}  # {session_id: vector_db}

@app.get("/wake-up")
def wake_up():
    return {"status": "Backend is awake!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = None):
    """
    Uploads and processes a PDF file.
    If session_id is provided and exists, it overwrites the session in memory.
    Otherwise, creates a new session.
    """

    # Generate a new session if one isn't passed
    if not session_id or session_id not in sessions:
        session_id = str(uuid4())

    try:
        # Read and extract PDF text
        file.file.seek(0)
        reader = PdfReader(file.file)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in PDF.")

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Text extraction succeeded, but chunking failed.")

        # Create an in-memory vector store (no persist_directory = no file locking)
        vector_db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=f"docs_{session_id}"
        )

        # Overwrite or register in-memory session
        sessions[session_id] = vector_db

        return {
            "session_id": session_id,
            "message": f"Processed {len(chunks)} chunks"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class QuestionRequest(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """ Retrieves relevant document chunks and generates an AI answer. """

    if request.session_id not in sessions:
        raise HTTPException(status_code=400, detail="Invalid session. Upload a document first.")

    try:
        vector_db = sessions[request.session_id]

        # Retrieve relevant chunks
        docs = vector_db.similarity_search(request.question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer with Groq's LLM
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )

        response = llm.invoke(f"Context:\n{context}\n\nQuestion: {request.question}\nAnswer:")

        return {"answer": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

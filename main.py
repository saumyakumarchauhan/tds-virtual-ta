# main.py
# ✅ Build main.py (or app.py) with FastAPI that calls the above logic along with image OCR support
# main.py

# main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import httpx
import json
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from slugify import slugify 
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_API_URL = os.getenv("AIPIPE_API_URL")

app = FastAPI()

# Optional (Windows users): Uncomment and set the path to tesseract.exe
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
with open("embedding_combined.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)
index = faiss.read_index("faiss_index.idx")

# Request and response models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]
def build_discourse_url(window):
    base_url = "https://discourse.onlinedegree.iitm.ac.in"
    slug = slugify(window.get("topic_title", "discussion"))
    topic_id = window.get("topic_id", "")
    post_number = window.get("post_number", 1)
    return f"{base_url}/t/{slug}/{topic_id}/{post_number}"

def retrieve(query, top_k=10):
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(np.array([query_emb]), top_k)

    results = []
    used_urls = set()

    for score, idx in zip(D[0], I[0]):
        window = embedding_data[idx]
        content = window.get("content", "")
        url = window.get("url") or build_discourse_url(window)
        used_urls.add(url)

        results.append({
            "score": float(score),
            "topic_id": window.get("topic_id", ""),
            "topic_title": window.get("topic_title", "Untitled"),
            "combined_text": content[:500] + "...",
            "url": url
        })

    # Ensure original URL is included if referenced in the query
    for window in embedding_data:
        if window.get("url") and window["url"] in query and window["url"] not in used_urls:
            content = window.get("content", "")
            results.append({
                "score": 1.0,
                "topic_id": window.get("topic_id", ""),
                "topic_title": window.get("topic_title", "Untitled"),
                "combined_text": content[:500] + "...",
                "url": window["url"]
            })
            break

    return results





# Use AIPipe proxy to generate an answer
def generate_answer(query: str, context_texts: list) -> str:
    context = "\n\n---\n\n".join(context_texts)
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on forum discussions."},
            {"role": "user", "content": f"Based on these forum excerpts:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    response = httpx.post(
        AIPIPE_API_URL,
        headers=headers,
        json=payload,
        timeout=25.0
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise HTTPException(status_code=500, detail=f"AIPipe Error: {response.text}")

# Endpoint to handle question + optional image
@app.post("/api/", response_model=QueryResponse)
async def handle_query(data: QueryRequest):
    question = data.question.strip()
    ocr_text = ""

    # OCR processing if image provided
    if data.image:
        try:
            decoded_img = base64.b64decode(data.image)
            img = Image.open(BytesIO(decoded_img))
            img = img.convert("L")  # Convert to grayscale
            ocr_text = pytesseract.image_to_string(img).strip()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image provided or OCR failed")

    # Retrieve relevant context
    results = retrieve(question, top_k=10)
    context_texts = [r['combined_text'] for r in results]

    # Add OCR text to context if available
    if ocr_text:
        context_texts.insert(0, f"[Text extracted from image]:\n{ocr_text}")

    # Generate answer
    answer = generate_answer(question, context_texts)
    links = [{"url": r["url"], "text": r["topic_title"]} for r in results]

    return {"answer": answer, "links": links}

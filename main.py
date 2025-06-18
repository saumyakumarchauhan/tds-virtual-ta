# main.py
# ✅ Build main.py (or app.py) with FastAPI that calls the above logic along with image OCR support
# main.py

# main.py

import os
import json
import base64
import numpy as np
import faiss
import httpx
from io import BytesIO
from PIL import Image
import pytesseract
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch
import re

# Load environment variables
load_dotenv()
AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_API_URL = os.getenv("AIPIPE_API_URL")

# Model loading with GPU/CPU support
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device=device)

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.idx")
with open("embedding_combined.json", "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

# FastAPI app
app = FastAPI()

# Request & Response Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# OCR Function
def extract_ocr_text(image_base64):
    try:
        decoded_img = base64.b64decode(image_base64)
        img = Image.open(BytesIO(decoded_img)).convert("L")
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        return f"OCR failed: {str(e)}"

# Semantic Search
def retrieve(query, top_k=10):
    query_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    query_emb = np.array(query_emb, dtype="float32").reshape(1, -1)

    print("Query embedding shape:", query_emb.shape)
    print("Index expected dimension:", index.d)

    D, I = index.search(query_emb, top_k)

    results = []
    seen_urls = set()

    for score, idx in zip(D[0], I[0]):
        window = embedding_data[idx]
        text = window.get("combined_text", "")
        url = window.get("url", "#")
        url = re.sub(r'/\d+$', '', url) 
        if url in seen_urls:
            continue
        seen_urls.add(url)

        results.append({
            "score": float(score),
            "topic_title": window.get("topic_title", "Untitled"),
            "combined_text": text[:700] + ("..." if len(text) > 700 else ""),
            "url": url
        })

    return results


# LLM-based Answer Generation via AIPipe
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
            {"role": "user", "content": f"Based on the following:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    try:
        response = httpx.post(AIPIPE_API_URL, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Sorry, no response generated.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AIPipe Error: {str(e)}")


# API Endpoint
@app.post("/api/", response_model=QueryResponse)
async def handle_query(data: QueryRequest):
    query = data.question.strip() if data.question else ""
    ocr_text = ""

    if not query and not data.image:
        raise HTTPException(status_code=400, detail="Please provide either a question or an image.")

    if data.image:
        ocr_text = extract_ocr_text(data.image)
        query = f"{query} {ocr_text}".strip()

    # Semantic Search
    results = retrieve(query, top_k=10)
    context_texts = [r['combined_text'] for r in results]

    if ocr_text:
        context_texts.insert(0, f"[Extracted from image]: {ocr_text}")
    if not context_texts:
        context_texts.append("No context available.")

    try:
        answer = generate_answer(query, context_texts)
    except Exception as e:
        answer = f"Error during answer generation: {str(e)}"

    # Preserve full link titles and answers
    links = [{"url": r["url"], "text": r["topic_title"]} for r in results]

    return QueryResponse(answer=answer, links=[Link(**link) for link in links])

# this will combine the two embedding json files to single called "embedding_combined.json"
# create_faiss_index.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === Load the model ===
model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
print(" Model loaded")

# === Load Discourse embeddings ===
with open("embedding_data.json", "r", encoding="utf-8") as f:
    discourse_data = json.load(f)

# === Load Markdown embeddings ===
with open("embedding_md_data.json", "r", encoding="utf-8") as f:
    md_data = json.load(f)

# === Prepare combined text and metadata ===
all_texts = []
metadata = []

# Process Discourse data
for item in discourse_data:
    text = item["combined_text"]
    all_texts.append(text)
    item["source"] = "discourse"
    metadata.append(item)

# Process Markdown data
for item in md_data:
    text = item["chunk"]
    all_texts.append(text)
    metadata.append({
        "combined_text": text,
        "topic_title": "TDS Docs",
        "topic_id": "md",
        "url": item["original_url"],
        "source": "markdown"
    })

# === Generate embeddings (normalized) ===
print(" Generating embeddings for", len(all_texts), "chunks...")
embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
print(" Embeddings shape:", embeddings.shape)

# === Create FAISS index (Inner Product for normalized vectors) ===
dimension = embeddings.shape[1]
print(" Creating FAISS index with dimension:", dimension)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# === Save FAISS index and metadata ===
faiss.write_index(index, "faiss_index.idx")
with open("embedding_combined.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(" FAISS index and combined metadata created successfully!")

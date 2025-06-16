# this will combine the two embedding json files to single called "embedding_combined.json"
# create_faiss_index.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load both JSONs
with open("embedding_data.json", "r", encoding="utf-8") as f1:
    discourse_data = json.load(f1)

with open("embedding_md_data.json", "r", encoding="utf-8") as f2:
    md_data = json.load(f2)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Collect all texts and metadata
all_texts = []
metadata = []

# Discourse
for item in discourse_data:
    all_texts.append(item["combined_text"])
    item["source"] = "discourse"
    metadata.append(item)

# Markdown
for item in md_data:
    all_texts.append(item["chunk"])
    metadata.append({
        "combined_text": item["chunk"],
        "topic_title": "TDS Docs",
        "topic_id": "md",
        "url": item["original_url"],
        "source": "markdown"
    })

# Compute embeddings
embeddings = model.encode(all_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# Create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Using Inner Product (dot product) with normalized vectors

# Add embeddings
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "faiss_index.idx")

# Save unified metadata
with open("embedding_combined.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("FAISS index and metadata created.")

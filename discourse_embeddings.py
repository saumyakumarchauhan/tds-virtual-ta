import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os

def clean_text(text):
    return " ".join(text.strip().split())

# === Load your data ===
with open("discourse_posts.json", "r", encoding="utf-8") as f:
    posts_data = json.load(f)

# === Group posts by topic_id ===
topics = {}
for post in posts_data:
    topic_id = post["topic_id"]
    if topic_id not in topics:
        topics[topic_id] = {"topic_title": post.get("topic_title", ""), "posts": []}
    topics[topic_id]["posts"].append(post)

for topic_id in topics:
    topics[topic_id]["posts"].sort(key=lambda p: p["post_number"])

print(f"Loaded {len(posts_data)} posts across {len(topics)} topics.")

model = SentenceTransformer("all-MiniLM-L6-v2")  # Faster than GritLM for large datasets

embedding_data = []
embeddings = []

def build_reply_map(posts):
    reply_map = defaultdict(list)
    posts_by_number = {}
    for post in posts:
        posts_by_number[post["post_number"]] = post
        parent = post.get("reply_to_post_number")
        reply_map[parent].append(post)
    return reply_map, posts_by_number

def extract_subthread(root_post_number, reply_map, posts_by_number):
    collected = []
    def dfs(post_num):
        post = posts_by_number[post_num]
        collected.append(post)
        for child in reply_map.get(post_num, []):
            dfs(child["post_number"])
    dfs(root_post_number)
    return collected

print("Generating embeddings and metadata...")

for topic_id, topic_data in tqdm(topics.items()):
    posts = topic_data["posts"]
    topic_title = topic_data["topic_title"]

    reply_map, posts_by_number = build_reply_map(posts)
    root_posts = reply_map[None]

    for root_post in root_posts:
        root_num = root_post["post_number"]
        subthread_posts = extract_subthread(root_num, reply_map, posts_by_number)

        combined_text = f"Topic title: {topic_title}\n\n"
        combined_text += "\n\n---\n\n".join(clean_text(p["content"]) for p in subthread_posts)

        emb = model.encode(combined_text, convert_to_numpy=True, normalize_embeddings=True)
        embedding_data.append({
            "topic_id": topic_id,
            "topic_title": topic_title,
            "root_post_number": root_num,
            "post_numbers": [p["post_number"] for p in subthread_posts],
            "combined_text": combined_text,
            "url": root_post.get("url", f"https://discourse.example.com/t/{topic_id}/{root_num}")
        })
        embeddings.append(emb)

# Save embedding metadata
with open("embedding_data.json", "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, indent=2)



print(" saved embedding_data.json.")

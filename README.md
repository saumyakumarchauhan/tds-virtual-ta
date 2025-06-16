# TDS Virtual TA (Teaching Assistant)

A FastAPI-based intelligent assistant built to support the **Tools in Data Science (TDS)** course offered by **IIT Madras**. This assistant can answer student queries by retrieving relevant forum and documentation content, powered by **semantic search**, **OCR (image-to-text)**, and **language models**.

---

## 🚀 Features

- 🔍 Semantic search using FAISS + Sentence Transformers
- 🧠 Language model-based answers (via AIPipe using `gpt-4.1-mini`)
- 🖼️ OCR support to handle image-based queries
- 🗂️ Support for both Discourse forum and TDS HTML documentation
- 📎 Source-aware linking to the original content

---

## 🧪 Scraping Overview

### 🔹 Discourse Forum

- **URL**: [TDS Discourse Forum](https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34)
- **Date Range**: 01 Jan 2025 – 14 Apr 2025
- **Method**:
  - Used `httpx` + `BeautifulSoup` to scrape posts.
  - Focused on extracting:
    - `topic_id`
    - `topic_title`
    - `post_number`
    - `content`
    - `tags`, `category_id`
  - Cleaned the HTML to remove PII and extra formatting.
  - Stored results in: `discourse_posts.json`

### 🔹 TDS HTML Documentation

- **URL**: [TDS Module Pages](https://tds.s-anand.net/#/2025-01/)
- **Method**:
  - Downloaded the page sources using Playwright.
  - Converted HTML to markdown using `markdownify`.
  - Broke content into chunks (~500 tokens) for embedding.
  - Stored chunks in: `embedding_md_data.json`

### 📦 Storage Format

- **Discourse**: JSON with metadata
- **HTML Docs**: Markdown in chunks (JSON-formatted)

---

## 🧠 Embedding Generation

### 🔸 For Forum & Markdown

- **Model Used**: `all-MiniLM-L6-v2` (from `sentence-transformers`)
- Normalized and converted embeddings to `float32`

### 🔸 Files Generated

- `embedding_data.json`: Embeddings for `discourse_posts.json`
- `embedding_md_data.json`: Embeddings for TDS documentation
- **Merged into**: `embedding_combined.json`

### 🔸 Indexing

- Used `faiss.IndexFlatIP` (Inner Product)
- Saved as: `faiss_index.idx`
- Created via: `create_faiss_index.py`

---

## ⚙️ Core Logic

### 🔸 `main.py`

- FastAPI app with `/api/` endpoint
- Accepts JSON with `question` and optional base64 `image`
- Performs:
  - OCR (via `pytesseract`) if image is provided
  - Embedding-based retrieval
  - Response generation using AIPipe API

---

## 🖼️ OCR Support

If the user supplies a base64-encoded image, `main.py` extracts the text using Tesseract OCR:

```python
img = Image.open(BytesIO(base64.b64decode(image)))
ocr_text = pytesseract.image_to_string(img)
```

The extracted text is included as context for the language model.

To generate `image.txt` from an image (e.g. PNG or WEBP):

```bash
base64 -w 0 project-tds-virtual-ta-q1.webp > image.txt
```

---

## 🔬 How to Test

### ✅ With Question + Image

```bash
curl -X POST http://127.0.0.1:8000/api/   -H "Content-Type: application/json"   -d "{"question": "What does the image say?", "image": "$(tr -d '\n' < image.txt)"}"
```

### ✅ With Question Only

```bash
curl -X POST http://127.0.0.1:8000/api/   -H "Content-Type: application/json"   -d '{"question": "What are deployment tools in TDS?"}'
```

---

## 📁 Project Structure

```
tds-virtual-ta/
├── main.py
├── create_faiss_index.py
├── discourse_posts.json
├── embedding_data.json
├── embedding_md_data.json
├── embedding_combined.json
├── faiss_index.idx
├── image.txt
├── project-tds-virtual-ta-q1.webp
├── project-tds-virtual-ta-promptfoo.yaml
├── LICENSE
├── README.md
├── requirements.txt
```

---

## 🧾 YAML Prompt Configuration

The `project-tds-virtual-ta-promptfoo.yaml` contains pre-defined prompt templates used for structured testing or model evaluations (e.g., via **Promptfoo**).

#### 📄 Sample Format:

```yaml
- prompt: "Answer based on context: {{ context }}\n\nQuestion: {{ question }}"
  model: gpt-4.1-mini
  variables:
    - question
    - context
```

#### ✅ How to Test with Promptfoo CLI:

1. Install [Promptfoo](https://github.com/promptfoo/promptfoo):

   ```bash
   npm install -g promptfoo
   ```

2. **Evaluate your application**  
   Here are a few sample questions and evaluation parameters. These are indicative. The actual evaluation could ask any realistic student question.

   **Steps to run:**

   - Edit `project-tds-virtual-ta-promptfoo.yaml` to replace:
     ```yaml
     providers[0].config.url: http://your-api-url/api/
     ```
   - Then run the script:

     ```bash
     npx -y promptfoo eval --config project-tds-virtual-ta-promptfoo.yaml
     ```

> ⚠️ Make sure to set your `OPENAI_API_KEY` or `AIPIPE_API_KEY` in a `.env` file and load it via `dotenv` in your Python app if needed.

---

## 📄 License

This project is licensed under the **MIT License**. You're free to use, modify, and distribute it with attribution.

See [LICENSE](./LICENSE) for details.

---

## 👥 Credits

Developed by: *Saumya Kumar* (IIT Madras - B.S. Data Science, IIIT Kota - B.Tech CSE)

Powered by:

- [FastAPI](https://fastapi.tiangolo.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [AIPipe Proxy](https://aipipe.org)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Discourse](https://discourse.onlinedegree.iitm.ac.in)
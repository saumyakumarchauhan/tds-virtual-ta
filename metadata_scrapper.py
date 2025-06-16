import os
import json
import re
from datetime import datetime
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright

BASE_URL = "https://tds.s-anand.net/#/2025-01/"
BASE_ORIGIN = "https://tds.s-anand.net"
OUTPUT_DIR = "tds_pages_md"
METADATA_FILE = "embedding_md_data.json"

visited = set()
metadata = []

def sanitize_filename(title):
    return re.sub(r'[\\/*?:"<>|]', "_", title).strip().replace(" ", "_")

def extract_all_internal_links(page):
    links = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
    return list(set(
        link for link in links
        if BASE_ORIGIN in link and '/#/' in link
    ))

def wait_for_article_and_get_html(page):
    page.wait_for_selector("article.markdown-section#main", timeout=10000)
    return page.inner_html("article.markdown-section#main")

def crawl_page(page, url):
    if url in visited:
        return
    visited.add(url)

    print(f" Visiting: {url}")
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(1000)
        html = wait_for_article_and_get_html(page)
    except Exception as e:
        print(f" Error loading page: {url}\n{e}")
        return

    # Extract and convert to markdown
    markdown = md(html)

    metadata.append({
        "chunk": markdown,
        "original_url": url
    })

    # Recursively crawl all internal links
    links = extract_all_internal_links(page)
    for link in links:
        if link not in visited:
            crawl_page(page, link)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global visited, metadata

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        crawl_page(page, BASE_URL)

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\nCompleted. {len(metadata)} chunks saved.")
        browser.close()

if __name__ == "__main__":
    main()

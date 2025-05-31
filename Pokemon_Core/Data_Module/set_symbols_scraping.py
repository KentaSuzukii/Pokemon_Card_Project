import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ----------------------------
# 📁 Output Configuration
# ----------------------------
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data", "External", "Set_Symbols"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "pokemon_tcg_sets.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ----------------------------
# 🧼 Helper Functions
# ----------------------------
def sanitize_filename(name):
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()

# ----------------------------
# 🌐 Get All Set URLs (Working)
# ----------------------------
def get_set_links():
    print("🧠 Using in-browser JavaScript to extract ALL set links...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        print("⏳ Navigating to pokesymbols.com/tcg/sets...")
        page.goto("https://pokesymbols.com/tcg/sets", wait_until="load", timeout=60000)
        page.wait_for_timeout(5000)  # Extra time for hydration

        links = page.evaluate("""
            () => Array.from(document.querySelectorAll('a'))
                .map(a => a.href)
                .filter(h => h.includes('/tcg/sets/') && !h.endsWith('/tcg/sets'))
        """)
        browser.close()

    print(f"✅ Total sets found: {len(links)}")
    return sorted(set(links))

# ----------------------------
# 🔍 Parse Individual Set Page
# ----------------------------
def parse_set_page(url):
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    name_tag = soup.find("h1")
    meta_tags = {m.get('property') or m.get('name'): m.get('content') for m in soup.find_all("meta")}

    # Basic Fields
    set_name = name_tag.text.strip() if name_tag else meta_tags.get("og:title", "Unknown").replace(" | Pokemon Symbols", "")
    img_url = meta_tags.get("og:image")
    desc = meta_tags.get("description")
    release_date = None

    # Parse Release Date (inside <p> tags)
    for p in soup.find_all("p"):
        if "Released:" in p.text:
            release_date = p.text.split("Released:")[-1].strip()

    # Download Symbol Image
    safe_name = sanitize_filename(set_name)
    img_path = os.path.join(OUTPUT_DIR, f"{safe_name}_symbol.png")
    if img_url:
        try:
            img = requests.get(img_url, headers=HEADERS)
            with open(img_path, "wb") as f:
                f.write(img.content)
            print(f"🖼️  Saved: {img_path}")
        except Exception as e:
            print(f"❌ Image download failed: {e}")
            img_path = ""

    return {
        "Set Name": set_name,
        "Description": desc,
        "Release Date": release_date,
        "Image URL": img_url,
        "Image Path": img_path,
        "Detail URL": url
    }

# ----------------------------
# 🔁 Main Scraping Loop
# ----------------------------
def scrape_all_sets():
    urls = get_set_links()
    print(f"\n🔍 Starting scrape of {len(urls)} sets...\n")

    data = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            data.append(parse_set_page(url))
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
    return pd.DataFrame(data)

# ----------------------------
# 🚀 Entry Point
# ----------------------------
if __name__ == "__main__":
    print("✅ RUNNING: set_symbols_scraping.py\n")
    df = scrape_all_sets()
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✅ DONE! {len(df)} sets saved to:\n{CSV_PATH}")

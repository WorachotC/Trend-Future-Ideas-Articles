import asyncio
from playwright.async_api import async_playwright
import json
import os
import re

# --- Configuration ---
# เริ่มต้นที่ 2 หน้าหลัก (ไทยและอังกฤษ)
START_URLS = [
    "https://www.jenosize.com/th/ideas",
    "https://www.jenosize.com/en/ideas"
]
OUTPUT_FILE = "data/processed/jenosize_train_data.jsonl"

def clean_text(text):
    """Cleans whitespace and standardizes text format."""
    if not text: return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def get_all_links(page, start_url):
    """Crawls article links from a specific main page."""
    print(f"🔍 Crawling main page: {start_url}")
    try:
        await page.goto(start_url, timeout=60000)
        
        # Wait for grid
        try:
            await page.wait_for_selector('div.grid', timeout=15000)
        except:
            print(f"⚠️ Warning: Grid not found on {start_url}, trying anyway...")

        # Scroll down for lazy loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(3000) 

        # Extract links (Accept both /th/ideas/ and /en/ideas/)
        links = await page.evaluate('''() => {
            const anchors = Array.from(document.querySelectorAll('a'));
            return anchors
                .map(a => a.href)
                .filter(href => href.includes('/ideas/') && 
                                !href.endsWith('/ideas') && 
                                !href.endsWith('/ideas/') &&
                                !href.includes('#'));
        }''')
        
        return list(set(links))
    except Exception as e:
        print(f"❌ Error crawling {start_url}: {e}")
        return []

async def scrape_jenosize():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    all_links = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # 1. Collect links from BOTH languages 🇹🇭 🇬🇧
        for start_url in START_URLS:
            links = await get_all_links(page, start_url)
            all_links.extend(links)
        
        # Remove duplicates across languages (if any)
        unique_links = list(set(all_links))
        print(f"✅ Total unique articles found: {len(unique_links)}")

        data = []
        
        # 2. Extract content
        print(f"🚀 Starting Extraction on {len(unique_links)} articles...")
        
        for i, url in enumerate(unique_links):
            print(f"[{i+1}/{len(unique_links)}] Visiting: {url}")
            try:
                await page.goto(url, timeout=30000)
                
                # Determine language from URL
                lang = "th" if "/th/" in url else "en"
                
                try:
                    await page.wait_for_selector('div.content-detail', timeout=5000)
                except:
                    pass

                # Extract Title
                title = await page.title()
                h1 = await page.query_selector('h1')
                if h1: title = await h1.inner_text()
                title = clean_text(title)

                # Extract Content
                content_parts = []
                elements = await page.query_selector_all('div.content-detail p, div.content-detail h3, div.content-detail li')
                
                for el in elements:
                    text = await el.inner_text()
                    text = clean_text(text)
                    # Filter short texts
                    if len(text) > 30 and "cookie" not in text.lower() and "สงวนลิขสิทธิ์" not in text:
                        content_parts.append(text)
                
                full_body = "\n\n".join(content_parts)

                if len(full_body) > 200:
                    # Customize instruction based on language
                    instruction = "Write a creative trend analysis article in Jenosize's professional style."
                    if lang == "th":
                        instruction = "เขียนบทความวิเคราะห์เทรนด์ธุรกิจในสไตล์ Jenosize ที่มีความคิดสร้างสรรค์และเป็นมืออาชีพ"

                    entry = {
                        "instruction": instruction,
                        "input": f"Topic: {title}",
                        "output": full_body,
                        "language": lang # Optional: Tagging for reference
                    }
                    data.append(entry)
                    print(f"   ✅ Saved ({lang.upper()}): {len(full_body)} chars")
                else:
                    print("   ⚠️ Content too short.")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        await browser.close()

    # Save
    if data:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        print(f"\n🎉 ALL DONE! Saved {len(data)} bilingual articles to {OUTPUT_FILE}")
    else:
        print("\n❌ No data collected.")

if __name__ == "__main__":
    asyncio.run(scrape_jenosize())
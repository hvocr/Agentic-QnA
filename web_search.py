import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from readability import Document
import time
from typing import List, Optional
import streamlit as st

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def search_serpapi(query: str, api_key: str, num_results: int = 5) -> List[str]:
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": num_results,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        urls = []
        for result in data.get("organic_results", []):
            if "link" in result:
                urls.append(result["link"])
        return urls
    except Exception as e:
        st.error(f"SerpAPI error: {e}")
        return []

def is_scraping_allowed(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    rp = RobotFileParser()
    rp.set_url(f"{base}/robots.txt")
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return True

def is_garbled(line: str) -> bool:
    """
    Detects lines that are mostly letter-spaced or otherwise mangled text
    (a common artifact when a page has hidden/duplicate text nodes, e.g.
    accessibility spans that spell out numbers/words one character at a time).
    """
    words = line.split()
    if not words:
        return False
    single_char = sum(1 for w in words if len(w) == 1)
    return (single_char / len(words)) > 0.4


def scrape_page(url: str, timeout: int = 15) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        try:
            doc = Document(resp.text)
            html = doc.summary()
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator="\n", strip=True)
        except:
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            text_parts = []
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                txt = tag.get_text(strip=True)
                if txt:
                    text_parts.append(txt)
            text = "\n".join(text_parts)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = [line for line in lines if not is_garbled(line)]
        return "\n".join(lines)
    except Exception as e:
        return None

def fetch_web_content(query: str, serp_api_key: str, max_pages: int = 10) -> str:
    urls = search_serpapi(query, serp_api_key, num_results=max_pages + 3)
    if not urls:
        return ""

    st.session_state.debug_info = []
    texts = []
    for url in urls:
        debug_line = f"Checking: {url}"
        if not is_scraping_allowed(url):
            debug_line += " - blocked by robots.txt"
            st.session_state.debug_info.append(debug_line)
            continue
        debug_line += " - scraping..."
        st.session_state.debug_info.append(debug_line)
        text = scrape_page(url)
        if text and len(text) > 200:
            texts.append(text)
            st.session_state.debug_info.append(f"✓ Scraped {len(text)} chars")
            if len(texts) >= max_pages:
                break
        else:
            st.session_state.debug_info.append(f"✗ Scraped too little text or failed")
        time.sleep(1)

    if not texts:
        st.session_state.debug_info.append("No usable content found.")
        return ""
    return "\n\n".join(texts)
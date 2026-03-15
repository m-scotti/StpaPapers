"""
ingest.py
---------
Ingestion pipeline for the RAG-based research paper knowledge base.

Supports:
  - Fetching papers by URL (HTML or PDF)
  - Fetching papers by DOI (resolves via doi.org)
  - Crawling a webpage and ingesting all linked PDFs
  - Extracting text from HTML pages and PDFs
  - Calling Claude to produce structured JSON summaries
  - Batch processing from a papers.txt file

Output schema per paper:
  {
    "title": str,
    "authors": [str],
    "year": int,
    "industry_domain": str,
    "abstract_summary": str,
    "key_findings": [str],
    "software_dev_relevance": {
        "summary": str,
        "specific_applications": [str],
        "relevance_score": int  # 1-10
    },
    "tags": [str]
  }

Usage:
  python ingest.py --url https://arxiv.org/abs/2301.00001
  python ingest.py --doi 10.1145/3290605.3300695
  python ingest.py --pdf ./my_paper.pdf
  python ingest.py --batch papers.txt
  python ingest.py --crawl https://example.com/research
  python ingest.py --crawl https://example.com/research --delay 3 --limit 50
"""

import os
import time
import json
import argparse
import requests
import fitz  # PyMuPDF
import anthropic
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# --- Configuration ---
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OUTPUT_FILE = "summaries.json"
DEFAULT_CRAWL_DELAY = 2      # seconds between requests (be polite)

SYSTEM_PROMPT = """You are a research paper analyst. Given the text of a research paper, extract and return a structured JSON summary.

Return ONLY valid JSON with no preamble, explanation, or markdown backticks. The JSON must match this exact schema:

{
  "title": "Full title of the paper",
  "authors": ["Author 1", "Author 2"],
  "year": 2024,
  "industry_domain": "e.g. healthcare, finance, manufacturing, logistics, education",
  "abstract_summary": "2-3 sentence summary of what the paper is about and what it found",
  "key_findings": [
    "Finding 1",
    "Finding 2",
    "Finding 3"
  ],
  "software_dev_relevance": {
    "summary": "How the concepts, methods, or findings in this paper relate to software development practices",
    "specific_applications": [
      "Specific way this could apply to software dev 1",
      "Specific way this could apply to software dev 2"
    ],
    "relevance_score": 7
  },
  "tags": ["tag1", "tag2", "tag3"]
}

Be specific and insightful, especially for the software_dev_relevance section — look for cross-industry connections even if the paper is not directly about software."""


# --- Text Extraction ---

def fetch_url(url: str) -> tuple[str, str]:
    """
    Fetch a URL and return (text_content, content_type).
    Handles both HTML and PDF responses.
    """
    headers = {"User-Agent": "Mozilla/5.0 (research-paper-ingestion-bot)"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        text = extract_text_from_pdf_bytes(response.content)
    else:
        text = extract_text_from_html(response.text)

    return text, content_type


def fetch_doi(doi: str) -> tuple[str, str]:
    """Resolve a DOI to its URL and fetch the paper."""
    doi = doi.strip().lstrip("https://doi.org/").lstrip("doi:")
    url = f"https://doi.org/{doi}"
    print(f"  Resolving DOI: {doi} -> {url}")
    return fetch_url(url)


def extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping nav/header/footer noise."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def extract_text_from_pdf_path(path: str) -> str:
    """Extract text from a local PDF file."""
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def truncate_text(text: str, max_chars: int = 40000) -> str:
    """
    Truncate text to fit within Claude's context.
    Keeps the beginning (abstract/intro) and end (conclusions).
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... middle truncated ...]\n\n" + text[-half:]


# --- Crawling ---

def find_pdf_links(page_url: str) -> list[str]:
    """
    Fetch a webpage and return all absolute URLs pointing to PDFs.
    Resolves relative links against the base page URL.
    """
    headers = {"User-Agent": "Mozilla/5.0 (research-paper-ingestion-bot)"}
    response = requests.get(page_url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_urls = []
    seen = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if (parsed.path.lower().endswith(".pdf") or "pdf" in parsed.query.lower()):
            if absolute not in seen:
                seen.add(absolute)
                pdf_urls.append(absolute)

    return pdf_urls


def crawl_and_ingest(
    page_url: str,
    delay: float = DEFAULT_CRAWL_DELAY,
    limit: int = None,
) -> None:
    """
    Crawl a page, find all PDF links, and ingest each one.
    Skips already-ingested papers on re-runs (resume-safe).

    Args:
        page_url: URL of the page listing the PDFs.
        delay:    Seconds to wait between requests (default 2s).
        limit:    Max number of new PDFs to process (None = all).
    """
    summaries = load_existing_summaries()
    ingested_sources = {s.get("source") for s in summaries}

    print(f"\nCrawling for PDF links: {page_url}")
    pdf_urls = find_pdf_links(page_url)

    if not pdf_urls:
        print("  No PDF links found on this page. Check the URL or page structure.")
        return

    print(f"  Found {len(pdf_urls)} PDF link(s) on page")

    # Filter already-ingested
    new_urls = [u for u in pdf_urls if u not in ingested_sources]
    skipped = len(pdf_urls) - len(new_urls)
    if skipped:
        print(f"  Skipping {skipped} already-ingested paper(s)")

    if limit:
        new_urls = new_urls[:limit]

    print(f"  Processing {len(new_urls)} new paper(s)\n")

    if not new_urls:
        print("  Nothing new to ingest.")
        return

    errors = 0
    for i, pdf_url in enumerate(new_urls, 1):
        print(f"[{i}/{len(new_urls)}] {pdf_url}")
        try:
            text, _ = fetch_url(pdf_url)
            if not text.strip():
                print("  Warning: No text extracted, skipping.")
                summaries.append({"source": pdf_url, "error": "Empty text extracted"})
                errors += 1
            else:
                summary = summarize_paper(text, source=pdf_url)
                summary["source"] = pdf_url
                summaries.append(summary)

        except Exception as e:
            print(f"  Error: {e}")
            summaries.append({"source": pdf_url, "error": str(e)})
            errors += 1

        # Save after every paper so progress is never lost
        save_summaries(summaries)

        if i < len(new_urls):
            print(f"  Waiting {delay}s...\n")
            time.sleep(delay)

    succeeded = len(new_urls) - errors
    print(f"\nCrawl complete: {succeeded}/{len(new_urls)} succeeded, {errors} error(s)")
    print(f"Total in {OUTPUT_FILE}: {len(summaries)}")


# --- Claude Summarization ---

def summarize_paper(text: str, source: str = "") -> dict:
    """
    Call Claude to produce a structured JSON summary of the paper text.
    Returns a parsed dict.
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    truncated = truncate_text(text)
    user_message = f"Please summarize this research paper:\n\nSource: {source}\n\n---\n\n{truncated}"

    print(f"  Calling Claude ({ANTHROPIC_MODEL})...")
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON response: {e}")
        print(f"  Raw response: {raw[:300]}...")
        return {"error": str(e), "raw_response": raw, "source": source}


# --- Output Handling ---

def load_existing_summaries() -> list[dict]:
    """Load existing summaries from the output file if it exists."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return []


def save_summaries(summaries: list[dict]) -> None:
    """Save all summaries to the output file."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  Saved {len(summaries)} total paper(s) to {OUTPUT_FILE}")


def already_ingested(summaries: list[dict], source: str) -> bool:
    """Check if a source has already been processed."""
    return any(s.get("source") == source for s in summaries)


# --- Single Paper Ingestion ---

def ingest_url(url: str, summaries: list[dict]) -> dict | None:
    """Ingest a paper from a URL."""
    print(f"\nIngesting URL: {url}")
    if already_ingested(summaries, url):
        print("  Already ingested, skipping.")
        return None
    text, _ = fetch_url(url)
    summary = summarize_paper(text, source=url)
    summary["source"] = url
    return summary


def ingest_doi(doi: str, summaries: list[dict]) -> dict | None:
    """Ingest a paper from a DOI."""
    print(f"\nIngesting DOI: {doi}")
    if already_ingested(summaries, doi):
        print("  Already ingested, skipping.")
        return None
    text, _ = fetch_doi(doi)
    summary = summarize_paper(text, source=doi)
    summary["source"] = doi
    return summary


def ingest_batch(filepath: str) -> None:
    """
    Process a batch file of URLs and DOIs.
    Lines starting with '#' are treated as comments.
    """
    summaries = load_existing_summaries()

    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Found {len(lines)} entries in {filepath}")

    for line in lines:
        try:
            if line.startswith("10.") or line.lower().startswith("doi:"):
                result = ingest_doi(line, summaries)
            else:
                result = ingest_url(line, summaries)

            if result:
                summaries.append(result)
                save_summaries(summaries)

        except Exception as e:
            print(f"  Error processing '{line}': {e}")
            summaries.append({"source": line, "error": str(e)})
            save_summaries(summaries)

    print(f"\nDone. {len(summaries)} total paper(s) in {OUTPUT_FILE}")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest research papers into the knowledge base.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",   help="Fetch and ingest a paper from a URL")
    group.add_argument("--doi",   help="Fetch and ingest a paper by DOI")
    group.add_argument("--pdf",   help="Ingest a local PDF file")
    group.add_argument("--batch", help="Process a batch file of URLs/DOIs (papers.txt)")
    group.add_argument("--crawl", help="Crawl a webpage and ingest all linked PDFs")

    parser.add_argument("--delay", type=float, default=DEFAULT_CRAWL_DELAY,
                        help=f"Seconds between PDF requests during crawl (default: {DEFAULT_CRAWL_DELAY})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of new PDFs to process during crawl (default: all)")

    args = parser.parse_args()
    summaries = load_existing_summaries()

    if args.url:
        result = ingest_url(args.url, summaries)
        if result:
            summaries.append(result)
            save_summaries(summaries)

    elif args.doi:
        result = ingest_doi(args.doi, summaries)
        if result:
            summaries.append(result)
            save_summaries(summaries)

    elif args.pdf:
        print(f"\nIngesting local PDF: {args.pdf}")
        if already_ingested(summaries, args.pdf):
            print("  Already ingested, skipping.")
        else:
            text = extract_text_from_pdf_path(args.pdf)
            result = summarize_paper(text, source=args.pdf)
            result["source"] = args.pdf
            summaries.append(result)
            save_summaries(summaries)

    elif args.batch:
        ingest_batch(args.batch)

    elif args.crawl:
        crawl_and_ingest(args.crawl, delay=args.delay, limit=args.limit)

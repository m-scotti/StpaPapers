"""
vectorstore.py
--------------
Handles indexing and retrieval of research paper summaries in ChromaDB.

Assumes paper summaries are structured JSON dicts as produced by ingest.py:
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
"""

import json
import hashlib
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"

# Uses sentence-transformers locally (no API key needed)
# Model is small (~80MB) and well-suited for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_collection():
    """Initialize persistent Chroma client and return the collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _build_document_text(paper: dict) -> str:
    """
    Combine the most semantically rich fields into a single string for embedding.
    This is what gets vectorized — richer text = better retrieval.
    """
    parts = [
        paper.get("title", ""),
        paper.get("abstract_summary", ""),
        " ".join(paper.get("key_findings", [])),
        paper.get("software_dev_relevance", {}).get("summary", ""),
        " ".join(paper.get("software_dev_relevance", {}).get("specific_applications", [])),
        " ".join(paper.get("tags", [])),
    ]
    return " ".join(p for p in parts if p).strip()


def _build_metadata(paper: dict) -> dict:
    """
    Flatten paper fields into a Chroma-compatible metadata dict.
    Chroma metadata values must be str, int, float, or bool — no lists/dicts.
    """
    sdr = paper.get("software_dev_relevance", {})
    return {
        "title": paper.get("title", ""),
        "authors": ", ".join(paper.get("authors", [])),
        "year": int(paper.get("year") or 0),
        "industry_domain": paper.get("industry_domain", ""),
        "relevance_score": int(sdr.get("relevance_score", 0)),
        "tags": ", ".join(paper.get("tags", [])),
        "specific_applications": ", ".join(sdr.get("specific_applications", [])),
        "abstract_summary": paper.get("abstract_summary", ""),
        "software_dev_summary": sdr.get("summary", ""),
    }


def _paper_id(paper: dict) -> str:
    """Generate a stable ID for a paper based on its title + year."""
    key = f"{paper.get('title', '')}_{paper.get('year', '')}".lower()
    return hashlib.md5(key.encode()).hexdigest()


# --- Public API ---

def add_paper(paper: dict) -> str:
    """
    Add a single paper summary to the vector store.
    Returns the paper ID. Skips if already indexed (idempotent).
    """
    collection = _get_collection()
    doc_id = _paper_id(paper)

    existing = collection.get(ids=[doc_id])
    if existing["ids"]:
        print(f"  [skip] Already indexed: {paper.get('title', doc_id)}")
        return doc_id

    document_text = _build_document_text(paper)
    metadata = _build_metadata(paper)

    collection.add(
        ids=[doc_id],
        documents=[document_text],
        metadatas=[metadata],
    )
    print(f"  [added] {paper.get('title', doc_id)}")
    return doc_id


def add_papers(papers: list[dict]) -> list[str]:
    """Add a list of paper summaries. Returns list of IDs."""
    ids = []
    for paper in papers:
        ids.append(add_paper(paper))
    print(f"\nIndexed {len(ids)} paper(s) into ChromaDB at '{CHROMA_DB_PATH}'")
    return ids


def search(
    query: str,
    n_results: int = 5,
    min_relevance_score: int = 0,
    industry_domain: str = None,
) -> list[dict]:
    """
    Semantic search over indexed papers.

    Args:
        query: Natural language query string.
        n_results: Number of results to return.
        min_relevance_score: Filter by minimum software_dev relevance score (1-10).
        industry_domain: Optional filter by domain (e.g. "healthcare", "finance").

    Returns:
        List of result dicts with metadata + distance score.
    """
    collection = _get_collection()

    where_filter = {}
    if min_relevance_score > 0:
        where_filter["relevance_score"] = {"$gte": min_relevance_score}
    if industry_domain:
        where_filter["industry_domain"] = {"$eq": industry_domain}

    query_kwargs = dict(
        query_texts=[query],
        n_results=min(n_results, collection.count() or 1),
        include=["metadatas", "distances", "documents"],
    )
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    output = []
    for i, meta in enumerate(results["metadatas"][0]):
        output.append({
            "rank": i + 1,
            "distance": round(results["distances"][0][i], 4),
            **meta,
        })
    return output


def list_all(limit: int = 100) -> list[dict]:
    """Return all indexed papers (metadata only)."""
    collection = _get_collection()
    results = collection.get(limit=limit, include=["metadatas"])
    return results["metadatas"]


def count() -> int:
    """Return total number of indexed papers."""
    return _get_collection().count()


def delete_paper(paper: dict) -> None:
    """Remove a paper from the index by its derived ID."""
    collection = _get_collection()
    doc_id = _paper_id(paper)
    collection.delete(ids=[doc_id])
    print(f"Deleted: {paper.get('title', doc_id)}")


# --- CLI for quick testing ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vectorstore.py index <summaries.json>   # Index papers from JSON file")
        print("  python vectorstore.py search '<query>'         # Search indexed papers")
        print("  python vectorstore.py list                     # List all indexed papers")
        print("  python vectorstore.py count                    # Count indexed papers")
        sys.exit(0)

    command = sys.argv[1]

    if command == "index" and len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            papers = json.load(f)
        if isinstance(papers, dict):
            papers = [papers]
        add_papers(papers)

    elif command == "search" and len(sys.argv) >= 3:
        query = sys.argv[2]
        min_score = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
        results = search(query, n_results=5, min_relevance_score=min_score)
        print(f"\nTop results for: '{query}'\n")
        for r in results:
            print(f"  [{r['rank']}] {r['title']} ({r['year']}) — score: {r['relevance_score']}/10, dist: {r['distance']}")
            print(f"       {r['abstract_summary'][:120]}...")
            print()

    elif command == "list":
        papers = list_all()
        print(f"\n{len(papers)} paper(s) indexed:\n")
        for p in papers:
            print(f"  - {p['title']} ({p['year']}) [{p['industry_domain']}] score={p['relevance_score']}/10")

    elif command == "count":
        print(f"Total papers indexed: {count()}")

    else:
        print(f"Unknown command: {command}")

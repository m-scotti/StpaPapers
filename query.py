"""
query.py
--------
Query layer for the RAG-based research paper knowledge base.

Takes a natural language question, retrieves the most relevant papers
from ChromaDB, and uses Claude to synthesize a response — highlighting
cross-industry connections to software development.

Usage:
  python query.py "How can safety analysis methods improve software architecture?"
  python query.py "What are the key findings about healthcare IT safety?"
  python query.py --top 3 "How is STPA applied in practice?"
  python query.py --raw "organizational design"   # skip Claude, show raw results
"""

import argparse
import json
import anthropic
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TOP_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a research synthesis assistant specializing in cross-industry knowledge transfer.

You will be given a user's question and a set of relevant research paper summaries retrieved from a knowledge base.

Your job is to:
1. Directly answer the user's question using insights from the papers
2. Highlight connections between findings and software development practices
3. Note where multiple papers reinforce or complement each other
4. Be specific — reference paper titles and authors where relevant

Format your response as:
- A direct answer to the question (2-4 paragraphs)
- A "Software Development Connections" section with concrete, actionable insights
- A "Key Papers" section listing the most relevant papers with one-line summaries

Be insightful and practical. The goal is to help software developers learn from research in other industries."""


def load_collection():
    """Load the ChromaDB collection with the same embedding function used during indexing."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def search_papers(collection, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search ChromaDB for papers relevant to the query."""
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    papers = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        def safe_json(val, fallback):
            if not val:
                return fallback
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return fallback

        # Reconstruct the summary dict from stored metadata + document
        paper = {
            "title": metadata.get("title", "Unknown"),
            "authors": safe_json(metadata.get("authors"), []),
            "year": metadata.get("year"),
            "industry_domain": metadata.get("industry_domain", ""),
            "relevance_score": metadata.get("relevance_score", "N/A"),
            "distance": round(distance, 4),
            "abstract_summary": metadata.get("abstract_summary", ""),
            "key_findings": [],  # not stored in metadata, omitted
            "software_dev_relevance": {
                "summary": metadata.get("software_dev_summary", ""),
                "specific_applications": safe_json(metadata.get("specific_applications"), []),
                "relevance_score": metadata.get("relevance_score", "N/A"),
            },
            "tags": safe_json(metadata.get("tags"), []),
            "source": metadata.get("source", ""),
        }
        papers.append(paper)

    return papers


def format_papers_for_prompt(papers: list[dict]) -> str:
    """Format retrieved papers into a prompt-friendly string for Claude."""
    sections = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors += " et al."

        findings = "\n".join(f"    - {f}" for f in p["key_findings"][:3])
        applications = "\n".join(
            f"    - {a}" for a in p["software_dev_relevance"]["specific_applications"][:2]
        )

        section = f"""[Paper {i}]
Title: {p['title']}
Authors: {authors}
Year: {p['year']}
Domain: {p['industry_domain']}
SW Relevance Score: {p['relevance_score']}/10

Summary: {p['abstract_summary']}

Key Findings:
{findings}

Software Dev Applications:
{applications}

SW Relevance: {p['software_dev_relevance']['summary']}
"""
        sections.append(section)

    return "\n---\n".join(sections)


def synthesize_with_claude(query: str, papers: list[dict]) -> str:
    """Pass retrieved papers to Claude and get a synthesized response."""
    client = anthropic.Anthropic()

    papers_text = format_papers_for_prompt(papers)
    user_message = f"""Question: {query}

Retrieved Papers:
---
{papers_text}
---

Please synthesize an answer to my question using these papers."""

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


def print_raw_results(papers: list[dict]) -> None:
    """Print raw search results without Claude synthesis."""
    print(f"\nTop {len(papers)} result(s):\n")
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p["authors"][:2])
        if len(p["authors"]) > 2:
            authors += " et al."
        print(f"  [{i}] {p['title']} ({p['year']})")
        print(f"       Authors: {authors}")
        print(f"       Domain: {p['industry_domain']} | SW Score: {p['relevance_score']}/10 | Distance: {p['distance']}")
        print(f"       {p['abstract_summary'][:150]}...")
        print()


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the research paper knowledge base."
    )
    parser.add_argument("question", help="Natural language question to ask")
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of papers to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw search results without Claude synthesis",
    )
    args = parser.parse_args()

    print(f"\nSearching for: '{args.question}'")
    print(f"Retrieving top {args.top} papers...\n")

    collection = load_collection()
    papers = search_papers(collection, args.question, top_k=args.top)

    if not papers:
        print("No results found. Have you run 'python vectorstore.py index summaries.json' yet?")
        exit(1)

    if args.raw:
        print_raw_results(papers)
    else:
        print(f"Found {len(papers)} relevant paper(s). Synthesizing with Claude...\n")
        answer = synthesize_with_claude(args.question, papers)
        print("=" * 70)
        print(answer)
        print("=" * 70)

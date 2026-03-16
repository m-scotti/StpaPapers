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

SYSTEM_PROMPT = """You are a research synthesis assistant helping engineering managers learn from safety-critical industries and apply those lessons to software development.

You will be given a question and a set of research paper summaries. Each paper includes findings, methodology, limitations, and software development relevance extracted from the original research.

Your job:
1. Answer the question directly using specific evidence from the papers — cite findings, quote authors where relevant, and reference methodology to establish credibility
2. Draw out implications for engineering leadership: team structure, decision-making, risk management, process design, and organizational learning
3. Where papers reinforce or contradict each other, say so explicitly
4. Be honest about limitations — if the research has caveats that affect how confidently a manager should act on it, say so
5. For each key lesson, trace it explicitly back to its source paper and explain the direct analogy to software development

Format your response as follows:

## Answer
2-4 paragraphs directly addressing the question. Reference specific papers and findings — don't speak in generalities.

## Implications for Engineering Leadership
3-5 concrete, actionable takeaways framed for a manager. Focus on process, team, and organizational decisions — not implementation details.

## Lessons from the Research — Applied to Software Development
For each major lesson, use this structure:
**[Lesson title]**
- Source: which paper and what finding
- In [industry]: what they observed or did
- In software: the direct analogy and how a team would apply it
- Watch out for: one gotcha or difference between the two domains

## Caveats & Limitations
1-2 sentences on what the research doesn't cover or where findings should be applied carefully.

## Key Papers
A one-line summary of each relevant paper and why it matters for this question.

Tone: authoritative but accessible. Avoid jargon. Write for someone who reads HBR and leads engineering teams of 10-100 people."""


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

def run_query(collection, question: str, top_k: int, raw: bool) -> None:
    """Run a single query and print results."""
    print(f"\nSearching for: '{question}'")
    print(f"Retrieving top {top_k} papers...\n")

    papers = search_papers(collection, question, top_k=top_k)

    if not papers:
        print("No results found. Have you run 'python vectorstore.py index summaries.json' yet?")
        return

    if raw:
        print_raw_results(papers)
    else:
        print(f"Found {len(papers)} relevant paper(s). Synthesizing with Claude...\n")
        answer = synthesize_with_claude(question, papers)
        print("=" * 70)
        print(answer)
        print("=" * 70)


def run_interactive(collection, top_k: int, raw: bool) -> None:
    """Interactive REPL loop for querying the knowledge base."""
    print("\n" + "=" * 70)
    print("  Research Paper Knowledge Base — Interactive Mode")
    print("  Type your question and press Enter. Type 'exit' to quit.")
    print("=" * 70)

    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        run_query(collection, question, top_k, raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the research paper knowledge base."
    )
    parser.add_argument("question", nargs="?", help="Question to ask (omit for interactive mode)")
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
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive query session",
    )
    args = parser.parse_args()

    collection = load_collection()

    if args.interactive or not args.question:
        run_interactive(collection, top_k=args.top, raw=args.raw)
    else:
        run_query(collection, args.question, top_k=args.top, raw=args.raw)
        print("=" * 70)

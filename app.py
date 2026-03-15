"""
app.py
------
Flask web interface for the RAG-based research paper knowledge base.

Provides:
  - /           Browse all indexed papers
  - /search     Search and synthesize answers with Claude
  - /api/search JSON API for search queries
  - /api/papers JSON API listing all papers

Usage:
  pip install flask
  python app.py
  open http://localhost:5000
"""

import json
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, render_template_string, request, jsonify

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

Be insightful and practical."""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Research Knowledge Base</title>
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #0a0a0f;
      --surface: #12121a;
      --surface2: #1a1a26;
      --border: #2a2a3d;
      --accent: #7b6cff;
      --accent2: #ff6c9d;
      --text: #e8e8f0;
      --text-muted: #7a7a9a;
      --green: #4dffb4;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'DM Mono', monospace;
      min-height: 100vh;
      line-height: 1.6;
    }

    /* Grain overlay */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
      pointer-events: none;
      z-index: 0;
      opacity: 0.4;
    }

    header {
      border-bottom: 1px solid var(--border);
      padding: 1.5rem 2.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      background: rgba(10,10,15,0.92);
      backdrop-filter: blur(12px);
      z-index: 100;
    }

    .logo {
      font-family: 'Syne', sans-serif;
      font-weight: 800;
      font-size: 1.1rem;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .logo-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 12px var(--accent);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; box-shadow: 0 0 12px var(--accent); }
      50% { opacity: 0.5; box-shadow: 0 0 4px var(--accent); }
    }

    .paper-count {
      font-size: 0.7rem;
      color: var(--text-muted);
      background: var(--surface2);
      padding: 0.3rem 0.75rem;
      border-radius: 100px;
      border: 1px solid var(--border);
    }

    .tabs {
      display: flex;
      gap: 0.25rem;
    }

    .tab {
      padding: 0.4rem 1rem;
      border-radius: 6px;
      font-size: 0.75rem;
      cursor: pointer;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-family: 'DM Mono', monospace;
      transition: all 0.15s;
    }

    .tab.active, .tab:hover {
      background: var(--surface2);
      color: var(--text);
    }

    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 2.5rem;
      position: relative;
      z-index: 1;
    }

    /* Search Panel */
    .search-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 2rem;
      margin-bottom: 2.5rem;
      position: relative;
      overflow: hidden;
    }

    .search-panel::before {
      content: '';
      position: absolute;
      top: -60px; right: -60px;
      width: 200px; height: 200px;
      background: radial-gradient(circle, rgba(123,108,255,0.12) 0%, transparent 70%);
      pointer-events: none;
    }

    .search-label {
      font-family: 'Syne', sans-serif;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 0.75rem;
      display: block;
    }

    .search-row {
      display: flex;
      gap: 0.75rem;
      align-items: stretch;
    }

    .search-input {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.85rem 1.25rem;
      color: var(--text);
      font-family: 'DM Mono', monospace;
      font-size: 0.9rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .search-input:focus {
      border-color: var(--accent);
    }

    .search-input::placeholder { color: var(--text-muted); }

    .search-btn {
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 0.85rem 1.5rem;
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }

    .search-btn:hover { background: #9a8eff; transform: translateY(-1px); }
    .search-btn:active { transform: translateY(0); }
    .search-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

    .search-options {
      display: flex;
      gap: 1.5rem;
      margin-top: 1rem;
      align-items: center;
    }

    .search-options label {
      font-size: 0.75rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 0.4rem;
      cursor: pointer;
    }

    .search-options input[type=range] { accent-color: var(--accent); }

    /* Answer Panel */
    .answer-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 2rem;
      margin-bottom: 2rem;
      display: none;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

    .answer-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }

    .answer-badge {
      font-size: 0.65rem;
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 0.25rem 0.6rem;
      border-radius: 100px;
      background: rgba(123,108,255,0.15);
      color: var(--accent);
      border: 1px solid rgba(123,108,255,0.3);
    }

    .answer-query {
      font-size: 0.8rem;
      color: var(--text-muted);
      font-style: italic;
    }

    .answer-content {
      font-size: 0.88rem;
      line-height: 1.8;
      color: var(--text);
      white-space: pre-wrap;
    }

    .answer-content h2, .answer-content strong {
      color: var(--green);
      font-family: 'Syne', sans-serif;
    }

    /* Papers Grid */
    .section-header {
      display: flex;
      align-items: baseline;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .section-title {
      font-family: 'Syne', sans-serif;
      font-weight: 800;
      font-size: 1.1rem;
    }

    .section-count {
      font-size: 0.7rem;
      color: var(--text-muted);
    }

    .papers-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 1rem;
    }

    .paper-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      transition: all 0.2s;
      cursor: default;
      position: relative;
      overflow: hidden;
    }

    .paper-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(123,108,255,0.1);
    }

    .paper-card::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      opacity: 0;
      transition: opacity 0.2s;
    }

    .paper-card:hover::after { opacity: 1; }

    .paper-domain {
      font-size: 0.65rem;
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--accent2);
      margin-bottom: 0.5rem;
    }

    .paper-title {
      font-family: 'Syne', sans-serif;
      font-weight: 700;
      font-size: 0.9rem;
      line-height: 1.4;
      margin-bottom: 0.5rem;
      color: var(--text);
    }

    .paper-meta {
      font-size: 0.7rem;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
    }

    .paper-abstract {
      font-size: 0.78rem;
      color: var(--text-muted);
      line-height: 1.6;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .paper-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 1rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--border);
    }

    .score-badge {
      display: flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.7rem;
      color: var(--green);
    }

    .score-bar {
      width: 50px;
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      overflow: hidden;
    }

    .score-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--green));
      border-radius: 2px;
    }

    .paper-tags {
      display: flex;
      gap: 0.3rem;
      flex-wrap: wrap;
    }

    .tag {
      font-size: 0.6rem;
      padding: 0.15rem 0.5rem;
      border-radius: 100px;
      background: var(--surface2);
      color: var(--text-muted);
      border: 1px solid var(--border);
    }

    /* Loading */
    .loading {
      display: none;
      align-items: center;
      gap: 0.75rem;
      color: var(--text-muted);
      font-size: 0.8rem;
      padding: 1rem 0;
    }

    .spinner {
      width: 16px; height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    /* Empty state */
    .empty-state {
      text-align: center;
      padding: 4rem 2rem;
      color: var(--text-muted);
    }

    .empty-state .icon { font-size: 2.5rem; margin-bottom: 1rem; }
    .empty-state p { font-size: 0.85rem; }

    /* Sections toggle */
    .view { display: none; }
    .view.active { display: block; }
  </style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    Research KB
  </div>
  <div class="tabs">
    <button class="tab active" onclick="showView('search')">Search</button>
    <button class="tab" onclick="showView('browse')">Browse</button>
  </div>
  <div class="paper-count" id="paperCount">— papers</div>
</header>

<main>

  <!-- SEARCH VIEW -->
  <div class="view active" id="view-search">
    <div class="search-panel">
      <span class="search-label">Ask a question</span>
      <div class="search-row">
        <input
          class="search-input"
          id="queryInput"
          type="text"
          placeholder="How can safety analysis improve software architecture?"
          onkeydown="if(event.key==='Enter') doSearch()"
        />
        <button class="search-btn" id="searchBtn" onclick="doSearch()">Search</button>
      </div>
      <div class="search-options">
        <label>
          Top results: <strong id="topKVal">5</strong>
          <input type="range" min="1" max="10" value="5" id="topK"
            oninput="document.getElementById('topKVal').textContent=this.value"/>
        </label>
      </div>
    </div>

    <div class="loading" id="loading">
      <div class="spinner"></div>
      Retrieving papers and synthesizing answer...
    </div>

    <div class="answer-panel" id="answerPanel">
      <div class="answer-header">
        <span class="answer-badge">Claude Synthesis</span>
        <span class="answer-query" id="answerQuery"></span>
      </div>
      <div class="answer-content" id="answerContent"></div>
    </div>

    <div id="searchResults"></div>
  </div>

  <!-- BROWSE VIEW -->
  <div class="view" id="view-browse">
    <div class="section-header">
      <span class="section-title">All Papers</span>
      <span class="section-count" id="browseCount"></span>
    </div>
    <div class="papers-grid" id="papersGrid">
      <div class="empty-state"><div class="icon">📚</div><p>Loading papers...</p></div>
    </div>
  </div>

</main>

<script>
  let allPapers = [];

  function showView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('view-' + name).classList.add('active');
    event.target.classList.add('active');
    if (name === 'browse' && allPapers.length === 0) loadPapers();
  }

  async function loadPapers() {
    const res = await fetch('/api/papers');
    const data = await res.json();
    allPapers = data.papers || [];
    document.getElementById('paperCount').textContent = allPapers.length + ' papers';
    document.getElementById('browseCount').textContent = allPapers.length + ' indexed';
    renderPapersGrid(allPapers, 'papersGrid');
  }

  function renderPapersGrid(papers, targetId) {
    const grid = document.getElementById(targetId);
    if (!papers.length) {
      grid.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><p>No papers found.</p></div>';
      return;
    }
    grid.innerHTML = papers.map(p => {
      const score = p.relevance_score || 0;
      const pct = (score / 10) * 100;
      const tags = (p.tags || []).slice(0, 3).map(t => `<span class="tag">${t}</span>`).join('');
      const authors = (p.authors || []).slice(0,2).join(', ') + ((p.authors||[]).length > 2 ? ' et al.' : '');
      return `
        <div class="paper-card">
          <div class="paper-domain">${p.industry_domain || 'Research'}</div>
          <div class="paper-title">${p.title}</div>
          <div class="paper-meta">${authors} · ${p.year || ''}</div>
          <div class="paper-abstract">${p.abstract_summary || ''}</div>
          <div class="paper-footer">
            <div class="score-badge">
              <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
              SW ${score}/10
            </div>
            <div class="paper-tags">${tags}</div>
          </div>
        </div>`;
    }).join('');
  }

  async function doSearch() {
    const q = document.getElementById('queryInput').value.trim();
    if (!q) return;

    const topK = parseInt(document.getElementById('topK').value);
    const btn = document.getElementById('searchBtn');
    const loading = document.getElementById('loading');
    const answerPanel = document.getElementById('answerPanel');
    const resultsDiv = document.getElementById('searchResults');

    btn.disabled = true;
    loading.style.display = 'flex';
    answerPanel.style.display = 'none';
    resultsDiv.innerHTML = '';

    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: topK })
      });
      const data = await res.json();

      if (data.error) {
        answerPanel.style.display = 'block';
        document.getElementById('answerContent').textContent = 'Error: ' + data.error;
        return;
      }

      // Show answer
      document.getElementById('answerQuery').textContent = '"' + q + '"';
      document.getElementById('answerContent').textContent = data.answer;
      answerPanel.style.display = 'block';

      // Show paper cards
      if (data.papers && data.papers.length) {
        const header = document.createElement('div');
        header.className = 'section-header';
        header.style.marginTop = '1.5rem';
        header.innerHTML = `<span class="section-title">Retrieved Papers</span><span class="section-count">${data.papers.length} results</span>`;
        const grid = document.createElement('div');
        grid.className = 'papers-grid';
        grid.id = 'searchResultsGrid';
        resultsDiv.appendChild(header);
        resultsDiv.appendChild(grid);
        renderPapersGrid(data.papers, 'searchResultsGrid');
      }

    } catch (err) {
      alert('Request failed: ' + err.message);
    } finally {
      btn.disabled = false;
      loading.style.display = 'none';
    }
  }

  // Init
  fetch('/api/papers').then(r => r.json()).then(d => {
    document.getElementById('paperCount').textContent = (d.papers||[]).length + ' papers';
  });
</script>
</body>
</html>
"""

app = Flask(__name__)

# Load collection once at startup
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def safe_json(val, fallback):
    if not val:
        return fallback
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return fallback


def metadata_to_paper(metadata: dict, distance: float = None) -> dict:
    paper = {
        "title": metadata.get("title", "Unknown"),
        "authors": safe_json(metadata.get("authors"), []),
        "year": metadata.get("year"),
        "industry_domain": metadata.get("industry_domain", ""),
        "relevance_score": metadata.get("relevance_score", 0),
        "abstract_summary": metadata.get("abstract_summary", ""),
        "software_dev_relevance": {
            "summary": metadata.get("software_dev_summary", ""),
            "specific_applications": safe_json(metadata.get("specific_applications"), []),
        },
        "tags": safe_json(metadata.get("tags"), []),
        "source": metadata.get("source", ""),
    }
    if distance is not None:
        paper["distance"] = round(distance, 4)
    return paper


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/papers")
def api_papers():
    results = collection.get(include=["metadatas"])
    papers = [metadata_to_paper(m) for m in results["metadatas"]]
    return jsonify({"papers": papers, "count": len(papers)})


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    query = data.get("query", "").strip()
    top_k = min(int(data.get("top_k", DEFAULT_TOP_K)), collection.count())

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Retrieve from Chroma
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    papers = [
        metadata_to_paper(results["metadatas"][0][i], results["distances"][0][i])
        for i in range(len(results["ids"][0]))
    ]

    # Synthesize with Claude
    papers_text = "\n---\n".join([
        f"Title: {p['title']}\nDomain: {p['industry_domain']}\nYear: {p['year']}\n"
        f"Summary: {p['abstract_summary']}\nSW Relevance: {p['software_dev_relevance']['summary']}"
        for p in papers
    ])

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {query}\n\nPapers:\n{papers_text}"}],
    )
    answer = response.content[0].text

    return jsonify({"answer": answer, "papers": papers})


if __name__ == "__main__":
    print("\n  Research Knowledge Base")
    print(f"  {collection.count()} papers indexed")
    print("  Running at http://localhost:5000\n")
    app.run(debug=True, port=5000)

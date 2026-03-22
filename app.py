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

import os
import json
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from flask import Flask, render_template_string, request, jsonify

# --- Configuration ---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TOP_K = 20
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a research synthesis engine for engineering managers. Your sole job is to extract non-obvious, evidence-backed patterns from safety-critical industry research and translate them into specific software engineering decisions.

ABSOLUTE RULES — violating any of these invalidates the response:
- BANNED PHRASES: "can be applied to software", "relevant to software teams", "similar principles apply", "this methodology can help". If you find yourself writing these, stop and name the exact scenario instead.
- NO REPETITION: Every bullet, sentence, and section must add new information. If you made the point already, do not make it again in different words.
- NO VAGUE SCENARIOS: Every software insight must specify at minimum two of: team size, system scale, org structure, deployment frequency, incident type, meeting format, role. "Engineering teams should consider X" is not acceptable. "A 3-team microservices org doing continuous deployment should do X" is acceptable.
- CONVERGENCE IS EVIDENCE: If 3+ papers find the same thing, treat it as strong evidence and say so explicitly. If papers conflict, name both findings and the tension between them — do not average them out.
- SURPRISE FIRST: Lead with findings that contradict common software engineering assumptions. Save the obvious stuff for last or cut it.

## Answer
2-4 paragraphs directly addressing the question. Name specific papers and their findings. Flag convergence and divergence explicitly.

## Patterns Across the Research
Exactly 3-5 patterns. No more.
**[Pattern name: specific enough that two different patterns could not share it]**
- Evidence: paper names + specific finding + sample size or scope if available
- Contradicts the assumption that: [name the common software belief this overturns]
- Concrete scenario: [name the exact team structure, system type, or process where this applies]
- Confidence: High (3+ papers) / Medium (2 papers) / Low (1 paper or indirect evidence)

## Implications for Engineering Leadership
3-5 decisions — not principles. Each must be a thing a manager could do differently in their next sprint planning, 1:1, architecture review, or incident process. Start each with a verb.

## Where the Analogy Breaks Down
What structural differences between safety-critical systems and software make these findings less applicable? Be specific about which findings are most at risk of false transfer.

## Key Papers
Only papers whose findings directly answer the question — omit tangential ones. Format: [Title] — [single most relevant finding] — [why it matters for this specific question].

Tone: skeptical, precise, evidence-first. This reader has heard every generic "learnings from aviation" talk and is tired of it. Earn their attention with specificity."""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>STPA Research Knowledge Base</title>
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #f7f6f2;
      --surface: #ffffff;
      --surface2: #f0efe9;
      --border: #e2e0d8;
      --border-strong: #c8c5b8;
      --accent: #1d3461;
      --accent-light: #e8edf5;
      --accent2: #c0392b;
      --accent2-light: #fdf0ef;
      --green: #166534;
      --green-light: #dcfce7;
      --text: #1a1917;
      --text-muted: #6b6860;
      --text-light: #9c9a94;
      --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
      --shadow-lg: 0 20px 50px rgba(0,0,0,0.12), 0 8px 20px rgba(0,0,0,0.06);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Source Serif 4', Georgia, serif;
      min-height: 100vh;
      line-height: 1.65;
      font-size: 16px;
    }

    /* Header */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 2.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 64px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: var(--shadow);
    }

    .logo {
      font-family: 'Fraunces', Georgia, serif;
      font-weight: 700;
      font-size: 1.15rem;
      color: var(--accent);
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .logo-mark {
      width: 28px;
      height: 28px;
      background: var(--accent);
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 0.75rem;
      font-weight: 600;
      font-family: 'IBM Plex Mono', monospace;
      flex-shrink: 0;
    }

    .header-right {
      display: flex;
      align-items: center;
      gap: 1.5rem;
    }

    .paper-count {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--text-muted);
      background: var(--surface2);
      padding: 0.3rem 0.75rem;
      border-radius: 100px;
      border: 1px solid var(--border);
    }

    .tabs {
      display: flex;
      gap: 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: var(--surface2);
    }

    .tab {
      padding: 0.4rem 1.1rem;
      font-size: 0.8rem;
      font-family: 'Source Serif 4', serif;
      cursor: pointer;
      border: none;
      background: transparent;
      color: var(--text-muted);
      transition: all 0.15s;
      border-right: 1px solid var(--border);
    }

    .tab:last-child { border-right: none; }

    .tab.active {
      background: var(--surface);
      color: var(--accent);
      font-weight: 600;
    }

    .tab:hover:not(.active) {
      background: var(--border);
      color: var(--text);
    }

    /* Main layout */
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 2.5rem 2.5rem;
    }

    /* Search panel */
    .search-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2rem;
      margin-bottom: 2rem;
      box-shadow: var(--shadow);
    }

    .search-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      font-weight: 500;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
      display: block;
    }

    .search-row {
      display: flex;
      gap: 0.75rem;
    }

    .search-input {
      flex: 1;
      background: var(--bg);
      border: 1.5px solid var(--border);
      border-radius: 8px;
      padding: 0.8rem 1.1rem;
      color: var(--text);
      font-family: 'Source Serif 4', serif;
      font-size: 1rem;
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s;
    }

    .search-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(29, 52, 97, 0.08);
    }

    .search-input::placeholder { color: var(--text-light); }

    .search-btn {
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 8px;
      padding: 0.8rem 1.6rem;
      font-family: 'Source Serif 4', serif;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.15s;
      white-space: nowrap;
      letter-spacing: 0.01em;
    }

    .search-btn:hover { background: #152849; transform: translateY(-1px); box-shadow: var(--shadow-md); }
    .search-btn:active { transform: translateY(0); }
    .search-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

    .search-hint {
      margin-top: 0.85rem;
      font-size: 0.82rem;
      color: var(--text-muted);
      font-style: italic;
    }

    /* Answer panel */
    .answer-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 2rem;
      display: none;
      animation: fadeIn 0.3s ease;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

    .answer-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 1.25rem 1.75rem;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
    }

    .answer-badge {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      background: var(--accent-light);
      color: var(--accent);
      border: 1px solid rgba(29, 52, 97, 0.15);
    }

    .answer-query {
      font-size: 0.85rem;
      color: var(--text-muted);
      font-style: italic;
    }

    .answer-content {
      padding: 1.75rem;
      font-size: 0.95rem;
      line-height: 1.8;
      color: var(--text);
    }

    /* Markdown rendered styles */
    .answer-content h2 {
      font-family: 'Fraunces', serif;
      font-size: 1rem;
      font-weight: 600;
      color: var(--accent);
      margin: 1.75rem 0 0.6rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--border);
      letter-spacing: -0.01em;
    }
    .answer-content h2:first-child { margin-top: 0; }
    .answer-content h3 {
      font-family: 'Fraunces', serif;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      margin: 1.25rem 0 0.4rem;
    }
    .answer-content p { margin-bottom: 0.85rem; }
    .answer-content strong { color: var(--text); font-weight: 600; }
    .answer-content ul, .answer-content ol { padding-left: 1.5rem; margin-bottom: 0.85rem; }
    .answer-content li { margin-bottom: 0.35rem; line-height: 1.7; }
    .answer-content hr { border: none; border-top: 1px solid var(--border); margin: 1.25rem 0; }
    .answer-content em { color: var(--text-muted); }

    /* Loading */
    .loading {
      display: none;
      align-items: center;
      gap: 0.75rem;
      color: var(--text-muted);
      font-size: 0.85rem;
      padding: 1.25rem 0;
      font-style: italic;
    }

    .spinner {
      width: 16px; height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      flex-shrink: 0;
    }

    @keyframes spin { to { transform: rotate(360deg); } }

    /* Section header */
    .section-header {
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }

    .section-title {
      font-family: 'Fraunces', serif;
      font-weight: 600;
      font-size: 1.2rem;
      color: var(--text);
      letter-spacing: -0.01em;
    }

    .section-count {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    /* Papers grid */
    .papers-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
      gap: 1rem;
    }

    .paper-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem 1.35rem;
      transition: all 0.18s;
      cursor: pointer;
      box-shadow: var(--shadow);
    }

    .paper-card:hover {
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }

    .paper-domain {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent2);
      background: var(--accent2-light);
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 3px;
      margin-bottom: 0.6rem;
    }

    .paper-title {
      font-family: 'Fraunces', serif;
      font-weight: 600;
      font-size: 0.95rem;
      line-height: 1.4;
      margin-bottom: 0.4rem;
      color: var(--text);
      letter-spacing: -0.01em;
    }

    .paper-meta {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      color: var(--text-light);
      margin-bottom: 0.75rem;
    }

    .paper-abstract {
      font-size: 0.85rem;
      color: var(--text-muted);
      line-height: 1.65;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .paper-sw-insight {
      font-size: 0.82rem;
      color: var(--text-muted);
      line-height: 1.6;
      margin-top: 0.75rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--border);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .paper-sw-label {
      display: inline-block;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.6rem;
      font-weight: 500;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--green);
      background: var(--green-light);
      padding: 0.1rem 0.4rem;
      border-radius: 3px;
      margin-right: 0.4rem;
      vertical-align: middle;
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
      gap: 0.4rem;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      color: var(--green);
    }

    .score-bar {
      width: 44px;
      height: 4px;
      background: var(--border);
      border-radius: 2px;
      overflow: hidden;
    }

    .score-fill {
      height: 100%;
      background: var(--green);
      border-radius: 2px;
    }

    .paper-tags {
      display: flex;
      gap: 0.3rem;
      flex-wrap: wrap;
    }

    .tag {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.6rem;
      padding: 0.15rem 0.45rem;
      border-radius: 3px;
      background: var(--surface2);
      color: var(--text-muted);
      border: 1px solid var(--border);
    }

    /* Empty state */
    .empty-state {
      text-align: center;
      padding: 4rem 2rem;
      color: var(--text-muted);
    }

    .empty-state .icon { font-size: 2rem; margin-bottom: 0.75rem; }
    .empty-state p { font-size: 0.9rem; font-style: italic; }

    /* Views */
    .view { display: none; }
    .view.active { display: block; }

    /* Browse controls */
    .browse-controls {
      display: flex;
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }

    .browse-controls .search-input { flex: 1; }

    .browse-sort {
      background: var(--surface);
      border: 1.5px solid var(--border);
      border-radius: 8px;
      padding: 0.8rem 1rem;
      color: var(--text);
      font-family: 'Source Serif 4', serif;
      font-size: 0.85rem;
      outline: none;
      cursor: pointer;
      white-space: nowrap;
    }

    .browse-sort:focus { border-color: var(--accent); }

    /* Modal */
    .modal-overlay {
      position: fixed; inset: 0;
      background: rgba(20, 18, 15, 0.55);
      backdrop-filter: blur(4px);
      z-index: 500;
      display: flex; align-items: center; justify-content: center;
      padding: 2rem;
      opacity: 0; pointer-events: none;
      transition: opacity 0.2s;
    }

    .modal-overlay.open { opacity: 1; pointer-events: all; }

    .modal {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      max-width: 760px; width: 100%;
      max-height: 88vh; overflow-y: auto;
      position: relative;
      transform: translateY(12px);
      transition: transform 0.25s ease;
      box-shadow: var(--shadow-lg);
    }

    .modal-overlay.open .modal { transform: translateY(0); }

    .modal-header {
      padding: 1.75rem 2rem 1.25rem;
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0;
      background: var(--surface); z-index: 10;
      border-radius: 14px 14px 0 0;
    }

    .modal-domain {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem; font-weight: 500;
      letter-spacing: 0.1em; text-transform: uppercase;
      color: var(--accent2);
      background: var(--accent2-light);
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 3px;
      margin-bottom: 0.6rem;
    }

    .modal-title {
      font-family: 'Fraunces', serif;
      font-weight: 700;
      font-size: 1.25rem;
      line-height: 1.35;
      margin-bottom: 0.5rem;
      color: var(--text);
      letter-spacing: -0.02em;
    }

    .modal-meta {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--text-muted);
    }

    .modal-close {
      position: absolute; top: 1.25rem; right: 1.25rem;
      background: var(--surface2);
      border: 1px solid var(--border);
      color: var(--text-muted);
      border-radius: 6px;
      width: 30px; height: 30px;
      cursor: pointer; font-size: 0.9rem;
      display: flex; align-items: center; justify-content: center;
      transition: all 0.15s;
    }

    .modal-close:hover { color: var(--text); border-color: var(--accent); background: var(--accent-light); }

    .modal-body {
      padding: 1.75rem 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.75rem;
    }

    .modal-section-title {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.65rem; font-weight: 500;
      letter-spacing: 0.1em; text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 0.6rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--border);
    }

    .modal-text {
      font-size: 0.9rem;
      line-height: 1.75;
      color: var(--text);
    }

    .modal-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
    }

    .modal-list li {
      font-size: 0.88rem;
      line-height: 1.65;
      padding-left: 1.25rem;
      position: relative;
      color: var(--text);
    }

    .modal-list li::before {
      content: '→';
      position: absolute;
      left: 0;
      color: var(--accent);
      font-size: 0.75rem;
      top: 0.15rem;
    }

    .modal-sw-box {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
    }

    .modal-score-row {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }

    .modal-score-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      color: var(--text-muted);
    }

    .modal-score-bar {
      flex: 1;
      height: 5px;
      background: var(--border);
      border-radius: 3px;
      overflow: hidden;
    }

    .modal-score-fill {
      height: 100%;
      background: var(--green);
      border-radius: 3px;
    }

    .modal-score-val {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      font-weight: 500;
      color: var(--green);
    }

    .modal-tags { display: flex; flex-wrap: wrap; gap: 0.35rem; }

    .modal-source {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--accent);
      word-break: break-all;
      text-decoration: none;
    }

    .modal-source:hover { text-decoration: underline; }

    .modal-loading {
      text-align: center;
      padding: 3rem;
      color: var(--text-muted);
      font-size: 0.9rem;
      font-style: italic;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
    }

    blockquote {
      border-left: 3px solid var(--accent);
      padding-left: 1rem;
      margin: 0;
      font-style: italic;
      font-size: 0.88rem;
      color: var(--text-muted);
      line-height: 1.7;
    }
  </style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">KB</div>
    STPA Research Knowledge Base
  </div>
  <div class="header-right">
    <div class="tabs">
      <button class="tab active" onclick="showView('search', event)">Search</button>
      <button class="tab" onclick="showView('browse', event)">Browse</button>
    </div>
    <div class="paper-count" id="paperCount">— papers</div>
  </div>
</header>

<main>

  <!-- SEARCH VIEW -->
  <div class="view active" id="view-search">
    <div class="search-panel">
      <span class="search-label">Ask a research question</span>
      <div class="search-row">
        <input
          class="search-input"
          id="queryInput"
          type="text"
          placeholder="How does safety analysis improve system architecture decisions?"
          onkeydown="if(event.key==='Enter') doSearch()"
        />
        <button class="search-btn" id="searchBtn" onclick="doSearch()">Search</button>
      </div>
      <p class="search-hint">Retrieves the 20 most relevant papers and synthesizes an answer with Claude.</p>
    </div>

    <div class="loading" id="loading">
      <div class="spinner"></div>
      Retrieving papers and synthesizing answer…
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
    <div class="browse-controls">
      <input
        class="search-input"
        id="browseSearch"
        type="text"
        placeholder="Filter by title, author, domain, or keyword…"
        oninput="filterPapers()"
      />
      <select id="browseSort" class="browse-sort" onchange="filterPapers()">
        <option value="relevance">Sort: SW Relevance</option>
        <option value="year">Sort: Year</option>
        <option value="title">Sort: Title A–Z</option>
      </select>
    </div>
    <div id="browseEmpty" class="empty-state" style="display:none">
      <div class="icon">🔍</div><p>No papers match your filter.</p>
    </div>
    <div class="papers-grid" id="papersGrid">
      <div class="empty-state"><div class="icon">📚</div><p>Loading papers…</p></div>
    </div>
  </div>

</main>

<!-- PAPER DETAIL MODAL -->
<div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this) closeModal()">
  <div class="modal" id="modal">
    <div id="modalContent">
      <div class="modal-loading"><div class="spinner"></div>Loading paper…</div>
    </div>
  </div>
</div>

<script>
  let allPapers = [];

  function showView(name, event) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('view-' + name).classList.add('active');
    if (event && event.target) event.target.classList.add('active');
    if (name === 'browse' && allPapers.length === 0) loadPapers();
  }

  function filterPapers() {
    const query = document.getElementById('browseSearch').value.toLowerCase().trim();
    const sort = document.getElementById('browseSort').value;

    let filtered = allPapers.filter(p => {
      if (!query) return true;
      const searchable = [
        p.title || '',
        (p.authors || []).join(' '),
        p.industry_domain || '',
        p.abstract_summary || '',
        (p.tags || []).join(' '),
        (p.software_dev_relevance || {}).summary || '',
      ].join(' ').toLowerCase();
      return searchable.includes(query);
    });

    filtered.sort((a, b) => {
      if (sort === 'relevance') return (b.relevance_score || 0) - (a.relevance_score || 0);
      if (sort === 'year') return (b.year || 0) - (a.year || 0);
      if (sort === 'title') return (a.title || '').localeCompare(b.title || '');
      return 0;
    });

    const empty = document.getElementById('browseEmpty');
    if (filtered.length === 0) {
      empty.style.display = 'block';
      document.getElementById('papersGrid').innerHTML = '';
    } else {
      empty.style.display = 'none';
      renderPapersGrid(filtered, 'papersGrid');
    }

    document.getElementById('browseCount').textContent =
      filtered.length === allPapers.length
        ? allPapers.length + ' indexed'
        : filtered.length + ' of ' + allPapers.length;
  }

  async function loadPapers() {
    const res = await fetch('/api/papers');
    const data = await res.json();
    allPapers = data.papers || [];
    document.getElementById('paperCount').textContent = allPapers.length + ' papers';
    document.getElementById('browseCount').textContent = allPapers.length + ' indexed';
    renderPapersGrid(allPapers, 'papersGrid');
  }

  function paperCardHTML(p) {
    const score = p.relevance_score || 0;
    const pct = (score / 10) * 100;
    const tags = (p.tags || []).slice(0, 3).map(t => `<span class="tag">${t}</span>`).join('');
    const authors = (p.authors || []).slice(0,2).join(', ') + ((p.authors||[]).length > 2 ? ' et al.' : '');
    const swSummary = (p.software_dev_relevance || {}).summary || '';
    return `
      <div class="paper-card" data-title="${(p.title||'').replace(/"/g, '&quot;')}">
        <div class="paper-domain">${p.industry_domain || 'Research'}</div>
        <div class="paper-title">${p.title}</div>
        <div class="paper-meta">${authors}${p.year ? ' · ' + p.year : ''}</div>
        <div class="paper-abstract">${p.abstract_summary || ''}</div>
        ${swSummary ? `<div class="paper-sw-insight"><span class="paper-sw-label">SW Insight</span>${swSummary}</div>` : ''}
        <div class="paper-footer">
          <div class="score-badge">
            <div class="score-bar"><div class="score-fill" style="width:${pct}%"></div></div>
            SW ${score}/10
          </div>
          <div class="paper-tags">${tags}</div>
        </div>
      </div>`;
  }

  function renderPapersGrid(papers, targetId) {
    const grid = document.getElementById(targetId);
    if (!papers.length) {
      grid.innerHTML = '<div class="empty-state"><div class="icon">🔍</div><p>No papers found.</p></div>';
      return;
    }
    grid.innerHTML = papers.map(paperCardHTML).join('');
  }

  async function openPaper(title) {
    const overlay = document.getElementById('modalOverlay');
    const content = document.getElementById('modalContent');
    content.innerHTML = '<div class="modal-loading"><div class="spinner"></div>Loading paper…</div>';
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';

    try {
      const res = await fetch('/api/paper?title=' + encodeURIComponent(title));
      const p = await res.json();
      if (p.error) { content.innerHTML = `<div class="modal-loading">Error: ${p.error}</div>`; return; }

      const authors = (p.authors || []).join(', ');
      const swRel = p.software_dev_relevance || {};
      const score = swRel.relevance_score || 0;
      const pct = (score / 10) * 100;
      const swApps = swRel.specific_applications || [];
      const antiPatterns = swRel.anti_patterns || [];
      const findings = p.key_findings || [];
      const limitations = p.limitations || [];
      const relatedWork = p.related_work || [];
      const quotes = p.notable_quotes || [];
      const tags = p.tags || [];
      const source = p.source || '';

      const section = (label, html) => `
        <div>
          <div class="modal-section-title">${label}</div>
          ${html}
        </div>`;

      const list = items => `<ul class="modal-list">${items.map(i => `<li>${i}</li>`).join('')}</ul>`;

      content.innerHTML = `
        <div class="modal-header">
          <button class="modal-close" onclick="closeModal()">✕</button>
          <div class="modal-domain">${p.industry_domain || 'Research'}</div>
          <div class="modal-title">${p.title}</div>
          <div class="modal-meta">${authors}${p.year ? ' · ' + p.year : ''}</div>
        </div>
        <div class="modal-body">

          ${section('Abstract', `<div class="modal-text">${p.abstract_summary || 'N/A'}</div>`)}

          ${findings.length ? section('Key Findings', list(findings)) : ''}

          ${p.methodology ? section('Methodology', `<div class="modal-text">${p.methodology}</div>`) : ''}

          ${limitations.length ? section('Limitations', list(limitations)) : ''}

          ${relatedWork.length ? section('Related Work & Frameworks', list(relatedWork)) : ''}

          ${quotes.length ? section('Notable Quotes', `
            <div style="display:flex;flex-direction:column;gap:0.75rem">
              ${quotes.map(q => `<blockquote>${q}</blockquote>`).join('')}
            </div>`) : ''}

          <div>
            <div class="modal-section-title">Software Development Relevance</div>
            <div class="modal-sw-box">
              <div class="modal-score-row">
                <span class="modal-score-label">Relevance Score</span>
                <div class="modal-score-bar"><div class="modal-score-fill" style="width:${pct}%"></div></div>
                <span class="modal-score-val">${score}/10</span>
              </div>
              <div class="modal-text" style="margin-bottom:${swApps.length ? '1rem' : '0'}">${swRel.summary || 'N/A'}</div>
              ${swApps.length ? `
                <div class="modal-section-title" style="margin-top:0.75rem">Specific Applications</div>
                ${list(swApps)}` : ''}
              ${antiPatterns.length ? `
                <div class="modal-section-title" style="margin-top:1rem;color:var(--accent2)">Anti-Patterns to Avoid</div>
                <ul class="modal-list">
                  ${antiPatterns.map(a => `<li style="color:var(--text)">${a}</li>`).join('')}
                </ul>` : ''}
            </div>
          </div>

          ${tags.length ? section('Tags', `<div class="modal-tags">${tags.map(t => `<span class="tag">${t}</span>`).join('')}</div>`) : ''}

          ${source ? section('Source', `<a class="modal-source" href="${source}" target="_blank" rel="noopener">${source}</a>`) : ''}

        </div>`;
    } catch(e) {
      content.innerHTML = `<div class="modal-loading">Failed to load paper details.</div>`;
    }
  }

  function closeModal() {
    document.getElementById('modalOverlay').classList.remove('open');
    document.body.style.overflow = '';
  }

  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  function renderMarkdown(text) {
    if (typeof marked !== 'undefined') return marked.parse(text);
    const div = document.createElement('div');
    div.textContent = text;
    return '<pre style="white-space:pre-wrap;font-family:inherit">' + div.innerHTML + '</pre>';
  }

  async function doSearch() {
    const q = document.getElementById('queryInput').value.trim();
    if (!q) return;

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
        body: JSON.stringify({ query: q })
      });
      const data = await res.json();

      if (data.error) {
        answerPanel.style.display = 'block';
        document.getElementById('answerContent').textContent = 'Error: ' + data.error;
        return;
      }

      document.getElementById('answerQuery').textContent = '"' + q + '"';
      document.getElementById('answerContent').innerHTML = renderMarkdown(data.answer);
      answerPanel.style.display = 'block';

      if (data.papers && data.papers.length) {
        const header = document.createElement('div');
        header.className = 'section-header';
        header.style.marginTop = '2rem';
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

  document.addEventListener('click', e => {
    const card = e.target.closest('.paper-card');
    if (card && card.dataset.title) openPaper(card.dataset.title);
  });

  fetch('/api/papers').then(r => r.json()).then(d => {
    document.getElementById('paperCount').textContent = (d.papers||[]).length + ' papers';
  });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
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

@app.route("/api/paper")
def api_paper():
    """Return full details for a single paper by title from summaries.json."""
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "No title provided"}), 400
    summaries_path = "./summaries.json"
    if not os.path.exists(summaries_path):
        return jsonify({"error": "summaries.json not found"}), 404
    with open(summaries_path) as f:
        summaries = json.load(f)
    for paper in summaries:
        if paper.get("title", "").strip().lower() == title.lower():
            return jsonify(paper)
    return jsonify({"error": "Paper not found"}), 404


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
    port = int(os.environ.get("PORT", 5000))
    print("\n  Research Knowledge Base")
    print(f"  {collection.count()} papers indexed")
    print(f"  Running at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

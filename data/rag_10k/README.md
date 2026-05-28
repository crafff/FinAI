# 10-K RAG Chunking & Vectorization

This module implements Task 7: chunk a 10-K's full text, vectorize it,
and expose it as a retrievable tool function for the downstream agents.

The module takes:

    - ticker
    - accession_number
    - cleaned 10-K plain text (from EDGAR retrieval, Task 1)
    - optional cache directory
    - optional embedding model name (default BAAI/bge-small-en-v1.5)
    - optional chunk size and overlap

and returns a retrieval index plus a callable tool. The Fundamental
agent (Task 10) and the Risk qualitative agent (Task 12) use the
callable to ask focused questions of the 10-K instead of stuffing the
whole document into a prompt.

The module performs five main steps:

1. Section split

   The cleaned 10-K text is split on the regex

       ^Item\s+(\d+[A-Z]?)\.\s*(.*)$    (multiline, case-insensitive)

   Each Item header found at a line start opens a new section. The
   body that follows runs up to the next Item header. Sections shorter
   than 200 characters are dropped, which removes the table of contents
   that appears at the top of every 10-K.

   The regex captures two groups:

       - section_code   "1", "1A", "7", "7A", ...
       - section_title  "Risk Factors", "MD&A", or "" if absent

   Inline-title variants such as "Item 1A.Risk Factors" (AMZN) and
   "Item 1.BUSINESS" (AMGN) are handled by the same regex.

2. Size chunking

   Each surviving section is chunked into ~1000-character pieces with
   150 characters of overlap between consecutive chunks. Section
   metadata (code and title) is inherited so the agent can filter on
   it later. Char offsets in the source text are preserved for
   citation back to the original.

3. Embedding

   Chunks are embedded with sentence-transformers. The default model
   is BAAI/bge-small-en-v1.5 (384-dim). Embeddings are L2-normalized
   at write time so cosine similarity reduces to a single dot product.

4. Caching

   When a cache directory is provided, the index is written under
   <cache_dir>/<TICKER>/ as two files per filing:

       - <accession>.chunks.json      list of chunk dicts (text + metadata)
       - <accession>.embeddings.npz   (N, d) float32 matrix + model_name

   Writes go through .tmp + rename so an interrupted run cannot leave
   a partial cache entry. A cached embeddings.npz whose model_name
   does not match the requested model is treated as invalid and the
   index is rebuilt. This prevents silently mixing 384-dim and
   1536-dim spaces if the embedding model is swapped later.

5. Retrieval

   retrieve(index, query, k=5, section=None) embeds the query, dots
   it against the stored matrix, and returns the top-k chunks (each
   with a "similarity" float). The section filter accepts a single
   code or a list of codes, case-insensitive, so the Risk qualitative
   agent can request only Item 1A.

The chunk schema is:

    {
        "chunk_id": int,
        "section_code": str,
        "section_title": str,
        "text": str,
        "char_start": int,
        "char_end": int,
    }

The index schema is:

    {
        "ticker": str,
        "accession_number": str,
        "model_name": str,
        "chunks": list[chunk_dict],
        "embeddings": np.ndarray  (N, d) float32, L2-normalized,
    }

## Retrieval as a tool

make_retrieval_tool(index) returns a closure-bound callable suitable
for LangGraph tool binding:

    retrieve_10k(query: str, k: int = 5, section: str | None = None) -> str

Calling it returns a formatted string with one block per hit:

    [chunk 12 | Item 1A. Risk Factors | sim=0.81]
    <chunk text>

    [chunk 27 | Item 7. MD&A | sim=0.76]
    <chunk text>

Each ticker gets its own bound tool; the orchestrator decides which
to load for a given run.

## Usage

### 1. Install dependencies

    cd data/rag_10k
    pip install -r requirements.txt

requirements.txt pulls in numpy, sentence-transformers, and pytest.
The embedding model itself (BAAI/bge-small-en-v1.5, ~130 MB) is
downloaded lazily on the first call that actually embeds something.

### 2. Run the tests

    pytest -v

All unit tests run offline. The embedder is monkeypatched to a
deterministic hash-based fake (16-d vectors) so chunking, retrieval,
section filtering, fallback behavior, and cache invalidation are all
tested without any model download. One smoke test exercises the real
embedder; it is automatically skipped if sentence-transformers is
not installed.

### 3. Run against real 10-K data

This module reads cleaned text produced by EDGAR_retrieval, so make
sure data/EDGAR_retrieval/cache/ is populated first by running
data/EDGAR_retrieval/run_fetch.py.

Single filing, from a Python REPL or script:

    from pathlib import Path
    from rag_10k import build_or_load_index, retrieve, make_retrieval_tool

    text = Path(
        "../EDGAR_retrieval/cache/AAPL/0000320193-25-000079.txt"
    ).read_text(encoding="utf-8")

    index = build_or_load_index(
        ticker="AAPL",
        accession_number="0000320193-25-000079",
        text=text,
        cache_dir=Path("./cache"),
    )

    hits = retrieve(
        index,
        query="supply chain risks in China",
        k=3,
        section="1A",
    )

    for h in hits:
        print(f"[{h['chunk_id']} | Item {h['section_code']} | "
              f"sim={h['similarity']:.3f}] {h['text'][:200]} ...")

    tool = make_retrieval_tool(index)
    print(tool("How does the company manage cybersecurity?", k=2))

All 30 cached filings at once:

    python run_build_indexes.py

The runner walks every <accession>.meta.json under
../EDGAR_retrieval/cache/, reads the matching .txt, and calls
build_or_load_index. Per-ticker failures do not stop the run; a
summary prints at the end. Already-built indexes return immediately
from disk, so reruns are near-instant.

Two 10-K filings (HON and MCD) do not use Item-prefixed body section
headers and trigger the whole-document fallback, so their chunks
carry section_code="" and the section filter has no effect on them.
Semantic retrieval still works for those tickers, just without the
section-code filter.

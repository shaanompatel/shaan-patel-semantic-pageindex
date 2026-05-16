# SemanticPageIndex

## Overview

SemanticPageIndex is an extension of [PageIndex](https://github.com/VectifyAI/PageIndex) — VectifyAI's open-source, vectorless, reasoning-based RAG system — redesigned to generate high-quality hierarchical tree indexes for **general-purpose documents**, not just structured ones with explicit tables of contents.

The original PageIndex excels at professional documents (financial reports, legal filings, technical manuals) where a printed TOC and explicit section headings already encode document structure. But for novels, essays, transcripts, narrative non-fiction, mixed-structure PDFs, or any weakly-structured document, the heading-centric pipeline produces shallow or inaccurate trees.

SemanticPageIndex addresses this by adding an **automatic document routing layer** and a new **semantic segmentation path** that identifies section boundaries by detecting shifts in topic, tone, scene, setting, and narrative — not just by finding headings. The result is a much richer, more accurate table of contents for general documents, while still falling back to the original TOC-centric pipeline for structured documents where it performs best.

This is a second iteration built directly on the PageIndex open-source codebase. The original README and project background are preserved in the references section below.

## Setup

### Python and pip versions

This module was developed and tested with:

- **Python 3.11.9**
- **pip 24.0**

Other Python 3.9+ versions should work, but these were the versions used during development.

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SemanticPageIndex
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

The key dependencies are:

| Package | Version | Purpose |
|---|---|---|
| `litellm` | 1.83.0 | Multi-provider LLM routing (OpenAI, Anthropic, OpenRouter, etc.) |
| `pymupdf` | 1.26.4 | Robust PDF text extraction |
| `PyPDF2` | 3.0.1 | Fallback PDF parser |
| `python-dotenv` | 1.1.0 | Environment variable loading |
| `pyyaml` | 6.0.2 | Config file parsing |
| `openai-agents` | latest | Required only for the agentic RAG demo |

### 3. Configure your API key

Copy the config template and fill in your API key:

```bash
cp pageindex/config-template.yaml pageindex/config.yaml
# Then edit config.yaml and set model_api_key (and agent_api_key if using the demo)
```

`pageindex/config.yaml` is listed in `.gitignore` and will never be committed. You can also export keys as environment variables instead of writing them to the file:

```bash
export PAGEINDEX_MODEL_API_KEY=your_key_here
export PAGEINDEX_AGENT_API_KEY=your_key_here   # only for the agentic demo
```

The config supports any LLM provider via [LiteLLM](https://docs.litellm.ai/docs/providers). To use OpenAI directly, set:

```yaml
model: "gpt-4o"
model_api_base: ""
model_api_key: ""  # or export OPENAI_API_KEY
```

To use OpenRouter (as in the template):

```yaml
model: "openai/gpt-5.4-nano"
model_api_base: https://openrouter.ai/api/v1
model_api_key: "sk-or-v1-..."
```

### 4. Run the tests

```bash
python -m pytest tests/
```

The test suite covers provider config loading and key resolution logic.

## Codebase Structure

```
SemanticPageIndex/
    - requirements.txt
    - run_pageindex.py              # CLI entry point
    - PAGEINDEX_DEEP_DIVE.md        # Technical deep-dive on the indexing pipeline
    - PageIndex-vs-PageIndex-General-indexing-differences.md  # Diff from original PageIndex
    - pageindex/
        -- config-template.yaml     # Safe template to copy → config.yaml
        -- config.yaml              # Local config with your keys (gitignored)
        -- page_index.py            # Core PDF indexing pipeline (heavily extended)
        -- page_index_md.py         # Markdown indexing (header-based tree)
        -- utils.py                 # LLM helpers, PDF I/O, tree utilities
        -- retrieve.py              # Retrieval helpers (get_document_structure, get_page_content)
        -- client.py                # High-level PageIndexClient with workspace storage
    - examples/
        -- agentic_vectorless_rag_demo.py   # End-to-end agentic RAG with OpenAI Agents SDK
        -- tutorials/               # Doc-search and tree-search worked examples
    - cookbook/
        -- pageindex_RAG_simple.ipynb       # Minimal vectorless RAG notebook
        -- vision_RAG_pageindex.ipynb       # OCR-free, vision-based RAG
        -- agentic_retrieval.ipynb
        -- pageIndex_chat_quickstart.ipynb
    - tests/
        -- test_provider_config.py  # Provider config and API key resolution tests
    - financebench/                 # FinanceBench evaluation data and playground
    - logs/                         # Runtime diagnostic JSON logs (gitignored)
    - results/                      # Generated tree structure JSON files (gitignored)
```

Key files and components:

- `pageindex/page_index.py`: The core PDF pipeline. Contains all original PageIndex logic plus the new document routing layer, semantic segmentation path, header/footer removal, and parent-child boundary enforcement.
- `pageindex/utils.py`: Shared helpers — PDF text extraction, LLM wrappers with retry/backoff, tree traversal, token counting, and the `ConfigLoader`.
- `pageindex/page_index_md.py`: Markdown indexing via `#`-header hierarchy, with optional tree thinning for over-fragmented trees.
- `pageindex/client.py`: `PageIndexClient` — persists indexed documents to a workspace directory and exposes `get_document`, `get_document_structure`, and `get_page_content` as retrieval tools.
- `pageindex/retrieve.py`: Retrieval layer delegated to by the client.
- `run_pageindex.py`: CLI with full argument coverage including the new `--document-type` and `--no-header-footer-removal` flags.

## Functional Design (Usage)

### CLI: index a PDF or Markdown file

```bash
# Index a PDF (auto-detects document type)
python3 run_pageindex.py --pdf_path /path/to/document.pdf

# Force semantic mode for a clearly unstructured doc
python3 run_pageindex.py --pdf_path /path/to/novel.pdf --document-type unstructured

# Force structured mode (original PageIndex behavior)
python3 run_pageindex.py --pdf_path /path/to/annual-report.pdf --document-type structured

# Disable header/footer removal (e.g. if removal causes issues on a specific PDF)
python3 run_pageindex.py --pdf_path /path/to/document.pdf --no-header-footer-removal

# Index a Markdown file
python3 run_pageindex.py --md_path /path/to/document.md
```

Output is written to `results/<document_name>_structure.json`.

Full CLI reference:

```
--pdf_path                  Path to input PDF
--md_path                   Path to input Markdown file
--model                     LLM model (overrides config.yaml)
--model-api-base            Custom API base URL
--model-api-key             API key (prefer env var instead)
--toc-check-pages           Pages to scan for embedded TOC (default: 20)
--max-pages-per-node        Max pages per tree node (default: 10)
--max-tokens-per-node       Max tokens per tree node (default: 20000)
--document-type             {auto, structured, unstructured} — routing mode
--no-header-footer-removal  Disable repeated header/footer stripping
--if-add-node-id            Include node IDs in output (yes/no)
--if-add-node-summary       Generate LLM summaries per node (yes/no)
--if-add-doc-description    Generate one-sentence doc description (yes/no)
--if-add-node-text          Include raw page text in output (yes/no)
--skip-toc-detection        Skip embedded TOC detection entirely (yes/no)

# Markdown only:
--if-thinning               Merge tiny nodes into parents (yes/no, default: no)
--thinning-threshold        Min token count to keep a node (default: 5000)
--summary-token-threshold   Min tokens in a node before summarizing (default: 200)
```

### Python API

```python
from pageindex import page_index_main
from pageindex.utils import ConfigLoader

opt = ConfigLoader().load({})
structure = page_index_main("path/to/document.pdf", opt)
# structure is a nested dict tree:
# { "title": "...", "node_id": "0001", "start_index": 1, "end_index": 12,
#   "summary": "...", "nodes": [ ... ] }
```

```python
from pageindex.client import PageIndexClient

client = PageIndexClient(workspace_dir="./my_index")

# Index a document (stores to disk, returns doc_id)
doc_id = client.index("path/to/document.pdf")

# Retrieve the tree without page text (token-efficient)
structure = client.get_document_structure(doc_id)

# Fetch specific page content by range string
pages = client.get_page_content(doc_id, "5-8")   # pages 5 through 8
pages = client.get_page_content(doc_id, "3,7,12") # specific pages

# Get metadata
meta = client.get_document(doc_id)
# { "doc_name": "...", "doc_description": "...", "page_count": 42 }
```

```python
from pageindex.page_index_md import md_to_tree
import asyncio

tree = asyncio.run(md_to_tree(
    md_path="path/to/document.md",
    if_thinning=True,
    min_token_threshold=5000,
    if_add_node_summary="yes",
))
```

## Demo Video

[Watch the demo walkthrough on Google Drive](https://drive.google.com/file/d/1AU7UaSypaeh9AaMVnDeOfDeVYd1AKPdj/view?usp=drive_link)

## Algorithmic Design

### Original PageIndex pipeline

PageIndex operates in two phases:

1. **Indexing**: parse the document into a hierarchical tree of sections with page-range boundaries (`start_index`, `end_index`), optional LLM summaries, and stable node IDs.
2. **Retrieval**: an LLM agent reads the tree (`get_document_structure`), reasons about which page ranges are relevant to a query, and fetches only those pages (`get_page_content`). No vector embeddings are used at any stage.

For structured PDFs, the original pipeline detects an embedded TOC, maps printed page numbers to physical PDF pages, verifies each section's page assignment against the actual text, and runs correction loops for any mismatches.

### SemanticPageIndex extensions

SemanticPageIndex adds the following components on top of the original pipeline:

#### 1. Document routing layer

Before any indexing begins, the document is classified along two axes: **structural density** (how many explicit headings it contains) and **genre** (report, narrative, essay, mixed, etc.). Based on this classification, `document_type` is resolved to either `structured` or `unstructured`. The user can override this with `--document-type`.

- `structured` → original PageIndex TOC-centric path
- `unstructured` → new semantic segmentation path
- `auto` (default) → classifier decides

This replaces the original default of always attempting TOC detection first. SemanticPageIndex defaults to `skip_toc_detection: "yes"` and `document_type: "auto"`, trusting the classifier over brute-force TOC scanning for general documents.

**Justification**: TOC detection on a novel or an essay always fails (there is no TOC), wasting LLM calls and often degrading to a worse fallback. Routing up front avoids this and lets the semantic path do its job cleanly.

#### 2. Semantic segmentation path

For unstructured documents, three new functions handle indexing:

- **`generate_semantic_toc_init(chunk)`**: Given the first text chunk of the document, the LLM identifies the initial set of sections by detecting shifts in topic, setting, tone, scene, narrator, or subject — not by finding headings.
- **`generate_semantic_toc_continue(chunk, prior_sections)`**: Extends the section list for each subsequent chunk, anchored to the previously identified boundaries so the list stays globally consistent.
- **`refine_semantic_boundaries(sections, page_list)`**: A refinement pass that reviews the raw boundary assignments and adjusts them for coherence — merging over-fragmented transitions and splitting sections that span too many distinct topics.

The final flat section list is converted to a nested tree using the same `post_processing` / `list_to_tree` utilities as the structured path.

**Justification**: A novel chapter doesn't start with "Chapter 3" on every page. A scene break in a short story has no heading. The only way to produce a meaningful hierarchy for these documents is to model the *content* transitions, not scan for typographic signals. The semantic path asks the LLM to do what a careful reader would: notice when the subject changes.

#### 3. Header/footer removal (`remove_repeated_headers_and_footers`)

Before indexing, a sampling pass identifies lines that appear repeatedly near the top or bottom of pages across the document (running headers like "CHAPTER 2 — THE JOURNEY" or footers like "© 2024 Publisher — Page N"). These are stripped from page text before any LLM prompts are constructed.

The algorithm:
1. Samples every Nth page (default: every 3rd).
2. Counts normalized occurrences of the first and last `edge_line_window` (default: 3) lines.
3. Marks any line appearing on ≥ 65% of sampled pages as a common header/footer.
4. Strips matching lines from all pages before they enter the LLM prompts.

**Justification**: Running headers are noise for boundary detection. If every page of a chapter starts with "PART II: CONSEQUENCES", the LLM will incorrectly anchor on that as a section signal. Removing it lets the semantic content come through cleanly.

#### 4. Semantic-aware recursive sub-indexing

The original PageIndex recursively splits large nodes using only the `process_no_toc` path (heading-based inference from body text). SemanticPageIndex preserves the active indexing strategy during recursion: if a node was indexed semantically, its over-large children are also re-indexed semantically (`child_mode = process_semantic`).

**Justification**: Forcing heading-based inference on a node that belongs to an unstructured document is inconsistent and tends to produce poor sub-trees. Method consistency at depth produces more coherent hierarchies.

#### 5. Parent-child boundary enforcement (`enforce_parent_child_page_bounds`)

After tree construction, a post-processing pass verifies that every child node's `[start_index, end_index]` range is properly contained within its parent's range. Any violations (which can arise from imperfect LLM boundary assignments) are clamped to the parent's span.

**Justification**: Without this, downstream retrieval calls using `get_page_content` on a child node can silently return pages that belong to a sibling or adjacent section, corrupting the evidence retrieved by the agent.

#### 6. Genre-aware summary generation

When generating LLM summaries for nodes, SemanticPageIndex passes the inferred `genre` and `document_type` into the summary prompt so the model can write appropriate summaries — a summary of a narrative scene should read differently from a summary of a financial disclosure.

### Architecture diagram

```
PDF or Markdown
      │
      ▼
┌─────────────────────────────────────────┐
│ Document Routing (NEW)                  │
│  classify structural density + genre    │
│  → document_type: structured / unstr.  │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
 structured        unstructured
       │                │
       │   ┌────────────────────────────┐
       │   │ Header/Footer Removal (NEW) │
       │   │ sample pages → strip noise  │
       │   └────────────┬───────────────┘
       │                │
       ▼                ▼
 TOC Detection    Semantic Segmentation (NEW)
 + Page Mapping   generate_semantic_toc_init
 + Verification   generate_semantic_toc_continue
 + Fix Loops      refine_semantic_boundaries
       │                │
       └───────┬────────┘
               ▼
     post_processing: flat list → nested tree
               │
               ▼
     enforce_parent_child_page_bounds (NEW)
               │
               ▼
     Large-node recursive split
     (preserves semantic mode) (NEW)
               │
               ▼
     Optional enrichment:
     summaries (genre-aware) (NEW), node IDs, doc description
               │
               ▼
     Stored tree index (results/<name>_structure.json)
               │
               ▼
     ┌──────────────────────────────┐
     │ Retrieval (unchanged)        │
     │ get_document_structure       │
     │     → LLM reasons over tree  │
     │ get_page_content(pages)      │
     └──────────────────────────────┘
```

## Issues and Future Work

- **Semantic boundary quality degrades on very long documents**: `generate_semantic_toc_continue` processes the document in token-budget chunks with one-page overlap between chunks. Transitions that span chunk boundaries can be misidentified. A sliding-window approach with larger overlap would help.
- **Genre classification is prompt-based and not calibrated**: The document classifier uses a single LLM call with no few-shot examples. Misclassifications (e.g. a densely-formatted essay being called `structured`) will route to the wrong pipeline.
- **No evaluation benchmark for general documents**: PageIndex's structured-document quality is validated against FinanceBench. There is no equivalent benchmark for narrative/unstructured documents that would let us measure the improvement from semantic indexing quantitatively.
- **Header/footer removal uses fixed thresholds**: The 65% repetition threshold and 3-line edge window work well on typical trade paperbacks but may need tuning for documents with irregular formatting (e.g. screenplays, transcripts).
- **Semantic mode does not run the title-appearance verification loop**: Unlike the TOC-centric path (which verifies each section's title appears on its assigned page), semantic sections are not verified against explicit title text because unstructured sections often have no title. A coherence-based verification pass would strengthen confidence in boundary placement.
- **Markdown path is unchanged from upstream**: The header-based markdown indexing does not benefit from semantic routing. Extending semantic segmentation to Markdown (useful for long prose documents in `.md` format) is a natural next step.

## Change Log

This is the second iteration of the project. The first iteration was the upstream [PageIndex](https://github.com/VectifyAI/PageIndex) open-source repository by VectifyAI (Mingtian Zhang, Yu Tang et al.). This iteration adds the semantic indexing path, document routing, header/footer removal, boundary enforcement, and genre-aware summarization described above.

A detailed technical comparison between this fork and the upstream codebase is in [`PageIndex-vs-PageIndex-General-indexing-differences.md`](PageIndex-vs-PageIndex-General-indexing-differences.md).

A full walkthrough of the indexing and retrieval pipeline is in [`PAGEINDEX_DEEP_DIVE.md`](PAGEINDEX_DEEP_DIVE.md).

## References

- **Upstream project**: [PageIndex by VectifyAI](https://github.com/VectifyAI/PageIndex) — the open-source codebase this project extends.
- **PageIndex framework blog post**: Mingtian Zhang, Yu Tang and PageIndex Team, "PageIndex: Next-Generation Vectorless, Reasoning-based RAG", PageIndex Blog, Sep 2025. https://pageindex.ai/blog/pageindex-intro
- **FinanceBench benchmark**: Referring Mafin 2.5 results: https://github.com/VectifyAI/Mafin2.5-FinanceBench
- **FinanceBench paper**: Islam et al. (2023). FinanceBench: A new benchmark for financial question answering. https://arxiv.org/abs/2311.11944
- **LiteLLM**: Multi-provider LLM routing library used for all model calls. https://docs.litellm.ai/
- **PyMuPDF**: PDF text extraction library. https://pymupdf.readthedocs.io/
- **OpenAI Agents SDK**: Used in the agentic RAG demo. https://openai.github.io/openai-agents-python/

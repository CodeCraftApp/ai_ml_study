# Chapter 2: Building Retrieval-Augmented Generation (RAG)

> **Goal**: Understand every moving piece of a RAG system — the *why*, the *math*, the *algorithms*, the *code abstractions*, and how they all snap together — so you can design, debug, and improve one from scratch.

---

## Table of Contents

1. [Why RAG Exists — The Problem Statement](#1-why-rag-exists--the-problem-statement)
2. [RAG at 30,000 Feet — The Two-Phase Architecture](#2-rag-at-30000-feet--the-two-phase-architecture)
3. [Phase 1 — Indexing Pipeline (Offline)](#3-phase-1--indexing-pipeline-offline)
   - 3.1 [Document Loading & Preprocessing](#31-document-loading--preprocessing)
   - 3.2 [Chunking — Splitting Documents](#32-chunking--splitting-documents)
   - 3.3 [All Chunking Strategies (Definitions & Examples)](#33-all-chunking-strategies-definitions--examples)
      - [Strategy 1: Fixed-Size Character Splitting](#strategy-1-fixed-size-character-splitting)
      - [Strategy 2: Separator-Based Splitting](#strategy-2-separator-based-splitting)
      - [Strategy 3: Recursive Character Splitting (RECOMMENDED)](#strategy-3-recursive-character-splitting-recommended)
      - [Strategy 4: Token-Based Splitting](#strategy-4-token-based-splitting)
      - [Strategy 5: Sentence-Level Splitting](#strategy-5-sentence-level-splitting)
      - [Strategy 6: Semantic Chunking](#strategy-6-semantic-chunking)
      - [Strategy 7: Document-Structure-Aware Splitting](#strategy-7-document-structure-aware-splitting)
      - [Chunking Strategies Summary Table](#chunking-strategies-summary-table)
   - 3.4 [What Gets Tokenized, When, and Why](#34-what-gets-tokenized-when-and-why)
   - 3.5 [Embedding — From Text to Vectors (Step-by-Step)](#35-embedding--from-text-to-vectors-step-by-step)
      - [Step 1: Tokenization](#step-1-tokenization-covered-in-34-above)
      - [Step 2: Token Embedding Lookup](#step-2-token-embedding-lookup)
      - [Visualizing the Token-to-Vector Journey](#visualizing-the-token-to-vector-journey)
      - [Step 3: Positional Encoding — Adding Word Order](#step-3-positional-encoding--adding-word-order)
      - [Step 4: Self-Attention — The Core of the Transformer](#step-4-self-attention--the-core-of-the-transformer)
      - [Step 5: Mean Pooling — One Vector for the Whole Chunk](#step-5-mean-pooling--one-vector-for-the-whole-chunk)
   - 3.6 [Vector Store — Storing & Indexing Embeddings](#36-vector-store--storing--indexing-embeddings)
4. [Phase 2 — Retrieval & Generation Pipeline (Online)](#4-phase-2--retrieval--generation-pipeline-online)
   - 4.1 [Query Embedding](#41-query-embedding)
   - 4.2 [Retrieval — Finding Relevant Chunks](#42-retrieval--finding-relevant-chunks)
   - 4.3 [Augmented Prompt Construction](#43-augmented-prompt-construction)
   - 4.4 [LLM Generation](#44-llm-generation)
5. [The Mathematics — Deep Dive (Beginner Friendly)](#5-the-mathematics--deep-dive-beginner-friendly)
   - 5.1 [Vectors — The Language Computers Speak](#51-vectors--the-language-computers-speak)
   - 5.2 [Token Embeddings — How Words Become Numbers](#52-token-embeddings--how-words-become-numbers)
   - 5.3 [Positional Encoding — Teaching Order to the Model](#53-positional-encoding--teaching-order-to-the-model)
   - 5.4 [Self-Attention — How the Model "Reads"](#54-self-attention--how-the-model-reads)
   - 5.5 [Mean Pooling — Merging Tokens into One Chunk Vector](#55-mean-pooling--merging-tokens-into-one-chunk-vector)
   - 5.6 [Similarity Metrics — How Retrieval Ranks Chunks](#56-similarity-metrics--how-retrieval-ranks-chunks)
      - [Cosine Similarity](#cosine-similarity-most-common-in-rag)
      - [Dot Product](#dot-product-inner-product)
      - [Euclidean (L2) Distance](#euclidean-l2-distance)
   - 5.7 [Contrastive Loss — How Embedding Models are Trained](#57-contrastive-loss--how-embedding-models-are-trained)
   - 5.8 [BM25 — The Classic Keyword Algorithm](#58-bm25--the-classic-keyword-algorithm)
   - 5.9 [ANN Algorithms — Fast Search in Millions of Vectors](#59-ann-algorithms--fast-search-in-millions-of-vectors)
      - [HNSW (Hierarchical Navigable Small World)](#hnsw-hierarchical-navigable-small-world)
      - [IVF (Inverted File Index)](#ivf-inverted-file-index)
      - [Product Quantization (PQ)](#product-quantization-pq)
   - 5.10 [Attention in the Generator LLM](#510-attention-in-the-generator-llm)
6. [Key Algorithms Behind RAG](#6-key-algorithms-behind-rag)
7. [How Everything Plays Together — System Data Flow](#7-how-everything-plays-together--system-data-flow)
8. [What Each Technology Provides](#8-what-each-technology-provides)
9. [Advanced RAG Patterns](#9-advanced-rag-patterns)
   - 9.1 [Naive RAG vs. Advanced RAG vs. Modular RAG](#91-naive-rag-vs-advanced-rag-vs-modular-rag)
   - 9.2 [Query Transformation Techniques](#92-query-transformation-techniques)
   - 9.3 [Re-Ranking](#93-re-ranking)
10. [Conversation Memory in RAG](#10-conversation-memory-in-rag)
11. [Prompt Engineering for RAG](#11-prompt-engineering-for-rag)
12. [Evaluation Metrics for RAG](#12-evaluation-metrics-for-rag)
13. [End-to-End Code Walkthrough (LangChain)](#13-end-to-end-code-walkthrough-langchain)
14. [Reference Papers & Further Reading](#14-reference-papers--further-reading)

---

## 1. Why RAG Exists — The Problem Statement

Large Language Models (LLMs) like GPT, LLaMA, and Granite are trained on massive public corpora at a fixed point in time. This creates three fundamental problems:

| Problem | Description |
|---|---|
| **Hallucination** | The model confidently generates plausible but factually incorrect answers because it is pattern-completing, not fact-checking. |
| **Knowledge Cutoff** | Training data has a date boundary. Anything after that date is invisible to the model. |
| **No Private Data** | The model has never seen your company documents, policies, or proprietary data. |

### Why not just use long context windows?

Modern models support 128K+ token contexts. But:

1. **Input Dependency** — The user must already possess the source material.
2. **Capacity Limits** — 128K tokens ≈ 96K words. Many real corpora (legal databases, medical records) exceed this by orders of magnitude.
3. **Needle in a Haystack** — Stuffing irrelevant content dilutes the LLM's attention over the truly important passages.
4. **Latency** — Processing time scales with token count; longer prompts = slower responses.
5. **Cost** — API pricing is per-token. Unnecessary tokens waste money.

**RAG solves all five** by retrieving *only* the relevant slices of external data and injecting them into a compact, focused prompt.

---

## 2. RAG at 30,000 Feet — The Two-Phase Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG SYSTEM                              │
│                                                                 │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │   PHASE 1: INDEXING      │  │  PHASE 2: RETRIEVAL & GEN   │ │
│  │       (Offline)          │  │       (Online / Runtime)     │ │
│  │                          │  │                              │ │
│  │  Documents               │  │  User Query                  │ │
│  │     │                    │  │     │                        │ │
│  │     ▼                    │  │     ▼                        │ │
│  │  Load & Preprocess       │  │  Embed Query                 │ │
│  │     │                    │  │     │                        │ │
│  │     ▼                    │  │     ▼                        │ │
│  │  Chunk (Split)           │  │  Retrieve Top-K Chunks       │ │
│  │     │                    │  │     │                        │ │
│  │     ▼                    │  │     ▼                        │ │
│  │  Embed Chunks            │  │  Build Augmented Prompt      │ │
│  │     │                    │  │     │                        │ │
│  │     ▼                    │  │     ▼                        │ │
│  │  Store in Vector DB ─────┼──┼→ LLM Generates Response     │ │
│  │                          │  │                              │ │
│  └──────────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Intuition**: Phase 1 is like building an indexed library. Phase 2 is a librarian (retriever) fetching the right books for a reader (LLM) who writes an essay (response).

---

## 3. Phase 1 — Indexing Pipeline (Offline)

### 3.1 Document Loading & Preprocessing

Raw sources come in many formats: PDF, DOCX, HTML, TXT, databases, APIs. Before anything useful can happen, these must be converted to clean plain text.

**What happens here:**
- Format conversion (PDF → text via OCR or parsers)
- Metadata extraction (source filename, page number, creation date)
- Noise removal (headers, footers, boilerplate)

**Tech that provides this:**

| Tool | What It Does |
|---|---|
| `LangChain DocumentLoaders` | Unified API for loading PDF, TXT, HTML, CSV, Notion, Google Drive, etc. |
| `Unstructured.io` | Intelligent document parsing with layout awareness |
| `PyPDF2` / `pdfplumber` | Direct PDF text extraction |

```python
from langchain.document_loaders import TextLoader
loader = TextLoader("companyPolicies.txt")
documents = loader.load()
```

---

### 3.2 Chunking — Splitting Documents

**Why chunk?** Two reasons:

1. **Embedding model limit**: Embedding models have a fixed maximum input length (e.g., 512 tokens for many BERT-based models). A 10-page document won't fit.
2. **Retrieval precision**: A single vector representing an entire 100-page document would be a vague "average" of everything in it. A short user query like "What is the mobile policy?" would be similarly distant from this diluted vector and from a vector about cooking recipes. But a vector of just the mobile policy paragraph? That will be very close to the query.

**The core tradeoff:**

| Small Chunks (100–300 tokens) | Large Chunks (500–1500 tokens) |
|---|---|
| More precise retrieval | More context per chunk |
| May lose surrounding context | May introduce noise |
| More chunks to store & search | Fewer chunks, faster search |

**Basic Chunking Algorithm (simplified pseudocode):**

```
Input:  Document D of length L characters
Params: chunk_size = s, chunk_overlap = o, separator = sep

1. Split D on `sep` boundaries (e.g., "\n\n" for paragraphs)
2. Greedily merge adjacent splits until merged text reaches `s` characters
3. Finalize the chunk
4. Start the next chunk `o` characters BACK from where the last one ended
   (this is the "overlap" — it ensures continuity)
5. Repeat until all text is consumed
```

**Visual example of overlap:**

```
Document text: "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH"

chunk_size=16, chunk_overlap=4:

Chunk 1: [AAAA BBBB CCCC ]
Chunk 2:           [CCCC DDDD EEEE ]     ← "CCCC" appears in BOTH chunks
Chunk 3:                     [EEEE FFFF GGGG ]
Chunk 4:                               [GGGG HHHH]

WHY overlap? Without it, a sentence split across a boundary would be
incomplete in both chunks. Overlap guarantees at least one chunk has
the full sentence.
```

---

### 3.3 All Chunking Strategies (Definitions & Examples)

There is no single best chunking method. Each strategy suits different document types and retrieval needs.

---

#### Strategy 1: Fixed-Size Character Splitting

**Definition**: Split the text every N characters, regardless of content. Optionally overlap by M characters.

**How it works**: Count characters. When you reach N, cut. Go back M characters and start the next chunk.

**Pros**: Simple, predictable chunk sizes.
**Cons**: Can cut mid-word, mid-sentence, or mid-paragraph. Ignores document structure.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",         # split on every character boundary
    chunk_size=500,       # 500 characters per chunk
    chunk_overlap=50      # 50 characters overlap
)
```

**Example:**

```
Original: "The mobile policy requires all employees to safeguard their devices..."
                                                          ^ cut at char 500
Chunk 1: "The mobile policy requires all employees to safeg"
Chunk 2: "safeguard their devices..."
          ↑ overlap brings back "safeg..." as "safeguard"
```

**Best for**: Quick prototyping, uniform-length requirements.

---

#### Strategy 2: Separator-Based Splitting

**Definition**: Split only at specific separator characters (like paragraph breaks `\n\n` or newlines `\n`), then merge adjacent pieces until reaching the chunk size limit.

**How it works**: First split at every `\n\n`. Then glue small pieces back together until the combined text is close to `chunk_size`.

**Pros**: Respects paragraph boundaries. No mid-paragraph cuts.
**Cons**: If a paragraph is longer than `chunk_size`, it won't be split further.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",     # split on double newlines (paragraph breaks)
    chunk_size=1000,
    chunk_overlap=200
)
```

**Best for**: Documents with clear paragraph structure (policies, articles, books).

---

#### Strategy 3: Recursive Character Splitting (RECOMMENDED)

**Definition**: Try splitting on the most meaningful separator first. If chunks are still too large, fall back to less meaningful separators. This is a **hierarchy of separators**.

**How it works**: The splitter tries separators in this order:
1. `"\n\n"` (paragraph breaks) — best boundaries
2. `"\n"` (newlines) — line-level boundaries
3. `" "` (spaces) — word-level boundaries
4. `""` (every character) — last resort

At each level, it checks: "Are my chunks now small enough?" If yes, stop. If no, try the next separator.

**Pros**: Produces the most semantically coherent chunks. Adapts to document structure.
**Cons**: Slightly more complex.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)
```

**Example walkthrough:**

```
Input document (2500 chars):
  Paragraph 1 (800 chars) \n\n
  Paragraph 2 (600 chars) \n\n
  Paragraph 3 (1100 chars) \n\n    ← too big for chunk_size=1000

Step 1: Split on "\n\n" → [P1(800), P2(600), P3(1100)]
Step 2: P1 and P2 are ≤ 1000 → keep as chunks
Step 3: P3 is > 1000 → try next separator "\n" on P3
Step 4: P3 splits into [P3a(500), P3b(600)] → both ≤ 1000 → done

Result: 3 chunks (possibly 4 with overlap adjustments)
```

**Best for**: Almost everything. This is the **default recommendation** for RAG.

---

#### Strategy 4: Token-Based Splitting

**Definition**: Split based on **token count** (not character count), using the actual tokenizer of your embedding model.

**Why this matters**: Embedding models have limits in **tokens**, not characters. The word "unhappiness" is 1 word, ~11 characters, but might be 3 tokens: `["un", "happiness", "##ness"]`. A 500-character chunk might be 80 tokens or 150 tokens depending on the words used. Token-based splitting guarantees you never exceed the model's limit.

```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=256,       # 256 TOKENS (not characters)
    chunk_overlap=30      # 30 tokens overlap
)
```

**Best for**: When you need precise control over model input size. Production systems.

---

#### Strategy 5: Sentence-Level Splitting

**Definition**: Split the document into individual sentences (or groups of N sentences), keeping each sentence intact.

**How it works**: Use NLP sentence detection (periods, exclamation marks, question marks, plus rules for abbreviations like "Dr." and "U.S."). Then group N sentences per chunk.

```python
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=256,
    chunk_overlap=30
)
```

**Best for**: FAQ-style documents, short-form content, when each sentence carries independent meaning.

---

#### Strategy 6: Semantic Chunking

**Definition**: Split the document at points where the **meaning changes significantly**. Uses embeddings to detect topic boundaries.

**How it works**:
1. Embed each sentence individually
2. Compare consecutive sentence embeddings (cosine similarity)
3. When similarity drops below a threshold → that's a topic boundary → split here

```
Sentence 1: "The mobile policy covers device security." ──┐
Sentence 2: "Employees must not share passwords."        ──┤ similarity = 0.85 (high → same topic)
Sentence 3: "The smoking area is near the east exit."    ──┘ similarity = 0.21 (low → SPLIT HERE)
```

**Pros**: The most semantically coherent chunks possible.
**Cons**: Expensive (must embed every sentence), unpredictable chunk sizes.

**Best for**: Mixed-topic documents, research papers with abrupt topic shifts.

---

#### Strategy 7: Document-Structure-Aware Splitting

**Definition**: Use the document's own structure (headings, sections, pages, HTML tags, Markdown headers) as split points.

**Examples**:
- **Markdown**: Split on `## Heading` boundaries
- **HTML**: Split on `<h2>`, `<section>`, `<article>` tags
- **PDF**: Split on page boundaries
- **Code**: Split on function/class boundaries

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("##", "Section"),
        ("###", "Subsection"),
    ]
)
```

**Best for**: Structured documents (technical docs, legal contracts, code files).

---

#### Chunking Strategies Summary Table

| Strategy | Splits On | Preserves | Chunk Size Control | Best For |
|---|---|---|---|---|
| **Fixed-Size Character** | Every N characters | Nothing | Exact | Quick prototyping |
| **Separator-Based** | Specific delimiters (`\n\n`) | Paragraphs | Approximate | Structured text |
| **Recursive Character** | Hierarchy of separators | Best possible boundaries | Approximate | **General purpose (default)** |
| **Token-Based** | Token count | Token boundaries | Exact (tokens) | Production systems |
| **Sentence-Level** | Sentence boundaries | Sentences | Variable | FAQ, short-form |
| **Semantic** | Meaning changes | Topic coherence | Variable | Mixed-topic docs |
| **Structure-Aware** | Document structure | Sections/headings | Variable | Technical/legal docs |

---

### 3.4 What Gets Tokenized, When, and Why

This is a common point of confusion. Let's be crystal clear.

#### The Pipeline Has Two Separate Tokenization Moments

```
                     MOMENT 1                           MOMENT 2
                   (Embedding)                        (Generation)
                       │                                  │
  ┌────────────────────┼──────────────────────────────────┼──────────┐
  │                    │                                  │          │
  │  Chunk Text ──► TOKENIZER A ──► Embedding Model      │          │
  │  "The mobile       (of the embedding model)          │          │
  │   policy..."       e.g., MPNet's WordPiece            │          │
  │                    tokenizer                          │          │
  │                                                       │          │
  │  Augmented    ──────────────────────────────► TOKENIZER B        │
  │  Prompt                                        (of the LLM)     │
  │  "Context:                                     e.g., GPT's BPE  │
  │   [chunk text]                                 tokenizer        │
  │   Question: ..."                                                │
  └─────────────────────────────────────────────────────────────────┘
```

#### Moment 1: Tokenization INSIDE the Embedding Model

**What gets tokenized?** The text of **each individual chunk** — its words, punctuation, everything.

**When?** During the embedding step (Phase 1, after chunking).

**Why?** The embedding model is a neural network. Neural networks only understand numbers. So we must convert the chunk's text into numbers before the model can process it.

**Step-by-step example:**

```
Chunk text: "Employees must safeguard mobile devices."

Step 1 — Tokenizer splits text into subword tokens:
  ["employees", "must", "safe", "##guard", "mobile", "devices", "."]
  
  Notice: "safeguard" was split into "safe" + "##guard" because
  the tokenizer's vocabulary doesn't have "safeguard" as one piece,
  but it DOES have "safe" and "guard" separately.
  
  The "##" prefix means "this is a continuation of the previous word,
  not a new word."

Step 2 — Each token is mapped to a number (token ID) from the vocabulary:
  ["employees", "must", "safe", "##guard", "mobile", "devices", "."]
  →  [6198,       2442,   3647,   18498,     4684,    4068,     1012]
  
  These numbers are just dictionary lookups. Token "employees" is the
  6198th word in the model's vocabulary. There's no mathematical
  meaning to the number 6198 itself.

Step 3 — These token IDs go into the embedding model's neural network
  → Out comes one vector per token (7 vectors, each 768-dimensional)
  → Mean pooling combines them into 1 vector (768-dimensional)
  → This single vector represents the ENTIRE chunk
```

**The same process happens for the user's query** during retrieval (Phase 2):

```
Query: "What is the mobile phone policy?"
  → tokenize → embed → 1 vector (768-dimensional)
  → compare this vector to all chunk vectors → find top-K most similar
```

#### Moment 2: Tokenization INSIDE the Generator LLM

**What gets tokenized?** The entire **augmented prompt** — instructions + retrieved chunk texts + user question.

**When?** At generation time (Phase 2, after retrieval).

**Why?** The LLM also only understands numbers. But it uses a **different tokenizer** than the embedding model (e.g., BPE instead of WordPiece), and it uses the tokens for a different purpose: predicting the next token in the response.

**Key point**: The embedding model's tokenizer and the LLM's tokenizer are completely separate. They have different vocabularies, different algorithms, and different token IDs. They never interact.

#### Summary: What Gets Tokenized

| What | Tokenized By | When | Purpose |
|---|---|---|---|
| Each chunk's text | Embedding model's tokenizer | Indexing (offline) | Convert chunk text → numbers → neural network → chunk vector |
| User's query text | Embedding model's tokenizer | Retrieval (online) | Convert query text → numbers → neural network → query vector |
| Augmented prompt (instruction + retrieved chunks + query) | LLM's tokenizer | Generation (online) | Convert prompt text → numbers → LLM → generated response |

---

### 3.5 Embedding — From Text to Vectors (Step-by-Step)

An **embedding model** converts a piece of text into a fixed-length list of numbers (a vector) that captures its meaning. Let's walk through every step with real numbers.

---

#### Step 1: Tokenization (covered in 3.4 above)

Input chunk: `"Mobile policy prohibits sharing"`

```
Tokens:    ["Mobile", "policy", "prohibit", "##s", "sharing"]
Token IDs: [4521,     3621,     18923,      1055,  6214]
```

---

#### Step 2: Token Embedding Lookup

The model has a giant lookup table (called an **embedding matrix**) that was learned during training. Think of it as a dictionary:

```
Token ID 4521 ("Mobile")    → look up row 4521    → [0.12, -0.34, 0.56, ..., 0.08]  (768 numbers)
Token ID 3621 ("policy")    → look up row 3621    → [0.45, 0.23, -0.11, ..., 0.67]  (768 numbers)
Token ID 18923 ("prohibit") → look up row 18923   → [-0.22, 0.78, 0.33, ..., -0.15] (768 numbers)
... and so on for each token
```

**The math (it's just a table lookup):**

The embedding matrix $E$ has dimensions $V \times d$ where:
- $V$ = vocabulary size (e.g., 30,522 for BERT)
- $d$ = embedding dimension (e.g., 768)

For a token with ID $w_i$, the initial embedding is just the $w_i$-th row of $E$:

$$e_i = E[w_i]$$

**In plain English**: Go to row number $w_i$ in the table, read off 768 numbers. That's your token's initial vector.

**Worked example** (using tiny 3-dimensional vectors for clarity):

```
Vocabulary size V = 5, Embedding dimension d = 3

Embedding matrix E (learned during training):
  Row 0: [0.1, 0.2, 0.3]    ← token ID 0
  Row 1: [0.4, 0.5, 0.6]    ← token ID 1 ("cat")
  Row 2: [0.7, 0.8, 0.9]    ← token ID 2 ("sat")
  Row 3: [0.2, 0.4, 0.6]    ← token ID 3
  Row 4: [0.5, 0.1, 0.3]    ← token ID 4

If token "cat" has ID=1:  e_cat = E[1] = [0.4, 0.5, 0.6]
If token "sat" has ID=2:  e_sat = E[2] = [0.7, 0.8, 0.9]
```

These initial vectors are **context-free** — "bank" gets the same vector whether it means "river bank" or "savings bank". The Transformer layers that come next will fix this.

---

#### Visualizing the Token-to-Vector Journey

Let's trace a single chunk through the **entire** pipeline, watching the shape of the data transform at every stage. We'll use a realistic example with `all-mpnet-base-v2` (768 dimensions), but show simplified 8D snippets so you can see the patterns.

---

**STAGE 0 — Raw Text (just a string of characters)**

```
"Employees must safeguard mobile devices."
 ↑
 This is just characters in memory: E, m, p, l, o, y, e, e, s, ...
 A computer sees: 69, 109, 112, 108, 111, 121, 101, 101, 115, ...  (ASCII codes)
 No meaning. No structure. Just bytes.
```

---

**STAGE 1 — Tokenization (string → list of subwords)**

```
         "Employees must safeguard mobile devices."
                          │
                    ┌─────┴─────┐
                    │ TOKENIZER │  (WordPiece for MPNet)
                    └─────┬─────┘
                          │
                          ▼
   ┌────────────┬──────┬──────┬─────────┬────────┬─────────┬─────┐
   │ "employees"│"must"│"safe"│"##guard" │"mobile"│"devices" │ "." │
   └────────────┴──────┴──────┴─────────┴────────┴─────────┴─────┘
         7 tokens (note: "safeguard" became 2 tokens)

   Each token is then looked up in the vocabulary dictionary:
   ┌──────┬──────┬──────┬───────┬──────┬──────┬──────┐
   │ 6198 │ 2442 │ 3647 │ 18498 │ 4684 │ 4068 │ 1012 │   ← Token IDs
   └──────┴──────┴──────┴───────┴──────┴──────┴──────┘
         Just integer labels. No meaning yet.
```

---

**STAGE 2 — Embedding Lookup (integer IDs → initial vectors)**

Each token ID indexes into the embedding matrix $E$ (30,522 rows × 768 columns).
Think of each row as a "starting personality" for that word, learned during training.

```
Embedding Matrix E (30,522 × 768):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Row     0: [ 0.021, -0.013,  0.045,  0.008, ..., -0.031, 0.019] │
  │ Row     1: [-0.005,  0.033, -0.012,  0.041, ...,  0.027, 0.009] │
  │ Row     2: [ 0.018,  0.029, -0.037, -0.006, ...,  0.014, 0.022] │
  │   ...         ...     ...     ...     ...           ...     ...   │
  │ Row  2442: [ 0.041, -0.028,  0.033,  0.019, ..., -0.045, 0.011] │ ← "must"
  │   ...                                                             │
  │ Row  4684: [ 0.067,  0.023, -0.041,  0.055, ...,  0.031,-0.018] │ ← "mobile"
  │   ...                                                             │
  │ Row  6198: [ 0.034, -0.019,  0.052, -0.027, ...,  0.038, 0.015] │ ← "employees"
  │   ...                                                             │
  │ Row 30521: [-0.007,  0.025,  0.018, -0.033, ...,  0.042,-0.021] │
  └─────────────────────────────────────────────────────────────────┘

  Result: 7 tokens → 7 vectors, each 768 numbers long:

  Token 0 "employees" → [ 0.034, -0.019,  0.052, -0.027, ...,  0.038,  0.015]
  Token 1 "must"      → [ 0.041, -0.028,  0.033,  0.019, ..., -0.045,  0.011]
  Token 2 "safe"      → [ 0.029,  0.064, -0.017,  0.043, ...,  0.021, -0.033]
  Token 3 "##guard"   → [-0.015,  0.038,  0.027, -0.008, ...,  0.056,  0.044]
  Token 4 "mobile"    → [ 0.067,  0.023, -0.041,  0.055, ...,  0.031, -0.018]
  Token 5 "devices"   → [ 0.051, -0.034,  0.029,  0.062, ..., -0.023,  0.037]
  Token 6 "."         → [-0.003,  0.011,  0.005, -0.002, ...,  0.008, -0.001]
```

**What does a 768-dimensional vector "look like"?**

We can't draw 768 dimensions, but we can visualize the numbers as a **heat strip** — each cell is one dimension, colored by its value:

```
  "employees" (768 dims shown as a heat strip):
  dim:   1    2    3    4    5    6    7    8   ...  766  767  768
       ┌────┬────┬────┬────┬────┬────┬────┬────┬───┬────┬────┬────┐
       │+.03│-.02│+.05│-.03│+.01│+.04│-.01│+.06│...│+.02│+.04│+.02│
       └────┴────┴────┴────┴────┴────┴────┴────┴───┴────┴────┴────┘
        ▓▓   ░░   ▓▓▓  ░░░  ▓    ▓▓   ░    ▓▓▓     ▓▓   ▓▓   ▓▓
        (▓ = positive values, ░ = negative values, size = magnitude)

  "mobile" (768 dims):
       ┌────┬────┬────┬────┬────┬────┬────┬────┬───┬────┬────┬────┐
       │+.07│+.02│-.04│+.06│+.03│-.02│+.05│+.01│...│-.01│+.03│-.02│
       └────┴────┴────┴────┴────┴────┴────┴────┴───┴────┴────┴────┘
        ▓▓▓  ▓▓   ░░░  ▓▓▓  ▓▓   ░░   ▓▓▓  ▓        ░   ▓▓   ░░

  Key insight: Each of the 768 positions captures a different "feature"
  of the word. No single dimension means "this word is about phones."
  Instead, the PATTERN across all 768 dimensions collectively encodes
  the meaning. Similar words have similar patterns.
```

---

**STAGE 3 — Add Positional Encoding (so the model knows word order)**

Each position (0, 1, 2, ...) gets a unique wave pattern added to its vector:

```
  Position 0 ("employees"):  PE = [ 0.000,  1.000,  0.000,  1.000, ...]
  Position 1 ("must"):       PE = [ 0.841,  0.540,  0.010,  0.9999, ...]
  Position 2 ("safe"):       PE = [ 0.909, -0.416,  0.020,  0.9998, ...]
  Position 3 ("##guard"):    PE = [ 0.141, -0.990,  0.030,  0.9996, ...]
  ...

  After adding (element-wise):
  x_employees = e_employees + PE(0) = [0.034+0.000, -0.019+1.000, ...] = [0.034, 0.981, ...]
  x_must      = e_must      + PE(1) = [0.041+0.841, -0.028+0.540, ...] = [0.882, 0.512, ...]

  Now every token "knows" its position. The same word at position 0
  vs position 5 will have different vectors.
```

---

**STAGE 4 — Transformer Layers (context-free → contextualized)**

This is where the real magic happens. The model has 12 layers (for MPNet-base). Each layer runs self-attention so every token can "look at" every other token. After 12 layers, each token's vector has been deeply influenced by its neighbours.

```
  BEFORE Transformer (context-free):
  ┌─────────────────────────────────────────────────────────────┐
  │ "employees" → [0.034, 0.981, 0.052, 0.973, ...]            │
  │ "must"      → [0.882, 0.512, 0.043, 1.019, ...]            │
  │ "safe"      → [0.938, 0.648, 0.003, 1.043, ...]            │
  │ "##guard"   → [0.126, 0.048, 0.057, 0.992, ...]            │
  │ "mobile"    → [0.908, 0.563, 0.021, 1.055, ...]            │
  │ "devices"   → [0.892, 0.506, 0.049, 1.062, ...]            │
  │ "."         → [0.838, 0.551, 0.025, 0.998, ...]            │
  └─────────────────────────────────────────────────────────────┘
              │
              │   12 Transformer layers, each with self-attention
              │   
              │   Layer 1:  each token attends to every other token
              │             "safe" looks at "##guard" → learns it's "safeguard"
              │             "mobile" looks at "devices" → learns it's about tech
              │   
              │   Layer 2:  builds on Layer 1's richer representations
              │             "employees" now sees "safeguard mobile devices" as a phrase
              │   
              │   ...
              │   
              │   Layer 12: deeply contextual — every token now "knows"
              │             the full meaning of the entire sentence
              ▼
  AFTER Transformer (contextualized):
  ┌─────────────────────────────────────────────────────────────┐
  │ "employees" → [-0.312,  0.587,  0.821, -0.145, ...]        │
  │ "must"      → [ 0.234, -0.456,  0.678,  0.123, ...]        │
  │ "safe"      → [ 0.445,  0.312, -0.234,  0.567, ...]        │  Dramatically
  │ "##guard"   → [ 0.451,  0.298, -0.241,  0.573, ...]        │  different from
  │ "mobile"    → [ 0.623, -0.178,  0.534,  0.289, ...]        │  the input!
  │ "devices"   → [ 0.598, -0.201,  0.512,  0.267, ...]        │
  │ "."         → [ 0.089,  0.045, -0.023,  0.034, ...]        │
  └─────────────────────────────────────────────────────────────┘

  Notice: "safe" and "##guard" now have VERY similar vectors (0.445 vs 0.451,
  0.312 vs 0.298, etc.) — the Transformer merged them into a unified
  "safeguard" concept. They started with completely different vectors!

  Notice: "mobile" and "devices" also converged — the Transformer learned
  they form the phrase "mobile devices."
```

**Visualizing how context changes the same word:**

```
  The word "bank" in TWO different chunks:

  Chunk A: "The bank approved the loan"
  Chunk B: "The bank of the river flooded"

  BEFORE Transformer (same word = same initial vector):
    "bank" in A → [0.23, -0.15, 0.44, 0.67, ...]  ← IDENTICAL
    "bank" in B → [0.23, -0.15, 0.44, 0.67, ...]  ← IDENTICAL

  AFTER Transformer (context makes them different):
    "bank" in A → [0.71, -0.33, 0.82, 0.12, ...]  ← "financial institution"
    "bank" in B → [-0.15, 0.64, 0.21, 0.55, ...]  ← "river edge"
                    ↑ completely different numbers!

  The self-attention mechanism let "bank" look at "loan" and "approved"
  in chunk A, pulling it toward financial meanings. In chunk B, "bank"
  looked at "river" and "flooded", pulling it toward geographical meanings.
```

---

**STAGE 5 — Mean Pooling (7 token vectors → 1 chunk vector)**

```
  7 contextualized token vectors (each 768D):

  h_0 "employees" → [-0.312,  0.587,  0.821, -0.145, ..., 0.234]  ─┐
  h_1 "must"      → [ 0.234, -0.456,  0.678,  0.123, ..., 0.156]   │
  h_2 "safe"      → [ 0.445,  0.312, -0.234,  0.567, ..., 0.089]   │
  h_3 "##guard"   → [ 0.451,  0.298, -0.241,  0.573, ..., 0.092]   ├── Average
  h_4 "mobile"    → [ 0.623, -0.178,  0.534,  0.289, ..., 0.345]   │   all rows
  h_5 "devices"   → [ 0.598, -0.201,  0.512,  0.267, ..., 0.312]   │   column
  h_6 "."         → [ 0.089,  0.045, -0.023,  0.034, ..., 0.011]  ─┘   by column
                     ──────  ──────  ──────  ──────       ──────
                       │        │       │       │            │
                       ▼        ▼       ▼       ▼            ▼
  dim 1 avg:  (-0.312 + 0.234 + 0.445 + 0.451 + 0.623 + 0.598 + 0.089) / 7
            = 2.128 / 7 = 0.304

  dim 2 avg:  (0.587 + (-0.456) + 0.312 + 0.298 + (-0.178) + (-0.201) + 0.045) / 7
            = 0.407 / 7 = 0.058

  ... (repeat for all 768 dimensions) ...

  FINAL CHUNK VECTOR (768D):
  ┌─────────────────────────────────────────────────────────────────┐
  │ v_chunk = [0.304, 0.058, 0.292, 0.244, ..., 0.177]             │
  │                                                                 │
  │ This single vector of 768 numbers IS the chunk's embedding.     │
  │ It captures the meaning of "Employees must safeguard mobile     │
  │ devices." in a form that can be compared to other vectors.      │
  └─────────────────────────────────────────────────────────────────┘
```

---

**STAGE 6 — Storage (vector goes into the database)**

```
  Vector Store (e.g., ChromaDB):
  ┌────────────┬──────────────────────────────────────────────────┐
  │  Chunk ID  │  Vector (768 dimensions)                         │
  ├────────────┼──────────────────────────────────────────────────┤
  │  chunk_0   │ [0.304, 0.058, 0.292, 0.244, ..., 0.177]       │ ← our chunk
  │  chunk_1   │ [0.156, 0.423, -0.087, 0.331, ..., 0.265]      │
  │  chunk_2   │ [-0.211, 0.178, 0.543, -0.089, ..., 0.098]     │
  │  chunk_3   │ [0.089, -0.334, 0.267, 0.512, ..., -0.145]     │
  │  ...       │ ...                                              │
  │  chunk_999 │ [0.178, 0.067, -0.312, 0.445, ..., 0.234]      │
  └────────────┴──────────────────────────────────────────────────┘

  Each row is a point in 768-dimensional space.
  Nearby points = similar meaning. Distant points = different meaning.
```

---

**THE COMPLETE TRANSFORMATION AT A GLANCE:**

```
  "Employees must safeguard mobile devices."

       │ Stage 0: Raw characters
       │ Shape: a string (42 characters)
       ▼
  ["employees", "must", "safe", "##guard", "mobile", "devices", "."]

       │ Stage 1: Tokenized subwords
       │ Shape: 7 strings
       ▼
  [6198, 2442, 3647, 18498, 4684, 4068, 1012]

       │ Stage 2: Token IDs (vocabulary lookup)
       │ Shape: 7 integers
       ▼
  [[0.034, -0.019, ...(768)...],    ← "employees"
   [0.041, -0.028, ...(768)...],    ← "must"
   [0.029,  0.064, ...(768)...],    ← "safe"
   [-0.015, 0.038, ...(768)...],    ← "##guard"
   [0.067,  0.023, ...(768)...],    ← "mobile"
   [0.051, -0.034, ...(768)...],    ← "devices"
   [-0.003, 0.011, ...(768)...]]    ← "."

       │ Stage 3: Embedding lookup (table → initial vectors)
       │ Shape: 7 × 768 matrix
       ▼
  (same shape, but with positional encoding added)

       │ Stage 4: + Positional Encoding
       │ Shape: 7 × 768 matrix (values shifted)
       ▼
  [[-0.312,  0.587, ...(768)...],   ← "employees" (now context-aware)
   [ 0.234, -0.456, ...(768)...],   ← "must"
   [ 0.445,  0.312, ...(768)...],   ← "safe"
   [ 0.451,  0.298, ...(768)...],   ← "##guard"
   [ 0.623, -0.178, ...(768)...],   ← "mobile"
   [ 0.598, -0.201, ...(768)...],   ← "devices"
   [ 0.089,  0.045, ...(768)...]]   ← "."

       │ Stage 5: Transformer (12 layers of self-attention)
       │ Shape: 7 × 768 matrix (values dramatically changed)
       ▼
  [0.304, 0.058, 0.292, 0.244, ...(768)..., 0.177]

       │ Stage 6: Mean pooling (average all 7 rows)
       │ Shape: 1 × 768 vector  ← THIS IS THE CHUNK EMBEDDING
       ▼
  Stored in ChromaDB / FAISS / Pinecone
  Ready for similarity search!
```

---

**Size intuition at each stage:**

| Stage | Shape | Total Numbers | Human Analogy |
|---|---|---|---|
| Raw text | 1 string | 42 characters | A sentence on paper |
| Tokens | 7 strings | 7 subwords | Sentence broken into word fragments |
| Token IDs | 7 integers | 7 numbers | Each fragment gets a dictionary page number |
| Initial embeddings | 7 × 768 | 5,376 numbers | Each fragment gets a 768-trait personality profile |
| + Positional encoding | 7 × 768 | 5,376 numbers | Each profile now includes "where I sit" info |
| After Transformer | 7 × 768 | 5,376 numbers | Each profile now deeply understands its context |
| After mean pooling | 1 × 768 | 768 numbers | One unified profile for the whole sentence |

---

#### Step 3: Positional Encoding — Adding Word Order

The Transformer processes all tokens **in parallel** (not one-by-one like an RNN). So it has no idea that "Mobile" comes before "policy". We fix this by adding a **position signal** to each token's embedding.

**The intuition**: Imagine you have 5 numbered seats in a theater. Each seat has a unique "vibe" — seat 1 is near the door, seat 5 is by the window. We add each seat's "vibe" to the person sitting there, so the Transformer knows who is sitting where.

**The math:**

For each position $pos$ (0, 1, 2, ...) and each dimension $k$ (0, 1, 2, ..., d/2):

$$PE_{(pos, 2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right)$$

$$PE_{(pos, 2k+1)} = \cos\left(\frac{pos}{10000^{2k/d}}\right)$$

**What do these formulas actually do? Let's compute one.**

Say we have $d = 4$ dimensions and want the positional encoding for position $pos = 2$ (the third token):

```
Dimension 0 (k=0, even):  sin(2 / 10000^(0/4)) = sin(2 / 1) = sin(2) = 0.909
Dimension 1 (k=0, odd):   cos(2 / 10000^(0/4)) = cos(2 / 1) = cos(2) = -0.416
Dimension 2 (k=1, even):  sin(2 / 10000^(2/4)) = sin(2 / 100) = sin(0.02) = 0.020
Dimension 3 (k=1, odd):   cos(2 / 10000^(2/4)) = cos(2 / 100) = cos(0.02) = 0.9998

So: PE(pos=2) = [0.909, -0.416, 0.020, 0.9998]
```

**Why sine and cosine?** Each position gets a unique combination of waves at different frequencies. Low dimensions change fast (high frequency — they can distinguish nearby positions), high dimensions change slowly (low frequency — they capture long-range position differences). This gives every position a unique "fingerprint."

**The final token representation before the Transformer layers:**

$$x_i = e_i + PE(i)$$

Just add the position vector to the embedding vector, element by element.

```
Token "policy" at position 1:
  e_policy = [0.45, 0.23, -0.11, 0.67]  (from lookup table)
  PE(1)    = [0.841, 0.540, 0.010, 0.9999]  (from sine/cosine formula)
  x_policy = [0.45+0.841, 0.23+0.540, -0.11+0.010, 0.67+0.9999]
           = [1.291, 0.770, -0.100, 1.670]
```

---

#### Step 4: Self-Attention — The Core of the Transformer

This is where the magic happens. Self-attention allows each token to "look at" every other token in the chunk and decide which ones are relevant to it.

**The everyday analogy**: You're at a party. When someone says "The bank near the river flooded," the word "bank" looks around (attends) to the other words. It sees "river" and "flooded" and thinks: "Ah, I'm the river kind of bank, not the money kind." Self-attention is this "looking around" process, but in math.

**The three actors: Query (Q), Key (K), Value (V)**

Think of it like a library search:
- **Query (Q)**: "What am I looking for?" — Each token asks a question
- **Key (K)**: "What do I contain?" — Each token advertises its content
- **Value (V)**: "Here is my actual information" — Each token's payload

These are created by multiplying each token's vector by three different learned weight matrices:

$$Q = X \cdot W^Q$$
$$K = X \cdot W^K$$
$$V = X \cdot W^V$$

where $X$ is the matrix of all token vectors stacked together (one row per token).

**The attention formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

Let's break this down piece by piece with a **worked example**.

Say we have 3 tokens and $d_k = 2$ (tiny dimensions for clarity):

```
Q (what each token is looking for):
  Token 0 "Mobile":   [1.0, 0.5]
  Token 1 "policy":   [0.8, 0.3]
  Token 2 "sharing":  [0.2, 0.9]

K (what each token advertises):
  Token 0 "Mobile":   [0.9, 0.4]
  Token 1 "policy":   [0.7, 0.6]
  Token 2 "sharing":  [0.3, 0.8]
```

**Step 4a: Compute Q · K^T (how much each token "matches" every other token)**

Each entry = dot product of one token's Q with another token's K.

```
Q · K^T = 
             Key0    Key1    Key2
  Query0  [ 1.0×0.9+0.5×0.4,  1.0×0.7+0.5×0.6,  1.0×0.3+0.5×0.8 ]
  Query1  [ 0.8×0.9+0.3×0.4,  0.8×0.7+0.3×0.6,  0.8×0.3+0.3×0.8 ]
  Query2  [ 0.2×0.9+0.9×0.4,  0.2×0.7+0.9×0.6,  0.2×0.3+0.9×0.8 ]

        = [ 1.10, 1.00, 0.70 ]     ← "Mobile" matches strongly with "Mobile" and "policy"
          [ 0.84, 0.74, 0.48 ]     ← "policy" also matches "Mobile" most
          [ 0.54, 0.68, 0.78 ]     ← "sharing" matches "sharing" most (makes sense!)
```

**Step 4b: Divide by √d_k (scaling)**

$\sqrt{d_k} = \sqrt{2} = 1.414$

```
Scaled scores = Q·K^T / 1.414:
  [ 0.778, 0.707, 0.495 ]
  [ 0.594, 0.523, 0.339 ]
  [ 0.382, 0.481, 0.552 ]
```

**Why divide by √d_k?** Without scaling, when $d_k$ is large (like 768), the dot products become very large numbers. Large numbers fed into softmax produce outputs very close to 0 or 1 (like a step function). This means the model would "hard-attend" to just one token and ignore all others. Dividing by $\sqrt{d_k}$ keeps the numbers in a moderate range, allowing the model to attend to multiple tokens softly.

**Step 4c: Apply softmax (turn scores into probabilities)**

Softmax converts each row into probabilities that sum to 1:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```
For row 0 [0.778, 0.707, 0.495]:
  e^0.778 = 2.177,  e^0.707 = 2.028,  e^0.495 = 1.641
  sum = 5.846
  
  Attention weights for "Mobile": [2.177/5.846, 2.028/5.846, 1.641/5.846]
                                = [0.372,      0.347,       0.281]
```

**Reading these weights**: "Mobile" pays 37.2% attention to itself, 34.7% to "policy", and 28.1% to "sharing".

**Step 4d: Multiply weights × V (get the final output)**

```
V (each token's value/payload):
  Token 0: [0.5, 0.3]
  Token 1: [0.2, 0.8]
  Token 2: [0.7, 0.4]

Output for "Mobile" = 0.372 × [0.5, 0.3] + 0.347 × [0.2, 0.8] + 0.281 × [0.7, 0.4]
                    = [0.186, 0.112] + [0.069, 0.278] + [0.197, 0.112]
                    = [0.452, 0.502]
```

This output vector for "Mobile" now **encodes information from all tokens it attended to**, weighted by relevance. "Mobile" has absorbed context from "policy" and "sharing" — it now "knows" it's about a mobile policy regarding sharing.

**This is what makes embeddings "contextual"**: The same word "bank" would get a different output vector depending on whether the surrounding words are about rivers or money, because the attention weights would be different.

---

#### Step 5: Mean Pooling — One Vector for the Whole Chunk

After the Transformer's final layer, we have one vector per token. But we need ONE vector for the ENTIRE chunk to store in the vector database. The simplest approach: **average all token vectors**.

$$v_{\text{chunk}} = \frac{1}{n} \sum_{i=1}^{n} h_i$$

where $h_i$ is the final-layer output for token $i$, and $n$ is the number of tokens.

**Worked example:**

```
Final layer outputs for 3 tokens (2D for simplicity):
  Token 0 "Mobile":   h_0 = [0.452, 0.502]
  Token 1 "policy":   h_1 = [0.390, 0.610]
  Token 2 "sharing":  h_2 = [0.580, 0.440]

Mean pooling:
  v_chunk = ( [0.452, 0.502] + [0.390, 0.610] + [0.580, 0.440] ) / 3
          = [1.422/3, 1.552/3]
          = [0.474, 0.517]
```

This single vector $[0.474, 0.517]$ **IS** the chunk's embedding. It gets stored in the vector database. Later, when a user asks "What is the mobile policy?", that query gets embedded the same way, and we compare the two vectors.

---

### 3.6 Vector Store — Storing & Indexing Embeddings

After embedding every chunk, we need a data structure that supports **fast similarity search** over potentially millions of vectors.

#### Why not just a Python list?

Brute-force search compares the query vector against every stored vector. For $N$ chunks with $d$-dimensional vectors, each query requires:

$$\text{Operations} = N \times d$$

For 1 million chunks at 768 dimensions: $1{,}000{,}000 \times 768 = 768{,}000{,}000$ multiplications per query. At 100 queries/second, that's 76.8 billion operations/second. Not practical.

#### Approximate Nearest Neighbour (ANN) Algorithms

Vector databases use ANN algorithms to trade a small amount of accuracy for massive speedups. Details in Section 5.9.

**Tech that provides this:**

| Vector Store | ANN Algorithm | Notes |
|---|---|---|
| **ChromaDB** | HNSW | Simple, great for prototyping |
| **FAISS** (Facebook) | IVF, PQ, HNSW, Flat | Production-grade, GPU support |
| **Milvus** | IVF, HNSW, DiskANN | Distributed, scalable |
| **Pinecone** | Proprietary | Managed cloud service |
| **Weaviate** | HNSW | Hybrid search (vector + keyword) |
| **Qdrant** | HNSW | Filtering + vector search |

```python
from langchain.vectorstores import Chroma

vectorstore = Chroma.from_documents(chunks, embeddings)
```

---

## 4. Phase 2 — Retrieval & Generation Pipeline (Online)

### 4.1 Query Embedding

The user's query is embedded using the **exact same** embedding model used for the document chunks. This is critical — different models produce incompatible vector spaces.

$$q = \text{EmbedModel}(\text{"What is the mobile phone policy?"})$$

The result is a single vector (e.g., 768 numbers) in the same space as all the chunk vectors.

### 4.2 Retrieval — Finding Relevant Chunks

The retriever compares the query vector $q$ against all chunk vectors $\{c_1, c_2, \ldots, c_N\}$ and returns the top-K most similar chunks.

**Retrieval strategies:**

| Strategy | Description |
|---|---|
| **Top-K chunks** | Return the K individual chunks most similar to the query |
| **Parent document** | Return the entire parent document of the best matching chunk |
| **Multi-query** | Generate multiple reformulations of the query, retrieve for each, merge results |
| **Contextual compression** | Retrieve chunks, then use an LLM to extract only the relevant sentences |
| **Hybrid (vector + BM25)** | Combine dense vector similarity with sparse keyword matching |

### 4.3 Augmented Prompt Construction

The retrieved chunks are combined with the original query into a structured prompt:

```
┌─────────────────────────────────────────────┐
│              AUGMENTED PROMPT                │
│                                             │
│  System: You are a helpful assistant.       │
│  Use ONLY the context below to answer.      │
│  If the answer isn't in the context,        │
│  say "I don't know."                        │
│                                             │
│  Context:                                   │
│  ─────────                                  │
│  [Retrieved Chunk 1]                        │
│  [Retrieved Chunk 2]                        │
│  [Retrieved Chunk 3]                        │
│                                             │
│  Question: {user's original query}          │
│                                             │
│  Answer:                                    │
└─────────────────────────────────────────────┘
```

**Why structured templates matter**: Without explicit instructions, the LLM might ignore the context and answer from its parametric memory, or hallucinate details not present in the retrieved text.

### 4.4 LLM Generation

The augmented prompt is passed to the generator LLM, which produces a natural language response grounded in the retrieved context.

**Key generation parameters:**

| Parameter | Effect |
|---|---|
| `temperature` | Controls randomness. 0 = deterministic (greedy), 1 = more creative |
| `max_new_tokens` | Maximum length of the generated response |
| `top_p` (nucleus sampling) | Only sample from tokens whose cumulative probability ≥ p |
| `decoding_method` | `greedy` (always pick highest probability token) vs `sample` |

---

## 5. The Mathematics — Deep Dive (Beginner Friendly)

Every formula below is explained in three ways:
1. **The intuition** (everyday analogy)
2. **The formula** (using standard math notation rendered properly)
3. **A worked example** (plug in real numbers, compute by hand)

---

### 5.1 Vectors — The Language Computers Speak

Before anything else, let's understand what a **vector** is.

**A vector is just a list of numbers.** That's it.

```
A 2D vector:   [3, 4]           → a point on a flat surface
A 3D vector:   [1, 2, 5]       → a point in 3D space
A 768D vector: [0.12, -0.34, 0.56, ..., 0.08]  → a point in 768-dimensional space
```

**Why use vectors for text?** Because computers can't do math on words like "policy" and "regulation", but they CAN do math on numbers. If we can convert "policy" → $[0.45, 0.23, -0.11]$ and "regulation" → $[0.43, 0.25, -0.09]$, then the computer can see that these two vectors are **very close together** (similar numbers), meaning the words have **similar meanings**.

This is the fundamental idea behind all of RAG.

---

### 5.2 Token Embeddings — How Words Become Numbers

**Intuition**: Every word (or subword) gets assigned a list of numbers. Initially this is just a random starting point (learned during training). The key insight is that words used in similar contexts end up with similar numbers.

**The math** (this is just a lookup, no computation):

Given a vocabulary of $V$ words and embedding dimension $d$, the embedding matrix is:

$$E \in \mathbb{R}^{V \times d}$$

This is a table with $V$ rows (one per word in the vocabulary) and $d$ columns. For a token with ID $w_i$:

$$e_i = E[w_i] \in \mathbb{R}^d$$

**What $\mathbb{R}^{V \times d}$ means in plain English**: A table of real numbers with $V$ rows and $d$ columns.

**What $e_i = E[w_i]$ means**: "Go to row $w_i$ of the table $E$ and read the numbers."

**Why this works** (the training story):

During training (the Word2Vec intuition), the model learns these embeddings by reading billions of sentences and adjusting the numbers so that words appearing in similar contexts get similar vectors.

The Skip-gram training objective:

$$\max \sum_{t=1}^{T} \sum_{\substack{j=-c \\ j \neq 0}}^{c} \log P(w_{t+j} \mid w_t)$$

**Breaking this down piece by piece:**
- $T$ = total number of words in the training corpus (billions)
- $c$ = context window size (e.g., 5 — look 5 words left and right)
- $w_t$ = the word at position $t$ (the center word)
- $w_{t+j}$ = a word near position $t$ (a context word)
- $P(w_{t+j} \mid w_t)$ = "the probability of seeing context word $w_{t+j}$ near center word $w_t$"
- $\log$ = we use log for numerical stability
- $\max$ = adjust the embeddings to make this sum as large as possible

**In English**: "For every word in the corpus, look at the words nearby. Adjust the embedding numbers so that words that frequently appear near each other have similar embeddings."

**Worked example** (the result of training):

```
After training on millions of sentences:
  embed("king")  = [0.8, 0.2, 0.5, 0.9]
  embed("queen") = [0.7, 0.3, 0.5, 0.8]  ← similar to king (both royalty)
  embed("car")   = [0.1, 0.9, 0.2, 0.1]  ← very different (not royalty)
  
The famous result: king - man + woman ≈ queen
  [0.8, 0.2, 0.5, 0.9] - [0.6, 0.1, 0.4, 0.7] + [0.5, 0.2, 0.5, 0.6]
  = [0.7, 0.3, 0.6, 0.8]  ← close to queen's vector!
```

---

### 5.3 Positional Encoding — Teaching Order to the Model

(Detailed step-by-step in Section 3.5, Step 3 above.)

**Quick recap formula:**

$$PE_{(pos, 2k)} = \sin\left(\frac{pos}{10000^{2k/d}}\right) \qquad PE_{(pos, 2k+1)} = \cos\left(\frac{pos}{10000^{2k/d}}\right)$$

**What $10000^{2k/d}$ controls**: The "frequency" of the wave. Small $k$ (low dimensions) = high frequency (changes rapidly between positions). Large $k$ (high dimensions) = low frequency (changes slowly). Think of it like a clock: the second hand moves fast (low $k$), the hour hand moves slow (high $k$). Together they uniquely encode any time of day.

---

### 5.4 Self-Attention — How the Model "Reads"

(Full step-by-step worked example in Section 3.5, Step 4 above.)

**The formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

**What each piece means:**

| Symbol | Meaning | Size |
|---|---|---|
| $Q$ | Query matrix — "what is each token looking for?" | $n \times d_k$ ($n$ tokens, each with a $d_k$-dimensional query) |
| $K$ | Key matrix — "what does each token contain?" | $n \times d_k$ |
| $V$ | Value matrix — "what information does each token carry?" | $n \times d_v$ |
| $K^T$ | Key matrix transposed (rows ↔ columns) | $d_k \times n$ |
| $Q \cdot K^T$ | Match scores: how well each query matches each key | $n \times n$ (every token compared to every token) |
| $\sqrt{d_k}$ | Scaling factor to keep numbers moderate | A single number |
| $\text{softmax}$ | Converts scores to probabilities (each row sums to 1) | $n \times n$ |
| Final $\cdot V$ | Weighted combination of values | $n \times d_v$ |

**The softmax function in detail:**

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

**What it does**: Takes any list of numbers and converts them into positive numbers that sum to 1 (i.e., probabilities).

**Worked example:**

```
Input:  [2.0, 1.0, 0.1]
Step 1: e^2.0 = 7.389,  e^1.0 = 2.718,  e^0.1 = 1.105
Step 2: sum = 7.389 + 2.718 + 1.105 = 11.212
Step 3: [7.389/11.212, 2.718/11.212, 1.105/11.212] = [0.659, 0.242, 0.099]

Result: [0.659, 0.242, 0.099]  ← sums to 1.0
The largest input (2.0) gets the largest probability (0.659).
```

---

### 5.5 Mean Pooling — Merging Tokens into One Chunk Vector

(Detailed in Section 3.5, Step 5.)

$$v_{\text{chunk}} = \frac{1}{n} \sum_{i=1}^{n} h_i$$

**In English**: Add up all the token vectors, then divide by the number of tokens. This averages them into a single representative vector.

---

### 5.6 Similarity Metrics — How Retrieval Ranks Chunks

When the retriever has the query vector $q$ and all chunk vectors $\{c_1, c_2, \ldots\}$, it needs to compute "how similar is $q$ to each $c_i$?" Here are the three main ways:

---

#### Cosine Similarity (most common in RAG)

**Intuition**: Imagine two arrows drawn from the center of a room. Cosine similarity measures the **angle** between them. Two arrows pointing the same direction have cosine similarity = 1, regardless of how long the arrows are. Two arrows at 90° have cosine similarity = 0.

**Formula:**

$$\text{cosine\_sim}(q, c) = \frac{q \cdot c}{\|q\| \times \|c\|}$$

**Let's unpack each piece:**

The **dot product** $q \cdot c$ (numerator):

$$q \cdot c = q_1 \times c_1 + q_2 \times c_2 + \ldots + q_d \times c_d = \sum_{i=1}^{d} q_i \times c_i$$

Multiply corresponding elements and add them up.

The **magnitude** $\|q\|$ (denominator part):

$$\|q\| = \sqrt{q_1^2 + q_2^2 + \ldots + q_d^2} = \sqrt{\sum_{i=1}^{d} q_i^2}$$

The "length" of the vector — how far the arrow stretches from the origin.

**Full worked example with 2D vectors:**

```
Query vector (user asked "phone policy"):
  q = [0.8, 0.6]

Chunk vectors:
  c1 = [0.7, 0.7]   (chunk about mobile device security)
  c2 = [-0.5, 0.9]  (chunk about smoking areas)

─── Compare q to c1 ───

Numerator (dot product):
  q · c1 = (0.8 × 0.7) + (0.6 × 0.7)
         =     0.56     +     0.42
         = 0.98

Denominator (magnitudes):
  ‖q‖  = √(0.8² + 0.6²) = √(0.64 + 0.36) = √1.0 = 1.0
  ‖c1‖ = √(0.7² + 0.7²) = √(0.49 + 0.49) = √0.98 = 0.990

Cosine similarity:
  cos_sim(q, c1) = 0.98 / (1.0 × 0.990) = 0.98 / 0.990 = 0.990

  → 0.990 is very close to 1.0 → VERY SIMILAR ✓

─── Compare q to c2 ───

Numerator (dot product):
  q · c2 = (0.8 × -0.5) + (0.6 × 0.9)
         =     -0.40     +     0.54
         = 0.14

Denominator:
  ‖q‖  = 1.0 (same as before)
  ‖c2‖ = √((-0.5)² + 0.9²) = √(0.25 + 0.81) = √1.06 = 1.030

Cosine similarity:
  cos_sim(q, c2) = 0.14 / (1.0 × 1.030) = 0.14 / 1.030 = 0.136

  → 0.136 is close to 0 → NOT SIMILAR ✗

─── Retrieval decision ───
  c1 (0.990) >> c2 (0.136)
  → Retrieve c1 (mobile device security) ✓
  → Don't retrieve c2 (smoking areas) ✓
  The system correctly identifies the mobile policy chunk as relevant!
```

**Range**: -1 (opposite) to 0 (unrelated) to +1 (identical meaning)

---

#### Dot Product (Inner Product)

**Intuition**: Like cosine similarity, but it also cares about the **length** of the vectors. A longer vector = more "important" or "confident."

**Formula:**

$$\text{dot}(q, c) = q \cdot c = \sum_{i=1}^{d} q_i \times c_i$$

Same as the numerator of cosine similarity, but WITHOUT dividing by the magnitudes.

**Worked example:**

```
q = [0.8, 0.6]
c1 = [0.7, 0.7]
c2 = [1.4, 1.4]    ← same DIRECTION as c1, but TWICE as long

dot(q, c1) = 0.56 + 0.42 = 0.98
dot(q, c2) = 1.12 + 0.84 = 1.96    ← c2 scores higher just because it's longer!

cosine(q, c1) = 0.990
cosine(q, c2) = 0.990    ← cosine gives SAME score (same direction)
```

**When to use**: When the embedding model does NOT normalize vectors to unit length, and longer vectors carry meaning (e.g., the model is more "confident" about longer vectors).

**Range**: $(-\infty, +\infty)$ — unbounded.

---

#### Euclidean (L2) Distance

**Intuition**: The straight-line distance between two points. Like measuring with a ruler on a map.

**Formula:**

$$d_{L2}(q, c) = \sqrt{\sum_{i=1}^{d} (q_i - c_i)^2}$$

**Worked example:**

```
q  = [0.8, 0.6]
c1 = [0.7, 0.7]

d_L2(q, c1) = √((0.8 - 0.7)² + (0.6 - 0.7)²)
            = √((0.1)² + (-0.1)²)
            = √(0.01 + 0.01)
            = √0.02
            = 0.141

Very small distance → very similar ✓
```

**Range**: 0 (identical) to $\infty$ (very different). NOTE: lower = MORE similar (opposite of cosine).

---

#### Summary: Which Metric When?

| Metric | Measures | Range | Lower or Higher = Similar? | Default for RAG? |
|---|---|---|---|---|
| **Cosine Similarity** | Angle between vectors | -1 to +1 | Higher = more similar | **Yes (most common)** |
| **Dot Product** | Alignment + magnitude | $-\infty$ to $+\infty$ | Higher = more similar | Sometimes |
| **Euclidean Distance** | Straight-line distance | 0 to $\infty$ | Lower = more similar | Rarely |

---

### 5.7 Contrastive Loss — How Embedding Models are Trained

The embedding model (like `all-mpnet-base-v2`) was trained using **contrastive learning**: it learns by seeing examples of similar pairs and dissimilar pairs.

**Intuition**: Show the model a pair of sentences that mean the same thing, and a pair that don't. Adjust the model's weights until similar sentences have similar vectors and dissimilar sentences have distant vectors.

**The InfoNCE loss formula:**

$$\mathcal{L} = -\log \frac{e^{\text{sim}(v_i, v_i^+) / \tau}}{e^{\text{sim}(v_i, v_i^+) / \tau} + \sum_{j=1}^{N} e^{\text{sim}(v_i, v_j^-) / \tau}}$$

**Let's decode every symbol:**

| Symbol | Meaning |
|---|---|
| $v_i$ | The vector of the anchor sentence (e.g., "mobile policy") |
| $v_i^+$ | The vector of a sentence that means the same thing (e.g., "phone guidelines") |
| $v_j^-$ | The vector of a sentence that means something different (e.g., "smoking area rules") |
| $\text{sim}(a, b)$ | Cosine similarity between vectors $a$ and $b$ |
| $\tau$ | Temperature parameter (controls how "sharp" the distinction is; typically 0.05–0.1) |
| $e^x$ | The exponential function ($e \approx 2.718$ raised to the power $x$) |
| $\log$ | Natural logarithm |

**Worked example (tiny numbers):**

```
Anchor: "mobile policy"        → v_i  = [0.8, 0.6]
Positive: "phone guidelines"   → v_i+ = [0.7, 0.7]   (similar meaning)
Negative: "smoking area rules" → v_1- = [-0.5, 0.9]   (different meaning)
Temperature: τ = 0.1

Step 1: Compute similarities
  sim(v_i, v_i+) = cosine_sim([0.8,0.6], [0.7,0.7]) = 0.990
  sim(v_i, v_1-) = cosine_sim([0.8,0.6], [-0.5,0.9]) = 0.136

Step 2: Divide by temperature
  0.990 / 0.1 = 9.90
  0.136 / 0.1 = 1.36

Step 3: Compute exponentials
  e^9.90 = 19930
  e^1.36 = 3.896

Step 4: Compute the fraction
  19930 / (19930 + 3.896) = 19930 / 19933.9 = 0.9998

Step 5: Compute the loss
  L = -log(0.9998) = 0.0002

Very small loss → the model is doing well! The positive pair is much more
similar than the negative pair, which is exactly what we want.

If the model were confused and gave similar scores to positive and negative:
  sim(v_i, v_i+) = 0.5, sim(v_i, v_1-) = 0.5
  Both → e^(0.5/0.1) = e^5 = 148.4
  Fraction = 148.4 / (148.4 + 148.4) = 0.5
  L = -log(0.5) = 0.693  ← much higher loss → model needs to learn more
```

---

### 5.8 BM25 — The Classic Keyword Algorithm

BM25 is a **keyword-based** ranking algorithm used in hybrid retrieval alongside vector similarity. It's from the TF-IDF family.

**Intuition**: A document is relevant if it contains the query's keywords, especially if those keywords are rare (more distinctive). A document that says "mobile" 5 times is more relevant than one that says it once, but with diminishing returns.

**Formula:**

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

**Symbol glossary:**

| Symbol | Meaning | Typical Value |
|---|---|---|
| $t$ | A term (word) from the query | — |
| $q$ | The user's query (a set of terms) | — |
| $d$ | A document (chunk) being scored | — |
| $f(t, d)$ | How many times term $t$ appears in document $d$ (term frequency) | — |
| $\|d\|$ | Length of document $d$ (in words) | — |
| $\text{avgdl}$ | Average document length across all documents | — |
| $k_1$ | Controls term frequency saturation | 1.2 – 2.0 |
| $b$ | Controls length normalization | 0.75 |
| $\text{IDF}(t)$ | Inverse document frequency — how rare/distinctive the term is | — |

**IDF formula:**

$$\text{IDF}(t) = \log \frac{N - n(t) + 0.5}{n(t) + 0.5}$$

where $N$ = total documents, $n(t)$ = documents containing term $t$.

**Worked example:**

```
Query: "mobile policy"
We have 100 document chunks total. Average length = 50 words.

Document d1: "The mobile policy covers mobile device security" (7 words)
  f("mobile", d1) = 2  (appears twice)
  f("policy", d1) = 1  (appears once)

Document d2: "Smoking is prohibited in all areas" (6 words)
  f("mobile", d2) = 0
  f("policy", d2) = 0

IDF calculations:
  n("mobile") = 8  (8 out of 100 docs contain "mobile")
  IDF("mobile") = log((100 - 8 + 0.5) / (8 + 0.5)) = log(92.5 / 8.5) = log(10.88) = 2.39

  n("policy") = 15  (15 out of 100 docs contain "policy")
  IDF("policy") = log((100 - 15 + 0.5) / (15 + 0.5)) = log(85.5 / 15.5) = log(5.52) = 1.71

BM25 for d1 (term "mobile"):
  Using k1=1.5, b=0.75:
  
  Numerator = f("mobile", d1) × (k1 + 1) = 2 × 2.5 = 5.0
  Denominator = f("mobile", d1) + k1 × (1 - b + b × |d1|/avgdl)
              = 2 + 1.5 × (1 - 0.75 + 0.75 × 7/50)
              = 2 + 1.5 × (0.25 + 0.105)
              = 2 + 1.5 × 0.355
              = 2 + 0.533
              = 2.533
  
  Score for "mobile" in d1 = IDF × (Num/Denom) = 2.39 × (5.0/2.533) = 2.39 × 1.974 = 4.72

BM25 for d1 (term "policy"):
  Numerator = 1 × 2.5 = 2.5
  Denominator = 1 + 1.5 × 0.355 = 1.533
  Score = 1.71 × (2.5/1.533) = 1.71 × 1.631 = 2.79

Total BM25(q, d1) = 4.72 + 2.79 = 7.51  ← HIGH score (relevant!)

BM25 for d2: 
  Both terms have f=0, so both scores = 0
  Total BM25(q, d2) = 0  ← ZERO score (no matching keywords)
```

**Why combine BM25 with vector search?** Vector search catches semantic similarity ("phone guidelines" ≈ "mobile policy"), but might miss exact keyword matches for rare terms like product codes. BM25 catches exact matches. Together they're more robust.

---

### 5.9 ANN Algorithms — Fast Search in Millions of Vectors

When you have millions of chunk vectors, comparing the query to every single one (brute-force) is too slow. ANN algorithms organize vectors into clever data structures for fast approximate search.

---

#### HNSW (Hierarchical Navigable Small World)

**Intuition**: Imagine a city with highways, main roads, and residential streets. To get from point A to point B:
1. First take the highway (few stops, big jumps) to get to the right neighborhood
2. Then take main roads to get to the right block
3. Then walk along residential streets to find the exact house

HNSW builds a similar multi-level network of vectors.

**Structure:**

```
Layer 2 (express):  A ─────────────────── D ──────────── G
                    (very few nodes, long-range connections)

Layer 1 (medium):   A ──── B ──── C ──── D ──── E ──── G
                    (more nodes, medium-range connections)

Layer 0 (local):    A─B─C─D─E─F─G─H─I─J─K─L─M─N─O─P─Q
                    (ALL nodes, short-range connections only)
```

**Search algorithm:**

```
1. Start at the entry point on the top layer
2. Greedily walk to the closest node to the query on this layer
3. Drop down to the next layer, starting from that node
4. Repeat until Layer 0
5. Do a thorough search on Layer 0 around the landing point
6. Return the K nearest neighbours found
```

**Time complexity**: $O(\log N)$ per query (vs $O(N)$ brute-force)

**Why it works**: The top layers let you "teleport" to the right region of the vector space quickly. The bottom layer gives precise local search.

---

#### IVF (Inverted File Index)

**Intuition**: Divide all vectors into groups (clusters). At query time, figure out which group the query belongs to, then only search that group.

**Offline (one-time setup)**: Run k-means to create $C$ cluster centers:

$$\min_{\mu_1, \ldots, \mu_C} \sum_{i=1}^{N} \min_{j=1}^{C} \|c_i - \mu_j\|^2$$

**In English**: Find $C$ center points such that every vector is as close as possible to its nearest center.

**Online (per query):**

1. Find the $n_{\text{probe}}$ nearest cluster centers to $q$
2. Only search vectors inside those clusters

**Complexity**: Searches roughly $N \times n_{\text{probe}} / C$ vectors instead of all $N$.

```
Example: N=1,000,000 vectors, C=1000 clusters, n_probe=10
  Brute force: search all 1,000,000
  IVF:         search ~10,000 (100x faster!)
```

---

#### Product Quantization (PQ)

**Intuition**: Compress each 768-dimensional vector into a much smaller code (e.g., 96 bytes). This uses less memory AND enables faster distance computation using precomputed lookup tables.

**How it works:**

1. Split each 768D vector into 96 sub-vectors of 8 dimensions each
2. For each sub-vector group, learn a codebook of 256 representative sub-vectors (using k-means)
3. Replace each sub-vector with the ID (0-255) of the closest codebook entry

```
Original vector (768 floats × 4 bytes = 3072 bytes):
  [0.12, -0.34, 0.56, 0.08, 0.91, ..., 0.23]

Compressed code (96 bytes — 32x smaller!):
  [42, 117, 5, 230, 89, ..., 156]
  Each number is an index into a codebook of 256 entries
```

---

### 5.10 Attention in the Generator LLM

When the LLM processes the augmented prompt (instructions + retrieved chunks + query), it uses the same attention mechanism described in Section 5.4, but with two key differences:

1. **Causal masking**: The LLM can only attend to **previous** tokens (not future ones), since it generates left-to-right.
2. **Multi-head attention**: Multiple attention mechanisms run in parallel, each learning to focus on different types of information.

**Multi-head attention formula:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h) \cdot W^O$$

where each head computes attention independently:

$$\text{head}_i = \text{Attention}(Q W_i^Q, \; K W_i^K, \; V W_i^V)$$

**Intuition**: Head 1 might focus on entity names in the retrieved context ("mobile", "device"), head 2 on action words ("prohibit", "require"), head 3 on numbers and dates. The model runs all these in parallel, then combines them.

---

## 6. Key Algorithms Behind RAG

| Algorithm | Where Used | Purpose |
|---|---|---|
| **Byte-Pair Encoding (BPE)** | Tokenization | Builds a subword vocabulary by iteratively merging frequent character pairs |
| **WordPiece** | Tokenization (BERT) | Similar to BPE but uses likelihood instead of frequency to choose merges |
| **Transformer Encoder** | Embedding model | Produces contextualized token representations via self-attention |
| **Mean Pooling** | Embedding model | Averages token-level vectors into a single sentence/chunk vector |
| **k-Means Clustering** | IVF index | Partitions vector space into searchable regions |
| **HNSW Graph** | Vector store | Enables $O(\log N)$ approximate nearest neighbour search |
| **Product Quantization** | FAISS compression | Compresses vectors for memory-efficient storage and fast distance computation |
| **Greedy / Beam Search** | LLM decoding | Selects the next token during response generation |
| **Softmax** | Attention + output layer | Converts raw scores into probability distributions |
| **Contrastive Learning (InfoNCE)** | Embedding model training | Trains embeddings so similar texts cluster and dissimilar texts separate |
| **BM25** | Hybrid retrieval | Classic keyword ranking algorithm (TF-IDF family) |

---

## 7. How Everything Plays Together — System Data Flow

```
┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────┐
│ Raw Docs │────▶│  Loader  │────▶│   Splitter   │────▶│  Chunks[]  │
│ (PDF,TXT)│     │(LangChain│     │(Recursive    │     │            │
│          │     │ Loaders) │     │ Char/Token)  │     │            │
└─────────┘     └──────────┘     └──────────────┘     └─────┬──────┘
                                                            │
                              Each chunk's text gets        │
                              tokenized into subwords,      │
                              then fed through the          ▼
                              Transformer encoder      ┌──────────────┐
                              to produce a single      │  Embedding   │
                              768-dim vector           │    Model     │
                                                       │ (e.g., MPNet)│
                                                       └──────┬───────┘
                                                              │
                                   One vector per chunk       │
                                   gets stored                ▼
                                                       ┌──────────────┐
                                                       │ Vector Store │
                                                       │ (ChromaDB /  │
                                                       │  FAISS)      │
                                                       └──────┬───────┘
                                                              │
                    ┌─────────────────────────────────────────┘
                    │ (vectors indexed and ready)
                    │
                    ▼
 User Query ──▶ Embed Query ──▶ Similarity Search ──▶ Top-K Chunks
  (tokenized      (same model     (cosine/dot           │
   by the same     as chunks)      product)              │
   embedding                                             ▼
   model's                                      ┌────────────────┐
   tokenizer)                                   │  Prompt        │
                                                │  Template      │
                                                │  (instruction  │
                                                │  + context     │
                                                │  + question)   │
                                                └───────┬────────┘
                                                        │
                            The LLM tokenizes the       │
                            ENTIRE augmented prompt      ▼
                            using its OWN tokenizer ┌────────────────┐
                            (different from the     │   Generator    │
                            embedding model's)      │     LLM        │
                                                    │ (Granite/GPT/  │
                                                    │  LLaMA)        │
                                                    └───────┬────────┘
                                                            │
                                                            ▼
                                                       Response to
                                                         User
```

**The critical insight**: The embedding model and the generator LLM are **different models** with **different jobs** and **different tokenizers**:
- **Embedding model** (encoder): Maps text → fixed-length vector. Tokenizer A (e.g., WordPiece).
- **Generator LLM** (decoder): Takes augmented prompt → generates text. Tokenizer B (e.g., BPE).

The vector store is the **bridge**: it stores the embedding model's outputs and feeds them (as original text, not vectors) into the generator's input.

---

## 8. What Each Technology Provides

| Technology | Role in RAG | What It Provides |
|---|---|---|
| **LangChain** | Orchestration framework | Chains together loaders, splitters, embeddings, retrievers, LLMs, memory, and prompts into a coherent pipeline |
| **HuggingFace Transformers** | Model hub + inference | Pre-trained embedding models and LLMs, tokenizers, model architectures |
| **Sentence-Transformers** | Embedding specialization | Models fine-tuned specifically for producing high-quality sentence/paragraph embeddings |
| **ChromaDB** | Vector storage | Lightweight, in-memory or persistent vector database with HNSW indexing |
| **FAISS** | Vector search library | High-performance similarity search with GPU support, multiple index types (IVF, PQ, HNSW) |
| **Milvus** | Distributed vector DB | Handles billions of vectors with horizontal scaling, supports hybrid search |
| **IBM watsonx.ai** | LLM provider | Hosted LLM inference (Granite, LLaMA, etc.) with enterprise features |
| **OpenAI API** | LLM + embedding provider | GPT models for generation, embedding models for vectorization |
| **Pinecone / Weaviate / Qdrant** | Managed vector DBs | Cloud-hosted vector search with filtering, metadata, and hybrid search capabilities |

---

## 9. Advanced RAG Patterns

### 9.1 Naive RAG vs. Advanced RAG vs. Modular RAG

```
Naive RAG:       Chunk → Embed → Retrieve → Generate
                 (simple but limited)

Advanced RAG:    Pre-retrieval optimization (query rewriting, HyDE)
                    ↓
                 Multi-stage retrieval (coarse → fine)
                    ↓
                 Post-retrieval processing (re-ranking, compression)
                    ↓
                 Generate with refined context

Modular RAG:     Swappable components — mix retrieval strategies,
                 add routing, caching, feedback loops
```

### 9.2 Query Transformation Techniques

| Technique | Description | Why It Helps |
|---|---|---|
| **Multi-Query** | Generate N paraphrases of the query using an LLM, retrieve for each, take union | Increases recall by covering different phrasings |
| **HyDE** (Hypothetical Document Embeddings) | Ask the LLM to generate a hypothetical answer, embed that instead of the query | The hypothetical answer is closer in embedding space to the real answer chunks |
| **Step-Back Prompting** | Ask a more general version of the query first | Helps when the query is too specific for direct retrieval |

### 9.3 Re-Ranking

After initial retrieval of top-K candidates (fast but approximate), apply a **cross-encoder** re-ranker that jointly encodes (query, chunk) pairs for more accurate relevance scoring:

$$\text{relevance}(q, c) = \sigma\big(W \cdot \text{BERT}([q \; ; \; \text{SEP} \; ; \; c]) + b\big)$$

where $\sigma$ is the sigmoid function.

Cross-encoders are more accurate than bi-encoders (separate embedding) but too slow for searching the entire corpus — hence the two-stage pipeline.

---

## 10. Conversation Memory in RAG

A single-turn RAG system treats each query independently. But real users have multi-turn conversations where "it" refers to something mentioned earlier.

### The Problem

```
User: "What is the mobile policy?"   → retrieves mobile policy chunks ✓
User: "List points in it?"           → "it" = ??? The retriever doesn't know
```

### The Solution: ConversationBufferMemory

LangChain's `ConversationBufferMemory` stores the full chat history. Before each retrieval, the system rewrites the current query to be self-contained by incorporating context from the history.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=False
)
```

**What happens internally:**

1. User asks: "List points in it?"
2. LangChain sees chat history: previous Q was about mobile policy
3. An LLM rewrites the query: "List the key points of the mobile phone policy"
4. This standalone query goes through the normal RAG retrieval pipeline

### Memory Types

| Memory Type | Behavior | Tradeoff |
|---|---|---|
| `ConversationBufferMemory` | Stores all messages verbatim | Complete history but grows without bound |
| `ConversationSummaryMemory` | Uses an LLM to summarize older messages | Bounded size but lossy |
| `ConversationBufferWindowMemory` | Keeps only last K turns | Simple and bounded but loses early context |
| `ConversationTokenBufferMemory` | Keeps messages up to a token limit | Token-aware, fits model context windows |

---

## 11. Prompt Engineering for RAG

The prompt template is the **contract** between your system and the LLM. A well-designed template prevents hallucination and controls output format.

### Template Anatomy

```python
prompt_template = """Use the following pieces of context to answer the question.
If you don't know the answer, say "I don't know." Do NOT make up an answer.

Context:
{context}

Question: {question}

Answer:"""
```

### Key Design Principles

1. **Ground the LLM**: Explicitly instruct it to use *only* the provided context.
2. **Fallback behavior**: Tell it what to do when the context doesn't contain the answer.
3. **Output format**: Specify if you want bullet points, JSON, citations, etc.
4. **Persona**: Set the tone — "You are a helpful HR assistant" changes response style.

### Chain Types in LangChain

| Chain Type | How It Uses Retrieved Chunks | When to Use |
|---|---|---|
| **`stuff`** | Concatenates all chunks into a single prompt | Few chunks, small total size |
| **`map_reduce`** | Sends each chunk to the LLM separately, then combines answers | Many chunks, large documents |
| **`refine`** | Iteratively refines the answer by processing one chunk at a time | Need high-quality synthesis |
| **`map_rerank`** | Scores each chunk's answer, returns the highest-scoring one | Need the single best answer |

---

## 12. Evaluation Metrics for RAG

RAG evaluation requires measuring both **retrieval quality** and **generation quality**.

### Retrieval Metrics

| Metric | Formula | Measures |
|---|---|---|
| **Recall@K** | $\frac{\|\text{relevant} \cap \text{retrieved}\|}{\|\text{relevant}\|}$ | What fraction of relevant docs were retrieved? |
| **Precision@K** | $\frac{\|\text{relevant} \cap \text{retrieved}\|}{K}$ | What fraction of retrieved docs were relevant? |
| **MRR** (Mean Reciprocal Rank) | $\frac{1}{\|Q\|}\sum_{i=1}^{\|Q\|}\frac{1}{\text{rank}_i}$ | How high does the first relevant doc rank? |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Measures ranking quality with graded relevance |

### Generation Metrics

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Does the answer only contain information from the retrieved context? (no hallucination) |
| **Answer Relevance** | Is the answer actually addressing the user's question? |
| **Context Relevance** | Are the retrieved chunks relevant to the question? |
| **RAGAS Score** | Composite metric combining faithfulness, answer relevance, and context relevance |

---

## 13. End-to-End Code Walkthrough (LangChain)

```python
# ─── 1. Load ───
from langchain.document_loaders import TextLoader
loader = TextLoader("companyPolicies.txt")
documents = loader.load()

# ─── 2. Split ───
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# ─── 3. Embed & Store ───
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# ─── 4. Build LLM ───
from ibm_watsonx_ai.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

model = Model(model_id="ibm/granite-3-3-8b-instruct", params={...}, credentials={...}, project_id="...")
llm = WatsonxLLM(model=model)

# ─── 5. Create Retrieval Chain ───
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""Use the context below to answer. If unsure, say "I don't know."

{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# ─── 6. Query ───
result = qa_chain.invoke("What is the mobile phone policy?")
print(result["result"])

# ─── 7. Add Memory for Multi-Turn ───
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_with_memory = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

qa_with_memory.invoke({"question": "What is the smoking policy?"})
qa_with_memory.invoke({"question": "List all its points."})  # "its" resolves correctly
```

---

## 14. Reference Papers & Further Reading

### Foundational Papers

| Paper | Year | Key Contribution |
|---|---|---|
| **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** | 2020 | The original RAG paper by Lewis et al. (Facebook AI). Introduced the RAG framework combining a parametric seq2seq model with a non-parametric retrieval component. |
| **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** | 2017 | Vaswani et al. Introduced the Transformer architecture that underlies all modern embedding models and LLMs. |
| **[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)** | 2018 | Devlin et al. Bidirectional encoder used as the basis for many embedding models. |
| **[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)** | 2019 | Reimers & Gurevych. Made BERT practical for sentence similarity by using siamese/triplet networks. |
| **[Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906)** | 2020 | Karpukhin et al. Showed that learned dense representations outperform BM25 for passage retrieval. |

### Advanced RAG Papers

| Paper | Year | Key Contribution |
|---|---|---|
| **[HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)** | 2022 | Gao et al. Use the LLM to generate a hypothetical answer, then embed that for retrieval. |
| **[Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511)** | 2023 | Asai et al. The model learns when to retrieve and self-evaluates its outputs. |
| **[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)** | 2024 | Sarthi et al. Builds a hierarchical tree of document summaries for multi-scale retrieval. |
| **[Corrective RAG (CRAG)](https://arxiv.org/abs/2401.15884)** | 2024 | Yan et al. Adds a self-correction mechanism to evaluate and improve retrieved documents. |

### Vector Search & Indexing

| Paper | Year | Key Contribution |
|---|---|---|
| **[Efficient and Robust Approximate Nearest Neighbor Search (HNSW)](https://arxiv.org/abs/1603.09320)** | 2016 | Malkov & Yashunin. The HNSW algorithm used by most vector databases. |
| **[Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202)** | 2011 | Jegou et al. Foundation of FAISS's compression techniques. |
| **[Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)** | 2017 | Johnson et al. (Facebook). The FAISS library for efficient similarity search. |

### Evaluation

| Paper | Year | Key Contribution |
|---|---|---|
| **[RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)** | 2023 | Es et al. Framework for evaluating RAG systems on faithfulness, relevance, and context quality. |
| **[Benchmarking Large Language Models in RAG](https://arxiv.org/abs/2309.01431)** | 2023 | Comprehensive benchmarking of different LLMs in RAG settings. |

### Recommended Reading Order for Beginners

1. Start with the **original RAG paper** (Lewis et al., 2020) for the big picture
2. Read **"Attention Is All You Need"** to understand Transformers
3. Read **Sentence-BERT** to understand how embeddings work
4. Read **DPR** to understand dense retrieval
5. Read **HNSW** to understand vector search
6. Explore advanced patterns: HyDE, Self-RAG, CRAG

---

> **Summary**: RAG bridges the gap between a frozen LLM and the ever-changing world of private, fresh data. It does this by (1) chunking and embedding documents into a vector space, (2) retrieving semantically similar chunks for a given query using ANN algorithms, and (3) grounding the LLM's generation in that retrieved context. The math is rooted in Transformer attention, contrastive embedding losses, and efficient nearest-neighbour search. The engineering is glued together by orchestration frameworks like LangChain. Master these pieces, and you can build, debug, and scale any RAG system.

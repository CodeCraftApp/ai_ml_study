# The Beginner's Foundations: From Data to LLMs

> A structured guide for AI/ML founding engineers who want to understand what's actually happening under the hood — not just use the API.

---

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [1. Supervised vs. Unsupervised Learning](#1-supervised-vs-unsupervised-learning)
    - [The Core Distinction](#the-core-distinction)
    - [Supervised Learning — In Depth](#supervised-learning--in-depth)
    - [Unsupervised Learning — In Depth](#unsupervised-learning--in-depth)
    - [Self-Supervised Learning — The Bridge](#self-supervised-learning--the-bridge)
  - [2. Foundation Models](#2-foundation-models)
    - [What Is a Foundation Model?](#what-is-a-foundation-model)
    - [The "Before and After"](#the-before-and-after)
    - [How Adaptation Works](#how-adaptation-works)
    - [A Note on Terminology Honesty](#a-note-on-terminology-honesty)
  - [3. Large Language Models (LLMs)](#3-large-language-models-llms)
    - [What Makes an LLM Different from Traditional ML?](#what-makes-an-llm-different-from-traditional-ml)
    - [The Training Objective: Why "Predict the Next Token" Is So Powerful](#the-training-objective-why-predict-the-next-token-is-so-powerful)
    - [The Transformer — The Architecture That Made This Possible](#the-transformer--the-architecture-that-made-this-possible)
    - [Emergent Capabilities](#emergent-capabilities)
  - [4. Tokens — How Models Actually "Read"](#4-tokens--how-models-actually-read)
    - [Why Not Just Use Words?](#why-not-just-use-words)
    - [What Is a Token?](#what-is-a-token)
    - [How Tokenizers Are Built — Byte-Pair Encoding (BPE)](#how-tokenizers-are-built--byte-pair-encoding-bpe)
    - [Token ≠ Word — The Numbers](#token--word--the-numbers)
    - [Why Tokens Matter to You as a Founder](#why-tokens-matter-to-you-as-a-founder)
- [Glossary](#glossary)
- [Success Criteria Self-Check](#success-criteria-self-check)

---

<!-- ============================================================ -->
<!-- SECTION: INTRODUCTION                                         -->
<!-- ============================================================ -->

## Introduction

Every AI system — from a spam filter to GPT — runs on the same fundamental loop: **data in, pattern learned, prediction out.** The difference between a 1990s spam filter and a modern LLM isn't a difference in *kind*; it's a difference in *scale, architecture, and the type of pattern being learned.*

This guide covers four concepts that form the spine of that progression:

1. **Supervised vs. Unsupervised Learning** — the two paradigms for how machines learn from data.
2. **Foundation Models** — the architectural shift that changed AI from "one model per task" to "one model, many tasks."
3. **Large Language Models (LLMs)** — what makes them different from everything that came before.
4. **Tokens** — the atomic unit LLMs actually operate on (spoiler: it's not words).

By the end, you should be able to explain — to an investor, a teammate, or yourself at 2 a.m. — why an LLM doesn't "read" English, why tokens matter for your costs, and where foundation models fit in the stack.

> **Founder's Tip:** When evaluating AI products, the first question to ask is *"What kind of learning does this use, and what data did it learn from?"* That single question cuts through 90% of marketing noise.

---

<!-- ============================================================ -->
<!-- SECTION: CORE CONCEPTS                                        -->
<!-- ============================================================ -->

## Core Concepts

---

### 1. Supervised vs. Unsupervised Learning

<!-- <thought>
The best analogy for a beginner: Supervised learning is a student studying
with an answer key. Every practice problem has the correct answer printed
next to it. The student learns by comparing their attempt to the right answer
and adjusting. Unsupervised learning is an archaeologist handed a box of
unlabeled artifacts — no guidebook, no labels. They have to find structure
themselves: "These 50 artifacts are made of bronze and date to the same era;
these 30 are ceramic and more recent." The learning happens by discovering
groupings and relationships, not by being told what's right.
</thought> -->

#### The Core Distinction

| | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Training data** | Labeled — every input has a known correct output | Unlabeled — just raw data, no "answers" |
| **What the model learns** | A mapping from inputs to outputs: *f(X) → Y* | Hidden structure, patterns, or groupings in data |
| **Goal** | Predict the right answer for new, unseen inputs | Discover what's *in* the data that humans haven't explicitly tagged |
| **Analogy** | A student studying with an answer key | An archaeologist sorting unlabeled artifacts into groups |

#### Supervised Learning — In Depth

In supervised learning, you give the model pairs of **(input, correct output)** and it learns the function that maps one to the other.

**Two main flavors:**

**Classification** — the output is a discrete category.

```
Input: email text       → Output: "spam" or "not spam"
Input: chest X-ray      → Output: "pneumonia" or "healthy"
Input: transaction data  → Output: "fraudulent" or "legitimate"
```

The model learns a **decision boundary** — a line (or complex surface) in feature space that separates classes. A new data point falling on one side of the boundary gets one label; on the other side, the other.

**Regression** — the output is a continuous number.

```
Input: square footage, location, bedrooms  → Output: house price ($425,000)
Input: patient vitals, lab results          → Output: blood pressure (next week)
Input: ad spend, channel, time of year      → Output: expected revenue ($1.2M)
```

The model learns a function that fits a curve through the data points, minimizing the difference between predicted and actual values.

**How training works (the learning loop):**

1. The model makes a prediction on a training example.
2. A **loss function** measures how wrong the prediction is (e.g., cross-entropy for classification, mean squared error for regression).
3. **Backpropagation** computes how much each parameter contributed to the error.
4. **Gradient descent** adjusts the parameters slightly to reduce the error.
5. Repeat millions of times across the dataset.

The model converges when it can't reduce the loss much further. At that point, you test it on held-out data it has never seen to check if it **generalizes** or just memorized the training set (overfitting).

> **Founder's Tip:** When building a supervised learning product, the bottleneck is almost never the model — it's the **labeled data.** Labeling is expensive, slow, and error-prone. Before committing to a supervised approach, ask: *"Do I have enough high-quality labels, and can I afford to get more?"* If the answer is no, look at self-supervised or unsupervised approaches first.

#### Unsupervised Learning — In Depth

In unsupervised learning, the model receives data **with no labels at all.** It must find structure on its own.

**Key techniques:**

**Clustering** — grouping similar data points together without being told what the groups should be.

```
Input: customer purchase histories (no labels)
Output: 5 discovered segments
  - Cluster 1: "Budget weekend shoppers"
  - Cluster 2: "Premium weekday buyers"
  - Cluster 3: "Seasonal bulk purchasers"
  ...
```

Algorithms like **K-Means** partition data into *k* groups by minimizing the distance between each point and its cluster center. **DBSCAN** finds clusters of arbitrary shape by identifying dense regions. The model doesn't know what these clusters "mean" — a human interprets that.

**Dimensionality Reduction** — compressing high-dimensional data into fewer dimensions while preserving important structure.

A dataset with 10,000 features (columns) might have redundancies. Techniques like **PCA (Principal Component Analysis)** and **t-SNE** find the most important axes of variation and compress the data down — from 10,000 dimensions to 50, or even 2 for visualization.

```
Input: gene expression data with 20,000 genes per sample
PCA output: 50 principal components that capture 95% of the variance
Use case: visualization, denoising, speeding up downstream models
```

**Anomaly Detection** — finding data points that don't fit the learned pattern.

```
Input: network traffic logs (no labels for "attack" vs "normal")
Model learns: what "normal" traffic looks like
Output: flags unusual patterns as potential intrusions
```

**The crucial insight:** Unsupervised learning isn't "worse" than supervised — it solves a different problem. You use it when you *don't have labels* and want to discover structure, or when you want to *learn representations* of data that can be used downstream. In fact, the pre-training phase of every modern LLM is unsupervised (technically **self-supervised**, which we'll cover below).

#### Self-Supervised Learning — The Bridge

There's a third paradigm that doesn't fit neatly into either box, and it's arguably the most important one in modern AI:

**Self-supervised learning** creates its own labels from the structure of the data.

```
Original sentence: "The cat sat on the [MASK]."
Self-generated label: "mat"
```

The model takes unlabeled text, masks out a piece, and trains itself to predict the missing piece. The *data itself* provides the supervision signal — no human labeling needed. This is how BERT, GPT, and every modern foundation model is pre-trained.

**Why this matters:** It unlocks the entire internet as training data. You don't need humans to label billions of web pages — the text supervises itself.

> **Founder's Tip:** When someone pitches you an AI product, ask whether their model requires labeled data *specific to your domain.* If yes, factor in the labeling cost. If they're using a foundation model that was pre-trained via self-supervision, the heavy lifting is already done — your domain-specific data just fine-tunes the last mile.

---

### 2. Foundation Models

<!-- <thought>
The best analogy: A foundation model is like a liberal arts education. You
spend four years learning broadly — history, science, writing, math — and
then you specialize in med school, law school, or engineering. The four-year
foundation isn't wasted; it gives you reasoning skills, general knowledge,
and adaptability that you carry into your specialty. A foundation model is
the same: trained broadly on massive data, then specialized (fine-tuned)
for specific tasks. Before foundation models, building an AI was like going
straight to med school with no prior education — you had to start from
scratch every time.
</thought> -->

#### What Is a Foundation Model?

A foundation model is a large AI model **pre-trained on broad, diverse data** at massive scale, designed to be **adapted for many downstream tasks** rather than built for a single purpose.

The term was coined by Stanford's Center for Research on Foundation Models (CRFM) in 2021. The key properties:

| Property | What It Means |
|---|---|
| **Scale** | Trained on terabytes to petabytes of data (text, images, code, audio) using thousands of GPUs |
| **Generality** | Not designed for one task — the same base model can be adapted for translation, summarization, code generation, question-answering, etc. |
| **Transfer learning** | Knowledge learned during pre-training transfers to downstream tasks, dramatically reducing the data and compute needed for specialization |
| **Emergence** | Capabilities appear at scale that weren't explicitly trained for (e.g., in-context learning, chain-of-thought reasoning) |

#### The "Before and After"

**Before foundation models (pre-2018):**

```
Task: Spam detection    → Train a spam classifier from scratch
Task: Sentiment analysis → Train a sentiment model from scratch
Task: Translation       → Train a translation model from scratch
Task: Summarization     → Train a summarization model from scratch
```

Each task required its own dataset, its own training run, its own architecture decisions. There was no shared knowledge between them.

**After foundation models:**

```
Foundation model: Pre-trained on massive diverse data (one expensive training run)
    ├── Fine-tune for spam detection      (small dataset, cheap)
    ├── Fine-tune for sentiment analysis  (small dataset, cheap)
    ├── Fine-tune for translation         (small dataset, cheap)
    └── Use directly via prompting        (zero additional training)
```

One massive investment in pre-training pays dividends across dozens of downstream tasks. This is why companies like OpenAI, Google, Meta, and IBM invest hundreds of millions in training a single base model.

#### How Adaptation Works

There are multiple ways to take a foundation model and make it do what you need:

**1. Prompting (no training at all)**

You write a natural language instruction and the model follows it. This works because the model learned to follow patterns during pre-training.

```
"Classify this email as spam or not spam: [email text]"
```

**2. Fine-tuning (light training)**

You further train the model on a small, task-specific dataset. The pre-trained weights give it a massive head start — you're adjusting, not teaching from zero.

```
Dataset: 5,000 (email, spam/not-spam) labeled pairs
Training: 1-3 epochs, a few GPU-hours
Result: Model now excels at spam detection while retaining general capabilities
```

**3. Retrieval-Augmented Generation (RAG)**

Instead of retraining, you give the model access to an external knowledge base at inference time. The model retrieves relevant documents and uses them to ground its response.

```
User: "What's our refund policy?"
RAG: Retrieves policy doc from vector store → Model answers using that doc
```

**4. Adapters / LoRA (parameter-efficient fine-tuning)**

Instead of modifying all model parameters, you freeze the base model and train a small set of adapter weights (often < 1% of total parameters). This is cheaper and lets you maintain multiple task-specific adaptations of the same base model.

> **Founder's Tip:** For most startups, the foundation model is *not* your moat — it's a commodity. Your moat is the **data you fine-tune with**, the **retrieval pipeline you build around it**, and the **domain-specific UX** you deliver. Don't try to train your own foundation model unless you have a very good reason and very deep pockets.

#### A Note on Terminology Honesty

The term "foundation model" is widely used but not universally agreed upon in its exact boundaries. Some researchers argue it overstates the generality of these models, since they still fail at tasks outside their training distribution. Others note that calling them "foundational" implies a stability and reliability they don't always have. This guide uses the term as the industry commonly understands it, but be aware that it carries implicit assumptions about generality that deserve scrutiny in production systems.

---

### 3. Large Language Models (LLMs)

<!-- <thought>
The best way to explain how LLMs differ from traditional ML: Traditional ML
is a specialist — you train a model to do ONE thing (predict house prices,
classify images, detect fraud). An LLM is a generalist that was trained on
so much text that it developed broad capabilities. The key difference isn't
just scale; it's the training objective. Traditional models have task-specific
objectives ("minimize classification error on this dataset"). LLMs have a
universal objective ("predict the next word") that, at scale, produces
emergent general intelligence. It's the difference between training someone
to be a plumber vs. giving someone such a broad education that they can
figure out plumbing, along with a thousand other things, on the fly.
</thought> -->

#### What Makes an LLM Different from Traditional ML?

An LLM is a foundation model specialized in language. But the differences from traditional ML go deeper than just "it's bigger":

| Dimension | Traditional ML | Large Language Models |
|---|---|---|
| **Training objective** | Task-specific (e.g., "minimize classification error on this dataset") | Universal: "predict the next token" — one objective, applied to all of language |
| **Data** | Curated, labeled datasets (thousands to millions of examples) | Massive unlabeled text corpora (trillions of tokens from the internet, books, code) |
| **Architecture** | Varies widely (decision trees, SVMs, CNNs, RNNs...) | Almost exclusively the **Transformer** architecture (2017) |
| **Input/Output** | Fixed format (tabular data → number, image → class label) | Flexible natural language (text → text, covering almost any task) |
| **How you "program" it** | Feature engineering + training | **Prompting** — you describe what you want in natural language |
| **Generalization** | Narrow — only does what it was trained for | Broad — handles tasks it was never explicitly trained for (zero-shot) |

#### The Training Objective: Why "Predict the Next Token" Is So Powerful

A traditional sentiment classifier's objective is narrow:

```
Minimize: error on predicting positive/negative/neutral for movie reviews
```

An LLM's training objective is universal:

```
Minimize: error on predicting the next token, across ALL text ever written
```

This single objective, applied at massive scale, forces the model to learn:

- **Grammar and syntax** — to predict the next word, you need to understand sentence structure.
- **Facts and knowledge** — "The capital of France is ___" requires world knowledge to predict "Paris."
- **Reasoning patterns** — "If all dogs are animals, and Rex is a dog, then Rex is ___" requires logical reasoning to predict "an animal."
- **Code structure** — predicting the next token in source code requires understanding programming logic.
- **Social/emotional patterns** — predicting dialogue requires understanding human communication.

No one explicitly taught the model any of these skills. They **emerged** from the scale of the objective and the data. This is the key insight of the LLM paradigm: a sufficiently powerful model with a simple objective and enough data develops complex capabilities.

#### The Transformer — The Architecture That Made This Possible

Before 2017, language models used **Recurrent Neural Networks (RNNs)** that processed text one word at a time, left to right. This created two problems:

1. **Vanishing gradients** — by the time the model reached word 500 in a document, it had largely "forgotten" word 1.
2. **Sequential processing** — each word had to wait for the previous word to be processed. No parallelism. Slow training.

The **Transformer architecture** (Vaswani et al., 2017 — "Attention Is All You Need") solved both:

**Self-Attention** — instead of processing tokens sequentially, the model looks at *all* tokens simultaneously and computes how much each token should "attend to" every other token.

```
Sentence: "The animal didn't cross the street because it was too tired."

Question: What does "it" refer to?

Self-attention computes:
  "it" attends strongly to "animal" (high attention weight)
  "it" attends weakly to "street" (low attention weight)
  → The model resolves "it" = "the animal"
```

Mathematically, attention is computed as:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \]

Where:
- **Q (Query):** "What am I looking for?" — what the current token wants to know.
- **K (Key):** "What do I contain?" — what each other token advertises about itself.
- **V (Value):** "What information do I carry?" — the actual content to retrieve.
- **\( d_k \):** The dimension of the key vectors (used for scaling to prevent extreme values).

The dot product \( QK^T \) measures similarity between the query and each key. Softmax converts these scores into a probability distribution (weights that sum to 1). Those weights are applied to the values to produce a context-aware representation.

**Multi-Head Attention** — instead of one attention computation, the model runs several in parallel (typically 32–128 "heads"), each learning to attend to different types of relationships (syntactic, semantic, positional, etc.). The outputs are concatenated and combined.

**Why this enables scale:** Since attention processes all tokens in parallel (not sequentially), Transformers can be trained on massive datasets using GPU parallelism. This is what unlocked the path from small language models to billion-parameter LLMs.

#### Emergent Capabilities

One of the most striking properties of LLMs is **emergence** — capabilities that appear only when the model reaches a certain scale, without being explicitly trained for them:

- **In-context learning:** The model can learn new tasks from examples provided in the prompt, without any parameter updates (this is what few-shot prompting exploits).
- **Chain-of-thought reasoning:** Larger models can break down multi-step problems when prompted to "think step by step."
- **Code generation:** Models trained primarily on text also learn to write functional code, because code was part of the training data.
- **Instruction following:** With alignment training (RLHF/DPO), models learn to follow complex, multi-part instructions.

These capabilities aren't present in smaller models and appear somewhat discontinuously as model size increases — a phenomenon that is still not fully understood by the research community.

> **Founder's Tip:** Don't conflate "LLM" with "AI." An LLM is one type of model optimized for language tasks. If your problem is tabular data prediction (churn, pricing, fraud), a traditional ML model (XGBoost, logistic regression) will almost certainly outperform an LLM, cost 1000x less, and be easier to explain. Use the right tool for the problem.

---

### 4. Tokens — How Models Actually "Read"

<!-- <thought>
The critical misconception to address: beginners assume the model processes
words. It doesn't. It processes tokens, which are sub-word units. The best
analogy: Imagine you're a foreigner learning English, and you don't have a
complete dictionary. When you encounter "unbreakable," you don't know the
whole word, but you recognize the pieces: "un" + "break" + "able." Each piece
carries meaning you can combine. That's tokenization — the model breaks text
into learned sub-word pieces it can work with. This also directly impacts
cost (APIs charge per token) and context limits (models have a max token
window, not a max word window).
</thought> -->

#### Why Not Just Use Words?

The naive approach — treating each word as a unit — has serious problems:

1. **Vocabulary explosion:** English alone has hundreds of thousands of words. Add misspellings, slang, technical jargon, code, other languages, and the vocabulary becomes unmanageable.
2. **Unknown words:** Any word not in the vocabulary is unprocessable. A model trained on standard English would choke on "defragmentation" or "Kubernetes" if those words weren't in the training vocabulary.
3. **Morphological blindness:** The model can't see that "run," "running," "runner," and "ran" share a root. Each is a completely independent symbol.

Tokenization solves all of these.

#### What Is a Token?

A token is the **atomic unit of text** that a model processes. Modern LLMs use **sub-word tokenization** — they break text into pieces that are *between* characters and words in granularity.

```
Word:   "unbreakable"
Tokens: ["un", "break", "able"]

Word:   "Tokenization"
Tokens: ["Token", "ization"]

Word:   "AI"
Tokens: ["AI"]              ← common words stay whole

Word:   "xylophone"
Tokens: ["xy", "lo", "phone"]  ← rare words get split
```

The key insight: **common words and sub-words stay intact; rare words get broken into recognizable pieces.** The model never encounters a truly "unknown" word because it can decompose anything into known sub-word tokens.

#### How Tokenizers Are Built — Byte-Pair Encoding (BPE)

The most widely used algorithm is **Byte-Pair Encoding**, used by GPT, Llama, and most modern LLMs. Here's how it works:

**Step 1: Start with individual characters as the initial vocabulary.**

```
Vocabulary: {a, b, c, d, e, f, ..., z, A, B, ..., Z, 0, ..., 9, !, ?, ...}
```

**Step 2: Count the most frequent adjacent pair in the training corpus.**

```
Corpus: "the cat sat on the mat the cat"
Most frequent pair: ("t", "h") → appears many times
```

**Step 3: Merge that pair into a new token and add it to the vocabulary.**

```
New token: "th"
Vocabulary: {..., "th"}
Corpus re-encoded: "th e c a t s a t o n th e m a t th e c a t"
```

**Step 4: Repeat steps 2–3 thousands of times.**

```
Next merge: ("th", "e") → "the"
Next merge: ("c", "a") → "ca"
Next merge: ("ca", "t") → "cat"
...continue until vocabulary reaches target size (e.g., 32,000 or 100,000 tokens)
```

The result is a vocabulary of sub-word units where frequent words are single tokens and rare words decompose into multiple tokens. The vocabulary size is a design choice — GPT-4 uses ~100,000 tokens; Llama uses ~32,000.

#### Token ≠ Word — The Numbers

A rough rule of thumb for English text:

```
1 token ≈ 0.75 words    (or equivalently, 1 word ≈ 1.33 tokens)
100 tokens ≈ 75 words
```

But this varies significantly:

| Text Type | Tokens per Word (approx.) |
|---|---|
| Simple English prose | ~1.2–1.3 |
| Technical / medical text | ~1.5–1.8 (more rare words = more splits) |
| Source code | ~2.0–3.0+ (symbols, indentation, syntax) |
| Non-English languages | ~1.5–4.0+ (depends on how well the tokenizer represents that language) |
| Emojis / special characters | Often 1–3 tokens *each* |

**Practical example:**

```
Text: "The quick brown fox jumps over the lazy dog."

Word count:  9 words
Token count: 9 tokens (all common English words → 1 token each)

Text: "Deoxyribonucleic acid encodes genetic information."

Word count:  5 words
Token count: 11 tokens (rare/long words split into subwords)
```

#### Why Tokens Matter to You as a Founder

**1. Cost — APIs charge per token, not per word.**

```
GPT-4o pricing (example): $2.50 per 1M input tokens, $10.00 per 1M output tokens
A 1,000-word prompt ≈ 1,333 tokens → actual cost depends on token count
```

If your application sends long system prompts, chat history, or retrieved documents to the model, your costs scale with *tokens*, not the number of user-facing words. Optimizing prompt length directly reduces your bill.

**2. Context window — the model's "working memory" is measured in tokens.**

```
GPT-4o:     128,000 token context window
Llama 3:    8,000 – 128,000 tokens (varies by variant)
Granite:    8,000 – 128,000 tokens
```

Everything the model can "see" at once — your system prompt, the user's message, any retrieved documents, and the generated response — must fit within this window. If your retrieval pipeline pulls in 50,000 tokens of documents, that leaves only 78,000 tokens for everything else in a 128K model.

**3. Non-English tax — tokenizers are biased toward English.**

Most tokenizers are trained predominantly on English text, so English gets efficient tokenization (fewer tokens per word). Other languages — especially those with non-Latin scripts — get penalized:

```
English:  "Hello, how are you?"       → 6 tokens
Japanese: "こんにちは、お元気ですか？"  → 11 tokens  (same meaning, ~2x the tokens)
```

This means non-English users hit context limits sooner and pay more per API call. If your product serves a global audience, this is a critical design consideration.

> **Founder's Tip:** Before building, run your actual prompts through a tokenizer to understand your real costs. OpenAI's [tiktoken](https://github.com/openai/tiktoken) library and Hugging Face's [tokenizers](https://github.com/huggingface/tokenizers) let you count tokens programmatically. Build token budgeting into your architecture from day one — it's much harder to retrofit.

---

<!-- ============================================================ -->
<!-- SECTION: GLOSSARY                                             -->
<!-- ============================================================ -->

## Glossary

| Term | Definition |
|---|---|
| **Supervised Learning** | Training a model on labeled (input, output) pairs so it learns to predict outputs for new inputs. |
| **Unsupervised Learning** | Training a model on unlabeled data to discover hidden structure (clusters, patterns, anomalies). |
| **Self-Supervised Learning** | A form of unsupervised learning where the model generates its own labels from the data structure (e.g., predicting masked words). |
| **Foundation Model** | A large model pre-trained on broad data, designed to be adapted for many downstream tasks via fine-tuning or prompting. |
| **Large Language Model (LLM)** | A foundation model specialized in language, trained on massive text corpora to predict the next token. |
| **Transformer** | The neural network architecture (2017) that uses self-attention to process all tokens in parallel, enabling modern LLMs. |
| **Self-Attention** | A mechanism where each token computes relevance scores against all other tokens to build context-aware representations. |
| **Token** | The atomic sub-word unit an LLM processes. Not a word — common words are single tokens; rare words are split into multiple tokens. |
| **Tokenizer** | The algorithm (e.g., BPE) that converts raw text into a sequence of token IDs from a fixed vocabulary. |
| **BPE (Byte-Pair Encoding)** | A tokenization algorithm that iteratively merges the most frequent adjacent character pairs to build a sub-word vocabulary. |
| **Fine-Tuning** | Further training a pre-trained model on a smaller, task-specific dataset to specialize its behavior. |
| **RAG (Retrieval-Augmented Generation)** | A pattern where the model retrieves external documents at inference time to ground its response in specific knowledge. |
| **LoRA (Low-Rank Adaptation)** | A parameter-efficient fine-tuning method that trains small adapter weights while keeping the base model frozen. |
| **Context Window** | The maximum number of tokens the model can process in a single forward pass (prompt + response combined). |
| **Emergence** | Capabilities that appear in large models but are absent in smaller ones, despite no explicit training for those capabilities. |
| **RLHF** | Reinforcement Learning from Human Feedback — a technique to align model outputs with human preferences using a learned reward model. |
| **Overfitting** | When a model memorizes training data instead of learning generalizable patterns, performing well on training data but poorly on new data. |
| **Loss Function** | A mathematical function that measures how wrong a model's prediction is. Training minimizes this function. |
| **Gradient Descent** | An optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss. |

---

## Success Criteria Self-Check

After reading this guide, you should be able to answer:

- [ ] *"What's the difference between supervised and unsupervised learning?"* → One uses labeled data to learn input→output mappings; the other finds hidden structure in unlabeled data.
- [ ] *"What is a foundation model?"* → A large model pre-trained on broad data that can be adapted for many tasks, replacing the old "one model per task" approach.
- [ ] *"How is an LLM different from traditional ML?"* → Traditional ML has a task-specific objective and fixed I/O. An LLM has a universal objective (predict next token) and flexible natural language I/O, enabling broad generalization.
- [ ] *"What's the difference between a token and a word?"* → A token is a sub-word unit the model actually processes. Common words are single tokens; rare/long words get split into multiple tokens. APIs charge per token, not per word.
- [ ] *"Why do tokens matter for cost and context limits?"* → Because both are measured in tokens. More tokens = higher cost and less room in the context window.

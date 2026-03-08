# The Beginner's Foundations: From Data to LLMs

> A structured guide for AI/ML founding engineers who want to understand what's actually happening under the hood — not just use the API.

---

## Table of Contents

- [Introduction](#introduction)
  - [What Is Machine Learning?](#what-is-machine-learning)
  - [Types of Machine Learning Systems](#types-of-machine-learning-systems)
- [Core Concepts](#core-concepts)
  - [1. Supervised vs. Unsupervised Learning](#1-supervised-vs-unsupervised-learning)
    - [The Core Distinction](#the-core-distinction)
    - [Supervised Learning — In Depth](#supervised-learning--in-depth)
    - [Unsupervised Learning — In Depth](#unsupervised-learning--in-depth)
  - [2. Reinforcement Learning](#2-reinforcement-learning)
  - [3. Generative AI](#3-generative-ai)
    - [Input-to-Output Types](#input-to-output-types)
    - [How Does Generative AI Work?](#how-does-generative-ai-work)
  - [4. Self-Supervised Learning — The Bridge](#4-self-supervised-learning--the-bridge)
  - [5. Foundation Models](#5-foundation-models)
    - [What Is a Foundation Model?](#what-is-a-foundation-model)
    - [The "Before and After"](#the-before-and-after)
    - [How Adaptation Works](#how-adaptation-works)
    - [A Note on Terminology Honesty](#a-note-on-terminology-honesty)
  - [6. Large Language Models (LLMs)](#6-large-language-models-llms)
    - [What Makes an LLM Different from Traditional ML?](#what-makes-an-llm-different-from-traditional-ml)
    - [The Training Objective: Why "Predict the Next Token" Is So Powerful](#the-training-objective-why-predict-the-next-token-is-so-powerful)
    - [The Transformer — The Architecture That Made This Possible](#the-transformer--the-architecture-that-made-this-possible)
    - [Emergent Capabilities](#emergent-capabilities)
  - [7. Tokens — How Models Actually "Read"](#7-tokens--how-models-actually-read)
    - [Why Not Just Use Words?](#why-not-just-use-words)
    - [What Is a Token?](#what-is-a-token)
    - [How Tokenizers Are Built — Byte-Pair Encoding (BPE)](#how-tokenizers-are-built--byte-pair-encoding-bpe)
    - [Token ≠ Word — The Numbers](#token--word--the-numbers)
    - [Why Tokens Matter to You as a Founder](#why-tokens-matter-to-you-as-a-founder)
- [Glossary](#glossary)
- [Success Criteria Self-Check](#success-criteria-self-check)

---







## Introduction

### What Is Machine Learning?

Machine learning is the process of training a piece of software — called a **model** — to make useful predictions or generate content from data. Instead of writing explicit rules ("if humidity > 80% and cloud_cover > 70%, predict rain"), you give the model data and let it *discover* the rules itself.

> **Traditional approach vs. ML approach:**
>
> Suppose we want to predict rainfall. The traditional approach builds a physics-based simulation of Earth's atmosphere — massive fluid dynamics equations, incredibly difficult to get right.
>
> The ML approach gives a model an enormous amount of historical weather data until it *learns* the mathematical relationships between weather patterns that produce rain. Give it today's conditions, and it predicts tomorrow's rainfall — without a single line of physics.

Machine learning powers translation apps, autonomous vehicles, recommendation engines, medical imaging, fraud detection, code generation, and much more. It provides a fundamentally different way to solve problems: instead of programming the solution, you program the *learning process* and let the solution emerge from data.

### Types of Machine Learning Systems

ML systems are differentiated by *how they learn*. Every approach in this guide falls into one of these categories:

```mermaid
flowchart TD
    ML["Machine Learning"] --> SUP["Supervised Learning\n(labeled data)"]
    ML --> UNSUP["Unsupervised Learning\n(no labels)"]
    ML --> RL["Reinforcement Learning\n(rewards & penalties)"]
    ML --> GENAI["Generative AI\n(creates new content)"]
    SUP --> CLASS["Classification"]
    SUP --> REG["Regression"]
    UNSUP --> CLUST["Clustering"]
    UNSUP --> DIM["Dimensionality Reduction"]
    UNSUP --> ANOM["Anomaly Detection"]
    GENAI --> SELFSUP["Self-Supervised\nPre-training"]
    GENAI --> FM["Foundation Models"]
    FM --> LLMS["Large Language\nModels"]
```



This guide walks through each branch of this tree, from the fundamentals up to LLMs and tokens:

1. **Supervised Learning** — learning with labeled "answers" (classification and regression).
2. **Unsupervised Learning** — discovering hidden patterns in raw, unlabeled data.
3. **Reinforcement Learning** — learning by trial, error, and reward signals.
4. **Generative AI** — models that create new content (text, images, code, video).
5. **Self-Supervised Learning** — the bridge between unsupervised and supervised that powers modern AI.
6. **Foundation Models** — the "one model, many tasks" paradigm shift.
7. **Large Language Models (LLMs)** — what makes them different from everything before.
8. **Tokens** — the atomic unit LLMs actually operate on (spoiler: it's not words).

By the end, you should be able to explain — to an investor, a teammate, or yourself at 2 a.m. — why an LLM doesn't "read" English, why tokens matter for your costs, and where foundation models fit in the stack.

> **Founder's Tip:** When evaluating AI products, the first question to ask is *"What kind of learning does this use, and what data did it learn from?"* That single question cuts through 90% of marketing noise.

---







## Core Concepts

---

### 1. Supervised vs. Unsupervised Learning



#### The Core Distinction


|                           | Supervised Learning                              | Unsupervised Learning                                               |
| ------------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| **Training data**         | Labeled — every input has a known correct output | Unlabeled — just raw data, no "answers"                             |
| **What the model learns** | A mapping from inputs to outputs: *f(X) → Y*     | Hidden structure, patterns, or groupings in data                    |
| **Goal**                  | Predict the right answer for new, unseen inputs  | Discover what's *in* the data that humans haven't explicitly tagged |
| **Analogy**               | A student studying with an answer key            | An archaeologist sorting unlabeled artifacts into groups            |


#### Supervised Learning — In Depth

In supervised learning, you give the model pairs of **(input, correct output)** and it learns the function that maps one to the other. Think of it like a student studying with an answer key — the student learns by comparing their attempt to the correct answer and adjusting.

The entire supervised learning lifecycle has five stages:

```mermaid
flowchart LR
    A["Data\n(features + labels)"] --> B["Model\n(mathematical function)"]
    B --> C["Training\n(learn from errors)"]
    C --> D["Evaluating\n(test on held-out data)"]
    D --> E["Inference\n(predict on new data)"]
    D -.->|"poor results"| C
```



##### Stage 1: Data — The Driving Force

Data is made up of **features** and **labels**:

- **Features** are the input values the model uses to make predictions (temperature, humidity, wind direction).
- **Labels** are the correct answers (rainfall amount, spam/not-spam).

A dataset is characterized by its **size** (number of examples) and **diversity** (range of scenarios covered). Good datasets are both large *and* diverse:


| Dataset                                   | Large? | Diverse? | Problem                                     |
| ----------------------------------------- | ------ | -------- | ------------------------------------------- |
| 100 years of data, but only July          | Yes    | No       | Can't predict January rainfall              |
| Every month represented, but only 2 years | No     | Yes      | Not enough years to account for variability |
| 50 years, all months, multiple regions    | Yes    | Yes      | Strong foundation for generalization        |


Datasets also vary in the number of features. A weather dataset might have hundreds of features (satellite imagery, barometric pressure, dew point) or just 3–4 (temperature, humidity, wind). More *relevant* features help the model discover better patterns, but irrelevant features add noise — not every column helps.

##### Stage 2: The Model & Its Parameters

In supervised learning, a model is a mathematical function that maps input features to output labels. But what *is* that function made of? **Parameters.**

**What is a parameter?**

A parameter is a single numerical value inside the model that gets adjusted during training. Think of parameters as the "knobs" the model can turn to improve its predictions. Before training, these knobs are set to random values. Training is the process of finding the right setting for every knob.

There are two types of parameters:


| Type        | What It Does                                                                                                                        | Analogy                                                                |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **Weights** | Control how much influence each input feature has on the output. Every connection between neurons in a neural network has a weight. | Volume knobs — they amplify or dampen each input signal.               |
| **Biases**  | Allow the model to shift its output up or down independently of the inputs. Each neuron has a bias term.                            | A baseline offset — like starting a thermostat at 20°C instead of 0°C. |


**A concrete example:**

In a simple linear model predicting house price from square footage:

```
price = weight × square_footage + bias
price = 200 × 1500 + 50,000
price = $350,000
```

This model has **2 parameters**: one weight (200) and one bias (50,000). Training adjusts these two numbers until the predictions match actual prices as closely as possible.

Real models have far more parameters:


| Model                    | Parameter Count | Context                           |
| ------------------------ | --------------- | --------------------------------- |
| Simple linear regression | 2–100           | One weight per feature + bias     |
| Small neural network     | ~10,000         | A few hidden layers               |
| BERT (2018)              | 110 million     | NLP breakthrough model            |
| GPT-3 (2020)             | 175 billion     | First "large" LLM                 |
| Llama 3 8B               | 8 billion       | Efficient open-weight model       |
| Llama 3 70B              | 70 billion      | High-capability open-weight model |
| GPT-4 (estimated)        | ~1.8 trillion   | Mixture-of-experts architecture   |


When someone says "Llama 3 is a 70B model," they mean it has **70 billion parameters** — 70 billion individual numbers that were tuned during training. More parameters generally means the model can learn more complex patterns, but also requires more data, more compute, more memory, and more cost to run.

**Parameters vs. hyperparameters:**

Don't confuse these — they sound similar but are fundamentally different:


|                               | Parameters                       | Hyperparameters                                          |
| ----------------------------- | -------------------------------- | -------------------------------------------------------- |
| **Set by**                    | The training process (automatic) | The engineer (manual, before training)                   |
| **Examples**                  | Weights, biases                  | Learning rate, batch size, number of layers, temperature |
| **Adjusted during training?** | Yes — this *is* training         | No — fixed before training starts                        |
| **Analogy**                   | What the student learns          | The study schedule the teacher sets                      |


> **Founder's Tip:** When evaluating models, parameter count is a rough proxy for capability — but not a guarantee. A well-trained 8B model can outperform a poorly trained 70B model on specific tasks. What matters is parameter count *combined with* training data quality, training compute, and alignment. Always benchmark on *your* use case rather than trusting parameter counts alone.

##### Stage 3: Training — The Learning Loop

**Two main task types:**

**Regression** — the label is a continuous number.


| Scenario    | Input Features                                              | Predicted Value            |
| ----------- | ----------------------------------------------------------- | -------------------------- |
| House price | Square footage, zip code, bedrooms, lot size, interest rate | Price ($425,000)           |
| Travel time | Distance, traffic conditions, weather, road type            | Minutes to arrive (23 min) |
| Rainfall    | Temperature, humidity, pressure, cloud cover, wind          | Millimeters of rain (12mm) |


**Classification** — the label is a discrete category.


| Type                           | Example                     | Output                             |
| ------------------------------ | --------------------------- | ---------------------------------- |
| **Binary classification**      | Is this email spam?         | `spam` or `not spam`               |
| **Binary classification**      | Will it rain tomorrow?      | `rain` or `no rain`                |
| **Multi-class classification** | What type of precipitation? | `rain`, `hail`, `snow`, or `sleet` |


The model learns a **decision boundary** (for classification) or a **best-fit curve** (for regression) through the training loop:

1. The model makes a prediction on a training example.
2. A **loss function** measures how wrong the prediction is (cross-entropy for classification, mean squared error for regression).
3. **Backpropagation** computes how much each parameter contributed to the error.
4. **Gradient descent** adjusts the parameters slightly to reduce the error.
5. Repeat millions of times across the dataset.

```
Example: Model predicts 115mm of rain. Actual was 75mm.
Loss = |115 - 75| = 40mm error
→ Model adjusts its parameters so next prediction is closer to 75mm.
→ After seeing all examples, it arrives at the best average prediction.
```

The model converges when it can't reduce the loss much further. This gradual refinement is why large, diverse datasets produce better models — the model has seen more scenarios and refined its understanding across a wider range of situations.

##### Stage 4: Evaluating

We evaluate a trained model by giving it labeled data it has **never seen during training** — we provide only the features and compare the model's predictions to the actual labels.

This tests **generalization**: did the model learn real patterns, or did it just memorize the training set? If it performs well on training data but poorly on held-out data, it has **overfit** — memorized instead of learned.

Depending on evaluation results, you may go back to training: adjust features, add more data, or tune the model's configuration.

##### Stage 5: Inference

Once you're satisfied with evaluation results, the model makes predictions — called **inferences** — on new, unlabeled data in the real world.

```
Inference: Give the model today's temperature, pressure, and humidity
         → It predicts 12mm of rainfall tomorrow
```

This is the model "in production" — doing the job it was trained for.

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

---

### 2. Reinforcement Learning

Reinforcement learning (RL) is fundamentally different from both supervised and unsupervised learning. Instead of learning from a dataset, an RL agent learns by **interacting with an environment** and receiving **rewards or penalties** based on its actions.

```mermaid
flowchart LR
    A["Agent"] -->|"takes action"| B["Environment"]
    B -->|"returns state + reward"| A
    A -->|"updates policy"| A
```



The agent's goal is to learn a **policy** — a strategy that maximizes cumulative reward over time. It discovers the best actions through trial and error, not from labeled examples.

**Key concepts:**


| Concept         | What It Means                                                                |
| --------------- | ---------------------------------------------------------------------------- |
| **Agent**       | The learner / decision-maker                                                 |
| **Environment** | The world the agent interacts with                                           |
| **State**       | The current situation the agent observes                                     |
| **Action**      | What the agent does in response to a state                                   |
| **Reward**      | A numerical signal (positive or negative) indicating how good the action was |
| **Policy**      | The learned strategy: given a state, which action to take                    |


**Real-world examples:**

- **Game playing:** DeepMind's AlphaGo learned to beat the world champion at Go by playing millions of games against itself. Each win = reward, each loss = penalty.
- **Robotics:** Robots learn to walk by trying different motor commands. Falling down = negative reward. Moving forward = positive reward.
- **Recommendation systems:** Suggesting content, observing whether the user clicks (reward) or scrolls past (penalty), and adapting the strategy.
- **LLM alignment (RLHF):** Reinforcement Learning from Human Feedback is used to align LLMs with human preferences — the reward signal comes from human ratings of model outputs.

**How it differs from supervised learning:** In supervised learning, every training example has the "right answer." In RL, the agent only gets a reward signal — it has to figure out *which* of its many actions led to the reward, and how to do better next time. This is called the **credit assignment problem**.

> **Founder's Tip:** RL shines in sequential decision-making problems where the "right answer" isn't known in advance — robotics, game AI, dynamic pricing, autonomous driving. But RL is notoriously hard to train: unstable, sample-inefficient, and sensitive to reward design. For most startup use cases, start with supervised or self-supervised approaches and reach for RL only when the problem truly requires it.

---

### 3. Generative AI

Generative AI is a class of models that **creates new content** from user input. Rather than classifying, predicting, or clustering existing data, these models produce novel outputs — text, images, code, music, video — that didn't exist before.

#### Input-to-Output Types

Generative AI is often described by its input/output modality:


| Model Type                | Input               | Output           | Example                                                |
| ------------------------- | ------------------- | ---------------- | ------------------------------------------------------ |
| **Text-to-text**          | Text prompt         | Text response    | "Summarize this article" → summary paragraph           |
| **Text-to-image**         | Text description    | Image            | "A sunset over mountains, watercolor style" → painting |
| **Text-to-code**          | Natural language    | Source code      | "Write a Python sort function" → working code          |
| **Text-to-speech**        | Text                | Audio            | "Read this paragraph aloud" → spoken audio             |
| **Text-to-video**         | Text description    | Video clip       | "A teddy bear swimming in the ocean" → video           |
| **Speech-to-text**        | Audio               | Transcribed text | Spoken words → written transcript                      |
| **Image-to-text**         | Image               | Description      | Photo of a flamingo → "This is a flamingo"             |
| **Image & text-to-image** | Image + instruction | Modified image   | Photo + "remove the background" → clean cutout         |


#### How Does Generative AI Work?

Generative models learn the **patterns and structure** of their training data, then produce new instances that are statistically similar but novel. You can think of it as:

- A comedian who studies other comedians' timing, structure, and delivery — then writes original jokes in a similar style.
- An artist who studies thousands of impressionist paintings — then creates a new one that "fits" the style without copying any specific work.

The training process typically involves:

1. **Unsupervised / self-supervised pre-training** on massive datasets — the model learns the statistical structure of text, images, or other data.
2. **Supervised fine-tuning** on task-specific data — teaching the model to follow instructions, answer questions, or generate particular types of content.
3. **Reinforcement learning from human feedback (RLHF)** — aligning the model's outputs with human preferences for quality, safety, and helpfulness.

This three-stage pipeline is how modern LLMs like GPT, Llama, and Claude are built. Each stage adds a layer of capability on top of the previous one.

> **Founder's Tip:** Generative AI is advancing rapidly — new modalities (text-to-3D, text-to-music) are emerging constantly. When evaluating opportunities, focus on the *workflow* you're improving rather than the specific model capability. Models are commoditizing fast; the value is in the application layer.

---

### 4. Self-Supervised Learning — The Bridge

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

### 5. Foundation Models



#### What Is a Foundation Model?

A foundation model is a large AI model **pre-trained on broad, diverse data** at massive scale, designed to be **adapted for many downstream tasks** rather than built for a single purpose.

The term was coined by Stanford's Center for Research on Foundation Models (CRFM) in 2021. The key properties:


| Property              | What It Means                                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Scale**             | Trained on terabytes to petabytes of data (text, images, code, audio) using thousands of GPUs                                             |
| **Generality**        | Not designed for one task — the same base model can be adapted for translation, summarization, code generation, question-answering, etc.  |
| **Transfer learning** | Knowledge learned during pre-training transfers to downstream tasks, dramatically reducing the data and compute needed for specialization |
| **Emergence**         | Capabilities appear at scale that weren't explicitly trained for (e.g., in-context learning, chain-of-thought reasoning)                  |


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

### 6. Large Language Models (LLMs)



#### What Makes an LLM Different from Traditional ML?

An LLM is a foundation model specialized in language. But the differences from traditional ML go deeper than just "it's bigger":


| Dimension                | Traditional ML                                                        | Large Language Models                                                               |
| ------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Training objective**   | Task-specific (e.g., "minimize classification error on this dataset") | Universal: "predict the next token" — one objective, applied to all of language     |
| **Data**                 | Curated, labeled datasets (thousands to millions of examples)         | Massive unlabeled text corpora (trillions of tokens from the internet, books, code) |
| **Architecture**         | Varies widely (decision trees, SVMs, CNNs, RNNs...)                   | Almost exclusively the **Transformer** architecture (2017)                          |
| **Input/Output**         | Fixed format (tabular data → number, image → class label)             | Flexible natural language (text → text, covering almost any task)                   |
| **How you "program" it** | Feature engineering + training                                        | **Prompting** — you describe what you want in natural language                      |
| **Generalization**       | Narrow — only does what it was trained for                            | Broad — handles tasks it was never explicitly trained for (zero-shot)               |


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

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where:

- **Q (Query):** "What am I looking for?" — what the current token wants to know.
- **K (Key):** "What do I contain?" — what each other token advertises about itself.
- **V (Value):** "What information do I carry?" — the actual content to retrieve.
- $d_k$ — The dimension of the key vectors (used for scaling to prevent extreme values).

The dot product $QK^T$ measures similarity between the query and each key. Softmax converts these scores into a probability distribution (weights that sum to 1). Those weights are applied to the values to produce a context-aware representation.

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

### 7. Tokens — How Models Actually "Read"



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


| Text Type                   | Tokens per Word (approx.)                                              |
| --------------------------- | ---------------------------------------------------------------------- |
| Simple English prose        | ~1.2–1.3                                                               |
| Technical / medical text    | ~1.5–1.8 (more rare words = more splits)                               |
| Source code                 | ~2.0–3.0+ (symbols, indentation, syntax)                               |
| Non-English languages       | ~1.5–4.0+ (depends on how well the tokenizer represents that language) |
| Emojis / special characters | Often 1–3 tokens *each*                                                |


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







## Glossary


| Term                                     | Definition                                                                                                                                |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Machine Learning**                     | The process of training software (a model) to make predictions or generate content from data, rather than explicitly programming rules.   |
| **Supervised Learning**                  | Training a model on labeled (input, output) pairs so it learns to predict outputs for new inputs.                                         |
| **Classification**                       | A supervised learning task where the model predicts a discrete category (e.g., spam or not spam).                                         |
| **Regression**                           | A supervised learning task where the model predicts a continuous number (e.g., house price, rainfall).                                    |
| **Unsupervised Learning**                | Training a model on unlabeled data to discover hidden structure (clusters, patterns, anomalies).                                          |
| **Clustering**                           | An unsupervised technique that groups similar data points together without predefined categories.                                         |
| **Reinforcement Learning**               | Training an agent to make decisions by interacting with an environment and receiving reward/penalty signals.                              |
| **Policy**                               | In RL, the learned strategy that maps states to actions to maximize cumulative reward.                                                    |
| **Generative AI**                        | A class of models that creates new content (text, images, code, video) from user input.                                                   |
| **Self-Supervised Learning**             | A form of unsupervised learning where the model generates its own labels from the data structure (e.g., predicting masked words).         |
| **Foundation Model**                     | A large model pre-trained on broad data, designed to be adapted for many downstream tasks via fine-tuning or prompting.                   |
| **Large Language Model (LLM)**           | A foundation model specialized in language, trained on massive text corpora to predict the next token.                                    |
| **Transformer**                          | The neural network architecture (2017) that uses self-attention to process all tokens in parallel, enabling modern LLMs.                  |
| **Self-Attention**                       | A mechanism where each token computes relevance scores against all other tokens to build context-aware representations.                   |
| **Token**                                | The atomic sub-word unit an LLM processes. Not a word — common words are single tokens; rare words are split into multiple tokens.        |
| **Tokenizer**                            | The algorithm (e.g., BPE) that converts raw text into a sequence of token IDs from a fixed vocabulary.                                    |
| **BPE (Byte-Pair Encoding)**             | A tokenization algorithm that iteratively merges the most frequent adjacent character pairs to build a sub-word vocabulary.               |
| **Feature**                              | An input variable the model uses to make predictions (e.g., temperature, square footage).                                                 |
| **Label**                                | The correct output value in a supervised learning dataset — the "answer" the model learns to predict.                                     |
| **Parameter**                            | A single numerical value inside the model (weight or bias) that is adjusted during training. "70B model" = 70 billion parameters.         |
| **Hyperparameter**                       | A configuration value set by the engineer *before* training (learning rate, batch size, number of layers). Not learned from data.         |
| **Inference**                            | Using a trained model to make predictions on new, unseen data in production.                                                              |
| **Fine-Tuning**                          | Further training a pre-trained model on a smaller, task-specific dataset to specialize its behavior.                                      |
| **RAG (Retrieval-Augmented Generation)** | A pattern where the model retrieves external documents at inference time to ground its response in specific knowledge.                    |
| **LoRA (Low-Rank Adaptation)**           | A parameter-efficient fine-tuning method that trains small adapter weights while keeping the base model frozen.                           |
| **Context Window**                       | The maximum number of tokens the model can process in a single forward pass (prompt + response combined).                                 |
| **Emergence**                            | Capabilities that appear in large models but are absent in smaller ones, despite no explicit training for those capabilities.             |
| **RLHF**                                 | Reinforcement Learning from Human Feedback — a technique to align model outputs with human preferences using a learned reward model.      |
| **Overfitting**                          | When a model memorizes training data instead of learning generalizable patterns, performing well on training data but poorly on new data. |
| **Loss Function**                        | A mathematical function that measures how wrong a model's prediction is. Training minimizes this function.                                |
| **Gradient Descent**                     | An optimization algorithm that iteratively adjusts model parameters in the direction that reduces the loss.                               |
| **Backpropagation**                      | The algorithm that computes how much each model parameter contributed to the prediction error, enabling gradient descent.                 |


---

## Success Criteria Self-Check

After reading this guide, you should be able to answer:

- *"What is machine learning?"* → Training software to make predictions or generate content from data, rather than explicitly programming rules.
- *"What's the difference between supervised and unsupervised learning?"* → Supervised uses labeled data to learn input→output mappings; unsupervised finds hidden structure in unlabeled data.
- *"What is reinforcement learning?"* → An agent learns by interacting with an environment, receiving rewards/penalties, and optimizing a policy to maximize cumulative reward.
- *"What is generative AI?"* → Models that create new content (text, images, code, video) from user input, trained via self-supervised pre-training + fine-tuning + RLHF.
- *"What are the five stages of supervised learning?"* → Data → Model → Training → Evaluating → Inference.
- *"What is a foundation model?"* → A large model pre-trained on broad data that can be adapted for many tasks, replacing the old "one model per task" approach.
- *"How is an LLM different from traditional ML?"* → Traditional ML has a task-specific objective and fixed I/O. An LLM has a universal objective (predict next token) and flexible natural language I/O, enabling broad generalization.
- *"What's the difference between a token and a word?"* → A token is a sub-word unit the model actually processes. Common words are single tokens; rare/long words get split into multiple tokens. APIs charge per token, not per word.
- *"Why do tokens matter for cost and context limits?"* → Because both are measured in tokens. More tokens = higher cost and less room in the context window.


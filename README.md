# 🍃 AA Assistant — LLM Zoomcamp Final Project

> **A Retrieval-Augmented Generation (RAG) chatbot for Alcoholics Anonymous information**  
> Created by **Marcelo Nieva** for the [DataTalksClub LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp)

---

## 🧠 Project Overview

**AA Assistant** is an intelligent conversational agent designed to provide accurate, empathetic, and trustworthy information about **Alcoholics Anonymous (AA)** to individuals seeking help or guidance with alcohol-related concerns.

The chatbot leverages a **Retrieval-Augmented Generation (RAG)** pipeline built entirely from official AA sources, ensuring that all responses are grounded in verified documentation. This project demonstrates a complete end-to-end LLM application—from data collection and indexing to semantic retrieval, generation, and comprehensive evaluation.

🌐 Multilingual Support: The assistant operates in both Spanish and English, making AA information accessible to a broader audience across different regions and language preferences.

<p align="center">
  <img width="1536" height="600" alt="AA Assistant Chatbot" src="https://github.com/user-attachments/assets/a08ec447-4dab-4e79-a106-bfc731f74b64" />
  <br>

  <em>AA Assistant</em>
</p>

---

## 🎯 Project Goals & LLM Zoomcamp Evaluation Criteria

This project addresses all key evaluation criteria outlined in the LLM Zoomcamp:

| Criterion | Implementation | Location in Code |
|-----------|----------------|------------------|
| **Problem Definition** | Clear use case: providing reliable AA information to people struggling with alcohol | This README, `src/RAG/prompts.py` |
| **Data Collection** | Web scraping from official AA websites (global & Argentina) | `data/final/*.json` |
| **Data Indexing** | Vector embeddings using Jina AI v2 (Spanish/English) + sparse BM25 indexing | `src/ingest.py`, `src/config/db.py` |
| **Retrieval Strategy** | Hybrid search (semantic + lexical) with DBSF and RRF reranking | `src/RAG/main.py`, `src/eval/retrival/` |
| **LLM Integration** | NVIDIA NIM endpoints with three models evaluated | `src/LLM/main.py` |
| **Application Interface** | FastAPI serving custom HTML/CSS/JS interface | `src/server/app.py`, `src/server/templates/` |
| **Monitoring & Evaluation** | Comprehensive evaluation framework for retrieval and generation quality | `src/eval/` directory |
| **Documentation** | Detailed README with setup instructions, architecture, and evaluation results | This file |

---

## 📚 Data Sources

All knowledge in the chatbot comes from **official Alcoholics Anonymous sources**:

### 🌍 Global Resources
- **AA Official Website (Spanish):** [https://www.aa.org/es](https://www.aa.org/es)
- Core AA literature: The Twelve Steps, The Big Book, FAQs

### 🇦🇷 Local Resources  
- **AA Argentina Official Website:** [https://aa.org.ar/](https://aa.org.ar)
- Regional meeting information, local resources, and Argentina-specific guidance

### 📁 Data Structure

```
data/final/
├── FAQS.json                      # Frequently Asked Questions
├── FAQS_IDX.json                  # Indexed FAQs with IDs
├── Ground_Truth.json              # Evaluation ground truth dataset
├── Ground_Truth_IDX.json          # Indexed ground truth
├── answers_gpt_20b.json           # Generated answers from GPT-20B
├── answers_kimi_k2.json           # Generated answers from Kimi K2
└── answers_llama4_scout.json     # Generated answers from Llama 4 Scout
```

---

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                     │
│              (FastAPI + HTML/CSS/JS)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   RAG Pipeline                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Semantic   │  │   Lexical    │  │   Hybrid     │   │
│  │   Search     │  │   Search     │  │   (DBSF/RRF) │   │
│  │  (Jina v2)   │  │   (BM25)     │  │  Reranking   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 Vector Database                         │
│                    (Qdrant)                             │
│  • Dense vectors: Jina Embeddings v2 (768 dims)         │
│  • Sparse vectors: BM25 tokenization                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               LLM Generation Layer                      │
│                  (NVIDIA NIM)                           │
│  • openai/gpt-oss-20b                                   │
│  • moonshotai/kimi-k2-instruct                          │
│  • meta/llama-4-scout-17b-16e-instruct                  │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
src/
├── RAG/                          # Retrieval-Augmented Generation
│   ├── main.py                   # Core RAG orchestration
│   └── prompts.py                # System and user prompts
├── LLM/                          # Language Model interface
│   └── main.py                   # NVIDIA NIM client wrapper
├── eval/                         # Evaluation
│   ├── retrival/                 # Retrieval evaluation
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── llm/                      # Generation evaluation
│   │   ├── generate_answers.py
│   │   └── cosine_similarity/
│   │       └── evaluate.py
│   ├── generate_ground_truth.py
│   └── prompts.py
├── server/                       # Web application
│   ├── app.py                    # FastAPI server
│   └── templates/
│       └── index.html            # Frontend interface
├── config/                       # Configuration management
│   ├── db.py                     # Qdrant database setup
│   ├── envs.py                   # Environment variables
│   ├── paths.py                  # Path constants
│   └── utils.py                  # Utility functions
├── ingest.py                     # Data ingestion pipeline
└── main.py                       # Entry point
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker
- NVIDIA NIM API key

### 🚀 Quick Start

#### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:
```bash
uv --version
```

#### 2. Clone Repository

```bash
git clone https://github.com/marcelonieva7/AA_Bot.git
cd AA_BOT
```

#### 3. Create Virtual Environment

```bash
uv venv
```

#### 4. Install Dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` using locked versions from `uv.lock`.

#### 5. Configure Environment Variables

Create a `.env` file in the project root by copying the provided template:

```bash
cp .env.template .env
```

Then edit `.env` with your configuration:

```env
# Qdrant Vector Database
QDRANT_URL=http://localhost:6333

# NVIDIA NIM API Configuration
NVIDIA_API_KEY=
NVIDIA_URL=
```

**Configuration Details:**

| Variable | Description | Default/Example | Required |
|----------|-------------|-----------------|----------|
| `QDRANT_URL` | Qdrant vector database endpoint | `http://localhost:6333` | ✅ Yes |
| `NVIDIA_API_KEY` | Your NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com/) | `nvapi-xxx...` | ✅ Yes |
| `NVIDIA_URL` | NVIDIA NIM API base URL | `https://integrate.api.nvidia.com/v1` | ✅ Yes |

**Getting Your NVIDIA API Key:**

1. Visit [https://build.nvidia.com/](https://build.nvidia.com/)
2. Sign in or create a free account
3. Navigate to your API keys section
4. Generate a new API key
5. Copy the key to your `.env` file

**Note for Docker Users:**

If deploying with Docker Compose, you can also configure `.env.docker` with the same variables. The Docker setup uses:

```env
QDRANT_URL=http://qdrant:6333  # Note: uses service name instead of localhost
```

#### 6. Initialize Vector Database

**Step 1: Start Qdrant Database**

First, launch the Qdrant vector database using Docker:

```bash
docker run --rm -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/docker_volumes/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

This command:
- Exposes port `6333` for the REST API
- Exposes port `6334` for the gRPC API
- Persists data in `./docker_volumes/qdrant_storage/`
- Runs in the foreground (use `Ctrl+C` to stop)

Verify Qdrant is running:
```bash
curl http://localhost:6333/
```

You should see a JSON response with version information.

---

**Step 2: Ingest and Index Documents**

Once Qdrant is running, populate the vector database:

```bash
uv run -m src.ingest
```

This ingestion pipeline will:
- 📂 Load documents from `data/final/FAQS.json` and related files
- 🧠 Generate dense embeddings using **Jina Embeddings v2** (768 dimensions)
- 🔤 Create sparse BM25 vectors for lexical search
- 💾 Index all vectors in Qdrant with hybrid search capabilities
- ✅ Create collection with optimized indexing parameters

#### 7. Run the Application

```bash
uv run uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000** to access the chatbot interface.

---

## 🐳 Docker Deployment

### Environment Configuration

Use `.env.docker` for Docker-specific settings:

```env
QDRANT_URL=http://qdrant:6333
NVIDIA_API_KEY=
NVIDIA_URL=https://integrate.api.nvidia.com/v1
```

### Using Docker Compose

```bash
docker compose up --build
```

This starts:
- **FastAPI server** on port 8000
- **Qdrant vector database** on port 6333

---


## 🔍 Retrieval Strategy

The chatbot implements a **sophisticated hybrid retrieval system** combining semantic and lexical search:

### Search Types Evaluated

```python
search_types = [
    'semantic',              # Dense vector similarity (Jina v2)
    'lexical',               # Sparse BM25 keyword matching
    ['hybrid', 'DBSF'],      # Hybrid with Distribution-Based Score Fusion
    ['hybrid', 'RRF']        # Hybrid with Reciprocal Rank Fusion
]
```

### Embedding Models

| Model Type | Model Name | Purpose | Dimensions |
|------------|------------|---------|------------|
| **Dense** | `jinaai/jina-embeddings-v2-base-es` | Semantic similarity in Spanish and English | 768 |
| **Sparse** | `Qdrant/bm25` | Lexical keyword matching | Variable |

### Why Hybrid Search?

- **Semantic search** captures conceptual similarity and handles paraphrasing
- **Lexical search** ensures exact term matches (important for AA-specific terminology)
- **Reranking algorithms** (DBSF/RRF) combine both strengths for optimal retrieval

---

## 🤖 LLM Models Evaluated

The project compares three state-of-the-art language models via **NVIDIA NIM** endpoints:

| Model | Identifier | Strengths | Use Case |
|-------|------------|-----------|----------|
| **GPT-20B** | `openai/gpt-oss-20b` | Balanced performance, good Spanish support | General purpose answering |
| **Kimi K2** | `moonshotai/kimi-k2-instruct` | Long context, instruction following | Complex multi-step reasoning |
| **Llama 4 Scout** | `meta/llama-4-scout-17b-16e-instruct` | Efficient, fast inference | Quick responses |

All models are accessed through the [NVIDIA NIM API](https://build.nvidia.com/), enabling consistent evaluation and deployment.

---

## 📊 Evaluation Framework

### 1. Retrieval Evaluation

Located in `src/eval/retrival/`, this module measures retrieval quality using two key metrics:

#### Evaluation Metrics

**Hit Rate (Recall@k)**
- Measures the proportion of queries where at least one relevant document appears in the top-k results
- Formula: `(Number of queries with relevant docs in top-k) / (Total queries)`
- Range: 0.0 to 1.0 (higher is better)

**Mean Reciprocal Rank (MRR)**
- Evaluates how high the first relevant document ranks in the results
- Formula: `Average(1 / rank of first relevant document)`
- Range: 0.0 to 1.0 (higher is better)
- Emphasizes ranking quality—finding relevant docs early is rewarded

#### Running the Evaluation

```bash
uv run -m src.eval.retrival.evaluate
```

This script:
1. Loads ground truth questions with known relevant document IDs
2. Executes each search type (semantic, lexical, hybrid)
3. Compares retrieved document IDs against ground truth
4. Calculates Hit Rate and MRR for each configuration
5. Saves results to `src/eval/retrival/Eval_results.csv`

#### Results

Based on evaluation across **580 test queries** from the ground truth dataset:

| Search Type | Hit Rate | MRR | Interpretation |
|-------------|----------|-----|----------------|
| **Semantic** | **0.9276** | 0.6741 | Finds relevant docs 92.8% of the time, typically ranked 3rd-4th |
| Lexical | 0.7845 | 0.5505 | Baseline BM25 performance—finds docs 78.5% of time |
| Hybrid (DBSF) | 0.9224 | **0.7090** | Best ranking quality—relevant docs appear higher (avg rank ~2) |
| **Hybrid (RRF)** | **0.9379** | 0.6985 | **Best hit rate**—finds relevant docs 93.8% of time |

#### Key Findings

✅ **Winner: Hybrid RRF** for production deployment
- Highest hit rate (93.79%) ensures users almost always get relevant information
- Strong MRR (0.6985) means relevant docs typically appear in top 2-3 positions
- Combines strengths of both semantic understanding and exact term matching

📊 **Best Ranking: Hybrid DBSF**
- Highest MRR (0.7090) means relevant documents rank slightly higher on average
- Nearly as good hit rate (92.24%)
- Excellent choice if ranking position is critical

🎯 **Semantic-Only Performance**
- Strong hit rate (92.76%) shows Jina embeddings work well for Spanish AA content
- Lower MRR suggests relevant docs sometimes appear lower in results
- Good fallback if computational resources are limited

⚠️ **Lexical-Only Limitations**
- Lowest performance (78.45% hit rate) confirms pure keyword matching isn't sufficient
- Struggles with paraphrasing and conceptual questions
- Important as a component but insufficient alone

#### Visualization

```
Hit Rate Comparison (Higher is Better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hybrid (RRF)    ████████████████████ 93.79%
Semantic        ████████████████████ 92.76%
Hybrid (DBSF)   ████████████████████ 92.24%
Lexical         ███████████████      78.45%

MRR Comparison (Higher is Better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hybrid (DBSF)   ████████████████████ 0.7090
Hybrid (RRF)    ███████████████████  0.6985
Semantic        ██████████████████   0.6741
Lexical         ███████████████      0.5505
```

### 2. Generation Evaluation

Located in `src/eval/llm/`, this module includes:

#### a) Cosine Similarity Analysis

Measures semantic overlap between generated answers and ground truth using vector embeddings. This quantitative metric evaluates how closely each model's responses align with reference answers.

**Running the Evaluation:**
```bash
uv run -m src.eval.llm.cosine_similarity.evaluate
```

**Methodology:**
- Embeds both generated answers and ground truth using the same embedding model
- Computes cosine similarity scores (range: -1 to 1, where 1 = identical meaning)
- Analyzes distribution across 580 test question-answer pairs

**Results Summary:**

| Model | Mean Similarity | Median | Std Dev | Min | Max | 25th %ile | 75th %ile |
|-------|----------------|--------|---------|-----|-----|-----------|-----------|
| **Llama 4 Scout** | **0.7756** | **0.7786** | 0.1370 | -0.0223 | 1.0000 | 0.7024 | 0.8626 |
| **Kimi K2** | **0.7425** | 0.7617 | 0.1509 | -0.0435 | 1.0000 | 0.6675 | 0.8411 |
| GPT-20B | 0.6801 | 0.7162 | 0.2075 | -0.1596 | 1.0000 | 0.6182 | 0.8002 |

**Detailed Statistics:**

<details>
<summary><b>GPT-20B (openai/gpt-oss-20b)</b></summary>

```
count  580.000000
mean     0.680088
std      0.207454
min     -0.159631
25%      0.618194
50%      0.716159
75%      0.800209
max      1.000000
```

**Analysis:**
- Lowest mean similarity (0.68) suggests more creative/varied responses
- Highest standard deviation (0.21) indicates inconsistent alignment with ground truth
- Some negative scores show occasional semantic drift
- 75% of responses still achieve >0.62 similarity

</details>

<details>
<summary><b>Llama 4 Scout (meta/llama-4-scout-17b-16e-instruct)</b></summary>

```
count  580.000000
mean     0.775588
std      0.136974
min     -0.022276
25%      0.702364
50%      0.778617
75%      0.862613
max      1.000000
```

**Analysis:**
- **Highest mean similarity (0.78)** — best alignment with reference answers
- Lowest standard deviation (0.14) shows most consistent performance
- Minimum score near 0 (vs. negative for others) indicates fewer outliers
- 75% of responses achieve >0.86 similarity (excellent)
- **Winner for semantic accuracy**

</details>

<details>
<summary><b>Kimi K2 (moonshotai/kimi-k2-instruct)</b></summary>

```
count  580.000000
mean     0.742493
std      0.150890
min     -0.043473
25%      0.667533
50%      0.761724
75%      0.841071
max      1.000000
```

**Analysis:**
- Strong performance (0.74 mean) — second place overall
- Moderate standard deviation (0.15) shows good consistency
- Balanced distribution with median close to mean
- 75% of responses achieve >0.84 similarity
- Good middle ground between accuracy and creativity

</details>

**Key Insights:**

🥇 **Llama 4 Scout emerges as the winner** for semantic accuracy:
- Highest average similarity (77.6%) to ground truth
- Most consistent performance across all queries
- Fewest outliers and semantic drift cases

🥈 **Kimi K2 provides strong balanced performance**:
- Second-best similarity (74.2%)
- Good for complex queries requiring nuanced understanding

⚠️ **GPT-20B shows higher variance**:
- More creative/diverse responses (can be positive or negative depending on use case)
- Less predictable alignment with reference answers
- May provide alternative valid perspectives not captured in ground truth

**Distribution Visualization:**

```
Similarity Score Distribution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Llama 4 Scout:  [████████████████████████] 77.6% avg
                     ↑ Most consistent

Kimi K2:        [██████████████████████  ] 74.2% avg
                     ↑ Balanced performance

GPT-20B:        [███████████████████     ] 68.0% avg
                     ↑ More variance
```

**Recommendation:**
For production deployment, **Llama 4 Scout** is recommended when semantic accuracy to established AA content is prioritized. However, all three models perform above the 0.68 threshold, indicating they all generate semantically relevant responses.

#### b) LLM-as-a-Judge Evaluation

Uses an advanced language model to qualitatively assess answer quality through structured evaluation prompts. This approach provides nuanced assessment beyond pure numerical metrics.

**Running the Evaluation:**
```bash
uv run -m src.eval.llm.llm_judge.evaluate
```

**Methodology:**

Two complementary evaluation perspectives are used to assess each model's responses across 580 test cases:

**Prompt 1: Answer-to-Answer Comparison**
- Compares generated answer against a reference ground truth answer
- Evaluates semantic preservation and information completeness
- Stricter evaluation—checks if the model maintains factual accuracy

```python
"Compare the generated answer to the original reference answer 
and classify relevance as: NON_RELEVANT | PARTLY_RELEVANT | RELEVANT"
```

**Prompt 2: Question-to-Answer Alignment**
- Evaluates how well the generated answer addresses the original question
- Focuses on user satisfaction and practical utility
- More lenient—allows for valid alternative phrasings and approaches

```python
"Evaluate how well the generated answer responds to the question
and classify relevance as: NON_RELEVANT | PARTLY_RELEVANT | RELEVANT"
```

---

**Results Summary:**

| Model | Prompt Type | Relevant | Partly Relevant | Non-Relevant | Success Rate* |
|-------|-------------|----------|-----------------|--------------|---------------|
| **Llama 4 Scout** | Answer Comparison (P1) | 274 (47.2%) | 305 (52.6%) | 1 (0.2%) | **99.8%** |
| **Llama 4 Scout** | Question Alignment (P2) | **501 (86.4%)** | 77 (13.3%) | 2 (0.3%) | **99.7%** |
| **Kimi K2** | Answer Comparison (P1) | 259 (44.7%) | 313 (54.0%) | 8 (1.4%) | 98.6% |
| **Kimi K2** | Question Alignment (P2) | 488 (84.1%) | 86 (14.8%) | 6 (1.0%) | 99.0% |
| GPT-20B | Answer Comparison (P1) | 242 (41.7%) | 308 (53.1%) | 30 (5.2%) | 94.8% |
| GPT-20B | Question Alignment (P2) | 488 (84.1%) | 63 (10.9%) | 29 (5.0%) | 95.0% |

*Success Rate = (Relevant + Partly Relevant) / Total

---

**Detailed Results:**

<details>
<summary><b>GPT-20B (openai/gpt-oss-20b)</b></summary>

**Prompt 1 - Answer Comparison:**
```
PARTLY_RELEVANT    308 (53.1%)
RELEVANT           242 (41.7%)
NON_RELEVANT        30 (5.2%)
```

**Prompt 2 - Question Alignment:**
```
RELEVANT           488 (84.1%)
PARTLY_RELEVANT     63 (10.9%)
NON_RELEVANT        29 (5.0%)
```

**Analysis:**
- Shows largest gap between strict (P1) and lenient (P2) evaluation
- 5% non-relevant rate highest among all models—indicates more factual drift
- Strong question-answering capability (84% fully relevant)
- When aligned, provides comprehensive answers
- ⚠️ Higher risk of deviating from reference material

</details>

<details>
<summary><b>Llama 4 Scout (meta/llama-4-scout-17b-16e-instruct)</b></summary>

**Prompt 1 - Answer Comparison:**
```
PARTLY_RELEVANT    305 (52.6%)
RELEVANT           274 (47.2%)
NON_RELEVANT         1 (0.2%)  ← Best
```

**Prompt 2 - Question Alignment:**
```
RELEVANT           501 (86.4%)  ← Best
PARTLY_RELEVANT     77 (13.3%)
NON_RELEVANT         2 (0.3%)
```

**Analysis:**
- **🏆 Winner: Best overall performance**
- Virtually no non-relevant answers (0.2-0.3%)
- Highest "Relevant" score on question alignment (86.4%)
- Excellent balance between accuracy and utility
- Most reliable for production deployment

</details>

<details>
<summary><b>Kimi K2 (moonshotai/kimi-k2-instruct)</b></summary>

**Prompt 1 - Answer Comparison:**
```
PARTLY_RELEVANT    313 (54.0%)
RELEVANT           259 (44.7%)
NON_RELEVANT         8 (1.4%)
```

**Prompt 2 - Question Alignment:**
```
RELEVANT           488 (84.1%)
PARTLY_RELEVANT     86 (14.8%)
NON_RELEVANT         6 (1.0%)
```

**Analysis:**
- Strong performance, second place overall
- Low non-relevant rate (1.0-1.4%)
- Tied with GPT-20B on question alignment (84.1%)
- Slightly more conservative than Llama (more "partly relevant" classifications)
- Good choice for complex, nuanced queries

</details>

---

**Key Insights:**

📊 **Evaluation Method Comparison:**

The two prompts reveal different aspects of model performance:

| Metric | Prompt 1 (Strict) | Prompt 2 (Lenient) | Insight |
|--------|-------------------|-------------------|---------|
| **Average "Relevant"** | 45.5% | 85.5% | Models better at answering questions than matching reference style |
| **Average "Non-Relevant"** | 2.3% | 2.1% | Consistent failure rate across evaluation methods |

🎯 **Performance Ranking:**

**By Accuracy (Prompt 1 - Answer Fidelity):**
1. 🥇 Llama 4 Scout: 99.8% success rate, 0.2% failures
2. 🥈 Kimi K2: 98.6% success rate, 1.4% failures  
3. 🥉 GPT-20B: 94.8% success rate, 5.2% failures

**By Utility (Prompt 2 - Question Answering):**
1. 🥇 Llama 4 Scout: 86.4% fully relevant, 0.3% failures
2. 🥈 Kimi K2 / GPT-20B: 84.1% fully relevant (tied)

💡 **Practical Implications:**

**For Production Use:**
- **Llama 4 Scout** recommended: highest reliability + best question-answering
- Maintains factual accuracy while providing helpful responses
- Lowest risk of hallucination or irrelevant content

**For Specialized Cases:**
- **Kimi K2**: Excellent for long-context queries requiring deep understanding
- **GPT-20B**: Consider when creative rephrasing is valued over strict accuracy

---

**Visual Comparison:**

```
Success Rate (Relevant + Partly Relevant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Answer Comparison (Strict):
Llama 4 Scout  [████████████████████████] 99.8%
Kimi K2        [████████████████████████] 98.6%
GPT-20B        [███████████████████████ ] 94.8%

Question Alignment (Lenient):
Llama 4 Scout  [████████████████████████] 99.7%
Kimi K2        [████████████████████████] 99.0%
GPT-20B        [███████████████████████ ] 95.0%
```

---

**Recommendation:**

Based on combined evaluation (cosine similarity + LLM-as-a-judge), **Llama 4 Scout** emerges as the optimal model for the AA Assistant chatbot:

✅ Highest semantic similarity (77.6%)  
✅ Best LLM-judge scores (86.4% fully relevant)  
✅ Lowest failure rate (0.2-0.3% non-relevant)  
✅ Consistent performance across evaluation methods  
✅ Balanced accuracy and user satisfaction  

This model is currently deployed in production.

---

## 📸 Screenshots

### Main Interface
The chatbot provides a clean, accessible interface for users seeking AA information:
<img width="1338" height="1165" alt="chat bot web interface" src="https://github.com/user-attachments/assets/80b495ca-f574-4ef9-a5eb-e1cdf343a794" />

### Sample Conversation
Example interaction showing empathetic and informative responses:
<img width="1338" height="1189" alt="chat bot web interface, sample conversation" src="https://github.com/user-attachments/assets/39c2333c-09f5-4b26-a551-2291f857439f" />

---

## 🎥 Demo Video

> **Coming Soon**: A video walkthrough demonstrating the chatbot's capabilities, retrieval process, and evaluation results will be added before final project submission.

---

## 📈 Key Results

### Retrieval Performance

Evaluated across **58 ground truth queries** using Hit Rate and Mean Reciprocal Rank (MRR):

| Search Type | Hit Rate | MRR | Performance Summary |
|-------------|----------|-----|---------------------|
| Semantic | 0.9276 | 0.6741 | Strong recall, moderate ranking |
| Lexical | 0.7845 | 0.5505 | Baseline BM25 performance |
| Hybrid (DBSF) | 0.9224 | **0.7090** | **Best ranking quality** |
| **Hybrid (RRF)** | **0.9379** | 0.6985 | **Best overall - highest hit rate** |

**Winner**: **Hybrid search with Reciprocal Rank Fusion (RRF)** achieves the best retrieval performance with 93.79% hit rate, ensuring users almost always receive relevant AA information.

---

### LLM Comparison

Comprehensive evaluation across **580 test cases** using multiple metrics:

#### Semantic Similarity (Cosine)

| Model | Mean Similarity | Median | Consistency (Std Dev) | Ranking |
|-------|----------------|--------|----------------------|---------|
| **Llama 4 Scout** | **0.7756** | **0.7786** | 0.1370 (Best) | 🥇 |
| Kimi K2 | 0.7425 | 0.7617 | 0.1509 | 🥈 |
| GPT-20B | 0.6801 | 0.7162 | 0.2075 | 🥉 |

#### LLM-as-a-Judge Quality Assessment

**Answer Fidelity (vs. Ground Truth):**

| Model | Relevant | Partly Relevant | Non-Relevant | Success Rate |
|-------|----------|-----------------|--------------|--------------|
| **Llama 4 Scout** | 47.2% | 52.6% | **0.2%** ✨ | **99.8%** |
| Kimi K2 | 44.7% | 54.0% | 1.4% | 98.6% |
| GPT-20B | 41.7% | 53.1% | 5.2% | 94.8% |

**Question-Answering Quality:**

| Model | Relevant | Partly Relevant | Non-Relevant | User Satisfaction |
|-------|----------|-----------------|--------------|-------------------|
| **Llama 4 Scout** | **86.4%** ✨ | 13.3% | 0.3% | **99.7%** |
| Kimi K2 | 84.1% | 14.8% | 1.0% | 99.0% |
| GPT-20B | 84.1% | 10.9% | 5.0% | 95.0% |

---

### Overall Winner: **Llama 4 Scout** 🏆

**meta/llama-4-scout-17b-16e-instruct** is deployed in production based on:

✅ **Highest semantic accuracy** (77.6% cosine similarity)  
✅ **Best consistency** (lowest standard deviation)  
✅ **Exceptional reliability** (99.8% success rate)  
✅ **Top question-answering** (86.4% fully relevant responses)  
✅ **Lowest failure rate** (only 0.2-0.3% non-relevant answers)  

**Model Characteristics:**

| Model | Strengths | Best Use Case | Production Status |
|-------|-----------|---------------|-------------------|
| **Llama 4 Scout** | Accuracy, consistency, reliability | General AA information queries | ✅ **Deployed** |
| Kimi K2 | Long context, nuanced understanding | Complex multi-part questions | Available |
| GPT-20B | Creative rephrasing, diversity | Alternative perspectives | Available |

---

**Configuration:**

The production deployment uses:
- **Retrieval**: Hybrid RRF (93.79% hit rate)
- **Generation**: Llama 4 Scout (86.4% relevance)
- **Combined**: Provides accurate, empathetic, and trustworthy AA information

## 💡 Future Improvements

- [ ] **Conversation Memory**: Implement persistent chat history
- [ ] **Regional Meeting Finder**: Integrate location-based AA meeting search
- [ ] **Human Feedback Loop**: Collect user ratings to improve responses
- [ ] **Fine-tuned Model**: Train a specialized model on AA literature

---

## 🤝 Contributing

While this is a personal project for the LLM Zoomcamp, feedback and suggestions are welcome! Please open an issue or reach out directly.

---

## 👤 Author

**Marcelo Nieva**  
Final Project for DataTalksClub LLM Zoomcamp

- 📧 Email: [marcelonieva7@gmail.com](mailto:marcelonieva7@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/marcelo-nieva](https://www.linkedin.com/in/marcelo-nieva/)

---

## 📄 License

This project is released under the **MIT License**

---

## ⚠️ Important Disclaimer

**This chatbot is not affiliated with or endorsed by Alcoholics Anonymous.**

This is an educational project built to demonstrate RAG systems and improve access to publicly available AA information. It should **never replace professional medical advice, therapy, or in-person AA meetings**.

If you or someone you know is struggling with alcohol use:
- 🌐 Visit the official AA website: [aa.org](https://www.aa.org)
- 📞 Contact a local AA chapter
- 🏥 Seek professional medical help

**In case of emergency, call your local emergency services immediately.**

---

## 🙏 Acknowledgments

- **DataTalksClub** for the excellent LLM Zoomcamp course
- **Alcoholics Anonymous** for their invaluable resources and decades of helping people
- **NVIDIA** for providing accessible LLM inference via NIM
- **Qdrant** for their powerful vector search engine
- The open-source community for tools like FastAPI, FastEmbed, and uv

---

<p align="center">
  <strong>Built with ❤️ to help people find information about recovery</strong>
</p>

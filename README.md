
# DAT401 Evidence-Grounded Research Assistant

A research assistant designed to retrieve, rank, and generate evidence-grounded answers from a curated academic research corpus.

## 1. System Architecture

The application implements Configuration D:

BM25 + BGE Dense Retrieval
→ Reciprocal Rank Fusion (RRF)
→ Cross-Encoder Reranking
→ Evidence-Grounded Answer Generation through OpenRouter.

The system retrieves relevant research passages and uses them as evidence when generating answers.

## 2. Project Structure

```text
LULC_Research_Assistant/
│
├── app.py
├── config.py
├── generation.py
├── retrieval.py
├── indexing.py
├── requirements.txt
├── README.md
│
├── data/
│   └── section_aware_chunks.jsonl
│
├── artifacts/
│   ├── bm25.pkl
│   ├── chunks.pkl
│   ├── dense.faiss
│   └── index_metadata.json
│
└── .streamlit/
    └── secrets.toml
```

## 3. Technologies Used

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Keyword retrieval | BM25 |
| Dense retrieval | BAAI/bge-small-en-v1.5 |
| Vector search | FAISS |
| Hybrid fusion | Reciprocal Rank Fusion |
| Reranking | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Answer generation | OpenRouter API |
| Backend language | Python |

## 4. Retrieval Configuration

The application uses:

- BM25 initial candidates: 20
- Dense retrieval candidates: 20
- Reciprocal Rank Fusion constant: 60
- Cross-Encoder reranking
- Final evidence passages: 5

The existing BM25 and FAISS indexes are loaded from the artifacts directory.

## 5. Language Model Integration

The language model is hosted through OpenRouter and accessed using an OpenAI-compatible API endpoint.

The application sends the retrieved research passages together with the user's question to the hosted language model.

The generation process uses:
- Temperature: 0
- Maximum output tokens: 500
- Evidence-grounded instructions
- Article and page citation requirements
- Abstention when evidence is insufficient

The deployed model identifier is configured in `config.py`.

Note: The deployed language model may differ from the Qwen2.5-3B-Instruct model used in the original experimental evaluation. Deployment results should therefore be distinguished from the original experimental results.

## 6. Local Installation

### Step 1: Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure OpenRouter

Create `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "your_api_key"
```

Never upload this file to a public repository.

### Step 4: Verify project artifacts

Ensure the following files exist:

- `artifacts/bm25.pkl`
- `artifacts/chunks.pkl`
- `artifacts/dense.faiss`

Also ensure the embedding and reranking models can be downloaded by the application.

### Step 5: Run the application

```bash
streamlit run app.py
```

## 7. Deployment

The application can be deployed using Streamlit Community Cloud.

Deployment requirements:

1. Push the application source code to GitHub.
2. Configure the required dependencies.
3. Provide the OpenRouter API key through Streamlit secrets.
4. Make the required retrieval artifacts accessible to the application.
5. Deploy the Streamlit application.
6. Test retrieval, answer generation, evidence display, and citations.

The hosting environment requires sufficient memory and storage for the retrieval models and index artifacts.

## 8. Evaluation and Limitations

- The application displays retrieved evidence for inspection.
- Citation traceability checks article and page metadata.
- Citation traceability does not establish semantic claim-level correctness.
- Generated answers may contain unsupported claims.
- Important answers should be checked against the original research papers.
- The deployed language model may differ from the model used during the original experimental evaluation.
- Experimental configurations A, B, and C remain separate from the deployed Configuration D.

## 9. Academic Integrity

The project uses an external hosted language model for answer generation. The model provider, model identifier, deployment configuration, and any differences from the experimental setup should be disclosed in the final report.

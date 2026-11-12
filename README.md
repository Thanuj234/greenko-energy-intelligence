# Greenko Energy Intelligence – AI Operations Assistant

A responsive, document-grounded Retrieval-Augmented Generation (RAG) prototype developed for the Greenko candidate case study.

The application helps renewable-energy operations teams retrieve relevant information from operational documents, generate evidence-based responses, and view source citations.

## 1. Problem Statement

Renewable-energy operations teams work with technical documents covering equipment maintenance, operational procedures, safety requirements, and grid operations.

Finding relevant information in these documents can take time.

Greenko Energy Intelligence addresses this problem through a document-grounded question-answering assistant that retrieves relevant information from the supplied operational documents and presents answers with supporting evidence.

## 2. Features

- Loads four supplied Markdown operational documents.
- Word-based text chunking with source metadata.
- Sentence Transformer embeddings when the required dependencies are available.
- TF-IDF sparse retrieval as a fallback.
- Cosine similarity-based document retrieval.
- Evidence filtering using a configurable retrieval threshold.
- Optional Groq LLM integration for answer generation.
- Extractive fallback when an LLM API key is unavailable.
- Source citations and retrieved evidence.
- Unsupported-question handling through retrieval threshold checks.
- Heuristic grounding overlap check.
- Evaluation endpoint using `data/questions.csv`.
- Responsive HTML, CSS, and JavaScript frontend.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Text processing | Python |
| Embedding option | Sentence Transformers |
| Retrieval fallback | Scikit-learn TF-IDF |
| Similarity | Cosine similarity |
| LLM integration | Groq API (optional) |
| Data format | Markdown, CSV |
| Deployment | Railway |
| Version control | Git, GitHub |

## 4. RAG Workflow

1. **Document ingestion:** Load the four operational Markdown documents.
2. **Text chunking:** Divide document content into smaller overlapping text chunks.
3. **Embedding and representation:** Generate Sentence Transformer embeddings when available; otherwise use TF-IDF representations.
4. **Retrieval:** Compare the user query against indexed chunks using cosine similarity.
5. **Evidence filtering:** Retain relevant chunks that meet the configured retrieval threshold.
6. **Answer generation:** Generate a response using the optional Groq LLM or extract relevant source text through the extractive fallback.
7. **Citation generation:** Display the source documents supporting the response.
8. **Unsupported-question handling:** Abstain when sufficient supporting evidence is unavailable.

## 5. Project Structure

```text
greenko_energy_intelligence/
│
├── app.py
├── rag_engine.py
├── requirements.txt
├── Procfile
├── runtime.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── solar_inverter_sop.md
│   ├── wind_turbine_maintenance.md
│   ├── hydro_safety_brief.md
│   ├── grid_operations_note.md
│   └── questions.csv
│
├── static/
│   └── index.html
│
├── README.md
└── PROJECT_EXPLANATION.txt
```

## 6. Run Locally (Windows PowerShell)

### Prerequisites

- Python 3.11
- VS Code
- Git (optional)

### Installation

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the application:

```powershell
python -m uvicorn app:app --reload
```

Open the application:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

## 7. Optional Groq LLM Integration

The application supports optional Groq-based LLM answer generation.

To configure it:

1. Create an account at https://console.groq.com/.
2. Generate an API key.
3. Copy `.env.example` to `.env`.
4. Add your API key and model configuration.

Example:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The application loads local environment variables through `python-dotenv`.

**Security:** Never commit `.env` or expose API keys in GitHub.

Without a configured Groq API key, the application uses its extractive answer-generation fallback.

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Opens the frontend |
| GET | `/api/health` | Application and model status |
| GET | `/api/documents` | Lists indexed documents |
| POST | `/api/run` | Processes a user question |
| GET | `/api/evaluate` | Runs the labelled evaluation questions |

Example request:

```json
{
  "text": "What should be checked when an inverter exceeds 75C?",
  "top_k": 3
}
```

## 9. Evaluation Results

The project uses an expanded evaluation dataset containing 10 labelled questions covering all four operational documents.

| Document | Questions | Correct source retrievals |
|---|---:|---:|
| Solar Inverter SOP | 3 | 3 |
| Wind Turbine Maintenance | 3 | 3 |
| Hydro Safety Brief | 2 | 2 |
| Grid Operations Note | 2 | 2 |
| **Total** | **10** | **10** |

**Observed source hit rate: 100% (10/10).**

The source hit rate measures whether the expected source document was retrieved for each labelled question.

This is a small prototype evaluation, not a statistically reliable production benchmark or a measure of complete answer accuracy.

Additional manual tests verified:
- Relevant source retrieval and citation.
- Filtering of irrelevant evidence.
- Abstention when the supplied documents do not contain sufficient information.

## 10. Deployment to Railway

1. Push the project to a GitHub repository.
2. Create a Railway account at https://railway.com/.
3. Create a new project.
4. Select Deploy from GitHub Repo.
5. Connect the project repository.
6. Configure the start command if required:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

7. Add environment variables under Railway service Variables if using Groq.
8. Deploy the application.
9. Open the generated public domain and test the application.

Verify the following endpoints after deployment:

- `/`
- `/api/health`
- `/api/documents`
- `/api/evaluate`
- `/api/run` (using a POST request)

## 11. Limitations and Future Improvements

### Current limitations

- The knowledge base contains only four short operational documents.
- The expanded evaluation dataset contains only 10 labelled questions.
- The current Windows environment uses sparse TF-IDF retrieval because Sentence Transformer dependencies could not be loaded under the local Windows Application Control policy.
- TF-IDF is lexical retrieval and does not provide dense semantic embeddings.
- The Groq LLM is optional and is not configured in the current local environment.
- The extractive fallback is used when an LLM is unavailable.
- The grounding overlap score is a heuristic, not a formal hallucination detector.
- The retrieval threshold has not been calibrated using a large validation dataset.
- The application is not connected to SCADA, CMMS, live plant telemetry, or operational control systems.

### Future improvements

- Expand the knowledge base with validated operational documents.
- Evaluate retrieval using a larger and more diverse question dataset.
- Configure and validate dense semantic embeddings.
- Improve answer-grounding and hallucination evaluation.
- Add user authentication and role-based access control.
- Introduce audit logs and monitoring.
- Integrate approved enterprise operational systems where authorized.
- Add human approval workflows for safety-critical information.

## 12. Safety Disclaimer

This prototype is intended for document retrieval and informational assistance.

It must not replace approved operational procedures, permit-to-work requirements, lockout-tagout processes, control-room authorization, or qualified engineering judgment.

Any safety-critical action must be verified against approved procedures and authorized by the responsible personnel.

---

**Project:** Greenko Energy Intelligence – AI Operations Assistant  
**Purpose:** Document-grounded question answering for renewable-energy operations  
**Implementation:** Python, FastAPI, RAG, TF-IDF, optional LLM integration
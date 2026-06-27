# Personal-Study-Assistant
RAG-based study assistant built from scratch. Upload PDFs and get grounded answers, flashcards, and quizzes. Structure-first chunking, persistent ChromaDB embeddings, Pydantic structured outputs, Gemini via LangChain.

Built to learn how production AI retrieval systems actually work, not just how to call an API.
Upload a PDF -> ask a question -> get a grounded answer with section citations, flashcards, and a quiz.

## Demo: 
<img width="2038" height="1222" alt="Image 2026-06-27 at 12 28 AM" src="https://github.com/user-attachments/assets/ef526b91-0feb-4e26-939f-d00bf0f1b077" />
<img width="2038" height="1222"" alt="Image 2026-06-27 at 12 31 AM" src="https://github.com/user-attachments/assets/9034ca8c-ee3d-4464-b074-db6db6875041" />
<img width="2038" height="1222" alt="Image 2026-06-27 at 12 30 AM" src="https://github.com/user-attachments/assets/2ad377a5-064e-46ba-9528-605d891b7e6c" />


## Tech Stack
| Layer | Tool | Why |
|---|---|---|
| LLM calls | LangChain `ChatGoogleGenerativeAI` | Model-agnostic — swap provider in one line |
| LLM model | `gemini-2.0-flash` | Fast, capable, free tier available |
| PDF extraction | `pypdf` | Simple, reliable for text-based PDFs |
| Chunking | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | Structure-first hybrid |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Free, runs locally, no API cost |
| Vector DB | `ChromaDB` `PersistentClient` | Local persistence, no account needed |
| Structured output | Pydantic + `.with_structured_output()` | Type-safe, zero JSON parsing |
| UI | Streamlit | Demo-ready in minimal code |
| Env management | `python-dotenv` | Keep API keys out of code |

## Design Decisions

### Structure-first chunking over character splitting
Character splitting is blind — it cuts mid-sentence and merges unrelated concepts based purely on a character count. smart_split() splits by document headings first (respecting the structure the author put there), then applies a size-based fallback only for oversized sections. Each chunk maps to one coherent idea, which means cosine similarity search finds the right section instead of a fragment that happens to contain a keyword.

### ChromaDB PersistentClient + MD5 hash check
chromadb.Client() is in-memory — embeddings die when the script ends, forcing a full re-embed on every run. PersistentClient writes to disk and survives restarts. On top of that, an MD5 hash of the PDF path is used as the collection name: if the collection already exists on startup, embedding is skipped entirely. A 50-page PDF goes from a 30-second cold start to an instant load on every subsequent run.

### Pydantic schemas + .with_structured_output() over prompt-based JSON
Asking the LLM to "return only JSON" in a prompt is ~85% reliable — models add markdown fences, hallucinate field names, and fail silently. Binding a Pydantic schema with .with_structured_output() constrains the model to match the schema exactly (~99% reliable) and returns a typed Python object. No json.loads(), no strip(), no try/except. result.cards[0].front just works.

### LangChain for LLM calls only — not for retrieval logic
LangChain's ChatGoogleGenerativeAI wrapper gives a unified .invoke() interface across every model provider. Switching from Gemini to GPT-4 or Claude is one line. But LangChain chains and retrievers wrap too much logic in abstraction — retrieval, chunking, and prompt construction are plain Python so every step is explicit, debuggable, and yours to reason about.

### Standard RAG over a RAG Agent
This assistant has one job i.e. answer questions from your notes. That's a single, well-defined retrieval task. Agents add non-determinism, multiple LLM calls, and debugging complexity with no benefit for this use case. The fixed pipeline is faster, cheaper, and easy to explain step by step in an interview.

## Setup
1. Clone and create a virtual environment
    git clone https://github.com/YOUR_USERNAME/study-assistant.git
    cd study-assistant
    python -m venv venv
    source venv/bin/activate        # Mac/Linux
    # venv\Scripts\activate         # Windows

2. Install dependencies
     pip install -r requirements.txt

4. Set up your API key
   Get your free key at: aistudio.google.com/apikey
   cp .env.example .env
  #### Add your key to .env:
  #### GOOGLE_API_KEY=your_key_here

4. Drop a PDF into the data/ folder
   cp your_notes.pdf data/

5. Run with Streamlit UI
    streamlit run src/app.py

The first run embeds your PDF and writes to chroma_db/. Every subsequent run skips embedding entirely and loads instantly.


## Project Structure

study-assistant/
├── .env                     ← GOOGLE_API_KEY (never commit this)
├── .env.example             ← safe to commit, shows required keys
├── .gitignore               ← includes .env and chroma_db/
├── requirements.txt
├── main.py                  ← terminal interface for testing
├── data/                    ← drop your PDFs here
├── chroma_db/               ← auto-created, persists embeddings
└── src/
    ├── llm.py               ← LangChain LLM setup (one swap point)
    ├── outputs.py           ← Pydantic schemas + structured LLMs
    ├── loader.py            ← PDF extraction + smart_split()
    ├── vectorstore.py       ← ChromaDB embed + persist + retrieve
    ├── rag.py               ← full RAG pipeline
    └── app.py               ← Streamlit UI

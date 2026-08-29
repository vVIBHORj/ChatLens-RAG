# 📱 ChatLens — Corrective-RAG Chat Pattern Analyzer

> Ask semantic questions about your WhatsApp conversations and surface behavioral patterns — powered by a local, self-correcting RAG pipeline. Fully offline, Hinglish-aware, zero API costs.

---

## ✨ What this is

ChatLens parses a raw WhatsApp `.txt` export and turns it into:

1. **A semantic Q&A interface** — ask natural-language questions about the conversation ("How did our chats change after March?") and get answers grounded in retrieved excerpts, not hallucinated guesses.
2. **A behavioral pattern dashboard** — sentiment trends, response-time trends, and engagement balance over time, computed with real statistics (not an LLM inventing a score).

Everything runs **100% locally** via [Ollama](https://ollama.com) — no data ever leaves your machine, no API keys, no per-token cost.

> ⚠️ **This is a pattern-analysis tool, not a psychological or relationship diagnostic.** It surfaces descriptive statistics (e.g. "average reply time increased over time"), not judgments about character or trust. Only analyze chats you have the right to — your own conversations, or with explicit consent from everyone involved.

## 🧠 Why this project is interesting

Most "chat with your PDF" RAG tutorials stop at naive retrieve-then-generate. This project instead implements:

- **Corrective RAG (CRAG)** via LangGraph — retrieved documents are graded for relevance by the LLM itself; if nothing relevant comes back, the query is automatically rewritten and retried (bounded, to avoid infinite loops).
- **Groundedness / hallucination checking** — after generating an answer, a second LLM pass verifies the answer is actually supported by the retrieved context, and flags it if not.
- **Hinglish-aware sentiment analysis** — instead of a rule-based English lexicon (which silently fails on code-mixed Hindi-English text), sentiment is classified by batched, structured-JSON calls to a local LLM that actually understands code-switched slang and tone.
- **Deterministic analytics, stochastic reasoning kept separate** — trends, averages, and engagement stats are computed with plain pandas, never guessed by an LLM. The LLM is used only where it's actually good: open-ended reasoning over retrieved text.

## 🏗️ Architecture

```
WhatsApp .txt export
        │
        ▼
   parser.py          → structured (timestamp, sender, message) records
        │
        ▼
   enrich.py           → sentiment (llama3.2, Hinglish-aware, batched),
        │                 response time, question detection
        ├──────────────────────────────┐
        ▼                              ▼
   vectorstore.py                analytics.py
   (daily chunks →                (sender stats, weekly trends,
    mxbai-embed-large →            engagement balance — pure
    Chroma vector DB)              pandas, no LLM guessing)
        │
        ▼
   rag_graph.py (LangGraph — Corrective RAG)
   retrieve → grade_documents → [relevant?] → generate → check_grounded
                              → [not relevant?] → rewrite_query → retrieve
        │
        ▼
   app.py (Streamlit UI)
   Tab 1: Ask Questions   |   Tab 2: Pattern Dashboard
```

## 🛠️ Tech stack

| Layer | Tool |
|---|---|
| LLM (reasoning, grading, sentiment) | [Ollama](https://ollama.com) + `llama3.2` |
| Embeddings | `mxbai-embed-large` (via Ollama) |
| Orchestration | LangChain + LangGraph |
| Vector store | ChromaDB (local, persisted to disk) |
| Analytics | pandas |
| UI | Streamlit + Plotly |

## 🚀 Getting started

### 1. Install Ollama and pull the models
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### 2. Clone and install dependencies
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### 3. Export your WhatsApp chat
WhatsApp → open chat → tap contact/group name → **Export Chat** → **Without Media** → save the `.txt` file.

### 4. Run it
```bash
streamlit run app.py
```
Upload your `.txt` export in the browser tab that opens.

## 📂 Project structure

```
.
├── app.py                       # Streamlit UI — Q&A tab + pattern dashboard
├── parser.py                    # WhatsApp .txt export → structured messages
├── enrich.py                    # Sentiment (LLM, Hinglish-aware) + behavioral features
├── llm_sentiment.py              # Batched llama3.2 sentiment classification
├── enrich_vader_fallback.py      # Original VADER-based sentiment (English-only, faster)
├── vectorstore.py                # Daily chunking + mxbai-embed-large + Chroma
├── rag_graph.py                  # LangGraph Corrective-RAG pipeline
├── analytics.py                  # Deterministic pattern statistics (pandas)
├── sample_chat.txt               # Synthetic example chat for testing
└── requirements.txt
```

## 🧪 Testing individual components

```bash
python parser.py sample_chat.txt        # test parsing
python enrich.py                        # test sentiment enrichment
python vectorstore.py                   # test daily chunking
python analytics.py                     # test pattern statistics
python rag_graph.py "your question"     # ask a question (build vectorstore first)
```

## 🗺️ Roadmap / ideas for extending

- [ ] Multi-turn conversation memory in the Q&A tab (LangGraph checkpointing)
- [ ] GraphRAG: extract entities/topics per day for multi-hop questions
- [ ] Evaluation harness (e.g. RAGAS) to measure retrieval precision / answer faithfulness
- [ ] Swap in a dedicated multilingual/Hinglish transformer sentiment model for comparison
- [ ] Support for Telegram/Signal/iMessage exports

## 📄 License

MIT — free to use, modify, and build on.

## ⚖️ Disclaimer

This tool is intended for personal reflection and educational/portfolio purposes. It does not provide clinical, psychological, or relationship advice. Always obtain consent before analyzing conversations involving other people.

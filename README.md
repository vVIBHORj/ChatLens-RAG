# 📱 ChatLens — Corrective-RAG Chat Pattern Analyzer(UNDER CONSTRUCTION)

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





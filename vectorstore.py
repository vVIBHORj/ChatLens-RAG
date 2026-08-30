"""
vectorstore.py
Groups enriched messages into daily conversation "chunks" (single messages
lack context for semantic search) and embeds them into a local Chroma vector
store using mxbai-embed-large via Ollama.

Requires a running local Ollama instance:
    ollama pull mxbai-embed-large
"""

import shutil
from pathlib import Path
from typing import List
from collections import defaultdict

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from enrich import EnrichedMessage

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "whatsapp_chat"


def build_daily_chunks(enriched: List[EnrichedMessage]) -> List[Document]:
    """Group messages by calendar day into a single Document per day.

    Each Document's page_content is a readable transcript of that day's
    conversation; metadata carries aggregate stats used for filtering /
    for the analytics layer to cross-reference later.
    """
    by_day = defaultdict(list)
    for m in enriched:
        if not m.timestamp:
            continue
        day_key = m.timestamp.date().isoformat()
        by_day[day_key].append(m)

    documents = []
    for day, msgs in sorted(by_day.items()):
        lines = []
        sentiments = []
        senders = set()
        for m in msgs:
            time_str = m.timestamp.strftime("%I:%M %p")
            lines.append(f"[{time_str}] {m.sender}: {m.message}")
            sentiments.append(m.sentiment_compound)
            if m.sender:
                senders.add(m.sender)

        transcript = "\n".join(lines)
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        documents.append(
            Document(
                page_content=transcript,
                metadata={
                    "date": day,
                    "num_messages": len(msgs),
                    "avg_sentiment": round(avg_sentiment, 4),
                    "participants": ", ".join(sorted(senders)),
                },
            )
        )

    return documents


def build_vectorstore(
    documents: List[Document],
    persist_dir: str = PERSIST_DIR,
    reset: bool = True,
) -> Chroma:
    """Embed documents into a Chroma collection.

    reset=True (default) wipes any existing persisted collection first. This
    matters because each app run processes a single chat export -- without a
    reset, re-uploading a new/different export would silently accumulate
    into the same collection as a previous run, so RAG answers could be
    grounded in the wrong conversation's chunks.
    """
    if reset and Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(persist_dir: str = PERSIST_DIR) -> Chroma:
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


if __name__ == "__main__":
    from parser import parse_whatsapp_export
    from enrich import enrich_messages

    msgs = parse_whatsapp_export("sample_chat.txt")
    enriched = enrich_messages(msgs)
    docs = build_daily_chunks(enriched)

    print(f"Built {len(docs)} daily chunks\n")
    for d in docs:
        print(f"--- {d.metadata['date']} (avg sentiment: {d.metadata['avg_sentiment']}) ---")
        print(d.page_content)
        print()

    print("NOTE: building the actual Chroma vectorstore requires a running")
    print("Ollama instance with 'mxbai-embed-large' pulled. Run this on your")
    print("own machine, not in a sandbox without Ollama.")

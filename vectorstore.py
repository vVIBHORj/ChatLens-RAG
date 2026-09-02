"""
vectorstore.py

Builds and loads the Chroma vector database for ChatLens-RAG.

Architecture:

    EnrichedMessage
          ↓
    Conversation Episodes
          ↓
    Overlapping Conversation Chunks
          ↓
    Qwen3 Embeddings
          ↓
    Chroma


Chroma is used as the semantic retrieval layer.

SQLite remains the source of truth for the original messages
and is used later for context expansion.
"""

# =====================================================================
# Imports
# =====================================================================

from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from enrich import EnrichedMessage


# =====================================================================
# Configuration
# =====================================================================

OLLAMA_BASE_URL = "http://127.0.0.1:11434"

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

PERSIST_DIR = "./chroma_db"

COLLECTION_NAME = "whatsapp_chat"

# Maximum size of a single conversation chunk.
MAX_CHARS_PER_CHUNK = 6000

# Maximum messages in a chunk.
MAX_MESSAGES_PER_CHUNK = 50

# Number of messages shared between consecutive chunks.
CHUNK_OVERLAP_MESSAGES = 10

# A gap this large starts a new conversation episode.
EPISODE_GAP_MINUTES = 120


# =====================================================================
# Embeddings
# =====================================================================

def get_embeddings():
    """
    Create the local Ollama embedding model.
    """

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


# =====================================================================
# Conversation formatting
# =====================================================================

def format_message(
    message: EnrichedMessage,
) -> str:
    """
    Convert one enriched message into a searchable transcript line.
    """

    if message.timestamp is not None:

        timestamp = message.timestamp.strftime(
            "%d/%m/%Y %I:%M %p"
        )

    else:

        timestamp = ""

    sender = message.sender or "Unknown"

    text = message.message or ""

    return (
        f"[ID {message.id}] "
        f"[{timestamp}] "
        f"{sender}: "
        f"{text}"
    )


# =====================================================================
# Build conversation episodes
# =====================================================================

def build_conversation_episodes(
    messages: List[EnrichedMessage],
):
    """
    Group messages into conversation episodes.

    A new episode begins when the gap between consecutive messages
    is >= EPISODE_GAP_MINUTES.

    Example:

        20:00
        20:02
        20:10
        22:15  <-- new episode
    """

    valid_messages = [
        message
        for message in messages
        if message.timestamp is not None
        and message.message_type in ("text", "media")
    ]

    if not valid_messages:

        return []

    episodes = []

    current_episode = [
        valid_messages[0]
    ]

    for message in valid_messages[1:]:

        previous = current_episode[-1]

        gap_minutes = (
            message.timestamp - previous.timestamp
        ).total_seconds() / 60.0

        if gap_minutes >= EPISODE_GAP_MINUTES:

            episodes.append(
                current_episode
            )

            current_episode = [
                message
            ]

        else:

            current_episode.append(
                message
            )

    if current_episode:

        episodes.append(
            current_episode
        )

    return episodes


# =====================================================================
# Build conversation chunks
# =====================================================================

def build_conversation_chunks(
    messages: List[EnrichedMessage],
) -> List[Document]:
    """
    Convert enriched WhatsApp messages into overlapping
    conversation-aware Chroma documents.

    Important:

        We do NOT split the conversation purely by calendar day.

        Instead, we first identify conversation episodes based on
        inactivity gaps and then create overlapping chunks.

    This preserves local conversational context.
    """

    episodes = build_conversation_episodes(
        messages
    )

    documents = []

    global_chunk_index = 0

    for episode_id, episode in enumerate(
        episodes
    ):

        if not episode:

            continue

        start_index = 0

        chunk_index = 0

        while start_index < len(episode):

            chunk_messages = []

            current_chars = 0

            current_index = start_index

            # ---------------------------------------------------------
            # Build one chunk
            # ---------------------------------------------------------

            while (
                current_index < len(episode)
                and len(chunk_messages)
                < MAX_MESSAGES_PER_CHUNK
            ):

                message = episode[
                    current_index
                ]

                formatted = format_message(
                    message
                )

                additional_chars = len(
                    formatted
                )

                # Don't add another message if doing so would exceed
                # the maximum size, unless the chunk is currently empty.
                if (
                    chunk_messages
                    and
                    current_chars + additional_chars
                    > MAX_CHARS_PER_CHUNK
                ):

                    break

                chunk_messages.append(
                    message
                )

                current_chars += (
                    additional_chars + 1
                )

                current_index += 1

            # ---------------------------------------------------------
            # Safety check
            # ---------------------------------------------------------

            if not chunk_messages:

                break

            # ---------------------------------------------------------
            # Transcript
            # ---------------------------------------------------------

            content = "\n".join(
                format_message(message)
                for message in chunk_messages
            )

            # ---------------------------------------------------------
            # Metadata
            # ---------------------------------------------------------

            participants = sorted(
                {
                    message.sender
                    for message in chunk_messages
                    if message.sender
                }
            )

            sentiments = [
                message.sentiment_compound
                for message in chunk_messages
            ]

            average_sentiment = (
                sum(sentiments) / len(sentiments)
                if sentiments
                else 0.0
            )

            has_questions = any(
                message.is_question
                for message in chunk_messages
            )

            first_message = chunk_messages[0]

            last_message = chunk_messages[-1]

            document = Document(
                page_content=content,

                metadata={
                    "episode_id":
                        episode_id,

                    "chunk_index":
                        chunk_index,

                    "global_chunk_index":
                        global_chunk_index,

                    "start_message_id":
                        first_message.id,

                    "end_message_id":
                        last_message.id,

                    "start_timestamp":
                        first_message.timestamp.isoformat()
                        if first_message.timestamp
                        else "",

                    "end_timestamp":
                        last_message.timestamp.isoformat()
                        if last_message.timestamp
                        else "",

                    "date":
                        first_message.timestamp.strftime(
                            "%Y-%m-%d"
                        )
                        if first_message.timestamp
                        else "",

                    "num_messages":
                        len(chunk_messages),

                    "participants":
                        ", ".join(participants),

                    "avg_sentiment":
                        float(average_sentiment),

                    "has_questions":
                        bool(has_questions),

                    "source":
                        "conversation_chunk",
                },
            )

            documents.append(
                document
            )

            chunk_index += 1

            global_chunk_index += 1

            # ---------------------------------------------------------
            # Move forward with overlap
            # ---------------------------------------------------------

            if current_index >= len(episode):

                break

            next_start = (
                current_index
                - CHUNK_OVERLAP_MESSAGES
            )

            # Never move backwards.
            if next_start <= start_index:

                next_start = (
                    start_index + 1
                )

            start_index = next_start

    return documents


# =====================================================================
# Load existing Chroma vector store
# =====================================================================

def load_vectorstore():
    """
    Load the existing persistent Chroma vector store.

    This is the function required by rag_graph.py.
    """

    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,

        embedding_function=embeddings,

        persist_directory=PERSIST_DIR,
    )

    return vectorstore


# =====================================================================
# Build / recreate Chroma database
# =====================================================================

def build_vectorstore(
    messages: List[EnrichedMessage],
):
    """
    Build the Chroma vector store from enriched messages.

    WARNING:

        This deletes the existing collection so that the database
        is rebuilt from the current conversation.
    """

    print(
        "\nBuilding conversation chunks..."
    )

    documents = build_conversation_chunks(
        messages
    )

    print(
        f"Created {len(documents)} conversation chunks."
    )

    if not documents:

        raise ValueError(
            "No conversation chunks were created."
        )

    print(
        "\nCreating embeddings and storing in Chroma..."
    )

    embeddings = get_embeddings()

    # -------------------------------------------------------------
    # Open Chroma
    # -------------------------------------------------------------

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,

        embedding_function=embeddings,

        persist_directory=PERSIST_DIR,
    )

    # -------------------------------------------------------------
    # Remove previous documents
    # -------------------------------------------------------------

    try:

        existing = vectorstore.get()

        existing_ids = existing.get(
            "ids",
            []
        )

        if existing_ids:

            print(
                f"Removing {len(existing_ids)} "
                "existing Chroma documents..."
            )

            vectorstore.delete(
                ids=existing_ids
            )

    except Exception as e:

        print(
            f"Warning while clearing Chroma: {e}"
        )

    # -------------------------------------------------------------
    # Add documents in batches
    # -------------------------------------------------------------

    batch_size = 25

    total = len(documents)

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        batch = documents[
            start:end
        ]

        ids = [
            f"whatsapp_chunk_{i}"
            for i in range(
                start,
                end,
            )
        ]

        print(
            f"Embedding documents "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        vectorstore.add_documents(
            documents=batch,
            ids=ids,
        )

    print(
        "\n✓ Chroma vector store built successfully."
    )

    print(
        f"✓ Documents stored: {total}"
    )

    return vectorstore


# =====================================================================
# Vector store statistics
# =====================================================================

def get_vectorstore_stats():
    """
    Print basic Chroma statistics.
    """

    vectorstore = load_vectorstore()

    collection = vectorstore._collection

    count = collection.count()

    return {
        "collection":
            COLLECTION_NAME,

        "documents":
            count,

        "persist_directory":
            PERSIST_DIR,
    }


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    from parser import (
        parse_whatsapp_export,
    )

    from enrich import (
        enrich_messages,
    )

    print(
        "=" * 70
    )

    print(
        "ChatLens-RAG Vector Store Test"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------

    print(
        "\nParsing WhatsApp export..."
    )

    messages = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed {len(messages)} messages."
    )

    # -------------------------------------------------------------
    # Enrich
    # -------------------------------------------------------------

    print(
        "\nEnriching messages..."
    )

    enriched = enrich_messages(
        messages
    )

    print(
        f"Enriched {len(enriched)} messages."
    )

    # -------------------------------------------------------------
    # Build chunks
    # -------------------------------------------------------------

    print(
        "\nBuilding conversation chunks..."
    )

    documents = build_conversation_chunks(
        enriched
    )

    print(
        f"Created {len(documents)} chunks."
    )

    # -------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------

    if documents:

        print(
            "\nFirst chunk:"
        )

        print(
            "-" * 70
        )

        print(
            documents[0].page_content
        )

        print(
            "\nMetadata:"
        )

        print(
            documents[0].metadata
        )

    # -------------------------------------------------------------
    # Build Chroma
    # -------------------------------------------------------------

    print(
        "\nBuilding Chroma..."
    )

    vectorstore = build_vectorstore(
        enriched
    )

    # -------------------------------------------------------------
    # Test similarity search
    # -------------------------------------------------------------

    print(
        "\nTesting semantic search..."
    )

    results = vectorstore.similarity_search(
        "sports",
        k=5,
    )

    print(
        f"\nRetrieved {len(results)} results."
    )

    for index, document in enumerate(
        results,
        start=1,
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Result {index}"
        )

        print(
            "-" * 70
        )

        print(
            document.page_content
        )

        print(
            "\nMetadata:"
        )

        print(
            document.metadata
        )

    # -------------------------------------------------------------
    # Final stats
    # -------------------------------------------------------------

    stats = get_vectorstore_stats()

    print(
        "\n" + "=" * 70
    )

    print(
        "VECTOR STORE STATS"
    )

    print(
        "=" * 70
    )

    print(
        stats
    )

    print(
        "\n✓ Vector store test completed."
    )

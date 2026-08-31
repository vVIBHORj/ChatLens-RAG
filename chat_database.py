"""
chat_database.py

Stores the complete enriched WhatsApp conversation in SQLite.

SQLite is the source of truth for the original conversation.

Chroma is used for semantic retrieval.
SQLite is used for exact message lookup and context expansion.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional

from enrich import EnrichedMessage


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATABASE_PATH = "./chat_data.db"


# ---------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------
def rebuild_fts_index(
    database_path: str = DATABASE_PATH,
) -> None:
    """
    Rebuild the SQLite FTS5 index from the messages table.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO messages_fts(
            messages_fts
        )
        VALUES ('rebuild')
        """
    )

    connection.commit()

    connection.close()

    print(
        "✓ SQLite FTS index rebuilt."
    )
def search_messages(
    query: str,
    limit: int = 20,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Search messages using SQLite FTS5.

    Returns the most relevant exact/keyword matches.
    """

    if not query or not query.strip():
        return []

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    # FTS5 has its own query syntax. Quoting the individual
    # terms makes normal user questions safer.
    words = query.strip().split()

    safe_words = []

    for word in words:

        cleaned = word.replace('"', '""')

        if cleaned:
            safe_words.append(
                f'"{cleaned}"'
            )

    if not safe_words:
        connection.close()
        return []

    fts_query = " OR ".join(
        safe_words
    )

    cursor.execute(
        """
        SELECT
            messages.*,
            messages_fts.rank AS fts_rank
        FROM messages_fts
        JOIN messages
            ON messages.id = messages_fts.rowid
        WHERE messages_fts MATCH ?
        ORDER BY messages_fts.rank
        LIMIT ?
        """,
        (
            fts_query,
            limit,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


def get_connection(
    database_path: str = DATABASE_PATH,
) -> sqlite3.Connection:
    """
    Create a SQLite connection.
    """

    connection = sqlite3.connect(
        database_path
    )

    connection.row_factory = sqlite3.Row

    return connection


# ---------------------------------------------------------------------
# Create database
# ---------------------------------------------------------------------

def create_database(
    database_path: str = DATABASE_PATH,
) -> None:
    """
    Create the messages table if it doesn't exist.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY,

            timestamp TEXT,

            sender TEXT,

            message TEXT,

            message_type TEXT,

            sentiment_compound REAL,

            sentiment_label TEXT,

            is_question INTEGER,

            message_length INTEGER,

            response_time_minutes REAL,

            reply_time_minutes REAL
        )
        """
    )

    # Indexes for fast retrieval.
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp
        ON messages(timestamp)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_sender
        ON messages(sender)
        """
    )

    connection.commit()

    connection.close()


# ---------------------------------------------------------------------
# Save messages
# ---------------------------------------------------------------------

def save_messages(
    messages: List[EnrichedMessage],
    database_path: str = DATABASE_PATH,
    reset: bool = True,
) -> None:
    """
    Save all enriched WhatsApp messages into SQLite.

    If reset=True, the previous message table is cleared first.
    """

    create_database(
        database_path
    )

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    if reset:

        cursor.execute(
            "DELETE FROM messages"
        )

    rows = []

    for message in messages:

        rows.append(
            (
                message.id,

                (
                    message.timestamp.isoformat()
                    if message.timestamp
                    else None
                ),

                message.sender,

                message.message,

                message.message_type,

                message.sentiment_compound,

                message.sentiment_label,

                int(message.is_question),

                message.message_length,

                message.response_time_minutes,

                message.reply_time_minutes,
            )
        )

    cursor.executemany(
        """
        INSERT OR REPLACE INTO messages (
            id,
            timestamp,
            sender,
            message,
            message_type,
            sentiment_compound,
            sentiment_label,
            is_question,
            message_length,
            response_time_minutes,
            reply_time_minutes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.commit()

    connection.close()

    print(
        f"✓ Saved {len(rows)} messages to SQLite."
    )


# ---------------------------------------------------------------------
# Get message by ID
# ---------------------------------------------------------------------

def get_message(
    message_id: int,
    database_path: str = DATABASE_PATH,
) -> Optional[sqlite3.Row]:
    """
    Retrieve one message by ID.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE id = ?
        """,
        (message_id,),
    )

    row = cursor.fetchone()

    connection.close()

    return row


# ---------------------------------------------------------------------
# Get surrounding messages
# ---------------------------------------------------------------------

def get_context(
    message_id: int,
    before: int = 10,
    after: int = 10,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Retrieve messages surrounding a target message.

    Example:

        message_id = 100

        before = 10
        after = 10

    returns approximately:

        90 ... 100 ... 110
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    start_id = max(
        0,
        message_id - before
    )

    end_id = message_id + after

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE id BETWEEN ? AND ?
        ORDER BY id ASC
        """,
        (
            start_id,
            end_id,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# ---------------------------------------------------------------------
# Get message range
# ---------------------------------------------------------------------

def get_message_range(
    start_id: int,
    end_id: int,
    database_path: str = DATABASE_PATH,
) -> List[sqlite3.Row]:
    """
    Retrieve an exact message ID range.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM messages
        WHERE id BETWEEN ? AND ?
        ORDER BY id ASC
        """,
        (
            start_id,
            end_id,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return rows


# ---------------------------------------------------------------------
# Database statistics
# ---------------------------------------------------------------------

def get_database_stats(
    database_path: str = DATABASE_PATH,
) -> dict:
    """
    Return basic database statistics.
    """

    connection = get_connection(
        database_path
    )

    cursor = connection.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM messages"
    )

    total_messages = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT COUNT(DISTINCT sender)
        FROM messages
        WHERE sender IS NOT NULL
        """
    )

    total_senders = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT MIN(timestamp), MAX(timestamp)
        FROM messages
        """
    )

    first_timestamp, last_timestamp = cursor.fetchone()

    connection.close()

    return {
        "total_messages": total_messages,
        "total_senders": total_senders,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from parser import parse_whatsapp_export
    from enrich import enrich_messages

    print("=" * 70)
    print("ChatLens-RAG SQLite Database Test")
    print("=" * 70)

    # -------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------

    print("\nParsing WhatsApp export...")

    msgs = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed {len(msgs)} messages."
    )

    # -------------------------------------------------------------
    # Enrich
    # -------------------------------------------------------------

    print("\nEnriching messages...")

    enriched = enrich_messages(
        msgs
    )

    print(
        f"Enriched {len(enriched)} messages."
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    print("\nSaving database...")

    save_messages(
        enriched,
        reset=True,
    )

    # -------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------

    stats = get_database_stats()

    print("\nDatabase statistics:")

    for key, value in stats.items():

        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------
    # Test context retrieval
    # -------------------------------------------------------------

    if enriched:

        test_id = enriched[
            min(10, len(enriched) - 1)
        ].id

        print(
            f"\nTesting context retrieval "
            f"around message ID {test_id}..."
        )

        context = get_context(
            test_id,
            before=3,
            after=3,
        )

        for row in context:

            print(
                f"[{row['id']}] "
                f"{row['timestamp']} "
                f"- "
                f"{row['sender']}: "
                f"{row['message']}"
            )

    print("\n✓ SQLite test completed.")
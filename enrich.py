"""
enrich.py

Adds per-message metadata:
    - stable message ID
    - sentiment
    - response time
    - message length
    - whether a message is a question

Uses VADER (rule-based, fast, no model download) for sentiment instead
of an LLM call per message.

The LLM is reserved for the reasoning-heavy Q&A layer.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from parser import ChatMessage


# ---------------------------------------------------------------------
# Sentiment analyzer
# ---------------------------------------------------------------------

_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------
# Enriched message
# ---------------------------------------------------------------------

@dataclass
class EnrichedMessage:
    """
    Parsed WhatsApp message plus analytical metadata.

    `id` is the original position of the message in the parsed
    WhatsApp export. This allows us to later retrieve messages before
    and after a relevant message for context expansion.
    """

    # Stable message ID
    id: int

    # Original message data
    timestamp: Optional[datetime]
    sender: Optional[str]
    message: str
    message_type: str

    # Sentiment
    sentiment_compound: float
    sentiment_label: str

    # Message properties
    is_question: bool
    message_length: int

    # Timing
    response_time_minutes: Optional[float]
    reply_time_minutes: Optional[float]


# ---------------------------------------------------------------------
# Sentiment label
# ---------------------------------------------------------------------

def _sentiment_label(compound: float) -> str:
    """
    Convert VADER compound score into a simple label.
    """

    if compound >= 0.05:
        return "positive"

    elif compound <= -0.05:
        return "negative"

    return "neutral"


# ---------------------------------------------------------------------
# Enrich messages
# ---------------------------------------------------------------------

def enrich_messages(
    messages: List[ChatMessage],
) -> List[EnrichedMessage]:
    """
    Add analytical metadata to parsed WhatsApp messages.

    Every parsed message gets a stable ID corresponding to its original
    position in the WhatsApp export.

    System messages are skipped from the enriched dataset, but their
    timestamps are still used for response-time continuity.
    """

    enriched: List[EnrichedMessage] = []

    # Timestamp of the previous parsed message
    last_timestamp: Optional[datetime] = None

    # sender -> timestamp of their most recent message
    last_sender_timestamp = {}

    # -----------------------------------------------------------------
    # Process messages
    # -----------------------------------------------------------------

    for message_id, m in enumerate(messages):

        # -------------------------------------------------------------
        # System messages
        # -------------------------------------------------------------

        if m.message_type == "system":

            # Keep timestamp continuity.
            last_timestamp = m.timestamp or last_timestamp

            continue

        # -------------------------------------------------------------
        # Message text
        # -------------------------------------------------------------

        text = m.message or ""

        # -------------------------------------------------------------
        # Sentiment
        # -------------------------------------------------------------

        scores = _analyzer.polarity_scores(text)

        compound = scores["compound"]

        # -------------------------------------------------------------
        # Response time
        # -------------------------------------------------------------

        response_time = None

        if last_timestamp is not None and m.timestamp is not None:

            response_time = (
                m.timestamp - last_timestamp
            ).total_seconds() / 60.0

        # -------------------------------------------------------------
        # Reply time
        # -------------------------------------------------------------

        reply_time = None

        if m.timestamp is not None:

            # Find the most recent timestamp belonging to another
            # participant.
            other_sender_timestamps = [
                ts
                for sender, ts in last_sender_timestamp.items()
                if sender != m.sender
                and ts is not None
            ]

            if other_sender_timestamps:

                most_recent_other_timestamp = max(
                    other_sender_timestamps
                )

                reply_time = (
                    m.timestamp
                    - most_recent_other_timestamp
                ).total_seconds() / 60.0

        # -------------------------------------------------------------
        # Create enriched message
        # -------------------------------------------------------------

        enriched.append(
            EnrichedMessage(
                # THIS FIXES YOUR ERROR
                id=message_id,

                timestamp=m.timestamp,

                sender=m.sender,

                message=text,

                message_type=m.message_type,

                sentiment_compound=compound,

                sentiment_label=_sentiment_label(compound),

                is_question="?" in text,

                message_length=len(text),

                response_time_minutes=response_time,

                reply_time_minutes=reply_time,
            )
        )

        # -------------------------------------------------------------
        # Update timing state
        # -------------------------------------------------------------

        last_timestamp = m.timestamp or last_timestamp

        if m.sender and m.timestamp:

            last_sender_timestamp[m.sender] = m.timestamp

    return enriched


# ---------------------------------------------------------------------
# Convert to DataFrame
# ---------------------------------------------------------------------

def to_dataframe(
    enriched: List[EnrichedMessage],
) -> pd.DataFrame:
    """
    Convert enriched messages into a pandas DataFrame.
    """

    return pd.DataFrame(
        [asdict(message) for message in enriched]
    )


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    from parser import parse_whatsapp_export

    print("=" * 70)
    print("ChatLens-RAG Enrichment Test")
    print("=" * 70)

    # -------------------------------------------------------------
    # Parse WhatsApp export
    # -------------------------------------------------------------

    print("\nParsing WhatsApp export...")

    msgs = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed {len(msgs)} messages."
    )

    # -------------------------------------------------------------
    # Enrich messages
    # -------------------------------------------------------------

    print("\nEnriching messages...")

    enriched = enrich_messages(msgs)

    print(
        f"Enriched {len(enriched)} messages."
    )

    # -------------------------------------------------------------
    # Show first 10 messages
    # -------------------------------------------------------------

    print("\nFirst 10 enriched messages:")

    for message in enriched[:10]:

        print("\n" + "-" * 70)

        print(
            f"ID: {message.id}"
        )

        print(
            f"Timestamp: {message.timestamp}"
        )

        print(
            f"Sender: {message.sender}"
        )

        print(
            f"Message: {message.message}"
        )

        print(
            f"Type: {message.message_type}"
        )

        print(
            f"Sentiment: {message.sentiment_label}"
        )

        print(
            f"Sentiment score: "
            f"{message.sentiment_compound}"
        )

        print(
            f"Question: {message.is_question}"
        )

        print(
            f"Message length: "
            f"{message.message_length}"
        )

        print(
            f"Response time: "
            f"{message.response_time_minutes}"
        )

        print(
            f"Reply time: "
            f"{message.reply_time_minutes}"
        )

    # -------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------

    df = to_dataframe(enriched)

    print("\n" + "=" * 70)
    print("DATAFRAME PREVIEW")
    print("=" * 70)

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    print(
        df[
            [
                "id",
                "timestamp",
                "sender",
                "message",
                "sentiment_label",
                "sentiment_compound",
                "is_question",
                "message_length",
                "response_time_minutes",
                "reply_time_minutes",
            ]
        ].head(10)
    )

    # -------------------------------------------------------------
    # ID verification
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ID VERIFICATION")
    print("=" * 70)

    if enriched:

        print(
            f"First enriched message ID: "
            f"{enriched[0].id}"
        )

        print(
            f"Last enriched message ID: "
            f"{enriched[-1].id}"
        )

        print(
            f"Total enriched messages: "
            f"{len(enriched)}"
        )

        print(
            "\n✓ Stable message IDs successfully created."
        )

    else:

        print(
            "\n⚠ No messages were enriched."
        )
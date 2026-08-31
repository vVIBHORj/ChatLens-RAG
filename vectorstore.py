"""
enrich.py

Adds per-message metadata:
    - stable message ID
    - sentiment
    - response time
    - message length
    - whether a message is a question

VADER is used for fast local sentiment analysis.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from parser import ChatMessage


_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------
# Enriched message
# ---------------------------------------------------------------------

@dataclass
class EnrichedMessage:
    """
    A parsed WhatsApp message with additional analysis metadata.
    """

    # Stable position/ID in the conversation
    id: int

    timestamp: Optional[datetime]
    sender: Optional[str]
    message: str
    message_type: str

    # Sentiment
    sentiment_compound: float
    sentiment_label: str

    # Message characteristics
    is_question: bool
    message_length: int

    # Timing
    response_time_minutes: Optional[float]
    reply_time_minutes: Optional[float]


# ---------------------------------------------------------------------
# Sentiment
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
# Enrichment
# ---------------------------------------------------------------------

def enrich_messages(
    messages: List[ChatMessage],
) -> List[EnrichedMessage]:
    """
    Add analytical metadata to parsed WhatsApp messages.

    Every message receives a stable sequential ID.

    Important:
        The ID represents the message's position in the parsed
        conversation and will later allow us to retrieve neighboring
        messages for context expansion.
    """

    enriched: List[EnrichedMessage] = []

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

            # System messages are retained by the parser but excluded
            # from normal enrichment/semantic analysis.
            last_timestamp = m.timestamp or last_timestamp
            continue

        # -------------------------------------------------------------
        # Sentiment
        # -------------------------------------------------------------

        text = m.message or ""

        scores = _analyzer.polarity_scores(text)

        compound = scores["compound"]

        # -------------------------------------------------------------
        # Response time
        # -------------------------------------------------------------

        response_time = None

        if last_timestamp and m.timestamp:

            response_time = (
                m.timestamp - last_timestamp
            ).total_seconds() / 60.0

        # -------------------------------------------------------------
        # Reply time
        # -------------------------------------------------------------

        reply_time = None

        if m.timestamp and m.sender:

            # Find the most recent message from another sender.
            other_sender_timestamps = [
                ts
                for sender, ts in last_sender_timestamp.items()
                if sender != m.sender and ts is not None
            ]

            if other_sender_timestamps:

                most_recent_other_timestamp = max(
                    other_sender_timestamps
                )

                reply_time = (
                    m.timestamp - most_recent_other_timestamp
                ).total_seconds() / 60.0

        # -------------------------------------------------------------
        # Create enriched message
        # -------------------------------------------------------------

        enriched.append(
            EnrichedMessage(
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
# DataFrame
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

    print("=" * 60)
    print("ChatLens-RAG Enrichment Test")
    print("=" * 60)

    # -------------------------------------------------------------
    # Parse
    # -------------------------------------------------------------

    msgs = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"\nParsed messages: {len(msgs)}"
    )

    # -------------------------------------------------------------
    # Enrich
    # -------------------------------------------------------------

    enriched = enrich_messages(msgs)

    print(
        f"Enriched messages: {len(enriched)}"
    )

    # -------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------

    for message in enriched[:10]:

        print("\n----------------------------------------")

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

    print("\n")
    print("=" * 60)
    print("DATAFRAME")
    print("=" * 60)

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
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
                "reply_time_minutes",
            ]
        ].head(10)
    )
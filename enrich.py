"""
enrich.py

Adds analytical metadata to parsed WhatsApp messages.

Features:
    - stable message ID
    - sentiment
    - sentiment label
    - response time
    - reply time
    - message length
    - question detection

VADER is used for fast local sentiment analysis.

Important architecture rule:

    parser.py
        ↓
    enrich.py
        ↓
    SQLite = source of truth
        ↓
    Chroma = semantic retrieval

The `id` field is the original position of the message
inside the parsed WhatsApp export. It is intentionally NOT
renumbered after filtering system messages.

This allows us to reliably locate neighboring messages later.
"""


# =====================================================================
# Imports
# =====================================================================

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer,
)

from parser import ChatMessage


# =====================================================================
# Configuration
# =====================================================================

# Messages separated by a very large time gap should not produce
# meaningful "response time" values.
#
# We still calculate the actual time difference, but this constant
# can later be used by analytics if desired.
MAX_REASONABLE_RESPONSE_GAP_MINUTES = 24 * 60


# =====================================================================
# Sentiment analyzer
# =====================================================================

_analyzer = SentimentIntensityAnalyzer()


# =====================================================================
# Enriched message
# =====================================================================

@dataclass
class EnrichedMessage:
    """
    Parsed WhatsApp message plus analytical metadata.

    `id`:
        Original position of the message in the parsed WhatsApp
        export.

        Example:

            parser message #0 -> id 0
            parser message #1 -> id 1
            parser message #2 -> id 2

        System messages can later be excluded from the enriched
        dataset while their IDs remain preserved.

    This makes the ID suitable for context expansion.
    """

    # -----------------------------------------------------------------
    # Stable message identity
    # -----------------------------------------------------------------

    id: int

    # -----------------------------------------------------------------
    # Original message information
    # -----------------------------------------------------------------

    timestamp: Optional[datetime]

    sender: Optional[str]

    message: str

    message_type: str

    # -----------------------------------------------------------------
    # Sentiment
    # -----------------------------------------------------------------

    sentiment_compound: float

    sentiment_label: str

    # -----------------------------------------------------------------
    # Message properties
    # -----------------------------------------------------------------

    is_question: bool

    message_length: int

    # -----------------------------------------------------------------
    # Timing
    # -----------------------------------------------------------------

    response_time_minutes: Optional[float]

    reply_time_minutes: Optional[float]


# =====================================================================
# Sentiment label
# =====================================================================

def _sentiment_label(
    compound: float,
) -> str:
    """
    Convert VADER compound score into a simple label.

    VADER convention:

        compound >= 0.05  -> positive
        compound <= -0.05 -> negative
        otherwise         -> neutral
    """

    if compound >= 0.05:

        return "positive"

    if compound <= -0.05:

        return "negative"

    return "neutral"


# =====================================================================
# Safe time difference
# =====================================================================

def _minutes_between(
    newer: Optional[datetime],
    older: Optional[datetime],
) -> Optional[float]:
    """
    Return the difference between two timestamps in minutes.

    Returns None if either timestamp is unavailable.

    Negative differences are treated as None because they usually
    indicate malformed/out-of-order timestamps rather than a real
    response time.
    """

    if newer is None or older is None:

        return None

    try:

        minutes = (
            newer - older
        ).total_seconds() / 60.0

    except Exception:

        return None

    if minutes < 0:

        return None

    return minutes


# =====================================================================
# Enrichment
# =====================================================================

def enrich_messages(
    messages: List[ChatMessage],
) -> List[EnrichedMessage]:
    """
    Add analytical metadata to parsed WhatsApp messages.

    Stable IDs
    ----------

    Every parsed message gets an ID equal to its original position
    in the parser output.

    Example:

        parsed index 0 -> id 0
        parsed index 1 -> id 1
        parsed index 2 -> id 2

    If a system message occurs at index 5, it is skipped from the
    enriched dataset, but the next message still has its original
    parser ID.

    This is important because SQLite context expansion can later
    use these IDs to locate neighboring messages.

    Response time
    -------------

    Time since the immediately previous parsed message.

    Reply time
    ----------

    Time since the most recent message from another participant.

    System messages
    ---------------

    System messages are excluded from the enriched dataset but their
    timestamps are retained for chronological continuity.
    """

    enriched: List[EnrichedMessage] = []

    # -----------------------------------------------------------------
    # Previous parsed message timestamp
    # -----------------------------------------------------------------

    last_timestamp: Optional[datetime] = None

    # -----------------------------------------------------------------
    # Most recent timestamp for every sender
    #
    # Example:
    #
    # {
    #     "Vibhor": datetime(...),
    #     "EXODIA": datetime(...)
    # }
    # -----------------------------------------------------------------

    last_sender_timestamp: Dict[
        str,
        datetime,
    ] = {}

    # -----------------------------------------------------------------
    # Process messages
    # -----------------------------------------------------------------

    for message_id, message in enumerate(
        messages
    ):

        # =============================================================
        # System message
        # =============================================================

        if message.message_type == "system":

            # Keep timestamp continuity.
            if message.timestamp is not None:

                last_timestamp = (
                    message.timestamp
                )

            continue

        # =============================================================
        # Basic message information
        # =============================================================

        text = (
            message.message
            or ""
        )

        sender = message.sender

        timestamp = message.timestamp

        # =============================================================
        # Sentiment
        # =============================================================

        scores = _analyzer.polarity_scores(
            text
        )

        compound = float(
            scores["compound"]
        )

        sentiment_label = _sentiment_label(
            compound
        )

        # =============================================================
        # Question detection
        # =============================================================

        is_question = (
            "?" in text
        )

        # =============================================================
        # Message length
        # =============================================================

        message_length = len(
            text
        )

        # =============================================================
        # Response time
        # =============================================================

        response_time = _minutes_between(
            timestamp,
            last_timestamp,
        )

        # =============================================================
        # Reply time
        # =============================================================

        reply_time = None

        if (
            timestamp is not None
            and sender is not None
        ):

            # Find the latest message belonging to someone else.
            #
            # This is intentionally based on sender identity rather
            # than simply using the previous message.
            other_timestamps = [
                previous_timestamp

                for previous_sender,
                previous_timestamp
                in last_sender_timestamp.items()

                if previous_sender != sender
            ]

            if other_timestamps:

                most_recent_other_timestamp = max(
                    other_timestamps
                )

                reply_time = _minutes_between(
                    timestamp,
                    most_recent_other_timestamp,
                )

        # =============================================================
        # Create enriched message
        # =============================================================

        enriched_message = EnrichedMessage(

            # ---------------------------------------------------------
            # Stable ID
            # ---------------------------------------------------------

            id=message_id,

            # ---------------------------------------------------------
            # Original fields
            # ---------------------------------------------------------

            timestamp=timestamp,

            sender=sender,

            message=text,

            message_type=message.message_type,

            # ---------------------------------------------------------
            # Sentiment
            # ---------------------------------------------------------

            sentiment_compound=compound,

            sentiment_label=sentiment_label,

            # ---------------------------------------------------------
            # Message properties
            # ---------------------------------------------------------

            is_question=is_question,

            message_length=message_length,

            # ---------------------------------------------------------
            # Timing
            # ---------------------------------------------------------

            response_time_minutes=response_time,

            reply_time_minutes=reply_time,
        )

        enriched.append(
            enriched_message
        )

        # =============================================================
        # Update timing state
        # =============================================================

        if timestamp is not None:

            last_timestamp = timestamp

        if (
            sender
            and timestamp is not None
        ):

            last_sender_timestamp[
                sender
            ] = timestamp

    return enriched


# =====================================================================
# Convert to DataFrame
# =====================================================================

def to_dataframe(
    enriched: List[EnrichedMessage],
) -> pd.DataFrame:
    """
    Convert enriched messages into a pandas DataFrame.

    Each dataclass field becomes a DataFrame column.
    """

    if not enriched:

        return pd.DataFrame(
            columns=[
                "id",
                "timestamp",
                "sender",
                "message",
                "message_type",
                "sentiment_compound",
                "sentiment_label",
                "is_question",
                "message_length",
                "response_time_minutes",
                "reply_time_minutes",
            ]
        )

    return pd.DataFrame(
        [
            asdict(message)
            for message in enriched
        ]
    )


# =====================================================================
# Validation
# =====================================================================

def validate_enriched_messages(
    enriched: List[EnrichedMessage],
) -> dict:
    """
    Validate the enriched dataset.

    Returns a dictionary of validation results.

    This is useful before inserting the data into SQLite or Chroma.
    """

    if not enriched:

        return {
            "valid": False,
            "message_count": 0,
            "unique_ids": 0,
            "duplicate_ids": 0,
            "missing_timestamps": 0,
            "missing_senders": 0,
        }

    ids = [
        message.id
        for message in enriched
    ]

    unique_ids = len(
        set(ids)
    )

    duplicate_ids = (
        len(ids) - unique_ids
    )

    missing_timestamps = sum(
        message.timestamp is None
        for message in enriched
    )

    missing_senders = sum(
        not message.sender
        for message in enriched
    )

    valid = (
        duplicate_ids == 0
        and all(
            isinstance(
                message.id,
                int,
            )
            for message in enriched
        )
    )

    return {
        "valid": valid,
        "message_count": len(enriched),
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "missing_timestamps": missing_timestamps,
        "missing_senders": missing_senders,
    }


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    from parser import (
        parse_whatsapp_export,
    )

    print(
        "=" * 70
    )

    print(
        "ChatLens-RAG Enrichment Test"
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

    msgs = parse_whatsapp_export(
        "sample_chat.txt"
    )

    print(
        f"Parsed messages: "
        f"{len(msgs)}"
    )

    # -------------------------------------------------------------
    # Enrich
    # -------------------------------------------------------------

    print(
        "\nEnriching messages..."
    )

    enriched = enrich_messages(
        msgs
    )

    print(
        f"Enriched messages: "
        f"{len(enriched)}"
    )

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    print(
        "\nValidating enriched messages..."
    )

    validation = (
        validate_enriched_messages(
            enriched
        )
    )

    for key, value in validation.items():

        print(
            f"{key}: {value}"
        )

    if validation["valid"]:

        print(
            "\n✓ Enriched dataset passed validation."
        )

    else:

        print(
            "\n⚠ Enriched dataset has validation issues."
        )

    # -------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FIRST 10 ENRICHED MESSAGES"
    )

    print(
        "=" * 70
    )

    for message in enriched[:10]:

        print(
            "\n" + "-" * 70
        )

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

    df = to_dataframe(
        enriched
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATAFRAME PREVIEW"
    )

    print(
        "=" * 70
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        200,
    )

    print(
        df.head(10)
    )

    # -------------------------------------------------------------
    # ID verification
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ID VERIFICATION"
    )

    print(
        "=" * 70
    )

    if enriched:

        ids = [
            message.id
            for message in enriched
        ]

        print(
            f"First enriched ID: "
            f"{ids[0]}"
        )

        print(
            f"Last enriched ID: "
            f"{ids[-1]}"
        )

        print(
            f"Total enriched messages: "
            f"{len(enriched)}"
        )

        print(
            f"Unique IDs: "
            f"{len(set(ids))}"
        )

        # Check whether IDs are unique.
        if len(ids) == len(set(ids)):

            print(
                "\n✓ IDs are unique."
            )

        else:

            print(
                "\n❌ Duplicate IDs detected."
            )

    else:

        print(
            "\n⚠ No messages were enriched."
        )

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ Enrichment test completed."
    )

    print(
        "=" * 70
    )

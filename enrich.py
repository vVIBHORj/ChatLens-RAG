"""
enrich.py
Adds per-message metadata: sentiment, response time, message length, and
whether a message is a question. Uses VADER (rule-based, fast, no model
download) for sentiment instead of an LLM call per message -- llama3.2 is
reserved for the reasoning-heavy Q&A layer, not cheap per-row scoring.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from parser import ChatMessage

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class EnrichedMessage:
    timestamp: Optional[datetime]
    sender: Optional[str]
    message: str
    message_type: str
    sentiment_compound: float   # -1 (very negative) to +1 (very positive)
    sentiment_label: str        # "positive" | "negative" | "neutral"
    is_question: bool
    message_length: int
    response_time_minutes: Optional[float]  # gap since previous message (any sender)
    reply_time_minutes: Optional[float]      # gap since previous message from the OTHER sender


def _sentiment_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


def enrich_messages(messages: List[ChatMessage]) -> List[EnrichedMessage]:
    enriched: List[EnrichedMessage] = []
    last_timestamp: Optional[datetime] = None
    last_sender_timestamp = {}  # sender -> last timestamp they saw a message from the OTHER person

    for m in messages:
        if m.message_type != "text":
            last_timestamp = m.timestamp or last_timestamp
            continue

        scores = _analyzer.polarity_scores(m.message)
        compound = scores["compound"]

        response_time = None
        if last_timestamp and m.timestamp:
            response_time = (m.timestamp - last_timestamp).total_seconds() / 60.0

        # reply_time: minutes since the OTHER sender last spoke (proxy for "did they leave me on read")
        reply_time = None
        # find timestamp of the most recent message from a different sender
        for s, ts in last_sender_timestamp.items():
            if s != m.sender and ts is not None:
                if m.timestamp:
                    candidate = (m.timestamp - ts).total_seconds() / 60.0
                    if reply_time is None or candidate < reply_time:
                        reply_time = candidate

        enriched.append(
            EnrichedMessage(
                timestamp=m.timestamp,
                sender=m.sender,
                message=m.message,
                message_type=m.message_type,
                sentiment_compound=compound,
                sentiment_label=_sentiment_label(compound),
                is_question="?" in m.message,
                message_length=len(m.message),
                response_time_minutes=response_time,
                reply_time_minutes=reply_time,
            )
        )

        last_timestamp = m.timestamp or last_timestamp
        if m.sender:
            last_sender_timestamp[m.sender] = m.timestamp

    return enriched


def to_dataframe(enriched: List[EnrichedMessage]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in enriched])


if __name__ == "__main__":
    from parser import parse_whatsapp_export

    msgs = parse_whatsapp_export("sample_chat.txt")
    enriched = enrich_messages(msgs)
    df = to_dataframe(enriched)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(df[["timestamp", "sender", "message", "sentiment_label",
              "sentiment_compound", "reply_time_minutes"]])

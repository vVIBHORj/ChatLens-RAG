"""
analytics.py
Aggregate behavioral pattern analysis over enriched chat data.

Deliberately separate from the RAG Q&A layer (rag_graph.py): this module
computes real statistics from the data (sentiment trends, response-time
trends, engagement balance) rather than asking an LLM to invent a single
"trust score". An LLM is optionally used only to turn the computed numbers
into a plain-language written summary -- it never produces the numbers
themselves.
"""

from typing import Optional
import pandas as pd


def sender_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sender aggregate stats: message count, avg sentiment, avg reply time,
    question-asking rate."""
    text_df = df[df["message_type"] == "text"]
    summary = text_df.groupby("sender").agg(
        message_count=("message", "count"),
        avg_sentiment=("sentiment_compound", "mean"),
        avg_reply_time_minutes=("reply_time_minutes", "mean"),
        median_reply_time_minutes=("reply_time_minutes", "median"),
        question_rate=("is_question", "mean"),
        avg_message_length=("message_length", "mean"),
    ).round(2)
    return summary.reset_index()


def weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly rollup of sentiment and responsiveness, for trend charts."""
    text_df = df[df["message_type"] == "text"].copy()
    text_df["week"] = pd.to_datetime(text_df["timestamp"]).dt.to_period("W").apply(lambda p: p.start_time)
    trend = text_df.groupby("week").agg(
        avg_sentiment=("sentiment_compound", "mean"),
        avg_reply_time_minutes=("reply_time_minutes", "mean"),
        message_count=("message", "count"),
    ).round(3).reset_index()
    return trend


def engagement_balance(df: pd.DataFrame) -> dict:
    """Who initiates conversations more, and how balanced is message volume."""
    text_df = df[df["message_type"] == "text"]
    counts = text_df["sender"].value_counts()
    total = counts.sum()
    balance = (counts / total).round(3).to_dict()

    # crude "initiator" proxy: messages sent after a gap of 3+ hours from
    # the last message (i.e. starting a new conversation thread)
    text_df = text_df.copy()
    text_df["gap_hours"] = text_df["response_time_minutes"].fillna(0) / 60.0
    initiators = text_df[text_df["gap_hours"] >= 3]["sender"].value_counts()

    return {
        "message_share": balance,
        "conversation_initiations": initiators.to_dict(),
    }


def generate_plain_language_summary(df: pd.DataFrame, llm=None) -> str:
    """Turn the computed stats into a human-readable paragraph. Uses an LLM
    (llama3.2 via ChatOllama) purely to phrase the ALREADY-COMPUTED numbers
    in plain language -- the LLM does not invent any figures itself."""
    stats = sender_summary(df)
    trend = weekly_trend(df)
    balance = engagement_balance(df)

    stats_text = stats.to_string(index=False)
    trend_text = trend.to_string(index=False)

    if llm is None:
        return (
            "Per-sender stats:\n" + stats_text +
            "\n\nWeekly trend:\n" + trend_text +
            f"\n\nMessage share: {balance['message_share']}" +
            f"\nConversation initiations: {balance['conversation_initiations']}"
        )

    prompt = (
        "You are summarizing behavioral patterns from a chat analysis in 3-4 "
        "plain-language sentences. Do NOT invent any numbers beyond what is "
        "given below -- only describe the trends these numbers show.\n\n"
        f"Per-sender stats:\n{stats_text}\n\n"
        f"Weekly trend:\n{trend_text}\n\n"
        f"Message share: {balance['message_share']}\n"
        f"Conversation initiations: {balance['conversation_initiations']}\n\n"
        "Plain-language summary:"
    )
    return llm.invoke(prompt).content.strip()


if __name__ == "__main__":
    from parser import parse_whatsapp_export
    from enrich import enrich_messages, to_dataframe

    msgs = parse_whatsapp_export("sample_chat.txt")
    enriched = enrich_messages(msgs)
    df = to_dataframe(enriched)

    print("=== Per-sender summary ===")
    print(sender_summary(df))
    print("\n=== Weekly trend ===")
    print(weekly_trend(df))
    print("\n=== Engagement balance ===")
    print(engagement_balance(df))
    print("\n=== Plain-language summary (no LLM, raw stats) ===")
    print(generate_plain_language_summary(df))
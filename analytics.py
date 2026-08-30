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
import re
from collections import Counter

import pandas as pd


# Standard English stopwords plus WhatsApp-specific noise tokens (media
# placeholders, link fragments, etc). Hardcoded rather than pulled from
# nltk/sklearn so this has no extra runtime dependency or data download.
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "than", "too", "very", "can", "will", "just", "don", "should", "now",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "should", "ought",
    "im", "ive", "id", "ill", "youre", "youve", "youll", "youd", "hes",
    "shes", "theyre", "theyve", "theyll", "theyd", "dont", "cant", "wont",
    "isnt", "arent", "wasnt", "werent", "havent", "hasnt", "hadnt",
    "doesnt", "didnt", "couldnt", "shouldnt", "wouldnt", "mustnt", "lets",
    "thats", "whos", "whats", "heres", "theres", "ok", "okay", "yeah",
    "yes", "hmm", "hm", "oh", "ah", "um", "uh", "like", "really",
    "actually", "kinda", "gonna", "wanna", "gotta",
})

WHATSAPP_NOISE = frozenset({
    "media", "omitted", "message", "deleted", "http", "https", "www",
    "com", "image", "video", "audio", "document", "sticker", "gif",
})

_WORD_RE = re.compile(r"[a-zA-Z']+")


def sender_word_frequencies(
    df: pd.DataFrame, top_n: int = 25, extra_stopwords: Optional[set] = None
) -> dict:
    """Per-sender word frequency counts ("word bag") after removing
    stopwords, punctuation, numbers, and WhatsApp artifacts. Powers the
    word-usage view on the dashboard."""
    text_df = df[df["message_type"] == "text"]
    stop = STOPWORDS | WHATSAPP_NOISE
    if extra_stopwords:
        stop = stop | {w.lower() for w in extra_stopwords}

    result = {}
    for sender, group in text_df.groupby("sender"):
        counter = Counter()
        for msg in group["message"]:
            for w in _WORD_RE.findall(str(msg).lower()):
                w = w.strip("'")
                if len(w) <= 2 or w in stop:
                    continue
                counter[w] += 1
        result[sender] = counter.most_common(top_n)
    return result


def build_style_profile(df: pd.DataFrame, examples_per_sender: int = 3) -> str:
    """Build a compact, WHOLE-CONVERSATION profile per sender: real computed
    stats plus a few representative message examples (most positive, most
    negative, sample questions asked). This stays a fixed, bounded size
    regardless of how long the conversation is.

    Used to answer broad/holistic questions ("describe each person's
    communication style", "how has the tone changed overall") without
    relying on top-k semantic retrieval, which only surfaces a handful of
    daily chunks that happen to word-match the question -- a biased slice
    of the conversation, not the whole thing.
    """
    stats = sender_summary(df).set_index("sender")
    word_freqs = sender_word_frequencies(df, top_n=10)
    text_df = df[df["message_type"] == "text"]

    sections = []
    for sender, group in text_df.groupby("sender"):
        if sender not in stats.index:
            continue
        s = stats.loc[sender]
        top_words = ", ".join(w for w, _ in word_freqs.get(sender, [])) or "(not enough data)"

        sorted_by_sentiment = group.sort_values("sentiment_compound")
        most_negative = sorted_by_sentiment.head(examples_per_sender)["message"].tolist()
        most_positive = sorted_by_sentiment.tail(examples_per_sender)["message"].tolist()
        sample_questions = group[group["is_question"]]["message"].head(examples_per_sender).tolist()

        sections.append(
            f"=== {sender} ===\n"
            f"Total messages: {int(s['message_count'])}\n"
            f"Average sentiment: {s['avg_sentiment']} (-1 very negative to +1 very positive)\n"
            f"Average message length: {s['avg_message_length']} characters\n"
            f"Question-asking rate: {s['question_rate']} (share of messages containing '?')\n"
            f"Average reply time: {s['avg_reply_time_minutes']} minutes\n"
            f"Frequently used words: {top_words}\n"
            f"Examples of their more negative-toned messages: {most_negative}\n"
            f"Examples of their more positive-toned messages: {most_positive}\n"
            f"Examples of questions they asked: {sample_questions}\n"
        )

    return "\n".join(sections)


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

    # crude "initiator" proxy: messages sent after a gap of 3+ hours since
    # the immediately preceding message from ANYONE (i.e. starting a fresh
    # conversation thread rather than continuing one). response_time_minutes
    # is already in minutes, so we convert the 3-hour threshold to minutes
    # rather than converting the column to hours.
    INITIATION_GAP_MINUTES = 3 * 60
    text_df = text_df.copy()
    initiators = text_df[
        text_df["response_time_minutes"].fillna(0) >= INITIATION_GAP_MINUTES
    ]["sender"].value_counts()

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

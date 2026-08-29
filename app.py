"""
app.py
Streamlit front end tying together parsing, enrichment, vector storage,
LangGraph RAG Q&A, and the analytics dashboard.

Run with:
    streamlit run app.py

Prerequisites (on the machine running this, not a sandbox):
    ollama pull llama3.2
    ollama pull mxbai-embed-large
    ollama serve   (usually already running as a background service)
"""

import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from parser import parse_whatsapp_export
from enrich import enrich_messages, to_dataframe
from vectorstore import build_daily_chunks, build_vectorstore
from analytics import sender_summary, weekly_trend, engagement_balance
from rag_graph import ask

st.set_page_config(page_title="Chat Pattern Analyzer", layout="wide")
st.title("📱 WhatsApp Chat Pattern Analyzer")
st.caption(
    "A local, private tool for semantic Q&A and behavioral pattern analysis "
    "over a WhatsApp export. Runs entirely on-device via Ollama. "
    "This is a pattern-analysis aid, not a psychological or trust diagnosis."
)

if "df" not in st.session_state:
    st.session_state.df = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False

uploaded = st.file_uploader("Upload your WhatsApp chat export (.txt)", type=["txt"])

if uploaded is not None and st.session_state.df is None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Parsing and analyzing messages..."):
        messages = parse_whatsapp_export(tmp_path)
        enriched = enrich_messages(messages)
        df = to_dataframe(enriched)
        st.session_state.df = df

    with st.spinner("Building local vector store (embedding with mxbai-embed-large)..."):
        docs = build_daily_chunks(enriched)
        build_vectorstore(docs)
        st.session_state.vectorstore_ready = True

    os.unlink(tmp_path)
    st.success(f"Processed {len(df)} messages across {df['sender'].nunique()} participants.")

if st.session_state.df is not None:
    df = st.session_state.df
    tab1, tab2 = st.tabs(["💬 Ask Questions", "📊 Pattern Dashboard"])

    with tab1:
        st.subheader("Ask semantic questions about the conversation")
        st.caption("Answers are grounded in retrieved chat excerpts and checked for hallucination.")
        question = st.text_input("Your question", placeholder="e.g. How did our conversations change after week 2?")
        if st.button("Ask") and question:
            with st.spinner("Retrieving context and reasoning with llama3.2..."):
                result = ask(question)
            st.markdown(f"**Answer:** {result['answer']}")
            st.caption(f"Grounded in retrieved context: {'✅ Yes' if result['grounded'] else '⚠️ Uncertain'}")
            with st.expander("Sources used"):
                for s in result["sources"]:
                    st.write(s)

    with tab2:
        st.subheader("Behavioral pattern summary")
        st.warning(
            "These are descriptive statistics about messaging patterns, "
            "not a validated measure of trust, character, or relationship health."
        )

        summary = sender_summary(df)
        st.dataframe(summary, use_container_width=True)

        col1, col2 = st.columns(2)

        trend = weekly_trend(df)
        with col1:
            fig1 = px.line(trend, x="week", y="avg_sentiment", title="Weekly Average Sentiment")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.line(trend, x="week", y="avg_reply_time_minutes", title="Weekly Average Reply Time (minutes)")
            st.plotly_chart(fig2, use_container_width=True)

        balance = engagement_balance(df)
        col3, col4 = st.columns(2)
        with col3:
            share_df = pd.DataFrame(
                balance["message_share"].items(), columns=["sender", "share"]
            )
            fig3 = px.pie(share_df, names="sender", values="share", title="Message Volume Share")
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            init_df = pd.DataFrame(
                balance["conversation_initiations"].items(), columns=["sender", "initiations"]
            )
            fig4 = px.bar(init_df, x="sender", y="initiations", title="Conversation Initiations")
            st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Upload a WhatsApp .txt export to get started. (WhatsApp: Chat > Export chat > Without media)")
"""
rag_graph.py
A Corrective-RAG (CRAG) pipeline built with LangGraph:

    retrieve -> grade_documents -> [relevant?] -> generate -> check_grounded -> END
                                  -> [not relevant?] -> rewrite_query -> retrieve (loop, capped)

Uses llama3.2 (via Ollama) as both the answering model and the "judge" for
grading retrieved chunks and checking whether the final answer is actually
grounded in the retrieved context.

Requires a running local Ollama instance:
    ollama pull llama3.2
    ollama pull mxbai-embed-large
"""

from typing import List, Optional, TypedDict

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from vectorstore import load_vectorstore
from analytics import build_style_profile

MAX_REWRITES = 2
llm = ChatOllama(model="qwen3:8b", temperature=0)


class GraphState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    rewrite_count: int
    grounded: bool
    df: Optional[pd.DataFrame]
    scope: str


# ---------- Nodes ----------

def classify_scope(state: GraphState) -> GraphState:
    """Decide whether this question needs a WHOLE-conversation view or can
    be answered from a few semantically retrieved chunks.

    'broad' = personality/communication-style/overall-pattern/general-tone
    questions -- these bypass vector retrieval entirely and use a profile
    built from real stats + samples spanning the full conversation.
    'specific' = a particular fact, date, or exact thing said -- these use
    the existing retrieve -> grade -> generate pipeline, which is the right
    tool when the answer really does live in one or two chunks.
    """
    prompt = (
        "Classify the following question about a chat conversation as "
        "exactly one word: 'broad' or 'specific'.\n\n"
        "'broad' = asking about overall patterns, personality, communication "
        "style, general mood or tone, or anything spanning the whole "
        "conversation rather than one moment in it.\n"
        "'specific' = asking about a particular fact, date, event, plan, or "
        "an exact thing someone said.\n\n"
        f"Question: {state['question']}\n\n"
        "Answer with exactly one word:"
    )
    response = llm.invoke(prompt).content.strip().lower()
    scope = "broad" if "broad" in response else "specific"
    return {**state, "scope": scope}


def build_profile_context(state: GraphState) -> GraphState:
    """For broad questions: wrap the whole-conversation stats profile as a
    single synthetic Document so it flows through the existing
    generate/check_grounded nodes unchanged."""
    df = state.get("df")
    if df is None or df.empty:
        return {**state, "documents": []}

    profile_text = build_style_profile(df)
    doc = Document(
        page_content=profile_text,
        metadata={"source": "whole_conversation_profile"},
    )
    return {**state, "documents": [doc]}

def retrieve(state: GraphState) -> GraphState:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}


def grade_documents(state: GraphState) -> GraphState:
    """Ask llama3.2 to judge whether ANY retrieved chunk is relevant enough
    to answer the question. Keeps only the relevant ones."""
    question = state["question"]
    relevant_docs = []

    for doc in state["documents"]:
        prompt = (
            "You are grading whether a document is relevant to a user question.\n"
            f"Document:\n{doc.page_content}\n\n"
            f"Question: {question}\n\n"
            "Answer with exactly one word: 'yes' if the document contains "
            "information that could help answer the question, otherwise 'no'."
        )
        response = llm.invoke(prompt).content.strip().lower()
        if "yes" in response:
            relevant_docs.append(doc)

    return {**state, "documents": relevant_docs}


def decide_to_generate(state: GraphState) -> str:
    if state["documents"]:
        return "generate"
    if state["rewrite_count"] >= MAX_REWRITES:
        # Give up rewriting; answer with whatever we have (likely "I don't know")
        return "generate"
    return "rewrite_query"


def rewrite_query(state: GraphState) -> GraphState:
    prompt = (
        "The following question did not retrieve relevant results from a "
        "WhatsApp chat archive. Rewrite it to be more specific and likely to "
        "match how people actually phrase things in casual chat, while "
        "preserving the original intent.\n\n"
        f"Original question: {state['question']}\n\n"
        "Rewritten question (just the question, nothing else):"
    )
    new_question = llm.invoke(prompt).content.strip()
    return {
        **state,
        "question": new_question,
        "rewrite_count": state["rewrite_count"] + 1,
    }


def generate(state: GraphState) -> GraphState:
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    is_profile = any(d.metadata.get("source") == "whole_conversation_profile" for d in state["documents"])

    if is_profile:
        context_note = (
            "The context below is a statistical profile covering the ENTIRE "
            "conversation for each participant (computed from all their "
            "messages, not a snippet), plus a few representative example "
            "messages. Base your answer on these whole-conversation patterns."
        )
    else:
        context_note = (
            "The context below consists of excerpts retrieved from specific "
            "days of the conversation that are most relevant to the question."
        )

    prompt = (
        "You are analyzing a WhatsApp chat history to answer a question about "
        f"the conversation. {context_note} Use ONLY the context below. If the "
        "context doesn't contain enough information, say so honestly instead "
        "of guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_question']}\n\n"
        "Answer:"
    )
    answer = llm.invoke(prompt).content.strip()
    return {**state, "generation": answer}


def check_grounded(state: GraphState) -> GraphState:
    """Verify the generated answer is actually supported by the retrieved
    context, rather than the model inventing details (hallucination check)."""
    if not state["documents"]:
        return {**state, "grounded": False}

    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    prompt = (
        "Does the ANSWER below rely only on facts present in the CONTEXT, "
        "with no invented details? Answer with exactly one word: 'yes' or 'no'.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{state['generation']}"
    )
    response = llm.invoke(prompt).content.strip().lower()
    grounded = "yes" in response
    return {**state, "grounded": grounded}


# ---------- Build the graph ----------

def route_by_scope(state: GraphState) -> str:
    return "profile" if state["scope"] == "broad" else "retrieve"


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_scope", classify_scope)
    workflow.add_node("build_profile_context", build_profile_context)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_grounded", check_grounded)

    workflow.set_entry_point("classify_scope")
    workflow.add_conditional_edges(
        "classify_scope",
        route_by_scope,
        {"profile": "build_profile_context", "retrieve": "retrieve"},
    )
    workflow.add_edge("build_profile_context", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "check_grounded")
    workflow.add_edge("check_grounded", END)

    return workflow.compile()


def ask(question: str, df: Optional[pd.DataFrame] = None) -> dict:
    """Convenience wrapper: run the graph and return the answer + metadata.

    Pass the enriched messages DataFrame (df) so broad/holistic questions
    can be answered from a whole-conversation profile instead of a handful
    of retrieved chunks. If df is omitted, broad questions will fall back to
    an empty profile and the model will say it lacks enough context.
    """
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
        "df": df,
        "scope": "",
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["generation"],
        "grounded": result["grounded"],
        "sources": [d.metadata for d in result["documents"]],
        "rewrites_used": result["rewrite_count"],
        "scope": result["scope"],
    }


if __name__ == "__main__":
    import sys

    from parser import parse_whatsapp_export
    from enrich import enrich_messages, to_dataframe

    q = sys.argv[1] if len(sys.argv) > 1 else "How did our conversations change over time?"
    msgs = parse_whatsapp_export("sample_chat.txt")
    df = to_dataframe(enrich_messages(msgs))

    print(f"Q: {q}\n")
    result = ask(q, df=df)
    print(f"A: {result['answer']}\n")
    print(f"Scope: {result['scope']}")
    print(f"Grounded: {result['grounded']}")
    print(f"Rewrites used: {result['rewrites_used']}")
    print(f"Sources: {result['sources']}")

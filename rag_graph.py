
"""
rag_graph.py

ChatLens-RAG retrieval and reasoning pipeline.

Architecture:

    User Question
          |
          v
    classify_scope
       /       \
   broad       specific
     |             |
     v             v
 whole-chat     hybrid retrieval
 profile          |
                  v
              grade docs
                  |
          +-------+-------+
          |               |
       relevant        irrelevant
          |               |
          v               v
    expand context      rewrite
          |               |
          v               |
       generate <---------+
          |
          v
    grounded check
          |
          v
         END


Retrieval architecture:

    Chroma
      |
      | semantic similarity
      v
    candidate chunks
      |
      +-------------------+
      |                   |
      v                   v
   semantic           SQLite FTS5
                       keyword
      |                   |
      +---------+---------+
                |
                v
         candidate pool
                |
                v
         LLM relevance grade
                |
                v
       SQLite context expansion
                |
                v
             Qwen


Important:

    Chroma = semantic locator
    SQLite = source of truth
    SQLite FTS5 = exact/keyword retrieval
    LLM = reasoning + grading + groundedness


Requirements:

    ollama pull qwen2.5:3b

    Chroma vector store must already exist.

    chat_data.db must already exist.
"""


# =====================================================================
# Imports
# =====================================================================

from typing import List, Optional, TypedDict
from datetime import datetime

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from vectorstore import load_vectorstore

from analytics import build_style_profile

from chat_database import (
    search_messages,
    get_message_range,
)


# =====================================================================
# Configuration
# =====================================================================

MAX_REWRITES = 2

SEMANTIC_TOP_K = 10

KEYWORD_TOP_K = 10

MAX_HYBRID_CANDIDATES = 20

CONTEXT_BEFORE = 10

CONTEXT_AFTER = 10


# =====================================================================
# LLM
# =====================================================================

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


# =====================================================================
# Graph state
# =====================================================================

class GraphState(TypedDict):
    question: str

    original_question: str

    documents: List[Document]

    generation: str

    rewrite_count: int

    grounded: bool

    df: Optional[pd.DataFrame]

    scope: str


# =====================================================================
# Scope classification
# =====================================================================

def classify_scope(
    state: GraphState,
) -> GraphState:
    """
    Decide whether the question requires a whole-conversation
    analysis or a specific retrieval-based answer.

    broad:
        Questions about overall patterns, communication style,
        general tone, personality-like behavioral patterns,
        changes over time, etc.

    specific:
        Questions about particular messages, dates, events,
        topics, people, or exact things said.
    """

    question = state["question"]

    prompt = (
        "Classify the following question about a WhatsApp "
        "conversation as exactly one word: 'broad' or 'specific'.\n\n"

        "'broad' means the user is asking about overall conversation "
        "patterns, communication style, general mood, general tone, "
        "changes over time, or something that requires looking across "
        "the whole conversation.\n\n"

        "'specific' means the user is asking about a particular "
        "message, topic, event, date, person, plan, or exact thing "
        "someone said.\n\n"

        f"Question: {question}\n\n"

        "Answer with exactly one word:"
    )

    response = llm.invoke(
        prompt
    ).content.strip().lower()

    if "broad" in response:

        scope = "broad"

    else:

        scope = "specific"

    return {
        **state,
        "scope": scope,
    }


# =====================================================================
# Whole-conversation profile
# =====================================================================

def build_profile_context(
    state: GraphState,
) -> GraphState:
    """
    Build a synthetic document containing whole-conversation
    statistics for broad questions.
    """

    df = state.get("df")

    if df is None or df.empty:

        return {
            **state,
            "documents": [],
        }

    profile_text = build_style_profile(
        df
    )

    document = Document(
        page_content=profile_text,

        metadata={
            "source": "whole_conversation_profile",
        },
    )

    return {
        **state,
        "documents": [document],
    }


# =====================================================================
# Expand one semantic document
# =====================================================================

def expand_document_context(
    document: Document,
    before: int = CONTEXT_BEFORE,
    after: int = CONTEXT_AFTER,
) -> Document:
    """
    Expand a Chroma result with surrounding original messages.

    Chroma provides the relevant conversation section.

    SQLite provides the original messages around that section.
    """

    metadata = document.metadata

    start_message_id = metadata.get(
        "start_message_id"
    )

    end_message_id = metadata.get(
        "end_message_id"
    )

    # -------------------------------------------------------------
    # Not a semantic chunk
    # -------------------------------------------------------------

    if (
        start_message_id is None
        or end_message_id is None
    ):

        return document

    start_message_id = int(
        start_message_id
    )

    end_message_id = int(
        end_message_id
    )

    expanded_start = max(
        0,
        start_message_id - before,
    )

    expanded_end = (
        end_message_id + after
    )

    rows = get_message_range(
        expanded_start,
        expanded_end,
    )

    if not rows:

        return document

    # -------------------------------------------------------------
    # Format transcript
    # -------------------------------------------------------------

    lines = []

    for row in rows:

        timestamp = row["timestamp"]

        try:

            dt = datetime.fromisoformat(
                timestamp
            )

            timestamp_text = dt.strftime(
                "%d/%m/%Y %I:%M %p"
            )

        except Exception:

            timestamp_text = (
                timestamp or ""
            )

        sender = row["sender"] or ""

        message = row["message"] or ""

        lines.append(
            f"[ID {row['id']}] "
            f"[{timestamp_text}] "
            f"{sender}: "
            f"{message}"
        )

    expanded_transcript = "\n".join(
        lines
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    new_metadata = dict(
        metadata
    )

    new_metadata[
        "context_expanded"
    ] = True

    new_metadata[
        "expanded_start_message_id"
    ] = expanded_start

    new_metadata[
        "expanded_end_message_id"
    ] = expanded_end

    new_metadata[
        "context_messages_before"
    ] = before

    new_metadata[
        "context_messages_after"
    ] = after

    return Document(
        page_content=expanded_transcript,
        metadata=new_metadata,
    )


# =====================================================================
# Keyword retrieval
# =====================================================================

def keyword_retrieve(
    question: str,
    limit: int = KEYWORD_TOP_K,
) -> List[Document]:
    """
    Retrieve exact/keyword matches from SQLite FTS5.

    This is complementary to semantic retrieval.

    Useful for:

        names
        exact words
        unusual spellings
        slang
        Hinglish
        exact phrases
        dates
        short expressions
    """

    rows = search_messages(
        question,
        limit=limit,
    )

    documents = []

    for row in rows:

        timestamp = row["timestamp"]

        sender = row["sender"] or ""

        message = row["message"] or ""

        content = (
            f"[ID {row['id']}] "
            f"[{timestamp}] "
            f"{sender}: "
            f"{message}"
        )

        documents.append(
            Document(
                page_content=content,

                metadata={
                    "source": "keyword_search",

                    "message_id": row["id"],

                    "timestamp": timestamp,

                    "sender": sender,

                    "fts_rank": row["fts_rank"],
                },
            )
        )

    return documents


# =====================================================================
# Hybrid retrieval
# =====================================================================

def retrieve(
    state: GraphState,
) -> GraphState:
    """
    Hybrid retrieval.

    Semantic:
        Chroma finds conceptually similar conversation sections.

    Keyword:
        SQLite FTS5 finds exact lexical matches.

    Results are combined into one candidate pool.
    """

    question = state["question"]

    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID RETRIEVAL"
    )

    print(
        "=" * 70
    )

    print(
        f"Question: {question}"
    )

    # -------------------------------------------------------------
    # Semantic retrieval
    # -------------------------------------------------------------

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": SEMANTIC_TOP_K,
        }
    )

    semantic_docs = retriever.invoke(
        question
    )

    # -------------------------------------------------------------
    # Keyword retrieval
    # -------------------------------------------------------------

    keyword_docs = keyword_retrieve(
        question,
        limit=KEYWORD_TOP_K,
    )

    # -------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------

    combined = []

    seen_semantic_ranges = set()

    seen_keyword_messages = set()

    # -------------------------------------------------------------
    # Semantic results
    # -------------------------------------------------------------

    for document in semantic_docs:

        start_id = document.metadata.get(
            "start_message_id"
        )

        end_id = document.metadata.get(
            "end_message_id"
        )

        key = (
            start_id,
            end_id,
        )

        if key in seen_semantic_ranges:

            continue

        seen_semantic_ranges.add(
            key
        )

        combined.append(
            document
        )

    # -------------------------------------------------------------
    # Keyword results
    # -------------------------------------------------------------

    for document in keyword_docs:

        message_id = document.metadata.get(
            "message_id"
        )

        if message_id in seen_keyword_messages:

            continue

        seen_keyword_messages.add(
            message_id
        )

        combined.append(
            document
        )

    # -------------------------------------------------------------
    # Bound candidate pool
    # -------------------------------------------------------------

    combined = combined[
        :MAX_HYBRID_CANDIDATES
    ]

    print(
        f"Semantic candidates: "
        f"{len(semantic_docs)}"
    )

    print(
        f"Keyword candidates:  "
        f"{len(keyword_docs)}"
    )

    print(
        f"Combined candidates:  "
        f"{len(combined)}"
    )

    return {
        **state,
        "documents": combined,
    }


# =====================================================================
# Grade retrieved documents
# =====================================================================

def grade_documents(
    state: GraphState,
) -> GraphState:
    """
    Ask the LLM whether each candidate contains information useful
    for answering the question.

    Context expansion happens AFTER grading so that the grader does
    not have to process unnecessarily large contexts.
    """

    question = state["question"]

    relevant_docs = []

    print(
        "\nGrading retrieved candidates..."
    )

    for index, document in enumerate(
        state["documents"],
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "semantic_search",
        )

        prompt = (
            "You are a relevance grader for a WhatsApp "
            "conversation retrieval system.\n\n"

            "Determine whether the document contains information "
            "that could help answer the question.\n\n"

            "Answer with exactly one word:\n"
            "'yes' = relevant\n"
            "'no' = not relevant\n\n"

            f"Question:\n{question}\n\n"

            f"Document:\n"
            f"{document.page_content}\n\n"

            "Decision:"
        )

        try:

            response = llm.invoke(
                prompt
            ).content.strip().lower()

        except Exception as e:

            print(
                f"  Candidate {index}: "
                f"LLM grading failed: {e}"
            )

            continue

        relevant = response.startswith(
            "yes"
        )

        print(
            f"  Candidate {index}: "
            f"{'RELEVANT' if relevant else 'not relevant'} "
            f"[{source}]"
        )

        if relevant:

            relevant_docs.append(
                document
            )

    print(
        f"Relevant documents: "
        f"{len(relevant_docs)}"
    )

    return {
        **state,
        "documents": relevant_docs,
    }


# =====================================================================
# Decide whether to generate or rewrite
# =====================================================================

def decide_to_generate(
    state: GraphState,
) -> str:
    """
    Decide what happens after document grading.
    """

    if state["documents"]:

        return "generate"

    if (
        state["rewrite_count"]
        >= MAX_REWRITES
    ):

        return "generate"

    return "rewrite_query"


# =====================================================================
# Rewrite query
# =====================================================================

def rewrite_query(
    state: GraphState,
) -> GraphState:
    """
    Rewrite a failed retrieval query into language more likely to
    match the WhatsApp archive.
    """

    original_question = (
        state["question"]
    )

    prompt = (
        "Rewrite the following question so that it is more likely "
        "to retrieve useful messages from a WhatsApp chat archive.\n\n"

        "Preserve the user's original intent.\n"

        "Use natural conversational terms that people might actually "
        "use in WhatsApp messages.\n\n"

        f"Original question:\n"
        f"{original_question}\n\n"

        "Return only the rewritten question."
    )

    new_question = llm.invoke(
        prompt
    ).content.strip()

    if not new_question:

        new_question = original_question

    print(
        f"\nQuery rewrite "
        f"{state['rewrite_count'] + 1}:"
    )

    print(
        f"  {new_question}"
    )

    return {
        **state,

        "question": new_question,

        "rewrite_count": (
            state["rewrite_count"] + 1
        ),
    }


# =====================================================================
# Expand relevant documents
# =====================================================================

def expand_context(
    state: GraphState,
) -> GraphState:
    """
    Expand relevant retrieval results using SQLite.

    Semantic result:
        expand using its start/end message IDs.

    Keyword result:
        expand around its individual message ID.
    """

    expanded_documents = []

    for document in state["documents"]:

        source = document.metadata.get(
            "source"
        )

        # ---------------------------------------------------------
        # Semantic Chroma result
        # ---------------------------------------------------------

        if source != "keyword_search":

            expanded_document = (
                expand_document_context(
                    document,
                    before=CONTEXT_BEFORE,
                    after=CONTEXT_AFTER,
                )
            )

        # ---------------------------------------------------------
        # Keyword SQLite result
        # ---------------------------------------------------------

        else:

            message_id = document.metadata.get(
                "message_id"
            )

            if message_id is None:

                expanded_document = document

            else:

                message_id = int(
                    message_id
                )

                start_id = max(
                    0,
                    message_id - CONTEXT_BEFORE,
                )

                end_id = (
                    message_id + CONTEXT_AFTER
                )

                rows = get_message_range(
                    start_id,
                    end_id,
                )

                if not rows:

                    expanded_document = (
                        document
                    )

                else:

                    lines = []

                    for row in rows:

                        timestamp = (
                            row["timestamp"]
                        )

                        sender = (
                            row["sender"]
                            or ""
                        )

                        message = (
                            row["message"]
                            or ""
                        )

                        lines.append(
                            f"[ID {row['id']}] "
                            f"[{timestamp}] "
                            f"{sender}: "
                            f"{message}"
                        )

                    expanded_document = (
                        Document(
                            page_content="\n".join(
                                lines
                            ),

                            metadata={
                                **document.metadata,

                                "context_expanded": True,

                                "expanded_start_message_id":
                                    start_id,

                                "expanded_end_message_id":
                                    end_id,

                                "context_messages_before":
                                    CONTEXT_BEFORE,

                                "context_messages_after":
                                    CONTEXT_AFTER,
                            },
                        )
                    )

        expanded_documents.append(
            expanded_document
        )

    print(
        f"\nContext expansion:"
        f" {len(expanded_documents)} "
        f"documents expanded."
    )

    return {
        **state,
        "documents": expanded_documents,
    }


# =====================================================================
# Generate answer
# =====================================================================

def generate(
    state: GraphState,
) -> GraphState:
    """
    Generate the final answer using the original question and the
    retrieved/expanded context.
    """

    documents = state["documents"]

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    is_profile = any(
        document.metadata.get("source")
        == "whole_conversation_profile"

        for document in documents
    )

    # -------------------------------------------------------------
    # Context description
    # -------------------------------------------------------------

    if is_profile:

        context_note = (
            "The context below is a whole-conversation statistical "
            "profile covering the participants and the conversation "
            "rather than a small retrieved excerpt."
        )

    else:

        context_note = (
            "The context below contains conversation sections "
            "retrieved using semantic and keyword search. Relevant "
            "sections have been expanded using surrounding original "
            "WhatsApp messages to preserve conversational context."
        )

    # -------------------------------------------------------------
    # No context
    # -------------------------------------------------------------

    if not context.strip():

        prompt = (
            "You are answering a question about a WhatsApp "
            "conversation.\n\n"

            "There is no reliable retrieved context available.\n\n"

            f"Question:\n"
            f"{state['original_question']}\n\n"

            "Do not invent an answer. Clearly say that the "
            "available chat context is insufficient."
        )

    else:

        prompt = (
            "You are analyzing a WhatsApp conversation.\n\n"

            f"{context_note}\n\n"

            "IMPORTANT RULES:\n"

            "1. Use ONLY information contained in the context.\n"
            "2. Do not invent names, dates, events, motivations, "
            "or statements.\n"
            "3. If the context is insufficient, say so.\n"
            "4. Distinguish direct evidence from reasonable "
            "interpretation.\n"
            "5. When discussing a sequence of messages, preserve "
            "the chronological order.\n"
            "6. Do not assume that every message in an expanded "
            "context section is directly relevant.\n\n"

            f"CONTEXT:\n"
            f"{context}\n\n"

            f"USER QUESTION:\n"
            f"{state['original_question']}\n\n"

            "ANSWER:"
        )

    try:

        answer = llm.invoke(
            prompt
        ).content.strip()

    except Exception as e:

        answer = (
            "I was unable to generate the answer "
            f"because the local language model returned "
            f"an error: {e}"
        )

    return {
        **state,
        "generation": answer,
    }


# =====================================================================
# Groundedness check
# =====================================================================

def check_grounded(
    state: GraphState,
) -> GraphState:
    """
    Check whether the generated answer is supported by the retrieved
    context.
    """

    documents = state["documents"]

    if not documents:

        return {
            **state,
            "grounded": False,
        }

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    answer = state["generation"]

    prompt = (
        "You are a factual groundedness checker.\n\n"

        "Determine whether the answer is supported by the supplied "
        "conversation context.\n\n"

        "Answer with exactly one word:\n"
        "'yes' = the answer is supported by the context\n"
        "'no' = the answer contains unsupported or invented claims\n\n"

        f"CONTEXT:\n"
        f"{context}\n\n"

        f"ANSWER:\n"
        f"{answer}\n\n"

        "Decision:"
    )

    try:

        response = llm.invoke(
            prompt
        ).content.strip().lower()

        grounded = response.startswith(
            "yes"
        )

    except Exception:

        grounded = False

    return {
        **state,
        "grounded": grounded,
    }


# =====================================================================
# Scope routing
# =====================================================================

def route_by_scope(
    state: GraphState,
) -> str:

    if state["scope"] == "broad":

        return "profile"

    return "retrieve"


# =====================================================================
# Build graph
# =====================================================================

def build_graph():

    workflow = StateGraph(
        GraphState
    )

    # -------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------

    workflow.add_node(
        "classify_scope",
        classify_scope,
    )

    workflow.add_node(
        "build_profile_context",
        build_profile_context,
    )

    workflow.add_node(
        "retrieve",
        retrieve,
    )

    workflow.add_node(
        "grade_documents",
        grade_documents,
    )

    workflow.add_node(
        "expand_context",
        expand_context,
    )

    workflow.add_node(
        "rewrite_query",
        rewrite_query,
    )

    workflow.add_node(
        "generate",
        generate,
    )

    workflow.add_node(
        "check_grounded",
        check_grounded,
    )

    # -------------------------------------------------------------
    # Entry
    # -------------------------------------------------------------

    workflow.set_entry_point(
        "classify_scope"
    )

    # -------------------------------------------------------------
    # Scope routing
    # -------------------------------------------------------------

    workflow.add_conditional_edges(
        "classify_scope",
        route_by_scope,
        {
            "profile":
                "build_profile_context",

            "retrieve":
                "retrieve",
        },
    )

    # -------------------------------------------------------------
    # Broad path
    # -------------------------------------------------------------

    workflow.add_edge(
        "build_profile_context",
        "generate",
    )

    # -------------------------------------------------------------
    # Retrieval path
    # -------------------------------------------------------------

    workflow.add_edge(
        "retrieve",
        "grade_documents",
    )

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate":
                "expand_context",

            "rewrite_query":
                "rewrite_query",
        },
    )

    # -------------------------------------------------------------
    # Context expansion
    # -------------------------------------------------------------

    workflow.add_edge(
        "expand_context",
        "generate",
    )

    # -------------------------------------------------------------
    # Query rewrite loop
    # -------------------------------------------------------------

    workflow.add_edge(
        "rewrite_query",
        "retrieve",
    )

    # -------------------------------------------------------------
    # Groundedness
    # -------------------------------------------------------------

    workflow.add_edge(
        "generate",
        "check_grounded",
    )

    workflow.add_edge(
        "check_grounded",
        END,
    )

    return workflow.compile()


# =====================================================================
# Public ask function
# =====================================================================

def ask(
    question: str,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run the complete ChatLens-RAG pipeline.

    Parameters
    ----------
    question:
        User's question.

    df:
        Enriched messages DataFrame.

        Required for broad/whole-conversation analysis.

    Returns
    -------
    dict
        answer
        grounded
        sources
        rewrites_used
        scope
    """

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    graph = build_graph()

    initial_state: GraphState = {

        "question":
            question.strip(),

        "original_question":
            question.strip(),

        "documents":
            [],

        "generation":
            "",

        "rewrite_count":
            0,

        "grounded":
            False,

        "df":
            df,

        "scope":
            "",
    }

    result = graph.invoke(
        initial_state
    )

    return {

        "answer":
            result["generation"],

        "grounded":
            result["grounded"],

        "sources":
            [
                document.metadata
                for document
                in result["documents"]
            ],

        "rewrites_used":
            result["rewrite_count"],

        "scope":
            result["scope"],
    }


# =====================================================================
# Standalone test
# =====================================================================

if __name__ == "__main__":

    import sys

    from parser import (
        parse_whatsapp_export,
    )

    from enrich import (
        enrich_messages,
        to_dataframe,
    )

    print(
        "=" * 70
    )

    print(
        "ChatLens-RAG Test"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Question
    # -------------------------------------------------------------

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What did we discuss about sports?"
    )

    print(
        f"\nQuestion:\n{question}"
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
        f"Parsed {len(msgs)} messages."
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
        f"Enriched {len(enriched)} messages."
    )

    # -------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------

    df = to_dataframe(
        enriched
    )

    # -------------------------------------------------------------
    # Run
    # -------------------------------------------------------------

    print(
        "\nRunning RAG pipeline..."
    )

    try:

        result = ask(
            question,
            df=df,
        )

    except Exception as e:

        print(
            "\n❌ RAG pipeline failed:"
        )

        print(
            e
        )

        raise

    # -------------------------------------------------------------
    # Result
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAnswer:\n{result['answer']}"
    )

    print(
        f"\nScope: "
        f"{result['scope']}"
    )

    print(
        f"Grounded: "
        f"{result['grounded']}"
    )

    print(
        f"Rewrites used: "
        f"{result['rewrites_used']}"
    )

    print(
        "\nSources:"
    )

    for source in result["sources"]:

        print(
            source
        )

    print(
        "\n✓ RAG test completed."
    )


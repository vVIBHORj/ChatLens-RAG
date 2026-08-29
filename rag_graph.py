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

from typing import List, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from vectorstore import load_vectorstore

MAX_REWRITES = 2
llm = ChatOllama(model="llama3.2", temperature=0)


class GraphState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    rewrite_count: int
    grounded: bool


# ---------- Nodes ----------

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
    prompt = (
        "You are analyzing a WhatsApp chat history to answer a question about "
        "the conversation. Use ONLY the context below. If the context doesn't "
        "contain enough information, say so honestly instead of guessing.\n\n"
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

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_grounded", check_grounded)

    workflow.set_entry_point("retrieve")
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


def ask(question: str) -> dict:
    """Convenience wrapper: run the graph and return the answer + metadata."""
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
        "grounded": False,
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["generation"],
        "grounded": result["grounded"],
        "sources": [d.metadata for d in result["documents"]],
        "rewrites_used": result["rewrite_count"],
    }


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "How did our conversations change over time?"
    print(f"Q: {q}\n")
    result = ask(q)
    print(f"A: {result['answer']}\n")
    print(f"Grounded: {result['grounded']}")
    print(f"Rewrites used: {result['rewrites_used']}")
    print(f"Sources: {result['sources']}")
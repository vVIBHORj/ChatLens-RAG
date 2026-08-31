from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://127.0.0.1:11434",
)

texts = [
    f"This is WhatsApp test message number {i}. "
    f"The conversation contains some normal text "
    f"for semantic embedding testing."
    for i in range(500)
]

print("Embedding 100 documents...")

result = embeddings.embed_documents(texts)

print("SUCCESS")
print("Embeddings:", len(result))
print("Dimensions:", len(result[0]))
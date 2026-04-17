from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
doc_embeddings = []

def ingest_docs(files):
    global documents, doc_embeddings
    documents, doc_embeddings = [], []

    for file in files:
        text = file.read().decode("utf-8", errors="ignore")

        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        for chunk in chunks:
            documents.append(chunk)
            emb = embed_model.encode(chunk)
            doc_embeddings.append(emb)

    return f"{len(documents)} chunks indexed."


def retrieve_context(query, top_k=3):
    if not documents:
        return ""

    query_emb = embed_model.encode(query)

    similarities = [
        np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        for emb in doc_embeddings
    ]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    context = "\n\n".join([documents[i] for i in top_indices])

    return context
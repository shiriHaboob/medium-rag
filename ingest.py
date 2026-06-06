import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
OVERLAP = int(CHUNK_SIZE * OVERLAP_RATIO)

EMBEDDING_BATCH_SIZE = 256
PINECONE_BATCH_SIZE = 100

load_dotenv()

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

embeddings = OpenAIEmbeddings(
    model="4UHRUIN-text-embedding-3-small",
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


df = pd.read_csv("data/medium-english-50mb.csv")
df_sample = df

texts = []
metadata_list = []
ids = []

for article_number, (article_id, row) in enumerate(df_sample.iterrows(), start=1):
    title = str(row["title"])
    text = str(row["text"])
    authors = str(row["authors"])
    url = str(row["url"])
    tags = str(row["tags"])

    chunks = chunk_text(text)

    print(f"Preparing article {article_number}/{len(df_sample)} | chunks: {len(chunks)}")

    for chunk_index, chunk in enumerate(chunks):
        texts.append(chunk)
        ids.append(f"article-{article_id}-chunk-{chunk_index}")
        metadata_list.append({
            "article_id": str(article_id),
            "title": title,
            "authors": authors,
            "url": url,
            "tags": tags,
            "chunk": chunk
        })

print(f"Total chunks prepared: {len(texts)}")
print("Creating embeddings in batches...")

total_uploaded = 0

for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
    end = start + EMBEDDING_BATCH_SIZE

    batch_texts = texts[start:end]
    batch_ids = ids[start:end]
    batch_metadata = metadata_list[start:end]

    print(f"Embedding chunks {start} to {min(end, len(texts))}")

    vectors = embeddings.embed_documents(batch_texts)

    pinecone_vectors = []

    for vector_id, vector, metadata in zip(batch_ids, vectors, batch_metadata):
        pinecone_vectors.append({
            "id": vector_id,
            "values": vector,
            "metadata": metadata
        })

    for i in range(0, len(pinecone_vectors), PINECONE_BATCH_SIZE):
        batch = pinecone_vectors[i:i + PINECONE_BATCH_SIZE]
        index.upsert(batch)
        total_uploaded += len(batch)
        print(f"Uploaded {total_uploaded} vectors so far")

print("Batch ingestion completed successfully!")
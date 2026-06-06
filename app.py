import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from prompts import RAG_PROMPT

CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 7
RETRIEVAL_K = 30

load_dotenv()

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = os.getenv("LLMOD_BASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

app = Flask(__name__)

embeddings = OpenAIEmbeddings(
    model="4UHRUIN-text-embedding-3-small",
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL
)

llm = ChatOpenAI(
    model="4UHRUIN-gpt-5-mini",
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL,
    temperature=1
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)


def ask_rag(question):
    query_vector = embeddings.embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=RETRIEVAL_K,
        include_metadata=True
    )

    matches = results["matches"]

    # Special improvement for topic-list questions about education:
    # Prefer articles whose tags explicitly contain Education.
    if "education" in question.lower():
        education_matches = []
        other_matches = []

        for match in matches:
            metadata = match["metadata"]
            tags = str(metadata.get("tags", "")).lower()
            title = str(metadata.get("title", "")).lower()
            chunk = str(metadata.get("chunk", "")).lower()

            if "education" in tags:
                education_matches.append(match)
            elif "education" in title:
                education_matches.append(match)
            elif "education" in chunk:
                other_matches.append(match)
            else:
                other_matches.append(match)

        matches = education_matches + other_matches

    context_items = []
    context_texts = []
    seen_articles = set()

    for match in matches:
        metadata = match["metadata"]

        article_key = metadata.get("article_id", metadata.get("title", ""))

        if article_key in seen_articles:
            continue

        seen_articles.add(article_key)

        item = {
            "article_id": metadata.get("article_id", ""),
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", ""),
            "tags": metadata.get("tags", ""),
            "chunk": metadata.get("chunk", ""),
            "score": match["score"]
        }

        context_items.append(item)

        context_texts.append(
            f"Article ID: {item['article_id']}\n"
            f"Title: {item['title']}\n"
            f"Authors: {item['authors']}\n"
            f"Tags: {item['tags']}\n"
            f"Chunk: {item['chunk']}"
        )

        if len(context_items) >= TOP_K:
            break

    context_text = "\n\n---\n\n".join(context_texts)

    system_prompt = RAG_PROMPT.messages[0].prompt.template
    user_prompt = RAG_PROMPT.messages[1].prompt.template.format(
        context=context_text,
        question=question
    )

    messages = RAG_PROMPT.invoke({
        "context": context_text,
        "question": question
    })

    response = llm.invoke(messages)

    return {
        "response": response.content,
        "context": context_items,
        "Augmented_prompt": {
            "System": system_prompt,
            "User": user_prompt
        }
    }


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    })


@app.route("/api/prompt", methods=["POST"])
def prompt():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    result = ask_rag(data["question"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
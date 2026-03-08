"""
Flask + LCEL Demo API
Exposes three endpoints that demonstrate different LangChain/LCEL patterns
backed by a local Ollama model.
"""

import os

from flask import Flask, jsonify, request
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

app = Flask(__name__)

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=MODEL, base_url=BASE_URL)
parser = StrOutputParser()


# ---------------------------------------------------------------------------
# Endpoint 1: Basic Chat  (demonstrates a simple sequential LCEL chain)
# ---------------------------------------------------------------------------

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Keep answers concise."),
    ("human", "{message}"),
])

chat_chain = chat_prompt | llm | parser


@app.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    message = body.get("message")
    if not message:
        return jsonify({"error": "Missing 'message' field"}), 400

    result = chat_chain.invoke({"message": message})
    return jsonify({"response": result})


# ---------------------------------------------------------------------------
# Endpoint 2: Summarize  (demonstrates a different prompt template)
# ---------------------------------------------------------------------------

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a summarization engine. "
     "Summarize the following text in 2-3 bullet points. "
     "Return only the bullet points, nothing else."),
    ("human", "{text}"),
])

summarize_chain = summarize_prompt | llm | parser


@app.route("/api/summarize", methods=["POST"])
def summarize():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = summarize_chain.invoke({"text": text})
    return jsonify({"summary": result})


# ---------------------------------------------------------------------------
# Endpoint 3: Analyze  (demonstrates RunnableParallel — three tasks at once)
# ---------------------------------------------------------------------------

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the text in one sentence."),
    ("human", "{text}"),
])

sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the sentiment of the text as one of: "
     "positive, negative, or neutral. Return only the label."),
    ("human", "{text}"),
])

keywords_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract 3-5 keywords from the text. "
     "Return them as a comma-separated list."),
    ("human", "{text}"),
])

analyze_chain = RunnablePassthrough() | RunnableParallel(
    summary=summary_prompt | llm | parser,
    sentiment=sentiment_prompt | llm | parser,
    keywords=keywords_prompt | llm | parser,
)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = analyze_chain.invoke({"text": text})
    return jsonify(result)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL})


if __name__ == "__main__":
    print(f"  Model  : {MODEL}")
    print(f"  Ollama : {BASE_URL}")
    print(f"  Docs   : see README.md for curl examples\n")
    app.run(debug=True, port=5001, threaded=True)

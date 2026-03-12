import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

faq_df = pd.read_csv("dataset/dataset.csv")

questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings
faq_embeddings = model.encode(questions)

# Todo: Add init -> 3 random suggestions for ui ini

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json["question"]
    query_embedding = model.encode([user_question])
    similarities = cosine_similarity(query_embedding, faq_embeddings)
    best_idx = similarities.argmax()

    response = {
        "question": questions[best_idx],
        "answer": answers[best_idx],
        "score": float(similarities[0][best_idx])
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
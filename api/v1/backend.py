import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

faq_df = pd.read_csv("dataset/dataset.csv")

questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

# Load model, Precompute embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(questions, convert_to_numpy=True)

THRESHOLD = 0.5

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    user_question = data["question"].strip()

    if user_question == "":
        return jsonify({"error": "Empty question"}), 400

    # Encode query
    query_embedding = model.encode([user_question], convert_to_numpy=True)

    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]

    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score < THRESHOLD:
        return jsonify({
            "answer": "I'm sorry, I don't have information on that topic.",
        })

    return jsonify({
        "answer": answers[best_idx],
        "question": questions[best_idx],
        "score": float(best_score),
    })


@app.route("/suggest", methods=["GET"])
def suggest():
    sample_indices = random.sample(range(len(questions)), 3)
    suggestions = [questions[i] for i in sample_indices]
    return jsonify({"suggestions": suggestions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
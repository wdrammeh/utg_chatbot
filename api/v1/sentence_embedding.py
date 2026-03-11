from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

faq_df = pd.read_csv("dataset/faq_dataset.csv")

questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings
faq_embeddings = model.encode(questions)


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
    app.run(port=5000)
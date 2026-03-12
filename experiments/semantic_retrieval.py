import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load dataset
dtst_path = "dataset/faq_dataset.csv"
test_dtst_path = "dataset/test_questions.csv"

faq_df = pd.read_csv(dtst_path)

questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

# Init/Load embedding retrieval model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode FAQ questions
faq_embeddings = model.encode(questions)


def retrieve_ans(query):
    # Encode query
    query_embedding = model.encode([query])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, faq_embeddings)

    # Best match
    best_idx = similarities.argmax()

    return questions[best_idx], answers[best_idx], similarities[0][best_idx]


def eval_ser():
    faq_df = pd.read_csv(dtst_path)
    test_df = pd.read_csv(test_dtst_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    faq_embeddings = model.encode(faq_df["question"].tolist())

    correct = 0
    total = len(test_df)

    for q in test_df["question"]:
        query_embedding = model.encode([q])

        sim = cosine_similarity(query_embedding, faq_embeddings)
        best_idx = sim.argmax()

        retrieved_category = faq_df.iloc[best_idx]["category"]

        true_category = test_df[test_df["question"] == q]["category"].values[0]

        if retrieved_category == true_category:
            correct += 1

    accuracy = correct / total
    print("Sentence Embedding Retrieval Accuracy:", accuracy)


# Interactive test
def int_test():
    while True:
        query = input("\nAsk a question (or type q): ")

        if query.lower() == "q":
            break

        matched_q, answer, score = retrieve_ans(query)

        print("\nMatched Question:", matched_q)
        print("Answer:", answer)
        print("Similarity Score:", score)


if __name__ == "__main__":
    # eval_ser()
    int_test()
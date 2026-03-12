import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
dtst_path = "dataset/dataset.csv"
test_dtst_path = "dataset/dataset-test-par.csv"

faq_df = pd.read_csv(dtst_path)

questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

# Build TF-IDF vectorizer - This is a traditional keyword retrieval
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(questions)


def retrieve_ans(query):
    # Convert query to vector
    query_vec = vectorizer.transform([query])

    # Compute similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix)

    # Get best match index
    best_idx = similarity.argmax()

    return questions[best_idx], answers[best_idx], similarity[0][best_idx]


def cb_eval():
    faq_df = pd.read_csv(dtst_path)
    test_df = pd.read_csv(test_dtst_path)

    vectorizer = TfidfVectorizer(stop_words="english")
    faq_vectors = vectorizer.fit_transform(faq_df["question"])

    correct = 0
    total = len(test_df)

    for q in test_df["question"]:
        query_vec = vectorizer.transform([q])
        sim = cosine_similarity(query_vec, faq_vectors)

        best_idx = sim.argmax()
        retrieved_category = faq_df.iloc[best_idx]["category"]
        true_category = test_df[test_df["question"] == q]["category"].values[0]

        if retrieved_category == true_category:
            correct += 1

    accuracy = correct / total
    print("TF-IDF Category-Based Retrieval Accuracy:", accuracy)


def exact_eval():
    faq_df = pd.read_csv(dtst_path)
    test_df = pd.read_csv(test_dtst_path)

    faq_questions = faq_df["question"].tolist()
    faq_answers = faq_df["answer"].tolist()

    test_questions = test_df["question"].tolist()
    test_answers = test_df["answer"].tolist()

    vectorizer = TfidfVectorizer()
    faq_tfidf = vectorizer.fit_transform(faq_questions)

    correct_tfidf = 0

    for q, true_ans in zip(test_questions, test_answers):
        q_vec = vectorizer.transform([q])

        sims = cosine_similarity(q_vec, faq_tfidf)
        best_idx = sims.argmax()
        predicted_ans = faq_answers[best_idx]

        if predicted_ans == true_ans:
            correct_tfidf += 1

    tfidf_accuracy = correct_tfidf / len(test_questions)
    print(f"TF-IDF Exact Answer Retrieval Accuracy: {tfidf_accuracy}") # {tfidf_accuracy:.2f}


def int_test():
    # Interactive test
    while True:
        user_input = input("\nAsk a question (or type q): ")
        if user_input.lower() == "q":
            break
        matched_q, answer, score = retrieve_ans(user_input)

        print("\nMatched Question:", matched_q)
        print("Answer:", answer)
        print("Similarity Score:", score)


if __name__ == "__main__":
    cb_eval()
    exact_eval()
    # int_test()
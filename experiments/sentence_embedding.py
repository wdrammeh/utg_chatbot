import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load dataset
dtst_path = "dataset/dataset.csv"
test_dtst_path = "dataset/testset.csv"

faq_df = pd.read_csv(dtst_path)
test_df = pd.read_csv(test_dtst_path)

questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()
categories = faq_df["category"].tolist()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
faq_embeddings = model.encode(questions, convert_to_numpy=True)


# Retrieval
def embed_retrieve(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)

    similarities = cosine_similarity(query_embedding, faq_embeddings)[0]
    ranked_indices = similarities.argsort()[::-1]

    return ranked_indices[:top_k]


# Category Accuracy
def embed_eval_category():
    correct = 0
    total = len(test_df)

    for _, row in test_df.iterrows():
        q = row["question"]
        true_category = row["category"]

        top_idx = embed_retrieve(q, top_k=1)[0]
        predicted_category = categories[top_idx]

        if predicted_category == true_category:
            correct += 1

    accuracy = correct / total
    print(f"Sentence Embedding Category Accuracy: {accuracy:.4f}")


# Recall@1
def embed_eval_top1():
    correct = 0
    total = len(test_df)

    for _, row in test_df.iterrows():
        q = row["question"]
        true_ans = row["answer"]

        top_idx = embed_retrieve(q, top_k=1)[0]
        predicted_ans = answers[top_idx]

        if predicted_ans == true_ans:
            correct += 1

    accuracy = correct / total
    print(f"Sentence Embedding Recall@1: {accuracy:.4f}")


# Recall@3
def embed_eval_top3():
    correct = 0
    total = len(test_df)

    for _, row in test_df.iterrows():
        q = row["question"]
        true_ans = row["answer"]

        top_indices = embed_retrieve(q, top_k=3)
        predicted_answers = [answers[i] for i in top_indices]

        if true_ans in predicted_answers:
            correct += 1

    accuracy = correct / total
    print(f"Sentence Embedding Recall@3: {accuracy:.4f}")


# Recall@5
def embed_eval_top5():
    correct = 0
    total = len(test_df)

    for _, row in test_df.iterrows():
        q = row["question"]
        true_ans = row["answer"]

        top_indices = embed_retrieve(q, top_k=5)
        predicted_answers = [answers[i] for i in top_indices]

        if true_ans in predicted_answers:
            correct += 1

    accuracy = correct / total
    print(f"Sentence Embedding Recall@5: {accuracy:.4f}")


# MRR
def embed_eval_mrr():
    total = len(test_df)
    reciprocal_ranks = []

    for _, row in test_df.iterrows():
        q = row["question"]
        true_ans = row["answer"]

        query_embedding = model.encode([q], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, faq_embeddings)[0]

        ranked_indices = similarities.argsort()[::-1]

        rank = None
        for idx, i in enumerate(ranked_indices):
            if answers[i] == true_ans:
                rank = idx + 1
                break

        if rank is not None:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    mrr = sum(reciprocal_ranks) / total
    print(f"Sentence Embedding MRR: {mrr:.4f}")


# Interactive test
def int_test():
    while True:
        query = input("\nAsk a question (or type q): ")
        if query.lower() == "q":
            break

        query_embedding = model.encode([query], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, faq_embeddings)[0]

        best_idx = similarities.argmax()

        print("\nAnswer:", answers[best_idx])
        print("Matched Question:", questions[best_idx])
        print("Similarity Score:", similarities[best_idx])


if __name__ == "__main__":
    embed_eval_category()
    embed_eval_top1()
    embed_eval_top3()
    embed_eval_top5()
    embed_eval_mrr()
    # int_test()
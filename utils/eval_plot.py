import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Metrics
    metrics = ["Recall@1", "Recall@3", "Recall@5", "MRR"]

    tfidf_scores = [0.5350, 0.7400, 0.7700, 0.6424]
    embed_scores = [0.7550, 0.9000, 0.9450, 0.8382]

    x = range(len(metrics))

    plt.figure()

    # Plot
    plt.plot(x, tfidf_scores, marker='o', label="TF-IDF")
    plt.plot(x, embed_scores, marker='o', label="Sentence Embedding")

    # Labels
    plt.xticks(x, metrics)
    plt.xlabel("Eval Metric")
    plt.ylabel("Score")
    # plt.title("Performance Comparison of Retrieval Models")
    plt.legend()

    plt.tight_layout()
    plt.savefig("eval-plot.png", dpi=300)
    plt.show()
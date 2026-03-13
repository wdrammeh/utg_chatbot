
if __name__ == "__main__":
    import pandas as pd

    # Load dataset and test set
    dataset = pd.read_csv("dataset/dataset.csv")      # your FAQ dataset
    testset = pd.read_csv("dataset/testset.csv")      # your paraphrased queries

    # Normalize text (lowercase + strip spaces)
    dataset_questions = dataset['question'].str.lower().str.strip()
    test_questions = testset['question'].str.lower().str.strip()

    # Find exact matches
    duplicates = set(dataset_questions).intersection(set(test_questions))

    # Print results
    if duplicates:
        print("Exact matches found:\n")
        for q in duplicates:
            print(q)
        print(f"\nTotal duplicates: {len(duplicates)}")
    else:
        print("No exact matches found between dataset and test set.")
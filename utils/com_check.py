import pandas as pd

if __name__ == "__main__":
    # Load the datasets
    data = pd.read_csv("dataset/dataset.csv")
    test = pd.read_csv("dataset/testset.csv")

    # Get all questions from test
    test_questions = set(test["question"])

    # Remove rows from data where question exists in test
    filtered_data = data[~data["question"].isin(test_questions)]

    # Save the result to a new file
    # filtered_data.to_csv("dataset/corpus.csv", index=False)
    print(filtered_data)
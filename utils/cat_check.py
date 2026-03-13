import pandas as pd

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv("dataset/dataset.csv")

    # data["category"] = data["category"].str.strip()
    
    # Debug
    # print(data["category"].unique())
    print(data.info())

    # Count distinct categories
    category_counts = data["category"].value_counts()

    n = 1
    t = 0
    for category, count in category_counts.items():
        print(f"{n}. {category}: {count}")
        n = n + 1
        t = t + count

    # print("\nTotal distinct categories:", data["category"].nunique())
    print("\nTotal:", t)
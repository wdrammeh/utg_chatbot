import csv
import pandas as pd
from collections import defaultdict


# Check category distribution in a set
def category_check(path):
    data = pd.read_csv(path)

    # data["category"] = data["category"].str.strip()
    
    # Debug
    print(data["category"].unique())
    # print(data.info())

    # Count distinct categories
    category_counts = data["category"].value_counts()

    i = 1
    total = 0
    for category, count in category_counts.items():
        print(f"{i}. {category}: {count}")
        i = i + 1
        total = total + count

    # print("\nTotal distinct categories:", data["category"].nunique())
    print("\nTotal:", total)


# Reveals duplicate question in a set (using question only - no regard to category)
def duplicate_check(set_path):
    # Dictionary to store: {question_text: [list_of_row_numbers]}
    question_map = defaultdict(list)
    duplicates_found = False
    
    print(f"--- Scanning for Duplicates in {set_path} ---")
    
    try:
        with open(set_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Start counting from 2 (Row 1 is the header)
            for row_num, row in enumerate(reader, 2):
                # Normalize: lowercase and strip spaces for a "fuzzy" match
                q_text = row['question'].strip().lower()
                question_map[q_text].append(row_num)
                
        # Iterate through the map to find keys with more than one row index
        for q_text, rows in question_map.items():
            if len(rows) > 1:
                duplicates_found = True
                print(f"DUPLICATE FOUND ({len(rows)} occurrences):")
                print(f"   Question: \"{q_text}\"")
                print(f"   Located at Rows: {', '.join(map(str, rows))}")
                print("-" * 10)

        if not duplicates_found:
            print("Clear! No duplicate questions were detected.")
            
    except FileNotFoundError:
        print(f"Error: The file '{set_path}' was not found.")
    except KeyError:
        print("Error: The CSV must have a 'question' column header.")


# Tells if exact test questions appear in dataset
def duplicate_question_check(set_path, test_path):
    # Load dataset and test set
    dataset = pd.read_csv(set_path)
    testset = pd.read_csv(test_path)

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


# Ensure all test answers are in fact in the dataset (using both question and category)
def test_ans_val(set_path, test_path):
    # We use a set of tuples (Answer, Category) as our unique lookup key.
    # Sets use hash tables, making lookups O(1) regardless of dataset size.
    master_registry = set()
    
    print("--- Phase 1: Indexing Master Dataset ---")
    try:
        with open(set_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Clean whitespace to prevent matching errors
                key = (row['answer'].strip(), row['category'].strip())
                master_registry.add(key)
        print(f"Successfully indexed {len(master_registry)} unique entries.\n")
    except FileNotFoundError:
        print(f"Error: File '{set_path}' not found.")
        return

    print("--- Phase 2: Validating Test Set ---")
    mismatches = []
    total_checked = 0

    try:
        with open(test_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                total_checked += 1
                test_key = (row['answer'].strip(), row['category'].strip())
                
                if test_key not in master_registry:
                    mismatches.append({
                        'row': i,
                        'question': row['question'],
                        'answer': row['answer'],
                        'category': row['category']
                    })
    except FileNotFoundError:
        print(f"Error: File '{test_path}' not found.")
        return

    # --- Final Report ---
    print(f"Total entries checked: {total_checked}")
    
    if not mismatches:
        print("SUCCESS: All test entries match the Master Dataset (Answer + Category).")
    else:
        print(f"FAILURE: Found {len(mismatches)} discrepancies.")
        print("-" * 30)
        for error in mismatches:
            print(f"Line {error['row']}: Key [\"{error['answer']}\" | {error['category']}] not found.")
            print(f"Context: {error['question'][:60]}...")
            print("-" * 30)


if __name__ == "__main__":
    set_path = "dataset/dataset.csv"
    test_path = "dataset/testset.csv"

    category_check(set_path)
    category_check(test_path)

    duplicate_check(set_path)
    duplicate_check(test_path)

    duplicate_question_check(set_path, test_path)

    test_ans_val(set_path, test_path)
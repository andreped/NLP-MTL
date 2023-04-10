import pandas as pd
from typing import List
import os

from model.Review import Review


data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data"))

def load_reviews() -> List[Review]:
    print("Loading dataset ...")
    path = os.path.join(data_dir, "reviews.csv")
    column_names = ["review_id", "product_id", "age", "title", "review_text", "rating", "recommends", "upvotes",
                    "division_name", "department_name", "class_name"]
    data = pd.read_csv(path, delimiter=",", names=column_names)
    data.fillna("")
    dataset = []
    for _, row in data.iterrows():
        try:
            entry = Review(str(row["review_id"]), str(row["product_id"]), int(row["age"]), str(row["title"]),
                           str(row["review_text"]), int(row["rating"]), bool(int(row["recommends"])),
                           int(row["upvotes"]), str(row["division_name"]), str(row["department_name"]), str(row["class_name"]))
            dataset.append(entry)
        except:
            pass
    print(f"Dataset loaded ({len(dataset)} rows).")
    return dataset

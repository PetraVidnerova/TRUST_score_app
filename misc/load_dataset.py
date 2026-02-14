import tqdm 
from datasets import load_dataset
import pandas as pd 

# INPUT_DATASET_NAME = "PetraV77/papers-references-abstracts"
# OUTPUT_PARQUET_FILE = "papers-references-abstracts.parquet"

# INPUT_DATASET_NAME = "PetraV77/papers-references-titles"
# OUTPUT_PARQUET_FILE = "papers-references-titles.parquet"

INPUT_DATASET_NAME = "PetraV77/papers-ref-data"
OUTPUT_PARQUET_FILE = "papers-ref-data.parquet"

print("Loading dataset...")
dataset = load_dataset(INPUT_DATASET_NAME, 
                       split="train")

print("Dataset loaded.") 
dataset.to_parquet(OUTPUT_PARQUET_FILE)
print("Done.")


# print("Converting it to dictionary ....")

# id_to_abstract = {}
# for row in tqdm.tqdm(dataset):
#     id_to_abstract[row["openalexid"]] = row["abstract"]
# #id_to_abstract = {row["openalexid"]: row["abstract"] for row in dataset}

    
# print(id_to_abstract["W1987125112"])



# print("Loading dataset from parquet file...")
# df = pd.read_parquet("papers-references-abstracts.parquet")
# # print(df.head())
# print("Dataset loaded.")

# print("Converting it to dictionary ....")
# id_to_abstract = {}

# for row in df.itertuples(index=False):
#     id_to_abstract[row.openalexid] = row.abstract

# print("Done.")

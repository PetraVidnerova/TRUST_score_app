import pickle
import pandas as pd
from datasets import Dataset, Features, Value
from huggingface_hub import login

# INPUT = "challenge/abstracts_cache.pickle"
# OUTPUT_DATASET_NAME = "papers-references-abstracts"

INPUT = "challenge/titles_cache.pickle"
OUTPUT_DATASET_NAME = "papers-references-titles"

# INPUT = "challenge/ref_data_cache.pickle"
# OUTPUT_DATASET_NAME = "papers-ref-data"


with open(INPUT, "rb") as f:
    abstracts_cache = pickle.load(f)

# for titles only
df = pd.read_csv("challenge/Metadata file COMBINED.csv")
for _, row in df.iterrows():
    openalexid = row["OpenAlexID (as URL)"].replace("https://openalex.org/", "")
    title = row["Title"]
    if not openalexid in abstracts_cache or abstracts_cache[openalexid] is None:
        abstracts_cache[openalexid] = title

print(f"Loaded abstracts cache with {len(abstracts_cache)} entries")

dataset = Dataset.from_generator(
    lambda: (
        {"openalexid": k, "abstract": v}
        for k, v in abstracts_cache.items()
    )
)


print(dataset)

login()
dataset.push_to_hub(
    f"PetraV77/{OUTPUT_DATASET_NAME}",
    private=True # it is possible to change via the Hugging Face website
)

print("Done.")

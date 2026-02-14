---
title: TRUST Score
emoji: 👁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
short_description: simple app for the purpose of novelty challenge
---

<hr>


# 📜 TRUST Score Calculator 


This repository contains the novelty score calculator that for a given scientific paper (OpenAlex ID) returns a real number indicating the predicted novelty of the paper. The higher the score, the higher the novelty prediction. 

You can try the demo on the [HuggingFace space TRUST_score](https://huggingface.co/spaces/PetraV77/TRUST_score) (please do not overload the HF space, it is intended for demonstration purposes only and to demonstrate reproducibility). Recommended usage is to download the source code to your computer and run it locally. See below the [instructions](#how-to-run-the-calculator-locally).

## About the score  



## The TRUST Score calculator App 

1. In case the paper is from the novelty challenge dataset, it finds in prepared datasets on HF the title, abstract and references, as well as titles and abstracts for all references. 
Otherwise: 
    + Fetches via OpenAlex API the title, abstract and references of the paper queried. 
    + Fetches titles and abstracts for referenced papers.
4. Converts couples (title, abstract) into embedding vectors using **allenai/specter2** model.
5. Calculates final scores. 

## How to run the calculator locally 
Local runs are recommended and enable you to process lists of papers efficiently. Since the calculator uses the **specter2** embedding model, it is recommended to run on a GPU (but CPU run is also possible). You will also need your API token to [OpenAlex](https://openalex.org) database to fetch the data via API. Store your API token in the file `openalex_api_key.txt` in the TRUST_score_app directory. 

The easiest way is to use `uv` to run the Python code. 
```sh
uv run calculate_scores.py data/Metadata\ file\ COMBINED.csv
```

Alternatively, you can install the references via pip.
```sh
pip install -r requirements.txt 
```
and run the script via Python
```sh
python calculate_scores.py data/Metadata\ file\ COMBINED.csv
```
The overview of **usage**:
```
Usage: calculate_scores.py [OPTIONS] [FILENAME]

Options:
  --log-level [DEBUG|INFO|WARNING|ERROR]
  --use-api-key
  --force-cpu
  --only-cached                   Only evaluate papers that have cached data
                                  available, skip others.
  --batch-size INTEGER
  --help                          Show this message and exit.
```
By default, the script runs with `device="auto"`, you can supress the use of GPU by `--force-cpu`. 
The `--batch-size` controls the number of (title, abstract) embeddings computed in one batch, you can experiment with this option if you have enough memory. Higher batches help to speed up the computation.

### How to reproduce our results
Since the OpenAlex API is not perfectly stable (sometimes data for a valid ID is not returned, and if it is a reference, our script skips it), you may get slightly different results from ours. If you want to reproduce the results exactly, download the prepared `.parquet` files with the downloaded titles, abstracts, and reference lists, and place them in the `data` directory. Then disable API calls with `--only-cached`.

```sh
uv run calculate_scores.py data/Metadata\ file\ COMBINED.csv --force-cpu --only-cached
```






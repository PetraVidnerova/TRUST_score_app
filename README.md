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

**Author:** Petra Vidnerová, Institute of Computer Science, The Czech Academy of Sciences  
**Acknowledgment**: todo TRUST project 

This repository contains the novelty score calculator that for a given scientific paper (OpenAlex ID) returns a real number indicating the predicted novelty of the paper. The higher the score, the higher the novelty prediction. 

You can try the demo on the [HuggingFace space TRUST_score](https://huggingface.co/spaces/PetraV77/TRUST_score) (please do not overload the HF space, it is just for demonstration purposes). Recommended usage is to download the source code to your computer and run it locally. See below the [instructions](#how-to-run-the-calculator-locally).

## About the score  



## The TRUST Score calculator App 

1. In case the paper is from the novelty challenge dataset, it finds in prepared datasets on HF the title, abstract and references, as well as titles and abstracts for all references. 
Otherwise: 
    + Fetches via OpenAlex API the title, abstract and references of the paper queried. 
    + Fetches titles and abstracts for referenced papers.
4. Converts couples (title, abstract) into embedding vectors using **allenai/specter2** model.
5. Calculates final scores. 

## How to run the calculator locally 
Local run is recommended and it enables you to process lists of papers efficiently. Since the calculator uses **specter2** embedding model, it is recommended to run on GPU (but CPU run is also possible). You will also need your API token to [OpenAlex](https://openalex.org) database to fetch the data via API. Store you API token in file `openalex_api_key.txt` in the TRUST_score_app directory. 

The easies way is to us `uv` to run the python code. 
```sh
uv run calculate_scores.py data/Metadata\ file\ COMBINED.csv
```

Alternatively you can install the references via pip.
```sh
pip install -r requirements.txt 
``




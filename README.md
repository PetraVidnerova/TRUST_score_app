<hr>

# 📜 TRUST Score Calculator 

This README file was generated on 2026-03-17 by Petra Vidnerová, petra@cs.cas.cz
Last updated: 2026-03-17.
Version: 1.0

----------------------
GENERAL INFORMATION
----------------------

- Software title: TRUST Score Calculator

- Software DOI: ZENODO

- Author: Petra Vidnerová, ORCID: https://sandbox.orcid.org/0009-0006-2424-7712

- Affiliation: Institute of Computer Science of the Czech Academy of Science

- ROR: https://ror.org/0496n6574 

- Description: This is an example softwere which doesn't do anything really...except for serving as an example.

- Contributors: 

- Funding sources: This work has been produced with the financial support of the European Union under the: 'TRUST' project no.: CZ.XXX via the Operational Programme Jan Ámos Komenský. 

- Language: English
  
- Key words: example, software, publication

- Field of study: [Pick one here](https://tacr.gov.cz/wp-content/uploads/documents/2021/04/26/1619432556_Struktura%20obor%C5%AF%20FORD.pdf)

----------------------
LEGAL AND ETHICAL ASPECTS
----------------------

- License: MIT
  
- Conditions of use: https://opensource.org/license/mit

- Copyrights/restrictions: none

- Ethical aspects: none


----------------------
FILE OVERVIEW
----------------------

- `hello_world.py`: Example code in Python 3.0.
- `LICENSE`: Example license file
- `.zenodo.json`: Configuration file for Zenodo-GitHub integration, which ensures that Zenodo records produced automatically
                with every release of this repository have correct metadata automatically filled in.
- `README.md`: This file.

----------------------
ORIGIN AND METHODOLOGY
----------------------

- This software was created from scratch by the author and Google Gemini 3 AI.

----------------------
HOW TO USE THIS SOFTWARE
----------------------

- Execute it with command `python3 hello_world.py` to see the beautiful message!
- *(Note: requirements.txt or environment.yml file should be included to specify the necessary libraries.)*
  
----------------------
CITATION OF THIS SOFTWARE
----------------------

Doe, John. (2026). Example Software. Zenodo. https://doi.org/10.5072/zenodo.447283





This repository contains the novelty score calculator that for a given scientific paper (OpenAlex ID) returns a real number indicating the predicted novelty of the paper. The higher the score, the higher the novelty prediction.

You can try the demo on the HuggingFace space [TRUST_score](https://huggingface.co/spaces/PetraV77/TRUST_score) (please do not overload the HF space, it is intended for demonstration purposes only and to demonstrate reproducibility). Recommended usage is to download the source code to your computer and run it locally. 
See below the [instructions](#how-to-run-the-calculator-locally).

## About the score  

Our indicator is based on text analysis and embedding representations, using document embeddings to capture the semantic content of scientific publications and their references. As input data, we use the title and abstract of a focal paper and the titles and abstracts of all papers it cites. If the abstract of a focal paper is not available or there are not enough abstracts of references, only titles are used.

Each document is represented using SPECTER2 embeddings [1], which are designed specifically for scientific texts. These embeddings allow us to measure conceptual similarity between papers in a continuous vector space.

The indicator consists of two complementary components:

+ **Paper–Reference Distance**

We compute the mean conceptual distance between the focal paper and each of its references. Distance is measured as cosine similarity between the corresponding embeddings subtracted from the constant value (we use 1.0, since scientific references are typically related to the citing paper and therefore cosine similarities are high, non-negative). Lower similarity (i.e. larger distance) indicates greater conceptual departure from prior work.

+ **Reference–Reference Distance**

We compute the mean pairwise distance between all cited references, again using cosine similarity. This component captures the extent to which a paper draws on conceptually diverse or weakly connected papers.

Both components are normalized to ensure comparability. The final indicator score is defined as the maximum of the two normalized components, reflecting the assumption that a paper can be considered novel if it satisfies at least one of the following conditions:

  +  it is substantially different from its prior literature, or

  +  it connects or combines conceptually distant areas of research.

We tried several ways of combining the two components, the maximum was chosen as it best fitted the novelty scores in dataset published with the paper [2].

Note: Our first intention was to work with full texts and extracts structured summary from them using LLMs, but this approach does not scale well to large number of papers and also we failed to download a full text in many cases. Therefore we decided to use only abstracts and titles. The future direction is to make structured summary (extracting methods, main ideas, key findings) from full texts and create embeddings from them.

[1] Singh, Amanpreet et al. “SciRepEval: A Multi-Format Benchmark for Scientific Document Representations.” Conference on Empirical Methods in Natural Language Processing (2022).

[2] Daeseong Jeon, Junyoup Lee, Joon Mo Ahn, Changyong Lee. Measuring the novelty of scientific publications: A fastText and local outlier factor approach, Journal of Informetrics, Volume 17, Issue 4, 2023, 101450, ISSN 1751-1577, https://doi.org/10.1016/j.joi.2023.101450


## The TRUST Score calculator App 

1. In case the paper is from the novelty challenge dataset, it finds in prepared datasets   the title, abstract and references, as well as titles and abstracts for all references. 
Otherwise: 
    + Fetches via OpenAlex API the title, abstract and references of the paper queried. 
    + Fetches titles and abstracts for referenced papers.
4. Converts couples (title, abstract) into embedding vectors using **allenai/specter2** model.
5. Calculates final scores. 

## How to run the calculator locally 
Local runs are recommended and enable you to process lists of papers efficiently. Since the calculator uses the **specter2** embedding model, it is recommended to run on a GPU (but CPU run is also possible). You will also need your API token to [OpenAlex](https://openalex.org) database to fetch the data via API. Store your API token in the file `openalex_api_key.txt` in the TRUST_score_app directory and use the flag `--use-api-key`. Alternatively, you can use the environment variable OPENALEX_API_KEY (`export OPENALEX_API_KEY=XXXXXX`).

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
By default, the script runs with `device="auto"`, you can suppress the use of GPU by `--force-cpu`. 
The `--batch-size` controls the number of (title, abstract) embeddings computed in one batch; you can experiment with this option if you have enough memory. Higher batches help to speed up the computation.

The input CSV files must have columns named “PaperProjectID”, “OpenAlexID (as URL)”, and optionally “Title” and “Abstract”. If the title or abstract is not included, data are downloaded from *OpenAlex API*.

The output is a file with `pickled` dictionary (you may specify its name using `--output` option). To convert these raw scores to final scores, use `convert_pickle_to_scores.py`:
```sh
uv run convert_pickle_to_scores.py data/challenge_scores.pickle  --output-file scores.csv
```

### How to reproduce our results
Since the OpenAlex API is not perfectly stable (sometimes data for a valid ID is not returned, and if it is a reference, our script skips it), you may get slightly different results from ours. If you want to reproduce the results exactly, download the prepared `.parquet` files with the downloaded titles, abstracts, and reference lists, and place them in the `data` directory. Then disable API calls with `--only-cached`.

**Warning**: Since the script loads large cached files, it is recommended to use a computer with 32GB of memory (the script uses approx. 8 GB of RAM). I can run it on the laptop with 16 GB, but it’s not ideal when other applications are also running.
 
```sh
uv run calculate_scores.py data/Metadata\ file\ COMBINED.csv  --only-cached
```
or on CPU 
```sh
uv run calculate_scores.py data/Metadata\ file\ COMBINED.csv  --force-cpu --only-cached
```

To fill the challenge result form, use the script `fill_result_form.py`:
```
Usage: fill_result_form.py [OPTIONS] [PICKLED_FILE]

Options:
  --template-file TEXT  Path to the response form template.
  --output-file TEXT    Path to save the filled response form.
  --help                Show this message and exit.
```

such as
```sh
uv run fill_result_form.py data/challenge_scores.pickle --template-data/Challenger_Response_Form.csv --output-file data/filed_form2.csv 
```
[]







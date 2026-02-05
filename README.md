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

<img src="http://github.com/PetraVidnerova/TRUST_score_app/blob/main/img/logo.png?raw=true" width="200px" align="right"/>

# 📜 TRUST Score Calculator 

**Author:** Petra Vidnerová, Institute of Computer Science, The Czech Academy of Sciences  
**Acknowledgment**: todo TRUST project 

This repository contains the novelty score calculator that for a given scientific paper (OpenAlex ID) returns a real number indicating the predicted novelty of the paper. The higher the score, the higher the novelty prediction. 

You can try the demo on the [HuggingFace space TRUST_score](https://huggingface.co/spaces/PetraV77/TRUST_score) or you can download the source code to your computer and run it locally.

## About the score  

TODO 

## The TRUST Score calculator App 

1. Fetches via OpenAlex API the title, abstract and references of the paper queried.
2. Fetches titles and abstracts for referenced papers.
3. Converts couples (title, abstract) into embedding vectors using allenai/specter2 model.
4. Calculates final scores. 

## How to run the calculator locally 
Local run is recommended and it enables you to process lists of papers effiently. Since the calculator uses **specter2** embedding model, it is recommended to run on GPU (but CPU run is also possible). You may also need a token to OpenAlex database to fetch the data via API more efficiently. 

TODO



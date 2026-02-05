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


# 📜 TRUST Score Calculator 
<img src="img/logo.png" width="200px" align="right"/>

This repository contains the novelty score calculator that for a given scientific paper (OpenAlex ID) returns a real number, the higher the number the more novel is the paper considered.

You can try the demo on the [HuggingFace space TRUST_score](https://huggingface.co/spaces/PetraV77/TRUST_score) or you can download the source to you computer and run it locally.

Local run is recommended and it enables you to process lists of papers effiently. Since the calculator uses **specter2** embedding model, it is recommended to run on GPU (but CPU run is also possible). You may also need a token to OpenAlex database to fetch the data via API more efficiently. 



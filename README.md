# An LLM-based Framework for Fingerprinting Internet-connected Devices

This repository includes code used for training and evaluating transformer-based
language models on banners obtained from global Internet scans. The resulting
models can be used to generate device embeddings (for downstream learning
tasks), as well as to analyze clustered embeddings and generate text-based
(regex) fingerprints for detecting software/hardware products.

## Installation

Run `python setup.py install` to install the package and its dependencies.

## Scripts

The `scripts` directory contains the scripts used for preparing datasets,
training the language models, and using clustered embeddings to generate regex
fingerprints. Note that for preparing datasets, one needs to provide banners
collected through Internet-wide scans and exported as JSON files. Models in the
paper are trained on snapshots from the [Censys Universal Internet BigQuery
Dataset](https://support.censys.io/hc/en-us/articles/360056063151-Universal-Internet-BigQuery-Dataset).

## Using Models

Trained models are available on the HuggingFace Hub, and can be further
fine-tuned on downstream applications. Currently, the following models are
available:

- [roberta-base-banner](https://huggingface.co/arsarabi/roberta-base-banner): A
  RoBERTa masked language model trained on banners from all protocols available
  in the Censys database.
- [roberta-embedding-http](https://huggingface.co/arsarabi/roberta-embedding-http):
  A model fine-tuned on HTTP banners (headers) using a contrastive loss function
  to generate temporally stable embeddings. See `scripts/compute_embeddings.py`
  on how to aggregate token embeddings from the last layer of the model to
  compute banner embeddings.

## Reference

- Armin Sarabi, Tongxin Yin, and Mingyan Liu. [An LLM-based Framework for Fingerprinting Internet-connected Devices](https://doi.org/10.1145/3618257.3624845). In Internet Measurement Conference (IMC), 2023.

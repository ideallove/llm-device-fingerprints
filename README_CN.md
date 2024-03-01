# 基于LLM的互联网连接设备指纹识别框架

This repository includes code used for training and evaluating transformer-based
language models on banners obtained from global Internet scans. The resulting
models can be used to generate device embeddings (for downstream learning
tasks), as well as to analyze clustered embeddings and generate text-based
(regex) fingerprints for detecting software/hardware products.

该仓库包含用于在从全球互联网扫描中获得的banner上训练和评估基于Transformer模型的语言模型的代码。
生成的模型可用于生成设备嵌入(device embeddings)（用于下游学习任务），
以及分析集群嵌入(clustered embeddings)
并生成基于文本（正则表达式）的指纹以检测软件/硬件产品。

## Installation 安装

Run `python setup.py install` to install the package and its dependencies.

运行 `python setup.py install` 来安装包及其依赖项。

## Scripts 脚本

The `scripts` directory contains the scripts used for preparing datasets,
training the language models, and using clustered embeddings to generate regex
fingerprints. Note that for preparing datasets, one needs to provide banners
collected through Internet-wide scans and exported as JSON files. Models in the
paper are trained on snapshots from the [Censys Universal Internet BigQuery
Dataset](https://support.censys.io/hc/en-us/articles/360056063151-Universal-Internet-BigQuery-Dataset).

目录`scripts`包含用于准备数据集、训练语言模型以及使用聚类嵌入生成正则表达式指纹的脚本。
请注意，为了准备数据集，需要提供通过互联网范围内的扫描收集并导出为 JSON 文件的banner。
本文中的模型是在[Censys Universal Internet BigQuery Dataset](https://support.censys.io/hc/en-us/articles/360056063151-Universal-Internet-BigQuery-Dataset)的快照上训练的。

## Using Models 使用模型

Trained models are available on the HuggingFace Hub, and can be further
fine-tuned on downstream applications. Currently, the following models are
available:

训练好的模型可以在HuggingFace Hub上找到，并且可以在下游应用中进一步微调。
目前，以下模型可用：

- [roberta-base-banner](https://huggingface.co/arsarabi/roberta-base-banner): A
  RoBERTa masked language model trained on banners from all protocols available
  in the Censys database.
在Censys数据库中所有可用协议的banner上训练的RoBERTa掩码语言模型。

- [roberta-embedding-http](https://huggingface.co/arsarabi/roberta-embedding-http):
  A model fine-tuned on HTTP banners (headers) using a contrastive loss function
  to generate temporally stable embeddings. See `scripts/compute_embeddings.py`
  on how to aggregate token embeddings from the last layer of the model to
  compute banner embeddings.
使用对比损失函数在HTTP banner（头部）上微调的模型，以生成时间稳定的嵌入。 
请参见`scripts/compute_embeddings.py`，了解如何聚合模型的最后一层的token嵌入以计算banner嵌入。

## Reference 参考

- Armin Sarabi, Tongxin Yin, and Mingyan Liu. [An LLM-based Framework for Fingerprinting Internet-connected Devices](https://doi.org/10.1145/3618257.3624845). In Internet Measurement Conference (IMC), 2023.

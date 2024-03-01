"""
Trains an embedding model using a contrastive loss function.
使用一个对比损失函数训练嵌入模型。
"""

import argparse
import json
from pathlib import Path
import random
from typing import Dict

import numpy as np
import sklearn.metrics
import transformers

import devicefingerprints as dfp


def compute_metrics(
        predictions: transformers.trainer_utils.EvalPrediction
) -> Dict[str, float]:
    # Computes performance metrics for the provided predictions.
    preds, labels = predictions
    preds = np.argmax(preds, axis=2).flatten()
    labels = labels.flatten()
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]
    return {
        'accuracy': sklearn.metrics.accuracy_score(labels, preds),
        'precision_pos': sklearn.metrics.precision_score(labels, preds),
        'recall_pos': sklearn.metrics.recall_score(labels, preds),
        'precision_neg': sklearn.metrics.precision_score(1 - labels, 1 - preds),
        'recall_neg': sklearn.metrics.recall_score(1 - labels, 1 - preds)
    }


def train_model(d1: dfp.InternetDataset, d2: dfp.InternetDataset,
                tokenizer: transformers.PreTrainedTokenizerBase,
                pretrained_model: transformers.AutoModelForTokenClassification,
                training_args: transformers.TrainingArguments,
                args: argparse.Namespace):
    # Fine-tunes a pre-trained model for extracting embeddings.
    d1_train, d1_test, d2_train, d2_test = d1.train_test_split(
        d2,
        test_size=args.test_size,
        max_test_size=args.max_test_size,
        seed=args.seed)
    data_collator = transformers.DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8)
    train_dataset = dfp.LcsExtractionDataset(
        (d1_train, d2_train),
        'banner',
        tokenizer,
        max_length=args.max_length,
        spanner=dfp.BannerSpanner(),
        min_match_length=args.min_match_length,
        min_match_total_length=args.min_match_total_length)
    eval_dataset = dfp.LcsExtractionDataset(
        (d1_test, d2_test),
        'banner',
        tokenizer,
        max_length=args.max_length,
        spanner=dfp.BannerSpanner(),
        min_match_length=args.min_match_length)
    trainer = dfp.trainer.EmbeddingTrainer(model=pretrained_model,
                                           args=training_args,
                                           data_collator=data_collator,
                                           train_dataset=train_dataset,
                                           eval_dataset=eval_dataset,
                                           tokenizer=tokenizer,
                                           compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(training_args.output_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_dir',
        nargs='+',
        type=lambda p: Path(p).resolve(),
        help='Path to Hugging Face dataset(s) for training the model.')
    parser.add_argument('--tokenizer-dir',
                        type=lambda p: Path(p).resolve(),
                        required=True,
                        help='Path to the pre-trained tokenizer.')
    parser.add_argument('--mlm-dir',
                        type=lambda p: Path(p).resolve(),
                        required=True,
                        help='Path to the pre-trained masked language model.')
    parser.add_argument('--output-dir',
                        default='embedding',
                        type=lambda p: Path(p).resolve(),
                        help='Path for saving the trained model.')
    parser.add_argument(
        '--training-args',
        type=lambda p: Path(p).resolve(),
        required=True,
        help='Path to a JSON file containing training arguments.')
    parser.add_argument('--service',
                        nargs='*',
                        default=['HTTP'],
                        type=str,
                        help='Service(s) to finetune on.')
    parser.add_argument(
        '--finetune-max-steps',
        default=20000,
        type=int,
        help='Total number of training steps for finetuning over each service.')
    parser.add_argument(
        '--test-size',
        default=0.2,
        type=float,
        help='Proportion of the input dataset to use for testing.')
    parser.add_argument('--max-test-size',
                        default=100000,
                        type=int,
                        help='Maximum number of examples to use for testing.')
    parser.add_argument('--max-length',
                        default=768,
                        type=int,
                        help='Maximum length for truncating sequences.')
    parser.add_argument('--min-match-length',
                        default=3,
                        type=int,
                        help='Minimum length (number of characters) for '
                        'individual (continguous) matches.')
    parser.add_argument('--min-match-total-length',
                        default=11,
                        type=int,
                        help='Minimum length (number of tokens) for an entire '
                        'match (i.e., a sequence of contiguous matches).')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randrange(2**32)

    # Load datasets and match by ID/service.
    datasets = [dfp.InternetDataset.load_from_disk(d) for d in args.dataset_dir]
    datasets_a = []
    datasets_b = []
    for i in range(len(datasets) - 1):
        d1, d2 = datasets[i].intersect(datasets[i + 1],
                                       match_service=True,
                                       keep_in_memory=True)
        datasets_a.append(d1)
        datasets_b.append(d2)

    # Fine-tune pre-trained masked language model for token classification.
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_dir)
    with open(args.training_args) as f:
        training_args = transformers.TrainingArguments(
            args.output_dir / 'all',
            evaluation_strategy='steps',
            seed=args.seed,
            fp16=True,
            dataloader_pin_memory=True,
            **json.load(f))

    pretrained_model = (
        transformers.AutoModelForTokenClassification.from_pretrained(
            args.mlm_dir, num_labels=2))
    d1 = dfp.InternetDataset.concatenate_datasets(datasets_a)
    d2 = dfp.InternetDataset.concatenate_datasets(datasets_b)
    train_model(d1, d2, tokenizer, pretrained_model, training_args, args)

    training_args.max_steps = args.finetune_max_steps
    for service_name in sorted(args.service):
        print(f'Training {service_name}')
        d1 = dfp.InternetDataset.concatenate_datasets([
            d.filter_service(service_name, keep_in_memory=True)
            for d in datasets_a
        ])
        d2 = dfp.InternetDataset.concatenate_datasets([
            d.filter_service(service_name, keep_in_memory=True)
            for d in datasets_b
        ])
        if not d1.num_rows:
            continue

        pretrained_model = (
            transformers.AutoModelForTokenClassification.from_pretrained(
                args.output_dir / 'all', num_labels=2))
        training_args.output_dir = args.output_dir / service_name
        train_model(d1, d2, tokenizer, pretrained_model, training_args, args)


if __name__ == '__main__':
    main()

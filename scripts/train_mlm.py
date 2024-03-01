"""
Trains a masked language model (MLM).
训练一个掩码语言模型（MLM）。
"""

import argparse
import json
from pathlib import Path
import random

import transformers

import devicefingerprints as dfp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_dir',
        nargs='+',
        type=lambda p: Path(p).resolve(),
        help='Path to Hugging Face dataset(s) for training the MLM.')
    parser.add_argument('--tokenizer-dir',
                        type=lambda p: Path(p).resolve(),
                        required=True,
                        help='Path to the pre-trained tokenizer.')
    parser.add_argument('--output-dir',
                        default='mlm',
                        type=lambda p: Path(p).resolve(),
                        help='Path for saving the trained MLM.')
    parser.add_argument('--roberta-config',
                        type=lambda p: Path(p).resolve(),
                        required=True,
                        help='Path to a JSON file containing configuration for '
                        'the RoBERTa model.')
    parser.add_argument(
        '--training-args',
        type=lambda p: Path(p).resolve(),
        required=True,
        help='Path to a JSON file containing training arguments.')
    parser.add_argument('--batch-size',
                        default=1024,
                        type=int,
                        help='Batch size for training.')
    parser.add_argument(
        '--max-tokens',
        default=12 * 768,
        type=int,
        help='Maximum number of tokens in each generated minibatch.')
    parser.add_argument('--max-length',
                        default=768,
                        type=int,
                        help='Maximum length for truncating sequences.')
    parser.add_argument('--ignore-overflowing-tokens',
                        action='store_true',
                        help='Where to ignore overflowing sequences. The '
                        'default is to return overflowing tokens.')
    parser.add_argument('--stride',
                        default=96,
                        type=int,
                        help='Stride length to use for overflowing sequences.')
    parser.add_argument(
        '--mlm-probability',
        default=0.15,
        type=float,
        help='Probability of masking tokens for training the MLM.')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Whether to resume training from the last checkpoint.')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randrange(2**32)

    # Prepare model configuration and training arguments.
    config = transformers.RobertaConfig.from_pretrained('roberta-base')
    with open(args.roberta_config) as f:
        config.update(json.load(f))

    with open(args.training_args) as f:
        training_args = transformers.TrainingArguments(
            args.output_dir,
            seed=args.seed,
            fp16=True,
            dataloader_pin_memory=True,
            **json.load(f))

    # Load dataset/tokenizer.
    dataset = dfp.InternetDataset.concatenate_datasets(
        [dfp.InternetDataset.load_from_disk(d) for d in args.dataset_dir])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Prepare training dataset.
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8)

    skip_batches = 0
    if args.resume_from_checkpoint:
        # If resuming training, compute the number of batches to skip.
        skip_batches = max(
            int(path.name.split('-')[1])
            for path in args.output_dir.glob('checkpoint-*'))
        print(f'Skipping {skip_batches} batches')

    train_dataset = dfp.dataset.SmartBatchDataset(
        dataset,
        'banner',
        tokenizer,
        data_collator,
        batch_size=args.batch_size,
        skip_batches=skip_batches,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        max_tokens=args.max_tokens,
        max_length=args.max_length,
        return_overflowing_tokens=not args.ignore_overflowing_tokens,
        stride=args.stride,
        shuffle=True,
        seed=args.seed)

    # Train model.
    model = transformers.RobertaForMaskedLM(config=config)
    trainer = dfp.trainer.SmartBatchTrainer(model=model,
                                            args=training_args,
                                            train_dataset=train_dataset)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()

"""
Trains a tokenizer for encoding banners.
训练一个用于编码横幅的分词器。
"""

import argparse
from pathlib import Path

import tokenizers
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
        help='Path to Hugging Face dataset(s) for training the tokenizer.')
    parser.add_argument('--output-dir',
                        default='tokenizer',
                        type=lambda p: Path(p).resolve(),
                        help='Path for saving the trained tokenizer.')
    parser.add_argument('--num-examples',
                        default=int(1e8),
                        type=int,
                        help='Number of examples to train on.')
    parser.add_argument('--batch-size',
                        default=1000,
                        type=int,
                        help='Batch size for reading from the dataset.')
    parser.add_argument('--vocab-size',
                        default=50000,
                        type=int,
                        help='Vocabulary size for the tokenizer.')
    parser.add_argument('--min-frequency',
                        default=2,
                        type=int,
                        help='See tokenizers.trainers.BpeTrainer.')
    parser.add_argument('--add-prefix-space',
                        action='store_true',
                        help='See tokenizers.pre_tokenizers.ByteLevel.')
    parser.add_argument('--no-trim-offsets',
                        action='store_true',
                        help='See tokenizers.processors.RobertaProcessing.')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')

    args = parser.parse_args()

    # Load dataset.
    dataset = dfp.InternetDataset.concatenate_datasets(
        [dfp.InternetDataset.load_from_disk(d) for d in args.dataset_dir])
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.with_format(columns=['banner'])

    def iterator():
        num_examples = args.num_examples
        batch_size = args.batch_size
        for i in range(0, num_examples, batch_size):
            yield dataset[i:min(i + batch_size, num_examples)]['banner']

    # Initialize the tokenizer.
    special_tokens = [
        '<s>', '<pad>', '</s>', '<unk>',
        tokenizers.AddedToken('<mask>', lstrip=True)
    ]
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=args.add_prefix_space)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    tokenizer.post_processor = tokenizers.processors.RobertaProcessing(
        ('</s>', special_tokens.index('</s>')),
        ('<s>', special_tokens.index('<s>')),
        trim_offsets=not args.no_trim_offsets,
        add_prefix_space=args.add_prefix_space)

    # Train the tokenizer.
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(iterator=iterator(),
                                  trainer=trainer,
                                  length=args.num_examples)

    # Save the tokenizer to disk.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.RobertaTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()

"""
Computes and saves banner embeddings using a pre-trained model.
计算并保存使用预训练模型的banner嵌入。
"""


import argparse
import functools
from pathlib import Path
import shutil
from typing import Dict

import numpy as np
import sklearn.utils
import torch
import transformers

import devicefingerprints as dfp


def transform(model: transformers.AutoModelForTokenClassification,
              inputs: Dict[str, torch.Tensor],
              model_type: str) -> Dict[str, np.ndarray]:
    # Computes and returns the embeddings of the provided examples.
    if hasattr(model, 'device'):
        inputs = inputs.to(model.device)

    outputs = model(input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True)
    if model_type == 'embedding':
        weights = torch.nn.functional.softmax(outputs['logits'], dim=-1)[:, :,
                                                                         1]
        weights = weights * inputs['attention_mask'].to(weights.device)

        embeddings = outputs['hidden_states'][-1]
        embeddings = (weights.unsqueeze(-1) * embeddings).sum(axis=1)
        embeddings = embeddings / weights.sum(axis=1, keepdim=True)
    elif model_type == 'mlm':
        embeddings = outputs['hidden_states'][-1][:, 0, :]
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    embeddings = embeddings.cpu().numpy().astype(np.float32)
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dataset_dir',
        nargs='+',
        type=lambda p: Path(p).resolve(),
        help='Path to Hugging Face dataset(s) for computing embeddings.')
    parser.add_argument('--tokenizer-dir',
                        type=lambda p: Path(p).resolve(),
                        required=True,
                        help='Path to the pre-trained tokenizer.')
    parser.add_argument(
        '--model-dir',
        type=lambda p: Path(p).resolve(),
        required=True,
        help='Path to the pre-trained model for computing embeddings.')
    parser.add_argument('--output-dir',
                        default='embeddings',
                        type=lambda p: Path(p).resolve(),
                        help='Path for saving the generated datasets '
                        'containing banner embeddings.')
    parser.add_argument('--model-type',
                        default='embedding',
                        type=str,
                        choices=['embedding', 'mlm'],
                        help='Type of the pre-trained model.')
    parser.add_argument('--service',
                        nargs='*',
                        default=['HTTP'],
                        type=str,
                        help='Service(s) to computes embeddings for.')
    parser.add_argument(
        '--num-samples',
        default=int(1e7),
        type=int,
        help='Maximum number of samples to transform for each service.')
    parser.add_argument('--batch-size',
                        default=64,
                        type=int,
                        help='Batch size for computing embeddings.')
    parser.add_argument('--max-length',
                        default=768,
                        type=int,
                        help='Maximum length for truncating sequences.')
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')

    args = parser.parse_args()

    # Load dataset/models.
    dataset = dfp.InternetDataset.concatenate_datasets(
        [dfp.InternetDataset.load_from_disk(d) for d in args.dataset_dir])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_dir)
    models = {}
    if args.model_type == 'embedding':
        for path in args.model_dir.iterdir():
            service_name = path.name
            if service_name in ['all'] + list(dataset.service_names):
                key = None if service_name == 'all' else service_name
                model = (transformers.AutoModelForTokenClassification.
                         from_pretrained(path))
                model = model.to(dtype=torch.float16, device='cuda')
                models[key] = torch.nn.DataParallel(model)
    else:
        model = transformers.AutoModelForMaskedLM.from_pretrained(model_dir)
        model = model.to(dtype=torch.float16, device='cuda')
        models[None] = torch.nn.DataParallel(model)

    # Transform samples from each service and save the results.
    transform_fn = functools.partial(transform, model_type=args.model_type)
    for service_name in args.service:
        print(f'Processing {service_name}')
        dset = dataset.filter_service(service_name)
        if dset.num_rows > args.num_samples:
            indices = sklearn.utils.resample(np.arange(dset.num_rows),
                                             replace=False,
                                             n_samples=args.num_samples,
                                             random_state=args.seed)
            dset = dset.select(np.sort(indices))

        embeddings = dset.transform('banner',
                                    tokenizer,
                                    models,
                                    transform_fn=transform_fn,
                                    batch_size=args.batch_size,
                                    max_length=args.max_length)
        torch.cuda.empty_cache()

        path = args.output_dir / service_name
        if path.is_dir():
            shutil.rmtree(path)

        path.mkdir(parents=True)
        np.savez(path / 'embeddings.npz', embeddings=embeddings)
        dset.flatten_indices()
        dset.save_to_disk(path)


if __name__ == '__main__':
    main()

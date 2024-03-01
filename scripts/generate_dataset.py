"""
Generates a dataset from a Censys Universal Internet Dataset snapshot.
从Centos Universal Internet Dataset的快照中生成数据集。
"""
# 解析命令行参数
import argparse
# 从路径中提取文件名
from pathlib import Path
# 生成随机数
import random
# 生成hash值
import xxhash
# 导入device-fingerprints模块
import devicefingerprints as dfp


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数 --data-dir 用于指定数据文件的路径
    parser.add_argument(
        '--data-dir',
        type=lambda p: Path(p).resolve(),
        required=True,
        help=('Path to the directory containing data files (JSON files '
              'exported from BigQuery).'))
    # 添加参数 --parquet-dir 用于指定Parquet文件的路径
    parser.add_argument(
        '--parquet-dir',
        default='data/parquet',
        type=lambda p: Path(p).resolve(),
        help='Path to the directory for saving the Parquet file.')
    # 添加参数 --dataset-dir 用于指定数据集的路径
    parser.add_argument(
        '--dataset-dir',
        default='data/dataset',
        type=lambda p: Path(p).resolve(),
        help='Path to the directory for saving the Hugging Face dataset.')
    # 添加参数 --detect-encoding 用于指定是否检测编码
    parser.add_argument(
        '--detect-encoding',
        action='store_true',
        help=('Whether to try and detect the encoding of banners for '
              'converting them to unicode strings.'))
    # 添加参数 --default-encoding 用于指定默认编码
    parser.add_argument(
        '--default-encoding',
        default='utf-8',
        type=str,
        help='Encoding to use to when encoding detection fails.')
    # 添加参数 --include-truncated 用于指定是否包含截断的例子
    parser.add_argument('--include-truncated',
                        action='store_true',
                        help='Whether to include truncated examples.')
    # 添加参数 --include-empty-banner 用于指定是否包含空的例子
    parser.add_argument(
        '--include-empty-banner',
        action='store_true',
        help='Whether to include examples with an empty banner.')
    # 添加参数 --sort 用于指定是否对例子进行排序
    parser.add_argument(
        '--sort',
        action='store_true',
        help='Whether to sort examples before writing to the output.')
    # 添加参数 --shuffle 用于指定是否对例子进行洗牌
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Whether to shuffle examples before writing to the output.')
    # 添加参数 --subsample 用于指定子样本的比例
    parser.add_argument(
        '--subsample',
        default=1.0,
        type=float,
        help='If less than one, (filtered) examples are sampled at this rate.')
    # 添加参数 --seed 用于指定随机种子
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='Random seed for shuffling/sampling examples.')
    # 添加参数 --num-workers 用于指定并行工作的数量
    parser.add_argument('--num-workers',
                        default=1,
                        type=int,
                        help='Number of parallel workers for parsing examples.')
    # 添加参数 --chunk-size 用于指定工作块的大小
    parser.add_argument('--chunk-size',
                        default=1000,
                        type=int,
                        help='Chunk size for workers.')
    # 添加参数 --batch-size 用于指定写入输出的批量大小
    parser.add_argument('--batch-size',
                        default=1000,
                        type=int,
                        help='Batch size for writing to the output.')
    # 解析参数
    args = parser.parse_args()
    # 获取数据文件
    data_files = sorted(args.data_dir.glob('*.json.gz'))
    # 创建解析器 parser
    parser = dfp.preprocessing.CensysParser(
        detect_encoding=args.detect_encoding,
        default_encoding=args.default_encoding)
    # 如果没有指定随机种子，则生成一个随机种子
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2 ** 64 - 1)

    # 定义过滤函数
    def filter_func(example: dict) -> bool:
        # 如果不包含截断的例子，则返回False
        drop = False
        # 如果不包含空的例子，则返回False
        if not args.include_truncated:
            drop |= example['truncated']
        if not args.include_empty_banner:
            drop |= not example['banner']
        # 如果子样本的比例小于1，则进行子样本采样
        if args.subsample < 1:
            # 生成hash值 digest
            digest = xxhash.xxh64_intdigest(f"{example['ip']}-{seed}")
            # 如果digest / 2 ** 64 >= args.subsample，则返回True
            drop |= (digest / 2 ** 64) >= args.subsample
        # 返回not drop
        return not drop

    # 创建ParquetGenerator对象
    generator = dfp.preprocessing.ParquetGenerator(data_files=data_files,
                                                   parser=parser,
                                                   filter_func=filter_func,
                                                   sort=args.sort,
                                                   shuffle=args.shuffle,
                                                   seed=args.seed,
                                                   num_workers=args.num_workers,
                                                   chunk_size=args.chunk_size,
                                                   batch_size=args.batch_size)
    # 创建Parquet文件
    args.parquet_dir.mkdir(parents=True, exist_ok=True)
    # 生成Parquet文件
    parquet_file = args.parquet_dir / 'data.parquet'
    generator.generate(parquet_file)

    # 创建数据集目录
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    # 将Parquet文件转换为数据集
    dfp.preprocessing.convert_parquet(parquet_file, args.dataset_dir, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

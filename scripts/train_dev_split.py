import random, os, math, sys
from typing import Any, Dict, Iterable, List, Sequence, Type, Union

def _shuffle_and_split(data: List, test_ratio=None, test_size=None, seed=0):
    # import random

    random.seed(seed)
    size = len(data)

    if test_ratio is not None:
        train_ratio = 1 - test_ratio
        train_size = math.floor(size * train_ratio)
    elif test_size is not None:
        train_size = size - test_size

    index = list(range(size))
    random.shuffle(index)
    train_index = index[:train_size]
    test_index = index[train_size:]

    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]

    return train_data, test_data


def _train_dev_split(
    input_path, train_output_path, dev_output_path, dev_size=5000, seed=0
):
    with open(input_path, "rb") as f:
        data = [line for line in f]
    print(f'read input: {os.path.abspath(input_path)}')

    train_data, dev_data = _shuffle_and_split(data, test_size=dev_size, seed=seed)

    with open(train_output_path, "wb") as f:
        [f.write(i) for i in train_data]
    print(f'write train split: {os.path.abspath(train_output_path)}')

    with open(dev_output_path, "wb") as f:
        [f.write(i) for i in dev_data]
    print(f'write dev split: {os.path.abspath(dev_output_path)}')


def train_dev_split(data_dir):
    data_dir = os.path.abspath(data_dir)
    print(f'data_dir: {data_dir}')
    for dataset_dir_name in ['ag_news_csv', 'dbpedia_csv', 'yelp_review_full_csv']:
        dataset_dir= os.path.join(data_dir, dataset_dir_name)
        print(f'\ndataset_dir: {dataset_dir}')
        input_path = os.path.join(dataset_dir, 'train.csv')
        train_output_path =  os.path.join(dataset_dir, 'train.train.csv')
        dev_output_path = os.path.join(dataset_dir, 'train.dev.csv')
        _train_dev_split(input_path, train_output_path, dev_output_path, dev_size=5000, seed=0)


if __name__ == "__main__":
    train_dev_split(sys.argv[1])

import json
from arguments import DataArguments


def load_gsm8k_dataset(data_args: DataArguments, logger=None):
    dataset = []
    with open(data_args.dataset_filepath, "r", encoding="utf-8") as fin:
        for line in fin:
            dataset.append(json.loads(line.strip()))
    return dataset

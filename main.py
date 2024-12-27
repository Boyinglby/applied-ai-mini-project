
from configuration import (
    CLASSIFIER_CONFIGS,
    DATASET_CONFIGS,
    METRICS_CONFIG,
)
from experiment import run
from utils import seperate


def main():
    dataset_configs = DATASET_CONFIGS
    num_datasets = len(dataset_configs)
    seperate(f"Start running experiments on {num_datasets} datasets")
    for type in ["supervised", "unsupervised"]:
        run(type, dataset_configs, METRICS_CONFIG, CLASSIFIER_CONFIGS)

if __name__=="__main__":
    main()
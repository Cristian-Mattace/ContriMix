"""
Evaluate predictions for Camelyon-17 WILDS datasets.
Usage:
python evaluate.py --predictions_dir /jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon
--output_dir /jupyter-users-home/dinkar-2ejuyal/all_log_dir/erm_camelyon/test_results
 --root_dir /jupyter-users-home/dinkar-2ejuyal/datasets/camelyon17 --run_on_splits val --run_on_seeds 0"""
import argparse
import json
import logging
import os
import urllib.request
from ast import literal_eval
from pathlib import Path
from typing import Dict
from typing import List
from urllib.parse import urlparse

import numpy as np
import torch

from ip_drit.common.metrics._all_metrics import binary_logits_to_pred
from ip_drit.datasets.camelyon17 import CamelyonDataset


def evaluate_benchmark(
    dataset_name: str,
    predictions_dir: str,
    output_dir: str,
    root_dir: str,
    splits: [List[str]] = [],
    seeds: [List[int]] = [],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate across multiple replicates for a single benchmark.

    Args:
        dataset_name: The name of the dataset to evaluate.
        predictions_dir: The path to the directory with predictions.
        output_dir: The output directory.
        root_dir: The directory where datasets can be found.
        splits (optional): Only generate results on given splits. Values can be subset of
        ['train', 'id_val', 'test', 'val']. If not specified, results generated on all of them except `train`
        seeds (optional): A list of seeds to aggregate results over. If not provided, 0 to 9 used
    Returns:
        Metrics as a dictionary with metrics as the keys and metric values as the values
    """

    def _get_replicates(seeds: List[int]) -> List[str]:
        if len(seeds) == 0:
            seeds = range(0, 10)
        return [f"seed:{seed}" for seed in seeds]

    def _get_prediction_file(predictions_dir: str, dataset_name: str, split: str, replicate: str) -> str:
        run_id = f"{dataset_name}_split:{split}_{replicate}"
        for file in os.listdir(predictions_dir):
            if file.startswith(run_id) and (file.endswith(".csv") or file.endswith(".pth")):
                return file
        raise FileNotFoundError(f"Could not find CSV or pth prediction file that starts with {run_id}.")

    def _get_metrics(dataset_name: str) -> List[str]:
        if "camelyon17" == dataset_name:
            return ["acc_avg"]
        else:
            raise ValueError(f"Invalid dataset: {dataset_name}")

    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"Predictions directory does not exist.")

    # Dataset will only be downloaded if it does not exist

    wilds_dataset = CamelyonDataset(dataset_dir=Path(root_dir), use_full_size=True)
    if len(splits) == 0:
        splits: List[str] = list(wilds_dataset.split_dict.keys())
    if "train" in splits:
        splits.remove("train")

    replicates_results: Dict[str, Dict[str, List[float]]] = dict()
    replicates: List[str] = _get_replicates(seeds)
    metrics: List[str] = _get_metrics(dataset_name)

    # Store the results for each replicate
    for split in splits:
        replicates_results[split] = {}
        for metric in metrics:
            replicates_results[split][metric] = []

        for replicate in replicates:
            predictions_file = _get_prediction_file(predictions_dir, dataset_name, split, replicate)
            logging.info(f"Processing split={split}, replicate={replicate}, predictions_file={predictions_file}...")
            full_path = os.path.join(predictions_dir, predictions_file)

            predicted_labels: torch.Tensor = get_predictions(full_path)

            metric_results = evaluate_replicate(wilds_dataset, split, predicted_labels)
            for metric in metrics:
                replicates_results[split][metric].append(metric_results[metric])

    aggregated_results: Dict[str, Dict[str, float]] = dict()

    # Aggregate results of replicates
    for split in splits:
        aggregated_results[split] = {}
        for metric in metrics:
            replicates_metric_values: List[float] = replicates_results[split][metric]
            aggregated_results[split][f"{metric}_std"] = np.std(replicates_metric_values, ddof=1)
            aggregated_results[split][metric] = np.mean(replicates_metric_values)

    # Write out aggregated results to output file
    logging.info(f"Writing aggregated results for {dataset_name} to {output_dir}...")
    Path(output_dir).mkdir(exist_ok=True)
    with open(os.path.join(output_dir, f"{dataset_name}_results.json"), "w") as f:
        json.dump(aggregated_results, f, indent=4)

    return aggregated_results


def evaluate_replicate(dataset, split: str, predicted_labels: torch.Tensor) -> Dict[str, float]:
    """
    Evaluates the given predictions and returns the appropriate metrics.

    Parameters:
        dataset : A WILDS Dataset
        split : split we are evaluating on
        predicted_labels : Predictions

    Returns:
        Metrics as a dictionary with metrics as the keys and metric values as the values
    """
    # Dataset will only be downloaded if it does not exist
    subset = dataset.get_subset(split)
    metadata: torch.Tensor = subset.metadata_array
    true_labels = subset.y_array
    if predicted_labels.shape != true_labels.shape:
        predicted_labels.unsqueeze_(-1)
    return dataset.eval(predicted_labels, true_labels, metadata)[0]


def get_predictions(path: str) -> torch.Tensor:
    """
    Extract out the predictions from the file at path.

    Parameters:
        path: The path to the file that has the predicted labels. It can be an URL.

    Return:
        Tensor representing predictions
    """
    if _is_path_url(path):
        data = urllib.request.urlopen(path)
    else:
        file = open(path, mode="r")
        data = file.readlines()
        file.close()

    predicted_labels = [literal_eval(line.rstrip()) for line in data if line.rstrip()]
    # this is needed as the output of the model are logits
    predicted_labels = [int(predicted_label > 0) for predicted_label in predicted_labels]
    return torch.from_numpy(np.array(predicted_labels))


def _is_path_url(path: str) -> bool:
    """
    Returns True if the path is a URL.
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def main():
    """
    Aggregate evaluation metrics on given splits and seeds and save them
    """
    evaluate_benchmark(
        args.dataset, args.predictions_dir, args.output_dir, args.root_dir, args.run_on_splits, args.run_on_seeds
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="An argument parser to evaluate predictions for Camelyon-17 WILDS datasets."
    )
    parser.add_argument(
        "--predictions_dir", type=str, help="A directory that contains the predictions in .csv or .pth files."
    )
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument(
        "--dataset", type=str, help="The WILDS dataset to evaluate for. Defaults to `camelyon17`", default="camelyon17"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data",
        help="The directory where the datasets can be found (or should be downloaded to, if they do not exist).",
    )

    parser.add_argument(
        "--run_on_splits",
        nargs="+",
        default=[],
        help="Only generate results on given splits. Values can be subset of"
        "['train', 'id_val', 'test', 'val']. If not specified, results generated on all of them except `train`",
    )

    parser.add_argument(
        "--run_on_seeds",
        nargs="+",
        default=[],
        help="Only generate results on given seeds. Values can be subset of"
        "integer list from 0 to 9, If not provided, results from all seeds from 0 to 9 expected in predictions_dir",
    )
    # Parse args and run this script
    args = parser.parse_args()
    main()

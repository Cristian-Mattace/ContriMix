"""A script to run the benchmark for the Camelyon dataset."""
import logging
from absl import app
from ip_drit.datasets.camelyon17 import CamelyonDataset
from pathlib import Path
def main(argv):
    del argv
    logging.info("Running the Camelyon 17 dataset benchmark.")
    camelyon_dataset = CamelyonDataset(dataset_dir=Path('~/Documents/datasets/camelyon17/'))

if __name__ == "__main__":
    app.run(main)
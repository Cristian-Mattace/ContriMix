"""A module that defines a the Camelyon 17 dataset."""
from typing import Optional
from typing import Dict
from pathlib import Path
from ._dataset import AbstractPublicDataset

class CamelyonDataset(AbstractPublicDataset):
    _dataset_name: Optional[str] = 'Camelyon17_WILDS'
    _DOWNLOAD_URL_BY_VERSION: Dict[str, str] = {
        '1.0': 'https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/',
    }
    def __init__(self, dataset_dir: Path) -> None:
        self._version = '1.0'
        super().__init__(dataset_dir=dataset_dir)
        self._patch_size_pixels = (96, 96)

        # Read in metadata
        #self._metadata_df = pd.read_csv(
        #    os.path.join(self._data_dir, 'metadata.csv'),
        #    index_col=0,
        #    dtype={'patient': 'str'})


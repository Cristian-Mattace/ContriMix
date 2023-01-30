import gzip
import hashlib
import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional

from tqdm import tqdm


def download_and_extract_archive(
    url: str,
    download_root: Path,
    extract_root: Optional[Path] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
    size: Optional[int] = None,
) -> None:
    """Download and extract a compressed file of the dataset.

    Args:
        url: The URL to download the data from.
        download_root: The target folder to download the data to.
        extract_root: The target folder to extract the data to.
        filename: The name of the compress file for the whole dataset.
        md5 (optional): MD5 checksum of the download. If None, do not check.
        remove_finished (optional): Delete the downloaded file if True. Defaults to False.
        size (optional): The size of the compress file to download the data from
    """
    download_root = os.path.expanduser(str(download_root))

    extract_root = download_root if extract_root is None else str(extract_root)

    if not filename:
        filename = os.path.basename(url)
    logging.info(f"Downloading the zipped data file from {download_root}.")
    _download_url(url, download_root, filename, md5, size)

    archive = os.path.join(download_root, filename)
    logging.debug("Extracting {} to {}".format(archive, extract_root))
    _extract_archive(archive, extract_root, remove_finished)


def _download_url(
    url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, size: Optional[int] = None
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url: URL to download file from
        root: Directory to place downloaded file in
        filename (optional): Name to save the file under. If None, use the basename of the URL
        md5 (optional): MD5 checksum of the download. If None, do not check.
        size (optional): The size of the compressed file to download.
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    file_path = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if _check_integrity(file_path, md5):
        logging.debug("Using downloaded and verified file: " + file_path)
    else:  # download the file
        try:
            logging.info("Downloading " + url + " to " + file_path)
            urllib.request.urlretrieve(url, file_path, reporthook=_gen_bar_updater(size))
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                logging.error(
                    "Failed download. Trying https -> http instead." " Downloading " + url + " to " + file_path
                )
                urllib.request.urlretrieve(url, file_path, reporthook=_gen_bar_updater(size))
            else:
                raise e
        # check integrity of downloaded file
        if not _check_integrity(file_path, md5):
            raise RuntimeError("File not found or corrupted.")


def _check_integrity(file_path: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(file_path):
        return False
    if md5 is None:
        return True
    return _check_md5(file_path, md5)


def _check_md5(file_path: str, md5: str, **kwargs: Any) -> bool:
    return md5 == _calculate_md5(file_path, **kwargs)


def _calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _gen_bar_updater(total) -> Callable[[int, int, int], None]:
    pbar = tqdm(total=total, unit="Byte")

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def _extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> None:
    if to_path is None:
        to_path = os.path.dirname(from_path)
    if _is_tar(from_path):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def _is_tarxz(filename: str) -> bool:
    return filename.endswith(".tar.xz")


def _is_tar(filename: str) -> bool:
    return filename.endswith(".tar")


def _is_targz(filename: str) -> bool:
    return filename.endswith(".tar.gz")


def _is_tgz(filename: str) -> bool:
    return filename.endswith(".tgz")


def _is_gzip(filename: str) -> bool:
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename: str) -> bool:
    return filename.endswith(".zip")

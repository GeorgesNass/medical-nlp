'''
__author__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "Download MeSH resources and store them in data/raw/mesh with basic integrity checks."
'''

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

from src.core.config import get_settings
from src.utils.logging_utils import get_logger


## ============================================================
## LOGGER
## ============================================================

logger = get_logger("download_mesh")


## ============================================================
## HASHING UTILITIES
## ============================================================

def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        file_path (Path): Target file path.
        chunk_size (int): Read chunk size in bytes.

    Returns:
        str: SHA256 hex digest.
    """

    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def write_checksum_file(file_path: Path, sha256: str) -> Path:
    """
    Write a .sha256 file next to the downloaded artifact.

    Args:
        file_path (Path): Downloaded file path.
        sha256 (str): SHA256 hex digest.

    Returns:
        Path: Path to checksum file.
    """

    checksum_path = file_path.with_suffix(file_path.suffix + ".sha256")
    checksum_path.write_text(f"{sha256}  {file_path.name}\n", encoding="utf-8")
    return checksum_path


## ============================================================
## DOWNLOAD UTILITIES
## ============================================================

def _download_to_path(url: str, output_path: Path, timeout_sec: int = 60) -> None:
    """
    Download an artifact from an URL to a local file path.

    Args:
        url (str): Source URL.
        output_path (Path): Destination path.
        timeout_sec (int): Request timeout in seconds.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ## Use a temporary file to avoid partial writes
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {output_path}")

    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)

    ## Atomic replace
    tmp_path.replace(output_path)
    logger.info(f"Download completed: {output_path}")


def download_mesh(
    url: str,
    file_name: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    """
    Download a MeSH artifact and generate a checksum file.

    Args:
        url (str): URL to download (e.g., MeSH XML/TTL/JSON).
        file_name (Optional[str]): Output filename override. If None, inferred from URL.
        overwrite (bool): If True, overwrite existing file.

    Returns:
        Tuple[Path, Path]: (downloaded_file_path, checksum_file_path)

    Raises:
        FileExistsError: If file exists and overwrite is False.
        ValueError: If output filename cannot be inferred.
    """

    settings = get_settings()
    raw_mesh_dir: Path = settings.raw_mesh_dir

    ## Infer filename from URL if not provided
    inferred_name = url.split("/")[-1].split("?")[0].strip()
    if file_name:
        out_name = file_name.strip()
    else:
        out_name = inferred_name

    if not out_name:
        raise ValueError("Could not infer output file name from URL. Please provide file_name.")

    output_path = raw_mesh_dir / out_name

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}. Use overwrite=True to replace it.")

    ## Download
    _download_to_path(url=url, output_path=output_path)

    ## Checksum
    sha256 = compute_sha256(output_path)
    checksum_path = write_checksum_file(output_path, sha256)

    logger.info(f"SHA256: {sha256}")
    logger.info(f"Checksum file: {checksum_path}")

    return output_path, checksum_path

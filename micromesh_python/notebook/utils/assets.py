import requests
import urllib.parse
from pathlib import Path
import py7zr

def fetch_url(url : str, filepath : Path, force : bool) -> bool:
    if filepath.is_file() and not force:
        print(f'{filepath} exists, skipping download')
        return True
    print(f'Fetching {url}...')
    result = requests.get(url)
    if result.status_code == 200:
        with open(str(filepath), 'wb') as file:
            print(f'Writing {filepath}')
            file.write(result.content)
            return True
    print(f'Error fetching remote file')
    return False

def fetch_and_extract(url : str, download_path : str, extract_path : str, force : bool = False):
    download_path = Path(download_path)
    if not download_path.is_dir():
        download_path.mkdir()

    download_filepath = Path(download_path / Path(urllib.parse.urlparse(url).path).name).resolve()
    extract_path = Path(extract_path).resolve()
    if fetch_url(url, download_filepath, force):
        if extract_path.is_dir() and not force:
            print(f'{extract_path} exists, skipping extraction')
            return True
        with py7zr.SevenZipFile(str(download_filepath), mode='r') as archive:
            print(f'Extracting {archive.getnames()}...')
            archive.extractall(path=str(extract_path))
            return True
    return False

import os
import re
import shutil
import zipfile

import requests
from tqdm import tqdm


class DownloadManager:
    base_directory = os.path.join(os.path.expanduser("~"), ".cache")

    def _create_directory(self, directory_name):
        directory_path = os.path.join(self.base_directory, directory_name)
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    @staticmethod
    def _clean_string(text):
        return re.sub(r'[^a-zA-Z0-9]', '_', text)

    @staticmethod
    def _delete_existing_file(file_path):
        if os.path.exists(file_path):
            os.unlink(file_path)

    @staticmethod
    def _download_file(url: str, filepath: str, verbose: bool = True):
        if os.path.exists(filepath):
            return

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as file:
            if verbose:
                with tqdm(
                        desc=f"Downloading: {filepath}",
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as progress_bar:
                    for data in response.iter_content(chunk_size=1024):
                        written_size = file.write(data)
                        progress_bar.update(written_size)
            else:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)

    @staticmethod
    def _delete_existing_folder(path: str):
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def _extract_zip_file(file_path, destination_path):
        if not os.path.exists(destination_path):

            with zipfile.ZipFile(file_path, "r") as zip_file:
                for file in tqdm(iterable=zip_file.namelist(),
                                 total=len(zip_file.namelist()),
                                 desc=f"Extracting: {destination_path}"):
                    zip_file.extract(member=file, path=destination_path)

    @classmethod
    def load_zip(cls, url, remove_existing=False, model_file_name='model.bin', verbose=True):
        zip_filename = cls._clean_string(url.split("/")[-1])
        zip_file_path = os.path.join(cls.base_directory, zip_filename)

        if remove_existing:
            cls._delete_existing_file(zip_file_path)

        cls._download_file(url, zip_file_path, verbose)

        extracted_data_directory_path = os.path.join(cls.base_directory, cls._clean_string(zip_filename) + "_data")
        if remove_existing:
            cls._delete_existing_folder(extracted_data_directory_path)

        cls._extract_zip_file(zip_file_path, extracted_data_directory_path)

        return os.path.join(extracted_data_directory_path, model_file_name)


if __name__ == "__main__":
    download_path = DownloadManager.load_zip(
        'http://vectors.nlpl.eu/repository/20/180.zip',
        remove_existing=True)
    print("Download 1 path:", download_path)

    download_path = DownloadManager.load_zip(
        'http://vectors.nlpl.eu/repository/20/180.zip',
        remove_existing=False)
    print("Download 2 path:", download_path)

    download_path = DownloadManager.load_zip(
        'http://vectors.nlpl.eu/repository/20/180.zip',
        remove_existing=True, verbose=False)
    print("Download 2 path:", download_path)

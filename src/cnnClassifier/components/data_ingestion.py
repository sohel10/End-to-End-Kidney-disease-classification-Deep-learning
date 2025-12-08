import os
import zipfile
import requests
from pathlib import Path

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # ---------------------------------------------------------
    # DOWNLOAD FILE (SKIPPED IF FILE EXISTS OR URL IS NULL)
    # ---------------------------------------------------------
    def download_file(self):
        if self.config.source_URL is None:
            logger.info("Skipping download: source_URL is null")
            return

        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading file from {self.config.source_URL} ...")

            response = requests.get(self.config.source_URL)
            with open(self.config.local_data_file, "wb") as f:
                f.write(response.content)

            logger.info(f"File downloaded → {self.config.local_data_file}")
        else:
            logger.info(f"File already exists → {self.config.local_data_file}")

    # ---------------------------------------------------------
    # EXTRACT ZIP FILE
    # ---------------------------------------------------------
    def extract_zip_file(self):
        unzip_dir = self.config.unzip_dir
        zip_path = self.config.local_data_file

        # Create target directory
        os.makedirs(unzip_dir, exist_ok=True)

        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP file not found → {zip_path}")

        logger.info(f"Extracting ZIP: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)

        logger.info(f"Extraction completed → {unzip_dir}")

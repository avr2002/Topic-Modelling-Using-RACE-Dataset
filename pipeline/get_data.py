import os
import csv
import json
import yaml
import tarfile
import logging
import urllib.request
import pandas as pd

# import hydra
# from omegaconf import DictConfig


logger = logging.getLogger(__name__)

CONFIG_PATH = "./config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    data_config = config['data_config']
    


def download_race_data(download_url=data_config['dataset_download_url'])->str:
    try:
        tgz_download_dir = data_config['tgz_download_dir']
        file_name = os.path.basename(download_url)
        tgz_file_path = os.path.join(tgz_download_dir, file_name)

        # if os.path.exists(tgz_download_dir):
        #     os.remove(tgz_download_dir)
        os.makedirs(tgz_download_dir, exist_ok=True)

        if os.path.isfile(tgz_file_path):
            print("\nFile is already Downloaded!")
        else:
            print(f"\nDownloading file from [{download_url}] into: [{tgz_file_path}]")
            urllib.request.urlretrieve(url=download_url, filename=tgz_file_path)
            print("Download Successful!\n")
        
        return tgz_file_path
    except Exception as e:
        logger.error(f"Error occured in reading downloading data file: {e}", exc_info=True)


def extract_tgz_file(tgz_file_path: str)->str:
    try:
        extraction_path = data_config['raw_data_dir']
        # extraction_path = config.data_config.raw_data_dir
        os.makedirs(extraction_path, exist_ok=True)

        if len(os.listdir(extraction_path))>0:
            print(f"Text Files already Extracted in [{extraction_path}]")
        else:
            print(f"Extracting tgz file: [{tgz_file_path}] into dir: [{extraction_path}]")

            with tarfile.open(tgz_file_path) as race_tgz_file_object:
                race_tgz_file_object.extractall(extraction_path)

            print("Extraction Completed!\n")
        return extraction_path
    except Exception as e:
        logger.error(f"Error occured while extracting the files: {e}", exc_info=True)


def txt_to_csv(extracted_dir_path: str) -> str:
    try:
        txt_data_dir = os.path.join(extracted_dir_path, 'RACE')
        
        ingested_data_dir = data_config['ingested_data']
        os.makedirs(ingested_data_dir, exist_ok=True)

        csv_file_path = os.path.join(ingested_data_dir, data_config['csv_text_file_name'])

        if os.path.isfile(csv_file_path):
            print(f"Text Data is in [{txt_data_dir}], and is already Converted to CSV File: [{csv_file_path}]")
        else:
            print("Extracting 'articles' Section of Text Files into a CSV File...")
            document_file = open(csv_file_path, mode="w", newline="")
            csv_writer = csv.writer(document_file)

            # Writing the header
            csv_writer.writerow(['document'])

            for folder in os.listdir(txt_data_dir)[::-1]:
                for sub_folder in os.listdir(os.path.join(txt_data_dir, folder)):
                    for file in os.listdir(os.path.join(txt_data_dir, folder, sub_folder)):
                        file_path = os.path.join(txt_data_dir, folder, sub_folder, file)

                        with open(file_path, "r") as json_file:
                            json_data = json.load(json_file)

                        # Writing the 'article' in 'document' column
                        csv_writer.writerow([json_data['article']])

            document_file.close()
            print(f"Extraction Successful! in [{csv_file_path}]\n")
        return csv_file_path
    except Exception as e:
        logger.error(f"\nError occured while parsing txt files to csv file:\n{e}\n", exc_info=True)



def get_and_read_data() -> pd.DataFrame:
    try:
        tgz_file_path = download_race_data()
        extracted_dir_path = extract_tgz_file(tgz_file_path)
        data_file_path = txt_to_csv(extracted_dir_path)

        df = pd.read_csv(data_file_path)
        print("Data Loaded Successfully!\n")
        return df
    except Exception as e:
        logger.error(f"Error while reading the data: {e}", exc_info=True)



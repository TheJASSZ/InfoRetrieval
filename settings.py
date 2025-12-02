import os
import logging
import configparser
import torch
from packages.logger import setup_logger

logger = setup_logger(name="api", log_filename="api_log.log", log_dir="logs", level=logging.DEBUG)

BASE_DIR = os.getcwd()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

config = configparser.ConfigParser()
config.optionxform = str  # keep case. configparser by default converts to lowercase
config.read('config/config.ini')
config_dict = {section: dict(config.items(section)) for section in config.sections()}


BLIP_FINETUNED_PATH = config_dict["MODELS"]["BLIP_FINETUNED_PATH"]
FLAN_T5_SMALL_PATH = config_dict["MODELS"]["FLAN_T5_SMALL_PATH"]

DB_PATH = config_dict["DATABASE"]["DB_PATH"]
INSERT_QUERY = config_dict["DATABASE"]["INSERT_QUERY"]
FETCH_QUERY = config_dict["DATABASE"]["FETCH_QUERY"]


if torch.cuda.is_available():
    DEVICE = 0  # 0 refers to the first CUDA device
    logger.info("Device- CUDA")
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # For acceleration onApple chips
    logger.info("Device- MPS")
else:
    DEVICE = -1  # -1 for CPU
    logger.info("Device- CPU")


import logging
import logging.handlers
import os


def setup_logger(name, log_filename, log_dir="logs", level=logging.INFO):
    """
    Creates a logger that writes to both a rotating file and the console.
    """

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times if function is called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    project_root = os.getcwd()
    full_log_dir = os.path.join(project_root, log_dir)

    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)

    full_log_path = os.path.join(full_log_dir, log_filename)

    # --- 1. Rotating File Handler ---
    # maxBytes=5MB: When file hits 5MB, it rotates
    # backupCount=3: Keeps 3 backup files
    file_handler = logging.handlers.RotatingFileHandler(
        full_log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)

    # --- 2. Stream Handler (Console) ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
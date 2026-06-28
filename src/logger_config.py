import logging
import os


def setup_script_logger(script_name: str) -> logging.Logger:
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(f"logs/{script_name}.log")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

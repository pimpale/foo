import logging


def get_logger(log_file="history.log", log_name="ProjectLogger", log_level=logging.DEBUG):
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


logger = get_logger()

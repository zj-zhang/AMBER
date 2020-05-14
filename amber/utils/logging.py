import logging
import os


def setup_logger(working_dir='.', verbose_level=logging.INFO):
    # setup logger
    logger = logging.getLogger('AMBER')
    logger.setLevel(verbose_level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(working_dir, 'log.AMBER.txt'))
    fh.setLevel(verbose_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(verbose_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


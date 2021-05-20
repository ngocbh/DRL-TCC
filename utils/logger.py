import logging
import os
from torch.utils.tensorboard import SummaryWriter

def make_logger(log_dir):
    global logger, writer
    log_dir = log_dir or 'logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'debug.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    writer = SummaryWriter(log_dir)
    return logger, writer


def init_writer(log_dir='logs', ept=None):
    global writer
    ept_dir = os.path.join(log_dir, ept) if ept is not None else log_dir
    writer = SummaryWriter(ept_dir)
    return writer


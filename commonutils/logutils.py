import datetime
import logging
import daiquiri
from constants import model_constants


def get_logger(name):
    daiquiri.setup(
        level=logging.DEBUG,
        outputs=(
            daiquiri.output.File(model_constants.LOG_FILE_PATH, level=logging.DEBUG),
            daiquiri.output.TimedRotatingFile(
                model_constants.LOG_FILE_PATH,
                level=logging.DEBUG,
                interval=datetime.timedelta(weeks=356))
        )
    )
    logger = daiquiri.getLogger(name)
    return logger

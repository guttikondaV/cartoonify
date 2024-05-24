# This is the enviroment point of the package. This module contains all the configs
# that are used to create the GAN model and train it.
# This module has side effects and is the only module that calls functions
# without guarding them with an if __name__ == "__main__": block.
import os
import pathlib
import random

import dotenv
import numpy as np
import torch

PROJECT_DIR = pathlib.Path(__file__).parent.parent


def seed_everything(seed_value: int = 42):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed_value {int} -- integer value
    """
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # gpu vars
    torch.backends.cudnn.deterministic = True  # needed
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_value)


dotenv.load_dotenv()

seed_everything()

import os
import random
import re

import numpy as np
import torch
from num2words import num2words


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def replace_numbers_with_words(sentence: str) -> str:
    sentence = re.sub(r"(\d+)", r" \1 ", sentence)  # add spaces around numbers

    def replace_with_words(match: re.Match):
        num = match.group(0)
        try:
            return num2words(num)
        except:
            return num

    return re.sub(r"\b\d+\b", replace_with_words, sentence)

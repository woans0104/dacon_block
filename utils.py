
import cv2
import numpy as np
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def random_label(labels, min_num = 4 , sort=False):
    rand_num = np.random.randint(min_num, len(labels)+1)
    rand_label = np.random.choice(labels,rand_num, replace=False)
    if sort :
        rand_label.sort()
    return rand_label

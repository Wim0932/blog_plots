import yaml
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm

CMAP = cm.YlGnBu
FUNC = lambda x: (x ** 2) * (x + 1) * (x - 1.25)  # noqa


def file_path(input_path: str):
    input_path = Path(input_path)
    return input_path


def read_yaml(yaml_path: Path):
    return yaml.safe_load(open(str(yaml_path), "r"))


def clear_plot(function):
    def wrapper(*args, **kwargs):
        func_val = function(*args, **kwargs)
        plt.cla()
        plt.clf()
        return func_val

    return wrapper

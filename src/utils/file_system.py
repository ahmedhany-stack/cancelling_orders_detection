import os 

from pathlib import Path


def create_directories(paths):

    if isinstance(paths, (str, Path)):
        paths = [paths]

    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def delete_file(path):
    if os.path.exists(path):
        os.remove(path)

def file_exists(path):
    return os.path.exists(path)
import joblib

def save_object(path, obj):
    joblib.dump(obj, path)

def load_object(path):
    return joblib.load(path)

import pandas as pd

import pandas as pd

def load_csv(path, encoding="utf-8"):
    return pd.read_csv(path, encoding=encoding)


def load_excel(path):
    return pd.read_excel(path)

def save_csv(df, path):
    df.to_csv(path, index=False)

import os

import json

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path) as f:
        return json.load(f)
    
import yaml

def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)
    
import yaml

def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)
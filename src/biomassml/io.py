import pickle
import json
import joblib


def dump_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f)


def dump_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def dump_joblib(obj, filename):
    with open(filename, "wb") as f:
        joblib.dump(obj, f)


def load_json(filename):
    with open( filename, "r") as f:
        return json.load(f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_joblib(filename):
    with open(filename, "rb") as f:
        return joblib.load(f)

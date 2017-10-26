import cloudpickle as pickle
import json

def pickle_load(fname):
    with open(fname, 'rb') as fh:
        return pickle.load(fh)

def pickle_dump(obj, fname):
    with open(fname, 'wb') as fh:
        return pickle.dump(obj, fh)


def json_load(fname):
    with open(fname, 'rt') as fh:
        return json.load(fh)

def json_dump(obj, fname):
    with open(fname, 'wt') as fh:
        return json.dump(obj, fh)

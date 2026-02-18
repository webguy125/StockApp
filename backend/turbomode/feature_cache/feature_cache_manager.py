import os
import pickle

CACHE_DIR = 'backend/turbomode/feature_cache/'
CACHE_VERSION = 'v1'

os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(name):
    return os.path.join(CACHE_DIR, f'{name}_{CACHE_VERSION}.pkl')


def save_cache(name, data):
    path = cache_path(name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return path


def load_cache(name):
    path = cache_path(name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def validate_cache(name):
    return os.path.exists(cache_path(name))

"""Used to load/store objects in ~/.seq2seq_cache/."""
import hashlib
import os.path
try:
    import cPickle as pickle
except ImportError:
    import pickle


CACHE_DIR = os.path.join(os.path.dirname(__file__), '.seq2seq_cache')
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)


def _filename(cache_id):
    return os.path.join(CACHE_DIR, '%s.pkl' % cache_id)


def get_id(*args):
    m = hashlib.md5()
    for arg in args:
        m.update(str(arg).encode('utf-8'))
    return m.hexdigest()


def exists(cache_id):
    return os.path.exists(_filename(cache_id))


def load(cache_id):
    with open(_filename(cache_id), 'rb') as infile:
        return pickle.load(infile)


def store(cache_id, obj):
    with open(_filename(cache_id), 'wb') as outfile:
        pickle.dump(obj, outfile, 2)
